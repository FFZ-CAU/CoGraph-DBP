from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import os
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import KFold
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
METRIC_COLUMNS = ['loss', 'specificity', 'accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc', 'mcc', 'threshold']

@dataclass
class RunConfig:
    root: str
    out_dir: str
    train_table: str
    train_split_root: str
    contact_subdir: str
    esm_map_subdir: str
    max_len: int
    k: int
    contact_graph_mode: str
    epochs: int
    batch_size: int
    lr: float
    variants: List[str]
    seeds: List[int]
    split_seed: Optional[int]
    folds_to_run: Optional[List[int]]
    preload: bool
    num_workers: int
    device: str
    save_best_checkpoint: bool

def parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(',') if item.strip()]

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_pickle_table(path: Path) -> List[Dict[str, object]]:
    with path.open('rb') as handle:
        raw = pickle.load(handle)
    rows = []
    for name, item in raw.items():
        seq, label, seq_len = (item[0], int(item[1]), int(item[2]))
        rows.append({'name': str(name), 'seq': seq, 'label': label, 'seq_len': seq_len})
    return rows

def pad_2d(array: np.ndarray, rows: int, cols: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    out = np.zeros((rows, cols), dtype=np.float32)
    r = min(rows, array.shape[0])
    c = min(cols, array.shape[1])
    out[:r, :c] = array[:r, :c]
    return out

def pad_square(array: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    out = np.zeros((size, size), dtype=np.float32)
    n = min(size, array.shape[0], array.shape[1])
    out[:n, :n] = array[:n, :n]
    return out

def normalize_adj(matrix: np.ndarray) -> np.ndarray:
    rowsum = np.asarray(matrix.sum(1), dtype=np.float32)
    inv_sqrt = np.power(rowsum, -0.5, out=np.zeros_like(rowsum), where=rowsum > 0)
    inv_sqrt[rowsum <= 0] = 0.0
    return inv_sqrt[:, None] * matrix * inv_sqrt[None, :]

def attention_to_knn_matrix(attention_matrix: np.ndarray, k: int, symmetric: bool=True, weighted: bool=False) -> np.ndarray:
    raw = np.array(attention_matrix, dtype=np.float32, copy=True)
    att = raw.copy()
    if att.shape[0] <= 1:
        return np.ones_like(att, dtype=np.float32)
    k = max(1, min(k, att.shape[0] - 1))
    np.fill_diagonal(att, -np.inf)
    indices = np.argpartition(att, -k, axis=1)[:, -k:]
    knn = np.zeros_like(att, dtype=np.float32)
    if weighted:
        values = np.take_along_axis(raw, indices, axis=1).astype(np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        min_value = float(values.min()) if values.size else 0.0
        if min_value < 0:
            values = values - min_value
        values = values + 1e-06
        np.put_along_axis(knn, indices, values, axis=1)
    else:
        np.put_along_axis(knn, indices, 1.0, axis=1)
    if symmetric:
        knn = np.maximum(knn, knn.T)
    return knn

def build_contact_graph(attention_matrix: np.ndarray, k: int, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == 'binary':
        return attention_to_knn_matrix(attention_matrix, k, symmetric=True, weighted=False)
    if mode == 'weighted':
        return attention_to_knn_matrix(attention_matrix, k, symmetric=True, weighted=True)
    raise ValueError(f'Unsupported contact graph mode: {mode}')

def distance_to_knn_matrix(distance_matrix: np.ndarray, k: int, symmetric: bool=True) -> np.ndarray:
    dist = np.array(distance_matrix, dtype=np.float32, copy=True)
    if dist.shape[0] <= 1:
        return np.ones_like(dist, dtype=np.float32)
    k = max(1, min(k, dist.shape[0] - 1))
    np.fill_diagonal(dist, np.inf)
    indices = np.argpartition(dist, k, axis=1)[:, :k]
    knn = np.zeros_like(dist, dtype=np.float32)
    np.put_along_axis(knn, indices, 1.0, axis=1)
    if symmetric:
        knn = np.maximum(knn, knn.T)
    return knn

class ProteinGraphDataset(Dataset):

    def __init__(self, rows: Sequence[Dict[str, object]], split_root: Path, max_len: int, k: int, contact_graph_mode: str='binary', contact_subdir: str='contact', esm_map_subdir: str='map-esm', preload: bool=False) -> None:
        self.rows = list(rows)
        self.split_root = split_root
        self.max_len = max_len
        self.k = k
        self.contact_graph_mode = contact_graph_mode
        self.contact_subdir = contact_subdir
        self.esm_map_subdir = esm_map_subdir
        self.preload = preload
        self.cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
        if preload:
            for row in self.rows:
                name = str(row['name'])
                self.cache[name] = self._load_item_arrays(name)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        name = str(row['name'])
        label = int(row['label'])
        seq_len = int(row['seq_len'])
        if name in self.cache:
            prot5, contact, esm_map, real_len = self.cache[name]
        else:
            prot5, contact, esm_map, real_len = self._load_item_arrays(name)
        usable_len = min(seq_len, real_len, self.max_len)
        return {'name': name, 'prot5': prot5, 'contact': contact, 'esm_map': esm_map, 'label': np.int64(label), 'length': np.int64(usable_len)}

    def _load_item_arrays(self, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        prot5_path = self.split_root / 'feature' / 'feature-prot5-xl' / f'{name}.npy'
        contact_path = self.split_root / self.contact_subdir / f'{name}.npy'
        esm_map_path = self.split_root / self.esm_map_subdir / f'{name}.npy'
        prot5_raw = np.load(prot5_path)
        real_len = int(prot5_raw.shape[0])
        prot5 = pad_2d(prot5_raw, self.max_len, 1024)
        contact_raw = np.load(contact_path)
        contact = build_contact_graph(contact_raw, self.k, self.contact_graph_mode)
        contact = normalize_adj(contact.astype(np.float32))
        contact = pad_square(contact, self.max_len)
        valid_nodes = min(real_len, self.max_len)
        valid_index = np.arange(valid_nodes)
        contact[valid_index, valid_index] = 1.0
        esm_map_raw = np.load(esm_map_path)
        esm_map = distance_to_knn_matrix(esm_map_raw, self.k, symmetric=True)
        esm_map = normalize_adj(esm_map.astype(np.float32))
        esm_map = pad_square(esm_map, self.max_len)
        esm_map[valid_index, valid_index] = 1.0
        return (prot5.astype(np.float32), contact.astype(np.float32), esm_map.astype(np.float32), real_len)

class CustomLoss(nn.Module):

    def __init__(self, w: float=2.0, r: float=0.0) -> None:
        super().__init__()
        self.w = w
        self.r = r

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred_prob = F.softmax(y_pred, dim=-1)
        y_true = y_true.float()
        eps = 1e-08
        loss_pos = -self.w * (1 - y_pred_prob[:, 1]) ** self.r * torch.log(y_pred_prob[:, 1] + eps)
        loss_neg = -y_pred_prob[:, 1] ** self.r * torch.log(1 - y_pred_prob[:, 1] + eps)
        return torch.where(y_true == 1, loss_pos, loss_neg).mean()

class Normalize(nn.Module):

    def __init__(self, features: int, epsilon: float=1e-06) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, dim: int=-1) -> torch.Tensor:
        mu = x.mean(dim=dim, keepdim=True)
        sigma = x.std(dim=dim, keepdim=True)
        shape = [1] * x.dim()
        shape[dim] = x.size(dim)
        gain = self.gain.view(shape)
        bias = self.bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias

class BaseGAT(nn.Module):

    def __init__(self, hidden_size: int=512, dropout: float=0.2) -> None:
        super().__init__()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = Normalize(hidden_size)
        self.w_attn = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x0 = x
        _, n_nodes, _ = x.size()
        adj_attn = self.sigmoid(torch.einsum('bij,bkj->bik', self.w_attn(x), x))
        adj_attn = adj_attn + 1e-05 * torch.eye(n_nodes, device=x.device)
        adj_attn = adj_attn * adj
        adj_attn_sum = adj_attn.sum(dim=2).clamp(min=1e-05)
        adj_attn = adj_attn / adj_attn_sum.unsqueeze(-1)
        x = torch.einsum('bij,bjd->bid', adj_attn, x)
        return self.norm(x0 + self.dropout(self.relu(x)))

class NetProtGAT(nn.Module):

    def __init__(self, hidden_size: int=512, num_layers: int=4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([BaseGAT(hidden_size) for _ in range(num_layers)])
        self.final = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        d = adj.sum(dim=2).clamp(min=1e-05)
        d_inv_sqrt = torch.sqrt(1.0 / d)
        norm_adj = d_inv_sqrt.unsqueeze(2) * adj * d_inv_sqrt.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, norm_adj)
        return self.final(x)

class ResidualGCNLayer(nn.Module):

    def __init__(self, hidden_size: int=512, dropout: float=0.2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = Normalize(hidden_size)
        self.weight = Parameter(torch.empty(hidden_size * 2, hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, x0: torch.Tensor, layer_id: int, adj: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        weight = self.weight.unsqueeze(0).expand(batch_size, -1, -1)
        propagated = torch.bmm(adj, x)
        theta = min(1.0, math.log(1.5 / layer_id + 1.0))
        support = torch.cat([propagated, x0], dim=-1)
        residual = 0.3 * propagated + 0.7 * x0
        out = theta * torch.bmm(support, weight) + (1.0 - theta) * residual
        return self.norm(x0 + self.dropout(self.relu(out)))

class NetProtGCN(nn.Module):

    def __init__(self, hidden_size: int=512, num_layers: int=4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ResidualGCNLayer(hidden_size) for _ in range(num_layers)])
        self.final = nn.Linear(hidden_size, hidden_size // 2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x0 = x
        for layer_id, layer in enumerate(self.layers, start=1):
            x = layer(x, x0, layer_id, adj)
        return self.final(x)

def masked_node_mean(x: torch.Tensor, node_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if node_mask is None:
        return x.mean(dim=1)
    denominator = node_mask.sum(dim=1).clamp_min(1.0)
    return (x * node_mask).sum(dim=1) / denominator

class GraphBlock(nn.Module):

    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = canonical_variant(variant)
        self.gat = NetProtGAT(512)
        self.gcn = NetProtGCN(512)
        self.drop = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, contact_adj: torch.Tensor, esm_adj: torch.Tensor, node_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        if self.variant == 'no_graph':
            return x
        x0 = x
        zeros = torch.zeros((*x.shape[:2], 256), dtype=x.dtype, device=x.device)
        if self.variant == 'contact_only':
            left = self.gat(x, contact_adj)
            right = zeros
        elif self.variant == 'esm_map_only':
            left = zeros
            right = self.gcn(x, esm_adj)
        elif self.variant == 'dual_graph':
            left = self.gat(x, contact_adj)
            right = self.gcn(x, esm_adj)
        else:
            raise ValueError(f'Unsupported variant: {self.variant}')
        graph_update = torch.cat([left, right], dim=2)
        if node_mask is not None:
            graph_update = graph_update * node_mask
        return x0 + self.drop(graph_update)

class CoGraphModel(nn.Module):

    def __init__(self, variant: str, num_graph_blocks: int=2) -> None:
        super().__init__()
        self.variant = canonical_variant(variant)
        self.proj = nn.Linear(1024, 512)
        self.blocks = nn.ModuleList([GraphBlock(self.variant) for _ in range(num_graph_blocks)])
        self.classifier = nn.Sequential(nn.Linear(512, 128), nn.LeakyReLU(), nn.Dropout(0.2), nn.Linear(128, 32), nn.LeakyReLU(), nn.Dropout(0.2), nn.Linear(32, 2))

    def forward(self, prot5: torch.Tensor, contact_adj: torch.Tensor, esm_adj: torch.Tensor, lengths: Optional[torch.Tensor]=None) -> torch.Tensor:
        node_mask: Optional[torch.Tensor] = None
        if lengths is not None:
            positions = torch.arange(prot5.shape[1], device=prot5.device).unsqueeze(0)
            node_mask = (positions < lengths.unsqueeze(1)).to(prot5.dtype).unsqueeze(-1)
        x = self.proj(prot5)
        if node_mask is not None:
            x = x * node_mask
        for block in self.blocks:
            x = block(x, contact_adj, esm_adj, node_mask=node_mask)
        x = masked_node_mean(x, node_mask)
        return self.classifier(x)

def canonical_variant(variant: str) -> str:
    allowed = {'no_graph', 'contact_only', 'esm_map_only', 'dual_graph'}
    if variant not in allowed:
        raise ValueError(f'Unsupported variant: {variant}')
    return variant

def batch_to_device(batch: Dict[str, object], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prot5 = batch['prot5'].to(device=device, dtype=torch.float32)
    contact = batch['contact'].to(device=device, dtype=torch.float32)
    esm_map = batch['esm_map'].to(device=device, dtype=torch.float32)
    labels = batch['label'].to(device=device, dtype=torch.long)
    lengths = batch['length'].to(device=device, dtype=torch.long)
    return (prot5, contact, esm_map, labels, lengths)

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    losses: List[float] = []
    for batch in loader:
        prot5, contact, esm_map, labels, lengths = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(prot5, contact, esm_map, lengths)
        loss = criterion(logits, labels.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, List[int], List[float]]:
    model.eval()
    losses: List[float] = []
    y_true: List[int] = []
    y_score: List[float] = []
    for batch in loader:
        prot5, contact, esm_map, labels, lengths = batch_to_device(batch, device)
        logits = model(prot5, contact, esm_map, lengths)
        loss = criterion(logits, labels.view(-1))
        prob = F.softmax(logits, dim=1)[:, 1]
        losses.append(float(loss.detach().cpu()))
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        y_score.extend(prob.detach().cpu().numpy().astype(float).tolist())
    return (float(np.mean(losses)) if losses else 0.0, y_true, y_score)

def analyze(y_true: Sequence[int], y_score: Sequence[float], loss: float, threshold: Optional[float]=None) -> Dict[str, float]:
    if threshold is None:
        best_mcc = -2.0
        best_threshold = 0.5
        for idx in range(1, 101):
            candidate = idx / 100.0
            pred = [1 if score >= candidate else 0 for score in y_score]
            mcc = metrics.matthews_corrcoef(y_true, pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = candidate
        threshold = best_threshold
    y_pred = [1 if score >= threshold else 0 for score in y_score]
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    if len(set(y_true)) == 2:
        auc = metrics.roc_auc_score(y_true, y_score)
        precisions, recalls, _ = metrics.precision_recall_curve(y_true, y_score)
        auprc = metrics.auc(recalls, precisions)
    else:
        auc = 0.0
        auprc = 0.0
    return {'loss': float(loss), 'specificity': float(tn / (tn + fp)) if tn + fp else 0.0, 'accuracy': float(metrics.accuracy_score(y_true, y_pred)), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'auc': float(auc), 'auprc': float(auprc), 'mcc': float(metrics.matthews_corrcoef(y_true, y_pred)), 'threshold': float(threshold)}

def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

def write_row(path: Path, row: Dict[str, object], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)

def metric_prefix(prefix: str, values: Dict[str, float]) -> Dict[str, float]:
    return {f'{prefix}_{key}': value for key, value in values.items()}

def checkpoint_config(model: nn.Module, config: RunConfig) -> Dict[str, object]:
    source_set = Path(config.train_split_root).name
    path = getattr(model, 'path', 'topology' if hasattr(model.blocks[0], 'contact_gcn2') else 'reference')
    mode = 'largest_directed_weighted' if path == 'topology' else getattr(model, 'relation_mode', 'smallest_symmetric_binary')
    return {'path': path, 'source_set': source_set, 'variant': model.variant, 'contact_subdir': config.contact_subdir, 'esm_map_subdir': config.esm_map_subdir, 'contact_graph_mode': config.contact_graph_mode, 'esm_graph_mode': mode, 'pooling': getattr(model, 'pooling', 'valid_residue_mean'), 'max_len': config.max_len, 'k': config.k}

def run_experiment(config: RunConfig) -> None:
    root = Path(config.root)
    out_dir = Path(config.out_dir)
    train_rows = load_pickle_table(root / config.train_table)
    train_split_root = root / config.train_split_root
    device = torch.device(config.device if config.device else 'cuda' if torch.cuda.is_available() else 'cpu')
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'config.json').open('w') as handle:
        json.dump(asdict(config), handle, indent=2)
    epoch_fields = ['timestamp', 'contact_graph_mode', 'variant', 'seed', 'fold', 'effective_seed', 'epoch', 'train_loss', *[f'val_{col}' for col in METRIC_COLUMNS]]
    best_fields = ['timestamp', 'contact_graph_mode', 'variant', 'seed', 'fold', 'effective_seed', 'best_epoch_by_val_mcc', *[f'best_val_{col}' for col in METRIC_COLUMNS]]
    print(f'Using device={device}, source_train={len(train_rows)}', flush=True)
    print(f'Writing results to {out_dir}', flush=True)
    for variant in config.variants:
        variant = canonical_variant(variant)
        for seed in config.seeds:
            set_seed(seed)
            names = np.array([row['name'] for row in train_rows])
            split_seed = config.split_seed if config.split_seed is not None else seed
            kfold = KFold(n_splits=5, shuffle=True, random_state=split_seed)
            for fold_id, (train_index, valid_index) in enumerate(kfold.split(names), start=1):
                if config.folds_to_run and fold_id not in config.folds_to_run:
                    continue
                effective_seed = seed
                set_seed(effective_seed)
                run_label = f'variant={variant} seed={seed} fold={fold_id} effective_seed={effective_seed}'
                print(f'\n========== {run_label} ==========', flush=True)
                fold_train_rows = [train_rows[i] for i in train_index]
                fold_valid_rows = [train_rows[i] for i in valid_index]
                train_dataset = ProteinGraphDataset(fold_train_rows, train_split_root, config.max_len, config.k, contact_graph_mode=config.contact_graph_mode, contact_subdir=config.contact_subdir, esm_map_subdir=config.esm_map_subdir, preload=config.preload)
                valid_dataset = ProteinGraphDataset(fold_valid_rows, train_split_root, config.max_len, config.k, contact_graph_mode=config.contact_graph_mode, contact_subdir=config.contact_subdir, esm_map_subdir=config.esm_map_subdir, preload=config.preload)
                train_loader = make_loader(train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers)
                valid_loader = make_loader(valid_dataset, 1, shuffle=False, num_workers=config.num_workers)
                model = CoGraphModel(variant).to(device)
                criterion = CustomLoss(w=2, r=0)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
                best_by_val: Optional[Dict[str, object]] = None
                best_state_by_val: Optional[Dict[str, torch.Tensor]] = None
                start = time.time()
                for epoch in range(1, config.epochs + 1):
                    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                    val_loss, val_true, val_score = evaluate(model, valid_loader, criterion, device)
                    val_metrics = analyze(val_true, val_score, val_loss, threshold=None)
                    row: Dict[str, object] = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'contact_graph_mode': config.contact_graph_mode, 'variant': variant, 'seed': seed, 'fold': fold_id, 'effective_seed': effective_seed, 'epoch': epoch, 'train_loss': train_loss}
                    row.update(metric_prefix('val', val_metrics))
                    write_row(out_dir / 'epoch_metrics.csv', row, epoch_fields)
                    if best_by_val is None or val_metrics['mcc'] > best_by_val['val_mcc']:
                        best_by_val = row
                        best_state_by_val = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
                    print(f"{run_label} epoch={epoch:02d} train_loss={train_loss:.4f} val_mcc={val_metrics['mcc']:.4f} val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f}", flush=True)
                assert best_by_val is not None and best_state_by_val is not None
                if config.save_best_checkpoint:
                    checkpoint_dir = out_dir / 'checkpoints'
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = checkpoint_dir / f'{variant}_seed{seed}_fold{fold_id}.pt'
                    torch.save({'model_state': best_state_by_val, 'config': checkpoint_config(model, config), 'variant': variant, 'contact_graph_mode': config.contact_graph_mode, 'train_table': config.train_table, 'train_split_root': config.train_split_root, 'contact_subdir': config.contact_subdir, 'esm_map_subdir': config.esm_map_subdir, 'max_len': config.max_len, 'k': config.k, 'seed': seed, 'split_seed': split_seed, 'fold': fold_id, 'effective_seed': effective_seed, 'best_epoch_by_val_mcc': best_by_val['epoch'], 'validation_threshold': best_by_val['val_threshold'], 'pooling': getattr(model, 'pooling', 'valid_residue_mean'), 'validation_metrics': {key: best_by_val[f'val_{key}'] for key in METRIC_COLUMNS}}, checkpoint_path)
                best_row: Dict[str, object] = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'contact_graph_mode': config.contact_graph_mode, 'variant': variant, 'seed': seed, 'fold': fold_id, 'effective_seed': effective_seed, 'best_epoch_by_val_mcc': best_by_val['epoch']}
                for key in METRIC_COLUMNS:
                    best_row[f'best_val_{key}'] = best_by_val[f'val_{key}']
                write_row(out_dir / 'best_metrics.csv', best_row, best_fields)
                elapsed = (time.time() - start) / 60.0
                print(f'Finished {run_label} in {elapsed:.1f} min', flush=True)

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train CoGraph-DBP neural paths.')
    parser.add_argument('--root', default='data')
    parser.add_argument('--out-dir', default='results/path_training')
    parser.add_argument('--train-table', default='pkl/Train-DBP.pkl')
    parser.add_argument('--train-split-root', default='Train/Train-1698')
    parser.add_argument('--contact-subdir', default='contact')
    parser.add_argument('--esm-map-subdir', default='map-esm')
    parser.add_argument('--max-len', type=int, default=1000)
    parser.add_argument('--k', type=int, default=25)
    parser.add_argument('--contact-graph-mode', choices=['binary', 'weighted'], default='binary')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-05)
    parser.add_argument('--variants', default='dual_graph')
    parser.add_argument('--seeds', default='525')
    parser.add_argument('--split-seed', type=int, default=525)
    parser.add_argument('--folds-to-run', default='')
    parser.add_argument('--preload', action='store_true')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', default='')
    parser.add_argument('--save-best-checkpoint', action='store_true', help='Save the checkpoint selected by validation MCC.')
    return parser

def main() -> None:
    args = build_arg_parser().parse_args()
    variants = [canonical_variant(item.strip()) for item in args.variants.split(',') if item.strip()]
    config = RunConfig(root=args.root, out_dir=args.out_dir, train_table=args.train_table, train_split_root=args.train_split_root, contact_subdir=args.contact_subdir, esm_map_subdir=args.esm_map_subdir, max_len=args.max_len, k=args.k, contact_graph_mode=args.contact_graph_mode, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, variants=variants, seeds=parse_int_list(args.seeds), split_seed=args.split_seed, folds_to_run=parse_int_list(args.folds_to_run) or None, preload=bool(args.preload), num_workers=args.num_workers, device=args.device, save_best_checkpoint=bool(args.save_best_checkpoint))
    run_experiment(config)
if __name__ == '__main__':
    main()
