from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import base_path as base

def canonical_variant(variant: str) -> str:
    if variant not in {'dual_graph', 'dual_plm_anchor_graph', 'dual_plm_concat_graph'}:
        raise ValueError(f'unsupported residue-path variant: {variant}')
    return variant

class DualPLMProteinGraphDataset(Dataset):

    def __init__(self, rows: Sequence[Dict[str, object]], split_root: Path, max_len: int, k: int, contact_graph_mode: str='binary', contact_subdir: str='contact', esm_map_subdir: str='map-esm', preload: bool=False) -> None:
        self.rows = list(rows)
        self.split_root = Path(split_root)
        self.max_len = max_len
        self.k = k
        self.contact_graph_mode = contact_graph_mode
        self.contact_subdir = contact_subdir
        self.esm_map_subdir = esm_map_subdir
        feature_root = os.environ.get('ESM2_RESIDUE_ROOT')
        if not feature_root:
            raise RuntimeError('ESM2_RESIDUE_ROOT is required')
        self.esm2_dir = Path(feature_root) / self.split_root.name
        self.cache: Dict[str, Tuple[np.ndarray, ...]] = {}
        if preload:
            for row in self.rows:
                name = str(row['name'])
                self.cache[name] = self._load_item_arrays(name)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        name = str(row['name'])
        arrays = self.cache.get(name)
        if arrays is None:
            arrays = self._load_item_arrays(name)
        prot5, esm2, contact, esm_map, real_len = arrays
        usable_len = min(int(row['seq_len']), int(real_len), self.max_len)
        return {'name': name, 'prot5': prot5, 'esm2': esm2, 'contact': contact, 'esm_map': esm_map, 'label': np.int64(row['label']), 'length': np.int64(usable_len)}

    def _load_item_arrays(self, name: str) -> Tuple[np.ndarray, ...]:
        prot5_raw = np.load(self.split_root / 'feature' / 'feature-prot5-xl' / f'{name}.npy')
        esm2_raw = np.load(self.esm2_dir / f'{name}.npy')
        prot5_len = min(int(prot5_raw.shape[0]), self.max_len)
        esm2_len = min(int(esm2_raw.shape[0]), self.max_len)
        real_len = min(prot5_len, esm2_len)
        if abs(prot5_len - esm2_len) > 1:
            raise ValueError(f'unaligned residue features for {name}: ProtT5={prot5_len}, ESM2={esm2_len}, max_len={self.max_len}')
        prot5 = base.pad_2d(prot5_raw, self.max_len, 1024)
        esm2 = base.pad_2d(esm2_raw, self.max_len, 1280)
        contact_raw = np.load(self.split_root / self.contact_subdir / f'{name}.npy')
        contact = base.build_contact_graph(contact_raw, self.k, self.contact_graph_mode)
        contact = base.normalize_adj(contact.astype(np.float32))
        contact = base.pad_square(contact, self.max_len)
        esm_map_raw = np.load(self.split_root / self.esm_map_subdir / f'{name}.npy')
        esm_map = base.distance_to_knn_matrix(esm_map_raw, self.k, symmetric=True)
        esm_map = base.normalize_adj(esm_map.astype(np.float32))
        esm_map = base.pad_square(esm_map, self.max_len)
        valid_nodes = min(real_len, self.max_len)
        valid_index = np.arange(valid_nodes)
        contact[valid_index, valid_index] = 1.0
        esm_map[valid_index, valid_index] = 1.0
        return (prot5.astype(np.float32), esm2.astype(np.float32), contact.astype(np.float32), esm_map.astype(np.float32), real_len)

class ResidueDualPLMModel(nn.Module):

    def __init__(self, variant: str, num_graph_blocks: int=2) -> None:
        super().__init__()
        self.variant = canonical_variant(variant)
        self.path = 'residue'
        if self.variant == 'dual_plm_concat_graph':
            self.prot5_256 = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256), nn.GELU())
            self.esm2_256 = nn.Sequential(nn.LayerNorm(1280), nn.Linear(1280, 256), nn.GELU())
            self.concat_norm = nn.LayerNorm(512)
        else:
            self.prot5_512 = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 512), nn.GELU())
            self.esm2_512 = nn.Sequential(nn.LayerNorm(1280), nn.Linear(1280, 512), nn.GELU())
            self.anchor_scale = nn.Parameter(torch.tensor(-2.0))
            self.anchor_norm = nn.LayerNorm(512)
        self.blocks = nn.ModuleList([base.GraphBlock('dual_graph') for _ in range(num_graph_blocks)])
        self.classifier = nn.Sequential(nn.Linear(512, 128), nn.LeakyReLU(), nn.Dropout(0.2), nn.Linear(128, 32), nn.LeakyReLU(), nn.Dropout(0.2), nn.Linear(32, 2))

    def forward(self, prot5: torch.Tensor, esm2: torch.Tensor, contact_adj: torch.Tensor, esm_adj: torch.Tensor, lengths: Optional[torch.Tensor]=None) -> torch.Tensor:
        node_mask: Optional[torch.Tensor] = None
        if lengths is not None:
            positions = torch.arange(prot5.shape[1], device=prot5.device).unsqueeze(0)
            node_mask = (positions < lengths.unsqueeze(1)).to(prot5.dtype).unsqueeze(-1)
        if self.variant == 'dual_plm_concat_graph':
            hidden = self.concat_norm(torch.cat([self.prot5_256(prot5), self.esm2_256(esm2)], dim=-1))
        else:
            scale = torch.sigmoid(self.anchor_scale)
            hidden = self.anchor_norm(self.prot5_512(prot5) + scale * self.esm2_512(esm2))
        if node_mask is not None:
            hidden = hidden * node_mask
        for block in self.blocks:
            hidden = block(hidden, contact_adj, esm_adj, node_mask=node_mask)
        pooled = base.masked_node_mean(hidden, node_mask)
        return self.classifier(pooled)

def batch_to_device(batch: Dict[str, object], device: torch.device):
    return (batch['prot5'].to(device=device, dtype=torch.float32), batch['esm2'].to(device=device, dtype=torch.float32), batch['contact'].to(device=device, dtype=torch.float32), batch['esm_map'].to(device=device, dtype=torch.float32), batch['label'].to(device=device, dtype=torch.long), batch['length'].to(device=device, dtype=torch.long))

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    losses = []
    for batch in loader:
        prot5, esm2, contact, esm_map, labels, lengths = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(prot5, esm2, contact, esm_map, lengths)
        loss = criterion(logits, labels.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    y_true = []
    y_score = []
    for batch in loader:
        prot5, esm2, contact, esm_map, labels, lengths = batch_to_device(batch, device)
        logits = model(prot5, esm2, contact, esm_map, lengths)
        loss = criterion(logits, labels.view(-1))
        probability = torch.softmax(logits, dim=1)[:, 1]
        losses.append(float(loss.detach().cpu()))
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        y_score.extend(probability.detach().cpu().numpy().astype(float).tolist())
    return (float(np.mean(losses)) if losses else 0.0, y_true, y_score)

def install() -> None:
    base.canonical_variant = canonical_variant
    base.ProteinGraphDataset = DualPLMProteinGraphDataset
    base.CoGraphModel = ResidueDualPLMModel
    base.train_one_epoch = train_one_epoch
    base.evaluate = evaluate
if __name__ == '__main__':
    install()
    base.main()
