from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
import base_path as base
import checkpoints
import topology_path

def load_inputs(row, split_root, esm2_dir, config, residue=False):
    name = str(row['name'])
    prot = np.load(split_root / 'feature/feature-prot5-xl' / f'{name}.npy', allow_pickle=False)
    if prot.ndim != 2 or prot.shape[1] != 1024:
        raise ValueError(f'{name}: ProtT5 must have 1024 channels')
    graph_nodes = min(len(prot), config['max_len'])
    length = min(int(row['seq_len']), graph_nodes)
    if length < 1:
        raise ValueError(f'{name}: empty usable input')
    raw = np.load(split_root / config['contact_subdir'] / f'{name}.npy', allow_pickle=False)
    contact = base.build_contact_graph(raw, config['k'], config['contact_graph_mode'])
    raw = np.load(split_root / config['esm_map_subdir'] / f'{name}.npy', allow_pickle=False)
    mode = config['esm_graph_mode']
    if mode == 'largest_symmetric_weighted':
        relation = base.attention_to_knn_matrix(raw, config['k'], symmetric=True, weighted=True)
    elif mode == 'largest_directed_weighted':
        relation = topology_path.directed_high_attention_graph(raw, config['k'])
    elif mode == 'smallest_symmetric_binary':
        relation = base.distance_to_knn_matrix(raw, config['k'], symmetric=True)
    else:
        raise ValueError(f'unsupported relation mode: {mode}')
    arrays = [base.pad_2d(prot, config['max_len'], 1024)]
    if residue:
        esm = np.load(esm2_dir / f'{name}.npy', allow_pickle=False)
        if esm.ndim != 2 or esm.shape[1] != 1280:
            raise ValueError(f'{name}: ESM2 must have 1280 channels')
        if abs(min(len(prot), config['max_len']) - min(len(esm), config['max_len'])) > 1:
            raise ValueError(f'{name}: unaligned ProtT5 and ESM2')
        graph_nodes = min(graph_nodes, len(esm))
        length = min(length, graph_nodes)
        arrays.append(base.pad_2d(esm, config['max_len'], 1280))
    for graph in [contact, relation]:
        graph = base.pad_square(base.normalize_adj(graph.astype(np.float32)), config['max_len'])
        np.fill_diagonal(graph[:graph_nodes, :graph_nodes], 1.0)
        arrays.append(graph)
    return (arrays, length)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-set', choices=['Train-1698', 'Train-2104'], required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--identifiers', type=Path)
    parser.add_argument('--split-root', type=Path, required=True)
    parser.add_argument('--esm2-dir', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=ROOT / 'models')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {str(row['name']): row for row in base.load_pickle_table(args.records)}
    if args.identifiers is None:
        names = list(rows)
    else:
        with np.load(args.identifiers, allow_pickle=False) as data:
            names = data['names'].astype(str).tolist()
    if len(names) != len(set(names)):
        raise ValueError('identifier list contains duplicates')
    if set(names) - rows.keys():
        raise ValueError('some requested identifiers are missing from the records')
    predictors = {}
    for role in ['reference', 'residue', 'topology', 'no_graph', 'contact_only', 'esm_map_only', 'identity_graph_control']:
        predictors[role] = checkpoints.load_model(args.models / f'{args.source_set}_{role}.pt', args.device)
    results = {role: [] for role in predictors}
    device = torch.device(args.device)
    with torch.no_grad():
        for name in names:
            for role, (model, config) in predictors.items():
                arrays, length = load_inputs(rows[name], args.split_root, args.esm2_dir, config, role == 'residue')
                inputs = [torch.as_tensor(a, dtype=torch.float32, device=device).unsqueeze(0) for a in arrays]
                logits = model(*inputs, torch.tensor([length], device=device))
                results[role].append(float(torch.softmax(logits, dim=1)[0, 1].cpu()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, names=np.asarray(names), **{role: np.asarray(p, dtype=np.float64) for role, p in results.items()}, dual_graph=np.asarray(results['topology'], dtype=np.float64))
if __name__ == '__main__':
    main()
