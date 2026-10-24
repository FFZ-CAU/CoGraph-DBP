from __future__ import annotations
import torch
import base_path as base
import reference_path
import residue_path
import topology_path

def make_model(config):
    path = config['path']
    variant = config['variant']
    if path == 'reference':
        return reference_path.ReferenceModel(variant, pooling=config['pooling'])
    if path == 'residue':
        return residue_path.ResidueDualPLMModel(variant)
    if path == 'topology':
        block, canonical = (base.GraphBlock, base.canonical_variant)
        try:
            base.GraphBlock = topology_path.TopologyGraphBlock
            base.canonical_variant = topology_path.canonical_topology_variant
            return base.CoGraphModel(variant)
        finally:
            base.GraphBlock, base.canonical_variant = (block, canonical)
    raise ValueError(f'unsupported predictor path: {path}')

def load_model(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    model = make_model(payload['config'])
    model.load_state_dict(payload['model_state'], strict=True)
    return (model.eval().to(device), payload['config'])
