from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
import structure_path as structure
import structure_support as support

def apply_model(names, scores, rows, parameters):
    matrices = {}
    for kind in ['attention', 'dual_contact']:
        matrix, columns = structure.derived_matrix(names, rows, scores, parameters['anchor'], kind)
        stage = parameters[kind]
        if columns != stage['columns']:
            raise ValueError(f'{kind} feature-column order differs from the saved model')
        weights = np.asarray(stage['weights'], dtype=np.float64)
        scale = np.asarray(stage['scale'], dtype=np.float64)
        if matrix.shape[1] != len(weights) or np.any(scale <= 0):
            raise ValueError(f'invalid {kind} parameter dimensions or scale')
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f'non-finite {kind} input')
        matrices[kind] = np.clip(matrix / scale @ weights, -stage['clip'], stage['clip'])
    z = support.logit(scores[parameters['anchor']])
    return expit(z + matrices['attention'] + matrices['dual_contact'])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters', type=Path, required=True)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    parameters = json.loads(args.parameters.read_text(encoding='utf-8'))
    with np.load(args.scores, allow_pickle=False) as data:
        names = data['names'].astype(str).tolist()
        scores = {variant: data[variant if variant in data.files else 'raw_' + variant] for variant in support.VARIANTS}
    rows = structure.read_feature_rows(args.features)
    probabilities = apply_model(names, scores, rows, parameters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, names=np.asarray(names), structure=probabilities)
if __name__ == '__main__':
    main()
