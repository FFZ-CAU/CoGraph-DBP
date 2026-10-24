import argparse
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'training'))
import base_path


def prepare(neural_path, structure_path, records_path, training):
    with np.load(neural_path, allow_pickle=False) as data:
        payload = {key: data[key].copy() for key in data.files}
    names = payload['names'].astype(str)
    if records_path is not None:
        rows = base_path.load_pickle_table(records_path)
        by_name = {row['name']: row for row in rows}
        payload['labels'] = np.asarray([by_name[name]['label'] for name in names], dtype=np.int64)
        if training:
            np.testing.assert_array_equal(names, np.asarray([row['name'] for row in rows]))
            splits = list(KFold(n_splits=5, shuffle=True, random_state=525).split(names))
            payload['train_index'], payload['valid_index'] = splits[2]
    elif training:
        raise ValueError('--training requires --records')
    if structure_path is not None:
        with np.load(structure_path, allow_pickle=False) as data:
            np.testing.assert_array_equal(names, data['names'].astype(str))
            payload['structure'] = data['structure'].copy()
    return payload


def main():
    parser = argparse.ArgumentParser(description='Assemble aligned path scores and source partitions.')
    parser.add_argument('--neural', type=Path, required=True)
    parser.add_argument('--structure', type=Path)
    parser.add_argument('--records', type=Path)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = prepare(args.neural, args.structure, args.records, args.training)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)


if __name__ == '__main__':
    main()
