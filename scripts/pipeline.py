import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('Train-1698', 'Train-2104')


def run(module, *args):
    subprocess.run([sys.executable, '-B', str(ROOT/module), *map(str, args)], check=True)


def infer_neural(source, records, split, esm2, models, output, device):
    run('scripts/predict_neural_paths.py', '--source-set', source, '--records', records,
        '--split-root', split, '--esm2-dir', esm2, '--models', models,
        '--output', output, '--device', device)


def main():
    parser = argparse.ArgumentParser(description='CoGraph-DBP training and prediction.')
    sub = parser.add_subparsers(dest='action', required=True)
    train = sub.add_parser('train')
    train.add_argument('--data-root', type=Path, required=True)
    train.add_argument('--esm2-root', type=Path, required=True)
    train.add_argument('--graph-features', type=Path, required=True)
    train.add_argument('--out-dir', type=Path, required=True)
    train.add_argument('--device', default='cuda:0')
    predict = sub.add_parser('predict')
    predict.add_argument('--source-set', choices=SOURCES, required=True)
    predict.add_argument('--records', type=Path, required=True)
    predict.add_argument('--split-root', type=Path, required=True)
    predict.add_argument('--esm2-dir', type=Path, required=True)
    predict.add_argument('--features', type=Path, required=True)
    predict.add_argument('--run-dir', type=Path, required=True)
    predict.add_argument('--out-dir', type=Path, required=True)
    predict.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if args.action == 'train':
        specs = []
        for source in SOURCES:
            directory = args.out_dir/source
            run('scripts/train.py', '--source-set', source, '--root', args.data_root,
                '--esm2-root', args.esm2_root, '--out-dir', directory, '--device', args.device)
            records = args.data_root/'pkl'/('Train-DBP.pkl' if source == 'Train-1698' else 'Train-2104.pkl')
            neural = directory/'neural_scores.npz'
            infer_neural(source, records, args.data_root/'Train'/source, args.esm2_root/source,
                         directory/'models', neural, args.device)
            scores = directory/'source_scores.npz'
            run('scripts/prepare_scores.py', '--neural', neural, '--records', records,
                '--training', '--output', scores)
            specs.extend(['--dataset', f'{source}|{scores}|{args.graph_features / (source + ".csv")}'])
        run('training/structure_path.py', *specs, '--out-dir', args.out_dir/'structure')
        for source in SOURCES:
            directory = args.out_dir/source
            structural = directory/'structure_scores.npz'
            run('training/replay_structure.py', '--parameters', args.out_dir/'structure'/f'{source}_structure.json',
                '--scores', directory/'source_scores.npz', '--features', args.graph_features/f'{source}.csv',
                '--output', structural)
            records = args.data_root/'pkl'/('Train-DBP.pkl' if source == 'Train-1698' else 'Train-2104.pkl')
            run('scripts/prepare_scores.py', '--neural', directory/'neural_scores.npz',
                '--structure', structural, '--records', records, '--training',
                '--output', directory/'path_scores.npz')
            run('scripts/fuse.py', 'fit', '--source-set', source, '--scores', directory/'path_scores.npz',
                '--output', directory/'fusion.json')
    else:
        source = args.source_set
        neural = args.out_dir/'neural_scores.npz'
        infer_neural(source, args.records, args.split_root, args.esm2_dir,
                     args.run_dir/source/'models', neural, args.device)
        structural = args.out_dir/'structure_scores.npz'
        run('training/replay_structure.py', '--parameters', args.run_dir/'structure'/f'{source}_structure.json',
            '--scores', neural, '--features', args.features, '--output', structural)
        run('scripts/prepare_scores.py', '--neural', neural, '--structure', structural,
            '--output', args.out_dir/'path_scores.npz')
        run('scripts/fuse.py', 'predict', '--scores', args.out_dir/'path_scores.npz',
            '--model', args.run_dir/source/'fusion.json', '--output', args.out_dir/'predictions.npz')


if __name__ == '__main__':
    main()
