import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def training_commands(source, data_root, esm2_dir, out_dir, device):
    table = 'Train-DBP.pkl' if source == 'Train-1698' else 'Train-2104.pkl'
    common = ['--root', str(data_root), '--train-table', f'pkl/{table}',
              '--train-split-root', f'Train/{source}', '--max-len', '1000',
              '--k', '25', '--epochs', '25', '--batch-size', '16', '--lr', '5e-5',
              '--seeds', '525', '--split-seed', '525', '--folds-to-run', '3',
              '--save-best-checkpoint', '--preload', '--device', device]
    variants = 'no_graph,contact_only,esm_map_only,identity_graph_control,dual_graph'
    residue_variant = 'dual_plm_anchor_graph' if source == 'Train-1698' else 'dual_plm_concat_graph'
    specs = [
        ('reference', 'reference_path.py', ['--source-set', source, '--contact-subdir',
            'contact' if source == 'Train-1698' else 'map', '--esm-map-subdir', 'map-esm',
            '--contact-graph-mode', 'weighted', '--variants', 'dual_graph']),
        ('topology', 'topology_path.py', ['--contact-subdir', 'contact', '--esm-map-subdir',
            'map', '--contact-graph-mode', 'weighted', '--variants', variants]),
        ('residue', 'residue_path.py', ['--contact-subdir', 'contact', '--esm-map-subdir',
            'map', '--contact-graph-mode', 'binary', '--variants', residue_variant])]
    return [(role, [sys.executable, str(ROOT/'training'/module), *common,
                   '--out-dir', str(out_dir/role), *extra]) for role, module, extra in specs]


def main():
    parser = argparse.ArgumentParser(description='Train neural paths and structural auxiliary models.')
    parser.add_argument('--source-set', choices=['Train-1698', 'Train-2104'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--esm2-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, ESM2_RESIDUE_ROOT=str(args.esm2_root.resolve()))
    for role, command in training_commands(args.source_set, args.root.resolve(), args.esm2_root.resolve(), args.out_dir.resolve(), args.device):
        subprocess.run(command, check=True, env=env)
    models = args.out_dir/'models'
    models.mkdir()
    residue = 'dual_plm_anchor_graph' if args.source_set == 'Train-1698' else 'dual_plm_concat_graph'
    locations = {'reference': ('reference', 'dual_graph'), 'residue': ('residue', residue),
                 'topology': ('topology', 'dual_graph')}
    locations.update({role: ('topology', role) for role in ['no_graph', 'contact_only', 'esm_map_only', 'identity_graph_control']})
    for role, (directory, variant) in locations.items():
        source = args.out_dir/directory/'checkpoints'/f'{variant}_seed525_fold3.pt'
        shutil.copy2(source, models/f'{args.source_set}_{role}.pt')


if __name__ == '__main__':
    main()
