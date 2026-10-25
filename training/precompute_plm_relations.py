import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import time

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer


def residue_attention(attention, special_tokens_mask):
    positions = torch.nonzero(special_tokens_mask == 0, as_tuple=False).flatten()
    mean_attention = attention.mean(dim=0)
    return mean_attention.index_select(0, positions).index_select(1, positions)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Generate residue-aligned ProtT5 relation matrices.')
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--progress-every', type=int, default=25)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.out_dir.glob('*.npy'))
    metadata_path = args.out_dir / 'generation_config.json'
    if existing and not args.resume:
        raise FileExistsError(args.out_dir)
    if existing and not metadata_path.is_file():
        raise ValueError('Existing arrays have no matching generation metadata.')
    with args.records.open('rb') as handle:
        records = pickle.load(handle)
    config = json.loads((args.model / 'config.json').read_text())
    if (config['num_layers'], config['num_heads'], config['d_model']) != (24, 32, 1024):
        raise ValueError('The model must be ProtT5-XL with 24 layers, 32 heads and 1024 channels.')
    model_files = [p for p in args.model.iterdir() if p.is_file() and
                   (p.suffix in {'.json', '.model'} or
                    (p.name.startswith('pytorch_model') and p.suffix == '.bin') or
                    (p.name.startswith('model') and p.suffix == '.safetensors'))]
    identity = {'schema_version': 1, 'token_selection': 'amino_acid_tokens',
                'records_sha256': file_hash(args.records),
                'model_files': {p.name: file_hash(p) for p in sorted(model_files)}}
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != identity:
        raise ValueError('Input records or model files differ from the generation metadata.')
    metadata_path.write_text(json.dumps(identity, indent=2), encoding='utf8')
    tokenizer = T5Tokenizer.from_pretrained(str(args.model), local_files_only=True)
    model = T5EncoderModel.from_pretrained(str(args.model), local_files_only=True,
                                         attn_implementation='eager').eval().to(args.device)
    torch.set_num_threads(4)
    started = time.time()
    entries = []
    for index, (name, record) in enumerate(records.items(), 1):
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError(f'Invalid protein identifier: {name}')
        sequence = record[0]
        expected = np.load(args.features / f'{name}.npy', mmap_mode='r', allow_pickle=False).shape[0]
        if expected != len(sequence):
            raise ValueError(f'Sequence and residue feature lengths differ for {name}')
        target = args.out_dir / f'{name}.npy'
        if args.resume and target.exists():
            matrix = np.load(target, allow_pickle=False)
            if matrix.shape != (expected, expected) or not np.isfinite(matrix).all():
                raise ValueError(f'Invalid relation matrix: {target}')
        else:
            text = ' '.join(re.sub(r'[UZOB]', 'X', sequence))
            encoded = tokenizer(text, add_special_tokens=True, return_tensors='pt',
                                return_special_tokens_mask=True)
            mask = encoded.pop('special_tokens_mask')[0].to(args.device)
            if int((mask == 0).sum()) != expected:
                raise ValueError(f'Token and residue counts differ for {name}')
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = model(**encoded, output_attentions=True, return_dict=True)
                matrix = residue_attention(output.attentions[-1][0], mask).float().cpu().numpy()
            if matrix.shape != (expected, expected) or not np.isfinite(matrix).all():
                raise ValueError(f'Invalid generated matrix for {name}')
            temporary = target.with_suffix('.tmp.npy')
            np.save(temporary, matrix, allow_pickle=False)
            os.replace(temporary, target)
            del output
        entries.append({'name': name, 'length': expected,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        if index % args.progress_every == 0 or index == len(records):
            status = {'processed': index, 'total': len(records),
                      'seconds': round(time.time() - started, 2)}
            (args.out_dir / 'progress.json').write_text(json.dumps(status), encoding='utf8')
            print(json.dumps(status), flush=True)
    metadata = {'model_layers': model.config.num_layers, 'model_heads': model.config.num_heads,
                'attention_layer': model.config.num_layers, 'head_reduction': 'mean',
                'tokens': 'residues only', 'entries': entries}
    (args.out_dir / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()

