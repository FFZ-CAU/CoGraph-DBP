from __future__ import annotations
import argparse
import sys
import base_path as base

class ReferenceModel(base.CoGraphModel):

    def __init__(self, variant: str, num_graph_blocks: int=2, pooling: str='valid_residue_mean') -> None:
        super().__init__(variant, num_graph_blocks)
        if pooling not in {'valid_residue_mean', 'all_padded_positions_mean'}:
            raise ValueError(f'unsupported pooling: {pooling}')
        self.pooling = pooling
        self.path = 'reference'

    def forward(self, prot5, contact_adj, esm_adj, lengths=None):
        if self.pooling == 'all_padded_positions_mean':
            lengths = None
        return super().forward(prot5, contact_adj, esm_adj, lengths)

def largest_symmetric_weighted(matrix, k, symmetric=True):
    return base.attention_to_knn_matrix(matrix, k, symmetric=symmetric, weighted=True)

def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--source-set', choices=['Train-1698', 'Train-2104'], required=True)
    args, remaining = parser.parse_known_args()
    pooling = 'all_padded_positions_mean' if args.source_set == 'Train-1698' else 'valid_residue_mean'
    if args.source_set == 'Train-2104':
        base.distance_to_knn_matrix = largest_symmetric_weighted

    class ConfiguredReference(ReferenceModel):

        def __init__(self, variant, num_graph_blocks=2):
            super().__init__(variant, num_graph_blocks, pooling=pooling)
            self.relation_mode = 'largest_symmetric_weighted' if args.source_set == 'Train-2104' else 'smallest_symmetric_binary'
    base.CoGraphModel = ConfiguredReference
    sys.argv = [sys.argv[0], *remaining]
    base.main()
if __name__ == '__main__':
    main()
