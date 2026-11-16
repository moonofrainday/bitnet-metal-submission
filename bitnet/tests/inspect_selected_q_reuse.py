#!/usr/bin/env python3
"""Inventory reuse opportunities for the selected Q shape; never infer latency."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
MODEL_SHA = '52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('--output must be fresh')
    digest = hashlib.sha256()
    with args.model.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != MODEL_SHA:
        p.error('This inventory is scoped to the frozen public GGUF')
    sys.path.insert(0, str(ROOT / 'gguf-py'))
    from gguf import GGUFReader
    reader = GGUFReader(args.model)
    suffixes = ('attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
                'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight')
    weights = [{'name': tensor.name, 'type': tensor.tensor_type.name,
                'shape': list(map(int, tensor.shape)), 'bytes': int(tensor.n_bytes)}
               for tensor in reader.tensors if tensor.name.endswith(suffixes)]
    result = {'model_sha256': MODEL_SHA, 'projection_weights': len(weights),
              'types': dict(collections.Counter(w['type'] for w in weights)),
              'Q_weights': sum(w['name'].endswith('attn_q.weight') for w in weights),
              'same_2560_square_TQ2_weights': sum(w['type'] == 'TQ2_0' and w['shape'] == [2560, 2560]
                                               for w in weights),
              'current_target': 'Layer-0 Qcur-0 only: one N1 projection per decode token',
              'interpretation': 'Structural reuse opportunity only; activation/guard/binding/numerical and performance qualification of other nodes is not implied. Counts and bytes are not GPU time or critical-path share.',
              'weights': weights}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'weights'}))


if __name__ == '__main__':
    main()
