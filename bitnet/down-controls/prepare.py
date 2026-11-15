#!/usr/bin/env python3
"""Bind the fixed real stock norm input and down weights for a 96-row control."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

MODEL = '52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2'
INPUT = 'c9e096fc21f1e6ab4dcc1c4329925942219f848b4ab082dc30b930fe2bf43e7a'
WEIGHT = 'c032780f9245f4ecfe37fe117f0b6f16b5a76cfb72734c60b1f7e4beabe99888'
FAILURES = [5, 338, 356, 362, 455, 502, 882, 1148, 1508, 1830, 2155, 2508]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weight', type=Path, required=True)
    parser.add_argument('--weight-sha256', required=True, help='Independently verified GGUF tensor-byte digest')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--libraries', type=Path, help='Fresh builder output; bind exact candidate library hashes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing an existing output directory')
    if args.weight_sha256 != WEIGHT or args.weight.stat().st_size != 4561920 or sha(args.weight) != WEIGHT:
        parser.error('Down weight extent or verified hash mismatch')
    if args.input.stat().st_size != 27648 or sha(args.input) != INPUT:
        parser.error('Expected the captured independent stock layer-0 / step-16 norm input')
    libraries = {}
    if args.libraries:
        build = json.loads((args.libraries / 'BUILD.json').read_text())
        if not build.get('build_completed') or build.get('exit_code') != 0:
            parser.error('Candidate libraries were not built successfully')
        for route, key in (('S-wrap', 'wrap'), ('correction-end', 'correction-end')):
            entry = build['products'][key]
            if sha(args.libraries / entry['file']) != entry['sha256']:
                parser.error('Candidate library checksum mismatch')
            libraries[route] = {'file': entry['file'], 'sha256': entry['sha256']}
    args.output.mkdir(parents=True)
    shutil.copy2(args.weight, args.output / 'w.tq2')
    shutil.copy2(args.input, args.output / 'x.f32')
    manifest = {'schema': 1, 'model_sha256': MODEL, 'layer': 0, 'step': 16,
                'history': 'independent', 'owner_route': 'S', 'shape': [2560, 6912],
                'weight': {'file': 'w.tq2', 'bytes': 4561920, 'sha256': args.weight_sha256},
                'input': {'file': 'x.f32', 'bytes': 27648, 'sha256': INPUT},
                'known_failed_rows': FAILURES, 'row_groups': sorted({(i // 8) * 8 for i in FAILURES}),
                'rows_per_group': 8, 'libraries': libraries,
                'tolerance': {'abs': 2e-4, 'relative': 2e-5}, 'raw_fixture_stays_local': True}
    (args.output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared 12 real eight-row groups, 96 outputs/path; no GPU work run.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
