#!/usr/bin/env python3
"""Compile and run the shared launch/cache-contract CPU checks; never opens Metal."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compiler', default='c++')
    args = p.parse_args()
    out = args.output.expanduser().resolve()
    if out.exists():
        p.error('Use a new output directory')
    out.mkdir(parents=True)
    files = ['ggml/src/ggml-metal/ggml-metal-bitnet-q-config.h',
             'ggml/src/ggml-metal/ggml-metal-bitnet-q-contract.h', 'bitnet/tests/test_q_kernel_contract.cpp']
    summary = {'CPU_only': True, 'GPU_executed': False, 'device_executed': False,
               'source_SHA256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
               'commands': [], 'pass': False}
    commands = [
        [args.compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT/'ggml/src/ggml-metal'),
         str(ROOT/files[-1]), '-o', str(out/'test-q-kernel-contract')],
        [str(out/'test-q-kernel-contract')],
    ]
    for index, argv in enumerate(commands):
        result = subprocess.run(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = f'{index}.log'; (out/log).write_text(result.stdout)
        summary['commands'].append({'argv': argv, 'exit_code': result.returncode, 'log': log})
        (out/'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
        if result.returncode:
            print(result.stdout, file=sys.stderr)
            return result.returncode if result.returncode > 0 else 1
    summary['pass'] = True
    (out/'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
