#!/usr/bin/env python3
"""Build this patched source in a fresh directory and run the TQ2_0 regression."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cmake', default='cmake')
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('This smoke entry requires Apple Silicon macOS.')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    for path in (args.build, args.output):
        if path.exists() and any(path.iterdir()):
            parser.error('Refusing to overwrite a nonempty directory: ' + str(path))
    cmake = shutil.which(args.cmake)
    if not cmake:
        parser.error('CMake not found; use --cmake /path/to/cmake')
    args.build = args.build.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if not env.get('DEVELOPER_DIR'):
        active = subprocess.check_output(['xcode-select', '-p'], text=True).strip()
        if 'CommandLineTools' in active:
            parser.error('Set DEVELOPER_DIR to a complete Xcode installation.')
    records = []
    def run(name, command, allowed=(0,)):
        with (args.output / (name + '.log')).open('w') as log:
            log.write('$ ' + shlex.join(map(str, command)) + '\n')
            log.flush()
            result = subprocess.run(list(map(str, command)), cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        records.append({'stage': name, 'command': list(map(str, command)), 'exit_code': result.returncode})
        (args.output / 'RUN.json').write_text(json.dumps({'stages': records}, indent=2) + '\n')
        if result.returncode not in allowed:
            raise SystemExit(result.returncode)
    config = [cmake, '-S', root, '-B', args.build, '-DCMAKE_BUILD_TYPE=Release',
              '-DCMAKE_OSX_ARCHITECTURES=arm64', '-DGGML_METAL=ON', '-DGGML_METAL_EMBED_LIBRARY=ON',
              '-DGGML_BLAS=ON', '-DGGML_OPENMP=OFF', '-DLLAMA_BUILD_TESTS=ON',
              '-DLLAMA_BUILD_COMMON=ON', '-DLLAMA_BUILD_TOOLS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF',
              '-DLLAMA_BUILD_APP=OFF', '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_MTMD=OFF', '-DLLAMA_OPENSSL=OFF']
    run('configure', config)
    run('build', [cmake, '--build', args.build, '--target', 'test-backend-ops', 'llama', '--parallel', args.jobs])
    # This pinned executable prints usage and returns 1 for --help.
    run('help', [args.build / 'bin/test-backend-ops', '--help'], allowed=(0, 1))
    help_text = (args.output / 'help.log').read_text()
    if '-p <params regex>' not in help_text or 'MTL0' not in help_text:
        raise SystemExit('Unexpected backend test help; refusing an unchecked filter.')
    if not args.build_only:
        clean = env.copy()
        for key in list(env):
            if key.startswith('GGML_METAL_BITNET_'):
                del env[key]
        run('tq2-regression', [args.build / 'bin/test-backend-ops', 'test', '-b', 'MTL0', '-o', 'MUL_MAT', '-p', 'type_a=tq2_0'])
        env.clear()
        env.update(clean)
        text = (args.output / 'tq2-regression.log').read_text()
        if '59/59 tests passed' not in text:
            print('Inspect regression counts; expected the pinned 59/59 test set.', file=sys.stderr)
            return 2
    binaries = {}
    for path in sorted((args.build / 'bin').glob('*')):
        if path.is_file():
            binaries[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / 'BINARIES.json').write_text(json.dumps(binaries, indent=2) + '\n')
    print('Build and selected checks completed. Logs: ' + str(args.output))
    return 0


if __name__ == '__main__':
    sys.exit(main())
