#!/usr/bin/env python3
"""Clean Mac-only core build for the Q contract tests; never execute GPU work."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bitnet/ios'))
from bitnet.ios.build import source_identity


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def native_identity():
    result = source_identity()
    for path in (ROOT / 'bitnet/actquant').glob('*'):
        if path.is_file() and path.suffix in {'.cpp', '.h'}:
            result[str(path.relative_to(ROOT))] = sha(path)
    return result


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--cmake', type=Path, help='CMake executable; otherwise search PATH')
    parser.add_argument('--developer-dir', type=Path, default=Path('/Applications/Xcode.app/Contents/Developer'))
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    build = args.build_dir.expanduser().resolve()
    developer = args.developer_dir.expanduser().resolve()
    require(platform.system() == 'Darwin', 'A native macOS host is required')
    require(ROOT != build and ROOT not in build.parents, 'Build artifacts must stay outside the source checkout')
    require(not build.exists(), 'Refuse an existing build directory, including an empty one')
    require(args.jobs > 0, '--jobs must be positive')
    require((developer / 'usr/bin/xcodebuild').is_file(), 'Select full Xcode with --developer-dir')
    executable = args.cmake or shutil.which('cmake')
    require(executable is not None, 'CMake is not on PATH; pass --cmake /absolute/path/to/cmake')
    cmake = Path(executable).expanduser().resolve()
    require(cmake.is_file() and os.access(cmake, os.X_OK), 'CMake executable is unavailable')
    identity = native_identity()
    recipe_sources = {name: sha(ROOT / name) for name in
                      ('bitnet/tests/build_q_contract.py', 'bitnet/actquant/CMakeLists.txt', 'bitnet/actquant/runner.h')}
    build.mkdir(parents=True, exist_ok=False)
    logs = build / 'build-logs'; logs.mkdir()
    env = os.environ.copy()
    removed = []
    for key in ('DYLD_LIBRARY_PATH', 'DYLD_FRAMEWORK_PATH', 'DYLD_FALLBACK_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES'):
        if key in env:
            removed.append(key); del env[key]
    env.update(DEVELOPER_DIR=str(developer), CCACHE_DISABLE='1')
    report = {'source_root': str(ROOT), 'build_dir': str(build), 'build_completed': False,
              'old_dylib_reused': False, 'GPU_executed': False, 'model_executed': False,
              'performance_measured': False, 'started': datetime.datetime.now().astimezone().isoformat(),
              'DEVELOPER_DIR': str(developer), 'CCACHE_DISABLE': '1',
              'removed_dyld_environment_keys': removed, 'source_files_sha256': identity,
              'build_recipe_sources_sha256': recipe_sources, 'commands': []}
    stage = 'preflight'

    def save():
        write_json(build / 'BUILD.json', report)

    def run(name, argv):
        nonlocal stage
        stage = name
        argv = list(map(str, argv))
        log = logs / ('%02d-%s.log' % (len(report['commands']), name))
        record = {'name': name, 'argv': argv, 'cwd': str(ROOT), 'log': str(log.relative_to(build))}
        report['commands'].append(record); save()
        started = time.monotonic()
        with log.open('x') as stream:
            process = subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record.update(exit_code=process.returncode, elapsed_seconds=time.monotonic() - started)
        save()
        require(process.returncode == 0, name + ' failed with exit ' + str(process.returncode) + '; see ' + str(log))
        return log.read_text(errors='replace').strip()

    try:
        save()
        report['uname'] = run('uname', ['uname', '-s'])
        require(report['uname'] == 'Darwin', 'Require the native Darwin execution environment')
        report['macOS'] = run('macOS', ['sw_vers'])
        report['Xcode'] = run('Xcode', ['xcodebuild', '-version'])
        report['SDK_version'] = run('SDK', ['xcrun', '--sdk', 'macosx', '--show-sdk-version'])
        report['CMake'] = run('CMake', [cmake, '--version'])
        flags = ['-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                 '-DGGML_METAL=ON', '-DLLAMA_BUILD_TESTS=ON']
        report['configure_flags'] = flags
        run('configure', [cmake, '-S', ROOT / 'bitnet/actquant', '-B', build, *flags])
        run('build', [cmake, '--build', build, '--target', 'llama', 'test-backend-ops',
                      'bitnet-actquant-runner', '--parallel', str(args.jobs)])
        require(native_identity() == identity, 'Native sources changed during build; no fresh marker will be issued')
        require(all(sha(ROOT / name) == value for name, value in recipe_sources.items()),
                'Build recipe changed during compilation')
        for filename in ('CMakeCache.txt', 'compile_commands.json'):
            require((build / filename).is_file(), 'Required configure record missing: ' + filename)
        cache = (build / 'CMakeCache.txt').read_text()
        for key, value in [('CMAKE_BUILD_TYPE', 'Release'), ('CMAKE_EXPORT_COMPILE_COMMANDS', 'ON'),
                           ('GGML_METAL', 'ON'), ('LLAMA_BUILD_TESTS', 'ON')]:
            require(any(line.startswith(key + ':') and line.endswith('=' + value) for line in cache.splitlines()),
                    'Effective CMake option differs: ' + key)
        product_paths = set((build / 'bin').glob('*.dylib'))
        product_paths.update(build / 'bin' / name for name in ('test-backend-ops', 'bitnet-actquant-runner'))
        required = ['libllama.dylib', 'libggml.dylib', 'libggml-base.dylib', 'libggml-metal.dylib', 'libggml-cpu.dylib',
                    'test-backend-ops', 'bitnet-actquant-runner']
        require(all((build / 'bin' / name).is_file() for name in required), 'A required fresh product is missing')
        for path in product_paths:
            require(path.is_file() and path.stat().st_size > 0 and path.resolve().is_relative_to(build),
                    'Product is empty or resolves outside the new build: ' + path.name)
        products = {str(path.relative_to(build)): sha(path) for path in sorted(product_paths)}
        marker = {'source_root': str(ROOT), 'source_files_sha256': identity,
                  'products_sha256': products, 'build_completed': True,
                  'old_dylib_reused': False, 'GPU_executed': False}
        report.update(build_completed=True, products_sha256=products, exit_code=0,
                      finished=datetime.datetime.now().astimezone().isoformat(),
                      configure_records_sha256={name: sha(build / name) for name in ('CMakeCache.txt', 'compile_commands.json')})
        save()
        temporary = build / 'FRESH_CORE_BUILD.json.tmp'; write_json(temporary, marker)
        temporary.rename(build / 'FRESH_CORE_BUILD.json')
        print(json.dumps({'build_completed': True, 'GPU_executed': False,
                          'marker': str(build / 'FRESH_CORE_BUILD.json'),
                          'source_files': len(identity), 'products': len(products)}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report.update(build_completed=False, failed_stage=stage, error=str(error), exit_code=1,
                      finished=datetime.datetime.now().astimezone().isoformat())
        save()
        print(json.dumps({'build_completed': False, 'error': str(error), 'failed_stage': stage,
                          'log_root': str(logs)}, indent=2), file=sys.stderr)
        return 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
