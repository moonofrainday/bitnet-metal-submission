#!/usr/bin/env python3
"""Compile two iPhoneOS down controls; do not run GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'bitnet/ios'))
from common import Commands, developer_environment, require, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--developer-dir', type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    require(not output.exists(), 'Refusing an existing output directory')
    identity = json.loads((HERE / 'SOURCES.json').read_text())
    for name, digest in identity['sources'].items():
        require(sha256(HERE / 'shaders' / name) == digest, 'Shader source changed: ' + name)
    stock = (ROOT / identity['stock_source']).read_text()
    start = stock.index('template<int nr0, typename args_t>\nvoid kernel_mul_mv_tq2_0_f32_impl(')
    end = stock.index('\n[[host_name("kernel_mul_mv_tq2_0_f32")]]', start)
    helper = stock[start:end].rstrip() + '\n'
    require(hashlib.sha256(helper.encode()).hexdigest() == identity['stock_helper_sha256'], 'Stock helper changed')
    env = developer_environment(args.developer_dir)
    output.mkdir(parents=True)
    commands = Commands(output / 'logs', env)
    record = {'build_completed': False, 'GPU_executed': False, 'sdk': 'iphoneos',
              'source_identity': identity, 'products': {}, 'function': 'stock_down_control',
              'math': 'Both variants use the same default -O3 Metal options; AIR is static evidence only.'}
    try:
        commands.run('metal-version', ['xcrun', '--sdk', 'iphoneos', 'metal', '--version'])
        for variant, source in (('wrap', 'wrap.metal'), ('correction-end', 'correction-end.metal')):
            air = output / ('down-' + variant + '.air')
            library = output / ('down-' + variant + '.metallib')
            common = ['xcrun', '--sdk', 'iphoneos', 'metal', '-O3', '-mios-version-min=16.4',
                      '-fmodules-cache-path=' + str(output / 'modules'), str(HERE / 'shaders' / source)]
            commands.run(variant + '-compile', common + ['-c', '-o', air])
            commands.run(variant + '-AIR', common + ['-S', '-emit-llvm', '-o', output / ('down-' + variant + '.ll')])
            commands.run(variant + '-link', ['xcrun', '--sdk', 'iphoneos', 'metallib', air, '-o', library])
            record['products'][variant] = {'file': library.name, 'sha256': sha256(library),
                                           'source': source, 'source_sha256': sha256(HERE / 'shaders' / source)}
        record.update({'build_completed': True, 'exit_code': 0})
        print(json.dumps(record, indent=2))
        return 0
    except (RuntimeError, OSError, ValueError) as error:
        record.update({'error': str(error), 'exit_code': 1})
        print(str(error), file=sys.stderr)
        return 1
    finally:
        write_json(output / 'BUILD.json', record)


if __name__ == '__main__':
    sys.exit(main())
