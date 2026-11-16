#!/usr/bin/env python3
"""Build the actual backend cache test; --run explicitly enables Metal setup.

No model is loaded. This test checks loader/cache behavior, not matvec numeric
accuracy or performance. Metal device initialization may run capability probes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bitnet.q_kernel_contract import variant_contract

EXPECTED_CASES = {
    'Q4_4x2_rejected', 'Q4_4x4_rejected', 'legacy_4x2_rejected', 'legacy_4x4_rejected',
    'Q4_unique_entry_load', 'Q4_unchanged_file_reuses_pipeline',
    'canonical_path_alias_reuses_pipeline', 'different_file_has_distinct_cache_identity',
    'C8_library_cannot_satisfy_cached_Q4', 'C8_unique_entry_load',
    'Q4_library_cannot_satisfy_cached_C8', 'missing_path_cannot_reuse_Q4',
    'deleted_cached_file_rejected', 'restored_new_file_identity_gets_new_pipeline',
    'same_path_replaced_with_C8_rejects_Q4', 'same_path_restored_Q4_loads_new_identity',
    'forged_entry_rejected_by_actual_getter', 'forged_groups_rejected_by_actual_getter',
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_verdict(runtime, code):
    require(runtime['exit_code'] == code and runtime['target_Q_dispatches'] == 0 and
            runtime['model_loaded'] is False and runtime['fp64_pass'] is None and
            runtime['model_output_pass'] is None and runtime['performance_measured'] is False,
            'Native result exit/scope differs')
    names = [x['name'] for x in runtime['cases']]
    require(len(names) == len(set(names)) and set(names) <= EXPECTED_CASES and
            all(type(x['pass']) is bool for x in runtime['cases']), 'Native case identities differ')
    if code == 0:
        require(runtime['runtime_contract_pass'] is True and set(names) == EXPECTED_CASES and
                all(x['pass'] for x in runtime['cases']), 'Runtime result incomplete')
        return True
    if code == 1:
        require(runtime['runtime_contract_pass'] is False and any(not x['pass'] for x in runtime['cases']),
                'Native failure has no observed failed required check')
        return False
    require(code == 2 and runtime['runtime_contract_pass'] is None, 'Blocked result has an inconsistent verdict')
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--shaders', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--developer-dir', type=Path, default=Path('/Applications/Xcode.app/Contents/Developer'))
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    build, shaders, output = (x.resolve() for x in (args.build_dir,args.shaders,args.output))
    require(not output.exists(), 'Refuse existing test output')
    require(ROOT != output and ROOT not in output.parents, 'Write artifacts outside source checkout')
    output.mkdir(parents=True)
    result = {'build_completed': False, 'runtime_requested': args.run,
              'runtime_contract_pass': None, 'fp64_pass': None, 'model_output_pass': None,
              'performance_measured': False, 'commands': []}
    try:
        identity = json.loads((build/'FRESH_CORE_BUILD.json').read_text())
        require(identity['build_completed'] and identity['old_dylib_reused'] is False and
                Path(identity['source_root']).resolve() == ROOT, 'Require matching fresh core build')
        for name,digest in identity['source_files_sha256'].items():
            require(sha(ROOT/name) == digest, 'Core source changed after build: '+name)
        for name,digest in identity['products_sha256'].items():
            require(sha(build/name) == digest, 'Fresh library changed: '+name)
        manifest = json.loads((shaders/'BUILD.json').read_text())
        require(manifest['build_completed'] and manifest['sdk']=='macosx', 'Require completed macOS shaders')
        libraries = {}
        for variant,product in [('stock4x1','stock4x1.metallib'),('c8x2','qcur0-c8x2.metallib')]:
            contract = variant_contract(variant)
            entry = manifest['products'][product]
            require(entry['q_contract'] == contract and entry['expected_symbols'] == [contract['entry']],
                    'Shader contract differs from current source')
            library = shaders/product
            require(sha(library) == entry['sha256'], 'Shader product hash differs')
            libraries[variant] = library
        env = os.environ.copy()
        env['DEVELOPER_DIR'] = str(args.developer_dir)
        env['DYLD_LIBRARY_PATH'] = str(build/'bin')
        source = ROOT/'bitnet/tests/q_pipeline_contract.cpp'
        binary = output/'q-pipeline-contract'
        command = ['xcrun','--sdk','macosx','clang++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
                   '-Wno-deprecated-declarations','-DBITNET_Q_CONTRACT_TEST_MAIN',
                   '-I',str(ROOT/'ggml/include'),'-I',str(ROOT/'ggml/src/ggml-metal'),
                   '-I',str(ROOT/'vendor'),str(source),'-L',str(build/'bin'),
                   '-Wl,-rpath,'+str(build/'bin'),'-lggml-metal','-lggml','-lggml-base',
                   '-framework','Foundation','-framework','Metal','-o',str(binary)]
        def execute(name, argv, allowed=(0,)):
            with (output/(name+'.log')).open('w') as log:
                run = subprocess.run(argv,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            result['commands'].append({'name':name,'argv':argv,'exit_code':run.returncode})
            require(run.returncode in allowed, name+' failed; see '+str(output/(name+'.log')))
            return run.returncode
        execute('build', command)
        result.update(build_completed=True, source_sha256=sha(source), executable_sha256=sha(binary),
                      core_build_manifest_sha256=sha(build/'FRESH_CORE_BUILD.json'),
                      shader_build_manifest_sha256=sha(shaders/'BUILD.json'))
        if args.run:
            native_code = execute('runtime',[str(binary),str(libraries['stock4x1']),str(libraries['c8x2']),str(output/'runtime')], (0,1,2))
            result['native_exit_code'] = native_code
            runtime = json.loads((output/'runtime/RESULT.json').read_text())
            result.update(runtime=runtime, native_runtime_contract_pass=runtime['runtime_contract_pass'])
            result['runtime_contract_pass'] = runtime_verdict(runtime,native_code)
            require(native_code == 0, 'Native runtime checks did not complete successfully')
        result['exit_code'] = 0
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        result.update(error=str(error), exit_code=result.get('native_exit_code') or 1)
    (output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('commands','runtime')},indent=2))
    return result['exit_code']


if __name__=='__main__':
    raise SystemExit(main())
