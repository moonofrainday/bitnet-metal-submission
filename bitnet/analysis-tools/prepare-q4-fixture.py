#!/usr/bin/env python3
"""Prepare a local Q operator fixture from an accepted real-phone Q capture."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

STEP16_X = '8314cd7fe2f5deaa64ef8a7c9cbb1d9254bc430eed863ea4792c46a3c99c887e'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read_json(path):
    require(path.is_file(), 'Missing JSON: ' + str(path))
    return json.loads(path.read_text())


def load_analyzer():
    source = Path(__file__).with_name('analyze-q4-complete.py')
    spec = importlib.util.spec_from_file_location('bitnet_complete_q_capture', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Accepted probe-on host or fetched device directory')
    parser.add_argument('--acceptance', type=Path, required=True, help='Its passing complete Q capture SUMMARY.json')
    parser.add_argument('--build-dir', type=Path, required=True, help='Same selected signed build directory')
    parser.add_argument('--output', type=Path, required=True, help='Fresh local directory outside the source checkout')
    parser.add_argument('--expect-step16-x', default=STEP16_X, help='Audited primary real-input X hash; do not silently change datasets')
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[2]
    output = args.output.expanduser().resolve()
    require(not output.exists(), 'Refuse an existing fixture output')
    require(output != source_root and source_root not in output.parents, 'Raw fixture output must remain outside the source checkout')
    acceptance_path = args.acceptance.expanduser().resolve()
    acceptance = read_json(acceptance_path)
    require(acceptance.get('analysis_completed') is True and acceptance.get('strict_exit_code') == 0
            and acceptance.get('overall_pass') is True, 'Complete Q capture acceptance has not passed')
    for flag in ('execution_pass', 'same_input_pass', 'parity_pass', 'fp64_pass', 'model_output_pass', 'probe_logit_nonperturbation_pass'):
        require(acceptance.get(flag) is True, 'Required acceptance flag did not pass: ' + flag)
    selected_build = read_json(args.build_dir.expanduser().resolve() / 'BUILD.json')
    build_sha = sha((args.build_dir.expanduser().resolve() / 'BUILD.json').read_bytes())
    identity = acceptance.get('selected_build_identity', {})
    require(identity.get('build_manifest_sha256') == build_sha and identity.get('unsigned') is False
            and selected_build.get('build_completed') is True and selected_build.get('exit_code') == 0
            and selected_build.get('unsigned') is False, 'Fixture build differs from the accepted signed app')
    m = load_analyzer()
    root = m.data_root(args.input)
    files, hashes, manifest_sha = m.read_capture(root, 'S', m.WEIGHT_SHA)
    require(acceptance.get('capture_manifest_sha256', {}).get('S') == manifest_sha,
            'Current capture manifest is not bound by the accepted analysis')
    result, request, config = m.identity(root, m.MODEL_SHA)
    require(result.get('q_readback_enabled') is True, 'Fixture source did not collect actual readbacks')
    require(hashes['step-16/x.f32'] == args.expect_step16_x, 'Step-16 X differs from the audited timing input')
    library_root = Path(selected_build['shaders'])
    libraries = {}
    for route, filename in (('C8', 'q-c8x2.metallib'), ('Q4', 'q-stock4x1.metallib')):
        path = library_root / filename
        require(path.is_file(), 'Missing freshly built shader: ' + str(path))
        libraries[route] = {'file': filename, 'sha256': sha(path.read_bytes())}
    require(libraries['Q4']['sha256'] == request['libraries']['Q4']['sha256'],
            'Q4 library differs from the accepted real-graph dispatch')
    import numpy as np
    decoded = m.decode_weights(np, files['weights/w.tq2'])
    generated = {}
    for step in (1, 16, 32):
        activation, actual = files['step-%d/x.f32' % step], files['step-%d/y.f32' % step]
        reference = m.cpu_reference(np, decoded, activation)
        metric = m.metrics(np, reference, np.frombuffer(actual, dtype='<f4'))
        require(metric['pass'], 'Actual stock output failed independent FP64 while preparing fixture')
        generated[step] = {'w.tq2': files['weights/w.tq2'], 'x.f32': activation,
                           'y.f32': actual, 'reference.f64': reference.astype('<f8').tobytes()}
    output.mkdir(parents=True)
    manifest = {'phase': 'CPU preparation from accepted real iPhone Q readbacks; no new GPU execution',
                'GPU_executed_by_preparer': False, 'actual_phone_capture_source': True,
                'model_sha256': m.MODEL_SHA, 'node': 'Qcur-0', 'weight': 'blk.0.attn_q.weight',
                'W_shape': [2560, 2560], 'X_shape': [2560, 1], 'activation': 'F32', 'output': 'F32',
                'weight_type': 'TQ2_0', 'prompt_ids': config['prompt_ids'],
                'continuation_ids': config['continuation_ids'], 'fixtures': {}, 'libraries': libraries,
                'acceptance_sha256': sha(acceptance_path.read_bytes()), 'capture_manifest_sha256': manifest_sha,
                'selected_build_manifest_sha256': build_sha, 'source_weight_sha256': m.WEIGHT_SHA,
                'timing_step16_activation_sha256': hashes['step-16/x.f32'],
                'tolerance': {'abs': m.ATOL, 'relative': m.RTOL},
                'whole_model_quality_acceptance_inferred': False}
    for step, payloads in generated.items():
        folder = output / ('step-%d' % step)
        folder.mkdir()
        manifest['fixtures'][str(step)] = {}
        for name, data in payloads.items():
            (folder / name).write_bytes(data)
            manifest['fixtures'][str(step)][name] = {'sha256': sha(data), 'bytes': len(data),
                                                    'source_role': 'actual GPU W/X/Y' if name != 'reference.f64' else 'independent CPU FP64'}
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'fixture_prepared': True, 'GPU_executed_by_preparer': False,
                      'output': str(output), 'weight_sha256': m.WEIGHT_SHA,
                      'step16_X_sha256': hashes['step-16/x.f32']}))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError, KeyError, ImportError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
