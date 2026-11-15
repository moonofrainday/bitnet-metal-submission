#!/usr/bin/env python3
"""CPU-only analyzer/packing negative controls; never compiles or launches Metal."""
import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
PACKAGE = Path(__file__).resolve().parent


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native', type=Path, help='Optional complete fresh GPU output for executable negative tests')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        print('Refuse existing output', file=sys.stderr)
        return 3
    out.mkdir(parents=True)
    a = module('clean_analysis', PACKAGE / 'analyze.py')
    p = module('clean_preparation', PACKAGE / 'prepare-real-inputs.py')
    tests = []
    def check(name, value, detail=None):
        tests.append({'name': name, 'pass': bool(value), 'detail': detail})
    zero = tuple(0.0 for _ in range(a.M))
    perturbed = list(zero)
    perturbed[0] = 1.0
    bad = a.metrics(tuple(perturbed), zero)
    check('all_zero_reference_exact_pass', a.metrics(zero, zero)['pass'])
    check('pointwise_numeric_failure_detected', not bad['pass'] and bad['failing_rows'] == 1)
    perturbed[0] = float('nan')
    nonfinite = a.metrics(tuple(perturbed), zero)
    check('nonfinite_fails_not_silently_ignored', not nonfinite['pass'] and nonfinite['nonfinite_rows'] == 1)
    block = b'\x55' * 64 + struct.pack('<e', 1.0)
    weight = bytearray(block * (a.M * 10))
    x = list(zero)
    for k, value in [(3, 1.25), (35, 4.0), (67, 2.0), (99, -.5), (385, 3.0)]:
        x[k] = value
    weight[3] = 0x24
    weight[64:66] = struct.pack('<e', .5)
    weight[66 + 33] = 0x56
    weight[66 + 64:66 + 66] = struct.pack('<e', .25)
    weight[660 + 3] = 0x56
    weight[660 + 64:660 + 66] = struct.pack('<e', 0.0)
    weight[(a.M - 1) * 660 + 3] = 0x54
    observed = [a.oracle_row(weight, tuple(x), row) for row in (0, 1, a.M - 1)]
    check('independent_hand_known_answer', observed == [1.375, 0.0, -1.25], observed)
    valid = a.validate_weights(weight)
    check('packed_known_answer_contract_pass', valid['pass'])
    illegal = bytearray(weight)
    illegal[0] = (illegal[0] & ~3) | 3
    check('reserved_trit_code_rejected', not a.validate_weights(illegal)['pass'])
    try:
        p.packed_validation(illegal)
        check('preparation_reserved_code_rejected', False)
    except ValueError:
        check('preparation_reserved_code_rejected', True)
    illegal = bytearray(weight)
    illegal[64:66] = struct.pack('<H', 0x7c00)
    check('nonfinite_scale_rejected', not a.validate_weights(illegal)['pass'])
    try:
        p.activation_validation(struct.pack('<2560f', float('inf'), *([0.0] * 2559)))
        check('preparation_nonfinite_activation_rejected', False)
    except ValueError:
        check('preparation_nonfinite_activation_rejected', True)
    if args.native:
        native = args.native.resolve(strict=True)
        def invoke(name, source, mode='strict', model_free=False):
            destination = out / (name + '-analysis')
            command = [sys.executable, str(PACKAGE / 'analyze.py'), '--input', str(source),
                       '--output', str(destination), '--exit-mode', mode]
            if model_free:
                command.append('--model-free')
            result = subprocess.run(command, capture_output=True, text=True)
            (out / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
            (out / (name + '.stdout.log')).write_text(result.stdout)
            (out / (name + '.stderr.log')).write_text(result.stderr)
            summary = json.loads((destination / 'SUMMARY.json').read_text()) if (destination / 'SUMMARY.json').exists() else None
            return result.returncode, summary
        code, summary = invoke('baseline', native)
        check('executable_positive_control_exit0', code == 0 and summary['overall_pass'] and summary['strict_exit_code'] == 0,
              {'actual_exit': code})
        numeric = out / 'numeric-mutated-copy'
        shutil.copytree(native, numeric, symlinks=False)
        q4 = numeric / 'known-answer/Q4.f32'
        data = bytearray(q4.read_bytes())
        struct.pack_into('<f', data, 0, struct.unpack_from('<f', data, 0)[0] + 1.0)
        q4.write_bytes(data)
        code, summary = invoke('numeric-strict', numeric)
        check('executable_numeric_failure_exit1', code == 1 and summary['fp64_pass'] is False and not summary['overall_pass'],
              {'actual_exit': code, 'strict_exit': summary['strict_exit_code'] if summary else None})
        code, summary = invoke('numeric-processing', numeric, 'processing')
        check('processing0_never_relabels_numeric_acceptance', code == 0 and summary['strict_exit_code'] == 1 and summary['fp64_pass'] is False and not summary['overall_pass'],
              {'actual_exit': code, 'strict_exit': summary['strict_exit_code'] if summary else None})
        missing = out / 'missing-data-copy'
        shutil.copytree(native, missing, symlinks=False)
        (missing / 'known-answer/Q4.f32').unlink()
        code, summary = invoke('missing-strict', missing)
        check('executable_missing_without_observed_failure_exit2', code == 2 and summary['fp64_pass'] is None and not summary['overall_pass'],
              {'actual_exit': code, 'strict_exit': summary['strict_exit_code'] if summary else None})
        nonfinite_copy = out / 'nonfinite-mutated-copy'
        shutil.copytree(native, nonfinite_copy, symlinks=False)
        target = nonfinite_copy / 'known-answer/Q4.f32'
        payload = bytearray(target.read_bytes())
        struct.pack_into('<f', payload, 0, float('nan'))
        target.write_bytes(payload)
        code, summary = invoke('nonfinite-strict', nonfinite_copy)
        case = next(c for c in summary['cases'] if c['id'] == 'known-answer')
        check('executable_nonfinite_exit1', code == 1 and case['comparisons']['Q4_vs_fp64']['nonfinite_rows'] == 1 and summary['fp64_pass'] is False,
              {'actual_exit': code})
        summary_copy = out / 'false-summary-copy'
        shutil.copytree(native, summary_copy, symlinks=False)
        target = summary_copy / 'summary.json'
        altered = json.loads(target.read_text())
        altered['fp64_pass'] = False
        target.write_text(json.dumps(altered, indent=2) + '\n')
        code, summary = invoke('false-summary-strict', summary_copy)
        check('native_declared_false_never_hidden_by_recomputed_rows', code == 1 and summary['fp64_pass'] is False and not summary['overall_pass'],
              {'actual_exit': code})
        subset = out / 'two-case-scope-copy'
        subset.mkdir()
        for case_id in ('known-answer', 'dense-seed42'):
            shutil.copytree(native / case_id, subset / case_id, symlinks=False)
        altered = json.loads((native / 'summary.json').read_text())
        altered['cases'] = [c for c in altered['cases'] if c['id'] in ('known-answer', 'dense-seed42')]
        altered['requested_cases'] = altered['processed_cases'] = 2
        altered['real_inputs_manifest'] = None
        (subset / 'summary.json').write_text(json.dumps(altered, indent=2) + '\n')
        code, summary = invoke('two-cases-default-strict', subset)
        check('missing_real_cases_never_auto_reduce_scope', code == 2 and len(summary['required_cases']) == 5 and not summary['overall_pass'],
              {'actual_exit': code})
        code, summary = invoke('two-cases-explicit-model-free', subset, model_free=True)
        check('explicit_model_free_two_case_scope_exit0', code == 0 and len(summary['required_cases']) == 2 and summary['captured_real_inputs_in_scope'] is False,
              {'actual_exit': code, 'scope_note': 'CPU-derived subset control; not a new native/GPU run'})
    report = {'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'CPU_only': True, 'GPU_or_build_executed': False,
              'original_native_data_modified': False, 'tests': tests,
              'pass': all(item['pass'] for item in tests),
              'evaluator_cli_controls_on_copied_native_outputs': bool(args.native),
              'native_harness_executed_by_controls': False}
    (out / 'CHECKS.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'pass': report['pass'], 'checks': len(tests), 'failed': [t['name'] for t in tests if not t['pass']]}))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
