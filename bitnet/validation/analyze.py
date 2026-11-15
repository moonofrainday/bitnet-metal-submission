#!/usr/bin/env python3
"""Independently recompute operator results from new native binary readbacks.

CPU only. Raw vectors/rows stay in --input. --output contains derived metrics,
coverage and a few independent FP16-unpack/math.fsum checks, never raw vectors.
Default exit: 0 all required checks pass; 1 observed failure; 2 missing data;
3 malformed input/configuration. Model/whole-layer correctness is not inferred.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

M = K = 2560
BLOCK_BYTES = 66
BLOCK_VALUES = 256
ROW_BYTES = 660
ATOL = 2e-4
RTOL = 2e-5
CASES = ('known-answer', 'dense-seed42', 'real-step-1', 'real-step-16', 'real-step-32')
ROUTES = ('S', 'PoC', 'Q4')
ORACLE_ATOL = 1e-11
ORACLE_RTOL = 1e-12


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def vector(path: Path, kind: str) -> tuple[bytes, tuple]:
    data = path.read_bytes()
    width = 4 if kind == 'f' else 8
    if len(data) != M * width:
        raise ValueError(f'Truncated/oversized {kind} vector: {path}: {len(data)} bytes')
    return data, struct.unpack('<' + str(M) + kind, data)


def metrics(actual: tuple, reference: tuple) -> dict:
    finite = [math.isfinite(a) and math.isfinite(r) for a, r in zip(actual, reference)]
    errors = [abs(a - r) if ok else None for a, r, ok in zip(actual, reference, finite)]
    tolerances = [ATOL + RTOL * abs(r) if math.isfinite(r) else None for r in reference]
    failed = [i for i in range(M) if not finite[i] or errors[i] > tolerances[i]]
    good = [i for i, ok in enumerate(finite) if ok]
    worst = max(good, key=lambda i: errors[i]) if good else None
    ratio_row = max(good, key=lambda i: errors[i] / tolerances[i]) if good else None
    finite_all = len(good) == M
    sq_error = math.fsum(errors[i] ** 2 for i in good)
    sq_reference = math.fsum(reference[i] ** 2 for i in good)
    zero_rows = sum(ok and r == 0 for ok, r in zip(finite, reference))
    nonzero = [i for i in good if reference[i] != 0]
    return {
        'rows': M, 'failing_rows': len(failed), 'first_failing_row': failed[0] if failed else None,
        'nonfinite_rows': M - len(good), 'max_abs_error': errors[worst] if worst is not None else None,
        'worst_row': worst, 'worst_reference': reference[worst] if worst is not None else None,
        'worst_actual': actual[worst] if worst is not None else None,
        'max_error_over_tolerance': errors[ratio_row] / tolerances[ratio_row] if ratio_row is not None else None,
        'worst_tolerance_ratio_row': ratio_row,
        'relative_l2': (math.sqrt(sq_error / sq_reference) if sq_reference else 0.0 if sq_error == 0 else None) if finite_all else None,
        'rmse': math.sqrt(sq_error / M) if finite_all else None,
        'nmse': (sq_error / sq_reference if sq_reference else 0.0 if sq_error == 0 else None) if finite_all else None,
        'max_relative_error_nonzero_reference': max((errors[i] / abs(reference[i]) for i in nonzero), default=None),
        'zero_reference_rows': zero_rows,
        'zero_reference_nonzero_error_rows': sum(finite[i] and reference[i] == 0 and errors[i] != 0 for i in range(M)),
        'pass': not failed,
    }


def validate_weights(weight: bytes) -> dict:
    if len(weight) != M * ROW_BYTES:
        raise ValueError(f'Weight vector has {len(weight)} bytes, expected {M * ROW_BYTES}')
    illegal = nonfinite = 0
    scale_hash = hashlib.sha256()
    for offset in range(0, len(weight), BLOCK_BYTES):
        for code in weight[offset:offset + 64]:
            illegal += sum(((code >> shift) & 3) == 3 for shift in (0, 2, 4, 6))
        scale = weight[offset + 64:offset + 66]
        scale_hash.update(scale)
        nonfinite += not math.isfinite(struct.unpack('<e', scale)[0])
    return {'illegal_code_slots': illegal, 'nonfinite_scales': nonfinite,
            'block_count': len(weight) // BLOCK_BYTES,
            'block_scale_sha256': scale_hash.hexdigest(), 'pass': not (illegal or nonfinite)}


def oracle_row(weight: bytes, x: tuple, row: int) -> float:
    # Index directly by logical K, independently of the native half/plane loop
    # and of both GPU kernels' packed-byte coefficient expressions.
    terms = []
    for k in range(K):
        within = k % BLOCK_VALUES
        block_start = row * ROW_BYTES + (k // BLOCK_VALUES) * BLOCK_BYTES
        byte_index = 32 * (within // 128) + (within % 32)
        plane = (within % 128) // 32
        code = (weight[block_start + byte_index] >> (plane * 2)) & 3
        scale = struct.unpack_from('<e', weight, block_start + 64)[0]
        terms.append((code - 1) * float(scale) * x[k])
    return math.fsum(terms)


def csv_fidelity(path: Path, reference: tuple, routes: dict, comparisons: dict) -> dict:
    rows = list(csv.DictReader(path.open(newline='')))
    errors = []
    if len(rows) != M:
        errors.append(f'CSV row count {len(rows)} != {M}')
    for index, row in enumerate(rows):
        if index >= M:
            break
        if int(row['row']) != index:
            errors.append(f'CSV row identity mismatch at {index}')
        values = {'reference_f64': reference[index], **{r + '_f32': routes[r][1][index] for r in ROUTES}}
        for field, expected in values.items():
            got = float(row[field])
            if math.isfinite(got) and math.isfinite(expected):
                fmt = '<d' if field == 'reference_f64' else '<f'
                if struct.pack(fmt, got) != struct.pack(fmt, expected):
                    errors.append(f'{field} differs from binary at row {index}')
            elif math.isfinite(got) != math.isfinite(expected):
                errors.append(f'{field} finite mask differs at row {index}')
        expected_tolerance = ATOL + RTOL * abs(reference[index]) if math.isfinite(reference[index]) else None
        if expected_tolerance is not None and abs(float(row['tolerance']) - expected_tolerance) > 1e-12:
            errors.append(f'CSV tolerance differs at row {index}')
    return {'pass': not errors, 'rows': len(rows), 'error_count': len(errors),
            'first_errors': errors[:12], 'scope': 'Binary/csv vector values, row identities and original tolerance'}


def native_metric_fidelity(native: dict, recomputed: dict) -> dict:
    checks = []
    aliases = {'max_abs': 'max_abs_error', 'relative_l2': 'relative_l2',
               'failing_rows': 'failing_rows', 'nonfinite_rows': 'nonfinite_rows', 'worst_row': 'worst_row'}
    for key, actual in recomputed.items():
        original = native.get(key)
        if not isinstance(original, dict):
            checks.append({'comparison': key, 'field': 'present', 'pass': False})
            continue
        for old_key, new_key in aliases.items():
            observed = original.get(old_key)
            expected = actual[new_key]
            if expected is None or observed is None:
                passed = expected is observed
            elif old_key in ('failing_rows', 'nonfinite_rows', 'worst_row'):
                passed = observed == expected
            else:
                passed = abs(float(observed) - expected) <= 1e-12 + 1e-10 * abs(expected)
            checks.append({'comparison': key, 'field': old_key, 'pass': passed})
    return {'pass': all(x['pass'] for x in checks), 'checks': checks,
            'scope': 'Reported scalar metrics versus independent recomputation; formatting allowance is not GPU tolerance'}


def run(args) -> dict:
    source = args.input.resolve(strict=True)
    native_path = source / 'summary.json'
    native = json.loads(native_path.read_text())
    required_cases = CASES[:2] if args.model_free else CASES
    native_cases = {case['id']: case for case in native.get('cases', [])}
    if len(native_cases) != len(native.get('cases', [])):
        raise ValueError('Native case IDs are duplicated')
    manifest_path = args.real_inputs
    if manifest_path is None and native.get('real_inputs_manifest'):
        manifest_path = Path(native['real_inputs_manifest'])
    manifest = json.loads(manifest_path.read_text()) if manifest_path else None
    identities = {case['id']: case for case in manifest.get('cases', [])} if manifest else {}
    missing, failed_checks, cases = [], [], []
    if native.get('mode') != 'gpu':
        missing.append('native_gpu_execution')
    if native.get('execution_pass') is not True:
        (failed_checks if native.get('execution_pass') is False else missing).append('native_execution')
    for name in ('known_answer_cpu_exact_pass', 'knownanswer_pass', 'fp64_pass', 'parity_pass', 'coverage_pass', 'overall_pass'):
        value = native.get(name)
        if value is False:
            failed_checks.append('native_declared_' + name)
        elif value is None:
            missing.append('native_declared_' + name)
    if native.get('abs_tolerance') != ATOL or native.get('rel_tolerance') != RTOL or native.get('required_row_failures') != 0:
        failed_checks.append('native_tolerance_contract')
    for case_id in required_cases:
        case = {'id': case_id, 'comparisons': {}, 'model_output_pass': None}
        folder = source / case_id
        expected_files = ['W.tq2_0', 'X.f32', 'reference.f64', 'S.f32', 'PoC.f32', 'Q4.f32', 'rows.csv']
        absent = [str(folder / name) for name in expected_files if not (folder / name).is_file()]
        if case_id not in native_cases:
            missing.append(case_id + ':native_case_record')
        if absent:
            missing.extend(absent)
            case.update({'missing': absent, 'fp64_pass': None, 'parity_pass': None, 'coverage_pass': None})
            cases.append(case)
            continue
        declared = native_cases.get(case_id, {})
        weight = (folder / 'W.tq2_0').read_bytes()
        weight_check = validate_weights(weight)
        _, x = vector(folder / 'X.f32', 'f')
        reference_bytes, reference = vector(folder / 'reference.f64', 'd')
        outputs = {route: vector(folder / (route + '.f32'), 'f') for route in ROUTES}
        hashes = {'weights_sha256': hashlib.sha256(weight).hexdigest(),
                  'activation_sha256': sha(folder / 'X.f32'),
                  'reference_f64_sha256': hashlib.sha256(reference_bytes).hexdigest(),
                  **{route + '_sha256': hashlib.sha256(outputs[route][0]).hexdigest() for route in ROUTES}}
        native_input_hash_pass = all(declared.get(k) == hashes[k] for k in ('weights_sha256', 'activation_sha256'))
        if case_id.startswith('real-step-'):
            expected = identities.get(case_id)
            if expected is None:
                # Native input hashes are a valid fallback only with exact named keys.
                candidate = declared.get('hashes', declared)
                expected = candidate if all(k in candidate for k in ('weights_sha256', 'activation_sha256')) else None
            if expected is None:
                missing.append(case_id + ':independent_prepared_input_identity')
                identity_pass = None
            else:
                identity_pass = all(hashes[k] == expected[k] for k in ('weights_sha256', 'activation_sha256'))
        else:
            identity_pass = True
        comparisons = {route + '_vs_fp64': metrics(outputs[route][1], reference) for route in ROUTES}
        comparisons.update({route + '_vs_S': metrics(outputs[route][1], outputs['S'][1]) for route in ('PoC', 'Q4')})
        bitwise = {route + '_vs_S': sum(outputs[route][0][i * 4:(i + 1) * 4] != outputs['S'][0][i * 4:(i + 1) * 4] for i in range(M)) for route in ('PoC', 'Q4')}
        fp64_pass = all(comparisons[route + '_vs_fp64']['pass'] for route in ROUTES)
        parity_pass = all(comparisons[route + '_vs_S']['pass'] for route in ('PoC', 'Q4'))
        csv_check = csv_fidelity(folder / 'rows.csv', reference, outputs, comparisons)
        native_check = native_metric_fidelity(declared.get('comparisons', {}), comparisons)
        oracle_indices = {0, 1, M - 1}
        for metric in comparisons.values():
            for key in ('worst_row', 'worst_tolerance_ratio_row'):
                if metric[key] is not None:
                    oracle_indices.add(metric[key])
        oracle = []
        if weight_check['pass'] and all(math.isfinite(v) for v in x):
            for row in sorted(oracle_indices):
                value = oracle_row(weight, x, row)
                diff = abs(value - reference[row]) if math.isfinite(reference[row]) else None
                tolerance = ORACLE_ATOL + ORACLE_RTOL * abs(value)
                oracle.append({'row': row, 'oracle': value, 'native_reference': reference[row] if math.isfinite(reference[row]) else None,
                               'abs_difference': diff, 'oracle_tolerance': tolerance,
                               'pass': diff is not None and diff <= tolerance})
        oracle_pass = bool(oracle) and all(row['pass'] for row in oracle)
        coverage = declared.get('coverage_pass')
        immutable = declared.get('input_immutable_pass')
        dispatches = declared.get('dispatches', [])
        route_map = {item.get('route'): item for item in dispatches}
        dispatch_pass = len(dispatches) == 3 and set(route_map) == set(ROUTES)
        for route in ROUTES:
            item = route_map.get(route, {})
            expected_groups = 1 if route == 'Q4' else 2
            dispatch_pass &= (item.get('command_status') == 4 and item.get('completed') is True
                              and item.get('canary_pass') is True and item.get('canary_corrupt_bytes') == 0
                              and item.get('coverage_pass') is True and item.get('finite_rows') == M
                              and item.get('input_immutable_pass') is True
                              and item.get('grid') == [M // (4 * expected_groups), 1, 1]
                              and item.get('threads') == [32, expected_groups, 1]
                              and item.get('buffer_offsets') == [0, 0, 128]
                              and item.get('timing_measured') is False)
        if len(route_map) == 3:
            dispatch_pass &= len({item.get('weight_handle') for item in dispatches}) == 1
            dispatch_pass &= len({item.get('activation_handle') for item in dispatches}) == 1
        known_pass = None
        if case_id == 'known-answer':
            expected = tuple(1.375 if row == 0 else -1.25 if row == M - 1 else 0.0 for row in range(M))
            known_pass = reference == expected and all(outputs[r][1] == expected for r in ROUTES)
        flags = {'packed_contract': weight_check['pass'], 'finite_X': all(math.isfinite(v) for v in x),
                 'input_identity': identity_pass, 'native_input_immutable': immutable,
                 'native_dumped_input_hashes': native_input_hash_pass,
                 'dispatch_completion_guards_and_geometry': dispatch_pass,
                 'fresh_GGML_dequantizer_crosscheck': declared.get('input_checks', {}).get('fresh_ggml_dequantizer_crosscheck_pass'),
                 'coverage': coverage, 'fp64': fp64_pass, 'parity': parity_pass,
                 'csv_fidelity': csv_check['pass'], 'native_metric_fidelity': native_check['pass'],
                 'independent_fp64_oracle': oracle_pass}
        for name, recomputed in (('fp64_pass', fp64_pass), ('parity_pass', parity_pass)):
            declared_value = declared.get(name)
            flags['declared_' + name + '_agrees'] = declared_value == recomputed if declared_value is not None else None
        if known_pass is not None:
            flags['known_answer_exact'] = known_pass
        for name, value in flags.items():
            if value is False:
                failed_checks.append(case_id + ':' + name)
            elif value is None:
                missing.append(case_id + ':' + name)
        case.update({'hashes': hashes, 'weight_contract': weight_check, 'comparisons': comparisons,
                     'bitwise_diff_rows': bitwise, 'checks': flags, 'fp64_pass': fp64_pass,
                     'parity_pass': parity_pass, 'coverage_pass': coverage,
                     'csv_fidelity': csv_check, 'native_metric_fidelity': native_check,
                     'oracle_rows': oracle, 'known_answer_exact': known_pass,
                     'dispatch_count': len(dispatches), 'dispatch_evidence_pass': dispatch_pass,
                     'native_declared_flags': {k: declared.get(k) for k in ('fp64_pass', 'parity_pass', 'coverage_pass')}})
        cases.append(case)
    def complete_flag(name):
        values = ([case.get(name) for case in cases] + [native.get(name)]
                  + [native_cases.get(case_id, {}).get(name) for case_id in required_cases])
        return False if False in values else None if None in values else True
    strict_exit = 1 if failed_checks else 2 if missing else 0
    return {
        'schema_version': 1, 'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'analysis_completed': True, 'CPU_only': True, 'GPU_executed_by_analyzer': False,
        'scope': ('standalone M2560_K2560_N1 operator; two model-free synthetic cases; three routes'
                  if args.model_free else 'standalone M2560_K2560_N1 operator; five fixed cases including three captured public-2B inputs; three routes'),
        'reference': 'fresh native independent FP64 binary; independently spot-checked via logical-K FP16 unpack + math.fsum',
        'tolerance': {'atol': ATOL, 'rtol': RTOL, 'zero_failed_rows_required': True},
        'oracle_reference_tolerance': {'atol': ORACLE_ATOL, 'rtol': ORACLE_RTOL, 'not_a_GPU_acceptance_rule': True},
        'model_free_requested_explicitly': args.model_free,
        'captured_real_inputs_in_scope': not args.model_free,
        'required_cases': list(required_cases), 'required_routes': list(ROUTES), 'cases': cases,
        'extra_native_cases_not_evaluated': sorted(set(native_cases) - set(required_cases)),
        'native_summary': str(native_path), 'native_summary_sha256': sha(native_path),
        'real_inputs_manifest': str(manifest_path) if manifest_path else None,
        'native_declared_flags': {k: native.get(k) for k in ('execution_pass', 'knownanswer_pass', 'fp64_pass', 'parity_pass', 'coverage_pass', 'overall_pass', 'exit_code')},
        'GPU_dispatch_records_checked': sum(case.get('dispatch_count', 0) for case in cases),
        'timing_or_speedup_measured': False,
        'parity_pass': complete_flag('parity_pass'), 'fp64_pass': complete_flag('fp64_pass'),
        'coverage_pass': complete_flag('coverage_pass'), 'model_output_pass': None,
        'whole_transformer_or_model_quality_measured': False,
        'missing': missing, 'failed_checks': failed_checks,
        'overall_pass': strict_exit == 0, 'strict_exit_code': strict_exit,
        'processing_success_does_not_override_strict_result': True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--real-inputs', type=Path, help='Optional prepared manifest; native summary otherwise supplies it')
    parser.add_argument('--model-free', action='store_true',
                        help='Explicitly request only the two synthetic cases; never implies three real-input cases passed')
    parser.add_argument('--exit-mode', choices=('strict', 'processing'), default='strict')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        print('ANALYSIS_BLOCKED: Refuse existing output', file=sys.stderr)
        return 3
    try:
        result = run(args)
        result['requested_exit_mode'] = args.exit_mode
        result['actual_cli_exit_code'] = result['strict_exit_code'] if args.exit_mode == 'strict' else 0
        output.mkdir(parents=True)
        (output / 'SUMMARY.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        with (output / 'RESULTS.csv').open('w', newline='') as stream:
            fields = ['case', 'comparison', 'rows', 'max_abs_error', 'relative_l2', 'rmse', 'nmse', 'failing_rows',
                      'nonfinite_rows', 'max_error_over_tolerance', 'worst_row', 'bitwise_diff_rows', 'pass']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for case in result['cases']:
                for comparison, metric in case['comparisons'].items():
                    row = {key: metric.get(key) for key in fields if key not in ('case', 'comparison', 'bitwise_diff_rows')}
                    row.update(case=case['id'], comparison=comparison,
                               bitwise_diff_rows=case.get('bitwise_diff_rows', {}).get(comparison))
                    writer.writerow(row)
        print(json.dumps({key: result[key] for key in ('analysis_completed', 'parity_pass', 'fp64_pass',
                                                      'coverage_pass', 'model_output_pass', 'overall_pass',
                                                      'strict_exit_code', 'actual_cli_exit_code', 'missing', 'failed_checks')}))
        return result['actual_cli_exit_code']
    except FileNotFoundError as error:
        print(json.dumps({'analysis_completed': False, 'missing': str(error), 'strict_exit_code': 2}), file=sys.stderr)
        return 2
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        print(json.dumps({'analysis_completed': False, 'malformed': f'{type(error).__name__}: {error}', 'strict_exit_code': 3}), file=sys.stderr)
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
