#!/usr/bin/env python3
"""Independently audit the selected A16 down outputs against logical-K math.fsum."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

ROWS = [0,336,352,360,448,496,880,1144,1504,1824,2152,2504]
ROUTES = ['S','S-wrap','correction-end']
KNOWN_FAILURES = [5,338,356,362,455,502,882,1148,1508,1830,2155,2508]
W_SHA = 'c032780f9245f4ecfe37fe117f0b6f16b5a76cfb72734c60b1f7e4beabe99888'
X_SHA = 'c9e096fc21f1e6ab4dcc1c4329925942219f848b4ab082dc30b930fe2bf43e7a'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text())


def reference(weights, x):
    require(len(weights) == 1782, 'Wrong single-row weight size')
    terms = []
    for block in range(27):
        offset = block * 66
        scale = struct.unpack_from('<e', weights, offset + 64)[0]
        require(math.isfinite(scale), 'Nonfinite FP16 scale')
        for half in range(2):
            for plane in range(4):
                for j in range(32):
                    code = (weights[offset + half*32+j] >> (2*plane)) & 3
                    require(code != 3, 'Invalid ternary code 3')
                    k = block*256 + half*128 + plane*32+j
                    terms.append((code - 1)*scale*x[k])
    require(len(terms) == 6912, 'Logical K coverage')
    ordered = 0.0
    for term in terms:
        ordered += term
    return math.fsum(terms), ordered


def metrics(rows):
    errors = [row['abs_error'] for row in rows]
    ref2 = math.fsum(row['reference']**2 for row in rows)
    err2 = math.fsum(error**2 for error in errors)
    result = {'rows': len(rows), 'failed_rows': sum(row['failed'] for row in rows),
            'failed_indices': [row['row'] for row in rows if row['failed']],
            'nonfinite_rows': sum(row['nonfinite'] for row in rows),
            'max_abs': max(errors), 'max_error_over_tolerance': max(row['error_over_tolerance'] for row in rows),
            'relative_L2': math.sqrt(err2/ref2) if ref2 else (0.0 if not err2 else None),
            'max_relative_error_nonzero_reference': max((row['relative_error'] for row in rows
                                                        if row['relative_error'] is not None), default=None),
            'zero_reference_rows': sum(row['reference'] == 0 for row in rows),
            'stock_bit_different_rows': sum(row['stock_bit_difference'] for row in rows),
            'stock_tolerance_failed_rows': sum(row['stock_tolerance_failed'] for row in rows),
            'stock_max_abs': max(row['stock_abs_error'] for row in rows),
            'worst_ratio_row': max(rows, key=lambda row: row['error_over_tolerance'])['row']}
    # A measured NaN/Inf is a numerical failure, not a JSON-processing error.
    unavailable = [key for key, value in result.items()
                   if isinstance(value, float) and not math.isfinite(value)]
    for key in unavailable:
        result[key] = None
    result['unavailable_aggregate_fields'] = unavailable
    result['aggregate_status'] = ('nonfinite_measured_output; unavailable aggregates are null'
                                  if result['nonfinite_rows'] else
                                  'nonfinite_derived_aggregate; unavailable aggregates are null'
                                  if unavailable else 'finite')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Fresh phone runner result directory')
    parser.add_argument('--old-stock', type=Path, help='Historical full 2560-output stock capture with identical input')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--strict', action='store_true', help='Return selected numerical strict status instead of processing status')
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing an existing analysis directory')
    source = args.input.resolve()
    operator = source/'operator'
    host = load(source/'summary.json')
    native = load(operator/'summary.json')
    require(host.get('execution_pass') is True and not host.get('errors'), 'Host execution was incomplete')
    require(native.get('execution_pass') is True and native.get('analysis_completed') is True, 'Native execution incomplete')
    require(native.get('device') == 'Apple A16 GPU', 'Wrong physical GPU')
    require(native.get('GPU_command_buffers_completed') == 36, 'Wrong GPU command count')
    weights = (source/'staged-fixture/w.tq2').read_bytes()
    input_bytes = (operator/'x.f32').read_bytes()
    require(len(weights) == 4561920 and sha(weights) == W_SHA, 'GGUF-bound down weight differs')
    require(len(input_bytes) == 27648 and sha(input_bytes) == X_SHA, 'Actual norm input differs')
    require(input_bytes == (source/'staged-fixture/x.f32').read_bytes(), 'Staged/native X differs')
    x = struct.unpack('<6912f', input_bytes)
    require(all(map(math.isfinite, x)), 'Nonfinite input')
    records = {route: [] for route in ROUTES}
    references = []
    byte_outputs = {}
    for start in ROWS:
        folder = operator/('group-'+str(start))
        subset = (folder/'w.tq2').read_bytes()
        require(subset == weights[start*1782:(start+8)*1782], 'Sliced W differs from actual full weight')
        native_reference_bytes = (folder/'reference.f64').read_bytes()
        require(len(native_reference_bytes) == 64, 'Wrong native FP64 extent')
        native_reference = struct.unpack('<8d', native_reference_bytes)
        outputs = {}
        for route in ROUTES:
            data = (folder/(route+'.f32')).read_bytes()
            require(len(data) == 32, 'Wrong output extent')
            outputs[route] = struct.unpack('<8f', data)
            byte_outputs[(start, route)] = data
        for local in range(8):
            row = start+local
            oracle, ordered = reference(subset[local*1782:(local+1)*1782], x)
            native_ref = native_reference[local]
            require(math.isfinite(oracle) and math.isfinite(native_ref), 'Nonfinite CPU reference')
            oracle_error = abs(native_ref-oracle)
            require(oracle_error <= 1e-10 + 1e-12*abs(oracle), 'Native reference differs materially from fsum oracle')
            references.append({'row': row, 'math_fsum': oracle, 'ordered_logical_K': ordered,
                               'native_reference': native_ref, 'native_minus_fsum': native_ref-oracle,
                               'native_ordered_bit_difference': struct.pack('<d', native_ref) != struct.pack('<d', ordered)})
            for route in ROUTES:
                actual = outputs[route][local]
                stock = outputs['S'][local]
                finite = math.isfinite(actual)
                error = abs(actual-oracle) if finite else math.inf
                tolerance = 2e-4 + 2e-5*abs(oracle)
                stock_error = abs(actual-stock) if finite and math.isfinite(stock) else math.inf
                records[route].append({'group': start, 'route': route, 'row': row,
                    'reference': oracle, 'native_reference': native_ref, 'actual': actual,
                    'abs_error': error, 'tolerance': tolerance, 'error_over_tolerance': error/tolerance,
                    'relative_error': error/abs(oracle) if oracle else None,
                    'failed': int(not finite or error > tolerance), 'nonfinite': int(not finite),
                    'stock_actual': stock, 'stock_abs_error': stock_error,
                    'stock_bit_difference': int(struct.pack('<f',actual) != struct.pack('<f',stock)),
                    'stock_tolerance_failed': int(not finite or stock_error > 2e-4+2e-5*abs(stock))})
    stats = {route: metrics(records[route]) for route in ROUTES}
    require(all(len(records[route]) == 96 for route in ROUTES), 'Wrong selected row count')
    stock_failure_set_reproduced = stats['S']['failed_indices'] == KNOWN_FAILURES
    for route in ROUTES:
        by_group = {start: [row for row in records[route] if row['group'] == start] for start in ROWS}
        for start, rows in by_group.items():
            case = [entry for entry in native['cases'] if entry['group'] == start and entry['route'] == route]
            require(len(case) == 1, 'Missing/duplicate native case')
            require(case[0]['rows'] == 8 and case[0]['GPU_completed'] is True and case[0]['GPU_status'] == 4,
                    'Native GPU completion missing')
            require(case[0]['output_canaries_pass'] is True and case[0]['inputs_unchanged'] is True,
                    'Native canary/input preservation failed')
            require(case[0]['failed_rows'] == sum(row['failed'] for row in rows), 'Native/independent failed count differs')
            require(case[0]['stock_bit_different_rows'] == sum(row['stock_bit_difference'] for row in rows),
                    'Native/independent parity count differs')
    completed = []
    for line in (source/'logs/launch.log').read_text(errors='replace').splitlines():
        if 'BITNET_DOWN_CONTROL_COMPLETE ' in line:
            fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
            require(fields.get('status') == '4' and fields.get('completed') == '1', 'Actual GPU completion failed')
            completed.append((fields['route'], int(fields['group'])))
    require(len(completed) == 36 and set(completed) == {(route,start) for route in ROUTES for start in ROWS},
            'Actual logged GPU completion coverage differs')
    old_comparison = None
    if args.old_stock:
        old_bytes = args.old_stock.read_bytes()
        require(len(old_bytes) == 10240, 'Historical stock extent')
        different = []
        for start in ROWS:
            for local in range(8):
                if byte_outputs[(start,'S')][4*local:4*local+4] != old_bytes[4*(start+local):4*(start+local+1)]:
                    different.append(start+local)
        old_comparison = {'source_sha256': sha(old_bytes), 'selected_rows': 96,
                          'different_rows': len(different), 'different_indices': different}
    compared = {}
    for route in ['S-wrap','correction-end']:
        pairs = list(zip(records['S'], records[route]))
        compared[route] = {'closer_to_fsum_rows': sum(b['abs_error'] < a['abs_error'] for a,b in pairs),
                          'farther_from_fsum_rows': sum(b['abs_error'] > a['abs_error'] for a,b in pairs),
                          'equal_error_rows': sum(b['abs_error'] == a['abs_error'] for a,b in pairs)}
    fp64 = all(stat['failed_rows'] == 0 and stat['nonfinite_rows'] == 0 for stat in stats.values())
    parity = stats['S-wrap']['stock_bit_different_rows'] == 0
    strict = 0 if fp64 and parity else 1
    require(native['fp64_pass'] == fp64 and native['parity_pass'] == parity, 'Native independent flags differ')
    require(host['fp64_pass'] == fp64 and host['parity_pass'] == parity, 'Host independent flags differ')
    summary = {'analysis_completed': True, 'CPU_only': True, 'GPU_executed_by_analyzer': False,
        'execution_pass': True, 'parity_pass': parity, 'fp64_pass': fp64, 'model_output_pass': None,
        'overall_pass': fp64 and parity, 'strict_exit_code': strict, 'processing_exit_code': 0,
        'scope': '96 selected real down outputs/path; actual captured stock norm input; not full-matrix/model acceptance',
        'tolerance': {'abs': 2e-4, 'relative': 2e-5},
        'reference': 'Independent logical-K TQ2_0 decode with math.fsum; each F32 input and FP16 block scale retained',
        'W_sha256': sha(weights), 'X_sha256': sha(input_bytes), 'routes': stats,
        'native_ordered_reference_bit_differences': sum(row['native_ordered_bit_difference'] for row in references),
        'native_reference_max_abs_vs_fsum': max(abs(row['native_minus_fsum']) for row in references),
        'GPU_completions_independently_checked': len(completed), 'native_summary_sha256': sha((operator/'summary.json').read_bytes()),
        'historical_stock_comparison': old_comparison, 'candidate_pointwise_error_changes': compared,
        'known_stock_failure_set_reproduced': stock_failure_set_reproduced,
        'analyzer_sha256': sha(Path(__file__).read_bytes()), 'performance_measured': False,
        'full_matrix_expanded': False}
    args.output.mkdir(parents=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    for name, rows in [('rows.csv',[row for route in ROUTES for row in records[route]]),('reference-audit.csv',references)]:
        with (args.output/name).open('w',newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    with (args.output/'route-summary.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['route','rows','failed_rows','nonfinite_rows','max_abs',
                                                  'max_error_over_tolerance','relative_L2','stock_bit_different_rows'])
        writer.writeheader()
        for route,stat in stats.items():
            writer.writerow({'route':route,**{key:stat[key] for key in writer.fieldnames if key!='route'}})
    print(json.dumps({'analysis_completed': True, 'processing_exit_code': 0, 'strict_exit_code': strict,
                      'parity_pass': parity, 'fp64_pass': fp64, 'routes': stats},allow_nan=False))
    return strict if args.strict else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError,struct.error) as error:
        print('Analysis error: '+str(error),file=sys.stderr)
        raise SystemExit(2)
