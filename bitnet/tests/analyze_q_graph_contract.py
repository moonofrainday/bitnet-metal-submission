#!/usr/bin/env python3
"""CPU-only audit of actual GGML Q guard/dispatch and saved FP64 comparisons."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bitnet.q_kernel_contract import load_contract


class ObservedFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ObservedFailure(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_values(path, count, code):
    data = path.read_bytes()
    width = struct.calcsize('<' + code)
    require(len(data) == count * width, 'Saved vector extent differs')
    return struct.unpack('<' + str(count) + code, data)


def blocks(text):
    result, name, lines = {}, None, []
    for line in text.splitlines():
        if line.startswith('Q_GRAPH_CASE_BEGIN '):
            require(name is None, 'Nested BEGIN marker')
            value = json.loads(line.split(' ', 1)[1]); name = value['name']
            require(name not in result, 'Duplicate case BEGIN')
            lines = []
        elif line.startswith('Q_GRAPH_CASE_END '):
            value = json.loads(line.split(' ', 1)[1])
            require(name == value['name'], 'END marker differs')
            result[name] = '\n'.join(lines); name = None
        elif name is not None:
            lines.append(line)
    require(name is None, 'Incomplete final case')
    return result


def audit(run):
    data = json.loads((run / 'RUN.json').read_text())
    if data.get('native_exit_code') == 2:
        raise ValueError('Native collection has a setup or evidence error')
    require(data['native_exit_code'] == 0 and data['native_numerical_pass'] is True, 'Native numerical collection failed')
    contract = load_contract()
    q4, c8 = (contract['variants'][x] for x in ('stock4x1', 'c8x2'))
    expected = {
        'stock': (None, None), 'Q4-valid': (q4, None),
        'Q4-wrong-4x2': (None, 'kernel_contract'), 'Q4-wrong-4x4': (None, 'kernel_contract'),
        'legacy-wrong-4x2': (None, 'kernel_contract'), 'legacy-wrong-4x4': (None, 'kernel_contract'),
        'C8-with-Q4-geometry': (None, 'kernel_contract'), 'Q4-descriptor-C8-library': (None, 'pipeline'),
        'C8-valid': (c8, None), 'C8-descriptor-Q4-library': (None, 'pipeline'),
        'same-path-Q4-initial': (q4, None), 'same-path-replace-C8': (None, 'pipeline'),
        'same-path-restore-Q4': (q4, None), 'same-path-missing-after-valid': (None, 'pipeline'),
        'same-path-restore-after-missing': (q4, None),
    }
    trace_path = run / 'backend-trace.log'
    trace = blocks(trace_path.read_text())
    require([x['name'] for x in data['cases']] == list(expected) == list(trace), 'Case coverage or ordering differs')
    require(data['graph_and_buffers_reused'] is True and data['same_calling_thread'] is True, 'Not the requested cache context')
    require(sha(run / 'reference.f64') == data['reference_sha256'], 'Reference bytes changed')
    reference = load_values(run / 'reference.f64', q4['nrows'], 'd')
    require(all(map(math.isfinite, reference)), 'Reference has nonfinite values')
    stock = (run / 'stock/y.f32').read_bytes()
    rows = []
    for item in data['cases']:
        name = item['name']; descriptor, fallback = expected[name]; text = trace[name]
        require(item['synchronized'] is True and item['graph_status'] == 0, name + ': compute did not complete')
        path = run / name / 'y.f32'
        require(sha(path) == item['output_sha256'], name + ': output hash differs')
        values = load_values(path, q4['nrows'], 'f')
        failures = sum(not math.isfinite(x) or abs(x - r) > 2e-4 + 2e-5 * abs(r) for x, r in zip(values, reference))
        require(failures == 0 and item['FP64_failures'] == 0, name + ': FP64 bound failed')
        for side in ('before', 'after'):
            require((run / name / ('canary-' + side + '.bin')).read_bytes() == bytes([0xa5]) * 512, name + ': canary changed')
        require(item['W_after_sha256'] == data['W_sha256'] and item['X_after_sha256'] == data['X_sha256'], name + ': W/X changed')
        if name != 'C8-valid':
            require(path.read_bytes() == stock and item['stock_bitwise_differences'] == 0, name + ': stock parity failed')
        encoded = re.findall(r'^BITNET_MULTILAYER_ENCODE route=(q-custom|q-stock) .*$', text, re.M)
        completed = re.findall(r'^BITNET_MULTILAYER_COMPLETE route=(q-custom|q-stock) .*completed=1$', text, re.M)
        require(len(encoded) == len(completed) == 1, name + ': dispatch/completion coverage differs')
        wanted_route = 'q-custom' if descriptor else 'q-stock'
        require(encoded == completed == [wanted_route], name + ': wrong actual dispatch route')
        enc_line = next(x for x in text.splitlines() if x.startswith('BITNET_MULTILAYER_ENCODE '))
        comp_line = next(x for x in text.splitlines() if x.startswith('BITNET_MULTILAYER_COMPLETE '))
        require(re.search(r'command=(\S+)', enc_line).group(1) == re.search(r'command=(\S+)', comp_line).group(1), name + ': completion command differs')
        entry = descriptor['entry'] if descriptor else 'kernel_mul_mv_tq2_0_f32'
        require('pipeline=' + entry + ' ' in enc_line and 'node=Qcur-0 ' in enc_line and 'n=1 ' in enc_line, name + ': pipeline/node differs')
        if descriptor:
            grid = ','.join(map(str, descriptor['grid'])); threads = ','.join(map(str, descriptor['threads']))
            require('grid=[' + grid + ']' in enc_line and 'threads=[' + threads + ']' in enc_line, name + ': launch differs')
            require('BITNET_Q_GEOMETRY ' in text and 'checked=1' in text, name + ': no successful contract evidence')
            require('BITNET_Q_FALLBACK ' not in text, name + ': unexpectedly fell back')
        elif fallback:
            require('BITNET_Q_FALLBACK node=Qcur-0 layer=0 reason=' + fallback in text, name + ': expected rejection not seen')
        rows.append({'case': name, 'actual_route': wanted_route, 'pipeline': entry, 'completed_dispatches': 1,
                     'FP64_failures': 0, 'stock_bitwise_required': name != 'C8-valid', 'canaries_and_inputs_pass': True})
    return {'CPU_only_analysis': True, 'GPU_executed_by_analyzer': False, 'analysis_completed': True,
            'scope': data['scope'], 'performance_measured': False, 'model_output_pass': None,
            'full_model_or_quality_pass': None, 'case_count': len(rows), 'dispatch_pass': True,
            'fp64_pass': True, 'stock_parity_for_required_cases_pass': True,
            'all_cases_pass': True, 'strict_exit_code': 0, 'cases': rows,
            'run_sha256': sha(run / 'RUN.json'), 'trace_sha256': sha(trace_path),
            'reference_scope': 'Saved output of unchanged independent validation/reference.cpp, recomputed per-value bound here; not a second FP64 implementation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        print('Refuse existing analyzer output', file=sys.stderr); return 2
    try:
        result = audit(args.run.resolve()); code = 0
    except ObservedFailure as error:
        result = {'analysis_completed': True, 'all_cases_pass': False, 'strict_exit_code': 1, 'error': str(error)}; code = 1
    except (OSError, KeyError, ValueError, TypeError, AttributeError) as error:
        result = {'analysis_completed': False, 'all_cases_pass': None, 'strict_exit_code': 2, 'error': str(error)}; code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result, indent=2)); return code


if __name__ == '__main__':
    raise SystemExit(main())
