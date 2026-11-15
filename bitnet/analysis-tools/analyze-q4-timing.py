#!/usr/bin/env python3
"""Independently recompute phone paired timings; never execute GPU/device work."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

MODEL_SHA = '52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2'
THERMAL = {0: 'nominal', 1: 'fair', 2: 'serious', 3: 'critical'}


class EvidenceError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise EvidenceError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, evidence):
    require(path.is_file(), 'Missing required JSON: ' + str(path))
    evidence[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    try:
        return json.loads(path.read_text())
    except ValueError as error:
        raise EvidenceError('Invalid JSON: ' + str(path)) from error


def read_csv(path, evidence):
    require(path.is_file(), 'Missing timing CSV: ' + str(path))
    evidence[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    with path.open(newline='') as handle:
        values = list(csv.DictReader(handle))
    require(bool(values), 'Timing CSV has zero samples')
    return values


def number(value, field, positive=False):
    result = float(value[field])
    require(math.isfinite(result) and (not positive or result > 0), 'Invalid timing field: ' + field)
    return result


def quantile(values, probability):
    require(bool(values), 'Cannot compute a quantile with no values')
    ordered = sorted(values)
    k = (len(ordered) - 1) * probability
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def stats(values):
    return {'n': len(values), 'p50': quantile(values, .5), 'p95': quantile(values, .95),
            'min': min(values), 'max': max(values)}


def paired(rows):
    a, b = [r['a'] for r in rows], [r['b'] for r in rows]
    delta = [y - x for x, y in zip(a, b)]
    return {'pairs': len(rows), 'A': stats(a), 'B': stats(b), 'paired_B_minus_A': stats(delta),
            'paired_relative_percent': stats([100 * d / x for x, d in zip(a, delta)]),
            'B_p50_relative_change_percent': 100 * (quantile(b, .5) / quantile(a, .5) - 1),
            'B_wins': sum(d < 0 for d in delta), 'A_wins': sum(d > 0 for d in delta),
            'ties': sum(d == 0 for d in delta)}


def grouped_stats(rows):
    result = paired(rows)
    result['execution_order'] = {order: paired([r for r in rows if r['order'] == order])
                                 for order in ('AB', 'BA') if any(r['order'] == order for r in rows)}
    n = len(rows)
    result['chronological_quartiles'] = {
        str(q + 1): paired(rows[q * n // 4:(q + 1) * n // 4]) for q in range(4)
        if rows[q * n // 4:(q + 1) * n // 4]}
    return result


def expected_order(pair, round_index):
    ab = pair % 4 in (0, 3)
    if round_index % 2 == 0:
        ab = not ab
    return 'AB' if ab else 'BA'


def resolve_input(path, mode):
    path = path.expanduser().resolve()
    if mode == 'operator':
        for candidate in (path, path / 'operator', path.parent):
            if (candidate / 'performance/samples.csv').is_file():
                return candidate
    else:
        for candidate in (path, path / 'device'):
            if (candidate / 'samples.csv').is_file() and (candidate / 'execution.json').is_file():
                return candidate
    raise EvidenceError('Cannot locate the requested native timing layout under ' + str(path))


def find_identity(path, explicit, evidence):
    if explicit:
        identity_path = explicit.expanduser().resolve()
    else:
        candidates = [parent / 'BUILD_IDENTITY.json' for parent in (path, path.parent, path.parent.parent)]
        identity_path = next((p for p in candidates if p.is_file()), None)
    require(identity_path is not None, 'Supply --identity-json or retain host BUILD_IDENTITY.json')
    identity = load(identity_path, evidence)
    require(identity.get('unsigned') is False and bool(identity.get('framework_sha256'))
            and bool(identity.get('app_files_sha256')), 'Missing selected signed build/framework identity')
    return identity


def verify_host_run(path, evidence):
    candidates = (path, path.parent, path.parent.parent)
    host = next((p for p in candidates if (p / 'logs/launch.log').is_file()
                 and (p / 'summary.json').is_file()), None)
    require(host is not None, 'Retain the host run summary and original launch log')
    summary = load(host / 'summary.json', evidence)
    require(summary.get('execution_pass') is True and summary.get('errors') == [], 'Host functional timing run did not pass')
    status, log = host / 'logs/launch.exit.txt', host / 'logs/launch.log'
    require(status.is_file() and status.read_text().strip() == '0', 'devicectl launch did not exit 0')
    require('The app terminated with the exit code 0.' in log.read_text(errors='replace'),
            'Actual application exit 0 is absent from the original launch log')
    for item in (status, log, host / 'logs/device-details.log', host / 'model-identity.json', host / 'config.json'):
        if item.is_file():
            evidence[str(item)] = {'sha256': sha(item), 'bytes': item.stat().st_size}
    result = load(host / 'device/result.json', evidence)
    require(result.get('device') == 'Apple A16 GPU' and result.get('model_sha256') == MODEL_SHA
            and result.get('process_exit_code') == 0, 'App model/device/completion identity differs')
    return {'functional_exit_verified': True, 'original_app_exit_verified': True,
            'physical_model_details_in_local_evidence': (host / 'logs/device-details.log').is_file(),
            'model_sha256': result['model_sha256'], 'device': result['device'], 'iOS': result.get('iOS')}


def endpoint_label(value):
    if isinstance(value, int):
        return THERMAL.get(value, 'unavailable(%s)' % value)
    return value if value in THERMAL.values() else 'unavailable'


def check_three_rounds(groups, comparisons):
    require(set(groups) == {(r, c) for r in (1, 2, 3) for c in comparisons},
            'Expected complete three-round contrast coverage')


def analyze_operator(root, evidence):
    summary = load(root / 'performance/summary.json', evidence)
    qualification = load(root / 'qualification.json', evidence)
    require(qualification.get('local_fp64_pass') is True and qualification.get('device') == 'Apple A16 GPU',
            'Native Q operator qualification/device failed')
    require(qualification.get('tolerance') == {'abs': 2e-4, 'relative': 2e-5}, 'Operator tolerance changed')
    cases = qualification.get('cases', [])
    require(len(cases) == 9 and {(v.get('step'), v.get('route')) for v in cases}
            == {(s, r) for s in (1, 16, 32) for r in ('S', 'C8', 'Q4')}, 'Incomplete operator qualification cases')
    require(all(v.get('rows') == 2560 and v.get('failed_rows') == 0 and v.get('nonfinite_rows') == 0
                and v.get('GPU_completed') is True for v in cases), 'Native qualification contains an observed failure')
    require(summary.get('inputs') == qualification.get('inputs') and summary.get('libraries') == qualification.get('libraries'),
            'Timed inputs/libraries differ from their qualification')
    require(summary.get('capture') is False and summary.get('diagnostic_readback') is False
            and summary.get('single_dispatch_per_command_buffer') is True,
            'Operator metric includes diagnostics or a different dispatch interval')
    values = read_csv(root / 'performance/samples.csv', evidence)
    groups, seen, sequence = defaultdict(list), set(), []
    expected_routes = {'AA': ('S', 'S'), 'SQ': ('S', 'Q4'), 'CQ': ('C8', 'Q4')}
    for source_index, value in enumerate(values):
        rr, contrast, index = int(value['round']), value['comparison'], int(value['pair'])
        require(rr in (1, 2, 3) and contrast in expected_routes and 0 <= index < 201, 'Invalid operator pair identity')
        require((rr, contrast, index) not in seen, 'Duplicate operator pair')
        seen.add((rr, contrast, index))
        require((value['first_route'], value['second_route']) == expected_routes[contrast], 'Operator route labels changed')
        require(value['order'] == expected_order(index, rr), 'Operator AB/BA schedule changed')
        a, b = number(value, 'first_us', True), number(value, 'second_us', True)
        require(abs(number(value, 'second_minus_first_us') - (b - a)) <= 1e-9, 'Operator recorded delta is inconsistent')
        groups[rr, contrast].append({'a': a, 'b': b, 'order': value['order'], 'pair': index, 'source_row': source_index})
        if not sequence or sequence[-1] != (rr, contrast):
            sequence.append((rr, contrast))
    check_three_rounds(groups, expected_routes)
    expected_sequence = [(rr, c) for rr in (1, 2, 3) for c in (list(expected_routes)[rr - 1:] + list(expected_routes)[:rr - 1])]
    require(sequence == expected_sequence, 'Operator contrast block order is not the recorded rotated protocol')
    native_groups = {(int(v['round']), v['comparison']): v for v in summary.get('groups', [])}
    require(len(summary.get('groups', [])) == 9 and set(native_groups) == set(groups), 'Native operator group summary is incomplete')
    result = []
    for (rr, contrast), rows in sorted(groups.items()):
        require(len(rows) == 201 and [r['pair'] for r in rows] == list(range(201)), 'Missing/out-of-order operator pairs')
        item = grouped_stats(rows)
        native = native_groups[rr, contrast]
        numeric_checks = {'first_p50_us': item['A']['p50'], 'first_p95_us': item['A']['p95'],
                          'second_p50_us': item['B']['p50'], 'second_p95_us': item['B']['p95'],
                          'paired_second_minus_first_p50_us': item['paired_B_minus_A']['p50']}
        require(all(abs(float(native[k]) - value) <= 1e-9 for k, value in numeric_checks.items()),
                'Native/recomputed operator quantiles differ')
        require(native.get('pairs') == len(rows) and native.get('second_wins') == item['B_wins'], 'Native/recomputed pair/win counts differ')
        before, after = native.get('before', {}), native.get('after', {})
        item.update({'round': rr, 'comparison': contrast, 'A_route': expected_routes[contrast][0],
                     'B_route': expected_routes[contrast][1], 'unit': 'us',
                     'thermal_sampling_resolution': 'Contrast-block endpoints; before includes warmup. Per-pair and continuous thermal state unavailable.',
                     'before': before, 'after': after,
                     'thermal_endpoint_cohort': endpoint_label(before.get('thermal')) + '->' + endpoint_label(after.get('thermal')),
                     'nominal_endpoints_imply_cold_device': False})
        result.append(item)
    require(len(values) == 1809, 'Operator actual pair count differs from 3 x 3 x 201')
    return {'mode': 'operator', 'metric': 'Single-dispatch GPU command-buffer elapsed (GPUEndTime-GPUStartTime) after completion',
            'unit': 'us', 'scope': 'One Qcur-0 operator on identical step-16 W/X; not whole-model decode',
            'actual_pairs': len(values), 'actual_measured_command_buffers': 2 * len(values),
            'warmups_per_logical_path_per_contrast': 30, 'raw_rows_excluded': 0,
            'groups': result, 'group_sequence': sequence, 'input_hashes': summary['inputs'],
            'library_hashes': summary['libraries'], 'native_summary_recomputed': True,
            'numerical_recomputed_by_this_timing_analyzer': False, 'prior_native_local_qualification_pass': True}


def decode_tokens(root, groups, evidence):
    rows = read_csv(root / 'token-times.csv', evidence)
    sample_lookup = {(rr, c, v['pair']): v for (rr, c), values in groups.items() for v in values}
    tokens, seen = defaultdict(dict), set()
    for value in rows:
        rr, c, index = int(value['round']), value['comparison'], int(value['pair'])
        path, step = value['path'], int(value['step'])
        require((rr, c, index) in sample_lookup and path in ('A', 'B') and 1 <= step <= 32, 'Unexpected per-token identity')
        key = (rr, c, index, path, step)
        require(key not in seen, 'Duplicate per-token interval')
        seen.add(key)
        sample = sample_lookup[rr, c, index]
        require(value['route'] == sample[path + '_route'] and value['order'] == sample['order'], 'Per-token route/order differs from pair')
        require(int(value['block_id']) == sample['block_id'] and int(value['campaign_order']) == sample['campaign_order'],
                'Per-token block identity differs from pair')
        elapsed, start, end = number(value, 'wall_ms', True), number(value, 'start_s'), number(value, 'end_s')
        require(end > start and abs((end - start) * 1000 - elapsed) <= .001, 'Invalid per-token wall-clock interval')
        require(start >= sample[path + '_start'] - 1e-8 and end <= sample[path + '_end'] + 1e-8,
                'Token interval falls outside its 32-token wall interval')
        tokens[rr, c, index, path][step] = {'elapsed': elapsed, 'start': start, 'end': end}
    require(len(rows) == 64 * len(sample_lookup), 'Per-token CSV does not cover all actual paths/steps')
    gaps = []
    for (rr, c, index), sample in sample_lookup.items():
        for path in ('A', 'B'):
            part = tokens[rr, c, index, path]
            require(set(part) == set(range(1, 33)), 'Missing per-token step coverage')
            require(all(part[s]['end'] <= part[s + 1]['start'] + 1e-8 for s in range(1, 32)), 'Per-token intervals overlap')
            total = math.fsum(part[s]['elapsed'] for s in range(1, 33))
            gap = sample['a' if path == 'A' else 'b'] - total
            require(gap >= -.001, 'Token intervals exceed the complete path wall interval')
            gaps.append({'round': rr, 'comparison': c, 'pair': index, 'path': path,
                         'wall_minus_token_intervals_ms': gap})
    step_summaries = []
    for (rr, c), values in sorted(groups.items()):
        for step in range(1, 33):
            part = [{'a': tokens[rr, c, v['pair'], 'A'][step]['elapsed'],
                     'b': tokens[rr, c, v['pair'], 'B'][step]['elapsed'], 'order': v['order']} for v in values]
            step_summaries.append({'round': rr, 'comparison': c, 'step': step, 'unit': 'ms', **grouped_stats(part)})
    return {'actual_token_rows': len(rows), 'step_summaries': step_summaries,
            'wall_minus_token_intervals': gaps,
            'note': 'Per-token intervals include normal decode CPU encoding and GPU completion; the wall-minus-interval gap is not a pure CPU encoding cost.'}


def analyze_decode(root, evidence):
    execution = load(root / 'execution.json', evidence)
    result = load(root / 'result.json', evidence)
    config = load(root / 'config.json', evidence)
    request = load(root / 'request.json', evidence)
    require(execution.get('execution_success') is True and execution.get('timing_clean') is True
            and execution.get('one_reused_model_and_context') is True and execution.get('reset_prefill_excluded') is True,
            'Decode execution/synchronization protocol failed')
    for field in ('callbacks', 'trace_during_timing', 'CPU_timers_during_timing', 'tensor_readback_during_timing'):
        require(execution.get(field) is False, 'Decode diagnostics are enabled/unverified: ' + field)
    require(result.get('process_exit_code') == 0 and result.get('q_readback_during_timing') is False,
            'Decode completion failed or Q readback is not explicitly disabled')
    require(result.get('device') == 'Apple A16 GPU' and result.get('model_sha256') == MODEL_SHA,
            'Decode model/device identity mismatch')
    require(config.get('layers') == '0' and config.get('n_ctx') == 512 and config.get('n_batch') == 128
            and config.get('n_ubatch') == 128 and config.get('threads') == 6
            and config.get('kv_k') == 'f16' and config.get('kv_v') == 'f16' and config.get('flash_attention') is False,
            'Decode effective configuration differs from the frozen layer-0 workload')
    require(len(config.get('prompt_ids', [])) == 27 and len(config.get('continuation_ids', [])) == 32,
            'Decode prompt/continuation ID coverage changed')
    require(result.get('routes') == ['S', 'Q4'] and execution.get('rounds') == 3
            and execution.get('tokens_per_path') == 32 and execution.get('prefill_tokens') == 27,
            'Decode routes/repetition/token protocol changed')
    expected = {'AA': ('S', 'S'), 'SQ4': ('S', 'Q4')}
    pairs = execution.get('pairs_per_comparison_per_round')
    require(isinstance(pairs, int) and pairs > 0, 'Missing actual decode pair count')
    values = read_csv(root / 'samples.csv', evidence)
    groups, seen, sequence = defaultdict(list), set(), []
    prior_end = None
    for source_index, value in enumerate(values):
        rr, contrast, index = int(value['round']), value['comparison'], int(value['pair'])
        require(rr in (1, 2, 3) and contrast in expected and 0 <= index < pairs, 'Invalid whole-decode pair identity')
        require((rr, contrast, index) not in seen, 'Duplicate whole-decode pair')
        seen.add((rr, contrast, index))
        require((value['a_route'], value['b_route']) == expected[contrast] and value['order'] == expected_order(index, rr),
                'Whole-decode route or AB/BA order changed')
        a, b = number(value, 'a_wall_ms', True), number(value, 'b_wall_ms', True)
        require(abs(number(value, 'b_minus_a_ms') - (b - a)) <= 1e-8, 'Whole-decode paired delta mismatch')
        ast, aend, bst, bend = (number(value, key) for key in ('a_start_s', 'a_end_s', 'b_start_s', 'b_end_s'))
        require(aend > ast and bend > bst and abs((aend - ast) * 1000 - a) <= .001
                and abs((bend - bst) * 1000 - b) <= .001, 'Whole-decode wall timestamps disagree with elapsed times')
        require((aend <= bst if value['order'] == 'AB' else bend <= ast), 'Recorded execution order disagrees with timestamps')
        require(prior_end is None or min(ast, bst) >= prior_end - 1e-8, 'Whole-decode pair timeline overlaps or is reordered')
        prior_end = max(aend, bend)
        tag = (rr, contrast)
        if not sequence or sequence[-1] != tag:
            sequence.append(tag)
        block_id, position = int(value['block_id']), int(value['campaign_order'])
        require(block_id == len(sequence) - 1 and position == 1 + ((len(sequence) - 1) % 2), 'Invalid contrast block/position')
        temperatures = {key: value[key] for key in ('a_thermal_before', 'a_thermal_after', 'b_thermal_before', 'b_thermal_after')}
        require(all(v in THERMAL.values() or v == 'unknown' for v in temperatures.values()), 'Unknown thermal field encoding')
        low = (value['a_low_power'], value['b_low_power'])
        require(all(v in ('0', '1') for v in low), 'Invalid low-power state')
        label = 'A:%s->%s|B:%s->%s|low_power:%s,%s' % (*temperatures.values(), *low)
        groups[tag].append({'a': a, 'b': b, 'pair': index, 'order': value['order'], 'source_row': source_index,
                            'block_id': block_id, 'campaign_order': position,
                            'A_route': value['a_route'], 'B_route': value['b_route'],
                            'A_start': ast, 'A_end': aend, 'B_start': bst, 'B_end': bend,
                            'endpoint_cohort': label})
    check_three_rounds(groups, expected)
    require(execution.get('schedule') == ['%d:%d:%s' % (rr, 1 + index % 2, contrast)
                                        for index, (rr, contrast) in enumerate(sequence)], 'Native/CSV contrast schedules differ')
    expected_sequence = [(1, 'AA'), (1, 'SQ4'), (2, 'SQ4'), (2, 'AA'), (3, 'AA'), (3, 'SQ4')]
    require(sequence == expected_sequence, 'Decode contrast rotation differs from the fixed AA/SQ4 protocol')
    answer = []
    for (rr, contrast), rows in sorted(groups.items()):
        require(len(rows) == pairs and [r['pair'] for r in rows] == list(range(pairs)), 'Incomplete whole-decode pair group')
        cohorts = defaultdict(list)
        for row in rows:
            cohorts[row['endpoint_cohort']].append(row)
        item = {'round': rr, 'comparison': contrast, 'A_route': expected[contrast][0],
                'B_route': expected[contrast][1], 'unit': 'ms', **grouped_stats(rows),
                'thermal_endpoint_cohorts': {key: grouped_stats(part) for key, part in cohorts.items()},
                'thermal_sampling_resolution': 'Per-path entry (before reset/prefix restoration) and after timed decode; no continuous or per-token thermal samples.',
                'nominal_endpoints_imply_cold_device': False}
        answer.append(item)
    require(result.get('sample_pairs') == len(values) == 3 * 2 * pairs, 'Native/actual whole-decode pair counts disagree')
    token_results = decode_tokens(root, groups, evidence)
    require(result.get('token_rows') == token_results['actual_token_rows'], 'Native/actual token count mismatch')
    return {'mode': 'decode', 'metric': 'Complete synchronized fixed 32-token decode host wall time; KV reset/prefix restoration excluded',
            'scope': 'Layer-0 Q4 versus full stock with normal Attention/FFN/down/residual execution',
            'unit': 'ms', 'actual_pairs': len(values), 'actual_token_rows': token_results['actual_token_rows'],
            'pairs_per_contrast_per_round': pairs, 'groups': answer, 'group_sequence': sequence,
            'round_power_snapshots': result.get('round_states'), 'raw_rows_excluded': 0,
            'input_identity': {'model_sha256': result['model_sha256'], 'prompt_ids_sha256': hashlib.sha256(json.dumps(config['prompt_ids']).encode()).hexdigest(),
                               'continuation_ids_sha256': hashlib.sha256(json.dumps(config['continuation_ids']).encode()).hexdigest(),
                               'library_hashes': result.get('library_sha256'), 'requested_libraries': request.get('libraries')},
            'token_results': token_results, 'whole_model_FP64_or_quality_acceptance_inferred': False}


def contrast_directions(result):
    groups = result['groups']
    calibrations = {v['round']: v for v in groups if v['comparison'] == 'AA'}
    output = {}
    for contrast in sorted({v['comparison'] for v in groups} - {'AA'}):
        items = [v for v in groups if v['comparison'] == contrast]
        per_round = []
        for item in items:
            aa = calibrations[item['round']]
            per_round.append({'round': item['round'], 'candidate_paired_median': item['paired_B_minus_A']['p50'],
                              'AA_paired_median': aa['paired_B_minus_A']['p50'],
                              'candidate_order_medians': {k: v['paired_B_minus_A']['p50'] for k, v in item['execution_order'].items()},
                              'AA_order_medians': {k: v['paired_B_minus_A']['p50'] for k, v in aa['execution_order'].items()},
                              'candidate_chronological_quartile_medians': [v['paired_B_minus_A']['p50'] for v in item['chronological_quartiles'].values()]})
        output[contrast] = {'three_rounds_available': len(items) == 3,
                            'B_faster_in_each_recorded_round_median': all(v['paired_B_minus_A']['p50'] < 0 for v in items),
                            'B_faster_in_each_recorded_order_group_median': all(x['paired_B_minus_A']['p50'] < 0 for v in items for x in v['execution_order'].values()),
                            'B_faster_in_each_recorded_chronological_quartile_median': all(x['paired_B_minus_A']['p50'] < 0 for v in items for x in v['chronological_quartiles'].values()),
                            'per_round_candidate_and_AA': per_round,
                            'automatic_statistical_significance_or_general_speedup_claim': False,
                            'note': 'A/A is reported as calibration, never subtracted. Thermal drift and serial correlation prevent an unconditional significance claim from these direction checks.'}
    return output


def flat_group(item):
    return {'round': item['round'], 'comparison': item['comparison'], 'A_route': item['A_route'], 'B_route': item['B_route'],
            'unit': item['unit'], 'pairs': item['pairs'], 'A_p50': item['A']['p50'], 'A_p95': item['A']['p95'],
            'B_p50': item['B']['p50'], 'B_p95': item['B']['p95'], 'paired_B_minus_A_p50': item['paired_B_minus_A']['p50'],
            'paired_B_minus_A_p95': item['paired_B_minus_A']['p95'],
            'paired_relative_percent_p50': item['paired_relative_percent']['p50'],
            'B_p50_relative_change_percent': item['B_p50_relative_change_percent'],
            'B_wins': item['B_wins'], 'A_wins': item['A_wins'], 'ties': item['ties'],
            'AB_paired_median': item['execution_order'].get('AB', {}).get('paired_B_minus_A', {}).get('p50'),
            'BA_paired_median': item['execution_order'].get('BA', {}).get('paired_B_minus_A', {}).get('p50')}


def write_csv(path, rows):
    require(bool(rows), 'No summary CSV rows to write')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('operator', 'decode'), required=True)
    parser.add_argument('--input', type=Path, required=True, help='Host run, fetched operator, or fetched device directory')
    parser.add_argument('--output', type=Path, required=True, help='Fresh local derived analysis directory')
    parser.add_argument('--identity-json', type=Path, help='Selected host BUILD_IDENTITY.json, otherwise discover near input')
    parser.add_argument('--accepted-q4', type=Path, help='Optional same-build complete Q capture SUMMARY.json')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error('Refusing existing output: ' + str(output))
    output.mkdir(parents=True)
    evidence = {}
    result = {'analysis_completed': False, 'CPU_only': True, 'GPU_executed_by_analyzer': False,
              'mode': args.mode, 'analysis_exit_code': 2, 'numerical_acceptance_evaluated_by_this_tool': False,
              'model_quality_tested': False, 'pooled_across_rounds_or_thermal_cohorts': False,
              'quantile_definition': 'Sorted linear interpolation at (n-1)*p; same p50/p95 definition as the native operator harness.',
              'negative_paired_B_minus_A_means_B_faster': True, 'errors': [],
              'analyzer_sha256': sha(Path(__file__))}
    try:
        root = resolve_input(args.input, args.mode)
        identity = find_identity(root, args.identity_json, evidence)
        result['selected_build_identity'] = identity
        result['host_run_checks'] = verify_host_run(root, evidence)
        if args.accepted_q4:
            acceptance = load(args.accepted_q4.expanduser().resolve(), evidence)
            require(acceptance.get('strict_exit_code') == 0 and acceptance.get('overall_pass') is True,
                    'The supplied Q acceptance did not pass its strict declared scope')
            require(acceptance.get('selected_build_identity') == identity, 'Timing/acceptance selected app/framework identities differ')
            result['prior_same_build_Q_capture_acceptance_pass'] = True
        else:
            result['prior_same_build_Q_capture_acceptance_pass'] = None
        details = analyze_operator(root, evidence) if args.mode == 'operator' else analyze_decode(root, evidence)
        result.update(details)
        result['contrast_direction_checks'] = contrast_directions(details)
        write_csv(output / 'ROUND_SUMMARY.csv', [flat_group(v) for v in details['groups']])
        if args.mode == 'decode':
            token_results = result.pop('token_results')
            token_rows = []
            for item in token_results['step_summaries']:
                item['A_route'] = 'S'
                item['B_route'] = 'S' if item['comparison'] == 'AA' else 'Q4'
                token_rows.append({'step': item['step'], **flat_group(item)})
            write_csv(output / 'TOKEN_STEP_SUMMARY.csv', token_rows)
            write_csv(output / 'WALL_MINUS_TOKEN_INTERVALS.csv', token_results['wall_minus_token_intervals'])
            result['per_token_note'] = token_results['note']
        result['analysis_completed'] = True
        result['analysis_exit_code'] = 0
        result['timing_evidence_pass'] = True
    except (EvidenceError, OSError, ValueError, KeyError, TypeError) as error:
        result['errors'].append(str(error))
        result['timing_evidence_pass'] = False
        print(str(error), file=sys.stderr)
    result['evidence_manifest'] = evidence
    (output / 'SUMMARY.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (output / 'PROVENANCE.json').write_text(json.dumps({'input_files': evidence, 'analyzer_sha256': result['analyzer_sha256']}, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('analysis_completed', 'analysis_exit_code', 'actual_pairs', 'actual_token_rows', 'timing_evidence_pass')}))
    return result['analysis_exit_code']


if __name__ == '__main__':
    sys.exit(main())
