#!/usr/bin/env python3
"""Audit captured layer-0 Q outputs on the CPU; never launch Metal or a phone."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

ROWS = COLS = 2560
BLOCK = 256
BLOCK_BYTES = 66
WEIGHT_BYTES = ROWS * (COLS // BLOCK) * BLOCK_BYTES
MODEL_SHA = '52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2'
WEIGHT_SHA = 'b2dea7f42cacfd08ed780df979f6f28240026795b1f5c87ca8c16fc7da23dc58'
ATOL, RTOL = 2e-4, 2e-5
ROUTES = ('S', 'Q4')
STEPS = range(1, 33)
LAYER_STEPS = (1, 16, 32)


class EvidenceError(RuntimeError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_json(path):
    if not path.is_file():
        raise EvidenceError('Missing required JSON: ' + str(path))
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise EvidenceError('Invalid JSON ' + str(path) + ': ' + str(error)) from error


def data_root(path):
    path = path.expanduser().resolve()
    return path / 'device' if (path / 'device/route-S').is_dir() else path


def checked_bytes(path, size, expected_sha=None):
    if not path.is_file():
        raise EvidenceError('Missing required payload: ' + str(path))
    data = path.read_bytes()
    if len(data) != size:
        raise EvidenceError('Wrong payload extent: ' + str(path))
    sha = digest(data)
    if expected_sha is not None and sha != expected_sha:
        raise EvidenceError('SHA-256 mismatch: ' + str(path))
    return data, sha


def required_capture_files():
    expected = {'weights/w.tq2': WEIGHT_BYTES}
    for step in STEPS:
        for role in ('x', 'y'):
            expected['step-%d/%s.f32' % (step, role)] = ROWS * 4
    for step in LAYER_STEPS:
        expected['step-%d/l_out.f32' % step] = ROWS * 4
    return expected


def read_capture(root, route, expected_weight_sha):
    folder = root / ('route-' + route) / 'q-readback'
    manifest = load_json(folder / 'manifest.json')
    entries = manifest.get('files')
    if not isinstance(entries, dict):
        raise EvidenceError('Capture manifest requires a files object: ' + str(folder))
    expected = required_capture_files()
    if set(entries) != set(expected):
        raise EvidenceError('Capture file coverage differs from the 68 required paths: ' + str(folder))
    files, hashes = {}, {}
    for relative, size in expected.items():
        entry = entries[relative]
        if not isinstance(entry, dict) or entry.get('bytes') != size:
            raise EvidenceError('Capture manifest has wrong byte extent: ' + relative)
        sha = entry.get('sha256')
        if not isinstance(sha, str) or len(sha) != 64:
            raise EvidenceError('Capture manifest lacks a digest: ' + relative)
        files[relative], hashes[relative] = checked_bytes(folder / relative, size, sha)
    if hashes['weights/w.tq2'] != expected_weight_sha:
        raise EvidenceError('Captured layer-0 weight differs from the audited public GGUF: ' + route)
    return files, hashes, digest((folder / 'manifest.json').read_bytes())


def decode_weights(np, payload):
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(ROWS, COLS // BLOCK, BLOCK_BYTES)
    packed = raw[:, :, :64].reshape(ROWS, COLS // BLOCK, 2, 1, 32)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 1, 4, 1)
    codes = (packed >> shifts) & 3
    if np.any(codes == 3):
        raise EvidenceError('Captured TQ2_0 weight contains illegal code 3')
    scales = raw[:, :, 64:].copy().view('<f2').reshape(ROWS, COLS // BLOCK).astype(np.float64)
    if not np.all(np.isfinite(scales)):
        raise EvidenceError('Captured TQ2_0 weight contains nonfinite scales')
    trits = codes.astype(np.int8).reshape(ROWS, COLS // BLOCK, BLOCK) - 1
    return trits.astype(np.float64) * scales[:, :, None]


def cpu_reference(np, decoded, activation):
    x = np.frombuffer(activation, dtype='<f4').astype(np.float64)
    if x.size != COLS or not np.all(np.isfinite(x)):
        raise EvidenceError('Captured Q activation is incomplete/nonfinite')
    # Independent logical K unpack and FP64 arithmetic. No stock coefficients,
    # packed-byte dot expression or SIMD reduction are reused.
    return np.einsum('rbk,bk->r', decoded, x.reshape(COLS // BLOCK, BLOCK), optimize=False)


def fsum_oracle(weight, activation, row):
    x = struct.unpack('<%df' % COLS, activation)
    terms = []
    for k in range(COLS):
        b, inner = divmod(k, BLOCK)
        half, within_half = divmod(inner, 128)
        plane, lane = divmod(within_half, 32)
        start = (row * (COLS // BLOCK) + b) * BLOCK_BYTES
        q = (weight[start + half * 32 + lane] >> (2 * plane)) & 3
        scale = struct.unpack_from('<e', weight, start + 64)[0]
        terms.append((q - 1) * scale * x[k])
    return math.fsum(terms)


def metrics(np, reference, actual):
    a, r = actual.astype(np.float64), reference.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(r)
    error = np.abs(a - r)
    tolerance = ATOL + RTOL * np.abs(r)
    failed = ~finite | (error > tolerance)
    safe_error = np.where(finite, error, np.inf)
    worst = int(np.argmax(safe_error / tolerance))
    ref2 = float(np.dot(r, r))
    err2 = float(np.dot(a - r, a - r))
    rel_l2 = math.sqrt(err2 / ref2) if ref2 else (0.0 if not err2 else math.inf)
    nonzero = r != 0
    max_relative = float(np.max(safe_error[nonzero] / np.abs(r[nonzero]))) if np.any(nonzero) else 0.0
    return {'rows': int(a.size), 'failed_rows': int(np.count_nonzero(failed)),
            'nonfinite_rows': int(np.count_nonzero(~finite)), 'max_abs': float(safe_error.max()),
            'max_error_over_tolerance': float((safe_error / tolerance).max()),
            'relative_L2': rel_l2, 'max_relative_nonzero_reference': max_relative, 'worst_row': worst,
            'zero_reference_rows': int(np.count_nonzero(r == 0)), 'pass': not bool(np.any(failed))}


def top_two(np, values):
    order = np.argsort(-values, kind='stable')[:2]
    return {'top1_token': int(order[0]), 'top1_logit': float(values[order[0]]),
            'top2_token': int(order[1]), 'top2_logit': float(values[order[1]]),
            'top1_top2_margin': float(values[order[0]]) - float(values[order[1]])}


def compare_logits(np, first_root, second_root, first_path, second_path, vocabulary, label):
    rows = []
    for step in range(33):
        a_bytes, a_sha = checked_bytes(first_root / first_path / 'logits' / ('step-%d.f32' % step), vocabulary * 4)
        b_bytes, b_sha = checked_bytes(second_root / second_path / 'logits' / ('step-%d.f32' % step), vocabulary * 4)
        a = np.frombuffer(a_bytes, dtype='<f4')
        b = np.frombuffer(b_bytes, dtype='<f4')
        metric = metrics(np, a, b)
        bits = int(np.count_nonzero(a.view('<u4') != b.view('<u4')))
        rows.append({'comparison': label, 'after_step': step, 'bit_different_values': bits,
                     'reference_sha256': a_sha, 'actual_sha256': b_sha, **metric,
                     'worst_token_id': metric['worst_row'],
                     'worst_reference_logit': float(a[metric['worst_row']]),
                     'worst_actual_logit': float(b[metric['worst_row']]),
                     'reference_top_two': top_two(np, a), 'actual_top_two': top_two(np, b),
                     'greedy_same': int(np.argmax(a)) == int(np.argmax(b))})
    return {'comparison': label, 'vectors': len(rows),
            'bit_different_values': sum(r['bit_different_values'] for r in rows),
            'failed_values': sum(r['failed_rows'] for r in rows),
            'nonfinite_values': sum(r['nonfinite_rows'] for r in rows),
            'tolerance_pass': all(r['pass'] for r in rows),
            'bitwise_pass': all(r['bit_different_values'] == 0 for r in rows),
            'greedy_same_all_vectors': all(r['greedy_same'] for r in rows), 'rows': rows}


def identity(root, expected_model):
    result = load_json(root / 'result.json')
    request = load_json(root / 'request.json')
    config = load_json(root / 'config.json')
    if result.get('model_sha256') != expected_model or request.get('model_sha256') != expected_model:
        raise EvidenceError('Model completion/request identity differs from the audited checkpoint')
    if result.get('device') != 'Apple A16 GPU':
        raise EvidenceError('Expected physical Apple A16 GPU completion identity')
    if result.get('process_exit_code') != 0:
        raise EvidenceError('App process did not report a successful completion')
    if result.get('routes') != ['S', 'Q4']:
        raise EvidenceError('This audit requires exactly stock and layer-0 Q4 routes')
    if config.get('n_ctx') != 512 or config.get('n_batch') != 128 or config.get('n_ubatch') != 128:
        raise EvidenceError('Unexpected context/batch configuration')
    if config.get('kv_k') != 'f16' or config.get('kv_v') != 'f16' or config.get('layers') != '0':
        raise EvidenceError('Unexpected KV precision/layer scope')
    if config.get('threads') != 6 or config.get('flash_attention') is not False:
        raise EvidenceError('Unexpected CPU thread or attention configuration')
    if config.get('callbacks') is not False:
        raise EvidenceError('Callback graph slicing is outside this capture protocol')
    if len(config.get('prompt_ids', [])) != 27 or len(config.get('continuation_ids', [])) != 32:
        raise EvidenceError('Expected exact 27-token prompt and 32 fixed continuation IDs')
    if not isinstance(config.get('vocabulary'), int) or config['vocabulary'] != 128256:
        raise EvidenceError('Unexpected public-model vocabulary')
    return result, request, config


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    return value


def save_json(path, value):
    path.write_text(json.dumps(safe_json(value), indent=2, allow_nan=False) + '\n')


def runtime_evidence(path):
    if not path.is_file():
        raise EvidenceError('Missing runtime log: ' + str(path))
    log = path.read_text(errors='replace')
    records = {'BITNET_Q_READBACK_SHADOW': [], 'BITNET_Q_READBACK_WEIGHT': [],
               'BITNET_FFN_SHADOW_ENCODE': [], 'BITNET_FFN_SHADOW_COMPLETE': [],
               'BITNET_FFN_SHADOW_DISPATCH': []}
    stock_counts = {route: {} for route in ROUTES}
    complete_counts = {route: {} for route in ROUTES}
    current, prefill = None, False
    prefill_begin, prefill_end = [], []
    for line in log.splitlines():
        fields = dict(re.findall(r'(\w+)=(\[[^\]]*\]|[^\s]+)', line))
        for prefix in records:
            if line.startswith(prefix + ' '):
                records[prefix].append(fields)
        if line.startswith('FFNK_PREFILL_BEGIN '):
            prefill = True
            prefill_begin.append(fields.get('route'))
        elif line.startswith('FFNK_PREFILL_END '):
            prefill = False
            prefill_end.append(fields.get('route'))
        elif line.startswith('FFNK_PATH_BEGIN '):
            current = fields.get('route')
        elif line.startswith('FFNK_PATH_END '):
            current = None
        elif line.startswith(('BITNET_MULTILAYER_ENCODE ', 'BITNET_MULTILAYER_COMPLETE ')):
            operation = fields.get('route', '')
            custom = operation == 'q-custom' or operation == 'ffn-fused' or operation.startswith(('ffn-mech-', 'ffn-c-'))
            if prefill and custom:
                raise EvidenceError('Unexpected custom prefill execution/completion')
            if current not in ROUTES:
                continue
            target = complete_counts if line.startswith('BITNET_MULTILAYER_COMPLETE ') else stock_counts
            target[current][operation] = target[current].get(operation, 0) + 1
            if custom and (current != 'Q4' or operation != 'q-custom' or fields.get('layer') != '0'):
                raise EvidenceError('Unexpected non-target custom pipeline in the runtime log')
            if target is stock_counts and operation == 'q-custom':
                if fields.get('node') != 'Qcur-0' or fields.get('weight') != 'blk.0.attn_q.weight' or fields.get('n') != '1':
                    raise EvidenceError('Custom Q dispatch identity/shape differs from the selected real node')
                if fields.get('pipeline') != 'tq2_0_matvec_poc_c8x2' or fields.get('grid') != '[640,1,1]' or fields.get('threads') != '[32,1,1]':
                    raise EvidenceError('Custom Q pipeline or Q4 launch differs from the selected specialization')
            if target is complete_counts and (fields.get('status') != '4' or fields.get('completed') != '1'):
                raise EvidenceError('A runtime dispatch did not complete successfully')
    if prefill or current is not None or prefill_begin != ['S', 'Q4'] or prefill_end != ['S', 'Q4']:
        raise EvidenceError('Missing/unclosed expected stock-prefill or decode path intervals')
    for route in ROUTES:
        if stock_counts[route] != complete_counts[route]:
            raise EvidenceError('Runtime pipeline encode/completion counts differ: ' + route)
        counts = stock_counts[route]
        if counts.get('q-stock', 0) != (960 if route == 'S' else 928) or counts.get('q-custom', 0) != (0 if route == 'S' else 32):
            raise EvidenceError('Runtime Q stock/custom counts are not the expected 960/928+32')
        if counts.get('ffn-stock-down', 0) != 960:
            raise EvidenceError('Stock down coverage differs from the fixed 32-step model replay')
    captures = {}
    for route in ROUTES:
        expected = required_capture_files()
        role_records = {}
        actual_copy_dispatches = 0
        for record in records['BITNET_Q_READBACK_SHADOW']:
            if record.get('route') != route:
                continue
            marker = '/route-' + route + '/q-readback/'
            actual_path = record.get('path', '')
            if marker not in actual_path:
                raise EvidenceError('GPU readback path does not match its named route')
            relative = actual_path.split(marker, 1)[1]
            if relative not in expected or relative in role_records:
                raise EvidenceError('Missing/duplicate/unexpected GPU readback role: ' + relative)
            size = expected[relative]
            if record.get('bytes') != str(size) or record.get('bounds') != '1':
                raise EvidenceError('Runtime readback byte size/bounds check failed')
            role = record.get('role')
            wanted_role = 'w' if relative == 'weights/w.tq2' else Path(relative).stem
            phase = {'w': 'before-Q', 'x': 'before-Q', 'y': 'after-Q-before-RoPE', 'l_out': 'after-layer'}
            if role != wanted_role or record.get('phase') != phase[wanted_role]:
                raise EvidenceError('Readback happened at an incorrect role/graph phase')
            expected_step = '1' if role == 'w' else relative.split('/')[0].split('-')[1]
            if record.get('step') != expected_step or (role == 'y' and record.get('consumed') != '1'):
                raise EvidenceError('Readback step/independent pre-RoPE materialization is inconsistent')
            ne = [2560, 2560, 1, 1] if role == 'w' else [2560, 1, 1, 1]
            nb = [66, 660, 1689600, 1689600] if role == 'w' else [4, 10240, 10240, 10240]
            try:
                actual_ne, actual_nb = json.loads(record['ne']), json.loads(record['nb'])
            except (KeyError, ValueError) as error:
                raise EvidenceError('Invalid runtime tensor shape/stride metadata') from error
            if actual_ne != ne or actual_nb != nb or record.get('type') != ('tq2_0' if role == 'w' else 'f32'):
                raise EvidenceError('Runtime tensor contract differs from the audited Q contract')
            if role == 'w' and record.get('node') != 'blk.0.attn_q.weight':
                raise EvidenceError('Captured the wrong model weight tensor')
            if role == 'y' and record.get('node') != 'Qcur-0':
                raise EvidenceError('Captured the wrong pre-RoPE Q output tensor')
            if role == 'l_out' and record.get('node') != 'l_out-0':
                raise EvidenceError('Captured the wrong residual layer output tensor')
            if record.get('copy_pipeline') != 'ffn_shadow_copy' or record.get('copy_threads') != '[256,1,1]':
                raise EvidenceError('Unexpected diagnostic copy pipeline/launch')
            chunks = 62 if role == 'w' else 1
            if record.get('copy_grid') != ('[27,1,1]' if role == 'w' else '[10,1,1]') or record.get('copy_last_grid') != ('[3,1,1]' if role == 'w' else '[10,1,1]'):
                raise EvidenceError('Unexpected diagnostic chunk-copy grid')
            if record.get('copy_dispatches') != str(chunks):
                raise EvidenceError('Diagnostic copy must cover the full tensor with bounded shader chunks')
            encode = [r for r in records['BITNET_FFN_SHADOW_ENCODE'] if r.get('path') == actual_path]
            complete = [r for r in records['BITNET_FFN_SHADOW_COMPLETE'] if r.get('path') == actual_path]
            if len(encode) != 1 or len(complete) != 1:
                raise EvidenceError('Missing/duplicate GPU encode/completion for ' + relative)
            if encode[0].get('command') != complete[0].get('command') or complete[0].get('status') != '4' or complete[0].get('saved') != '1':
                raise EvidenceError('Diagnostic GPU command did not complete and save successfully')
            if encode[0].get('bytes') != str(size) or complete[0].get('bytes') != str(size):
                raise EvidenceError('Diagnostic command byte ranges differ')
            dispatches = [r for r in records['BITNET_FFN_SHADOW_DISPATCH'] if r.get('path') == actual_path]
            if len(dispatches) != chunks or {r.get('chunk') for r in dispatches} != {str(i) for i in range(chunks)}:
                raise EvidenceError('Missing/duplicate diagnostic copy chunk dispatch')
            by_index = {int(r['chunk']): r for r in dispatches}
            copied = 0
            for chunk in range(chunks):
                entry = by_index[chunk]
                amount = min(6912 * 4, size - copied)
                grid = '[%d,1,1]' % (amount // (256 * 4))
                if entry.get('chunks') != str(chunks) or entry.get('command') != encode[0].get('command') or entry.get('diagnostic_only') != '1':
                    raise EvidenceError('Diagnostic chunk changed command buffer or copy count')
                if entry.get('shadow_offset') != str(copied) or entry.get('source_offset') != str(int(record['offset']) + copied) or entry.get('bytes') != str(amount):
                    raise EvidenceError('Diagnostic chunks overlap, omit bytes or read the wrong source offset')
                if entry.get('grid') != grid or entry.get('threads') != '[256,1,1]' or not 0 < amount <= 6912 * 4:
                    raise EvidenceError('Diagnostic chunk exceeds the frozen copy shader bounds')
                copied += amount
            if copied != size:
                raise EvidenceError('Diagnostic chunks do not cover every tensor byte')
            actual_copy_dispatches += chunks
            role_records[relative] = record
        if set(role_records) != set(expected):
            raise EvidenceError('GPU evidence does not cover all 68 required readback files: ' + route)
        weights = [r for r in records['BITNET_Q_READBACK_WEIGHT'] if r.get('route') == route]
        if len(weights) != 32 or {r.get('step') for r in weights} != {str(s) for s in STEPS}:
            raise EvidenceError('Expected 32 live weight binding checks: ' + route)
        weight_binding = {(r.get('buffer'), r.get('offset'), r.get('path')) for r in weights}
        if len(weight_binding) != 1 or any(r.get('bytes') != str(WEIGHT_BYTES) for r in weights):
            raise EvidenceError('Layer-0 weight binding changed across decode steps')
        if any((r.get('copied'), r.get('cached')) != (('1', '0') if r.get('step') == '1' else ('0', '1')) for r in weights):
            raise EvidenceError('Weight must be copied once then bound/checked on every later step')
        for step in STEPS:
            selected = [role_records['weights/w.tq2'], role_records['step-%d/x.f32' % step], role_records['step-%d/y.f32' % step]]
            for index, first in enumerate(selected):
                for second in selected[index + 1:]:
                    if first['buffer'] == second['buffer']:
                        a, an, b, bn = int(first['offset']), int(first['bytes']), int(second['offset']), int(second['bytes'])
                        if a < b + bn and b < a + an:
                            raise EvidenceError('Live Q input/output buffer byte ranges overlap')
        captures[route] = {'GPU_shadow_copies': len(role_records), 'weight_bindings_checked': len(weights),
                           'actual_diagnostic_copy_dispatches': actual_copy_dispatches,
                           'weight_copies': 1, 'pipeline_encodes': stock_counts[route],
                           'pipeline_completions': complete_counts[route], 'all_copy_commands_completed_and_saved': True}
    return {'pass': True, 'runtime_log_sha256': digest(path.read_bytes()), 'routes': captures,
            'note': 'Actual pipeline/copy encode+completion records, phase/type/shape/stride/bounds and live alias ranges checked; this is correctness instrumentation, not timing.'}


def audit(args, np, output, summary):
    root, off = data_root(args.input), data_root(args.probe_off)
    host_root = root if (root / 'BUILD_IDENTITY.json').is_file() else root.parent
    host_off = off if (off / 'BUILD_IDENTITY.json').is_file() else off.parent
    build, off_build = load_json(host_root / 'BUILD_IDENTITY.json'), load_json(host_off / 'BUILD_IDENTITY.json')
    if build != off_build or build.get('unsigned') is not False or not build.get('app_files_sha256') or not build.get('framework_sha256'):
        raise EvidenceError('Probe-on/off selected signed app/framework build identities differ or are missing')
    summary['selected_build_identity'] = build
    result, request, config = identity(root, args.model_sha256)
    off_result, off_request, off_config = identity(off, args.model_sha256)
    compare_keys = ('n_ctx', 'n_batch', 'n_ubatch', 'threads', 'kv_k', 'kv_v', 'flash_attention',
                    'prompt_ids', 'continuation_ids', 'vocabulary', 'layers')
    if any(config.get(k) != off_config.get(k) for k in compare_keys):
        raise EvidenceError('Probe-on/off effective inputs or context/KV configuration differ')
    if request.get('libraries', {}).get('Q4') != off_request.get('libraries', {}).get('Q4'):
        raise EvidenceError('Probe-on/off Q4 shader library identities differ')
    if result.get('q_readback_enabled') is not True or off_result.get('q_readback_enabled') is not False:
        raise EvidenceError('Expected explicitly enabled/disabled capture runs')
    runtime = runtime_evidence(args.runtime_log)
    save_json(output / 'RUNTIME_EVIDENCE.json', runtime)
    summary['execution_pass'] = runtime['pass']
    captures, hashes, manifests = {}, {}, {}
    for route in ROUTES:
        captures[route], hashes[route], manifests[route] = read_capture(root, route, args.weight_sha256)
    summary['capture_manifest_sha256'] = manifests
    same_weight = hashes['S']['weights/w.tq2'] == hashes['Q4']['weights/w.tq2']
    same_inputs = [step for step in STEPS if hashes['S']['step-%d/x.f32' % step] == hashes['Q4']['step-%d/x.f32' % step]]
    summary['same_input_pass'] = same_weight and len(same_inputs) == 32
    summary['matching_activation_steps'] = same_inputs
    save_json(output / 'INPUT_HASHES.json', {'route_files': hashes, 'capture_manifest_sha256': manifests})
    decoded = {route: decode_weights(np, captures[route]['weights/w.tq2']) for route in ROUTES}
    if same_weight:
        decoded['Q4'] = decoded['S']
    q_cases, parity, oracle = [], [], []
    with (output / 'q-rows.csv').open('w', newline='') as stream, (output / 'failed-q-rows.csv').open('w', newline='') as failed_stream:
        writer = csv.writer(stream)
        failed_writer = csv.writer(failed_stream)
        columns = ['step', 'route', 'row', 'reference_f64', 'actual_f32', 'abs_error', 'tolerance',
                   'error_over_tolerance', 'reference_zero', 'relative_abs_error_nonzero_reference', 'failed', 'nonfinite']
        writer.writerow(columns)
        failed_writer.writerow(columns)
        for step in STEPS:
            ref_cache = {}
            for route in ROUTES:
                activation = captures[route]['step-%d/x.f32' % step]
                x_sha = hashes[route]['step-%d/x.f32' % step]
                key = (hashes[route]['weights/w.tq2'], x_sha)
                if key not in ref_cache:
                    ref_cache[key] = cpu_reference(np, decoded[route], activation)
                ref = ref_cache[key]
                values = np.frombuffer(captures[route]['step-%d/y.f32' % step], dtype='<f4')
                metric = metrics(np, ref, values)
                q_cases.append({'step': step, 'route': route, 'weight_sha256': key[0], 'activation_sha256': x_sha, **metric})
                for row in range(ROWS):
                    e, tolerance = abs(float(values[row]) - float(ref[row])), ATOL + RTOL * abs(float(ref[row]))
                    finite = math.isfinite(float(values[row])) and math.isfinite(float(ref[row]))
                    failed = not finite or e > tolerance
                    record = (step, route, row, float(ref[row]), float(values[row]), e, tolerance, e / tolerance,
                              int(ref[row] == 0), e / abs(float(ref[row])) if ref[row] != 0 else None, int(failed), int(not finite))
                    writer.writerow(record)
                    if failed:
                        failed_writer.writerow(record)
                for row in sorted({0, ROWS - 1, metric['worst_row']}):
                    reference = fsum_oracle(captures[route]['weights/w.tq2'], activation, row)
                    error = abs(reference - float(ref[row]))
                    bound = 1e-10 * max(1.0, abs(reference))
                    oracle.append({'step': step, 'route': route, 'row': row, 'math_fsum': reference,
                                   'NumPy_FP64': float(ref[row]), 'abs_difference': error, 'consistency_bound': bound, 'pass': error <= bound})
            if summary['same_input_pass']:
                a = np.frombuffer(captures['S']['step-%d/y.f32' % step], dtype='<f4')
                b = np.frombuffer(captures['Q4']['step-%d/y.f32' % step], dtype='<f4')
                parity.append({'step': step, **metrics(np, a, b),
                               'bit_different_values': int(np.count_nonzero(a.view('<u4') != b.view('<u4')))})
    summary['Q_FP64_cases'] = q_cases
    summary['Q_stock_parity'] = parity
    summary['parity_pass'] = all(r['pass'] for r in parity) if len(parity) == 32 else None
    summary['Q_all32_bitwise_stock_pass'] = all(r['bit_different_values'] == 0 for r in parity) if len(parity) == 32 else None
    save_json(output / 'FP64_ORACLE.json', {'checks': oracle, 'pass': all(r['pass'] for r in oracle)})
    summary['independent_reference_oracle_pass'] = all(r['pass'] for r in oracle)
    if not summary['independent_reference_oracle_pass']:
        raise EvidenceError('Independent FP64 reference differs from the selected struct/math.fsum oracle')
    summary['fp64_pass'] = all(r['pass'] for r in q_cases)
    layers = []
    for step in LAYER_STEPS:
        a = np.frombuffer(captures['S']['step-%d/l_out.f32' % step], dtype='<f4')
        b = np.frombuffer(captures['Q4']['step-%d/l_out.f32' % step], dtype='<f4')
        layers.append({'step': step, **metrics(np, a, b), 'bit_different_values': int(np.count_nonzero(a.view('<u4') != b.view('<u4')))})
    summary['layer_output_comparisons'] = layers
    logit_checks = [compare_logits(np, root, root, 'route-S', 'route-Q4', config['vocabulary'], 'Q4_vs_S'),
                    compare_logits(np, root, root, 'route-S', 'repeat-S', config['vocabulary'], 'S_probe_on_vs_internal_probe_off'),
                    compare_logits(np, root, root, 'route-Q4', 'repeat-Q4', config['vocabulary'], 'Q4_probe_on_vs_internal_probe_off'),
                    compare_logits(np, root, off, 'route-S', 'route-S', config['vocabulary'], 'S_probe_on_vs_external_probe_off'),
                    compare_logits(np, root, off, 'route-Q4', 'route-Q4', config['vocabulary'], 'Q4_probe_on_vs_external_probe_off')]
    save_json(output / 'LOGITS.json', {'comparisons': logit_checks})
    summary['logit_comparisons'] = [{k: v for k, v in item.items() if k != 'rows'} for item in logit_checks]
    summary['model_output_pass'] = all(r['pass'] for r in layers) and logit_checks[0]['tolerance_pass']
    summary['probe_logit_nonperturbation_pass'] = all(r['bitwise_pass'] for r in logit_checks[1:])
    summary['probe_nonperturbation_scope'] = '33 full logits vectors only; uninstrumented Q/residual outputs are not available and no equality is inferred for them.'
    summary['captured_routes'] = ['S', 'Q4']
    summary['Q_GPU_values_checked'] = 32 * 2 * ROWS
    required = ('execution_pass', 'same_input_pass', 'parity_pass', 'fp64_pass', 'model_output_pass', 'probe_logit_nonperturbation_pass')
    summary['required_flags'] = list(required)
    summary['overall_pass'] = all(summary[k] is True for k in required)
    summary['strict_exit_code'] = 0 if summary['overall_pass'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Fetched probe-on data, or its host run directory')
    parser.add_argument('--probe-off', type=Path, required=True, help='Matched same-build capture-disabled data')
    parser.add_argument('--runtime-log', type=Path, required=True, help='Probe-on console log with original GPU encode/completion records')
    parser.add_argument('--output', type=Path, required=True, help='Fresh local directory; contains raw per-row analysis')
    parser.add_argument('--model-sha256', default=MODEL_SHA)
    parser.add_argument('--weight-sha256', default=WEIGHT_SHA)
    parser.add_argument('--exit-mode', choices=('strict', 'processing'), default='strict')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error('Refusing to reuse an output directory: ' + str(output))
    output.mkdir(parents=True)
    summary = {'schema_version': 1, 'scope': 'All 32 layer-0 Q outputs with identical captured W/X, sampled residual layer outputs and 33 stock-relative full logits vectors.',
               'reference': 'Independent logical TQ2_0 unpack and FP64 CPU arithmetic; selected independent struct/math.fsum rows.',
               'tolerance': {'abs': ATOL, 'relative': RTOL, 'required_failed_rows': 0},
               'analysis_completed': False, 'GPU_executed_by_analyzer': False,
               'execution_pass': None, 'same_input_pass': None, 'parity_pass': None, 'fp64_pass': None, 'model_output_pass': None,
               'probe_logit_nonperturbation_pass': None, 'overall_pass': False, 'strict_exit_code': 2,
               'whole_Transformer_FP64_reference': False, 'model_quality_tested': False,
               'separate_missing_library_or_geometry_fallback_controls_inferred': False,
               'missing_or_invalid': [], 'processing_success_does_not_override_strict_result': True,
               'analyzer_sha256': digest(Path(__file__).read_bytes())}
    try:
        import numpy as np
        summary['NumPy_version'] = np.__version__
        audit(args, np, output, summary)
        summary['analysis_completed'] = True
    except (EvidenceError, OSError, ValueError, KeyError, ImportError) as error:
        summary['missing_or_invalid'].append(str(error))
        summary['strict_exit_code'] = 2
        summary['overall_pass'] = False
        print(str(error), file=sys.stderr)
    actual = summary['strict_exit_code'] if args.exit_mode == 'strict' else (0 if summary['analysis_completed'] else 2)
    summary['actual_cli_exit_code'] = actual
    summary['requested_exit_mode'] = args.exit_mode
    save_json(output / 'SUMMARY.json', summary)
    print(json.dumps({k: summary[k] for k in ('analysis_completed', 'parity_pass', 'fp64_pass', 'model_output_pass', 'overall_pass', 'strict_exit_code', 'actual_cli_exit_code')}))
    return actual


if __name__ == '__main__':
    sys.exit(main())
