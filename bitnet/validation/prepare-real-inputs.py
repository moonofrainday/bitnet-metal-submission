#!/usr/bin/env python3
"""Copy and verify three captured Q inputs; no compilation, GPU or downloads.

The manifest contains only newly copied binaries. The historical audit supplies
expected input hashes, never expected GPU outputs or a reused FP64 reference.
Optional GGUF verification imports gguf-py only from the explicit fresh checkout.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys

M = K = 2560
BLOCK = 256
BLOCK_BYTES = 66
ROW_BYTES = (K // BLOCK) * BLOCK_BYTES
WEIGHT_BYTES = M * ROW_BYTES
ACTIVATION_BYTES = K * 4
STEPS = (1, 16, 32)
EXPECTED_QVAC = '2db8b046d555316019ae75d1be90484adc8b9330'
EXPECTED_MODEL = '52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2'


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def regular(path: Path) -> Path:
    need(path.is_file(), f'Missing regular input: {path}')
    return path.resolve(strict=True)


def packed_validation(data: bytes) -> dict:
    need(len(data) == WEIGHT_BYTES, 'Weight size violates M=K=2560 TQ2_0 contract')
    scales = bytearray()
    minimum = math.inf
    maximum = -math.inf
    illegal = 0
    nonfinite = 0
    for offset in range(0, len(data), BLOCK_BYTES):
        codes = data[offset:offset + 64]
        for value in codes:
            # Count all code slots, not just bytes containing a reserved code.
            illegal += sum(((value >> shift) & 3) == 3 for shift in (0, 2, 4, 6))
        half_bytes = data[offset + 64:offset + 66]
        scales.extend(half_bytes)
        scale = struct.unpack('<e', half_bytes)[0]
        nonfinite += not math.isfinite(scale)
        minimum = min(minimum, scale)
        maximum = max(maximum, scale)
    need(illegal == 0, f'Illegal TQ2_0 code 3: {illegal} slots')
    need(nonfinite == 0, f'Nonfinite FP16 block scales: {nonfinite}')
    return {
        'block_count': len(data) // BLOCK_BYTES,
        'block_scale_sha256': hashlib.sha256(scales).hexdigest(),
        'min_scale': minimum, 'max_scale': maximum,
        'illegal_code_slots': illegal, 'nonfinite_scales': nonfinite,
        'scale_encoding': 'little-endian IEEE binary16 at byte offsets 64:66',
    }


def activation_validation(data: bytes) -> dict:
    need(len(data) == ACTIVATION_BYTES, 'Activation must contain exactly 2560 F32 values')
    values = struct.unpack('<2560f', data)
    nonfinite = sum(not math.isfinite(value) for value in values)
    need(nonfinite == 0, f'Nonfinite activation values: {nonfinite}')
    return {'values': len(values), 'nonfinite': nonfinite,
            'minimum': min(values), 'maximum': max(values),
            'l2': math.sqrt(math.fsum(value * value for value in values))}


def checked_copy(source: Path, destination: Path, expected: str) -> dict:
    need(not destination.exists(), f'Refuse overwriting copied input: {destination}')
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination, follow_symlinks=True)
    after = source.stat()
    need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
         f'Input changed during copy: {source}')
    need(not destination.is_symlink(), 'Copied input cannot be a symlink')
    dest_stat = destination.stat()
    need((before.st_dev, before.st_ino) != (dest_stat.st_dev, dest_stat.st_ino),
         'Copied input must have independent file identity')
    observed = sha(destination)
    need(observed == expected, f'Copied SHA mismatch: {destination}')
    return {'source': str(source), 'copy': str(destination.resolve()),
            'bytes': dest_stat.st_size, 'sha256': observed,
            'independent_regular_file': True}


def verify_model(model: Path, source: Path, packed: bytes) -> dict:
    observed = sha(model)
    need(observed == EXPECTED_MODEL, 'GGUF SHA does not identify the validated public 2B model')
    module_dir = (source / 'gguf-py').resolve(strict=True)
    sys.path.insert(0, str(module_dir))
    gguf = importlib.import_module('gguf')
    numpy = importlib.import_module('numpy')
    need(Path(gguf.__file__).resolve().is_relative_to(module_dir),
         'gguf module did not come from the explicit fresh source')
    sizes = tuple(gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.TQ2_0])
    need(sizes == (BLOCK, BLOCK_BYTES), 'Fresh gguf-py TQ2_0 block ABI differs')
    reader = gguf.GGUFReader(str(model), 'r')
    matches = [tensor for tensor in reader.tensors if tensor.name == 'blk.0.attn_q.weight']
    need(len(matches) == 1, 'Expected exactly one blk.0.attn_q.weight')
    tensor = matches[0]
    need(tensor.tensor_type == gguf.GGMLQuantizationType.TQ2_0, 'Selected tensor is not TQ2_0')
    need(list(map(int, tensor.shape)) == [K, M], 'GGUF selected tensor shape mismatch')
    data = tensor.data.tobytes(order='C')
    need(data == packed, 'Captured packed weights differ from exact GGUF tensor bytes')
    return {
        'performed': True, 'model': str(model), 'model_bytes': model.stat().st_size,
        'model_sha256': observed, 'tensor': tensor.name, 'tensor_type': 'TQ2_0',
        'tensor_shape_ggml': list(map(int, tensor.shape)),
        'tensor_bytes': len(data), 'tensor_sha256': hashlib.sha256(data).hexdigest(),
        'captured_tensor_bytes_exact': True,
        'gguf_module': str(Path(gguf.__file__).resolve()), 'numpy_version': numpy.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True,
                        help='Captured stock root containing data/, or data/ itself')
    parser.add_argument('--audit', type=Path, required=True, help='Saved input audit JSON')
    parser.add_argument('--output', type=Path, required=True,
                        help='Fresh local-only inputs directory; existing paths are refused')
    parser.add_argument('--qvac-source', type=Path, required=True,
                        help='Explicit clean pinned QVAC checkout (never an installed gguf package)')
    parser.add_argument('--model', type=Path,
                        help='Optional existing public GGUF; read-only hash and tensor verification')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        print(f'INPUT_PREPARATION_BLOCKED: Refuse existing output: {output}', file=sys.stderr)
        return 2
    try:
        capture = args.capture_root.resolve(strict=True)
        data_root = capture / 'data' if (capture / 'data').is_dir() else capture
        audit_path = regular(args.audit)
        audit = json.loads(audit_path.read_text())
        source = args.qvac_source.resolve(strict=True)
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        need(commit == EXPECTED_QVAC, 'QVAC source is not the pinned baseline')
        dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--', 'gguf-py'], text=True)
        need(not dirty.strip(), 'Fresh gguf-py source has local modifications')
        weight_path = regular(data_root / 'weights/layer-0/q.tq2')
        weight = weight_path.read_bytes()
        expected_weight = audit['weight']['sha256']
        need(audit['weight']['bytes'] == WEIGHT_BYTES, 'Audit weight size mismatch')
        need(hashlib.sha256(weight).hexdigest() == expected_weight, 'Captured W differs from audited SHA')
        validation = packed_validation(weight)
        need(validation['block_scale_sha256'] == audit['weight']['block_scale_sha256'],
             'Captured block scales differ from audit')
        cases = []
        source_activations = []
        for step in STEPS:
            entry = next(item for item in audit['steps'] if item['step'] == step)
            need(entry['weight_stock_sha256'] == expected_weight, f'Step {step} weight identity mismatch')
            path = regular(data_root / f'step-{step}/layer-0/q/x.f32')
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            need(digest == entry['x_stock_sha256'], f'Step {step} X differs from audited SHA')
            source_activations.append((step, path, digest, activation_validation(payload)))
        model_check = verify_model(regular(args.model), source, weight) if args.model else {
            'performed': False, 'reason': '--model not supplied; no checkpoint claim is inferred'}
        output.mkdir(parents=True)
        copied_weight = checked_copy(weight_path, output / 'weights/q.tq2', expected_weight)
        copies = [copied_weight]
        for step, path, digest, x_validation in source_activations:
            copied_x = checked_copy(path, output / f'real-step-{step}/x.f32', digest)
            copies.append(copied_x)
            cases.append({'id': f'real-step-{step}', 'step': step,
                          'weights': copied_weight['copy'], 'activation': copied_x['copy'],
                          'weights_sha256': expected_weight, 'activation_sha256': digest,
                          'activation_validation': x_validation})
        manifest = {
            'schema_version': 1, 'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope': 'public-2B layer0 pre-RoPE Q; captured stock W/X; three operator inputs only',
            'GPU_executed': False, 'old_GPU_output_or_FP64_reference_reused': False,
            'contract': {'M': M, 'K': K, 'N': 1, 'weights': 'TQ2_0', 'activation': 'F32', 'output': 'F32',
                         'block_values': BLOCK, 'block_bytes': BLOCK_BYTES, 'row_bytes': ROW_BYTES,
                         'weights_bytes': WEIGHT_BYTES, 'activation_bytes': ACTIVATION_BYTES,
                         'W_ne': [K, M, 1, 1], 'W_nb': [66, 660, 1689600, 1689600]},
            'weight_validation': validation,
            'provenance': {'capture_root': str(capture), 'audit': str(audit_path),
                           'audit_sha256': sha(audit_path), 'qvac_source': str(source),
                           'qvac_commit': commit, 'gguf_py_clean': True,
                           'python': sys.version, 'script_sha256': sha(Path(__file__))},
            'model_verification': model_check, 'copies': copies, 'cases': cases,
        }
        target = output / 'real-inputs.json'
        target.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'input_preparation_pass': True, 'cases': len(cases),
                          'manifest': str(target), 'model_verification_performed': model_check['performed']}))
        return 0
    except (ValueError, OSError, KeyError, StopIteration, subprocess.CalledProcessError, ImportError) as error:
        report = {'input_preparation_pass': False, 'error': f'{type(error).__name__}: {error}',
                  'GPU_executed': False, 'old_data_modified': False}
        if output.is_dir():
            (output / 'PREPARATION_FAILURE.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
