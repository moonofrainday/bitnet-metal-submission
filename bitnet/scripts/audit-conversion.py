#!/usr/bin/env python3
"""Audit all 210 public BitNet matrices against direct AutoBitLinear scales.

This is the portable version of the corrected public-2B conversion audit.
It checks packed trits and FP16 block scales, not inference or model quality.
Requires NumPy, PyTorch, safetensors, and the repository's gguf-py package.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MATRICES = 210
PINNED_QVAC = "2db8b046d555316019ae75d1be90484adc8b9330"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True,
                        help="Converted public 2B TQ2_0 GGUF file")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Original packed public model.safetensors file")
    parser.add_argument("--output", type=Path, required=True,
                        help="Local JSON audit result, including per-matrix failures")
    return parser.parse_args()


def main():
    args = parse_args()
    model = args.gguf.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require(output not in [model, checkpoint], "Output must not overwrite an input file")
    report = {"model": str(model), "source": str(checkpoint),
              "pinned_qvac": PINNED_QVAC,
              "historical_implementation": "2026-11-07_133437-public-2b-compat/audit_conversion.py",
              "reference": "Original packed trits times direct AutoBitLinear weight_scale",
              "expected_matrix_count": EXPECTED_MATRICES,
              "matrix_count": 0, "rows": [], "passed": False,
              "GPU_executed": False}
    try:
        require(model.is_file(), "Missing GGUF: " + str(model))
        require(checkpoint.is_file(), "Missing safetensors checkpoint: " + str(checkpoint))
        require((ROOT / "gguf-py" / "gguf").is_dir(), "In-tree gguf-py package is missing")
        sys.path.insert(0, str(ROOT / "gguf-py"))
        import numpy as np
        from safetensors import safe_open
        import gguf

        reader = gguf.GGUFReader(model)
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        mapper = gguf.get_tensor_name_map(gguf.MODEL_ARCH.BITNET, 30)
        with safe_open(checkpoint, framework="pt", device="cpu") as packed:
            for name in packed.keys():
                if not name.endswith(".weight_scale"):
                    continue
                weight_name = name.removesuffix("_scale")
                target_name = mapper.get_name(weight_name, try_suffixes=(".weight", ".bias"))
                require(target_name in tensors, "Missing converted tensor: " + str(target_name))
                tensor = tensors[target_name]
                require(tensor.tensor_type == gguf.GGMLQuantizationType.TQ2_0,
                        "Expected TQ2_0: " + target_name)
                data = packed.get_tensor(weight_name).numpy()
                scale = float(packed.get_tensor(name).float().item())
                require(math.isfinite(scale) and scale > 0,
                        "Invalid direct AutoBitLinear scale: " + name)
                # The checkpoint uses contiguous row planes. TQ2_0 uses 32-wide bit planes.
                q = np.concatenate([((data >> shift) & 3).astype(np.int8) - 1
                                    for shift in (0, 2, 4, 6)], axis=0)
                require(bool(np.all((q >= -1) & (q <= 1))),
                        "Invalid source trit in " + weight_name)
                raw = np.asarray(tensor.data).reshape(-1, 66)
                qs = raw[:, :64].reshape(-1, 2, 1, 32)
                shifts = np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
                unpacked = (((qs >> shifts) & 3).astype(np.int8) - 1).reshape(q.shape)
                code_mismatches = int(np.count_nonzero(q != unpacked))
                scales = raw[:, 64:].copy().view("<f2").reshape(-1).astype(np.float32)
                expected_scale = float(np.float16(scale))
                require(math.isfinite(expected_scale), "Direct scale overflows FP16: " + name)
                expected_scales = np.where(np.any(q.reshape(-1, 256) != 0, axis=1),
                                           expected_scale, 0).astype(np.float32)
                scale_abs = float(np.max(np.abs(scales - expected_scales)))
                report["rows"].append({
                    "source": weight_name, "gguf": target_name, "shape": list(q.shape),
                    "source_direct_scale": scale,
                    "expected_fp16_block_scale": expected_scale,
                    "actual_block_scale_min": float(scales.min()),
                    "actual_block_scale_max": float(scales.max()),
                    "trit_mismatches": code_mismatches,
                    "scale_max_abs_error": scale_abs,
                })
        report["matrix_count"] = len(report["rows"])
        report["passed"] = report["matrix_count"] == EXPECTED_MATRICES and all(
            row["trit_mismatches"] == 0 and row["scale_max_abs_error"] == 0
            for row in report["rows"])
        report["gguf_sha256"] = sha256(model)
        report["checkpoint_sha256"] = sha256(checkpoint)
        report["exit_code"] = 0 if report["passed"] else 1
    except Exception as error:
        report["matrix_count"] = len(report["rows"])
        report["error"] = str(error)
        report["exit_code"] = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("packing/scale audit:", report["passed"], "matrices:", report["matrix_count"])
    if "error" in report:
        print(report["error"], file=sys.stderr)
    for row in report["rows"]:
        if row["trit_mismatches"] or row["scale_max_abs_error"]:
            print(json.dumps(row))
            break
    print(str(output))
    return report["exit_code"]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
