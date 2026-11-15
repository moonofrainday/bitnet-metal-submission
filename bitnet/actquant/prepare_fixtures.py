#!/usr/bin/env python3
"""Prepare local F32 ActQuant cases from the immutable Microsoft reference."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SOURCE_SHA = "a8b3170f1824f010701f4e033fbc33d199bc2ea344ddbf2554594b66fada8b1c"
EPS = 1e-5
BOUNDARY_BITS = [[1099879742, 3247363390, 1045950396, 3193434044, 0, 1029173180, 3176656828]]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_quantizer(path):
    if sha(path) != SOURCE_SHA:
        raise ValueError("The official ActQuant source hash does not match")
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ActQuant"]
    if len(selected) != 1:
        raise ValueError("Expected exactly one official ActQuant class")
    for node in ast.walk(selected[0]):
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = [d for d in node.decorator_list if ast.unparse(d) != "torch.compile"]
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    env = {"torch": torch}
    exec(compile(module, str(path), "exec"), env)
    return env["ActQuant"]


def array_file(path, values, dtype):
    a = np.asarray(values, dtype=dtype)
    a.tofile(path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}


def build_cases(real):
    cases = []
    for width in (2560, 6912):
        cases.append((f"zero-{width}-n1", np.zeros((1, width), np.float32), "All-zero denominator clamp"))
        x = np.zeros((3, width), np.float32)
        x[0, ::2] = np.float32(-0.0)
        x[1, :5] = [1e-8, -1e-8, 1e-7, -1e-7, 0]
        x[2, :5] = [1e-5, -1e-5, 2e-6, -2e-6, 0]
        cases.append((f"epsilon-{width}-n3", x, "Three independent tokens: signed zero, below epsilon, epsilon boundary"))
        x = np.zeros((1, width), np.float32)
        ties = np.array([.5, 1.5, 2.5, 3.5, -.5, -1.5, -2.5, -3.5,
                         126.5, -126.5, 127, -127, 0, -0.0], np.float32)
        x[0, :len(ties)] = ties
        cases.append((f"ties-{width}-n1", x, "Absmax 127 anchors exact F32 scale=1; positive and negative half ties"))
        x = np.zeros((3, width), np.float32)
        levels = [np.nextafter(np.float32(EPS), np.float32(0)),
                  np.float32(EPS), np.nextafter(np.float32(EPS), np.float32(np.inf))]
        for row, level in enumerate(levels):
            x[row, ::2] = level
            x[row, 1::2] = -level * np.float32(.25)
        cases.append((f"epsilon-neighbors-{width}-n3", x, "Adjacent F32 absmax values below/at/above epsilon"))
        for rows in (1, 3, 27):
            rng = np.random.default_rng(8419 + width + rows)
            x = rng.standard_normal((rows, width), dtype=np.float32)
            amplitudes = np.resize(np.array([.000003, 1, 103], np.float32), rows)
            x *= amplitudes[:, None]
            cases.append((f"dense-{width}-n{rows}", x, "Seeded dense inputs with different token amplitudes"))
    if real is not None:
        for step in (1, 16, 32):
            path = real / f"step-{step}" / "x.f32"
            x = np.fromfile(path, dtype="<f4")
            if x.size != 2560 or not np.isfinite(x).all():
                raise ValueError("Invalid captured Q input at step " + str(step))
            cases.append((f"real-q-step-{step}", x.reshape(1, -1), "Unchanged actual pre-RoPE layer-0 Q input"))
    return cases


def boundary_cases(capture, failures, counterexample):
    cases, provenance = [], {}
    if capture is not None:
        run_path = capture / "RUN.json"
        run = json.loads(run_path.read_text())
        records = json.loads(failures.read_text())
        for failure in records:
            record = run["checkpoints"][failure["capture"]]
            if record["node"] != failure["node"] or record["kind"] != "actquant":
                raise ValueError("Boundary capture identity differs")
            width = record["ne"][0]
            input_path = capture / record["input_path"]
            full = np.fromfile(input_path, dtype="<f4").reshape(-1, width)
            row = failure["row"]
            if record["input_positions"][row] != failure["input_position"]:
                raise ValueError("Boundary token mapping differs")
            x = np.array(full[row:row + 1], copy=True)
            name = f"boundary-real-layer{failure['layer']}-pos{failure['input_position']}"
            cases.append((name, x, "Complete unchanged real row that exposed a pinned-source floating evaluation difference"))
            provenance[name] = {"native_run_manifest_sha256": sha(run_path),
                                "capture_input_sha256": sha(input_path), "capture_index": failure["capture"],
                                "row": row, "node": record["node"], "input_position": failure["input_position"],
                                "failure_record_sha256": sha(failures)}
    bits = np.asarray(BOUNDARY_BITS, dtype="<u4")
    x = np.array(bits.view("<f4"), copy=True)
    name = "boundary-reciprocal-vs-divide-k7"
    provenance[name] = {"input_reconstruction": "Constructed IEEE754 uint32 constants in the source generator; no model data"}
    if counterexample is not None:
        record = json.loads(counterexample.read_text())
        if not np.array_equal(bits, np.asarray(record["input_u32_bits"], dtype="<u4")) or not np.array_equal(x.view(np.uint32), np.asarray(record["input"], dtype="<f4").view(np.uint32)):
            raise ValueError("Seven-value boundary input bits differ")
        provenance[name]["counterexample_sha256"] = sha(counterexample)
    cases.append((name, x, "Constructed seven-value counterexample to algebraic scale equivalence"))
    return cases, provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-source", type=Path, required=True)
    scope = p.add_mutually_exclusive_group(required=True)
    scope.add_argument("--real-fixture", type=Path)
    scope.add_argument("--synthetic-only", action="store_true", help="Explicit model-free scope; real capture checks remain missing")
    p.add_argument("--boundary-capture", type=Path)
    p.add_argument("--boundary-failures", type=Path)
    p.add_argument("--counterexample", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Output must be a new local directory")
    if (a.boundary_capture is None) != (a.boundary_failures is None):
        p.error("Boundary capture and failure records must be supplied together")
    if a.synthetic_only and a.boundary_capture is not None:
        p.error("Synthetic-only mode cannot include captured model rows")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reference = official_quantizer(a.official_source)
    a.output.mkdir(parents=True)
    items = []
    extra, provenance = boundary_cases(a.boundary_capture, a.boundary_failures, a.counterexample)
    with torch.inference_mode():
        for name, x, description in [*build_cases(a.real_fixture), *extra]:
            folder = a.output / name
            folder.mkdir()
            tensor = torch.from_numpy(x.copy())
            expected = reference.apply(tensor)
            scale = 127 / tensor.abs().max(dim=-1, keepdim=True).values.clamp(min=EPS)
            codes = (tensor * scale).round().clamp(-128, 127)
            reconstructed = codes / scale
            if not torch.equal(expected.view(torch.int32), reconstructed.view(torch.int32)):
                raise ValueError("Official output does not match metadata calculation")
            if name.startswith("ties-"):
                manual = torch.tensor([0, 2, 2, 4, 0, -2, -2, -4, 126, -126, 127, -127, 0, 0])
                if not torch.equal(codes[0, :manual.numel()].to(torch.int32), manual):
                    raise ValueError("The hand-derived ties-to-even control failed")
                if float(scale[0, 0]) != 1.0:
                    raise ValueError("Half-tie control does not have exact scale 1")
            if name == "boundary-reciprocal-vs-divide-k7":
                if codes[0].to(torch.int32).tolist() != [127, -127, 2, -2, 0, 0, 0]:
                    raise ValueError("Pinned eager floating profile changed at the reciprocal boundary")
            width, rows = x.shape[1], x.shape[0]
            records = {
                "input": array_file(folder / "x.f32", x, "<f4"),
                "expected": array_file(folder / "expected.f32", expected.numpy(), "<f4"),
                "scale": array_file(folder / "scale.f32", scale.numpy(), "<f4"),
                "int8_codes": array_file(folder / "q.s8", codes.numpy(), "i1"),
            }
            item = {"name": name, "description": description, "dtype": "F32", "rows": rows,
                    "width": width, "elements": int(x.size), "ne": [width, rows, 1, 1],
                    "nb": [4, 4 * width, 4 * width * rows, 4 * width * rows]}
            for role, record in records.items():
                item[role] = f"{name}/{record['path']}"
                item[role + "_sha256"] = record["sha256"]
            item["changed_values"] = int(np.count_nonzero(x != expected.numpy()))
            items.append(item)
    manifest = {"schema": 1, "scope": "CPU-generated local F32 ActQuant fixtures; no GPU executed",
                "source_revision": "096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4",
                "official_source_sha256": SOURCE_SHA, "epsilon": EPS, "cases": items,
                "torch": torch.__version__, "numpy": np.__version__,
                "rule": "F32 per-token absmax, denominator clamp1e-5, 127/denominator, ties-to-even, clamp[-128,127], divide by scale",
                "floating_execution_profile": "Exact pinned official AST under the recorded Torch eager CPU build. Its scalar127/tensor expression evaluates reciprocal(tensor)*127; algebraic direct division can differ near half boundaries.",
                "boundary_sources": provenance,
                "real_capture_included": a.real_fixture is not None,
                "missing_real_capture": a.real_fixture is None,
                "frozen_bound": {"absolute": 2e-4, "relative": 2e-5},
                "quantized_integer_codes_must_match": True,
                "model_semantics_pass": None, "model_quality_pass": None,
                "raw_payloads_must_remain_local": True}
    (a.output / "FIXTURES.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"fixture_cases": len(items), "values": sum(v["elements"] for v in items),
                      "official_source_verified": True, "hand_ties_verified": True,
                      "GPU_executed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
