#!/usr/bin/env python3
"""CPU-only verification of saved F32 ActQuant outputs and integer-grid codes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ABS = 2e-4
REL = 2e-5
SOURCE_SHA = "a8b3170f1824f010701f4e033fbc33d199bc2ea344ddbf2554594b66fada8b1c"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_array(path, dtype, elements):
    values = np.fromfile(path, dtype=dtype)
    if values.size != elements:
        raise ValueError("Unexpected element count for " + path.name)
    return values


def ulps(a, b):
    def ordered(x):
        u = np.asarray(x, dtype=np.float32).view(np.uint32)
        return np.where(u & 0x80000000, ~u, u | np.uint32(0x80000000)).astype(np.int64)
    return np.abs(ordered(a) - ordered(b))


def check(args):
    fixture_path = args.fixture / "FIXTURES.json"
    fixture = json.loads(fixture_path.read_text())
    if (fixture["schema"] != 1 or fixture["epsilon"] != 1e-5
            or fixture["official_source_sha256"] != SOURCE_SHA):
        raise ValueError("Unsupported fixture schema or epsilon")
    runtime_path = args.runtime or args.actual / "RUN.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else None
    if runtime is not None and runtime.get("fixture_sha256") != sha(fixture_path):
        raise ValueError("Runtime fixture hash differs")
    runtime_cases = {v["name"]: v for v in runtime.get("cases", [])} if runtime else {}
    cases = []
    failed_elements = []
    for case in fixture["cases"]:
        name, count, width, rows = case["name"], case["elements"], case["width"], case["rows"]
        if case["ne"] != [width, rows, 1, 1] or count != width * rows:
            raise ValueError("Fixture tensor extent differs")
        raw = {}
        for role, dtype, size in [("input", "<f4", count), ("expected", "<f4", count),
                                  ("scale", "<f4", rows), ("int8_codes", "i1", count)]:
            p = args.fixture / case[role]
            if sha(p) != case[role + "_sha256"]:
                raise ValueError("Fixture payload hash differs: " + name + "/" + role)
            raw[role] = read_array(p, dtype, size)
        path = args.actual / args.path_template.format(name=name)
        actual = read_array(path, "<f4", count)
        expected = raw["expected"]
        finite = np.isfinite(actual)
        reference_finite = np.isfinite(expected)
        if not reference_finite.all() or not np.isfinite(raw["input"]).all():
            raise ValueError("Nonfinite reference or input")
        scale = np.repeat(raw["scale"], width)
        if np.any(scale <= 0) or not np.isfinite(scale).all():
            raise ValueError("Invalid official scale")
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        tolerance = ABS + REL * np.abs(expected.astype(np.float64))
        numeric_bad = ~finite | (error > tolerance)
        with np.errstate(over="ignore", invalid="ignore"):
            # Use F64 recovery to avoid adding another F32 multiplication rounding.
            recovered = np.rint(actual.astype(np.float64) * scale.astype(np.float64))
        expected_codes = raw["int8_codes"].astype(np.float64)
        code_bad = ~finite | (recovered != expected_codes)
        bit_bad = actual.view(np.uint32) != expected.view(np.uint32)
        dist = ulps(actual, expected)
        native = runtime_cases.get(name)
        native_valid = None
        if native is not None:
            native_valid = (native.get("completed") is True
                            and native.get("input_sha256") == case["input_sha256"]
                            and native.get("output_sha256") == sha(path)
                            and native.get("ne") == case["ne"])
        if args.require_runtime and native_valid is not True:
            raise ValueError("Required native completion/input/output identity missing: " + name)
        details = {}
        native_scale = args.actual / name / "scale.f32"
        native_codes = args.actual / name / "q.s8"
        if native_scale.is_file():
            a_scale = read_array(native_scale, "<f4", rows)
            details["exported_scale_finite"] = bool(np.isfinite(a_scale).all())
            details["exported_scale_bitwise_differences"] = int(np.count_nonzero(a_scale.view(np.uint32) != raw["scale"].view(np.uint32)))
            details["exported_scale_max_ULP_distance"] = int(ulps(a_scale, raw["scale"]).max())
            scale_error = np.abs(a_scale.astype(np.float64) - raw["scale"].astype(np.float64))
            details["exported_scale_tolerance_failures"] = int(np.count_nonzero(~np.isfinite(a_scale) | (scale_error > ABS + REL * np.abs(raw["scale"].astype(np.float64)))))
        else:
            details["exported_scale_max_ULP_distance"] = None
            details["exported_scale_tolerance_failures"] = None
        if native_codes.is_file():
            codes = read_array(native_codes, "i1", count)
            details["exported_integer_code_mismatches"] = int(np.count_nonzero(codes != raw["int8_codes"]))
        else:
            details["exported_integer_code_mismatches"] = None
        finite_errors = error[finite]
        ref_norm = float(np.linalg.norm(expected.astype(np.float64)))
        item = {"name": name, "ne": case["ne"], "values": count,
                "input_sha256": case["input_sha256"], "reference_sha256": case["expected_sha256"],
                "output_sha256": sha(path), "nonfinite": int(np.count_nonzero(~finite)),
                "frozen_tolerance_failures": int(np.count_nonzero(numeric_bad)),
                "recovered_integer_code_mismatches": int(np.count_nonzero(code_bad)),
                "output_bitwise_differences": int(np.count_nonzero(bit_bad)),
                "max_abs_error": float(finite_errors.max()) if finite_errors.size else None,
                "max_error_over_tolerance": float((error[finite] / tolerance[finite]).max()) if finite_errors.size else None,
                "relative_L2": float(np.linalg.norm(actual.astype(np.float64) - expected.astype(np.float64)) / ref_norm) if finite.all() and ref_norm else None,
                "max_value_ULP_distance": int(dist[finite].max()) if finite.any() else None,
                "runtime_identity_and_completion_pass": native_valid, **details}
        item["numerical_pass"] = (item["nonfinite"] == 0 and item["frozen_tolerance_failures"] == 0
                                  and item["recovered_integer_code_mismatches"] == 0
                                  and details["exported_integer_code_mismatches"] in (None, 0)
                                  and details["exported_scale_tolerance_failures"] in (None, 0))
        cases.append(item)
        for index in np.flatnonzero(numeric_bad | code_bad):
            failed_elements.append({"case": name, "row": int(index // width), "column": int(index % width),
                                    "reference": float(expected[index]), "actual": float(actual[index]) if finite[index] else str(actual[index]),
                                    "absolute_error": float(error[index]) if finite[index] else None,
                                    "tolerance": float(tolerance[index]), "reference_code": int(expected_codes[index]),
                                    "recovered_output_code": int(recovered[index]) if np.isfinite(recovered[index]) else None,
                                    "frozen_tolerance_failed": bool(numeric_bad[index]), "integer_code_failed": bool(code_bad[index])})
    passed = all(v["numerical_pass"] for v in cases)
    result = {"schema": 1, "CPU_only_analysis": True, "model_or_GPU_executed_by_analyzer": False,
              "fixture_manifest_sha256": sha(fixture_path), "official_source_sha256": fixture["official_source_sha256"],
              "comparison_scope": "Same supplied F32 input, official F32 ActQuant output and quantized grid; no model-quality acceptance",
              "frozen_tolerance": {"absolute": ABS, "relative": REL},
              "required_integer_code_control": "Recover round(output * official_scale) in F64; this is an output-grid control, not direct observation of an internal integer register",
              "ULP_convention": "Monotone IEEE754 F32 bit ordering; negative/positive zero are adjacent and counted as different bits",
              "cases": cases, "values": sum(v["values"] for v in cases),
              "numerical_pass": passed, "execution_pass": all(v["runtime_identity_and_completion_pass"] is True for v in cases) if runtime else None,
              "model_output_pass": None, "model_semantics_pass": None, "model_quality_pass": None,
              "analysis_completed": True, "strict_operator_exit_code": 0 if passed else 1}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "SUMMARY.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.output / "failures-local.csv").open("w", newline="") as f:
        fields = ["case", "row", "column", "reference", "actual", "absolute_error", "tolerance", "reference_code", "recovered_output_code", "frozen_tolerance_failed", "integer_code_failed"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failed_elements)
    print(json.dumps({"cases": len(cases), "values": result["values"], "numerical_pass": passed,
                      "failed_values": len(failed_elements), "strict_operator_exit_code": result["strict_operator_exit_code"]}, indent=2))
    return result["strict_operator_exit_code"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--actual", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--path-template", default="{name}/y.f32")
    p.add_argument("--runtime", type=Path)
    p.add_argument("--require-runtime", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("Output must be new")
    try:
        return check(args)
    except FileNotFoundError as error:
        print(json.dumps({"analysis_completed": False, "missing": error.filename, "strict_operator_exit_code": 2}), file=sys.stderr)
        return 2
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        print(json.dumps({"analysis_completed": False, "malformed": str(error), "strict_operator_exit_code": 3}), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
