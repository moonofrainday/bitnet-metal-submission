#!/usr/bin/env python3
"""CPU-only reanalysis of the fixed Mac Qcur-0 operator archive.

Requires the six step-1/16/32 stock/Q4 correctness cases and the complete
three-round AA/SQ4 protocol. No GPU, old dylib, model or device is needed.
Exit 0 validates this archived operator scope; it is not model acceptance.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


MODEL_SHA256 = "52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2"
CASES = {(step, route) for step in (1, 16, 32) for route in ("S", "Q4")}


class EvidenceError(ValueError):
    pass


class QualificationFailure(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(row, field):
    value = float(row[field])
    require(math.isfinite(value), "Nonfinite field: " + field)
    return value


def percentile(values, p):
    ordered = sorted(values)
    require(bool(ordered), "Empty statistics")
    position = (len(ordered)-1)*p
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (position-lo)*(ordered[hi]-ordered[lo])


def stats(rows):
    a = [number(r, "a_us") for r in rows]
    b = [number(r, "b_us") for r in rows]
    difference = [bv-av for av, bv in zip(a, b)]
    return {"pairs": len(rows), "a_p50_us": percentile(a, .5), "a_p95_us": percentile(a, .95),
            "b_p50_us": percentile(b, .5), "b_p95_us": percentile(b, .95),
            "paired_B_minus_A_median_us": statistics.median(difference),
            "paired_relative_B_minus_A_median_percent": statistics.median(100*d/av for d, av in zip(difference, a)),
            "B_wins": sum(d < 0 for d in difference), "ties": sum(d == 0 for d in difference)}


def correctness(root, qualification):
    cases = qualification["cases"]
    by_case = {(r["step"], r["route"]): r for r in cases}
    require(len(cases) == len(by_case) == 6 and set(by_case) == CASES, "Expected six unique step/route cases")
    if qualification.get("pass") is False:
        raise QualificationFailure("Native operator qualification failed")
    require(qualification.get("pass") is True, "Missing native qualification")
    for case in cases:
        require(case["rows"] == 2560, "Wrong correctness extent")
        for field in ("canaries_unchanged", "inputs_unchanged", "GPU_completed"):
            if case.get(field) is False:
                raise QualificationFailure("Native safety/completion failure: " + field)
            require(case.get(field) is True, "Missing safety/completion evidence: " + field)
        if any(case[k] != 0 for k in ("FP64_failures", "stock_bitwise_differences", "nonfinite")):
            raise QualificationFailure("Native correctness case has failures")
    with (root/"correctness.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    require(len(rows) == 6*2560, "Incomplete per-row correctness CSV")
    grouped = defaultdict(dict)
    for row in rows:
        key = (int(row["step"]), row["route"])
        index = int(row["row"])
        require(key in CASES and 0 <= index < 2560 and index not in grouped[key], "Duplicate/unexpected correctness row")
        reference = number(row, "reference")
        actual = float(row["actual"])
        if not math.isfinite(actual):
            raise QualificationFailure("Nonfinite actual output")
        error = abs(actual-reference)
        tolerance = 2e-4 + 2e-5*abs(reference)
        require(math.isclose(error, number(row, "absolute_error"), rel_tol=1e-12, abs_tol=1e-14), "Stored row error differs")
        require(math.isclose(tolerance, number(row, "tolerance"), rel_tol=1e-12, abs_tol=1e-14), "Frozen tolerance differs")
        failed = int(row["failed"])
        different = int(row["stock_bitwise_different"])
        require(failed in (0, 1) and different in (0, 1) and failed == int(error > tolerance), "Invalid row verdict")
        if failed or different:
            raise QualificationFailure("Per-row FP64/stock comparison failed")
        grouped[key][index] = (reference, actual, error)
    result = []
    for key in sorted(CASES):
        values = grouped[key]
        require(set(values) == set(range(2560)), "Missing output rows")
        reference = [values[i][0] for i in range(2560)]
        errors = [values[i][2] for i in range(2560)]
        maximum = max(errors)
        relative_l2 = math.sqrt(math.fsum(v*v for v in errors)/math.fsum(v*v for v in reference))
        native = by_case[key]
        require(math.isclose(maximum, native["max_abs"], rel_tol=1e-12, abs_tol=1e-14) and
                math.isclose(relative_l2, native["relative_L2"], rel_tol=1e-10, abs_tol=1e-16), "Native/CSV correctness summaries differ")
        result.append({"step": key[0], "route": key[1], "rows": 2560, "FP64_failures": 0,
                       "nonfinite": 0, "stock_bitwise_differences": 0, "max_abs": maximum,
                       "relative_L2": relative_l2, "canaries_unchanged": True, "inputs_unchanged": True,
                       "GPU_completed": True})
    for step in (1, 16, 32):
        for index in range(2560):
            stock, candidate = grouped[(step, "S")][index], grouped[(step, "Q4")][index]
            require(stock[0] == candidate[0], "References differ between routes")
            if struct.pack("<f", stock[1]) != struct.pack("<f", candidate[1]):
                raise QualificationFailure("Recomputed stock/Q4 F32 bits differ")
    return result


def audit(root):
    native = json.loads((root/"SUMMARY.json").read_text())
    qualification = json.loads((root/"qualification.json").read_text())
    request = json.loads((root/"REQUEST.json").read_text())
    require(request["model_sha256"] == MODEL_SHA256 and request["entry"] == "tq2_0_matvec_stock4x1_v1" and
            request["rows"] == 4 and request["groups"] == 1 and
            [r["step"] for r in request["cases"]] == [1, 16, 32], "Unexpected model/kernel/fixture scope")
    require(qualification["config_sha256"] == native["source_config_sha256"] == sha(root/"REQUEST.json"), "Request identity differs")
    if native.get("qualification_pass") is False:
        raise QualificationFailure("Native operator qualification failed")
    require(native["execution_pass"] is True and native.get("qualification_pass") is True and native["device"] == "Apple M2 Pro" and
            native["stock_entry"] == "kernel_mul_mv_tq2_0_f32" and native["Q4_entry"] == request["entry"] and
            native["stock_grid"] == [320, 1, 1] and native["stock_threads"] == [32, 2, 1] and
            native["Q4_grid"] == [640, 1, 1] and native["Q4_threads"] == [32, 1, 1], "Native execution/pipeline/launch mismatch")
    require(native["warmups_per_path"] == 30 and native["rounds"] == 3 and native["pairs_per_contrast_round"] == 201 and
            native["single_dispatch_per_command_buffer"] is True and native["capture_diagnostics_readback_during_timing"] is False,
            "Changed timing protocol")
    require(native["historical_official_strict_exit_preserved"] == 1 and native["model_quality_pass"] is None and
            native["official_source_model_output_pass"] is None, "Operator result promoted to model acceptance")
    cases = correctness(root, qualification)
    with (root/"samples.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    schedule = [(round, contrast, pair) for round in (1, 2, 3)
                for contrast in (("AA", "SQ4") if round % 2 else ("SQ4", "AA")) for pair in range(201)]
    require(len(rows) == 1206 and [(int(r["round"]), r["contrast"], int(r["pair"])) for r in rows] == schedule,
            "Missing, duplicated, reordered or unexpected timing pairs")
    for row in rows:
        round, pair = int(row["round"]), int(row["pair"])
        require((row["a_route"], row["b_route"]) == (("S", "S") if row["contrast"] == "AA" else ("S", "Q4")), "Wrong paired routes")
        ab = (pair % 4 in (0, 3)) != (round % 2 == 0)
        require(row["order"] == ("AB" if ab else "BA"), "AB/BA order differs")
        a, b = number(row, "a_us"), number(row, "b_us")
        require(a > 0 and b > 0 and math.isclose(number(row, "b_minus_a_us"), b-a, rel_tol=1e-12, abs_tol=1e-10), "Invalid paired duration/difference")
    original = {(g["round"], g["contrast"]): g for g in native["groups"]}
    require(len(original) == len(native["groups"]) == 6, "Duplicate/missing native timing groups")
    groups = []
    for round, contrast in ((r, c) for r in (1, 2, 3) for c in ("AA", "SQ4")):
        selected = [row for row in rows if int(row["round"]) == round and row["contrast"] == contrast]
        value = stats(selected)
        require((round, contrast) in original, "Missing native comparison group")
        for field in ("a_p50_us", "a_p95_us", "b_p50_us", "b_p95_us", "paired_B_minus_A_median_us", "B_wins"):
            require(math.isclose(value[field], original[(round, contrast)][field], rel_tol=1e-10, abs_tol=1e-9), "Raw/native timing summary differs: " + field)
        groups.append({"round": round, "contrast": contrast, **value,
                       "order_groups": {order: stats([row for row in selected if row["order"] == order]) for order in ("AB", "BA")},
                       "time_blocks": [{"first_pair": first, "last_pair": first+66, **stats(selected[first:first+67])} for first in (0, 67, 134)],
                       "native_state_before": original[(round, contrast)]["before"],
                       "native_state_after": original[(round, contrast)]["after"]})
    return {"analysis_completed": True, "measurement_integrity_pass": True, "operator_qualification_pass": True,
            "same_input_FP64_pass": True, "stock_bitwise_pass": True, "strict_operator_exit_code": 0,
            "device": native["device"], "model_sha256": MODEL_SHA256, "request_sha256": sha(root/"REQUEST.json"),
            "cases": cases, "correctness_rows_checked": 15360, "timing_pairs": 1206, "timing_groups": groups,
            "warmups_per_path": 30, "metric": "Single-dispatch GPU command-buffer elapsed microseconds after completion",
            "scope": "Existing step16 Qcur-0 isolated timing; excludes guard/cache, ActQuant preparation and full-model scheduling",
            "time_block_note": "Three consecutive 67-pair sequence blocks per comparison; CSV has order but no absolute host timestamps",
            "local_three_round_negative_paired_medians_observed": all(g["paired_B_minus_A_median_us"] < 0 for g in groups if g["contrast"] == "SQ4"),
            "raw_sha256": {f: sha(root/f) for f in ("REQUEST.json", "SUMMARY.json", "qualification.json", "correctness.csv", "samples.csv")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Output must be fresh")
    args.output.mkdir(parents=True)
    result = {"schema": 1, "analysis_completed": False, "measurement_integrity_pass": None,
              "operator_qualification_pass": None, "same_input_FP64_pass": None, "stock_bitwise_pass": None,
              "GPU_executed_by_analyzer": False, "model_quality_pass": None, "official_source_model_output_pass": None,
              "full_model_FP64_pass": None, "full_model_speedup_established": None,
              "historical_official_strict_exit_preserved": 1}
    try:
        result.update(audit(args.run.resolve()))
        code = 0
    except QualificationFailure as error:
        result.update(error=str(error), operator_qualification_pass=False, strict_operator_exit_code=1)
        code = 1
    except (EvidenceError, OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        result.update(error=str(error), strict_operator_exit_code=2)
        code = 2
    result["exit_code"] = code
    (args.output/"SUMMARY.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("cases", "timing_groups", "raw_sha256")},indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
