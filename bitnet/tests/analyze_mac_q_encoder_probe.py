#!/usr/bin/env python3
"""CPU-only validation of the perturbative selected-Q encoder timestamp probe.

Exit 0: complete structural/timestamp evidence; 1: observed contract/check failure;
2: missing/malformed evidence or CLI/I/O failure. This is NOT a numerical/model
acceptance test and never establishes a normal-schedule critical-path fraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


class EvidenceError(Exception):
    pass


def integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvidenceError(f"{label}: expected integer >= {minimum}")
    return value


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise EvidenceError(f"{label}: expected finite number")
    return float(value)


def percentile(values, q):
    s = sorted(values)
    pos = (len(s) - 1) * q
    i = int(pos)
    return s[i] + (s[min(i + 1, len(s) - 1)] - s[i]) * (pos - i)


def abi(data):
    if not isinstance(data, str):
        raise EvidenceError("args_hex is not a string")
    try:
        raw = bytes.fromhex(data)
    except ValueError as exc:
        raise EvidenceError("malformed args_hex") from exc
    if len(raw) != 120:
        raise EvidenceError(f"stock MUL_MV ABI requires 120 bytes, got {len(raw)}")
    fields = {}
    for name, offset, code in [
        ("ne00", 0, "i"), ("ne01", 4, "i"), ("ne02", 8, "i"),
        ("nb00", 16, "Q"), ("nb01", 24, "Q"), ("nb02", 32, "Q"), ("nb03", 40, "Q"),
        ("ne10", 48, "i"), ("ne11", 52, "i"), ("ne12", 56, "i"),
        ("nb10", 64, "Q"), ("nb11", 72, "Q"), ("nb12", 80, "Q"), ("nb13", 88, "Q"),
        ("ne0", 96, "i"), ("ne1", 100, "i"), ("nr0", 104, "i"),
        ("r2", 108, "h"), ("r3", 110, "h"), ("eps", 112, "f"),
    ]:
        fields[name] = struct.unpack_from("<" + code, raw, offset)[0]
    return fields


def analyze(doc, mode, expected_cbs=None):
    checks = []
    def check(name, condition, detail=None):
        checks.append({"check": name, "pass": bool(condition), **({"detail": detail} if detail is not None else {})})

    if not isinstance(doc, dict):
        raise EvidenceError("diagnostic root must be an object")
    pool = doc.get("selected_q_encoder_probe")
    if not isinstance(pool, dict):
        raise EvidenceError("missing selected_q_encoder_probe object")
    records = pool.get("records")
    cbs = doc.get("command_buffers")
    if not isinstance(records, list) or not isinstance(cbs, list):
        raise EvidenceError("missing records or command_buffers list")
    check("diagnostic_complete", doc.get("complete") is True)
    check("diagnostic_no_drops", all(doc.get(k) == 0 for k in ("dropped_records", "command_buffer_drops", "pending_callback_handlers")) and doc.get("callback_wait_timed_out") is False)
    check("probe_complete", pool.get("enabled") is True and pool.get("mode") == mode and pool.get("ready") is True and pool.get("complete") is True and pool.get("error") == "none")
    check("probe_no_drops", pool.get("dropped_records") == 0)
    check("no_extra_CB_commit_CPU_wait", all(pool.get(k) == 0 for k in ("extra_command_buffers", "extra_commits", "extra_CPU_waits")))
    check("preallocated_counters_fences", all(pool.get(k) == 0 for k in ("counter_buffer_allocations_in_target_encode", "fence_allocations_in_target_encode")))
    check("clock_brackets_outside_wall", pool.get("clock_samples_taken_outside_decode_wall") is True)
    check("exact_32_records", len(records) == 32 and pool.get("records_count") == 32)
    if expected_cbs is not None:
        check("explicit_expected_CB_count", len(cbs) == expected_cbs, {"expected": expected_cbs, "observed": len(cbs)})
    check("all_recorded_CBs_completed", all(c.get("completed") is True and c.get("status") == 4 for c in cbs))
    for index, cb in enumerate(cbs):
        cbstart = finite(cb.get("gpu_start_s"), f"CB[{index}].gpu_start_s")
        cbend = finite(cb.get("gpu_end_s"), f"CB[{index}].gpu_end_s")
        check(f"CB_{index}:valid_GPU_interval", cbend >= cbstart > 0)

    cbefore = integer(pool.get("cpu_before"), "cpu_before", 1)
    cafter = integer(pool.get("cpu_after"), "cpu_after", 1)
    gbefore = integer(pool.get("gpu_before"), "gpu_before", 1)
    gafter = integer(pool.get("gpu_after"), "gpu_after", 1)
    check("valid_clock_bracket", cafter > cbefore and gafter > gbefore)
    if cafter <= cbefore or gafter <= gbefore:
        raise EvidenceError("non-positive clock bracket duration")
    ratio_ns = (cafter - cbefore) / (gafter - gbefore)
    steps = [integer(r.get("step"), "step", 1) for r in records]
    check("exact_steps_1_to_32", sorted(steps) == list(range(1, 33)))

    expected_weight = {"name": "blk.0.attn_q.weight", "type": "tq2_0", "ne": [2560, 2560, 1, 1], "nb": [66, 660, 1689600, 1689600], "bytes": 1689600}
    expected_input = {"name": "actq_qkv_out-0", "type": "f32", "ne": [2560, 1, 1, 1], "nb": [4, 10240, 10240, 10240], "bytes": 10240}
    expected_output = dict(expected_input, name="Qcur-0")
    expected_abi = {"ne00": 2560, "ne01": 2560, "ne02": 1, "nb00": 66, "nb01": 660, "nb02": 1689600, "nb03": 1689600,
                    "ne10": 2560, "ne11": 1, "ne12": 1, "nb10": 4, "nb11": 10240, "nb12": 10240, "nb13": 10240,
                    "ne0": 2560, "ne1": 1, "nr0": 4, "r2": 1, "r3": 1, "eps": 0.0}
    values, rows, all_indices = [], [], []
    pso_values = set()
    for index, rec in enumerate(records):
        p = f"step_{rec['step']}"
        check(p + ":identity", rec.get("node") == "Qcur-0" and rec.get("op") == "MUL_MAT" and rec.get("mode") == mode and rec.get("consumed_graph_nodes_required") == 1)
        bindings = {}
        for key, expected in (("weight", expected_weight), ("input", expected_input), ("output", expected_output)):
            t = rec.get(key)
            if not isinstance(t, dict) or not isinstance(t.get("binding"), dict):
                raise EvidenceError(f"{p}: missing {key} tensor/binding")
            check(p + ":" + key + "_contract", all(t.get(k) == v for k, v in expected.items()))
            b = t["binding"]
            buffer = integer(b.get("buffer"), f"{p}.{key}.buffer", 1)
            offset = integer(b.get("offset"), f"{p}.{key}.offset")
            length = integer(b.get("buffer_length"), f"{p}.{key}.buffer_length", 1)
            integer(b.get("hazard_tracking_mode"), f"{p}.{key}.hazard_tracking_mode")
            check(p + ":" + key + "_bounds", offset + expected["bytes"] <= length and offset % (2 if key == "weight" else 4) == 0)
            bindings[key] = b
        for a, b in (("weight", "input"), ("weight", "output"), ("input", "output")):
            aa, bb = bindings[a], bindings[b]
            asize, bsize = rec[a]["bytes"], rec[b]["bytes"]
            check(p + ":" + a + "_" + b + "_nonoverlap", aa["buffer"] != bb["buffer"] or aa["offset"] + asize <= bb["offset"] or bb["offset"] + bsize <= aa["offset"])
        planned_buffers = {"1": bindings["weight"], "2": bindings["input"], "3": bindings["output"], "4": bindings["output"], "5": bindings["input"]}
        check(p + ":planned_buffer_ABI", rec.get("planned_buffers") == planned_buffers)
        check(p + ":stock_pipeline", rec.get("planned_pipeline") == "kernel_mul_mv_tq2_0_f32")
        pso = integer(rec.get("planned_native_PSO"), p + ".planned_native_PSO", 1)
        pso_values.add(pso)
        integer(rec.get("planned_pipeline_wrapper"), p + ".planned_pipeline_wrapper", 1)
        for encoder in ("prefix_encoder", "selected_encoder", "suffix_encoder"):
            integer(rec.get(encoder), p + "." + encoder, 1)
        check(p + ":pipeline_parameters", [rec.get(k) for k in ("pipeline_nr0", "pipeline_nr1", "pipeline_nsg", "pipeline_execution_width")] == [4, 1, 2, 32] and integer(rec.get("pipeline_max_threads"), p + ".pipeline_max_threads", 1) >= 64)
        check(p + ":stock_geometry", rec.get("planned_grid") == [320, 1, 1] and rec.get("planned_threads") == [32, 2, 1])
        integer(rec.get("planned_smem"), p + ".planned_smem")
        check(p + ":stock_args_ABI", rec.get("planned_args_bytes") == 120 and abi(rec.get("planned_args_hex")) == expected_abi)
        check(p + ":ended_verified", rec.get("encoder_scope_ended") is True and rec.get("tuple_verified") is True)
        if mode == "target":
            check(p + ":one_actual_dispatch", rec.get("scope_dispatches") == 1)
            check(p + ":actual_tuple_matches", rec.get("actual_pipeline") == rec.get("planned_pipeline") and all(rec.get("actual_" + k) == rec.get("planned_" + k) for k in ("native_PSO", "args_hex", "args_bytes", "buffers", "grid", "threads", "smem")))
        else:
            check(p + ":empty_zero_dispatch", rec.get("scope_dispatches") == 0 and rec.get("actual_buffers") == {} and all("actual_" + k not in rec for k in ("native_PSO", "pipeline", "args_hex", "args_bytes", "grid", "threads", "smem")))
        start_i, end_i = integer(rec.get("sample_start_index"), p + ".sample_start_index"), integer(rec.get("sample_end_index"), p + ".sample_end_index")
        all_indices += [start_i, end_i]
        check(p + ":sample_indices", [start_i, end_i] == [2 * index, 2 * index + 1])
        graph, command = integer(rec.get("graph"), p + ".graph", 1), integer(rec.get("command"), p + ".command", 1)
        graph_index = integer(rec.get("graph_index"), p + ".graph_index")
        host = integer(rec.get("scope_host_ns"), p + ".scope_host_ns", 1)
        matches = [c for c in cbs if c.get("command") == command and c.get("graph") == graph and c.get("step") == rec["step"] and c.get("slice_start", -1) <= graph_index < c.get("slice_end", -1) and c.get("encode_start_ns", -1) <= host <= c.get("encode_end_ns", -1)]
        check(p + ":unique_natural_CB_join", len(matches) == 1 and rec.get("natural_CB_matches") == 1, {"independently_found": len(matches)})
        cb = matches[0] if len(matches) == 1 else {}
        check(p + ":GPU_completed", rec.get("GPU_command_completed") is True and cb.get("completed") is True and cb.get("status") == 4)
        start, end = integer(rec.get("raw_gpu_start"), p + ".raw_gpu_start", 1), integer(rec.get("raw_gpu_end"), p + ".raw_gpu_end", 1)
        check(p + ":valid_raw_interval", start != (1 << 64) - 1 and end != (1 << 64) - 1 and gbefore <= start <= end <= gafter and rec.get("valid") is True)
        if mode == "target":
            check(p + ":positive_target_duration", end > start)
        elapsed = (end - start) * ratio_ns / 1000
        reported = finite(rec.get("elapsed_us"), p + ".elapsed_us")
        check(p + ":calibration_recomputed", math.isclose(elapsed, reported, rel_tol=1e-12, abs_tol=1e-9))
        values.append(elapsed)
        rows.append({"step": rec["step"], "elapsed_us": elapsed, "raw_gpu_delta": end - start, "natural_CB_command": command, "natural_CB_graph": graph, "graph_index": graph_index, "CB_gpu_elapsed_us": (finite(cb["gpu_end_s"], p + ".CBgpu_end") - finite(cb["gpu_start_s"], p + ".CBgpu_start")) * 1e6 if cb else None})
    check("sample_indices_unique_exact_0_to_63", sorted(all_indices) == list(range(64)))
    check("single_stock_native_PSO_within_process", len(pso_values) == 1)
    passed = all(c["pass"] for c in checks)
    return {"schema": 1, "mode": mode, "measurement_integrity_pass": passed, "exit_code": 0 if passed else 1,
            "scope": "perturbative encoder interval including explicit fence wait/update and encoder boundaries; not normal-schedule critical-path share or pure Q kernel latency",
            "normal_schedule_Q_share": None, "fp64_pass": None, "model_output_pass": None,
            "requires_independent_numerical_probe_neutrality": True, "expected_CB_count_applied": expected_cbs,
            "command_buffers": len(cbs), "records": len(records), "stock_native_PSO_count": len(pso_values),
            "calibration": {"CPU_bracket_ns": cafter - cbefore, "raw_GPU_bracket_delta": gafter - gbefore, "ns_per_raw_GPU_tick": ratio_ns},
            "diagnostic_interval_us": {"p50": percentile(values, .5), "p95": percentile(values, .95), "min": min(values), "max": max(values), "sum": sum(values)} if values else None,
            "checks": checks, "failed_checks": [c for c in checks if not c["pass"]], "per_step": rows,
            "clock_rules": ["Host CLOCK_UPTIME_RAW values are used only for joins in that host clock.", "Raw counter deltas use their own MTLDevice sampleTimestamps bracket ratio.", "Metal command-buffer GPU seconds are reported separately; no absolute cross-clock comparison.", "Empty-control intervals are separate, never silently subtracted from target intervals."]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostics", type=Path, required=True)
    p.add_argument("--mode", choices=("target", "empty"), required=True)
    p.add_argument("--expected-cbs", type=int, help="Explicit normal diagnostic only; do not apply 64 to callback-capture runs")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    if a.output.exists():
        p.error("--output must be a new file; historical results are not overwritten")
    try:
        data = a.diagnostics.read_bytes()
        result = analyze(json.loads(data), a.mode, a.expected_cbs)
        result["input"] = {"filename": a.diagnostics.name, "sha256": hashlib.sha256(data).hexdigest()}
    except (OSError, ValueError, TypeError, KeyError, EvidenceError) as exc:
        result = {"schema": 1, "measurement_integrity_pass": None, "exit_code": 2, "error": f"{type(exc).__name__}: {exc}", "normal_schedule_Q_share": None, "fp64_pass": None, "model_output_pass": None}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("measurement_integrity_pass", "exit_code")}))
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
