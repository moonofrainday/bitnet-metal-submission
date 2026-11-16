#!/usr/bin/env python3
"""Recompute CPU/command-buffer diagnostics, without inference or device calls.

All CPU durations are elapsed diagnostic regions, including possible preemption
and waits. Nested/concurrent regions are nonadditive. GPU timestamps have a
separate clock domain; this script never subtracts GPU from host timestamps.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    require(values and all(math.isfinite(v) for v in values), "Missing/nonfinite distribution")
    ordered = sorted(values)
    def percentile(p):
        x = (len(ordered) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return ordered[lo] + (x - lo) * (ordered[hi] - ordered[lo])
    return {"count": len(values), "min": ordered[0], "p50": statistics.median(values),
            "p95": percentile(.95), "max": ordered[-1], "sum": sum(values)}


def intervals(rows):
    require(rows and all(math.isfinite(a) and math.isfinite(b) and a <= b for a, b in rows),
            "Missing/invalid interval")
    ordered = sorted(rows)
    start, end = ordered[0]
    merged = []
    for a, b in ordered[1:]:
        if a > end:
            merged.append((start, end))
            start, end = a, b
        else:
            end = max(end, b)
    merged.append((start, end))
    total = sum(b-a for a, b in rows)
    union = sum(b-a for a, b in merged)
    events = sorted((point, delta) for a, b in rows if a < b for point, delta in ((a, 1), (b, -1)))
    active, maximum = 0, 0
    for _, delta in events:
        active += delta
        maximum = max(maximum, active)
    span = max(b for _, b in rows) - min(a for a, _ in rows)
    return {"intervals": len(rows), "summed_elapsed": total, "union_elapsed": union,
            "first_start_to_last_end": span, "uncovered_inside_span": span-union,
            "summed_overlap": total-union, "maximum_concurrent_intervals": maximum}


TASK_FIELDS = ("native_encode_calls", "native_encode_ns", "matvec_calls", "q_identity_calls",
               "outer_eligible", "guard_attempts", "guard_passes", "guard_rejects",
               "custom_dispatches", "stock_target_dispatches", "guard_ns",
               "guard_tensor_buffer_ns", "getter_ns", "native_target_encode_ns")


def analyze_path(path, identity):
    raw = json.loads(path.read_text())
    require(raw["schema"] == 1 and raw["complete"] is True and raw["clock"] == "CLOCK_UPTIME_RAW" and
            raw["units"] == "nanoseconds" and raw["dropped_records"] == raw["command_buffer_drops"] ==
            raw["pending_callback_handlers"] == 0 and raw["callback_wait_timed_out"] is False,
            "Incomplete/native diagnostic identity")
    route = identity["route"]
    tasks, calls, buffers = raw["tasks"], raw["q_calls"], raw["command_buffers"]
    require(tasks and len(calls) == 32 and [v["step"] for v in calls] == list(range(1, 33)), "Missing Q call history")
    require(all(t["mode"] == route and 1 <= t["step"] <= 32 for t in tasks), "Task route mismatch")
    totals = {key: sum(t[key] for t in tasks) for key in TASK_FIELDS}
    expected = 32 if route != "S" else 0
    custom = 32 if route == "Q4" else 0
    require(totals["q_identity_calls"] == 960 and totals["outer_eligible"] == 32 and
            totals["guard_attempts"] == totals["guard_passes"] == expected and totals["guard_rejects"] == 0 and
            totals["custom_dispatches"] == custom and totals["stock_target_dispatches"] == 32-custom,
            "Missing actual guard/dispatch coverage")
    task_keys = Counter((t["step"], t["command"], t["graph"], t["slice_start"], t["slice_end"]) for t in tasks)
    buffer_keys = Counter((b["step"], b["command"], b["graph"], b["slice_start"], b["slice_end"]) for b in buffers)
    require(task_keys == buffer_keys and all(v == 1 for v in task_keys.values()), "Encoder/CB join missing or duplicated")
    call_stats = {key.removesuffix("_ns"): distribution([v[key] / 1000 for v in calls])
                  for key in ("tensor_buffer_ns", "getter_ns", "capability_ns")}
    call_stats["whole_target_MV_function"] = distribution([(v["end_ns"]-v["start_ns"]) / 1000 for v in calls])
    cache = {key.removesuffix("_ns"): distribution([v["cache"][key] / 1000 for v in calls])
             for key in ("lock_wait_ns", "snapshot_key_ns", "lookup_ns", "load_ns", "capability_ns")}
    cache_counts = {key: sum(v["cache"][key] for v in calls)
                    for key in ("hits", "misses", "invalid_paths", "stable_load_failures")}
    per_token = []
    for step in range(1, 33):
        chosen = [b for b in buffers if b["step"] == step]
        require(chosen and all(b["completed"] is True and b["status"] == 4 for b in chosen), "Incomplete CB completion")
        host = intervals([(b["encode_start_ns"], b["encode_end_ns"]) for b in chosen])
        gpu = intervals([(b["gpu_start_s"], b["gpu_end_s"]) for b in chosen])
        require(all(b["gpu_start_s"] > 0 for b in chosen), "GPU timestamps unavailable")
        per_token.append({"step": step, "host_encode_ns": host, "GPU_seconds": gpu})
    family = {}
    for task in tasks:
        for record in task["operator_family_cpu"]:
            item = family.setdefault(record["family"], {"encode_calls": 0, "elapsed_worker_ns": 0})
            item["encode_calls"] += record["encode_calls"]
            item["elapsed_worker_ns"] += record["encode_ns"]
    return {"route": route, "round": identity["round"], "contrast": identity["contrast"],
            "pair": identity["pair"], "side": identity["side"], "raw_sha256": sha(path),
            "tasks": len(tasks), "command_buffers": len(buffers), "all_tasks_joined_to_one_natural_CB": True,
            "raw_task_sums_ns_or_counts": totals, "per_call_elapsed_us": call_stats,
            "per_call_cache_elapsed_us": cache, "cache_counts": cache_counts,
            "clock_calibration": raw["calibration"], "per_token_intervals": per_token,
            "CB_GPU_elapsed_us": distribution([(b["gpu_end_s"]-b["gpu_start_s"]) * 1e6 for b in buffers]),
            "operator_family_CPU_work": family,
            "host_token_encode_union_us": distribution([v["host_encode_ns"]["union_elapsed"] / 1000 for v in per_token]),
            "host_token_encode_overlap_us": distribution([v["host_encode_ns"]["summed_overlap"] / 1000 for v in per_token]),
            "GPU_token_union_us": distribution([v["GPU_seconds"]["union_elapsed"] * 1e6 for v in per_token]),
            "GPU_token_uncovered_us": distribution([v["GPU_seconds"]["uncovered_inside_span"] * 1e6 for v in per_token]),
            "GPU_token_overlap_us": distribution([v["GPU_seconds"]["summed_overlap"] * 1e6 for v in per_token])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("--output must be fresh")
    args.output.mkdir(parents=True)
    result = {"schema": 1, "analysis_completed": False, "GPU_executed_by_analyzer": False,
              "model_output_acceptance": None, "formal_performance_conclusion": None,
              "scope": "Selected Qcur-0 CPU/CB diagnostic evidence only; no new kernel selection",
              "caveats": ["CPU regions include waits/preemption and diagnostic overhead, not pure CPU cycles.",
                          "Nested/concurrent region sums do not decompose model wall time.",
                          "CPU and GPU timestamps use separate domains; no cross-domain subtraction.",
                          "Operator-family CPU encode totals are not GPU timing or critical-path shares.",
                          "A small diagnostic prior cannot replace formal OFF/ON and matched wall campaigns."]}
    try:
        root = args.run.resolve()
        native = json.loads((root / "RUN.json").read_text())
        require(native["operation"] == "attribution" and native["execution_pass"] is True and
                native["activation_quantization"] is True and native["one_model_and_context"] is True,
                "Expected completed ActQuant-ON same-context native archive")
        paths = []
        for identity in native["paths"]:
            if identity["diagnostic_file"] is not None:
                file = (root / identity["diagnostic_file"]).resolve()
                require(file.is_relative_to(root), "Diagnostic path escapes archive")
                paths.append(analyze_path(file, identity))
        require(paths, "No diagnostic records")
        result.update(analysis_completed=True,device=native["device"],attribution_kind=native["attribution_kind"],
                      model_sha256=native["model_sha256"],tokens_sha256=native["tokens_sha256"],
                      native_RUN_sha256=sha(root / "RUN.json"),paths=paths,exit_code=0)
        code = 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(error=str(error),exit_code=2)
        code = 2
    (args.output / "SUMMARY.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="paths"},indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
