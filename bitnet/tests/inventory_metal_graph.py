#!/usr/bin/env python3
"""Inventory observed Metal graph work without assigning per-op GPU times.

This CPU-only tool reads saved diagnostics. Node visits, encoded dispatch traces,
pipeline compilation, and command-buffer completion are separate observations.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


SOURCE_ROOT = Path(__file__).resolve().parents[2]
METADATA_OPS = {"NONE", "RESHAPE", "VIEW", "PERMUTE", "TRANSPOSE"}
RELEVANT_OPS = {
    "ADD", "MUL", "SQR", "RMS_NORM", "MUL_MAT", "CPY", "CONT",
    "RESHAPE", "VIEW", "PERMUTE", "TRANSPOSE", "GET_ROWS", "SET_ROWS",
    "SOFT_MAX", "ROPE", "FLASH_ATTN_EXT", "UNARY", "ACT_QUANT",
}
EXPECTED_FAMILIES = {
    "output_head", "attention_projection", "attention_core", "attention_rope",
    "ffn_projection", "ffn_elementwise", "normalization", "act_quant",
    "residual", "kv_store", "embedding_lookup", "memory_movement",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def enum_ids(header):
    text = Path(header).read_text()
    body = text.split("enum ggml_op {", 1)[1].split("};", 1)[0]
    body = re.sub(r"/\*.*?\*/|//[^\n]*", "", body, flags=re.S)
    ids = {}
    value = -1
    for entry in body.split(","):
        entry = entry.strip()
        if not entry:
            continue
        match = re.fullmatch(r"GGML_OP_(\w+)(?:\s*=\s*(\d+))?", entry)
        if not match:
            raise ValueError(f"Unsupported ggml_op enum entry: {entry!r}")
        value = int(match[2]) if match[2] else value + 1
        ids[match[1]] = value
    return ids


def classify(op, name, weight=""):
    """Return a family for this op, never from a tensor name alone."""
    op = op.removeprefix("GGML_OP_")
    if op in METADATA_OPS:
        return "metadata_noop"
    if op == "ACT_QUANT":
        return "act_quant"
    if op == "MUL_MAT":
        if name == "result_output" or weight == "output.weight":
            return "output_head"
        if re.search(r"\.attn_(q|k|v|output)\.weight$", weight) or re.fullmatch(r"(?:[QKV]cur|attn_out)-\d+", name):
            return "attention_projection"
        if re.search(r"\.ffn_(gate|up|down)\.weight$", weight) or re.fullmatch(r"ffn_(?:gate|up|down)-\d+", name):
            return "ffn_projection"
        if re.fullmatch(r"(?:kq|kqv)(?:_mla)?-\d+", name):
            return "attention_core"
        return "other_matmul"
    if op == "FLASH_ATTN_EXT" or op == "SOFT_MAX":
        return "attention_core"
    if op == "ROPE":
        return "attention_rope"
    if op == "RMS_NORM" or (op == "MUL" and re.fullmatch(r"(?:(?:attn|ffn)(?:_sub)?_norm-\d+|result_norm)", name)):
        return "normalization"
    if op in {"UNARY", "SQR", "MUL"} and re.fullmatch(r"ffn_(?:relu|sqr\(relu\)|sub_out|gate_par)-\d+", name):
        return "ffn_elementwise"
    if op == "ADD" and re.fullmatch(r"(?:ffn_inp|l_out)-\d+", name):
        return "residual"
    if op in {"SET_ROWS", "CPY"} and re.search(r"cache_[kv]|(?:^|_)kv(?:_|$)", name):
        return "kv_store"
    if op == "GET_ROWS" and (name == "inp_embd" or name == "node_1"):
        return "embedding_lookup"
    if op in {"CPY", "CONT", "DUP", "GET_ROWS", "SET_ROWS"}:
        return "memory_movement"
    return "other"


def fields(text):
    return dict(re.findall(r"(\w+)=([^\s]+)", text))


def natural_command_buffers(path):
    """Read native attribution diagnostics; GPU intervals keep their own clock."""
    data = json.loads(Path(path).read_text())
    if data.get("schema") != 1 or data.get("clock") != "CLOCK_UPTIME_RAW":
        raise ValueError("Unsupported natural-command-buffer diagnostic schema")
    tasks = data.get("tasks", [])
    records = data.get("command_buffers", [])
    cpu = defaultdict(Counter)
    for task in tasks:
        for family in task.get("operator_family_cpu", []):
            cpu[task["mode"]][(family["family"], "encode_calls")] += family["encode_calls"]
            cpu[task["mode"]][(family["family"], "encode_ns")] += family["encode_ns"]
    intervals = defaultdict(list)
    invalid = []
    seen = set()
    for i, record in enumerate(records):
        start, end = record.get("gpu_start_s"), record.get("gpu_end_s")
        identity = tuple(record.get(k) for k in ["command", "graph", "step", "slice_start", "slice_end", "encode_start_ns"])
        if identity in seen:
            invalid.append({"record_index": i, "reason": "duplicate_natural_CB_record"})
            continue
        seen.add(identity)
        if not record.get("completed") or record.get("status") != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in [start, end]) or not 0 < start <= end:
            invalid.append({"record_index": i, "reason": "incomplete_or_invalid_GPU_interval"})
            continue
        intervals[record["step"]].append((start, end))
    gpu = []
    for step, values in sorted(intervals.items()):
        values.sort()
        start, end = values[0]
        union = 0.0
        for a, b in values[1:]:
            if a > end:
                union += end - start
                start, end = a, b
            else:
                end = max(end, b)
        union += end - start
        span = max(b for _, b in values) - values[0][0]
        gpu.append({"step": step, "natural_CB_records": len(values), "CB_interval_union_us": union * 1e6, "first_start_to_last_end_us": span * 1e6, "gaps_outside_all_CB_intervals_us": max(0.0, span - union) * 1e6})
    return {
        "input": {"file_name": Path(path).name, "sha256": sha256(path)},
        "native_complete": data.get("complete"),
        "native_dropped_records": data.get("dropped_records"),
        "native_CB_drops": data.get("command_buffer_drops"),
        "invalid_records": invalid,
        "all_observed_CB_intervals_valid": bool(records) and not invalid,
        "CPU_worker_family_intervals": [
            {"mode": mode, "family": family, "encode_calls": counter[(family, "encode_calls")], "encode_ns": counter[(family, "encode_ns")]}
            for mode, counter in sorted(cpu.items())
            for family in sorted({f for f, _ in counter})
        ],
        "GPU_command_buffer_intervals_by_step": gpu,
        "interpretation": [
            "CPU worker intervals are diagnostic work, not model critical-path shares; concurrent or nested intervals are not wall-clock additive.",
            "CB interval union describes elapsed CB execution envelopes, not measured GPU busy cycles or per-family GPU latency.",
            "Gaps outside CB intervals do not independently identify a CPU cause; submission, scheduling and other device work remain possibilities.",
            "CLOCK_UPTIME_RAW CPU timestamps are not subtracted from Metal GPU timestamp seconds; cross-domain queue latency is unavailable.",
            "Per-operator GPU timing remains unavailable even when each natural CB has family counts.",
        ],
    }


def inventory(path, header):
    ids = enum_ids(header)
    counts = defaultdict(Counter)
    phase_counts = defaultdict(Counter)
    pipelines = set()
    completions = Counter()
    phase_serial = 0
    phase = "unscoped"
    phase_metadata = {}
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        if "ACTQUANT_Q_PHASE_BEGIN " in line:
            phase_serial += 1
            f = fields(line)
            phase = f"{phase_serial}:{f.get('phase', 'unknown')}:{f.get('decode_step', '?')}"
            phase_metadata[phase] = {k: f[k] for k in ["phase", "decode_step", "input_position", "count", "q_route"] if k in f}
        match = re.search(r"node\[\s*(\d+)\]\s*-\s*([A-Z][A-Z0-9_]*)\s+(.+)$", line)
        if match:
            op = match[2]
            name = match[3].split()[0]
            family = "metadata_noop" if name == "(noop)" else classify(op, name)
            counts["graph_debug_node_visits"][(family, op)] += 1
            phase_counts[("graph_debug_node_visits", phase)][family] += 1
        if "ACT_QUANT_ENCODE " in line:
            f = fields(line)
            counts["targeted_encode_events"][("act_quant", "ACT_QUANT")] += 1
            phase_counts[("targeted_encode_events", phase)]["act_quant"] += 1
            if f.get("kernel"):
                pipelines.add(f["kernel"])
        elif "BITNET_MULTILAYER_ENCODE " in line:
            f = fields(line)
            name = f.get("node", f.get("output", ""))
            family = classify("MUL_MAT", name, f.get("weight", ""))
            counts["targeted_encode_events"][(family, "MUL_MAT")] += 1
            phase_counts[("targeted_encode_events", phase)][family] += 1
            if f.get("pipeline"):
                pipelines.add(f["pipeline"])
        if "compiling pipeline: " in line:
            match = re.search(r"name = '([^']+)'", line)
            if match:
                pipelines.add(match[1])
        for tag in ["BITNET_MULTILAYER_COMPLETE", "ACT_QUANT_COMPLETE", "FFN_ABLATION_GPU"]:
            if tag + " " in line:
                f = fields(line)
                completions[(tag, f.get("completed", "unknown"))] += 1

    observed = {family for counter in counts.values() for family, _ in counter}
    has_debug = bool(counts.get("graph_debug_node_visits"))
    has_targeted = bool(counts.get("targeted_encode_events"))
    return {
        "schema_version": 1,
        "input": {"file_name": Path(path).name, "sha256": sha256(path)},
        "ggml_header_sha256": sha256(header),
        "current_source_op_ids": {k: ids[k] for k in sorted(RELEVANT_OPS)},
        "coverage": {
            "kind": "graph_debug_observations" if has_debug else "targeted_partial" if has_targeted else "no_supported_observations",
            "complete_graph_confirmed": False,
            "unobserved_families_are_not_zero_work": True,
            "unobserved_families": sorted(EXPECTED_FAMILIES - observed),
            "per_operator_GPU_time": "unavailable",
            "critical_path_family_shares": "unavailable",
            "hardware_counters": "not_collected_by_this_tool",
        },
        "observations_by_origin": {
            origin: {
                "count": sum(counter.values()),
                "counts": [{"family": family, "op": op, "op_id": ids.get(op), "count": count} for (family, op), count in sorted(counter.items())],
            }
            for origin, counter in sorted(counts.items())
        },
        "phase_observations": [
            {"origin": origin, "scope": phase, "metadata": phase_metadata.get(phase), "families": dict(sorted(counter.items()))}
            for (origin, phase), counter in sorted(phase_counts.items())
        ],
        "pipeline_names_seen_in_compile_or_targeted_logs": sorted(pipelines),
        "completion_events_not_kernel_durations": [
            {"tag": tag, "completed": completed, "count": count} for (tag, completed), count in sorted(completions.items())
        ],
        "interpretation": [
            "Origins overlap; do not add node visits to targeted encode events.",
            "Graph node visits are not dispatch counts: fusion consumes nodes and metadata ops are noops.",
            "Pipeline compilation is not evidence of dispatch frequency or GPU elapsed time.",
            "Targeted traces cover selected Q/FFN/ActQuant nodes, not all model work.",
            "Completion events may repeat the same enclosing command buffer; never sum them as per-op GPU time.",
            "A command-buffer interval containing mixed families cannot be divided into per-family GPU times from node counts.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--diagnostics", type=Path, help="Saved schema-1 Q-attribution JSON; no native execution")
    parser.add_argument("--header", type=Path, default=SOURCE_ROOT / "ggml/include/ggml.h")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.log and not args.diagnostics:
        parser.error("provide --log and/or --diagnostics")
    result = inventory(args.log, args.header) if args.log else {"schema_version": 1}
    if args.diagnostics:
        result["natural_command_buffer_diagnostics"] = natural_command_buffers(args.diagnostics)
    if args.output.exists():
        parser.error("--output must be a fresh path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "coverage": result.get("coverage", {}).get("kind", "natural_CB_diagnostics_only"), "origins": {k: v["count"] for k, v in result.get("observations_by_origin", {}).items()}}))


if __name__ == "__main__":
    main()
