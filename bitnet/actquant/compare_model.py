#!/usr/bin/env python3
"""Compare saved ActQuant-model runs with immutable-source CPU references.

No model or GPU is executed. All generated CSV and detailed JSON remain local.
Processing completion, local quantizer correctness and full-model acceptance
are separate results. The original stock failure records are never modified.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch

from analyze_op import ABS, REL, ulps
from prepare_fixtures import SOURCE_SHA, official_quantizer

MODEL_SHA = "52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2"
VOCAB = 128256
STAGES = {"qkv", "attn_out", "ffn_up_gate", "ffn_down"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw(path, count):
    if path.stat().st_size != count * 4:
        raise ValueError("Unexpected F32 byte extent: " + path.name)
    return np.memmap(path, mode="r", dtype="<f4", shape=(count,))


def compare(actual, reference):
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    b = np.asarray(reference, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError("Comparison extents differ")
    finite = np.isfinite(a) & np.isfinite(b)
    error = np.abs(a - b)
    bound = ABS + REL * np.abs(b)
    fail = ~finite | (error > bound)
    ref_l2 = float(np.dot(b, b)) if finite.all() else None
    sq = float(np.dot(a - b, a - b)) if finite.all() else None
    metrics = {"values": int(a.size), "nonfinite_actual": int(np.count_nonzero(~np.isfinite(a))),
               "nonfinite_reference": int(np.count_nonzero(~np.isfinite(b))),
               "frozen_tolerance_failures": int(np.count_nonzero(fail)),
               "bitwise_differences": int(np.count_nonzero(np.asarray(actual, np.float32).reshape(-1).view(np.uint32)
                                                            != np.asarray(reference, np.float32).reshape(-1).view(np.uint32))),
               "max_abs_error": float(error[finite].max()) if finite.any() else None,
               "max_error_over_tolerance": float((error[finite] / bound[finite]).max()) if finite.any() else None,
               "RMSE": math.sqrt(sq / a.size) if sq is not None else None,
               "relative_L2": math.sqrt(sq / ref_l2) if sq is not None and ref_l2 else None,
               "squared_error": sq, "reference_squared_L2": ref_l2,
               "max_value_ULP_distance": int(ulps(np.asarray(actual, np.float32).reshape(-1)[finite], np.asarray(reference, np.float32).reshape(-1)[finite]).max()) if finite.any() else None}
    return metrics, fail


def aggregate(rows):
    if not rows:
        return None
    sq = math.fsum(v["squared_error"] for v in rows) if all(v["squared_error"] is not None for v in rows) else None
    rsq = math.fsum(v["reference_squared_L2"] for v in rows) if all(v["reference_squared_L2"] is not None for v in rows) else None
    count = sum(v["values"] for v in rows)
    return {"vectors_or_captures": len(rows), "values": count,
            "nonfinite_actual": sum(v["nonfinite_actual"] for v in rows),
            "nonfinite_reference": sum(v["nonfinite_reference"] for v in rows),
            "frozen_tolerance_failures": sum(v["frozen_tolerance_failures"] for v in rows),
            "bitwise_differences": sum(v["bitwise_differences"] for v in rows),
            "max_abs_error": max((v["max_abs_error"] for v in rows if v["max_abs_error"] is not None), default=None),
            "max_error_over_tolerance": max((v["max_error_over_tolerance"] for v in rows if v["max_error_over_tolerance"] is not None), default=None),
            "RMSE": math.sqrt(sq / count) if sq is not None else None,
            "relative_L2": math.sqrt(sq / rsq) if sq is not None and rsq else None}


def passed(metrics):
    return metrics is not None and metrics["nonfinite_actual"] == 0 and metrics["nonfinite_reference"] == 0 and metrics["frozen_tolerance_failures"] == 0


def nll(logits, target):
    x = np.asarray(logits, dtype=np.float64)
    maximum = float(x.max())
    return maximum + math.log(float(np.exp(x - maximum).sum(dtype=np.float64))) - float(x[target])


def ranking(row):
    x = np.asarray(row, dtype=np.float64)
    ids = np.lexsort((np.arange(x.size), -x))[:10]
    return {"greedy_token_id": int(x.argmax()), "top1_top2_margin": float(x[ids[0]] - x[ids[1]]),
            "max_logit_ties": int(np.count_nonzero(x == x.max())),
            "tie_policy": "First smallest token ID at equal maximum",
            "computations_minus_graphics": float(x[83699] - x[14515]),
            "specific": {str(i): {"logit": float(x[i]), "rank_competition": int(np.count_nonzero(x > x[i])) + 1} for i in (83699, 14515)},
            "top10": [{"token_id": int(i), "logit": float(x[i])} for i in ids]}


def dump_failures(writer, scope, position, node, a, b, bad, folder, masks):
    actual = np.asarray(a, dtype=np.float64).reshape(-1)
    reference = np.asarray(b, dtype=np.float64).reshape(-1)
    bad = np.asarray(bad, dtype=bool).reshape(-1)
    failed = np.flatnonzero(bad)
    if not failed.size:
        return
    filename = f"mask-{len(masks):06d}.bits"
    path = folder / filename
    np.packbits(bad, bitorder="little").tofile(path)
    masks.append({"scope": scope, "input_position": position, "node": node, "elements": int(bad.size),
                  "failed_elements": int(failed.size), "file": "failure-masks/" + filename,
                  "sha256": sha(path), "packing": "numpy.packbits little-bit-order; ignore padding past elements"})
    ratios = np.abs(actual - reference) / (ABS + REL * np.abs(reference))
    worst = failed[np.argsort(ratios[failed])[-16:]]
    selected = np.unique(np.concatenate((failed[:16], worst)))
    for index in selected:
        v, r = actual[index], reference[index]
        columns = actual.size // len(position) if isinstance(position, list) and position else actual.size
        mapped_position = position[int(index // columns)] if isinstance(position, list) and position else position
        writer.writerow({"scope": scope, "input_position": mapped_position, "node": node,
                         "element": int(index % columns), "flat_element": int(index),
                         "reference": float(r), "actual": float(v) if math.isfinite(v) else str(v),
                         "absolute_error": abs(float(v - r)) if math.isfinite(v) and math.isfinite(r) else None,
                         "tolerance": ABS + REL * abs(float(r)),
                         "relative_error": abs(float(v - r)) / abs(float(r)) if math.isfinite(v) and r else None,
                         "reference_zero": bool(r == 0)})


def analyze(a):
    run_path = a.run / "RUN.json"
    run = json.loads(run_path.read_text())
    if run["schema"] != 1 or run["operation"] != "model" or run["model_sha256"] != MODEL_SHA:
        raise ValueError("Run schema, operation or public model identity differs")
    workload = run["workload"]
    if workload not in ("teacher", "corpus") or type(run["activation_quantization"]) is not bool:
        raise ValueError("Invalid workload or quantizer switch")
    dirs = {"F32": a.reference_f32, "BF16": a.reference_bf16}
    ref_tokens = {k: json.loads((v / "tokens.json").read_text()) for k, v in dirs.items()}
    reference_identity = {}
    for precision, directory in dirs.items():
        manifest = json.loads((directory / "MANIFEST.json").read_text())
        if (manifest["official_transformers_commit"] != "096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4"
                or manifest["model_revision"] != "04c3b9ad9361b824064a1f25ea60a8be9599b127"
                or manifest["checkpoint_sha256"] != "8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b"
                or manifest["source_sha256"]["official-bitnet-096f25ae.py"] != SOURCE_SHA
                or manifest["dtype"] not in (("torch.float32", "float32") if precision == "F32" else ("torch.bfloat16", "bfloat16"))):
            raise ValueError("Reference identity or precision differs")
        reference_identity[precision] = {"manifest_sha256": sha(directory / "MANIFEST.json"),
                                         "tokens_sha256": sha(directory / "tokens.json"),
                                         "checkpoint_sha256": manifest["checkpoint_sha256"],
                                         "tokenizer_sha256": manifest["tokenizer_sha256"]}
    token_ids = run["token_ids"]
    key = "teacher_forced" if workload == "teacher" else "corpus"
    for precision, tokens in ref_tokens.items():
        if tokens[key] != token_ids:
            raise ValueError(precision + " reference token IDs differ")
        if workload == "teacher" and (run["prompt_ids"] != tokens["prompt"] or run["continuation_ids"] != tokens["continuation"]):
            raise ValueError("Prompt/history token IDs differ")
    positions = list(range(26, 59)) if workload == "teacher" else list(range(128))
    if [v["input_position"] for v in run["logits"]] != positions:
        raise ValueError("Missing, repeated or misordered logit position")
    if len(token_ids) != (59 if workload == "teacher" else 129):
        raise ValueError("Unexpected exact-token workload length")
    reference_logits = {}
    for precision, directory in dirs.items():
        for mode in ("quant_on", "quant_off"):
            suffix = "teacher-forced" if workload == "teacher" else "corpus"
            rows = 33 if workload == "teacher" else 129
            reference_logits[precision + "_" + mode] = raw(directory / f"{mode}-{suffix}-logits.f32", rows * VOCAB).reshape(rows, VOCAB)
    a.output.mkdir(parents=True, exist_ok=False)
    masks_dir = a.output / "failure-masks"
    masks_dir.mkdir()
    masks = []
    logit_rows, score_rows, ranking_rows = [], [], []
    actual_nll = []
    reference_nlls = {key: [] for key in reference_logits}
    with (a.output / "failures-local.csv").open("w", newline="") as f:
        fields = ["scope", "input_position", "node", "element", "flat_element", "reference", "actual", "absolute_error", "tolerance", "relative_error", "reference_zero"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(run["logits"]):
            position = record["input_position"]
            if record["vocab"] != VOCAB:
                raise ValueError("Vocabulary differs")
            actual = raw(a.run / record["path"], VOCAB)
            target = token_ids[position + 1] if position + 1 < len(token_ids) else None
            if "target_token" in record and record["target_token"] != target:
                raise ValueError("Logit target does not follow input position")
            finite = bool(np.isfinite(actual).all())
            rankings = {"actual": ranking(actual) if finite else None}
            score = {"input_position": position, "target_token": target, "actual_NLL": nll(actual, target) if target is not None and finite else None}
            if score["actual_NLL"] is not None:
                actual_nll.append(score["actual_NLL"])
            for name, matrix in reference_logits.items():
                reference = matrix[index]
                m, bad = compare(actual, reference)
                m.update(comparison=name, input_position=position, greedy_equal=bool(actual.argmax() == reference.argmax()) if finite else None)
                logit_rows.append(m)
                rankings[name] = ranking(reference)
                if target is not None:
                    score[name + "_NLL"] = nll(reference, target)
                    reference_nlls[name].append(score[name + "_NLL"])
                if name == "F32_quant_on":
                    dump_failures(writer, "F32_ON_logit", position, "result_output", actual, reference, bad, masks_dir, masks)
            ranking_rows.append({"input_position": position, "rankings": rankings})
            score_rows.append(score)
        quantizer, tensors = analyze_checkpoints(a, run, dirs, writer, masks_dir, masks)
    summaries = {}
    for name in reference_logits:
        rows = [v for v in logit_rows if v["comparison"] == name]
        summaries[name] = dict(aggregate(rows), greedy_equal_positions=sum(v["greedy_equal"] is True for v in rows))
    expected_targets = 128 if workload == "corpus" else 32
    score_summary = {"targets": len(actual_nll), "expected_targets": expected_targets,
                     "all_target_scores_available": len(actual_nll) == expected_targets,
                     "scope": "Whole 128-token test-split PPL smoke" if workload == "corpus" else "Fixed-continuation teacher-forced NLL/PPL, not held-out quality",
                     "actual_mean_NLL": math.fsum(actual_nll) / expected_targets if len(actual_nll) == expected_targets else None,
                     "actual_PPL": math.exp(math.fsum(actual_nll) / expected_targets) if len(actual_nll) == expected_targets else None,
                     "references": {k: {"mean_NLL": math.fsum(v) / len(v), "PPL": math.exp(math.fsum(v) / len(v))} for k, v in reference_nlls.items() if v}}
    expected_calls = 128 if workload == "corpus" else (59 if run["prefill"] == "sequential" else 33)
    if run.get("completed_decode_calls") != expected_calls:
        raise ValueError("Completed decode-call count differs from the declared execution path")
    quant_complete = quantizer_coverage(run, expected_calls)
    local_quant_pass = (all(v["numerical_pass"] for v in quantizer) if quantizer else None)
    if run["activation_quantization"] and not quant_complete and local_quant_pass is True:
        local_quant_pass = None
    primary_layers = [v for v in tensors if v["comparison"] == "F32_quant_on" and v["kind"] == "l_out"]
    layer_complete = layer_coverage(run)
    layer_summary = aggregate(primary_layers)
    logit_pass = passed(summaries["F32_quant_on"])
    if not logit_pass or (layer_summary is not None and not passed(layer_summary)):
        f32_output_pass = False
    else:
        f32_output_pass = passed(layer_summary) if layer_complete else None
    observed_fail = (local_quant_pass is False or f32_output_pass is False)
    strict = 1 if observed_fail else 0 if local_quant_pass is True and f32_output_pass is True else 2
    result = {"schema": 1, "analysis_completed": True, "model_or_GPU_executed_by_analyzer": False,
              "run_manifest_sha256": sha(run_path), "model_sha256": MODEL_SHA,
              "references": reference_identity,
              "workload": workload, "backend": run["backend"], "activation_quantization": run["activation_quantization"],
              "settings": {k: run[k] for k in ("prefill", "kv", "settings") if k in run},
              "exact_token_ID_alignment_pass": True, "frozen_tolerance": {"absolute": ABS, "relative": REL},
              "quantizer_captures": len(quantizer), "quantizer_capture_coverage_pass": quant_complete,
              "same_input_official_F32_ActQuant_pass": local_quant_pass,
              "quantizer_summary": aggregate([v["value_metrics"] for v in quantizer]),
              "recovered_integer_code_mismatches": sum(v["integer_code_mismatches"] for v in quantizer),
              "full_layer_capture_coverage_pass": layer_complete, "F32_ON_layer_output_summary": layer_summary,
              "logits": summaries, "scores": score_summary,
              "full_logits_F32_ON_frozen_pass": logit_pass,
              "model_output_pass": f32_output_pass, "model_output_reference": "F32 official-source ON, all declared l_out and logits; frozen tolerance",
              "fp64_pass": None, "model_quality_pass": None,
              "strict_scoped_model_exit_code": strict,
              "legacy_global_V1_exit_code": 2,
              "scope_limits": ["Official local ActQuant output is a F32 semantic oracle, not an FP64 full-model result.",
                               "CPU stock dot may repack activation to Q8_K; Metal/no-cache/source matmul and F16 KV have separate arithmetic differences.",
                               "F32 and BF16 profiles are compared separately; OFF alignment is a diagnostic, not the repaired semantic target.",
                               "Fixed history/128 targets and greedy agreement do not establish representative language quality.",
                               "No previous strict result or tolerance is overwritten."],
              "failure_storage": "Full failed-element masks are retained losslessly; CSV contains first16 and worst16 failures per comparison, not all rows"}
    for filename, value in [("SUMMARY.json", result), ("LOGITS.json", logit_rows), ("RANKINGS.json", ranking_rows),
                            ("TENSORS.json", tensors), ("ACTQUANT.json", quantizer), ("NLL.json", score_rows), ("FAILURE_MASKS.json", masks)]:
        (a.output / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("workload", "backend", "same_input_official_F32_ActQuant_pass", "model_output_pass", "scores", "strict_scoped_model_exit_code", "legacy_global_V1_exit_code")}, indent=2))
    return strict


def analyze_checkpoints(a, run, dirs, writer, masks_dir, masks):
    ref = official_quantizer(a.official_source)
    quantizer, tensors, cached = [], [], {}
    for index, record in enumerate(run.get("checkpoints", [])):
        kind, layer, positions = record["kind"], record["layer"], record["input_positions"]
        ne = record["ne"]
        if not 0 <= layer < 30 or any(not isinstance(p, int) or not 0 <= p < len(run["token_ids"]) for p in positions):
            raise ValueError("Checkpoint layer or positions invalid")
        count = math.prod(ne)
        output_path = a.run / record.get("output_path", record.get("path", ""))
        actual = raw(output_path, count)
        if kind == "actquant":
            width = ne[0]
            if ne != record["src_ne"] or ne != [width, len(positions), 1, 1] or width not in (2560, 6912):
                raise ValueError("Captured ActQuant extent differs")
            x = raw(a.run / record["input_path"], count).reshape(-1, width)
            if not np.isfinite(x).all():
                raise ValueError("Captured ActQuant input is nonfinite")
            with torch.inference_mode():
                xt = torch.from_numpy(np.array(x, copy=True))
                expected = ref.apply(xt).numpy()
                scale = (127 / xt.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)).numpy()
                codes = (xt * torch.from_numpy(scale)).round().clamp(-128, 127).numpy()
            actual_rows = actual.reshape(-1, width)
            m, bad = compare(actual_rows, expected)
            finite = np.isfinite(actual_rows)
            recovered = np.rint(actual_rows.astype(np.float64) * scale.astype(np.float64))
            code_bad = ~finite | (recovered != codes)
            item = {"capture": index, "node": record["node"], "layer": layer, "input_positions": positions,
                    "ne": ne, "input_sha256": sha(a.run / record["input_path"]),
                    "output_sha256": sha(output_path), "official_output_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                    "integer_code_mismatches": int(np.count_nonzero(code_bad)), "value_metrics": m,
                    "numerical_pass": passed(m) and not code_bad.any()}
            quantizer.append(item)
            if bad.any() or code_bad.any():
                dump_failures(writer, "same_input_ActQuant_value_or_code", positions, record["node"], actual_rows, expected, bad.reshape(actual_rows.shape) | code_bad, masks_dir, masks)
            continue
        if kind not in ("Q_preRoPE", "l_out"):
            raise ValueError("Unknown checkpoint kind: " + kind)
        if count != len(positions) * 2560:
            raise ValueError("Projection/l_out extent differs from token mapping")
        actual_rows = actual.reshape(len(positions), 2560)
        for precision, directory in dirs.items():
            for mode in ("quant_on", "quant_off"):
                if kind == "Q_preRoPE" and layer != 0:
                    continue
                suffix = "layer0-q-preRoPE" if kind == "Q_preRoPE" else f"layer-{layer}-l_out"
                filename = directory / f"{mode}-{suffix}.f32"
                if run["workload"] != "teacher":
                    continue
                cache_key = str(filename)
                if cache_key not in cached:
                    cached[cache_key] = raw(filename, 59 * 2560).reshape(59, 2560)
                expected = cached[cache_key][positions]
                m, bad = compare(actual_rows, expected)
                item = dict(m, capture=index, kind=kind, layer=layer, input_positions=positions,
                            node=record.get("node"), comparison=precision + "_" + mode)
                if kind == "Q_preRoPE" and "input_path" in record:
                    inp = raw(a.run / record["input_path"], count)
                    effective_suffix = "layer0-q-X-effective" if mode == "quant_on" else "layer0-q-X"
                    rfile = directory / f"{mode}-{effective_suffix}.f32"
                    rin = raw(rfile, 59 * 2560).reshape(59, 2560)[positions]
                    item["same_input_bitwise"] = bool(np.array_equal(inp.view(np.uint32).reshape(rin.shape), rin.view(np.uint32)))
                tensors.append(item)
                if precision == "F32" and mode == "quant_on":
                    dump_failures(writer, "F32_ON_" + kind, positions, record.get("node"), actual_rows, expected, bad, masks_dir, masks)
    return quantizer, tensors


def required_positions(run, layer, stage=None):
    if run["workload"] == "corpus":
        return set(range(128))
    if run["prefill"] == "sequential" or layer != 29 or stage in ("qkv", "attn_out"):
        return set(range(59))
    return set(range(26, 59))


def quantizer_coverage(run, calls):
    if not run["activation_quantization"]:
        return None
    records = [r for r in run.get("checkpoints", []) if r["kind"] == "actquant"]
    if len(records) != calls * 120 or run.get("actquant_callbacks_count") != len(records):
        return False
    observed = {}
    for record in records:
        match = re.fullmatch(r"actq_(qkv|attn_out|ffn_up_gate|ffn_down)_out-(\d+)", record["node"])
        if not match or int(match[2]) != record["layer"]:
            return False
        observed.setdefault((record["layer"], match[1]), set()).update(record["input_positions"])
    return all(observed.get((layer, stage)) == required_positions(run, layer, stage)
               for layer in range(30) for stage in STAGES)


def layer_coverage(run):
    if run["workload"] != "teacher":
        return False
    observed = {}
    for record in run.get("checkpoints", []):
        if record["kind"] == "l_out":
            observed.setdefault(record["layer"], set()).update(record["input_positions"])
    return all(observed.get(layer) == required_positions(run, layer) for layer in range(30))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--reference-f32", type=Path, required=True)
    p.add_argument("--reference-bf16", type=Path, required=True)
    p.add_argument("--official-source", type=Path, default=Path(__file__).resolve().parents[1] / "reference/official-bitnet-096f25ae.py")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--exit-mode", choices=("strict", "processing"), default="strict")
    a = p.parse_args()
    if a.output.exists():
        p.error("Output must be new")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        code = analyze(a)
        return code if a.exit_mode == "strict" else 0
    except FileNotFoundError as error:
        print(json.dumps({"analysis_completed": False, "missing": error.filename, "strict_scoped_model_exit_code": 2}), file=sys.stderr)
        return 2
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        print(json.dumps({"analysis_completed": False, "malformed": str(error), "strict_scoped_model_exit_code": 3}), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
