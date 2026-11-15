#!/usr/bin/env python3
"""CPU-only one-layer source probe using the existing exact 59-token history."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
from types import SimpleNamespace
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from compare_model import aggregate, compare


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--reference-cache", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1] / "reference")
    p.add_argument("--native-run", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Output must be new")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    torch.manual_seed(1)
    sys.path.insert(0, str(a.source_dir))
    from streamed_reference import make_env, replace_linears
    env, transformations = make_env(a.source_dir)
    cfg = json.loads((a.model_dir / "config.json").read_text())
    cfg.update(attention_bias=False, attention_dropout=0.0, _attn_implementation="eager", rope_scaling=None, head_dim=128)
    config = SimpleNamespace(**cfg)
    tokens = json.loads((a.reference_cache / "tokens.json").read_text())
    ids = tokens["teacher_forced"]
    if len(ids) != 59 or len(tokens["prompt"]) != 27 or len(tokens["continuation"]) != 32:
        raise ValueError("Expected the existing exact 27+32 token history")
    manifest = json.loads((a.reference_cache / "MANIFEST.json").read_text())
    if manifest["dtype"] not in ("torch.float32", "float32") or manifest["checkpoint_sha256"] != sha(a.model_dir / "model.safetensors"):
        raise ValueError("Checkpoint or F32 source profile identity differs")
    a.output.mkdir(parents=True)
    tensors = {}
    active = {"module": None}

    def record(name, tensor):
        if not isinstance(tensor, torch.Tensor):
            return
        x = tensor.detach().float().contiguous().numpy()
        path = a.output / (name.replace(".", "-").replace("/", "-") + ".f32")
        x.tofile(path)
        tensors[name] = {"file": path.name, "shape": list(x.shape), "sha256": sha(path),
                         "elements": int(x.size), "nonfinite": int(np.count_nonzero(~np.isfinite(x)))}

    original_quant = env["ActQuant"].forward

    def quant_probe(ctx, x):
        y = original_quant(ctx, x)
        record(active["module"] + "/effective", y)
        return y

    env["ActQuant"].forward = staticmethod(quant_probe)
    handles = []
    with torch.inference_mode(), safe_open(a.model_dir / "model.safetensors", framework="pt", device="cpu") as ckpt:
        embedding = ckpt.get_tensor("model.embed_tokens.weight")
        h = F.embedding(torch.tensor(ids).reshape(1, -1), embedding).float()
        del embedding
        record("layer/input", h)
        with torch.device("meta"):
            layer = env["BitNetDecoderLayer"](config, 0).float()
        replace_linears(layer, env, torch.float32)
        prefix = "model.layers.0."
        state = {k[len(prefix):]: ckpt.get_tensor(k) for k in ckpt.keys() if k.startswith(prefix)}
        state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
        layer.load_state_dict(state, strict=True, assign=True)
        layer.eval()
        del state
        for name, module in layer.named_modules():
            if isinstance(module, env["AutoBitLinear"]):
                def before(_module, args, name=name):
                    active["module"] = name
                    record(name + "/input", args[0])
                def after(_module, _args, y, name=name):
                    record(name + "/output", y)
                    active["module"] = None
                handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
            elif isinstance(module, env["BitNetRMSNorm"]):
                def before_norm(_module, args, name=name):
                    record(name + "/input", args[0])
                def after_norm(_module, _args, y, name=name):
                    record(name + "/output", y)
                handles.extend([module.register_forward_pre_hook(before_norm), module.register_forward_hook(after_norm)])
        def attn_after(_module, _args, y):
            record("attention/output", y[0])
        handles.append(layer.self_attn.register_forward_hook(attn_after))
        def activation_after(_module, _args, y):
            record("mlp/relu_squared", y)
        handles.append(layer.mlp.act_fn.register_forward_hook(activation_after))
        rotary = env["BitNetRotaryEmbedding"](config)
        positions = torch.arange(59).unsqueeze(0)
        mask = torch.full((59, 59), torch.finfo(torch.float32).min).triu(1).reshape(1, 1, 59, 59)
        output = layer(h, attention_mask=mask, position_ids=positions,
                       position_embeddings=rotary(h, positions))[0]
        record("layer/output", output)
        for handle in handles:
            handle.remove()
        reference = np.fromfile(a.reference_cache / "quant_on-layer-0-l_out.f32", dtype="<f4").reshape(1, 59, 2560)
        cached_metrics, _ = compare(output.numpy(), reference)
    probe_neutral = cached_metrics["bitwise_differences"] == 0
    for role in ("input", "effective"):
        gate = np.fromfile(a.output / tensors["mlp.gate_proj/" + role]["file"], dtype="<f4")
        up = np.fromfile(a.output / tensors["mlp.up_proj/" + role]["file"], dtype="<f4")
        if not np.array_equal(gate.view(np.uint32), up.view(np.uint32)):
            raise ValueError("Shared gate/up source input does not match")
    if not probe_neutral:
        result = {"probe_output_vs_existing_cache_bitwise": False, "cached_output_comparison": cached_metrics,
                  "native_attribution_valid": False, "reason": "Instrumented source output differs; do not attribute native stages"}
        (a.output / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 1
    roles = {"actq_qkv_out-0": "self_attn.q_proj", "actq_attn_out_out-0": "self_attn.o_proj",
             "actq_ffn_up_gate_out-0": "mlp.gate_proj", "actq_ffn_down_out-0": "mlp.down_proj"}
    runs = {}
    for native in a.native_run:
        run = json.loads((native / "RUN.json").read_text())
        if run["token_ids"] != ids or run["workload"] != "teacher" or not run["activation_quantization"]:
            raise ValueError("Native exact history or quantizer mode differs")
        rows = []
        for capture, node in enumerate(run["checkpoints"]):
            if node["layer"] != 0:
                continue
            pos = node["input_positions"]
            if node["kind"] == "actquant":
                module = roles[node["node"]]
                pairs = [("input", node["input_path"], module + "/input"),
                         ("effective", node["output_path"], module + "/effective")]
            elif node["kind"] == "Q_preRoPE":
                pairs = [("linear_output", node["path"], "self_attn.q_proj/output")]
            elif node["kind"] == "l_out":
                pairs = [("layer_output", node["path"], "layer/output")]
            else:
                continue
            for phase, path, target in pairs:
                metadata = tensors[target]
                width = metadata["shape"][-1]
                ref = np.fromfile(a.output / metadata["file"], dtype="<f4").reshape(59, width)[pos]
                actual = np.fromfile(native / path, dtype="<f4").reshape(len(pos), width)
                m, _ = compare(actual, ref)
                rows.append(dict(m, capture=capture, input_positions=pos, node=node["node"], phase=phase, reference=target))
        grouped = {}
        for row in rows:
            grouped.setdefault(row["node"] + "/" + row["phase"], []).append(row)
        runs[native.name] = {"run_manifest_sha256": sha(native / "RUN.json"), "kv": run["kv"], "prefill": run["prefill"],
                             "comparisons": rows, "aggregate_by_stage": {k: aggregate(v) for k, v in grouped.items()},
                             "first_bitwise_difference": next((v for v in rows if v["bitwise_differences"]), None),
                             "first_frozen_tolerance_failure": next((v for v in rows if v["frozen_tolerance_failures"]), None)}
    result = {"scope": "One-layer0 official-source F32 ON CPU probe only; no GPU or 30-layer model rerun",
              "probe_output_vs_existing_cache_bitwise": probe_neutral, "cached_output_comparison": cached_metrics,
              "checkpoint_sha256": manifest["checkpoint_sha256"], "source_transformations": transformations,
              "reference_tensors": tensors, "native_runs": runs,
              "peak_process_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "full_model_numerical_or_quality_acceptance": None}
    (a.output / "SUMMARY.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"probe_output_vs_existing_cache_bitwise": probe_neutral, "captured_source_tensors": len(tensors),
                      "native_runs": {k: {"first_frozen_tolerance_failure": v["first_frozen_tolerance_failure"]} for k, v in runs.items()},
                      "peak_process_RSS_bytes": result["peak_process_RSS_bytes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
