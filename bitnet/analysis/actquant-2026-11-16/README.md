# Public BitNet 2B ActQuant repair: Mac numerical qualification

> Historical Mac scope: the phone UNVERIFIED statements below describe the Mac experiment cutoff. A later physical A16 run from source `4c249e3` passes operator and captured same-input quantizer checks, while completed full-model reference comparison retains strict exit 1. See `../actquant-iphone-2026-11-16/` for the 16 November update. The later `edbd109` joint ActQuant/Q4 scope passes against stock ON; see `../actquant-q4-joint-2026-11-16/`. No historical verdict is overwritten.

This analysis concerns the public Microsoft checkpoint, not the team's unspecified checkpoint. It preserves the previous QVAC path, original strict failures, and the frozen bound `abs(actual-reference) <= 2e-4 + 2e-5*abs(reference)`. It reports no Mac performance measurements. The repaired path has **not been verified on a physical iPhone**.

## Implementation and reference profile

The opt-in `LLAMA_BITNET_ACTQUANT=1` adds the missing per-token activation fake-quantization stage before the public model's seven projections per layer. Four shared input stages per layer cover Q/K/V, attention output, gate/up and down: 120 quantizer nodes cover 210 linear projections across 30 layers. The tied `lm_head` is unchanged. Custom Q and FFN routes remain disabled. Flag OFF preserves the prior graph; start a new process/context when changing the flag because graph reuse does not key this environment variable.

For each finite, contiguous F32 token row, V2 computes:

```
d = max(max(abs(X)), float32(1e-5))
r = float32(1 / d)
s = float32(r * 127)
q = clamp(round_to_nearest_ties_even(float32(X * s)), -128, 127)
Y = float32(q / s)
```

`Y` is dequantized F32; this change does not introduce an I8 matmul. The F32 arithmetic matches the observed **pinned Torch eager source profile**, obtained from the immutable official function with its `torch.compile` decorator removed. It does not claim bitwise equivalence to a different compiled or BF16 execution profile. The initial implementation used `s=float32(127/d)`: mathematically equivalent, but a different floating-point evaluation.

## Why V1 failed, and what V2 fixes

| Qualification | V1 direct division | V2 reciprocal then multiply |
|---|---:|---:|
| Initial fixtures, per backend | 17 cases / 377,088 values | Included in expanded suite |
| Expanded fixtures, per backend | 20 cases / 386,567 values | 20 cases / 386,567 values |
| Expanded-suite recovered code mismatches | 4 | 0 |
| Expanded-suite frozen-bound failures | 4 | 0 |
| Expanded-suite operator strict exit | 1 | 0 |
| Expanded-suite bitwise differences to official F32 | 65,505 | 0 |
| Maximum value ULP, expanded suite | 8,388,607 | 0 |

These results hold separately for actual native CPU and MTL0 runs. CPU/Metal raw outputs are bitwise equal for the initial 377,088-value suite. The expanded suite adds two real full-graph boundary rows and a seven-value constructed boundary case. V1 fails one code in each real row and two in the constructed case. V2 matches every official output bit in the expanded suite; no nonfinite outputs were observed. Integer codes are recovered from the output grid in F64, not read directly from a GPU integer register.

The original 17-case suite had no wrong codes or frozen-bound failures, but 61,964 values differed by at most 2 ULP. Independent CPU controls reproduced every native output using direct division and every official output using reciprocal then multiplication. The real graph then exposed two code errors among 25,581,568 captured values: layer 18 prefill attention-output input and layer 25 decode FFN-down input. Their expected/native codes were respectively `-25/-24` and `13/12`. This is why the original small-suite pass was insufficient.

V2 changes only scale evaluation. Maximum reduction, rounding, clamp, output division, guards and launch remain unchanged. V1 source/binaries and failures are preserved locally. Offline AIR contains separate reciprocal division and multiplication; actual runtime compile-option getters confirm safe math and precise floating-point functions for the **new library only**. Stock library options and existing ROUND semantics are unchanged.

## Gates that remain separate

| Gate | Status and scope |
|---|---|
| Native local quantizer execution and official same-input comparison | V2 PASS: CPU and MTL0, 20 cases / 386,567 values each, strict exit 0 |
| Backend regressions | MTL0 ACT_QUANT 16/16; original TQ2_0 MUL_MAT 59/59, including V2 reruns |
| V2 actual-input quantizer and graph coverage | PASS: batched 3,960 captures / 25,581,568 values; sequential 7,080 / 25,827,840; all bits match official F32 on each same actual input, zero code/bound failures |
| V2 intermediate layers and teacher-forced full logits | FAIL: `model_output_pass=false`, strict scoped exit 1 in both prefill modes |
| V2 128-target test-split PPL smoke | Measured: 19.127701918; official F32 ON 19.129529529; model-quality acceptance remains null |
| Full-chain FP64 / product quality | UNVERIFIED (`null`); previous complete-FFN failures remain preserved |
| Physical iPhone validation | UNVERIFIED; a fresh unsigned iPhoneOS build succeeds, with no installation or device GPU execution |

The captured batched/sequential runs contain respectively 3,960/7,080 actual Metal ActQuant encode records and confirmed 31/31 layer offload. V2 capture/plain teacher runs produce bitwise identical full logits at all 33 positions. Original and V2 flag-OFF teacher outputs are also bitwise identical at all 33 positions.

| V2 comparison against F32 ON | Failed / total logit values | RMSE | Greedy equal positions | Strict exit |
|---|---:|---:|---:|---:|
| Teacher, batched prefill | 4,227,009 / 4,232,448 | 0.174339405 | 31/33 | 1 |
| Teacher, sequential prefill | 4,225,157 / 4,232,448 | 0.175027301 | 30/33 | 1 |
| Teacher, sequential F32-KV control | 4,223,875 / 4,232,448 | 0.146621968 | 31/33 | 1 |
| 128-target corpus | 16,384,271 / 16,416,768 | 0.124216825 | 126/128 | 1 |

Captured residual-layer outputs also fail the same bound: batched 4,462,437/4,464,640 values; sequential 4,529,005/4,531,200. Teacher-forced PPL is 1.826027356 (batched) or 1.822546944 (sequential), versus F32 ON 1.812893568. This fixed continuation is not held-out quality evaluation. The independent 128-target test-split PPL is only a small smoke; its near-reference score does not establish full-logit equivalence or a product-quality pass. Sequential prefill keeps local quantization exact but does not resolve the full-model discrepancy, so the discrepancy cannot be attributed solely to batched prefill.

Full-model comparison uses the same 27 prompt IDs plus 32 continuation IDs, F16 KV, context 512, batch/ubatch 128 and six CPU threads. F32 ON is the target reference; F32 OFF and BF16 profiles are separate controls. Remaining differences may include CPU Q8_K activation repacking, TQ2_0 folded scales, matmul/reduction order, F16 KV and source no-cache execution. A smaller RMSE, closer PPL or matching greedy token cannot overwrite the original elementwise failure. Extreme/subnormal F32 behavior is outside this fixture qualification.

## Remaining drift localized beyond the quantizer

A separate one-layer official-source probe reproduces its existing F32 ON layer-0 cache bitwise for all 59 positions / 151,040 values. It is a checked reference probe, not a new full-model implementation. Two controls distinguish the remaining drift from V2 quantizer defects:

- **Batched prefill Q:** with captured W/Y fixed, changing only reference X from F32 to an F16 round-trip reduces same-input FP64 failures from 11,168/69,120 to 0. Sequential N=1 Q passes same-input FP64 at all 59 positions. This localizes the initial prefill-Q discrepancy to the activation precision path; it does not repair the whole model.
- **Sequential propagation:** with F16 KV, the attention-output pre-quantization X differs from reference by at most `3.380700946e-5`, with zero frozen failures; after quantization, cross-reference output has 676 failures. At position 0, seven values first cross the bound. F32 KV reduces the pre-quantization X maximum difference to `4.202593118e-7`; the first position-0 failure then moves to FFN-down effective input, one value differing by `0.070197657`. Inputs that differ near quantization thresholds can legitimately produce different codes even when the quantizer is exact on each supplied input. At FFN-down column 1808, a `4.768371582e-7` input difference gives code 1 versus 2 while V2 still equals the official quantizer on its own input. The full F32-KV control has 7,080 captures / 25,827,840 bitwise-exact local quantizer outputs, but its full-logit strict exit remains 1; teacher-forced PPL is 1.815986257. This is a separate precision control, not a change to the F16-KV acceptance configuration.

The next check should target upstream arithmetic/precision at these boundaries, preserving same-input and propagation gates separately. A fresh unsigned iPhoneOS arm64 build succeeds, but installation and phone GPU execution are still UNVERIFIED.

The reusable implementation patch is `bitnet/patches/public-2b-actquant.patch`. A separate fresh build through the operator entry also passes all 20 cases / 386,567 values on CPU and MTL0, ACT_QUANT 16/16 and TQ2_0 MUL_MAT 59/59.

## Source walkthrough

- `src/models/bitnet.cpp`, `llama_model_bitnet::graph`: public-profile guard, explicit switch and quantizer placement.
- `ggml/include/ggml.h` and `ggml/src/ggml.c`, `ggml_act_quant`: append-only op/API and separate F32 result.
- `ggml/src/ggml-cpu/ops.cpp`, `ggml_compute_forward_act_quant`: token-wise CPU maximum, scale and ties-even reference execution.
- `ggml/src/ggml-metal/kernels/act_quant.metal`, `kernel_act_quant_f32`: one threadgroup/token, maximum reduction and precise quantization.
- `ggml/src/ggml-metal/ggml-metal-ops.cpp`, `ggml_metal_op_act_quant`: real buffer bounds/offsets, disjoint output, binding and launch.
- `ggml/src/ggml-metal/ggml-metal-device.m`: new-library-only math options and optional actual property getter trace.
- `tests/test-backend-ops.cpp`, `test_act_quant`: 16 backend cases including tail, zero, ties, epsilon and offset contiguous views.
- `bitnet/actquant/{prepare_fixtures,analyze_op,compare_model,run_mac}.py`: immutable source oracle, independent checks and fresh-build entry.

## Reproduction

Set `PY`, `ARTIFACTS`, `REAL_Q_FIXTURE`, `GGUF`, `TOKENS`, `REFERENCE_F32`, `REFERENCE_BF16` and `DEVELOPER_DIR` to local dependencies/cache locations. Keep `ARTIFACTS` outside this repository; use a new output directory. Raw model data and references are intentionally not published here.

```sh
"$PY" bitnet/actquant/test_analyze_op.py
"$PY" bitnet/actquant/test_model_contract.py
"$PY" bitnet/actquant/prepare_fixtures.py \
  --official-source bitnet/reference/official-bitnet-096f25ae.py \
  --real-fixture "$REAL_Q_FIXTURE" --output "$ARTIFACTS/fixtures"
"$PY" bitnet/actquant/run_mac.py --mode full \
  --output "$ARTIFACTS/new-run" --fixture "$ARTIFACTS/fixtures" \
  --model "$GGUF" --tokens "$TOKENS" \
  --reference-f32 "$REFERENCE_F32" --reference-bf16 "$REFERENCE_BF16" \
  --developer-dir "$DEVELOPER_DIR"
```

For all 20 cases, add the locally preserved `--boundary-capture`, `--boundary-failures` and `--counterexample` inputs when preparing fixtures. The strict operator analyzer returns 0 for pass, 1 for observed failure, 2 for missing evidence and 3 for malformed evidence. Model analysis retains its own strict scoped exit; processing success is a separate field. Do not use `--exit-mode processing` as acceptance.

`SUMMARY.json` contains only derived aggregates and evidence-name/hash digests. Local evidence names include `boundary-v1-{CPU,MTL0}-analysis/SUMMARY.json`, `operator-v2-{CPU,MTL0}-analysis/SUMMARY.json`, `ACTQUANT_FIRST_LOCAL_FAILURES.json`, `backend-actquant-v2.log`, `backend-tq2-v2.log` and the preserved V1/V2 source manifests. No raw logs, tensors, logits, CSV, device identifiers or local absolute paths are included in this analysis package.
