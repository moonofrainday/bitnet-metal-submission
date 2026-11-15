# iPhone ActQuant qualification

The repaired V2 ActQuant operator passes a new physical iPhone 14 Pro / Apple A16 GPU sample-I/O run: **20 cases, 386,567 values, every output bit equal to the pinned official F32 oracle, strict operator exit 0**. This is an ActQuant operator result, not full-model numerical acceptance. The same signed build also completes full-model diagnostics, but its frozen reference-output gate still fails (strict exit 1). No performance measurements are reported.

The baseline and frozen bound remain unchanged: `abs(actual-reference) <= 2e-4 + 2e-5*abs(reference)`. The public checkpoint, the team's unspecified checkpoint, the original matvec PoC and this activation-quantization repair remain separate scopes.

## Measured operator result

| Item | Observed result |
|---|---|
| Hardware / OS | Physical iPhone 14 Pro, Apple A16 GPU, iOS 27.2.1 |
| Build | Source commit `4c249e3e13dfce7b21bc2bab0912a1ff4807bc7e`; Xcode 27.2, iPhoneOS SDK 27.2 |
| Backend / actual shader | MTL0 / `kernel_act_quant_f32`; all output buffers report MTL0 |
| Execution evidence | 20 actual encode records, 20 synchronized completion markers; native/app/collection exit 0 |
| Independent official-F32 comparison | 386,567 finite values; 0 bit differences, 0 ULP, 0 frozen-bound failures |
| Recovered integer-grid code check | 0 mismatches; not a direct read of internal integer registers |
| Integrity | Every input and canary unchanged; all output hashes verified |
| Cross-device control | All 20 actual output byte files equal the prior Mac V2 MTL0 files |
| Strict operator analyzer | `numerical_pass=true`, strict exit 0 |
| Full-chain FP64 / model outputs / model quality | Not implied by the operator result; native collection keeps these gates null |

Fixtures cover zero and signed zero, epsilon and neighboring values, exact positive/negative half ties, seeded dense inputs, real Q inputs from steps 1/16/32, two captured real graph boundary rows and a seven-value division-order counterexample. Widths are 2560 and 6912 with 1/3/27 rows, plus the seven-element tail.

The iPhone entry includes the same `bitnet/actquant/runner.cpp` and calls the actual `ggml_act_quant` backend path. It does not substitute a standalone arithmetic clone. A read-only audit verifies all 1,067 framework-source, 25 package-source, three app-binary and six helper-object hashes recorded by the signed build against the current local files. Source identity matches the prior clean Mac operator build; platform binaries naturally differ.

## Arithmetic and runtime evidence

V2 uses F32 `(1 / max(absmax(X), 1e-5)) * 127`, ties-to-even rounding, clamp `[-128,127]`, then division by that scale. This matches the observed pinned Torch eager F32 profile, with the official function's compile decorator removed. It does not claim equivalence to every compiled or BF16 profile. Direct `127/denominator` V1 and its boundary failures remain preserved locally.

The A16 runtime log records actual compile-option getters for the new `act_quant` library: `languageVersion=262144`, `mathMode=0` equals `MTLMathModeSafe`, and `mathFloatingPointFunctions=1` equals Precise. The loaded pipeline has SIMD width 32 and threadgroup maximum 1024. Each token row uses one threadgroup of 256 threads. Live input/output offsets and disjoint ranges are logged. Existing stock libraries and ROUND semantics are unchanged. These records are execution/configuration evidence, not hardware counters or performance measurements.

## Full-model diagnostic result

The same signed app/framework completes teacher OFF, teacher ON capture, teacher ON plain, corpus OFF and corpus ON on the Apple A16 GPU. Every native/app process exits 0 with finite logits and confirmed 31/31 offload. Logs place the F16 K/V cache on MTL0: 37.50 MiB total, 18.75 MiB each. All final thermal observations are nominal; these observations are not a sustained-load or performance study.

Teacher runs consume the same 27 prompt IDs plus 32 fixed continuation IDs, producing 33 full 128,256-value logit vectors. Corpus runs consume the same BOS plus 128 fixed test-split targets and save 128 full logit vectors. Context is 512, batch/ubatch 128, CPU threads six and flash attention disabled. This is teacher forcing, not a new free-generation token-parity claim.

| Gate | Measured result |
|---|---|
| Real graph quantizer execution | 3,960 actual Metal encode records and callbacks; synchronized model-completion marker |
| All-stage same-input official-F32 comparison | 25,581,568 values bitwise exact; zero code/bound failures; coverage passes |
| Capture/plain control | All 33 teacher full-logit vectors bitwise identical |
| Captured residual-layer outputs versus F32 ON | 4,462,437 / 4,464,640 values exceed the frozen bound |
| Full-model output gate | `model_output_pass=false`, strict model analyzer exit 1 preserved |
| Full-chain FP64 / representative model quality | UNVERIFIED (`null`); local quantizer checks do not substitute |

| Comparison against source-derived F32 ON | Failed / total logit values | RMSE | Maximum absolute error | Greedy equal positions |
|---|---:|---:|---:|---:|
| Teacher OFF | 4,229,647 / 4,232,448 | 0.251979214 | 1.438055038 | 31/33 |
| Teacher ON, capture/plain | 4,227,009 / 4,232,448 | 0.174339405 | 1.203764677 | 31/33 |
| Corpus OFF | 16,396,195 / 16,416,768 | 0.213242325 | 1.826897383 | 126/128 |
| Corpus ON | 16,384,271 / 16,416,768 | 0.124216825 | 1.136595760 | 126/128 |

All five scoped model analyses retain strict exit 1. The corresponding ON and OFF teacher/corpus full-logit files are also bitwise identical to the previously saved same-configuration Mac files; this is a numerical cross-device control, not a performance result. OFF compared with its own F32 OFF reference has lower logit RMSE (teacher `0.000724058`, corpus `0.000536808`) but still exceeds the frozen rule. Disabling required quantization is not a correctness remedy.

A concrete token-18 witness remains: after the identical 27-token prompt and 17 shared continuation tokens (input position 43), OFF logits for `computations` (83699) and `graphics` (14515) are 22.3536472321 / 22.3587913513, margin `-0.0051441193`; ON logits are 22.3594665527 / 22.3896751404, margin `-0.0302085876`. Both A16 paths choose `graphics`. The cached source-derived F32 ON reference chooses `computations`, margin `+0.0949440002`. Thus this local ActQuant repair does not restore that reference greedy decision. This is a fixed-input numerical witness, not a language-quality conclusion.

The 128-target corpus PPL is **19.127701918 ON** and **19.088221109 OFF**. Source-derived F32 ON/OFF references are **19.129529529 / 19.088566108**. These small test-split smoke scores are recomputed using common FP64 logsumexp. Closer scores do not establish full-logit equivalence, representative model quality, or permission to omit ActQuant. Teacher fixed-continuation PPL is 1.826027356 ON versus source F32 ON 1.812893568; this continuation is not held-out quality evaluation.

A complete model can use an exact quantizer yet differ from its reference because upstream inputs differ. Existing Mac controls expose prefill activation narrowing, KV precision and quantization-threshold discontinuities; see `bitnet/analysis/actquant-2026-11-16/README.md`. Those controls are not new A16 localization experiments. The new phone evidence establishes operator execution and same-input quantizer correctness across the captured graph. Remaining full-model fidelity and product-quality acceptance are unresolved, and no historical strict failure is overwritten.

## Reproduce

Use local `PY`, `SIGNED_BUILD`, `IPHONE_DEVICE_ID`, `GGUF`, `FIXTURE`, `TOKENS`, `REFERENCE_F32`, `REFERENCE_BF16` and `ARTIFACTS` variables. Raw artifacts stay outside the repository; every new run needs a new output directory. Keep the phone connected and unlocked.

```sh
"$PY" bitnet/actquant/run_phone.py --run --operation operator \
  --build-dir "$SIGNED_BUILD" --device "$IPHONE_DEVICE_ID" \
  --model "$GGUF" --fixture "$FIXTURE" --trace \
  --output "$ARTIFACTS/operator"
"$PY" bitnet/actquant/analyze_op.py --fixture "$FIXTURE" \
  --actual "$ARTIFACTS/operator/device/native" \
  --runtime "$ARTIFACTS/operator/device/native/RUN.json" --require-runtime \
  --output "$ARTIFACTS/operator-analysis"
"$PY" bitnet/actquant/run_phone.py --run --operation model \
  --build-dir "$SIGNED_BUILD" --device "$IPHONE_DEVICE_ID" \
  --model "$GGUF" --tokens "$TOKENS" --actquant on --workload teacher \
  --capture --trace --operator-run "$ARTIFACTS/operator" \
  --operator-analysis "$ARTIFACTS/operator-analysis/SUMMARY.json" \
  --output "$ARTIFACTS/teacher-on-capture"
"$PY" bitnet/actquant/compare_model.py \
  --run "$ARTIFACTS/teacher-on-capture/device/native" \
  --reference-f32 "$REFERENCE_F32" --reference-bf16 "$REFERENCE_BF16" \
  --output "$ARTIFACTS/teacher-on-analysis"
```

`run_phone.py` exit 0 means collection and identity checks completed. Numerical acceptance requires the separate analyzer. A model process returning 0 does not override its analyzer's strict exit 1 or missing-gate status. Source and functions are described in `bitnet/actquant/README.md` and `bitnet/analysis/actquant-2026-11-16/README.md`. This directory contains only derived analysis and artifact-name/hash digests; raw logs, tensors, full logits, CSV, signed binaries and device/team identifiers remain local.
