# Changes from pinned QVAC

The source base is upstream commit `2db8b046d555316019ae75d1be90484adc8b9330`; its commit ancestry is intentionally not imported. The published base preserves all upstream paths, modes and blobs except the 63 files under `.github/workflows`. Those CI configurations are omitted because the GitHub credential has no workflow-creation scope; no runtime or local build/test source is excluded. The exact exclusion list, original blob hashes and published baseline import commit are recorded in `../provenance/BASELINE.json`. Compare that published baseline to the submission head to see our code and documentation changes:

```sh
BASE=$(python3 -c 'import json; print(json.load(open("bitnet/provenance/BASELINE.json"))["baseline_import_commit"])')
git diff --stat "$BASE" HEAD
git diff "$BASE" HEAD -- conversion src/models/bitnet.cpp ggml
```

## What already existed

QVAC already provides the BitNet graph, TQ2_0 packing/dequantization, CPU/Metal/Vulkan backends, `kernel_mul_mv_tq2_0_f32`, model/KV management, backend tests and the SwiftUI example with `FinetuneBridge`. We do not claim those as our implementation. The upstream README, license and existing code are preserved.

## Public-checkpoint compatibility: three files

| File | Our change |
|---|---|
| `conversion/base.py` | AutoBitLinear offline checkpoints use direct `trit * weight_scale`; preserve legacy inverse-scale handling for the other class. |
| `conversion/bitnet.py` | Recognize reconstructed offline ternary weights, avoid a second training-style quantization, and write activation metadata. |
| `src/models/bitnet.cpp` | Select existing ReLU-squared FFN support for explicit `relu2`; retain legacy SiLU behavior when the metadata is absent and reject unknown explicit activations. |

These changes adapt Microsoft's public packed checkpoint. They neither add I2_S support nor prove exact equivalence to Microsoft's W1.58A8 execution.

## Guarded Metal integration: five files

| File | Our change |
|---|---|
| `ggml/include/ggml-metal.h` | Diagnostic/replay C interfaces used by our harness. |
| `ggml/src/ggml-metal/ggml-metal-context.m` | Opt-in FFN/layer command-buffer timing and completion-wait hooks. |
| `ggml/src/ggml-metal/ggml-metal-device.h` | Internal declarations for external-library candidate pipelines and diagnostics. |
| `ggml/src/ggml-metal/ggml-metal-device.m` | Candidate pipeline loading/cache and capability checks. |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | Opt-in Q/FFN graph and consumer/encoding-boundary guards, real buffers/offsets, bounds/alias checks, launch selection and stock fallback; also a residual-binding correctness fix. |

The original packaged against-base patch is in `../patches/pinned-to-iphone.patch`. It is already applied to this repository: **do not apply it again here**. It changes eight upstream files in total (1,561 insertions, 14 deletions), before packaging/documentation additions. It includes experimental diagnostic hooks; a smaller production-only patch is future cleanup, not represented as complete. The subsequent `q4-readback-additive.patch` is relative to packaged commit `b6441351acb21418959a52b3864c11d55defa189` and adds the default-off Q diagnostic copies and their fixes to three backend files. Both patches are already applied in this checkout. The additive patch is not a replacement for the compatibility patch.

The residual-binding fix is unconditional: for a fused stock ADD, choose the other operand by tensor identity (`dst->src[0] == op`) rather than an operation-type heuristic that can misidentify it when both inputs are MUL_MAT. "Stock default" means the original stock shader/dispatch is selected by default; it does not mean no CPU-side backend code changed.

## New source under `bitnet/`

- `validation/`: standalone original PoC and Q4 shader, CPU FP64 unpack reference, native harness, independent analysis and explicit negative/exit-code checks.
- `shaders/`: source for opt-in Q4, G dual projection/postprocessing, F8, K and split-C controls. `scripts/build-shaders.py` rebuilds platform-specific metallibs; no prebuilt library is shipped.
- `analysis-tools/`: independent CPU-only complete Q/readback, paired-timing and selected Down analyzers; they distinguish processing success from numerical acceptance.
- `down-controls/`: a separately compiled stock wrapper and a correction-placement negative control, with explicit fixed-input preparation. It changes no default model shader.
- `ios/`: replay/operator helpers and a portable staged application build. Upstream SwiftUI/LibLlama remain the foundation; only the bridge hook and background start are added to the staged app.
- `docs/`, `analysis/`, `provenance/`: English scope, measured derived results, source identities and limitations.

Q4 retains the existing stock helper arithmetic. F8/C8 use different packed-dot expressions and accumulation behavior, so improved local FP64 error alone does not imply model-level stock equivalence. All original failure outcomes are retained in the validation notes.

## 15 November review addition

The default-off same-stock guard control adds `q-guard-control-2026-11-15.patch`, based on packaged commit `eb7858492d0f39e7a4f97f58ef53239c17bcb594` after the two earlier patches. It changes only the existing Metal device/ops files; all three patches are already applied in this checkout. No Metal shader arithmetic is changed by this review patch. The original and Q4 defaults remain distinct.

Source and controlled reference tests now confirm that the public-model graph omits Microsoft's per-token INT8 activation quantize/dequantize. Earlier successful loading, sample arithmetic and stock parity must not be described as faithful W1.58A8 execution. See `../analysis/review-2026-11-15/` for the measured semantic controls, exact original-PoC phone qualification and same-stock guard A/B. Team checkpoint rules remain a separate dependency.
