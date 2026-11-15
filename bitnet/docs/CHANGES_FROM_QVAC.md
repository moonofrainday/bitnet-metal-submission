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

One coherent against-base patch is in `../patches/pinned-to-iphone.patch`. It is already applied to this repository: **do not apply it again here**. It changes eight upstream files in total (1,561 insertions, 14 deletions), before packaging/documentation additions. It includes experimental diagnostic hooks; a smaller production-only patch is future cleanup, not represented as complete.

The residual-binding fix is unconditional: for a fused stock ADD, choose the other operand by tensor identity (`dst->src[0] == op`) rather than an operation-type heuristic that can misidentify it when both inputs are MUL_MAT. "Stock default" means the original stock shader/dispatch is selected by default; it does not mean no CPU-side backend code changed.

## New source under `bitnet/`

- `validation/`: standalone original PoC and Q4 shader, CPU FP64 unpack reference, native harness, independent analysis and explicit negative/exit-code checks.
- `shaders/`: source for opt-in Q4, G dual projection/postprocessing, F8, K and split-C controls. `scripts/build-shaders.py` rebuilds platform-specific metallibs; no prebuilt library is shipped.
- `ios/`: replay/operator helpers and a portable staged application build. Upstream SwiftUI/LibLlama remain the foundation; only the bridge hook and background start are added to the staged app.
- `docs/`, `analysis/`, `provenance/`: English scope, measured derived results, source identities and limitations.

Q4 retains the existing stock helper arithmetic. F8/C8 use different packed-dot expressions and accumulation behavior, so improved local FP64 error alone does not imply model-level stock equivalence. All original failure outcomes are retained in the validation notes.
