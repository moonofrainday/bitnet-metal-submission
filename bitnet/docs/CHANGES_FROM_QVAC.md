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

## Public ActQuant repair and current joint experiment, 16 November

The old eight-file integration map describes a historical snapshot, not the entire current diff. The already-applied `../patches/public-2b-actquant.patch` adds append-only `GGML_OP_ACT_QUANT` / `ggml_act_quant`, per-token F32 fake quantization and public-profile graph placements. Source `4c249e3` identifies the tested ActQuant A16 binary; `e84732c` consolidates its evidence/docs. Do not apply that patch again to this modified tree.

| Files/modules | Added behavior |
|---|---|
| `ggml/include/ggml.h`, `ggml/src/ggml.c` | Appended op/API, names and separate output tensor |
| `ggml/src/ggml-cpu/{ops.h,ops.cpp,ggml-cpu.c,ggml-cpu.cpp}` | CPU support, scheduling and F32 reciprocal-then-multiply scale with ties-even rounding |
| `ggml/src/ggml-metal/{ggml-metal-device.cpp,ggml-metal-device.m,CMakeLists.txt}` | New pipeline/library; safe/precise options restricted to the new library |
| `ggml/src/ggml-metal/{ggml-metal-impl.h,ggml-metal-ops.h,ggml-metal-ops.cpp,kernels/act_quant.metal}` | ABI, bounds/disjoint-buffer checks, encoding and one-token-row-per-threadgroup quantizer |
| `src/models/bitnet.cpp` | Default-off public-profile placement: 120 shared nodes cover 210 projections; lm_head unchanged |
| `tests/test-backend-ops.cpp`, `docs/ops.md` | Targeted backend cases and qualified support documentation |
| `bitnet/actquant/`, `bitnet/ios/src/iphone-actquant.mm` | Shared native runner, independent source oracle, strict analyzers and phone entry |

This is a quantizer repair, not a new INT8 matmul or a complete model-fidelity fix. A16 same-input operator and graph checks pass; the source-F32-ON model comparison is measured false, strict exit 1. Historical matmul/FFN failures remain unchanged. New joint ActQuant/Q4 integration adds default-off `LLAMA_BITNET_ACTQUANT_Q4=1`: only ActQuant ON, layer-0 Q, rows=4 and one SIMD group are permitted, with the live ACT_QUANT producer and epsilon checked. C8, FFN and other layers remain unauthorized. The joint implementation from `edbd109` passes its own two-history phone stock-ON Q-only qualification, including all-call FP64 and fallback. Fresh ON operator pairing measures 2.61%-2.68% reduction; joint decode and corpus/PPL are unmeasured. See `../analysis/actquant-q4-joint-2026-11-16/`; no trusted-reference or quality pass follows.

`../patches/actquant-q4-joint.patch` records the nine-file joint implementation from base `e84732ced307d7dbb0dabe1f7c60c8c4b3bdc091` to tested compute commit `edbd109bc152decbbc7274b703d21bd6099bf98c`. It is already applied here and is not a patch directly against pinned QVAC. Later CPU-only fixture/timing analyzers are shipped as source; they were not compiled into the tested phone binary. Rebuild from this full checkout using the documented fresh output directories.

## Q launch/cache safety revision

The optional Q4/C8 route now uses canonical versioned entries and exact launch descriptors; the external pipeline cache binds current file identity before lookup. The shared headers are `ggml-metal-bitnet-q-config.h` and `ggml-metal-bitnet-q-contract.h`. Shader arithmetic and stock defaults are preserved. Mac correctness is requalified. The later source-bound `14a9f103` package also passes fresh A16 scoped qualification; see [the safety note](Q_KERNEL_CONTRACT_FIX.md) for the original failure, patch and tests, and [current A16 results](../analysis/iphone-qcur0-attribution-2026-11-16/README.md) for the measured runtime and performance scope.

## A16 attribution and measured qualification

Default-off guard/getter/encode and natural-command-buffer diagnostics support same-context ActQuant-ON S/QB/Q4 replay and source-bound qualification. QB requires the same exact Q4/producer contract and explicit permission while retaining actual stock dispatch. The [completed A16 analysis](../analysis/iphone-qcur0-attribution-2026-11-16/README.md) records two independent 18/18 contract/cache passes, 20 ActQuant cases, and both histories' complete 32-call Q FP64 and 33-logit stock-parity checks. No shader arithmetic or frozen tolerance changed during this campaign.

The current package's isolated Q4 benefit is measured separately in Fair and post-model Serious cohorts. Complete 32-token decode is slower in all six formal Serious-state rounds: QB by 2.71%-3.74%, Q4 by 2.86%-3.76%. Instrumented warm snapshot/key work is a measured host region, not a complete CPU/GPU/thermal decomposition. Stock remains the default. These results do not revise the earlier `edbd109` binary's unmeasured full-decode scope or the trusted-model reference failure, strict exit 1.

## Historical Mac selected-Q value supplement

The [retained derived Mac analysis](../analysis/mac-qcur0-value-2026-11-16/README.md) evaluates the already selected Q4 operator. It separates a measured isolated GPU benefit from absent whole-decode gain and observed warm getter/identity costs. It is explicitly M2 Pro evidence, not a phone performance result.

The historical controller source is identified by commit `5a8b534a289c294a50ad2f2718f5d07fe2a2f78f`. A separate optional four-file encoder probe against packaged base `578ec123ec2f0eed01f3edf6dc09ac6dcbe6104a` is identified by isolated source commit `6c775b38cc0283ba126da3dc1890bfd9a2a6b864`. That probe changes Metal device/ops timing boundaries and the attribution harness, not shader arithmetic; it was never applied to the normal or signed-phone build. Its measured intervals include added fence/encoder boundaries and cannot establish normal critical-path share.

Mac-only controllers, the optional probe patch and detailed instructions are retained locally rather than in this focused phone submission. The English analysis and machine-readable aggregate evidence remain included because the report discusses the observed attribution limits. The submitted phone build and validation entries require none of that archived tooling.
