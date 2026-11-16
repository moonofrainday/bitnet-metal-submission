# BitNet Metal submission candidate

This private repository contains the pinned QVAC Fabric source plus our BitNet Metal kernels, guarded integration and correctness harness. It is a reviewable source package for the Tether assignment, not an upstream QVAC release or a claim that every experimental route passes numerical acceptance.

The source baseline is `tetherto/qvac-fabric-llm.cpp@2db8b046d555316019ae75d1be90484adc8b9330`. The published import preserves every upstream path/blob/mode **except the 63 upstream CI workflow files**, which are explicitly listed with their original hashes. Our code changes are in subsequent commits. See [baseline identity and exclusions](bitnet/provenance/BASELINE.json), [changes from QVAC](bitnet/docs/CHANGES_FROM_QVAC.md), and the preserved [upstream README](README.upstream.md). Upstream licenses and attribution remain in place.

**Current safety revision (16 November):** the optional Q launch/cache contract has been fixed and requalified on Mac. The new signed iPhone build is **not yet device-qualified or performance-measured**. The updated bilingual PDF/Word retains earlier A16 tables, adds the separate Mac attribution evidence and leaves new phone-result cells blank/PENDING. See [the contract fix](bitnet/docs/Q_KERNEL_CONTRACT_FIX.md) and [tonight's staged test commands](bitnet/docs/IPHONE_TEST_PLAN_2026-11-16.md).

**Evening attribution preparation:** [current test order and commands](bitnet/docs/IPHONE_TEST_PLAN_2026-11-16.md) cover identity/cache, both-history correctness, same-stock full-guard controls, isolated Q timing and whole-decode pairing. The [earlier preparation record](bitnet/docs/IPHONE_ATTRIBUTION_PLAN.md) remains historical. Mac correctness and collector integrity pass; the new signed phone build remains untested on A16. No new A16 performance result is claimed.

**Separate M2 Pro supplement:** the [selected-Q value experiment](bitnet/analysis/mac-qcur0-value-2026-11-16/README.md) measures the already selected Q4 candidate, not alternative kernel priorities. Isolated Q4 GPU command-buffer time improves by 8.30%-8.36%, but the current layer-0 integration does not improve whole-decode wall time. The full-guard same-stock control has measurable host-path cost. A separate optional encoder timestamp diagnostic preserves the tested outputs but perturbs scheduling; normal critical-path share remains unverified. This supplement does not update or replace the earlier physical-phone tables in the proposal.

## Start here

| Location | What to inspect |
|---|---|
| [Technical proposal](bitnet/proposal/README.md) | English PDF and editable Word draft covering all eleven assignment topics |
| [Chinese proposal companion](bitnet/proposal/README_ZH.md) | Chinese PDF and editable Word with the same sixteen sections, figures and measured values |
| [Validation](bitnet/validation/README.md) | Original TQ2_0 matvec PoC, stock4x1 specialization, independent FP64 reference, hand-checkable/dense/real-input tests |
| [Source walkthrough](bitnet/docs/SOURCE_WALKTHROUGH.md) | Input/output contract, graph flow, dispatch and stock fallback |
| [Build and run](bitnet/docs/BUILD_AND_RUN.md) | Fresh builds, standalone test and model conversion |
| [iPhone package](bitnet/ios/README.md) | Source-only SwiftUI background-init hook, replay/operator helpers, build and launch tools |
| [Historical iPhone Q4 acceptance](bitnet/docs/IPHONE_Q4_COMPLETE_2026-11-15.md) | ActQuant-OFF all-32-call verification, operator gain, no whole-decode gain, negative Down control |
| [Historical iPhone results](bitnet/docs/IPHONE_RESULTS.md) | Earlier A16 Q4 and G/F8 measurements and their limits |
| [Validation status](bitnet/docs/VALIDATION.md) | Separate parity, FP64 and model-output outcomes; preserved failures |
| [ActQuant experiment](bitnet/analysis/actquant-2026-11-16/README.md) | Default-off public-checkpoint quantizer, same-input CPU/Metal validation and preserved full-model reference failures |
| [New A16 ActQuant qualification](bitnet/analysis/actquant-iphone-2026-11-16/README.md) | Physical-phone all-case operator and real-graph quantizer pass; full-model frozen reference still fails |
| [Joint ActQuant-ON Q4 qualification](bitnet/analysis/actquant-q4-joint-2026-11-16/README.md) | Two histories on Mac and A16: all32 decode Q calls pass FP64 and stock-ON parity; new phone operator timing is separate from model fidelity |
| [Limitations](bitnet/docs/LIMITATIONS.md) | Unverified scope and remaining delivery work |

## Build a numerical smoke test

Requirements: Apple Silicon macOS, full Xcode with the Metal toolchain, CMake and Python 3.9 or newer. Select the full Xcode for the command/session; do not change global `xcode-select` merely for this project.

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
python3 bitnet/scripts/build-mac.py \
  --build ../bitnet-local/patched-build \
  --output ../bitnet-local/patched-checks
```

This builds **this modified source** and runs the original 59-case TQ2_0 Metal backend regression. Use a fresh output/build directory for each invocation. To compare the standalone PoC against a clean **unmodified** QVAC runtime, follow the separate [validation entry point](bitnet/validation/RERUN.md). Models and historical captures are optional for its two synthetic cases.

## Scope and selection

- **Stock remains the default.** All experimental routes require explicit opt-in and matching runtime contracts.
- **Original PoC:** an independently authored TQ2_0/F32 matvec. Local same-input FP64 success does not establish full-model stock compatibility.
- **Q stock4x1:** a fixed-shape/launch specialization that retains stock four-row arithmetic, not a new dot-product algorithm. Its integrated route is restricted to layer-0 `Qcur-0`, single-token decode.
- **G:** stock-arithmetic FFN dual projection with separate compatible postprocessing. Observed stock output parity does not repair inherited full-FFN FP64 failures.
- **F8/K:** performance research variants with recorded complete-chain FP64 and stock-relative model-output failures. They are not the recommended strict-acceptance baseline.

The checkpoint is the **public Microsoft BitNet b1.58 2B-4T**, converted to TQ2_0 with the documented compatibility changes. It is separate from the team's checkpoint. I2_S is not TQ2_0.

Primary proposal performance tables concern **physical iPhone 14 Pro/A16 measurements**. The separately authorized M2 Pro supplement reports Mac measurements explicitly and cannot predict A16 performance. Historical ActQuant-OFF results and the fresh joint ActQuant-ON Q4 campaign use different input profiles and are reported separately. Isolated Q measurements exclude activation-quantizer preparation and do not establish whole-model acceleration. Raw samples stay local.

## Publication contents

Included: base source, our source additions, patches, English instructions, derived summaries and digest-only evidence provenance. Excluded: our model checkpoints, raw tensor/logit payloads, per-row/raw timing CSVs, captures, compiled frameworks/libraries/apps, provisioning files and signing credentials. Public tokenizer-vocabulary GGUF fixtures inherited unchanged from QVAC remain upstream test assets; they are not our downloaded 2B model.

Raw experimental evidence remains local. A fresh run writes its own local logs and numerical outputs. The package is AI-assisted; final submission and explanations remain subject to the author's review. No access has been granted to the assignment team by creating this private repository.

Upstream `.github/workflows` is omitted because the available GitHub credential cannot create workflows, and upstream CI is not required for this private source package. This exclusion changes no runtime, model, shader or local test source. The complete exact upstream snapshot is preserved locally; it is not claimed that the published baseline tree is byte-identical to the full upstream tree.

## Review findings

The original authored 4x2 shader now has separate physical-A16 sample-I/O qualification; Q4 alone retains the current complete layer-0 graph acceptance. The preserved default public-model TQ2_0/F32 path omits Microsoft per-token INT8 activation quantize/dequantize. A new `LLAMA_BITNET_ACTQUANT=1` experiment implements the source-derived F32 stage on CPU and Metal: its expanded fixtures and all captured actual-input quantizers pass, while complete model-reference acceptance still fails. This experimental branch is not a promoted faithful-model implementation. Its new physical-phone operator and all captured actual-input quantizer tests pass; the complete model-reference gate remains failed. Stock agreement and successful generation do not establish faithful public-model execution. See [review controls](bitnet/analysis/review-2026-11-15/README.md), [ActQuant analysis](bitnet/analysis/actquant-2026-11-16/README.md) and the revised [technical proposal](bitnet/proposal/PROPOSAL.md). Raw tensors, corpus, logits, logs and timings remain local.
