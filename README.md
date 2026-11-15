# BitNet Metal submission candidate

This private repository contains the pinned QVAC Fabric source plus our BitNet Metal kernels, guarded integration and correctness harness. It is a reviewable source package for the Tether assignment, not an upstream QVAC release or a claim that every experimental route passes numerical acceptance.

The source baseline is `tetherto/qvac-fabric-llm.cpp@2db8b046d555316019ae75d1be90484adc8b9330`. The published import preserves every upstream path/blob/mode **except the 63 upstream CI workflow files**, which are explicitly listed with their original hashes. Our code changes are in subsequent commits. See [baseline identity and exclusions](bitnet/provenance/BASELINE.json), [changes from QVAC](bitnet/docs/CHANGES_FROM_QVAC.md), and the preserved [upstream README](README.upstream.md). Upstream licenses and attribution remain in place.

## Start here

| Location | What to inspect |
|---|---|
| [Technical proposal](bitnet/proposal/README.md) | English PDF and editable Word draft covering all eleven assignment topics |
| [Validation](bitnet/validation/README.md) | Original TQ2_0 matvec PoC, stock4x1 specialization, independent FP64 reference, hand-checkable/dense/real-input tests |
| [Source walkthrough](bitnet/docs/SOURCE_WALKTHROUGH.md) | Input/output contract, graph flow, dispatch and stock fallback |
| [Build and run](bitnet/docs/BUILD_AND_RUN.md) | Fresh builds, standalone test and model conversion |
| [iPhone package](bitnet/ios/README.md) | Source-only SwiftUI background-init hook, replay/operator helpers, build and launch tools |
| [Latest iPhone Q4 acceptance](bitnet/docs/IPHONE_Q4_COMPLETE_2026-11-15.md) | Fresh-build all-32-call numerical verification, operator gain, no whole-decode gain, negative Down control |
| [Historical iPhone results](bitnet/docs/IPHONE_RESULTS.md) | Earlier A16 Q4 and G/F8 measurements and their limits |
| [Validation status](bitnet/docs/VALIDATION.md) | Separate parity, FP64 and model-output outcomes; preserved failures |
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

All performance claims here concern **physical iPhone 14 Pro/A16 measurements**. Mac runs establish build and numerical reproducibility only. Historical phone results are explicitly distinguished from the fresh signed-build Q4 campaign documented above. Raw samples stay local in both cases.

## Publication contents

Included: base source, our source additions, patches, English instructions, derived summaries and digest-only evidence provenance. Excluded: our model checkpoints, raw tensor/logit payloads, per-row/raw timing CSVs, captures, compiled frameworks/libraries/apps, provisioning files and signing credentials. Public tokenizer-vocabulary GGUF fixtures inherited unchanged from QVAC remain upstream test assets; they are not our downloaded 2B model.

Raw experimental evidence remains local. A fresh run writes its own local logs and numerical outputs. The package is AI-assisted; final submission and explanations remain subject to the author's review. No access has been granted to the assignment team by creating this private repository.

Upstream `.github/workflows` is omitted because the available GitHub credential cannot create workflows, and upstream CI is not required for this private source package. This exclusion changes no runtime, model, shader or local test source. The complete exact upstream snapshot is preserved locally; it is not claimed that the published baseline tree is byte-identical to the full upstream tree.
