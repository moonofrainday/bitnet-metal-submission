# BitNet mobile GPU submission

This repository contains the source code, sample-I/O harness and English technical report for the Tether BitNet assignment. It uses the pinned QVAC Fabric baseline `2db8b046d555316019ae75d1be90484adc8b9330` and a public Microsoft BitNet checkpoint for the compatibility experiment. The team's checkpoint remains a separate dependency.

## Review the deliverables

| Entry | Purpose |
| --- | --- |
| [Technical report PDF](bitnet/proposal/BitNet_Mobile_GPU_Proposal.pdf) | Plan covering the eleven brief topics, implementation, measured results, limitations and risks |
| [Original PoC and independent harness](bitnet/validation/README.md) | Authored TQ2_0/F32 matvec, sample input/output, independent FP64 reference and clean stock comparison |
| [Source walkthrough](bitnet/docs/SOURCE_WALKTHROUGH.md) | Tensor contract, packing/scales, real graph selection, buffer offsets, dispatch and stock fallback |
| [Build and run](bitnet/docs/BUILD_AND_RUN.md) | Dependencies, clean build, conversion and physical-phone commands |
| [Submission scope and attribution](bitnet/docs/SUBMISSION_SCOPE.md) | Original shader versus stock-derived Q4; required code versus optional research dependencies |
| [Current iPhone results](bitnet/analysis/iphone-qcur0-attribution-2026-11-16/README.md) | Two-history correctness, stock/QB/Q4 controls, timing and attribution limits |
| [File review](bitnet/provenance/FILE_REVIEW.json) | Retained authored/modified files and their submission purpose |

The original `tq2_0_matvec_poc` is an authored unpack/dot kernel. The separately attributed `tq2_0_matvec_stock4x1_v1` reuses stock helper arithmetic and specializes the threadgroup configuration. Q4 replaces only layer-0 `Qcur-0` during single-token decode. The surrounding Transformer and stock default remain intact. FFN/down/PPL helpers retained by the current iOS builder are identified as supporting or optional research code; their presence does not make them accepted primary kernels.

## Measured status on iPhone 14 Pro

The current source-bound A16 package uses ActQuant ON and F16 KV. The fresh phone campaign completes two independent 18-case loader/cache checks, all 20 ActQuant cases, and both fixed histories. Each history checks all 32 single-token Q outputs against FP64 and all 33 full logits against same-build stock. QB actually executes the full guards and the same stock pipeline; Q4 records 32 custom dispatches. Prefill, non-target nodes and missing-library controls use stock as specified.

| Measurement | Observed paired result |
| --- | --- |
| Q4 isolated single-dispatch GPU elapsed, Fair cohort | 2.51%-2.78% less time than stock |
| Q4 isolated single-dispatch GPU elapsed, separate post-model Serious cohort | 2.18%-2.44% less time than stock |
| QB complete 32-token decode wall, Serious | 2.71%-3.74% more time than stock |
| Q4 complete 32-token decode wall, Serious | 2.86%-3.76% more time than stock |

These are distinct measurement scopes and thermal cohorts. Current integration has no demonstrated whole-model speedup. Warm snapshot/key construction is a measured host region, but complete CPU/GPU/thermal attribution and normal critical-path share remain unresolved. Detailed p50/p95, paired differences and order controls are in the linked analysis.

The frozen numerical tolerance is `abs(error) <= 2e-4 + 2e-5 * abs(reference)`. Local operator accuracy and stock-relative parity do not establish faithful public-model execution or language quality. The complete trusted-model comparison remains an observed failure with strict exit code **1**. The report retains all relevant negative results.

## Reproduce the numerical PoC

Use Apple Silicon macOS with full Xcode/Metal, CMake and Python 3.9+. The host runs below check arithmetic and rebuildability; submission performance results are labeled by their actual physical device.

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
python3 bitnet/validation/bootstrap.py \
  --qvac-repo /path/to/existing-qvac-repository \
  --work-root /path/to/fresh-clean-validation
```

This rebuilds the pristine pinned backend, runs hand-checkable and seeded-dense cases, and runs the stock TQ2_0 regression. Real steps 1/16/32 require separately captured local inputs and cannot silently degrade to synthetic-only testing. See [sample-I/O and real-input commands](bitnet/validation/RERUN.md).

For iPhone, use [the build guide](bitnet/ios/README.md) and [the physical-phone validation protocol](bitnet/docs/IPHONE_TEST_PLAN_2026-11-16.md). Build a fresh local package and qualify its identity before timing; signed products, source-bound build manifests and old passed gates are not interchangeable. The original shader uses `poc-verify`; current stock/QB/Q4 evaluation has a separate ActQuant-ON gate.

## Package boundaries

The complete pinned framework base is retained for build, vendor, SwiftUI staging, test and license consistency. [Baseline provenance](bitnet/provenance/BASELINE.json) records the existing 63 upstream CI-workflow exclusions; [the upstream README](README.upstream.md) and licenses are preserved. [Our change map](bitnet/docs/CHANGES_FROM_QVAC.md) and [source manifest](bitnet/provenance/SOURCE_MANIFEST.json) identify the overlay.

The current submission tree contains English instructions/report source, necessary source/build/tests, attribution, derived evidence and the PDF. Chinese companions, editable Word copies, obsolete preparation material and independent Mac research tooling are archived locally. Historical commits are retained. Upstream Unicode tokenizer/parser tests are functional source assets and are preserved.

Models, raw tensors/logits/timing samples, captures, compiled frameworks/libraries/apps, device identifiers and signing credentials stay outside Git. Small per-round or correctness-summary tables under `bitnet/analysis` are derived analysis, not raw timing samples. Upstream vocabulary GGUF fixtures and operation support tables are inherited test assets, not our downloaded checkpoint or experiment payloads.

The implementation is AI-assisted; the author remains responsible for reviewing and explaining the code. The repository is public so reviewers can access the source and report directly.
