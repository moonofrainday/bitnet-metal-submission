# FFN performance exploration on iPhone 14 Pro

This appendix consolidates existing physical-iPhone evidence. It adds no device run, shader change, numerical repair or threshold change. The tested public-model compatibility path is now known to omit Microsoft's per-token INT8 ActQuant; these measurements therefore do not establish faithful W1.58A8 inference. ActQuant repair remains separate and paused during this report update.

All results below were measured on **iPhone 14 Pro / Apple A16 GPU, iOS 27.2 (24B119)**, using Xcode 27.2 (27B52), iPhoneOS SDK 27.2, a Release framework and Debug signed test application. They are same-build research comparisons, not a production-app maximum-throughput claim. There are no Mac performance values in this appendix. The public GGUF SHA-256 is `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2`; pinned QVAC is `2db8b046d555316019ae75d1be90484adc8b9330`. The team's checkpoint is a separate dependency.

## Implementations and actual coverage

Gate/up each map 2,560 F32 activations through their own TQ2_0 weights and block scales to 6,912 F32 values. The activation is exactly `ReLU(gate)^2 * up`. Weighted RMSNorm precedes down, which returns 2,560 values. This is not ordinary ReGLU or SwiGLU. Q, Attention, down and residual operations remain stock in the FFN experiments.

| Route | Implementation | Independent whole-FFN dispatches | Real-graph coverage per 32-step history |
|---|---|---:|---|
| S | Actual stock projections, separate ReLU/square/multiply, stock weighted norm and down | 7 | Actual stock reference |
| G | One stock-helper four-row dual gate/up dispatch, one strict activation dispatch, stock weighted norm and down | 4 | All 30 layers: 960 dual + 960 strict-post dispatches |
| K | G's dual projection, one alias-safe in-place activation/weighted-norm dispatch, stock down | 3 | Actual 28 layers: 896 dual + 896 postnorm; layers 2 and 29 safely retain stock |
| F8 | Earlier C8x2 gate/up plus ReLU-squared multiplication in one dispatch, stock weighted norm and down | 3 | All 30 layers: 960 fused dispatches |

The dispatch counts describe the matched **independent whole-FFN harness**, not a constant dispatch count for every normal graph layer. For example, the actual layer-2 graph separates normalization and gamma multiplication. Replay numerically reproduces that layer but does not reproduce its native scheduling.

G reuses the complete stock four-row helper in a separately compiled pipeline; it is not the actual stock PSO and is not C8 dot arithmetic. Its dual dispatch is grid `[864,1,2]`, threads `[32,2,1]`; the two projections remain independent. G's strict activation uses grid `[27,1,1]`, threads `[256,1,1]`. F8 uses grid `[432,1,1]`, threads `[32,2,1]`, with eight rows per SIMD group. K's postnorm uses one 1,024-thread group, 27,776 bytes of queried static threadgroup memory and a device capacity of 32,768 bytes. These are launch/resource observations, not occupancy or bandwidth measurements.

K requires a seven-node encoder slice, exact graph edges and consumers, common X, per-layer weights/scales/gamma, complete type/shape/stride and buffer bounds, and exact aliases: gate equals activation, up equals normalized output, with the other ranges disjoint. Layer 2 rejects the encoder-slice contract; layer 29 rejects the alias contract. Thus requesting all layers does not mean K ran in all 30. A fresh-process missing-K-library control records zero K dispatches and stock-identical logits. Disabled switches and prefill retain stock. All recorded positive custom encodes have matching GPU completions.

## Earlier campaign with direct stock comparisons

Source experiment: `2026-11-12_202159+0900-iphone-candidate-sequence`.

Each percentage range below is the range of **separate round medians of pairwise relative reductions**. It is not a pooled estimate, a ratio of route medians, or a pure fusion-cost decomposition.

| Route versus actual stock S | Whole-FFN GPU reduction, 3 rounds | GPU paired candidate-minus-S median range | GPU wins | Complete 32-token wall reduction, 6 rounds | Decode wins |
|---|---:|---:|---:|---:|---:|
| G | 4.814%-5.070% | -86.667 to -82.250 us | 582/603 | 1.062%-2.664% | 121/126 |
| K | 5.440%-5.675% | -97.375 to -92.333 us | 597/603 | 1.263%-2.724% | 122/126 |
| F8 | 10.018%-10.145% | -172.125 to -170.583 us | 602/603 | 4.333%-5.121% | 126/126 |

The standalone interval starts from the same captured normalized layer-0 step-16 FFN X and ends after actual stock down. It includes all required work in one command buffer per path, with a common alias layout. There is no addition of separately measured kernel medians. The metric is completed **GPU command-buffer elapsed time**, `GPUEndTime-GPUStartTime`, in microseconds, with multiple dispatches per command buffer. Per path there are 30 warmups, 201 alternating pairs per round and three rounds. A/A, S/G, S/K, S/F8 and G/K make 3,015 pairs and 6,030 timed command buffers. Capture, diagnostics and tensor readback are off. All formal operator thermal snapshots are fair; low-power mode is off. Battery/charging bookends were not recorded, so USB connection is not converted into a charging claim.

In that independent FFN interval, direct K-minus-G paired medians are -10.958/-11.833/-14.583 us, with 478/603 K wins and relative reductions 0.679%-0.911%. All AB/BA subgroup medians favor K, but one chronological block favors G. This is a small interval-specific advantage, not a whole-model K-over-G result.

The complete-model interval is synchronized **host wall milliseconds for 32 fixed decode tokens**. Normal CPU encoding, GPU execution and completion are included; model loading, KV clearing and restoration of the same 27-token prefix are excluded. Effective settings are context 512, batch/ubatch 128, six CPU threads, F16 KV, flash attention off and 31/31 Metal offload. Two warmups per route/phase precede sampling. Two histories each have three rounds, four contrasts (AA/SG/SK/SF8) and 21 pairs: 504 qualified pairs and 32,256 token rows. This is direct S-versus-candidate evidence; it is not a same-round G/K/F8 ranking.

Absolute wall time drifts substantially. Primary round 1 starts fair and ends serious; the remaining qualified rounds are serious, charging at approximately 80%, low-power mode off. Every candidate's AB and BA subgroup median favors it in all six rounds. A/A round offsets are smaller than candidate medians, but individual A/A tails are broad. K's primary round-3 p95 worsens to 1,846.999 ms versus stock 1,718.053 ms. Do not claim universal tail improvement.

The original primary console monitor timed out while the phone continued. A partial fetch overlapped original round 3. Under a declared recovery protocol, that **entire 84-pair round remains diagnostic** and an unchanged-V2 replacement supplies qualified logical round 3; no high-tail filtering is used. There are 588 original raw pairs, 504 qualified pairs. Primary uses two processes with model/context reuse within each segment; independent uses one process. These limitations and actual transport/app/parser exit codes remain in the source notes.

## Later direct G versus F8 campaign

Source experiment: `2026-11-13_080244+0900-iphone-independent-gf8`.

This later campaign directly compares **G against F8**, with G/G calibration. It has **no formal actual-stock S contrast**. That limitation does not erase the earlier campaign's separate direct stock comparisons, and the two campaigns must not be subtracted or pooled.

| Direct complete-model contrast | Protocol and scope | Measured result |
|---|---|---|
| G versus F8 | Two histories; 3 rounds/history; 21 alternating pairs/contrast/round; one process/model/context per history | F8 wins 123/126 direct pairs |
| Pairwise relative reduction | Median of each actual pair's relative time change, then separate round medians | 1.961%-3.568% lower for F8 across 6 rounds |
| Paired F8-minus-G wall median | Direct 32-token synchronized host wall time | -105.156 to -32.744 ms |
| Ratio of route p50 values | Separate statistic, not the paired relative median | 2.287%-3.847% lower for F8 |
| Calibration and inventory | G/G plus G/F8; no excluded samples | 252 total pairs; 16,128 token rows; 12 contrast/round summaries |

Every G/F8 AB/BA subgroup and all 24 chronological quartile paired medians favor F8. All G/F8 timed samples are serious-thermal, charging near 90%, low-power mode off; primary G/G begins nominal/fair. Absolute latency drifts strongly, and only 21 pairs support each round's tails. All six observed F8 p95 values are lower, but this does not establish a general p95 or cool-device benefit. The metric includes normal encoding/execution/completion and excludes reset/prefix/model loading. Capture, diagnostic readback, trace and fine CPU timers are disabled. No GPU-counter cause is claimed.

G and F8 change both projection arithmetic/layout and fusion. This is not an isolated pure fusion-cost experiment, and it does not establish a new F8-versus-S percentage for the later binary/campaign.

## Correctness limits that accompany every performance claim

The frozen rule remains `abs(actual-reference) <= 2e-4 + 2e-5*abs(reference)`, with all rows required to pass. Scope, reference and missing fields remain explicit.

| Route | Verified parity and introduced-stage checks | Preserved numerical and model-output boundary |
|---|---|---|
| G | Both histories, all 30 layers; all 33 complete logits are stock-bitwise-identical. Layer-0 steps 1/16/32 raw gate/up and observable FFN boundaries match S. | G's broader complete-FFN FP64/residual-layer acceptance is not established by the initial phone audit. Stock-helper arithmetic and fusion parity do not repair stock's complete-chain failures; unmeasured required fields stay null. |
| K | Both histories, all 896 positive calls/history match independent old-K0 reference on captured inputs. Activation-local and weighted-RMSNorm-local FP64 each pass all 896 cases/history. | Sampled complete-chain FP64 fails: 84 cases/history include 15,353/16,944 failed final-down rows. Stock-relative full logits fail. Strict status 1. |
| F8 | Later completion covers 3,840/3,840 same-input cases across both histories and stock-X/F8-X streams; 11,520 independent replay command buffers complete. F8 and separate C8 activation, normalized/down input and down output are bit-exact; all outputs are finite. | Both stock and F8 fail complete-chain FP64. F8 stock-relative logit/model-output pass remains false (strict 1). Its internal unmaterialized gate/up values remain null. |

For the later independent prompt, stock-X has 890/960 stock and 875/960 F8 full-chain-down failed vectors (173,344 and 132,661 rows); F8-X has 891/960 and 874/960 (172,664 and 132,505 rows). These are same-input comparisons, not a quality ranking or evidence every F8 element is better.

The complete FFN reference starts at the captured normalized FFN X and includes independently decoded gate/up weights, ReLU-squared multiplication, weighted RMSNorm with actual epsilon/gamma, and down. It is not a reference for the full Transformer. Projection-error propagation and actual-input stock-down arithmetic both contribute; activation-rounding-only and normalization controls pass their specified scopes. Those conditional failure counts cannot be added as error shares. An analysis processing exit 0 does not overwrite numerical strict exit 1.

Original stock-relative F8 logit failures persist despite recorded fixed-history greedy agreement. No G/K/F8 language-quality acceptance follows. All these old tests also inherit the now-confirmed missing public ActQuant stage, which must be addressed separately before claiming faithful execution of the public model or transferring rules to the team checkpoint.

## Evidence and reproducibility

`SUMMARY.json` records full-precision aggregate round statistics, recalculated ranges/wins, experiment-relative source paths, sizes and SHA-256 digests. Raw pair/token CSV, tensors, device logs, binaries and personal host/device identifiers are not copied here. The raw source archives remain local.

| Experiment-relative source | What to inspect |
|---|---|
| `2026-11-12_202159+0900-iphone-candidate-sequence/SHADERS.md` | Actual shader entry points, ABI, arithmetic attribution and launch |
| Same experiment: `PHONE_CORRECTNESS.md` | Real dispatch/completion, K safety fallback, local versus full-chain scope and strict flags |
| Same experiment: `OPERATOR_RESULTS.md` and `analysis/operator-performance/summary.csv` | Whole-FFN GPU interval, direct stock and G/K comparisons, all 15 aggregate rows |
| Same experiment: `DECODE_RESULTS.md` and `analysis/decode-final/summary.csv` | Earlier direct stock wall comparisons, all 24 rows, recovery and process boundaries |
| `2026-11-12_231133+0900-iphone-f8-numerics/README.md` | Primary-prompt same-input and propagation findings before independent completion |
| `2026-11-13_080244+0900-iphone-independent-gf8/README.md` | Completed 3,840-case same-input numerical scope and retained failures |
| Same later experiment: `PERFORMANCE.md` and `analysis/direct-GF8/summary.csv` | Later direct G/F8 wall contrast, all 12 rows, thermal/order limits |

The local notes' existing CPU analyzers provide the original reproducibility commands. This curation only re-reads their aggregate CSVs, verifies source hashes against the preserved evidence index where available, and calculates min/max ranges and summed wins. It performs no new GPU test and changes no selection, threshold or raw data. Keep stock default; retain G as a stock-output-compatible research alternative and F8 as a faster measured research route with explicit numerical/semantic limitations. K remains a guarded in-place normalization experiment.
