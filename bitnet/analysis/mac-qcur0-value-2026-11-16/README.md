# Existing Qcur-0 kernel: scoped Mac value and integration cost

Q4 improves the isolated selected GPU operator by **8.30%–8.36%** in three rounds, but does **not** improve the measured 32-token model wall. Keep stock as the default. Q4 remains a correct, scoped PoC. Measurable host costs accompany its integration, but these measurements do not establish a complete causal decomposition of model-wall time. This is an evaluation of the already selected kernel, not a ranking of new algorithms.

## Scope and provenance

The public BitNet 2B TQ2_0 model runs on an M2 Pro with 16 GB unified memory, macOS 27.2.1 (26B131), Xcode 27.2 (27B52), macOS SDK 27.2. ActQuant is ON and K/V are F16 throughout. This campaign makes no new A16/iPhone claim. The official-source whole-model semantic/numerical acceptance remains unresolved; the historical strict comparison exit **1** is preserved. Scoped parity and local FP64 checks do not override it.

- Pinned QVAC: `2db8b046d555316019ae75d1be90484adc8b9330`.
- Lab baseline: `578ec123ec2f0eed01f3edf6dc09ac6dcbe6104a`; frozen native preparation: `14a9f103c23b5a24bd0da1e0dbc5a0d5a37ade3a`.
- Model SHA-256: `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2`.
- Runner SHA-256: `f6ae6e0ce4df36267a8ebcbe81afad61d93f1734d0bfa1d8fdac50cc9abab486`; the Mac gate binds 1,073 source files and the actual products.
- Q4 metallib SHA-256: `5a3f9dac0e21394f579e90d2af7fa13b3a2a85dd761801c827ccf0b594156347`.
- W SHA-256: `b2dea7f42cacfd08ed780df979f6f28240026795b1f5c87ca8c16fc7da23dc58`.
- Actual post-ActQuant step 16 X SHA-256: `faca9f62eb02a1da3d87cf39df20816a4a15b79806995b4225df68b008e0cbcb`.

| Route | What executes | Launch for 2,560 outputs |
|---|---|---|
| S | Actual `kernel_mul_mv_tq2_0_f32` | 4 rows/SIMD, 2 groups, grid 320, threads 32×2 |
| QB | Full candidate checks/getter, then the same actual stock PSO, ABI, bindings and launch | Same as S; zero custom dispatches |
| Q4 | `tq2_0_matvec_stock4x1_v1`, reusing stock-helper four-row arithmetic | 4 rows/SIMD, 1 group, grid 640, threads 32×1 |

Q4 is a stock-derived launch specialization, distinct from the original self-written 4×2 PoC and C8×2. Only layer 0 `Qcur-0` is replaced. Prefill and other nodes remain stock. Each replay has 32 selected Q dispatches and 960 all-layer Q identities. The model has 210 TQ2_0 projection weights, including 30 Q weights and 60 square 2560×2560 weights. These are **structural counts**, not measured time fractions or qualification of other nodes. Normal-schedule Q critical-path share remains **null**.

## What justifies this selected target

It is an actual ternary-weight projection in the requested transformer inference path, with a concrete single-token contract and a bounded integration/fallback point. It supports a reproducible single-layer matrix-kernel PoC; the measured local gain and structural presence of related weights support that limited choice. They do not prove that this projection dominates normal decode or deserves the highest system-optimization priority.

For Amdahl reasoning, normal critical-path fraction `f` is still unknown. A purely conditional budget illustrates the small current replacement scope: **if** the isolated saving of about 1.958 us transferred unchanged to each of 32 Q0 calls, with no change in overhead or overlap, the gross saving would be about 62.7 us, approximately 0.0145% of a 432 ms replay. That arithmetic is neither a measured end-to-end saving nor a guaranteed upper bound; normal scheduling, cache state and shared resources can change. It cannot be multiplied by 60/210 tensors without separate qualification and measurements. Current whole-model results, below, are the actual decision evidence.

## Correctness and comparison validity

The same-process two-history gate verifies S/QB stock PSO identity, exact stock ABI/launch/live bindings, all 32 actual target calls, local actual-input ActQuant checks, all 32 N=1 Q FP64 comparisons, saved residuals at steps 1/16/32 and all 33 full logits. Q4 matches stock-ON bitwise in the qualified scope. QB executes 32 full guards, 32 stock target dispatches and zero custom dispatches; Q4 executes 32 full guards and 32 custom dispatches. Missing-library fallback and diagnostic neutrality remain covered.

The independent isolated-operator check uses real steps 1/16/32, both S and Q4, all 2,560 rows: **15,360 values, 0 FP64 failures, 0 nonfinite values, 0 captured-stock bit differences**, with W/X immutability, guard canaries and GPU completion verified. The frozen bound is `abs(error) <= 2e-4 + 2e-5 * abs(reference)`. Maximum absolute error is 1.0247051e−5 (step 1), 8.8373527e−6 (step 16), 1.1220904e−5 (step 32), identical for S/Q4. This is local dot accuracy, not full-model FP64 acceptance.

## Isolated real-input GPU operator

One actual dispatch per command buffer; read `GPUEndTime-GPUStartTime` after completion. Each path has 30 warmups; each round has 201 alternating AB/BA pairs; 3 rounds each of A/A and S/Q4. Timed capture, logs and tensor readback are off. W/X are reused actual step 16 inputs. Guard/cache checks, ActQuant preparation, loading, CPU encoding and model scheduling are excluded. Block endpoints report nominal thermal state and low-power mode off.

| Round | Contrast | A p50/p95 µs | B p50/p95 µs | Paired B−A median µs | Paired relative median | B wins/201 |
|---:|---|---|---|---:|---:|---:|
| 1 | AA | 23.5417/27.7083 | 23.5417/24.1250 | +0.000000 | +0.0000% | 100 |
| 1 | SQ4 | 23.5000/24.0833 | 21.5000/21.9167 | -1.958324 | -8.3333% | 198 |
| 2 | SQ4 | 23.5000/23.8750 | 21.5417/22.0833 | -1.958324 | -8.3038% | 199 |
| 2 | AA | 23.5000/24.0000 | 23.5000/23.9167 | +0.000000 | +0.0000% | 94 |
| 3 | AA | 23.5000/24.0417 | 23.5000/24.0000 | -0.041662 | -0.1764% | 103 |
| 3 | SQ4 | 23.5000/23.8750 | 21.5000/22.1250 | -1.958339 | -8.3629% | 196 |

Q4 saves approximately 1.958 µs per isolated dispatch, with 196–199 wins/201. A/A paired medians are 0, 0 and −0.041662 µs. The 8.30%–8.36% figures are medians of paired relative differences, not ratios of separately pooled medians. Hot input reuse and isolated scheduling limit extrapolation.

## Formal fixed 32-token decode wall

One model/context per history; 2 full warmup replays per route; KV clear and identical 27-token prefix restoration before each path, outside the wall interval. The wall includes 32 N=1 `llama_decode` calls and existing per-token synchronization. Context 512, batch/ubatch 128, 6 CPU threads; flash attention off. Three rounds × 21 pairs/contrast, rotated AA/SQB/SQ4 blocks and balanced AB/BA. Trace, readback, capture and backend metrics are off. All final-logit hashes are checked outside the timer. No new synchronization was added.

| History | Round | Contrast | A p50/p95 ms | B p50/p95 ms | Paired B−A median ms | Paired relative median | B wins/21 |
|---|---:|---|---|---|---:|---:|---:|
| primary | 1 | AA | 431.8482/437.5914 | 431.5478/439.2638 | -0.187374 | -0.0434% | 12 |
| primary | 1 | SQB | 432.1112/447.5651 | 433.6790/436.8030 | +1.448124 | +0.3354% | 3 |
| primary | 1 | SQ4 | 432.7482/444.7362 | 433.9485/435.6794 | +1.200375 | +0.2774% | 5 |
| primary | 2 | AA | 432.9682/434.7202 | 432.4431/434.4656 | -0.254333 | -0.0584% | 14 |
| primary | 2 | SQB | 432.4142/441.5483 | 433.6626/436.1395 | +0.980000 | +0.2267% | 8 |
| primary | 2 | SQ4 | 432.2269/433.2256 | 433.2877/440.9279 | +1.074417 | +0.2490% | 3 |
| primary | 3 | AA | 431.8466/443.7660 | 431.6578/432.2054 | -0.444792 | -0.1032% | 14 |
| primary | 3 | SQB | 431.4040/440.0227 | 433.0423/434.8365 | +1.541458 | +0.3570% | 4 |
| primary | 3 | SQ4 | 432.9999/447.9428 | 434.4429/441.9273 | +1.492416 | +0.3447% | 6 |
| independent | 1 | AA | 431.9293/437.8436 | 431.5824/439.2016 | +0.333125 | +0.0771% | 8 |
| independent | 1 | SQB | 432.2159/433.1889 | 433.2462/439.4432 | +1.328000 | +0.3085% | 2 |
| independent | 1 | SQ4 | 432.7211/438.9908 | 433.1976/434.6598 | +0.103959 | +0.0241% | 10 |
| independent | 2 | AA | 433.1684/435.0740 | 432.6695/434.2262 | -0.521709 | -0.1204% | 15 |
| independent | 2 | SQB | 419.3163/442.9831 | 423.6232/454.3741 | +3.378500 | +0.8013% | 6 |
| independent | 2 | SQ4 | 419.4474/424.3732 | 421.4052/427.4733 | +1.957792 | +0.4668% | 3 |
| independent | 3 | AA | 432.4800/446.8074 | 432.4665/433.1926 | +0.048542 | +0.0112% | 10 |
| independent | 3 | SQB | 432.5987/433.7798 | 434.0867/438.9128 | +1.514833 | +0.3499% | 1 |
| independent | 3 | SQ4 | 432.4459/434.9418 | 433.1199/436.4239 | +0.750833 | +0.1734% | 5 |

SQ4 paired medians are positive (slower) in 6/6 rounds: **+0.104–+1.958 ms**, or +0.024%–+0.467%. SQB is also slower in 6/6: +0.980–+3.379 ms. This campaign establishes no full-model improvement. The smallest differences overlap A/A variation; the independent-history second round moves toward 419 ms from roughly 432 ms nearby. Whole-model thermal/low-power observations are unavailable. Retain per-round/order/time-block results instead of pooling away drift; this is not a stable-thermal regression estimate.

`SUMMARY.json` retains compact AB/BA and time-block medians. SQ4−SQB from separate blocks is not a pure shader measurement: bindings/encoding/scheduling and input-cache conditions also differ. The isolated GPU saving cannot be directly subtracted from model-wall medians.

## Measured host regions and attribution limits

The separate instrumented diagnostic has 32 full guard passes and 32 warm cache hits for QB/Q4, with zero misses, loads or failed paths. The getter includes file canonicalization/stat/identity-key validation. Its snapshot/key region is the dominant measured host region in these samples. The host intervals include potential preemption and measurement overhead; nested regions cannot be added.

| History/route | Guard total ms | Getter total ms | Tensor/buffer total ms | Snapshot/key total ms | Getter p50/p95 µs |
|---|---:|---:|---:|---:|---|
| primary QB | 1.193621 | 1.135084 | 0.056163 | 1.109212 | 34.958/47.071 |
| primary Q4 | 1.096088 | 1.039167 | 0.054919 | 1.014582 | 31.896/39.961 |
| independent QB | 4.361205 | 4.185085 | 0.166580 | 4.060374 | 92.333/287.555 |
| independent Q4 | 1.345125 | 1.281959 | 0.059917 | 1.244375 | 37.041/56.150 |

Primary QB snapshot/key p50/p95 is 34.292/45.948 µs; lock-wait totals 0.333 µs and load time is 0. This points to warm identity-snapshot/key work rather than repeated compilation or lock contention in those samples. The independent QB interval is disturbed: all native encode elapsed totals 102.758 ms, versus 33.229 ms in primary QB and 34.177 ms in independent Q4. This prevents assigning all wall variation to guards.

The separate metrics OFF/ON campaign is intrusive: all 18 route/round medians increase by 0.498–6.814 ms. Formal results therefore use metrics OFF. Each token has two natural CBs in the diagnostic; interval-union analysis avoids overlap double-counting. Host encoding and GPU timelines use distinct clock domains. Existing CB timing does not isolate the selected dispatch or prove its normal-schedule critical-path contribution. The old pre-hardening same-shader “no measurable overhead” result remains historical; current file-aware getter code has a different measured cost.

The historical phone guard-control difference of roughly 10-13 ms remains a separate observation. This Mac source profile, host diagnostics and thermal evidence do not retrospectively decompose that earlier phone result. At this campaign's cutoff, the prepared same-profile A16 controls had not run. Their later physical-device results are reported separately in the [current A16 analysis](../iphone-qcur0-attribution-2026-11-16/README.md).

## Separate selected-encoder timestamp probe

A four-file patch in an isolated Mac worktree adds a selected S encoder with explicit prefix/scope/suffix fences and stage-boundary timestamps. It is **not applied to the main or phone native build**. Primary OFF/ON have 199 saved payloads bitwise equal; primary OFF/ON and independent ON scoped gates pass. All 270 follow-up paths retain the stock final-logit hash. The CPU audit validates 29 selected paths/928 scopes, the actual stock PSO/120-byte ABI/all 5 buffer slots/geometry, exact steps 1–32, unique completed natural-CB joins and calibrated raw GPU timestamps.

| Diagnostic path | Mode | Dispatches/scope | Scopes | CBs | Scope p50/p95 µs | Sum µs |
|---|---|---:|---:|---:|---|---:|
| diagnostic-1/path-7 | target | 1 | 32 | 64 | 24.791500/25.816850 | 798.505000 |
| diagnostic-1/path-8 | target | 1 | 32 | 64 | 24.666500/25.540100 | 792.794000 |
| diagnostic-1/path-9 | target | 1 | 32 | 64 | 24.666500/25.120800 | 790.291000 |
| diagnostic-1/path-11 | target | 1 | 32 | 64 | 24.459000/25.495350 | 788.083000 |
| diagnostic-empty/path-7 | empty | 0 | 32 | 64 | 0.541000/0.579350 | 17.415000 |
| diagnostic-empty/path-8 | empty | 0 | 32 | 64 | 0.541000/0.542000 | 16.957000 |
| diagnostic-empty/path-9 | empty | 0 | 32 | 64 | 0.500000/0.560450 | 16.711000 |
| diagnostic-empty/path-11 | empty | 0 | 32 | 64 | 0.500000/0.542000 | 16.627000 |

Target scopes include explicit fence wait/update, encoder boundaries and counter sampling. Empty scopes contain zero dispatches; actual Q executes afterward. They are separate controls and are not silently subtracted. The raw-GPU bracket conversion is independently checked; no absolute host/GPU clock mixing occurs.

| Round | Metrics-only/no-split S paired overhead ms | Metrics+sampled-split S paired overhead ms |
|---:|---:|---:|
| 1 | +1.706667 | +3.027209 |
| 2 | +1.392333 | +3.047917 |
| 3 | +1.616416 | +2.737958 |

These sequential campaigns share the same probe build but do not directly pair selector 0 with 1. Do not subtract their medians as pure encoder/fence cost. The probe confirms small, measurable selected encoder intervals and intrusive diagnostic overhead; **normal-schedule Q time share remains unverified**.

## Decision and next bounded check

Keep stock as the default. Retain Q4 and its correctness proof as the scoped PoC optimization. Do not claim end-to-end, power, A16, full-model semantic equivalence or a hardware bandwidth/occupancy bottleneck. No new algorithm was selected or implemented in this campaign.

The next justified implementation question is whether immutable, session-bound pipeline planning can avoid repeated warm file-identity snapshot work while preserving the existing path-change/wrong-library/geometry/cache-invalidation safety tests. That is a proposed optimization, not a result here. The later [A16 qualification and measurement](../iphone-qcur0-attribution-2026-11-16/README.md) has its own actual results; the Mac gain is not carried over as a phone result.

Historical supplemental controller commit: `5a8b534a289c294a50ad2f2718f5d07fe2a2f78f`. The separate probe source commit is `6c775b38cc0283ba126da3dc1890bfd9a2a6b864`; its patch SHA-256 is `d140f98f968fb58e93e3c4efcf75f9ba02f03ee78c0457447540c552c39e4ebd`. The Mac-specific controllers, probe patch and detailed probe instructions are retained in the author's local archive, rather than the focused phone submission. The probe was never part of the normal or iPhone native build.

## Historical evidence and local archive

Raw logs, tensors and per-pair/per-step/per-row data remain local. This directory publishes only code-linked analysis and compact aggregates. Artifact names below are relative to local experiment `mac-qcur0-value-2026-11-16_152746+0900`; they are not bundled in Git. Their hashes are indexed in `SUMMARY.json`.

| Evidence | Local relative artifact |
|---|---|
| Frozen model/products/two-history same-process gate | `campaign/MAC_GATE.json` |
| Formal samples/config/results | `campaign/formal-{primary,independent}/{RUN.json,samples.csv,token-times.csv}` |
| Instrumentation controls | `campaign/overhead-{primary,independent}/` |
| Host diagnostic and CPU archive analysis | `campaign/diagnostic-{primary,independent}/`, `fresh-cpu-analysis-{primary,independent}/SUMMARY.json` |
| Isolated GPU benchmark and qualification | `operator-build/{BUILD.json,PLAN.json}`, `operator-build/run/{SUMMARY.json,qualification.json,samples.csv,correctness.csv}` |
| Structural projection counts | `SELECTED_KERNEL_REUSE.json` |
| Separate encoder probe | `profiling-feasibility/encoder-probe/{SOURCE-v2.json,PROBE_FINDINGS.json}`, its `runtime/` and `followup/` |

The historical supplemental controllers required Python 3.10+ and NumPy and used the original project analysis environment. The controller failure-status correction was checked by a CPU-only negative control after the successful campaign; it changed no native binary, shader, successful command or metric.

Those Mac-only controllers and the intrusive encoder-probe tooling are deliberately not part of this focused submission. Exact original commands, their source snapshots and raw evidence remain in the author's local archive under the experiment identity above. The entries in `SUMMARY.json` identify historical evidence and hashes; they are not dependencies that the current phone runner loads. No executable command in this document requires a removed Mac controller.

For the submitted phone reproduction, use the current [A16 analysis and procedure](../iphone-qcur0-attribution-2026-11-16/README.md). Native execution 0, scoped numerical/stock acceptance 0, and the historical official-model strict 1 remain distinct outcomes.
