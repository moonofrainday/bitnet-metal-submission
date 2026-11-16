# A16 stock QB Q4 attribution results

Physical iPhone 14 Pro / Apple A16 GPU, iOS27.2.1. Native source `14a9f103c23b5a24bd0da1e0dbc5a0d5a37ade3a`; QVAC base `2db8b046d555316019ae75d1be90484adc8b9330`. Both histories use27 prefill tokens and32 fixed continuation IDs, ActQuant ON, F16 KV, context512, batch/ubatch128,6 CPU threads and flash attention OFF. Only layer0 Qcur-0 is selected.

**Decision:** keep actual stock as the default. Q4 remains a correct stock-helper-derived single-projection specialization with a small isolated GPU benefit. The current guarded package is slower in the measured complete decode scope; no whole-model acceleration or faithful official-model/quality acceptance is claimed.

## Scope and routes

| Route | Checks and actual execution |
| --- | --- |
| S | Actual stock `kernel_mul_mv_tq2_0_f32`;4 rows/SIMD,2 groups/threadgroup,grid320,threads32x2 |
| QB | Full candidate node/layout/buffer/library checks; same-process actual stock PSO,120-byteABI, bindings, offsets, grid and threads;0 custom dispatches |
| Q4 | Same guarded scope; `tq2_0_matvec_stock4x1_v1`,stock-helper arithmetic,4 rows/SIMD,1 group/threadgroup,grid640,threads32x1 |

The original authored4x2 PoC and C8x2 research are separate implementations. This is not a whole custom Transformer block, all-layer Q deployment or FFN change.

## Correctness and prerequisite gates

- Actual loader/cache contract18/18, repeated18/18 in a second fresh process; incorrect entry/library/geometry and cache controls retained.
- ActQuant20cases,386,567values: exact oracle agreement, finite outputs, unchanged inputs and GPU completion.
- Each history: all32N=1 Q outputs (81,920values/route) pass independent FP64 at `abs(error)<=2e-4+2e-5*abs(reference)`. Maximum absolute errors: primary1.2679465726e-5; independent1.2671500372e-5. Zero failing/nonfinite values.
- QB32 full guards/32 targetstock/0 custom, Q432 custom per history. Same-process S/QB pipeline, ABI and live bindings checked.
- All33 full logits and saved Q/ActQuant/layer-output snapshots match same-build stock byte-for-byte. The27-token prefill Q is parity-only; the32single-token Q outputs are FP64-checked. Capture/plain and callback-null neutrality and missing-library fallback pass.31/31layers offload.
- This scoped pass does not repair the historical complete trusted-reference model failure. Strict comparison exit1 is preserved; quality and complete-model FP64 remain unverified.

## Formal complete decode wall time

Two complete replay warmups/route; three rounds x21pairs/contrast/history, A/A plus S/QB and S/Q4. Same model/context, KVclear and identical prefix restoration outside the timer; balanced AB/BA, rotated contrast blocks, per-token synchronization unchanged. Capture, diagnostic metrics and tensor readback OFF. Across two histories:378pairs,756timed paths,24,192token rows. Every formal stratum is Serious with low-power mode OFF; no temperature/frequency constancy is inferred.

Each contrast keeps its own stock baseline. Times are milliseconds; differences are medians of individual paired differences, not differences of marginal medians.

| History | Round | Contrast | A p50 / p95 ms | B p50 / p95 ms | Paired B-A ms | Paired change % | B wins /21 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| primary | 1 | AA | 2535.581 / 2666.600 | 2531.255 / 2679.830 | -7.101 | -0.281 | 13 |
| primary | 1 | SQB | 2525.289 / 2568.726 | 2616.669 / 2646.383 | +68.294 | +2.714 | 0 |
| primary | 1 | SQ4 | 2456.068 / 2509.500 | 2527.487 / 2578.849 | +68.844 | +2.858 | 1 |
| primary | 2 | SQB | 2367.436 / 2426.333 | 2464.360 / 2493.558 | +87.834 | +3.738 | 0 |
| primary | 2 | SQ4 | 2324.828 / 2409.873 | 2415.522 / 2487.857 | +87.130 | +3.759 | 1 |
| primary | 2 | AA | 2324.582 / 2341.150 | 2329.363 / 2374.274 | +2.191 | +0.094 | 9 |
| primary | 3 | SQ4 | 2311.729 / 2341.017 | 2381.225 / 2437.982 | +80.481 | +3.447 | 0 |
| primary | 3 | AA | 2389.822 / 2415.905 | 2394.660 / 2428.646 | +4.961 | +0.207 | 7 |
| primary | 3 | SQB | 2420.879 / 2491.471 | 2513.408 / 2545.520 | +67.963 | +2.800 | 0 |
| independent | 1 | AA | 2573.011 / 2803.386 | 2587.516 / 2771.559 | -3.167 | -0.124 | 13 |
| independent | 1 | SQB | 2603.999 / 2685.455 | 2697.339 / 2730.913 | +94.345 | +3.621 | 1 |
| independent | 1 | SQ4 | 2633.823 / 2683.142 | 2725.204 / 2769.471 | +76.634 | +2.922 | 0 |
| independent | 2 | SQB | 2667.072 / 2743.613 | 2760.507 / 2872.998 | +84.386 | +3.142 | 1 |
| independent | 2 | SQ4 | 2646.690 / 2822.090 | 2734.978 / 2811.872 | +87.702 | +3.314 | 2 |
| independent | 2 | AA | 2713.082 / 2915.383 | 2707.877 / 2893.915 | -3.152 | -0.117 | 12 |
| independent | 3 | SQ4 | 2690.501 / 2767.316 | 2769.761 / 2903.330 | +76.790 | +2.858 | 1 |
| independent | 3 | AA | 2696.299 / 2709.262 | 2700.117 / 2712.242 | +3.312 | +0.123 | 6 |
| independent | 3 | SQB | 2654.335 / 2714.620 | 2748.812 / 2802.862 | +76.822 | +2.922 | 1 |

All SQB/SQ4 AB/BA strata and chronological blocks remain slower. Same-shader QB regression is an observed integration-path effect in this package, not a pure shader penalty. A/A paired medians range-7.101to+4.961ms. The main positive differences exceed those A/A observations, but serious-state endpoints are coarse: thermal/preemption/clock effects are not fully separated.

## Isolated same-fixture GPU measurements

One dispatch per command buffer, completed `(GPUEndTime-GPUStartTime)*1e6`, microseconds.30 warmups/path,3rounds x201alternating pairs. No diagnostics, readback or capture in the timing interval. This excludes the guarded route, ActQuant, model preparation and graph scheduling. Every run freshly checks S/C8/Q4 on steps1/16/32 (9cases,23,040values); selected S/Q4 outputs are byte-equal with0 FP64failures. CQ in the fixed harness is extra C8/Q4 research, not S/Q4.

| Cohort | Round | Stock p50 / p95 us | Q4 p50 / p95 us | Paired Q4-S us | Paired change % | Q4 wins /201 |
| --- | --- | --- | --- | --- | --- | --- |
| initial Fair | 1 | 225.333 / 229.583 | 219.250 / 222.875 | -5.667 | -2.509 | 196 |
| initial Fair | 2 | 225.083 / 229.375 | 219.333 / 222.875 | -5.833 | -2.596 | 200 |
| initial Fair | 3 | 225.125 / 229.000 | 218.625 / 222.250 | -6.250 | -2.775 | 199 |
| post-model Serious | 1 | 226.125 / 230.375 | 220.125 / 224.125 | -5.375 | -2.377 | 198 |
| post-model Serious | 2 | 226.042 / 230.000 | 221.000 / 225.292 | -4.917 | -2.184 | 184 |
| post-model Serious | 3 | 225.250 / 229.458 | 219.375 / 224.458 | -5.458 | -2.441 | 194 |

Initial endpoints: Fair, charging50%, low-power OFF; post-model endpoints: Serious, charging70%, low-power OFF. Cohorts are not pooled. Each isolated campaign is about3seconds of timed dispatch work after qualification, not continuous long-duration throughput or energy evidence. The latter preserves the sign of the local benefit after the sustained model campaign; it does not cancel the complete-decode slowdown.

## Nested CPU diagnostics and attribution limit

| History | Route | Guard total ms /32 calls | Getter ms | Snapshot plus key ms | Getter p50 /p95 us | Cache hits/misses |
| --- | --- | --- | --- | --- | --- | --- |
| primary | QB | 96.083 | 95.621 | 95.245 | 3177.708 / 3379.140 | 32/0 |
| primary | Q4 | 68.328 | 68.003 | 67.691 | 2283.021 / 3384.216 | 32/0 |
| independent | QB | 95.828 | 95.385 | 95.032 | 3161.542 / 3390.229 | 32/0 |
| independent | Q4 | 95.698 | 95.270 | 94.931 | 3189.833 / 3453.777 | 32/0 |

Source: `ggml-metal-device.m` external library getter takes the shared library lock before file snapshot/key construction; `q_contract.h` performs the warm realpath/stat/attribute snapshot. Snapshot-plus-key dominates the measured nested getter region. No warm file reads, PSO loads or cache misses occurred.

These are worker elapsed intervals including waits/preemption, not pure CPU cycles or additive critical-path components. The32selected-Q lock waits do not measure other selectors blocked on that shared lock. The combined snapshot/key interval does not isolate realpath alone. Instrumentation ON/OFF effects change sign and baseline wall drifts; no universal instrumentation correction is applied. Natural64CBs/64tasks are retained. Their mixed operation intervals cannot assign normal per-dispatch critical-path share.

One bounded30s Metal System Trace attempt ended after1.362s with `Device disconnected` before recording readiness; no model was launched in that attempt. CLI exit0 is not accepted as usable trace evidence. A subsequent devicectl check still showed the phone connected. Normal GPU Q-share, hardware counters, bandwidth, occupancy, registers, energy and CPU/GPU/thermal decomposition remain unavailable/unverified.

## Next controlled experiment

Preserve all18contract/cache negative checks while separately testing whether a read-only file-identity snapshot/key can safely be constructed outside the shared pipeline-cache lock or cached at a validated lifecycle boundary. Change one factor, requalify both histories and repeat S/QB/Q4 in one build. This is a proposed follow-up, not an implemented fix or a claimed explanation for all regression.

## Evidence and reproduction

`SUMMARY.json` contains only derived aggregates, stable thermal strata, order/time blocks, identities and relative evidence filenames with SHA256. No raw tensor/logit/token/timing rows, device IDs, signing material, private paths, signed products, models or traces are uploaded. A separate optional post-model operator summary is retained when present.

CPU-only reanalysis of the retained local evidence:

```sh
python3 bitnet/tests/summarize_iphone_q_attribution.py \
  --experiment "$LOCAL_EVIDENCE" --output "$FRESH_ANALYSIS"
```

Exit0 means complete analysis evidence, not speedup or model-quality acceptance. An observed required correctness failure exits1; missing/malformed/stale evidence exits2. Raw data are required for reanalysis and remain on the author's Mac.

For a fresh phone campaign use `bitnet/docs/IPHONE_TEST_PLAN_2026-11-16.md`. Set local device, model, fixtures, token and package variables, run the scoped correctness gate first, then the isolated benchmark and diagnostic/overhead/formal planned commands serially. Run prepared-product checks from the original source checkout bound into its compile provenance; a different clone requires a clean package build. Exact executed argv/cwd/exit codes are preserved only in the local command indexes.

Host-only packaging repair: `bitnet/actquant/run_phone.py` now supports both relative package import and direct script import of q4_contract.2 CPU subprocess regression checks pass. The measured native app/framework/metallib were unchanged and launched from the frozen source using an explicit PYTHONPATH. This host repair is not a new phone-binary validation.
