# Profiling priority, matched guard control, and numerical acceptance

This appendix separates measured A16 behavior, arithmetic diagnostics and unverified model semantics. Performance below concerns the current public-checkpoint QVAC compatibility path. It does not demonstrate faithful execution of Microsoft's per-token INT8 activation quantization. The team checkpoint remains a separate dependency. Raw tensors, logits, traces and timing samples stay local; this directory contains analysis and portable reanalysis code.

## 1. Scope and priority of the current optimization

Q4 specializes layer-0 Q projection for single-token decode. It does not implement an entire Transformer block. The fresh isolated A16 measurements found approximately 6-7 us lower GPU command-buffer elapsed time per target dispatch, corresponding to paired reductions of 2.68%-3.11%. The normal 32-token decode campaigns did not show a model-speedup result. An isolated improvement should therefore be presented as a narrow specialization result, not a general inference-speed improvement.

The launch comparison is informative:

| Path | Threadgroups | SIMD groups/threadgroup | Rows/SIMD group | Total SIMD groups | Total threads |
|---|---:|---:|---:|---:|---:|
| Actual stock | 320 | 2 | 4 | 640 | 20,480 |
| Q4 | 640 | 1 | 4 | 640 | 20,480 |

Q4 does not increase per-SIMD activation reuse. It changes threadgroup granularity, and its fixed-shape wrapper changes ABI/compiler context relative to the native stock kernel. These are coupled differences in the measured implementation. Higher occupancy, lower register pressure, bandwidth saturation and ALU utilization have not been measured; none should be reported as the established reason for the improvement. AIR describes compiler output, not runtime stalls or hardware occupancy.

The A16 capture retained 725 dispatches in three command buffers, but no reliable per-dispatch time distribution. The available profiler reported `GPU profiling requires Apple M3 / A17 Pro or later`. Hardware bandwidth, occupancy, stalls and utilization therefore remain **unavailable**. This is an instrumentation limit, not evidence that any particular resource is the bottleneck.

## 2. The output head is a measurement priority, not a proven bottleneck

The GGUF contains a tied F16 embedding/output weight:

`128256 * 2560 * 2 = 656,670,720 bytes = 656.67072 MB = 626.25 MiB`.

That is **54.560524% of the 1,203,563,808-byte GGUF file**, or **54.918215% of its tensor payload**. The graph confirms an actual decode `result_output` matmul using F16 W `[2560,128256]`, F32 X `[2560,1]` and F32 Y `[128256,1]`. The tensor is counted once in stored unique parameters; backend placement can still introduce runtime mappings or copies. Stored bytes are not measured DRAM traffic, and neither stored bytes nor parameter count establishes latency share.

The next operator-priority study should include output projection, attention/KV, FFN, norms and other graph work. Measure normal full decode wall separately from diagnostic GPU ranges. If introducing command-buffer boundaries changes scheduling, label those timings diagnostic. A cached isolated head benchmark cannot directly establish its share of normal model critical-path time.

Amdahl's law is useful only with an explicitly measured or assumed fraction. If a target accounts for fraction `f` of original critical-path latency and its local time falls by fraction `r`, idealized total time becomes `T*(1-f*r)` and speedup is `1/(1-f*r)`, excluding added overhead and scheduling changes.

| Assumed critical-path fraction f | Assumed local reduction r | Idealized total latency reduction |
|---:|---:|---:|
| 1% | 3% | 0.03% |
| 5% | 3% | 0.15% |
| 10% | 3% | 0.30% |
| 50% | 3% | 1.50% |

These are conditional examples, not measured predictions. No runtime fraction is inferred from the 54.56% file share or from layer counts.

## 3. New physical-device guard-only control

A matched A/B/C control was implemented in a separate worktree:

- **S:** actual native stock dispatch.
- **QB:** perform the existing full Q4 guard, including candidate-pipeline lookup and capability checks, then return before encoder mutation and continue through native stock dispatch.
- **Q4:** the existing custom four-row/one-SIMD specialization.

QB checks actual node/weight identity, selected layer, single-token decode, no add/norm, types, shapes, strides, no views, live buffers, alignment, bounds, disjoint ranges and cached candidate capability. It does not consume nodes, bind the candidate, change encoder boundaries or dispatch a custom kernel. A rejected guard would be safe fallback behavior but an INVALID experimental B control.

The first phone attempt failed with SIGABRT on first QB diagnostic readback: a backend diagnostic allowlist still accepted only S/Q4. The failure and its source/binary record remain local. A minimal route-validation fix allowed QB and checked mutually exclusive route/flag identity; no shader arithmetic changed. Six host evidence-rejection tests and sixteen compiled tests of the actual backend readback-route function passed before rebuilding. The second physical-device run completed with host/app exit0.

Independent CPU audit of the fetched raw files and logs established:

| Check | Result |
|---|---|
| Device / OS | iPhone 14 Pro, Apple A16 GPU, iOS27.2.1 |
| Actual layer offload | 31/31 |
| QB successful full guard checks | 32/32 decode calls; zero in prefill |
| S / QB stock Q dispatches completed | 960 / 960 |
| QB custom Q dispatches | 0 |
| Q4 custom / stock Q dispatches completed | 32 / 928 |
| Native S/QB pipeline | Same actual compiled PSO and cached pipeline handle at all32 target calls |
| Native ABI and live bindings | Same120-byte argument ABI, six slots, W/X/Y/residual/norm bindings and offsets |
| Native launch | Same grid `[320,1,1]`, threads `[32,2,1]` |
| Captured W/X/Y | All32 target calls bitwise equal S/QB/Q4 |
| Residual checkpoints | Steps1/16/32 bitwise equal |
| Full logits | All33 vectors, 128,256 values each, bitwise equal across routes |
| Diagnostic neutrality | Each route's readback run matches its probe-off repeat |

The audit checked68 captured files per route and514,560 finite readback F32 values. It independently compared bytes rather than relying only on the app's boolean summary. New FP64 computation was outside this matched-control audit: `fp64_pass=null`, strict V1 exit2 remains appropriate for missing required checks. The control's success is not a claim of full FFN FP64 or original-model semantic acceptance.

### Short same-build decode screening

Correctness was followed by a deliberately short screen: one warmup per route/phase, three rounds, seven balanced direct pairs per contrast, with rotating A/A, S/QB and S/Q4 blocks. The same model/context was reused, KV was cleared, identical prefixes restored, and reset/prefill excluded from the32-token synchronized wall-clock interval. Timing disabled capture, trace, readback and CPU diagnostic timers. There were63 actual pairs and4,032 token rows.

All values below are milliseconds. The difference is computed per pair before taking its median.

| Contrast | Round | A p50 / p95 | B p50 / p95 | Paired B-A median | B wins |
|---|---:|---:|---:|---:|---:|
| A/A | 1 | 1289.345 /1294.705 | 1290.620 /1294.744 | +1.332 | 2/7 |
| A/A | 2 | 1641.452 /1699.434 | 1607.406 /1680.767 | +2.216 | 3/7 |
| A/A | 3 | 1679.235 /1687.446 | 1678.765 /1683.354 | +0.665 | 3/7 |
| S/QB | 1 | 1289.065 /1291.739 | 1301.048 /1303.715 | +12.588 | 1/7 |
| S/QB | 2 | 1289.646 /1500.303 | 1301.799 /1570.123 | +13.329 | 0/7 |
| S/QB | 3 | 1676.415 /1681.088 | 1686.018 /1688.431 | +10.020 | 0/7 |
| S/Q4 | 1 | 1287.186 /1290.483 | 1301.083 /1305.135 | +14.572 | 0/7 |
| S/Q4 | 2 | 1531.921 /1543.560 | 1540.838 /1578.920 | +9.658 | 0/7 |
| S/Q4 | 3 | 1675.731 /1679.408 | 1685.147 /1692.233 | +9.710 | 0/7 |

S/QB paired relative medians were +0.977%, +1.034% and +0.597%. Both AB and BA order-subgroup medians were positive in all three S/QB rounds. The per-token paired median was positive at30/32,32/32 and29/32 positions; the observed difference is not confined to first-token preparation. Nevertheless, this screen experienced nominal-to-serious thermal drift, large upper tails and a round2 A/A order subgroup of -22.937 ms. Battery level was100%, low-power mode was off, and thermal conditions were not constant.

**Conclusion:** the measured short screen shows a positive same-stock-shader guard-path wall-clock difference in this configuration. It does not justify the earlier Mac-only 'no measurable guard overhead' conclusion on A16. Its size and cause are not yet established as a stable universal cost. CPU selection/encoding time and GPU ranges were not measured here. Do not call B-A pure CPU instruction cost, and do not call Q4-B pure shader cost or subtract contrasts measured in different time blocks to isolate a component.

The smallest next control is a separate low-overhead CPU diagnostic around the read-only guard/cache lookup, with clock-cost calibration and an uninstrumented wall-clock comparison. Source inspection shows external library file loading and compilation only on a cache miss; a warm hit still takes the library lock, performs cache lookup and reads pipeline capability properties. Isolate those operations from tensor/buffer validation before attributing cost. Repeated file loading is not established by this run, and these source operations are not a measured cause. Keep stock as default until repeated end-to-end benefit is demonstrated.

## 4. Arithmetic accuracy, compatibility and quality require different references

The frozen contract remains:

`abs(output-reference) <= 2e-4 + 2e-5*abs(reference)`.

It is an empirical numerical regression rule, not a universal forward-error guarantee for every FP32 reduction or full-model logit. FP64 supplies a high-precision result for the specified inputs and operations; it is not a definition of model quality or Microsoft's INT8 semantics.

For round-to-nearest FP32, unit roundoff is `2^-24`. The standard accumulation factor `gamma_n=n*u/(1-n*u)` is approximately1.526e-4 at n2560 and4.122e-4 at n6912. In applicable error bounds these factors multiply sums of absolute products, not the final potentially cancelled dot result. A large `sum(abs(w*x))/abs(dot)` makes relative error sensitive to cancellation. Parallel reduction, FMA and transformed dot/correction expressions require their actual operation graph analyzed; these simple factors do not certify compiled fast-math kernels. See the [ALAFF dot-product analysis](https://www.cs.utexas.edu/~flame/laff/alaff/chapter06-backward-error-analysis-of-dot-general-case.html).

For the positive gate branch `h=g^2*u`, sensitivities are `2*g*u` to gate error and `g^2` to up error. Projection perturbations can therefore grow through squared gating; weighted RMSNorm then rescales and couples channels. Passing local Q FP64 does not mathematically imply later layer/logit values pass the same elementwise rule.

Existing diagnostics support this distinction. Local stock/C Q both passed FP64 on steps1/16/32, while a small pre-RoPE difference propagated to eight step16 residual failures. Restoring stock Q before RoPE restored downstream outputs bitwise. In full FFN analysis, fused and split C were bitwise equal; the798/880 figures describe a normalized-activation diagnostic, not weighted-RMSNorm output failures. Projection-error controls retained those deviations, while isolated FP32 activation rounding passed. Each down dot passed when referenced to its actual input, yet a full FP64 chain could still have one failing row. Those frozen failures remain recorded; they do not alone prove a packing bug or unacceptable task quality.

The fixed-prefix CPU/Metal token18 decision is also preserved, with the original strict comparison exit1. CPU's `computations - graphics` margin was +0.01025390625; Metal's was -0.00496673584. Controls identified blockwise CPU Q8_K activation packing versus direct F32 Metal decode, prefill half staging and K/V execution-setting contributions. A single-node Qcur-0 rounding change did not establish the whole-model flip's sole cause. Q8_K blockwise rounding is not automatically the official per-token INT8 quantizer. Backend parity and trusted-original-model fidelity remain different questions.

A separate acceptance ladder should be maintained:

1. **Model semantics:** trusted reference intermediates and teacher-forced logits, identical tokens and explicitly agreed quantizer/scale/rounding/clamping/norm/RoPE/KV rules.
2. **Exact controls:** identical compiled pipeline and inputs, copies, repeated execution and probe neutrality can require bitwise equality.
3. **Numerical accuracy:** independent same-input FP64, original thresholds, row failures/max error/relative L2/nonfinite counts; propagated-input comparisons reported separately.
4. **Application compatibility:** full logits, RMSE/centered RMSE, KL and top-k/margins. `top1 margin > 2*max_abs_logit_difference` certifies greedy stability only at the checked endpoint.
5. **Product quality:** independent held-out corpus/tasks, fixed data/tokenization/scoring and agreed noninferiority budgets. Do not derive a new pass threshold from observed maxima or treat fluent generated text as held-out evaluation.

Publish `parity_pass`, `fp64_pass` and `model_output_pass` separately, using null for unmeasured checks. Analysis exit0, token agreement and fewer failed rows cannot substitute for required acceptance. Proposed product-quality policies must not overwrite frozen V1 records.

## Evidence index and reanalysis

Local experiment identifiers, with raw data retained locally:

- `2026-11-15-review-221448/runs/guard-primary-readback`: original failed diagnostic attempt.
- `2026-11-15-review-221448/runs/guard-primary-readback-2`: successful raw correctness evidence.
- `2026-11-15-review-221448/runs/guard-screen`:63 paired decode samples and4,032 token rows.
- `2026-11-15_182149+0900-iphone-q4-complete`: earlier formal isolated Q4 campaign and whole-decode comparison.
- `2026-11-12_202159+0900-iphone-candidate-sequence/MODEL_WORKING_SET.md`: tensor accounting.
- `2026-11-10-iphone-layer-tuning`: A16 graph/capture and profiler limit.
- `2026-11-07_164204-public-2b-token-divergence` and `2026-11-07_165918+0900-qcur0-rounding-scope`: fixed-prefix numerical controls.
- `2026-11-11-mac-layer0-numerics`: arithmetic/propagation diagnostics, not phone performance claims.

Recompute the published guard analysis from separately supplied local evidence:

```sh
python3 bitnet/analysis/review-2026-11-15/analyze-guard-control.py --verify "$LOCAL_VERIFY_RUN" --timing "$LOCAL_SCREEN_RUN" --output "$ANALYSIS_JSON"
```

The script reads raw bytes/logs/CSV and outputs aggregate analysis without private device identifiers, process addresses or absolute local evidence paths. It does not rerun inference. GPU execution and strict V1 numerical acceptance remain separate from successful analysis.
