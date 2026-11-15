# Physical iPhone Q stock4x1 validation and measurements

This folder contains derived analysis from the 2026-11-15 iPhone experiment.
Raw tensors, logits, samples, device logs and signed applications remain local.
`EVIDENCE_INDEX.json` records experiment-relative identifiers, sizes and SHA-256
digests without copying those artifacts. All performance results were measured
on a physical iPhone 14 Pro with Apple A16 GPU and iOS 27.2.1. The build used
Xcode 27.2 (27B52), iPhoneOS SDK 27.2, pinned QVAC commit
`2db8b046d555316019ae75d1be90484adc8b9330`, and the public Microsoft 2B
checkpoint. This is separate from the team's checkpoint.

## Q-only correctness

The candidate is **Q stock4x1**, not the original 4x2 PoC or C8x2. Its historical
Metal symbol still contains `c8x2`; the validated launch actually uses four
output rows per SIMD group and one SIMD group per threadgroup. Only the
layer-0 `Qcur-0` decode projection is replaced. Other Q nodes and all FFN work
remain stock.

Both primary and independent fixed-token histories pass the declared complete
Q-only acceptance, with strict exit code 0. Each history checks all 32 real
decode calls on both stock and Q4: 64 vectors, 2,560 values each. There are no
nonfinite values or FP64 tolerance failures. Q4 and stock Q outputs are bitwise
equal throughout. Residual outputs at steps 1, 16 and 32 and all 33 complete
logit vectors are also bitwise equal. Probe-on/off equality is established for
the complete logits; it does not imply an unmeasured probe-off Q readback.

The frozen bound is `abs(actual-reference) <= 2e-4 + 2e-5*abs(reference)`.
Maximum Q absolute FP64 error is 1.576802e-5 for the primary history and
1.324259e-5 for the independent history. Independent selected-row `math.fsum`
checks agree with the vectorized reference. Actual weight capture hashes match
the model, and input hashes are checked before output comparisons.

Each route records 68 copied payloads, 129 diagnostic copy dispatches and GPU
completion evidence. The custom route records 32 custom Q completions and 928
stock Q completions; the reference records 960 stock Q completions. Prefill has
no custom Q dispatches. Missing-library and invalid-geometry controls produce
32 recorded fallbacks and zero custom dispatches. Their missing local FP64
verdict remains null; their portable full-V1 exit code 2 is retained.

An earlier invalid full-weight capture remains a failed local stage. The copy
shader's existing size bound required 62 bounded weight-copy dispatches in the
same command buffer. The fix changed CPU copy encoding, not the shader,
reference, model bytes or tolerance. No disk weight was substituted for a
claimed GPU capture.

## Performance scope and result

All timing campaigns use the **primary history only**. Independent-prompt
correctness is a separate result. A/A calibration is reported without
subtraction. No measured rows were excluded and no rounds or thermal cohorts
were pooled.

`operator-rounds.csv` has 18 groups: two campaigns, three rounds, and AA
(stock/stock), SQ (stock/Q4), CQ (C8/Q4) contrasts. Each group uses 30 warmups per
path and 201 alternating pairs. The metric is single-dispatch GPU
command-buffer elapsed time after completion,
`(GPUEndTime-GPUStartTime)*1e6`, in microseconds. It includes command-buffer
elapsed time and is not an isolated hardware shader-cycle counter.

Q4 is faster than stock in every recorded SQ round median, AB/BA group median
and chronological-quartile median. Paired Q4-minus-stock medians are:

| Campaign | Round 1 | Round 2 | Round 3 |
| --- | ---: | ---: | ---: |
| Nominal block endpoints | -6.250 us | -6.042 us | -6.000 us |
| Sustained/serious endpoints | -6.208 us | -7.000 us | -6.083 us |

These correspond to median per-pair relative changes of about -2.68% to
-3.11%. C8 remains faster than Q4 in this isolated operator comparison; its
previous model-compatibility failure is not repaired by these measurements.

`decode-rounds.csv` has 12 groups: two primary-history campaigns, three rounds,
and AA/SQ4 contrasts with 21 pairs per group. Its metric is synchronized host
wall time for the complete fixed 32-token decode in milliseconds. Each path
restores the same context and KV history; clearing/restoring the prefix is
outside this timing interval. Normal CPU encoding, GPU execution and completion
are included. Capture, diagnostic readback and tracing are disabled.

No complete-decode speedup is established. Paired Q4-minus-stock medians are
+1.510, +8.954 and +9.811 ms in the first campaign, and +7.261, +8.099 and
+0.339 ms in the sustained campaign. The final sustained result overlaps
order/time-block and A/A variation. These results do not establish a fixed
0.6% cost or localize a CPU-versus-GPU cause.

Thermal states are discrete endpoint snapshots. Nominal endpoints do not prove
a cold phone or fixed clocks. Operator snapshots bracket a contrast block and
its warmup; decode entry snapshots precede reset/prefix restoration. Absolute
decode time drifts substantially even within the serious category. Charging,
battery and low-power observations are retained in the aggregate data. No
continuous thermal trace, occupancy, bandwidth or ALU-counter claim is made.

## Selected down arithmetic control

`down-selected-summary.json` and `down-selected-routes.csv` cover 96 selected
outputs on one existing real stock-normalized down input. Stock and the
stock-helper wrapper are bitwise equal and each retains 12 FP64 failures.
Moving the correction term to the end changes 48 outputs: 22 become closer to
FP64, 26 farther and 48 have equal error. It still has 12 failures and worsens
the largest error/tolerance ratio from 1.5663 to 1.8298. Strict exit code 1 is
retained; processing exit code 0 only means analysis completed. This is not a
full-matrix, full-FFN or model-quality pass. No down performance was measured.

## Files and interpretation

- `SUMMARY.json`: Q contract, source/build/model hashes, two-history
  correctness, fallback outcomes, measurement scopes and limitations.
- `operator-rounds.csv`: all 18 operator group aggregates, p50/p95, paired
  differences, wins, AB/BA and time-quartile results, power/thermal endpoints.
- `decode-rounds.csv`: all 12 complete-decode group aggregates with separate
  thermal-cohort counts.
- `down-selected-summary.json` and `down-selected-routes.csv`: selected-row
  negative result and unchanged strict failure.
- `EVIDENCE_INDEX.json`: digest-only index of locally retained evidence.

Quantiles use linear interpolation at `(n-1)*p`. A negative paired B-minus-A
value favors B. A median of pairwise relative changes differs from a ratio of
the two route medians; both columns are retained. The source analysis tools and
runner provide reproducible computation; this folder adds no raw measurements.

Keep stock as the default. Q4 remains an opt-in layer-0 compatibility candidate
with a measured operator benefit, not demonstrated whole-model acceleration.
Full FFN/Transformer FP64, language-model quality, independent-prompt timing,
team-checkpoint validation and the historical CPU/Metal token divergence remain
outside this result.
