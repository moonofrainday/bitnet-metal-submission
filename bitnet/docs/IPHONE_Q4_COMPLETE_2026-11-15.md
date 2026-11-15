# iPhone Q4 completion: 2026-11-15

Q4 completed the declared layer-0 Q-only numerical acceptance on two fixed histories. Its isolated operator was consistently faster than actual stock in both recorded thermal cohorts, but neither whole-decode campaign demonstrated an end-to-end gain. The selected Down correction-placement control completed and failed the original FP64 criterion; it is a useful negative result, not a numerical repair. Raw evidence remains local in experiment `2026-11-15_182149+0900-iphone-q4-complete`; failed attempts remain preserved.

## Scope and frozen standard

Physical iPhone 14 Pro / Apple A16 GPU, iOS 27.2.1 (24B156), Xcode 27.2 (27B52), iPhoneOS SDK 27.2. The public Microsoft BitNet 2B TQ2_0 GGUF has SHA-256 `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2`. This is not the team's checkpoint.

The source derives from pinned QVAC `2db8b046d555316019ae75d1be90484adc8b9330` through submission commit `b6441351acb21418959a52b3864c11d55defa189`. All framework archives were rebuilt locally; no old phone backend or Mac dylib supplied the stock PSO. Q4 preserves the stock four-row helper and specializes the fixed shape/launch to one SIMD group. The original independently authored PoC remains separate and unchanged.

The canonical patch followed by [the additive Q-readback patch](../patches/q4-readback-additive.patch) was replayed on the published pristine source import; all eight modified upstream files matched the tested source byte for byte. The additive patch is 15,924 bytes, SHA-256 `5c44ac2e6f6c408dafc51b89d23cbbe9413ac68be556ce87fbd6bbec1816d2b2`. All 1,066 recorded framework compilation inputs match Signed4. Eleven of twelve recorded package inputs also match; the exception is the host Python runner's documented post-build same-install orchestration fix. It changes no app, framework or shader binary.

The frozen numerical bound is `abs(actual-reference) <= 2e-4 + 2e-5*abs(reference)`, with zero failing or nonfinite rows. Q's reference is an independent FP64 logical-K unpack/dot on the actual packed W and F32 X. Residual and logit comparisons use stock F32 outputs. They are **not** an FP64 implementation of the complete Transformer layer or a model-quality test.

## Completed Q-only numerical acceptance

Both fixed histories use 27-token stock prefill, 32 pinned continuation IDs, context 512, batch/ubatch 128, six CPU threads, F16 K/V and disabled flash attention. Only layer-0 Q is eligible; all FFN, down, normalization, other attention projections and residual operations remain stock.

| History | Q vectors / values checked, S plus Q4 | FP64 failed/nonfinite | S/Q4 bit differences | Maximum Q FP64 absolute error | Scoped strict exit |
|---|---:|---:|---:|---:|---:|
| Primary | 64 / 163,840 | 0 / 0 | 0 | 1.576801560077e-5 | 0 |
| Independent | 64 / 163,840 | 0 / 0 | 0 | 1.324259466173e-5 | 0 |

Each history has 32 actual Q4 encodes and matching GPU completions. Every step binds the audited TQ2_0 W[2560,2560] and F32 X/Y[2560] with checked live shapes, strides, buffers, byte ranges, offsets and grid `[640,1,1]` / threads `[32,1,1]`. The full W is copied once per route and matches the GGUF tensor SHA `b2dea7f42cacfd08ed780df979f6f28240026795b1f5c87ca8c16fc7da23dc58`.

Steps 1/16/32 `l_out-0` residual vectors and all 33 complete logits vectors are stock-bit-identical. Both within-process uninstrumented repeats and separate probe-off runs match the readback runs bit for bit on all 33 logits. That establishes probe neutrality at the observable full-logit boundary; an entirely uninstrumented pre-RoPE Q tensor was not itself read.

For this declared Q-only scope, `parity_pass`, `fp64_pass`, `model_output_pass`, `same_input_pass`, execution and probe-neutrality flags are all true. The CPU analyzer returns strict 0 only after every required check. Broader historical complete-FFN failures and their strict exit 1 are unchanged; processing success or Q-only acceptance does not clear them.

Evidence: `analysis/capture3-{primary,independent}/SUMMARY.json` and their per-row CSVs; `runs/capture3-{primary,independent}-{off,on}` contain original commands, app results, pipeline traces and payloads. These raw paths are relative to the local experiment and are not uploaded.

## Fallback and diagnostic corrections

Switch-off and stock prefill have zero custom calls. Missing-library and unsupported-geometry controls each produced 32 explicit rejections, zero custom Q calls, 960 stock Q calls and stock-identical logits. Unsupported geometry is not a malformed-stride test. Actual normal-run strides are checked; an invalid real graph was not manufactured to force an unsafe memory access.

Two readback defects were found and corrected without changing any computational shader:

1. `Qcur-0` also names the post-RoPE tensor. The initial name-only diagnostic guard aborted on that non-MUL_MAT node. Runtime skip logs now explicitly identify `ROPE`, while only the true MUL_MAT is captured before RoPE.
2. The reused `ffn_shadow_copy` kernel is bounded to 6,912 uint values. One large dispatch therefore truncated W. The corrected host encoding performs 61 chunks of 27,648 bytes plus one final 3,072-byte chunk, with rebased source/destination offsets. Each route has 68 semantic captures and 129 actual diagnostic dispatches, all in the original encoder/command buffers. The independent full-W hash gate rejected the old captures; those strict-2 records remain preserved.

Diagnostic copies alter scheduling and are never timing evidence. Formal timing unsets readback/trace controls.

## Single-operator timing: nominal thermal endpoints

Same accepted real step-16 W/X, actual embedded stock PSO versus external Q4; 30 warmups/path, 201 balanced alternating pairs/contrast/round, three rounds. Each command buffer contains one dispatch. Metric: `(GPUEndTime-GPUStartTime)*1e6` after completion, in microseconds, **GPU command-buffer elapsed time**.

| Round | Stock p50 / p95 us | Q4 p50 / p95 us | Paired Q4-stock median us | Median paired relative change | Q4 wins |
|---:|---:|---:|---:|---:|---:|
| 1 | 224.541665 / 228.499999 | 218.499999 / 221.999999 | -6.250000 | -2.787068% | 198/201 |
| 2 | 224.541667 / 228.333332 | 218.375000 / 222.625000 | -6.041668 | -2.699181% | 195/201 |
| 3 | 224.125000 / 228.500001 | 218.125000 / 221.000000 | -6.000000 | -2.680717% | 198/201 |

Q4 wins 591/603 direct stock pairs. All six AB/BA and twelve chronological-quartile medians favor Q4. A/A paired medians are +0.291666, +0.041666 and +0.124999 us. All block thermal endpoints were nominal, charging, battery 100%, low-power mode off. Continuous/per-pair thermal state, GPU clock, power, occupancy and bandwidth were not measured. This is a measured nominal-endpoint campaign, not proof of a universally cold or fixed-frequency device.

The complete operator campaign includes A/A, stock/Q4 and C8/Q4: 1,809 pairs and nine native summaries. Independent CPU recomputation matches every native summary and retains all rows. C8 is the mask 8x2 research variant, not the original 4x2 PoC; it is faster than Q4 here but does not inherit Q4's real-model acceptance.

The comparison measures the complete stock versus fixed-shape Q4 operator paths. It does not isolate only threadgroup size: wrapper/ABI and compilation context also differ. No hardware-counter cause is asserted.

Evidence: `runs/operator-nominal-start-2` and `analysis/operator-nominal-start` in the local experiment. Raw samples remain local.

## Reproducibility correction for operator timing

Mandatory App installation changed the absolute Documents-container UUID. The native qualification binds absolute input paths, so a previous installation's otherwise identical qualification correctly rejected timing. Actual device verification confirmed all nine relative file paths/hashes and library hashes were unchanged while the absolute roots differed.

The Python runner now installs once, performs a fresh mandatory nine-case verification, fetches and validates the qualification and actual app exit, then launches timing in that same installation. Old output folders remain untouched. The native hash/numerical gate is not relaxed, and no app, shader or backend rebuild was required for this orchestration fix.

## Single-operator timing after sustained model load

The second operator campaign reused the same accepted step-16 W/X, signed build, stock PSO, Q4 library and 30-warmup/201-pair/three-round protocol. All contrast-block thermal endpoints were **serious**; battery was full at 100%, low-power mode off. It again contains A/A, stock/Q4 and C8/Q4: 1,809 pairs, 3,618 measured command buffers and no excluded rows. It remains a single-operator GPU command-buffer measurement, not decode wall time.

| Round | Stock p50 / p95 us | Q4 p50 / p95 us | Paired Q4-stock median us | Median paired relative change | Q4 wins |
|---:|---:|---:|---:|---:|---:|
| 1 | 226.000 / 230.000 | 219.375 / 223.875 | -6.208 | -2.750956% | 194/201 |
| 2 | 226.125 / 229.875 | 219.167 / 223.042 | -7.000 | -3.112188% | 198/201 |
| 3 | 224.875 / 229.125 | 218.542 / 222.750 | -6.083 | -2.703704% | 197/201 |

Q4 wins 589/603 direct stock pairs. Every stock/Q4 AB/BA subgroup and chronological-quartile paired median favors Q4. A/A paired medians are +0.042, -0.333 and 0.000 us. Q4 is slower than C8 in every C8/Q4 pair in both campaigns; C8 is not the original PoC and has no new real-model acceptance here. The nominal-endpoint and serious-endpoint campaigns are reported separately, not pooled. The retained operator advantage does not establish a thermal cause, a general frequency-independent speedup or whole-model acceleration.

Evidence: `runs/operator-sustained/operator/performance/{samples.csv,summary.json}` and `analysis/operator-sustained/SUMMARY.json`.

## Whole-decode timing: no measured end-to-end benefit

Both campaigns use the **primary history only**, the same signed build, one reused model/context per campaign, stock versus layer-0 Q4, two route warmup passes and three rounds. Each round has 21 A/A and 21 direct stock/Q4 pairs with balanced AB/BA order and rotating contrast blocks: **126 pairs and 8,064 token records per campaign**, with no samples excluded. Every path clears KV and restores the same 27-token prefix before timing the same 32 continuation IDs. Reset, prefill, model loading and hashing are excluded. Normal CPU encoding, GPU execution/completion and per-token loop/clock overhead are included.

The metric is **synchronized 32-token host-wall milliseconds**. Capture, tensor readback and diagnostic trace are disabled. It is not the sum of separately timed GPU kernels. Negative paired Q4-stock would favor Q4; all six round medians below are positive.

| Campaign / round | Stock p50 / p95 ms | Q4 p50 / p95 ms | Paired Q4-stock median ms | Median paired relative change | Q4 wins |
|---|---:|---:|---:|---:|---:|
| First / 1 | 1523.294 / 1554.929 | 1523.119 / 1537.319 | +1.510 | +0.110357% | 9/21 |
| First / 2 | 1519.479 / 1532.117 | 1528.410 / 1536.074 | +8.954 | +0.589162% | 1/21 |
| First / 3 | 1672.692 / 1683.428 | 1683.458 / 1724.959 | +9.811 | +0.586542% | 3/21 |
| Sustained / 1 | 1838.651 / 1900.308 | 1847.512 / 1909.707 | +7.261 | +0.381563% | 2/21 |
| Sustained / 2 | 2000.696 / 2104.525 | 2018.455 / 2104.265 | +8.099 | +0.399383% | 7/21 |
| Sustained / 3 | 2418.593 / 2438.003 | 2421.630 / 2431.776 | +0.339 | +0.014108% | 10/21 |

The first campaign's Q4 win count is 13/63; the sustained campaign's is 19/63. A ratio of separately computed p50 values is not the median of paired relative changes: first-round Q4 p50 is slightly lower while the paired median is positive. All paired statistics use actual matched paths.

| Campaign | A/A paired medians, rounds 1/2/3, ms | Thermal observations |
|---|---|---|
| First | -1.093 / -2.575 / -0.399 | Begins nominal; direct round 1 moves from fair to serious. Later direct rounds are serious. |
| Sustained | +0.590 / -3.832 / -0.257 | All path thermal endpoints are serious. |

Battery stays at 100%; the recorded state changes from charging to full during the first campaign, and low-power mode is off throughout. Decode thermal observations are before/after each path, not continuous temperature or clock readings. There is strong absolute drift: the first campaign's first A/A p50 is about 1.15 s, while the sustained campaign's final A/A p50 is about 2.35 s. These are distinct sequential campaigns under changing conditions.

The first campaign's three direct AB and BA medians are all positive. In the sustained campaign, round 2 AB/BA medians are +12.615/-0.260 ms and round 3 is -0.907/+1.007 ms; chronological subgroups also change direction. Some observed differences are comparable to calibration/order variation. A/A is reported as calibration, never subtracted to manufacture an adjusted gain.

**Conclusion:** no end-to-end Q4 benefit was observed in either primary-history campaign. The observed penalty varies substantially with campaign, round and order; it is not a universal fixed regression. Replacing one Q projection per token affects only one node among the model's 30 blocks, and its approximately 6-7 us isolated saving cannot simply be multiplied into a guaranteed model gain. These measurements do not locate the remaining cost in guards, CPU encoding, shader execution or scheduling. Independent-history whole-decode performance remains unmeasured. Tail-latency improvement is also unestablished.

Evidence: `runs/decode-primary-{first,sustained}/device/{samples.csv,token-times.csv,result.json,execution.json}` and `analysis/decode-primary-{first,sustained}/SUMMARY.json`, including all order, time-quartile, token and thermal-cohort summaries.

## Selected Down correction-placement control: negative result

This separate operator test uses the actual independent-history stock stream at layer 0, step 16: F32 weighted-normalization input X[6912] and TQ2_0 down weights with 2,560 output rows. It tests twelve preselected eight-row groups, **96 outputs per route**, rather than the full matrix. The three paths are actual stock S, an independently compiled unmodified stock-helper bridge S-wrap, and a correction-end variant. The only intended arithmetic change in the variant moves the activation correction from the accumulator's initial value to after the four packed-dot additions, before block scale. Layout, coefficients, K/lane mapping, scale, reduction and launch remain fixed relative to S-wrap.

All 36 command buffers completed, canaries remained intact and W/X were unchanged. S-wrap is stock-bit-identical on all 96 outputs; actual S also reproduces the saved historical stock outputs on those rows. The CPU logical-K reference was independently checked with `math.fsum`, with at most 2.85e-14 difference from the native FP64 reference.

| Route | FP64 failed / 96 | Nonfinite | Stock bit differences | Max absolute FP64 error | Max error / frozen tolerance |
|---|---:|---:|---:|---:|---:|
| S | 12 | 0 | 0 | 3.891796691171e-4 | 1.566268 |
| S-wrap | 12 | 0 | 0 | 3.891796691171e-4 | 1.566268 |
| Correction-end | 12 | 0 | 48 | 3.891796691171e-4 | 1.829765 |

The same twelve rows fail in all paths: 5, 338, 356, 362, 455, 502, 882, 1148, 1508, 1830, 2155 and 2508. The variant is closer to the reference on 22 rows, farther on 26 and unchanged on 48; its worst normalized error increases at row 362. It removes no selected failure. All 48 stock bit differences nevertheless remain within the separate stock-relative tolerance. That distinction does not repair FP64 acceptance.

The native app and strict CPU verifier both return **1 for a completed measured numerical failure**; processing-only analysis returns 0 while retaining that strict result. The scoped flags are `execution_pass=true`, `parity_pass=true` for S-wrap/S, `fp64_pass=false`, `model_output_pass=null`, `overall_pass=false`. This is not a deployment failure and not a full-FFN/model test. No Down performance measurement or full 2,560-output expansion was performed. The data reject this correction-placement variant as a numerical repair; they do not show that every other arithmetic control must fail. Historical complete-FFN stock/F8 FP64 failures remain false and unchanged.

Evidence: `runs/down-selected-correction/operator/{summary.json,rows.csv,group-*/}`; independent CPU analysis `analysis/down-selected/summary.json`; strict invocation `analysis/down-selected-strict/summary.json`. The actual strict exit record is retained beside the original commands/logs.

## Reusable build, run and analysis entry points

Run from the source checkout. Keep all outputs in a **new external directory** and supply your local device/signing/model values through environment variables. `PYTHON_FP64` must select Python with NumPy; timing analysis uses only the standard library. No device identifier or signing credential is embedded below. The commands create new measurements; they do not reproduce an archived binary hash automatically.

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export ARTIFACTS=/absolute/path/to/new-local-phone-run
export PYTHON_FP64=/absolute/path/to/python-with-numpy
# Set APPLE_TEAM_ID, BUNDLE_ID, IPHONE_DEVICE_ID and BITNET_2B_GGUF locally.

python3 bitnet/ios/build.py --team "$APPLE_TEAM_ID" --bundle-id "$BUNDLE_ID" \
  --configuration Debug --allow-provisioning-updates --output "$ARTIFACTS/build"
export IOS_BUILD="$ARTIFACTS/build"

for HISTORY in primary independent; do
  python3 bitnet/ios/run.py --build-dir "$IOS_BUILD" --device "$IPHONE_DEVICE_ID" \
    --model "$BITNET_2B_GGUF" --mode graph-verify --routes S,Q4 --layers 0 --install \
    --history-dir "bitnet/ios/histories/$HISTORY" --output "$ARTIFACTS/runs/$HISTORY-off"
  python3 bitnet/ios/run.py --build-dir "$IOS_BUILD" --device "$IPHONE_DEVICE_ID" \
    --model "$BITNET_2B_GGUF" --mode graph-verify --routes S,Q4 --layers 0 --install --reuse-model \
    --history-dir "bitnet/ios/histories/$HISTORY" --q-readback --output "$ARTIFACTS/runs/$HISTORY-on"
  "$PYTHON_FP64" bitnet/analysis-tools/analyze-q4-complete.py \
    --input "$ARTIFACTS/runs/$HISTORY-on" --probe-off "$ARTIFACTS/runs/$HISTORY-off" \
    --runtime-log "$ARTIFACTS/runs/$HISTORY-on/logs/launch.log" --output "$ARTIFACTS/analysis/$HISTORY"
done

"$PYTHON_FP64" bitnet/analysis-tools/prepare-q4-fixture.py \
  --input "$ARTIFACTS/runs/primary-on" --acceptance "$ARTIFACTS/analysis/primary/SUMMARY.json" \
  --build-dir "$IOS_BUILD" --output "$ARTIFACTS/q-fixture"

python3 bitnet/ios/run.py --build-dir "$IOS_BUILD" --device "$IPHONE_DEVICE_ID" \
  --model "$BITNET_2B_GGUF" --mode q-benchmark --fixture "$ARTIFACTS/q-fixture" \
  --install --reuse-model --output "$ARTIFACTS/runs/operator"
python3 bitnet/analysis-tools/analyze-q4-timing.py --mode operator \
  --input "$ARTIFACTS/runs/operator" --accepted-q4 "$ARTIFACTS/analysis/primary/SUMMARY.json" \
  --output "$ARTIFACTS/analysis/operator"

python3 bitnet/ios/run.py --build-dir "$IOS_BUILD" --device "$IPHONE_DEVICE_ID" \
  --model "$BITNET_2B_GGUF" --mode decode-benchmark --routes S,Q4 --layers 0 \
  --comparisons AA,SQ4 --warmups 2 --rounds 3 --pairs 21 --install --reuse-model \
  --history-dir bitnet/ios/histories/primary --output "$ARTIFACTS/runs/decode-primary"
python3 bitnet/analysis-tools/analyze-q4-timing.py --mode decode \
  --input "$ARTIFACTS/runs/decode-primary" --accepted-q4 "$ARTIFACTS/analysis/primary/SUMMARY.json" \
  --output "$ARTIFACTS/analysis/decode-primary"
```

Use separate fresh run/analysis names for repeats and sustained-load campaigns. Save the actual thermal/power observations; do not label a run cold merely from its intended starting condition. The Q benchmark automatically performs fresh nine-case qualification after its one install and before timing, so no earlier installation's absolute input-path gate is reused.

For Down, first prepare the matching local fixture/libraries with [the Down instructions](../down-controls/README.md), then run:

```sh
python3 bitnet/ios/run.py --build-dir "$IOS_BUILD" --device "$IPHONE_DEVICE_ID" \
  --model "$BITNET_2B_GGUF" --mode down-verify --fixture "$DOWN_FIXTURE" \
  --down-libraries "$DOWN_LIBRARIES" --install --reuse-model --output "$ARTIFACTS/runs/down"
python3 bitnet/analysis-tools/analyze-down-selected.py --input "$ARTIFACTS/runs/down" \
  --output "$ARTIFACTS/analysis/down" --strict
```

The final CPU audit fixed a verifier-only nonfinite serialization edge: nonfinite aggregate metrics are explicit nulls with status, while failed/nonfinite counts and strict exit 1 remain. Four copied-evidence CPU controls returned the expected exits: finite pass 0, candidate-NaN numerical failure 1, missing output 2, invalid input hash 2. The actual 288 finite outputs were reanalyzed after that change; all errors, failed positions and conclusions are unchanged. This hardening changes no GPU shader, app or tolerance.

The archived Down fixture intentionally retains strict exit 1. Continue to the CPU analysis after preserving that expected measured-failure exit; do not use an unconditional success wrapper. The local processing form without `--strict` preserves the numerical failure in its summary. [The iPhone README](../ios/README.md) documents fallback modes, install semantics and native exit distinctions; [the analysis README](../analysis-tools/README.md) specifies scope, prerequisites and generated evidence.

On the original machine with the local raw artifact tree, the compact CPU-only evidence entry is:

```sh
python3 "$EXPERIMENT/rerun.py" --action analyze --output "$EXPERIMENT/fresh-evidence-recompute"
```

Set `EXPERIMENT` to that local experiment directory; the output must be new. This entry was actually executed as `final-evidence-recompute`: both Q analyzers and all four timing analyzers returned 0, while the selected Down strict analyzer returned the preserved expected **1**. `EXIT_CODES.json` records every actual and expected status. Completing that analysis sequence does not turn Down's numerical failure into a pass, and it executes no GPU work.

## Local evidence index and next decision

All paths below are relative to the local experiment, not URLs to uploaded raw data:

| Evidence | Relative path |
|---|---|
| Primary/independent all-32-call numerical acceptance | `analysis/capture3-{primary,independent}/SUMMARY.json` |
| Matched probe-off/on commands, runtime and payloads | `runs/capture3-{primary,independent}-{off,on}/` |
| Explicit missing-library/geometry fallback controls | `runs/fallback-{missing-library,invalid-geometry}/summary.json` |
| Nominal-endpoint operator raw/analysis | `runs/operator-nominal-start-2/`; `analysis/operator-nominal-start/` |
| Serious-endpoint operator raw/analysis | `runs/operator-sustained/`; `analysis/operator-sustained/` |
| First/sustained primary decode raw/analysis | `runs/decode-primary-{first,sustained}/`; `analysis/decode-primary-{first,sustained}/` |
| Selected Down actual GPU data / independent strict audit | `runs/down-selected-correction/operator/`; `analysis/down-selected-strict/` |
| Frozen signed build and command/source/binary identities | `signed-build-4/BUILD.json`; each run's `BUILD_IDENTITY.json` |
| Absolute-path relocation diagnostic | `debug-path-relocation/PROOF.json` |
| Same-install orchestration controls | `support/Q_SAME_INSTALL_CONTROLS.json` |
| Exact patch replay and built-source identity checks | `PATCH_REPLAY_CHECK.json`; `FINAL_BUILD_SOURCE_CHECK.json` |
| Actual final CPU recomputation exits | `final-evidence-recompute/EXIT_CODES.json` |

Keep stock as default and retain Q4 as a numerically accepted **layer-0 Q-only** operator candidate, with its measured isolated gain and absent end-to-end benefit both visible. Keep correction-end as a negative result; no Down/FFN repair is selected. No independent-history decode performance, full-model quality assessment or new G/F8 performance contrast was completed in this round.

The next discriminating experiment is a same-build **guard/stock-PSO A/B** in the real graph: A uses native stock dispatch; B actually performs the Q4 node/shape/stride/buffer/offset safety checks but dispatches the same compiled stock PSO with identical native ABI, binding, grid, threadgroup and scheduling. Confirm guard hits and bitwise parity before timing. B-A can estimate added guard/encoding-path effects. The existing Q4-B difference would still combine implementation, launch and scheduling effects; it must not be called pure shader time. This control is proposed, not implemented or measured here, and no current result forces an attribution to guards, CPU encoding or GPU scheduling.
