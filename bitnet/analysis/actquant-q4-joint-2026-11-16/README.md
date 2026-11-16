# Joint ActQuant + Q4: scoped graph qualification, 16 November 2026

The repaired public-model ActQuant graph and the stock-derived Q4 specialization now pass their **combined, layer-0 Q-only stock-compatibility gate** on M2 Pro and physical iPhone 14 Pro/A16, for both fixed histories. This is separate from the completed trusted-reference model comparison, which retains strict exit **1**. Fresh A16 post-ActQuant isolated-operator pairing measures a 2.61%-2.68% reduction; joint whole-decode and corpus/PPL are unmeasured.

## What changed and what is preserved

- Pinned QVAC: `2db8b046d555316019ae75d1be90484adc8b9330`. This joint implementation is source `edbd109bc152decbbc7274b703d21bd6099bf98c` on base `e84732ced307d7dbb0dabe1f7c60c8c4b3bdc091`. Earlier ActQuant-only phone results used `4c249e3`; evidence/docs were consolidated in `e84732c`. These are separate binary identities.
- New default-off `LLAMA_BITNET_ACTQUANT_Q4=1` permits only ActQuant ON plus stock-helper Q4, rows=4, one SIMD group, layer 0 and single-token `Qcur-0`. It verifies the actual `actq_qkv_out-0` ACT_QUANT producer and epsilon, in addition to existing node/weight/type/shape/stride/live-buffer/bounds/disjoint-range guards. C8, FFN, other layers and guard-only mixing are not authorized.
- Q4 reuses complete stock helper arithmetic; this is **not** the independently authored original 4x2 shader. Its actual pipeline symbol remains `tq2_0_matvec_poc_c8x2`, but source/build hashes and geometry identify stock4x1. The shader algorithm is unchanged in this joint experiment.
- Public GGUF SHA-256 is `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2`. W is TQ2_0 `[2560,2560]`; the actual post-ActQuant X and pre-RoPE Y are F32 `[2560,1]`. Both routes use identical live W/X; independent FP64 decodes the same packed weights and block scales.

## Measured correctness

Each history has 27 prefill tokens and 32 fixed continuation IDs, context512, batch/ubatch128, six threads, F16 KV and disabled flash attention. Each captured route includes one prefill plus all 32 N1 calls, not just steps 1/16/32. The FP64 gate covers the **32 decode dots**; prefill is checked for stock parity, not promoted as FP64-pass.

| Device | History | Q4 dispatch/completion | FP64 failing values per route | Max absolute FP64 error | Residual / full-logit vectors vs stock ON | Scoped exit |
|---|---|---|---|---|---|---|
| M2 Pro | primary | 32/32 | 0 / 81,920 | 1.267946573e-05 | 3 / 33, bitwise | 0 |
| M2 Pro | independent | 32/32 | 0 / 81,920 | 1.267150037e-05 | 3 / 33, bitwise | 0 |
| A16 | primary | 32/32 | 0 / 81,920 | 1.267946573e-05 | 3 / 33, bitwise | 0 |
| A16 | independent | 32/32 | 0 / 81,920 | 1.267150037e-05 | 3 / 33, bitwise | 0 |

For each device/history, stock and Q4 have zero bit differences in all 33 captured Q vectors. Each route checks 81,920 decode values versus independent FP64, using `abs(error) <= 2e-4 + 2e-5*abs(reference)`, with no failed or nonfinite rows. Three residual vectors at steps 1/16/32 and every one of33 full128256-logit vectors match the **same-build ActQuant-ON stock graph** bitwise. The largest error/bound ratios are 0.060105211 primary and 0.062065417 independent. A separate CPU-only archive audit additionally confirms all 33 captured layer-0 output vectors per phone history match bitwise; the strict analyzer retains its required three residual checkpoints. Missing checks outside these scopes remain null.

Actual trace accounting, per history: candidate32 custom Q encodes and32 completed custom calls; stock route0 custom. Both preserve30 stock prefill Q calls and928 non-target stock decode Q calls; all990 Q encodes complete. Probe-on/plain controls match all 33 logits for stock and Q4. Each missing-library run records32 eligible guard hits,32 fallbacks,0 custom dispatches and stock-identical33 logits. Six CPU-only startup controls accept the valid route and reject missing permission, unsupported geometry/layers, FFN mixing and permission without ActQuant; those are **not** device numerical tests.

Fresh A16 prerequisite quantizer validation independently passes 20 cases/386,567 values with 0 bit, integer-grid or frozen-bound failures, strict operator exit 0. Runtime compile-option getters report languageVersion262144, mathMode0/Safe and mathFloatingPointFunctions1/Precise for the **new quantizer library only**. Each captured Q-input quantizer scope contains 33 input/output pairs/151,040 values per route/history, bitwise equal to official F32 on its own actual input. This local capture scope does not assert a new full 120-node graph audit; the earlier complete audit is indexed separately.

## Fresh post-ActQuant A16 operator pairing

The actual physical-phone step-16 Q input is now used for a same-signed-build comparison against real stock. W SHA-256 is `b2dea7f42cacfd08ed780df979f6f28240026795b1f5c87ca8c16fc7da23dc58`; X SHA-256 is `faca9f62eb02a1da3d87cf39df20816a4a15b79806995b4225df68b008e0cbcb`. The Q4 metallib hash matches the passing joint model run. Fresh native verification precedes timing in the same installed app. No old fixture or legacy absolute Documents path supplies this campaign.

Each path has 30 warmups and 201 alternating pairs/round for 3 rounds, one dispatch per command buffer. Capture, diagnostic logs and tensor readback are off during timing. The reported metric is **completed `(GPUEndTime-GPUStartTime)*1e6`, GPU command-buffer elapsed microseconds**, not GPU instruction cycles. The fixed post-ActQuant X is precomputed outside the timer: quantizer preparation, CPU encoding, full decode and energy are excluded.

| Round | Stock p50 / p95 us | Q4 p50 / p95 us | Paired Q4 minus stock us | Median per-pair relative change | Q4 wins |
|---|---|---|---:|---:|---:|
| 1 | 225.0000 / 228.7917 | 218.7917 / 223.1250 | -5.9167 | -2.6089% | 195/201 |
| 2 | 224.7500 / 228.8750 | 218.2917 / 223.9583 | -5.9583 | -2.6754% | 197/201 |
| 3 | 224.5000 / 228.6250 | 217.9583 / 222.5000 | -5.8750 | -2.6509% | 196/201 |

A/A paired medians are -0.4167/-0.2500/-0.2083us. All AB/BA groups and all three chronological blocks in each stock/Q4 round favor Q4. Comparison order rotates; all samples remain in the local CSV. Across the entire nine-group campaign, all 18 thermal endpoints are nominal, charging at 100%, low-power mode off. Its approximately 2.90 seconds is **not** sustained-load or independent thermal-cohort evidence. The medians of per-pair percentages are distinct from ratios of marginal medians; A/A is not subtracted as a correction.

The observed benefit is about **2.61%-2.68% for this isolated Q operator on actual ActQuant-ON input**. New whole-decode and joint corpus/PPL remain UNMEASURED. The preserved harness also records three C8/Q4 groups as isolated research; C8 is not authorized by the new model route and receives no new model-acceptance claim. Detailed derived groups, order/time-block summaries and raw-file digests are in `SUMMARY.json`; raw samples remain local.

## Reference and claim boundaries

`parity_pass`, `fp64_pass`, and `model_output_pass` are true for the named joint Q-only gate. Here `model_output_pass` means residual/full-logit agreement with same-build **stock ON**, not agreement with the original Microsoft source execution. The joint analyzer does not recompute trusted-reference acceptance, full-Transformer FP64 or corpus/PPL. The earlier A16 source-derived eager F32 ON comparison was performed and failed strict 1; local quantizer correctness and Q4 parity do not erase that failure. Quality acceptance remains null.

Historical Q4, same-stock guard and FFN timing tables were collected with ActQuant OFF. Their 2.68%-3.11% isolated Q4 improvement and absent whole-decode gain cannot be transferred to this new ON graph. The new isolated-operator result above is measured on fresh ON input; **joint whole-model decode and corpus/PPL remain UNMEASURED**. No Mac performance is reported.

## Source and reproduction

Changed route checks: `src/models/bitnet.cpp`, `ggml/src/ggml-metal/ggml-metal-ops.cpp`. Shared runner: `bitnet/actquant/runner.cpp`; shader provenance: `q4_contract.py`; phone transport: `run_phone.py`; independent comparison: `analyze_q4.py`. Existing shader is `bitnet/shaders/stock4x1.metal`. Build from this source into a new signed output; do not reuse an older app/framework. Use the operator prerequisite and build commands in `../../docs/BUILD_AND_RUN.md` first.

Collect stock/Q4 captured, stock/Q4 plain and Q4 missing-library runs into separate fresh directories for each history. Set `$TOKENS` to the exact27+32 token file for that history. The launcher binds Q4 to the same signed build's stock4x1 metallib and checks hashes; it does not accept an arbitrary same-symbol C8 library.

```sh
"$PY" bitnet/actquant/run_phone.py --run --operation model \
  --build-dir "$SIGNED_BUILD" --device "$IPHONE_DEVICE_ID" \
  --model "$PUBLIC_2B_GGUF" --tokens "$TOKENS" --actquant on \
  --workload teacher --q-route Q4 --capture-scope q4-layer0 --capture --trace \
  --operator-run "$OPERATOR_RUN" --operator-analysis "$OPERATOR_ANALYSIS/SUMMARY.json" \
  --output "$ARTIFACTS/Q4-capture"
```

Use `--q-route stock` for the captured stock route; omit `--capture --trace` for plain controls. Add `--q-fallback missing-library` to a separate Q4 captured run for the fallback control. All runs use the same signed build and tokens. Then process the saved data without executing a model/GPU:

```sh
"$PY" bitnet/actquant/analyze_q4.py \
  --stock "$ARTIFACTS/stock-capture/device/native" \
  --candidate "$ARTIFACTS/Q4-capture/device/native" \
  --stock-log "$ARTIFACTS/stock-capture/logs/launch.log" \
  --candidate-log "$ARTIFACTS/Q4-capture/logs/launch.log" \
  --stock-plain "$ARTIFACTS/stock-plain/device/native" \
  --candidate-plain "$ARTIFACTS/Q4-plain/device/native" \
  --missing-library "$ARTIFACTS/Q4-missing-library/device/native" \
  --missing-library-log "$ARTIFACTS/Q4-missing-library/logs/launch.log" \
  --output "$ARTIFACTS/joint-analysis"
```

Expected scoped joint analyzer exit is 0 when every required gate passes. Complete evidence with a failed required gate returns 1; missing/malformed/invalid evidence, an `EvidenceError` contract rejection or malformed CLI arguments returns 2. Native/collection0 alone is not numerical acceptance. The separate trusted-source model analyzer's existing strict 1 is preserved.


## Reproduce the fresh operator timing and CPU audit

The fixture preparer requires an actual physical-A16 post-ActQuant graph capture, its passing joint analysis and the same signed build. It extracts steps 1/16/32; timing uses step 16. The legacy operator entry performs fresh verification before measurement in the same installed app and records the additional C8 research control separately.

```sh
"$PY" bitnet/actquant/prepare_q_fixture.py \
  --run "$ARTIFACTS/Q4-capture" \
  --acceptance "$ARTIFACTS/joint-analysis/SUMMARY.json" \
  --build-dir "$SIGNED_BUILD" --output "$ARTIFACTS/Q-fixtures"
python3 bitnet/ios/run.py --build-dir "$SIGNED_BUILD" \
  --device "$IPHONE_DEVICE_ID" --model "$PUBLIC_2B_GGUF" \
  --mode q-benchmark --fixture "$ARTIFACTS/Q-fixtures" --install \
  --output "$ARTIFACTS/Q-performance"
"$PY" bitnet/actquant/analyze_q_timing.py \
  --performance-run "$ARTIFACTS/Q-performance" \
  --fixture "$ARTIFACTS/Q-fixtures" \
  --joint-candidate "$ARTIFACTS/Q4-capture" \
  --output "$ARTIFACTS/Q-timing-analysis.json"
```

The portable timing audit is CPU-only and was independently reproduced with exit 0. An observed audit failure returns 1; missing/invalid evidence returns 2. An existing analysis file is never overwritten. Device identity/signing variables are supplied locally.

Local experiment identity: `actquant-q4-2026-11-16_084624+0900`, with phone subcampaign `phone-2026-11-16_090500+0900`. `SUMMARY.json` contains compact metrics, selected step1/16/32 W/X/Y digests and local evidence names/hashes. Raw arrays, logs, CSV, captures, models, signed builds and device/signing identities stay local. See `../actquant-iphone-2026-11-16/` for the distinct trusted-reference result.
