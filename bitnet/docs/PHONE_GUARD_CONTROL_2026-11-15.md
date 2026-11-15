# A16 same-stock-pipeline guard control

This is an opt-in diagnostic extension for the current public-checkpoint QVAC compatibility path. It does not restore Microsoft's missing per-token INT8 activation semantics. Runtime results must be reported only after a physical-device run; this document initially records implementation and acceptance rules.

## Routes and unchanged behavior

- `S`: default native stock encoder.
- `QB`: `GGML_METAL_BITNET_Q_GUARD_ONLY=1`; run the same full `ggml_metal_try_bitnet_poc` guard used by Q4, including candidate-pipeline lookup and execution-width/thread-limit checks. Return before binding, selecting or dispatching that candidate. Then continue through the existing native stock encoder.
- `Q4`: existing layer-0 custom stock-helper 4x1 route, unchanged.

QB and custom Q flags are mutually exclusive. The shared guard checks layer identity, geometry configuration, exact types/ne/nb, no views, buffer existence, 32-byte alignment, live bounds, disjoint output/input ranges, candidate availability and pipeline limits. The outer gate retains operation/weight identity, one-token decode, selected layer, no fused add/norm, and stock-only suppression. Neither QB nor its successful guard changes graph indexing, barriers, encoder boundaries, buffer bindings or output storage. Stock uses its native six-slot ABI and original grid320 / threads[32,2,1].

The extra QB plan returns a boolean; it does not alter encoder state. A guard rejection may continue safely through stock, but is a FAILED experimental B control, not a valid guard-overhead measurement.

## Correctness acceptance

Run one fresh same-build graph verification with `S,QB,Q4`, layer0 and fixed 27/32 tokens; retain both histories when practical. Diagnostic readback is independent of timing.

The runner requires:

- QB guard passes at all32 decode steps, no guard pass in prefill and no custom QB dispatch.
- S and QB both960 stock Q and960 stock down encodes/completions across30layers and32steps; Q4 retains32 custom Q and928 stock Q.
- Same actual stock PSO pointers and cached pipeline handles, native120-byte arguments, six slots, W/X/Y/residual/norm bindings and live offsets, grid and threads for S/QB at every target step.
-33 complete logit vectors bitwise equal S/QB and repeated-route outputs unchanged.
- With `--q-readback`, every captured S/QB W/X/Y and residual hash matches (68files/path), finite coverage is checked and readback/non-readback logits are identical.
- Metal device Apple A16 GPU and actual31/31 layer offload.

These checks establish a matched framework control. They do not establish original-model semantic fidelity, held-out quality, full FFN FP64 acceptance or a speedup.

## Minimal reproduction

Run from this worktree. Supply fresh artifact paths, a signed build, device ID and the existing checksum-validated GGUF:

```sh
python3 bitnet/ios/run.py --build-dir "$BUILD" --device "$IPHONE_DEVICE_ID" --model "$GGUF" --mode graph-verify --history-dir bitnet/ios/histories/primary --routes S,QB,Q4 --layers 0 --q-readback --warmups 1 --install --reuse-model --output "$VERIFY"
python3 bitnet/ios/run.py --build-dir "$BUILD" --device "$IPHONE_DEVICE_ID" --model "$GGUF" --mode decode-benchmark --history-dir bitnet/ios/histories/primary --routes S,QB,Q4 --layers 0 --comparisons AA,SQB,SQ4 --warmups 1 --rounds 3 --pairs 7 --install --reuse-model --output "$TIMING"
python3 bitnet/tests/test_guard_evidence.py
```

The initial benchmark is explicitly a short screening protocol: three rounds, seven balanced pairs per contrast, A/A plus direct S/QB and S/Q4. It uses the existing reset/KV-clear/prefix-restore protocol and 32-token synchronized wall clock; reset/prefill is excluded and timing has no trace/readback. It is not the earlier201-pair isolated-kernel campaign. A/B guard cost cannot be inferred from a noisy short result, nor can Q4-B be called pure shader cost. CPU instrumentation and GPU range profiling are not added here.

No new threshold replaces V1. `parity_pass`, `fp64_pass`, `model_output_pass` remain distinct. Unmeasured FP64 fields remain null; `--strict-v1` therefore cannot silently claim full acceptance. Raw phone artifacts remain local.

## Changed code

- `ggml/src/ggml-metal/ggml-metal-ops.cpp`: guard-only return before all encoder mutation and native-ABI trace.
- `ggml/src/ggml-metal/ggml-metal-device.m`: actual PSO handle in diagnostic dispatch trace.
- `bitnet/ios/src/CandidateReplay.mm`: mutually exclusive QB route and SQB comparison, exact-logit gate.
- `bitnet/ios/run.py`: full live-evidence validator and optional captured-input/output hash parity.
- `bitnet/tests/test_guard_evidence.py`: reject missing/failed guards, missing step, changed offsets/PSO and encoded custom work.
