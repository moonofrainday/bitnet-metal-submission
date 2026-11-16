# Optional Mac selected-Q encoder diagnostic

The ordinary selected-Q value campaign uses the prepared backend without this patch. The source-only `bitnet/patches/mac-qcur0-encoder-diagnostic.patch` is an optional diagnostic against packaged base `578ec123ec2f0eed01f3edf6dc09ac6dcbe6104a`; isolated source commit `6c775b38cc0283ba126da3dc1890bfd9a2a6b864` contains exactly the four patched files. It changes no shader algorithm, and was not deployed to a phone.

## Why it exists

The M2 Pro exposes Metal stage-boundary timestamp sampling and a Shared timestamp buffer, but reports dispatch-boundary sampling as unsupported. Instruments launches did not produce usable normal inference traces. The first target failed with an AMFI CMS-signature message; a separately development-signed copy also failed to launch under Instruments, for an unresolved reason. Neither failure proves that normal dispatch profiling is impossible on this machine.

The diagnostic splits the current compute encoder into prefix / selected Q / suffix inside the same command buffer. Explicit fences preserve dependencies even for untracked buffers. It reuses the actual stock `kernel_mul_mv_tq2_0_f32`, stock ABI and setters. Application queue, command-buffer commits and ordinary completion waits are unchanged. Metadata and encoder objects still add host work; timestamp/fence pools are allocated at reset outside the wall timer.

`GGML_METAL_QCUR0_ENCODER_PROBE` defaults to `0`. Value `1` additionally requires existing attribution diagnostics and the exact layer-0 `Qcur-0`, N=1, TQ2_0/F32, post-ActQuant contract. QB/Q4 remain unsplit controls. Value `empty` samples a zero-dispatch encoder before the actual stock Q.

## Verified scope

The corrected fresh v2 build binds 1,073 native-source hashes and 20 products. Primary-history OFF/ON and independent-history ON verification passed all 32 Q checks, the sampled residual outputs and 33 logits. Primary OFF/ON saved payloads were also independently byte-compared. Local FP64 tolerance remains `abs(error) <= 2e-4 + 2e-5 * abs(reference)`.

Four ordinary target paths each contain 32 valid one-dispatch scopes and 64 original command buffers. Their p50 intervals are 24.459-24.792 us. Four empty paths each contain 32 zero-dispatch scopes; p50 is 0.500-0.541 us. These are diagnostic observations, not three independent paired performance rounds.

The analyzer verifies actual pipeline identity, decoded 120-byte ABI, buffer slots and bounds, grid 320, threads 32x2, unique completed-CB association and timestamp conversion. It requires no dropped records or pending callbacks. Numerical verification is a separate gate; a metrology-only exit 0 does not imply numerical or full-model acceptance.

CPU samples from `sampleTimestamps` are already nanoseconds. Convert raw GPU deltas using the measured before/after CPU/GPU bracket ratio, rather than assuming the timestamp query frequency is the raw counter's conversion factor:

```text
elapsed_us = (endRaw - startRaw) / (gpuAfter - gpuBefore)
             * (cpuAfter_ns - cpuBefore_ns) / 1000
```

Do not apply `mach_timebase_info` to those CPU samples a second time. Host worker timings, GPU counter readings and GPU command-buffer times remain distinct domains and scopes.

The diagnostic changes encoder boundaries, dependency scheduling, possible Q/K/V overlap, cache state and clock behavior. Its result is neither pure kernel busy time nor normal critical-path share, and is not a proven upper bound. The empty encoder cannot simply be subtracted as a universal correction. Separate OFF/ON overhead campaigns also cannot be subtracted across time to isolate a pure encoder cost. ALU, bandwidth, occupancy, register-pressure and stall counters remain unavailable.

## Rebuild separately

From this checkout, with a locally installed CMake and full Xcode:

```sh
LAB="$PWD"
git worktree add --detach ../bitnet-q0-probe 578ec123ec2f0eed01f3edf6dc09ac6dcbe6104a
git -C ../bitnet-q0-probe apply "$LAB/bitnet/patches/mac-qcur0-encoder-diagnostic.patch"
cd ../bitnet-q0-probe
python3 bitnet/tests/build_q_contract.py \
  --build-dir "$FRESH_BUILD" --cmake "$CMAKE" \
  --developer-dir /Applications/Xcode.app/Contents/Developer --jobs 4
```

Use fresh outputs and the same model, token history, F16 KV and ActQuant-ON settings as the qualified campaign. For numerical verification set `operation=attribution`, `attribution_kind=verify` and `probe_verify=true` in the runner config, then run OFF and ON separately. Run the existing independent stock/Q4 numerical analyzer after the native process. For ordinary metrology use the existing diagnostic protocol, then:

```sh
python3 "$LAB/bitnet/tests/analyze_mac_q_encoder_probe.py" \
  --diagnostics "$DIAGNOSTIC_JSON" --mode target \
  --expected-cbs 64 --output "$FRESH_RESULT_JSON"
```

For the empty control use `--mode empty`. Do not require 64 command buffers on callback-capture verification runs; that pre-existing capture schedule has more buffers. The original commands, configs, hashes and raw evidence are retained locally under `profiling-feasibility/encoder-probe/` in the selected-Q experiment. Only code, the optional patch and derived aggregates are published.

The initial v1 build was never GPU-qualified. Its defensive partial-pair bounds check was corrected before the v2 fresh build and all reported runtime tests. This diagnostic does not repair the preserved official-model reference failure (strict exit 1).

See the [selected-Q findings](../analysis/mac-qcur0-value-2026-11-16/README.md) for the independent ordinary benchmark. Apple's [timestamp-conversion documentation](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time) defines the clock calibration used here.
