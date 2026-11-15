# Physical iPhone replay and operator harness

This source-only package carries the previously tested iPhone 14 Pro / Apple A16 replay design into this checkout. It reuses the pinned QVAC source and the compatibility/backend changes in this repository. It does not include a model, signing identity, device identifier, compiled framework, app, metallib, captured tensor, or raw benchmark result.

The fresh signed-build Q4 campaign is documented in [the 2026-11-15 result note](../docs/IPHONE_Q4_COMPLETE_2026-11-15.md). Historical G/F8 results belong to their archived builds. A new rebuild does not automatically inherit either result; record build and physical-phone verification separately.

## What is added

- `src/CandidateReplay.mm`: the V3 physical-device fixed-token driver, with S (stock), G (stock-arithmetic dual projection), K (dual projection plus in-place activation/weighted normalization), F8 (C8x2 gate/up plus squared-ReLU activation), and Q4 (layer-0 stock4x1 Q) routes.
- `src/iphone-q.mm`: an independent TQ2_0 unpack/FP64 Q reference, actual embedded stock pipeline selection, full-output verification, and paired operator timing.
- `src/iphone-operator.mm` and `alias-stock-ffn.h`: the separate FFN operator/control helper required by the V3 app. It uses the actual stock down pipeline.
- `src/llama_swiftuiApp.swift`: the small application hook. Model and Metal initialization run in a detached task after scene creation. This preserves the background initialization that resolved the historical launch-watchdog failure. The idle timer is disabled only while the experiment app runs.
- `histories/`: the two small fixed workload configurations: exact prompt text, 27 prompt token IDs, and 32 continuation IDs. They contain no captured tensor or output data.

The existing upstream SwiftUI application, FinetuneBridge, and LibLlama code are not our additions. The builder copies that application into the selected local build directory, appends a C declaration/include hook in the copy, and adds our replay source. Files under `examples/llama.swiftui` remain unchanged. `SOURCE_ORIGINS.json` identifies the original source artifacts; source hashes of each fresh build are recorded locally.

## Prerequisites

- macOS with full Xcode, the iPhoneOS SDK, and the Metal compiler installed.
- Python 3.9+, CMake, and the normal Apple command-line tools.
- For installation: a connected, paired, unlocked physical iPhone with Developer Mode, a user-selected development team, and a trusted development profile.
- The public converted TQ2_0 GGUF used by the experiment, obtained separately. Expected SHA-256:

  `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2`

The model is the public Microsoft BitNet 2B compatibility experiment, not the team's custom checkpoint. I2_S is not accepted as TQ2_0. Conversion and model identity belong to the repository's model-compatibility documentation.

## Fresh unsigned build

From this repository root:

```sh
python3 bitnet/ios/build.py \
  --unsigned \
  --developer-dir /Applications/Xcode.app/Contents/Developer \
  --output "$HOME/bitnet-local/ios-unsigned-$(date +%Y%m%d-%H%M%S)"
```

The output directory must not already exist. The builder configures fresh iPhoneOS arm64 CMake archives from **this checkout**, with Metal and embedded Metal library enabled; merges the required archives; links the dynamic `llama.framework` expected by SwiftUI; creates an iPhoneOS-only XCFramework; compiles all experimental shader sources for iPhoneOS; compiles the Q, FFN and Down-control Objective-C++ helper objects; stages the SwiftUI project; and builds it with `CODE_SIGNING_ALLOWED=NO`. It does not install or run anything.

The final framework linkage follows the tested upstream packaging pattern: static backend/llama libraries are combined into a dynamic framework for application embedding. No Simulator, Mac framework, or downloaded old dylib is reused. `--framework` may reuse a framework produced by this builder only when its adjacent `framework-provenance.json` matches the current source-file hashes and binary hash. Otherwise it fails.

A signed build uses the caller's team and a distinct fresh output directory:

```sh
python3 bitnet/ios/build.py \
  --team "$APPLE_TEAM_ID" \
  --bundle-id local.bitnet.submission.smoke \
  --allow-provisioning-updates \
  --output "$HOME/bitnet-local/ios-signed-$(date +%Y%m%d-%H%M%S)"
```

`DEVELOPER_DIR`, `APPLE_TEAM_ID`, and `BITNET_BUNDLE_ID` are optional environment inputs. No personal team, keychain access, certificate, provisioning profile, or device identifier is embedded in this package. An unsigned build cannot run on a physical device.

Each build leaves `BUILD.json`, exact commands, compiler/build logs, command exit codes, source hashes, object hashes, framework hashes, app hashes, and offline shader metadata in its local output directory. A failed stage returns nonzero and leaves the failure record. Offline compilation is not reported as GPU execution or correctness.

## Real-model fixed-token verification

First inspect the connected device identifier yourself or with:

```sh
xcrun devicectl list devices
```

Keep the phone unlocked. Use the signed build's actual output directory and the exact public GGUF path:

```sh
python3 bitnet/ios/run.py \
  --build-dir "$IOS_BUILD" \
  --device "$IPHONE_DEVICE_ID" \
  --model "$BITNET_2B_GGUF" \
  --mode graph-verify --routes S,Q4 --layers 0 \
  --history-dir bitnet/ios/histories/primary \
  --install \
  --output "$HOME/bitnet-local/phone-q4-primary-$(date +%Y%m%d-%H%M%S)"
```

Repeat with `histories/independent`. G can be checked separately with `--routes S,G --layers 0-29`; F8 uses `--routes S,F8 --layers 0-29`. K is a research route with known guard rejections; the archived all-layer campaign explicitly expected layers 2 and 29 to retain stock, so its reproduction must declare `--expect-fallback-layers 2,29` and audit actual rejection logs.

Every invocation requires `--install` and installs the exact signed app selected by the host `BUILD.json` before launching it. This prevents a previous app with the same bundle identifier from silently receiving a new host configuration. Host file hashes are checked, but no on-device executable hash is claimed. Reinstallation can change the absolute data-container UUID even when Documents contents survive. Q timing therefore performs fresh qualification and timing in one installation, as described below.

The runner verifies the local model hash, transfers it to the app's Documents container, transfers freshly compiled iPhoneOS libraries, and launches with a JSON configuration in `BITNET_CANDIDATE_CONFIG`. The app independently checks the model/library hashes. `--reuse-model` skips transfer only; it does not skip the on-phone model hash check.

The app uses context 512, batch/ubatch 128, six CPU threads, F16 K/V, disabled flash attention, stock prefill, the fixed 27-token prompt, and 32 replayed continuation IDs. Each replay synchronizes, clears KV data/metadata, restores the prompt, and checks sequence positions. It is fixed-token replay, not free generation or model-quality evaluation.

`graph-verify` requires actual Metal offload (31/31), the Apple A16 GPU, application exit 0, a completion record, route repeatability, exact eligible-layer/32-step dispatch counts, matching GPU completion counts, stock Q/down coverage, and no custom prefill dispatch. Q4 additionally requires frozen-tolerance full-logit agreement with stock; G requires bit-identical logits. Actual dispatch logs remain the evidence that the custom route ran. The default run has no pre-RoPE tensor copies; the explicit correctness-only readback below is separate from timing and from a full-model FP64 reference.

## Complete layer-0 Q readback and probe control

Run the same signed build and history once without `--q-readback`, then once with it, using fresh host outputs. The instrumented first S/Q4 traversals copy the actual Q input/output for all 32 decode calls, the weight once per route, and `l_out-0` at steps 1/16/32. Prefill, warmup and the repeat traversals have this probe disabled. Every decode synchronizes and waits for diagnostic GPU completion/file writes. No graph-slicing callback is added.

```sh
python3 bitnet/ios/run.py \
  --build-dir "$IOS_BUILD" --device "$IPHONE_DEVICE_ID" --model "$BITNET_2B_GGUF" \
  --mode graph-verify --routes S,Q4 --layers 0 --install --reuse-model \
  --history-dir bitnet/ios/histories/primary --q-readback \
  --output "$HOME/bitnet-local/phone-q4-readback-primary-$(date +%Y%m%d-%H%M%S)"
```

Repeat with the independent history. Each route has 68 payload files: `weights/w.tq2`, `step-1` through `step-32` with `x.f32`/`y.f32`, and `l_out.f32` at steps 1/16/32. `manifest.json` binds every relative file name, byte length and SHA-256. The Q output is captured before RoPE. The host runner checks all expected files, hashes and finite F32 values, and independently requires the 33 first/repeat full-logit vectors to match bit for bit. This checks the observed logit effect of enabling the probe; it does not claim unobserved internal tensors are unaffected.

Each host output includes `BUILD_IDENTITY.json` to bind the selected build manifest, framework and application binary hashes. Use it to verify the separate probe-off/on runs used the same executable. Independently compare captured W/X before same-input arithmetic comparison, then evaluate all Q outputs against the FP64 reference and the unchanged tolerance. Raw coverage alone leaves `fp64_pass` null. The complete FFN and model-quality limitations remain unchanged.

Two independent fallback controls are available with `--q-fallback missing-library` or `--q-fallback invalid-geometry`, without `--q-readback`. The first verifies the real library identity and then points the backend to a fresh nonexistent path; the second requests three rows/SIMD, which the guard rejects before dispatch. Each requires zero custom Q calls, all 960 stock Q calls, 32 matching fallback reasons and stock-relative full-logit agreement. Unsupported geometry is not a real stride-mismatch test. Each control must run in a fresh application process; neither counts as a successful custom dispatch.

## Independent Q operator correctness and timing

Captured W/X are intentionally not in Git. Supply an existing local fixture directory with `MANIFEST.json` and `step-{1,16,32}/{w.tq2,x.f32,y.f32}`. The manifest binds each file hash and the public model identity. The runner stages a local copy and updates only the two newly compiled library hashes; the original fixture remains unchanged.

```sh
python3 bitnet/ios/run.py \
  --build-dir "$IOS_BUILD" --device "$IPHONE_DEVICE_ID" --model "$BITNET_2B_GGUF" \
  --mode q-verify --fixture "$PHONE_Q_FIXTURES" --install \
  --operator-fixture candidate-q-input-v1 --operator-output candidate-q-check-v1 \
  --output "$HOME/bitnet-local/phone-q-verify-$(date +%Y%m%d-%H%M%S)"

python3 bitnet/ios/run.py \
  --build-dir "$IOS_BUILD" --device "$IPHONE_DEVICE_ID" --model "$BITNET_2B_GGUF" \
  --mode q-benchmark --operator-fixture candidate-q-input-v1 --install \
  --reuse-model \
  --output "$HOME/bitnet-local/phone-q-timing-$(date +%Y%m%d-%H%M%S)"
```

Verification sentinel-fills and checks every one of 2,560 outputs at all three steps, for actual embedded stock, C8x2, and stock4x1. The independent CPU code unpacks TQ2_0 and accumulates in FP64. The unchanged tolerance is `abs(actual-reference) <= 2e-4 + 2e-5*abs(reference)`; zero failed/nonfinite rows are required. It reports maximum absolute error, relative L2, stock bit differences, row-level data, hashes, and GPU completion. A fixture does not establish whole-model acceptance.

Both `q-verify` and `q-benchmark` require a fetched `qualification.json` with all nine step/route cases (three steps times three routes), 2,560 rows/case, zero failed/nonfinite rows, completed GPU work, stock-bit parity for S/Q4, unchanged tolerances, and current shader/input hashes. A functional completion marker cannot replace this artifact.

Q benchmarking installs the selected app exactly once, creates a fresh unique operator output, runs native nine-case verification, fetches and validates its qualification and actual app exit 0, and then launches native timing without reinstalling. The unchanged native timing gate still compares exact input-path/hash and library-hash dictionaries. This avoids stale absolute data-container paths after reinstallation and qualifies the selected backend build rather than an earlier binary. Existing qualification/performance folders remain untouched; a requested old `--operator-output` is recorded but replaced by a fresh name. `q-benchmark-protocol.json`, separate preflight logs, `q-preflight-qualification.json` and `BUILD_IDENTITY.json` bind both phases.

An existing device `--operator-fixture` is sufficient; alternatively provide `--fixture` to transfer local inputs. A separate earlier `q-verify` remains useful for inspection but is optional for timing because benchmarking repeats it in the same installation. Missing/failed qualification or an unsuccessful verification app exit stops before timing. Qualification is outside the timed interval.

Timing fixes step-16 W/X, uses 30 warmups per logical path, 201 balanced alternating pairs per round, and three rounds. It rotates S/S, S/Q4, and C8/Q4 contrasts and records thermal, low-power, and battery snapshots. Each command buffer contains exactly one dispatch. The metric is `GPUEndTime-GPUStartTime` after completion, in microseconds: **GPU command-buffer elapsed time**, not shader-only active time or model throughput. Capture, diagnostic logging, and tensor readback are disabled during timing.

The FFN helper is also available with `ffn-reference`, `ffn-verify`, and `ffn-benchmark`, taking its historical `weights/layer-*` and `step-*/layer-*` capture layout. Its local research qualification is distinct from complete-chain FP64 acceptance. Detailed all-layer F8 FP64 analysis belongs to the archived numerical tools; it is not silently replaced by the smaller V3 helper.

## Whole-decode research timing

`--mode decode-benchmark` reuses one model/context, excludes KV reset and prompt restoration from the measured 32-token synchronized host-wall interval, prewarms routes, and balances/rotates contrast blocks. Example compatible-route contrast: `--routes S,G --layers 0-29 --comparisons AA,SG --rounds 3 --pairs 21`. After the corresponding correctness checks, layer-0 Q4 can use `--routes S,Q4 --layers 0 --comparisons AA,SQ4 --warmups 2 --rounds 3 --pairs 21`. This is complete 32-token decode wall time, distinct from the operator's GPU command-buffer metric. Q readback, fallback controls and diagnostic trace are forbidden during formal timing. Historical Q4 results below did not include this new whole-decode contrast.

K/F8 decode timing requires `--research-gate`. This external JSON must bind the app hashes, relevant library hashes, model, layer scope, requested routes, and prior evidence hashes. It must state `research_timing_pass=true`, `fp64_pass=false`, and `strict_exit_code=1`. This explicitly preserves the known numerical failures. Timing authorization is not numerical acceptance, and making a gate is not a substitute for collecting its evidence.

## Exit codes and scope

- Build: 0 only when all requested compilation/link/build stages complete; otherwise nonzero.
- Runner: 0 only when the selected mode's execution and scoped checks complete. 1 means a required check, application, transfer, fetch, or tool stage failed. Missing runtime results are errors, not passes.
- `summary.json` keeps `parity_pass`, `fp64_pass`, and `model_output_pass` separate with their scope. Unchecked values are null. A local Q FP64 pass does not imply complete-model FP64 acceptance.
- `strict_v1_exit_code` remains 2 when required frozen data are missing, or 1 when a measured frozen check fails. This portable package does not yet collect all V1 mandatory model data and cannot claim a V1 pass. `--strict-v1` makes this strict code the process result rather than the selected mode's functional result.
- App errors and completion records are fetched after launch failure where possible. Logs preserve the exact `devicectl` exit, actual application-exit text, app error, and source of the completion record. Console stdout/stderr can interleave; a fetched `result.json` is authoritative but does not replace the application-exit check.

All model files, app/framework products, raw logs, tensor readbacks, row CSVs, timings, and device metadata belong in local artifact directories outside Git. Only source, workload configuration, and analysis summaries belong in the submission.

## Historical phone evidence, not new measurements

The public experiments used a physical iPhone 14 Pro / Apple A16 GPU, iOS 27.2, Xcode 27.2, and an iPhoneOS 27.2 SDK. The framework was Release; the signed UI harness was Debug. These are test configurations, not a production performance ceiling.

| Candidate | Verified historical scope | Historical performance and limitations |
|---|---|---|
| Q stock4x1 | All 7,680 sampled Q outputs passed FP64 and were stock-bit-identical; both histories had 32/32 layer-0 dispatches and 33 stock-identical logits vectors | Paired isolated-operator reduction 2.265%-3.291%, 599/603 wins. All thermal snapshots serious. No whole-model Q4 speed result or complete 32-call Q readback. |
| G | Both histories: 960 dual-projection plus 960 post dispatches; 33 full logits vectors matched stock bit for bit | Stock-output-compatible candidate; complete FFN FP64 was not repaired. |
| F8 | 3,840 same-input own-stream cases across two histories, all 30 layers and 32 steps; 11,520 replay command buffers completed. Fused/split-C8 activation, norm and down were bit-exact | Stock and F8 both failed the frozen complete-FFN FP64 standard; F8 retained stock-relative model-output failures. Strict numerical exit 1. Hidden fused gate/up were not materialized and remain unverified. |
| F8 vs G | Direct whole-model research campaign: 126 G/F8 pairs plus 126 G/G calibration pairs | F8 won 123/126; paired-relative median reduction 1.961%-3.568%. All direct G/F8 samples serious-thermal, with substantial absolute latency drift. No same-campaign formal stock/S comparison or pure fusion-cost attribution. |

Evidence origins in the separate experiment archive:

- `2026-11-12_202159+0900-iphone-candidate-sequence/{Q_RESULTS.md,SHADERS.md,PHONE_CORRECTNESS.md}`.
- `2026-11-12_231133+0900-iphone-f8-numerics/README.md` (primary numerical campaign).
- `2026-11-13_080244+0900-iphone-independent-gf8/{README.md,PERFORMANCE.md}` (independent completion and direct timing).

Raw evidence remains local by request. The archival preparation README predates actual Q-v3 activation; final `Q_RESULTS.md` and device artifacts determine the measured scope. No Mac timing is included as phone performance.


## Selected down numerical controls

The optional down-verify mode tests twelve selected eight-row groups (96 outputs per route) using one actual independent-history layer-0/step-16 stock normalization input. It runs the actual stock pipeline, a stock-helper wrapper, and a correction-order control. It is a diagnostic operator experiment, not full-FFN repair or model inference.

Build the app with the additional iphone-down-controls helper object, then provide the prepared local fixture and matching locally compiled libraries:

    python3 bitnet/ios/run.py --mode down-verify --build-dir "$IOS_BUILD" --device "$IPHONE_DEVICE_ID" --model "$BITNET_2B_GGUF" --fixture "$DOWN_FIXTURE" --down-libraries "$DOWN_LIBRARIES" --install --reuse-model --output "$DOWN_OUTPUT"

The helper writes all 36 case summaries and row outputs before returning. Native/app exit 1 denotes a completed measured FP64 failure; native exit 2 denotes an execution error. The runner accepts no deployment/launch failure as a numerical result: it requires the completion marker, fetched native summary, actual matching app exit, complete route/group coverage, 36 successful command buffers, consistent per-row failure counts, intact canaries and unchanged inputs. Nonfinite outputs remain numerical failures. A completed numerical failure retains execution_pass=true, fp64_pass=false, overall_pass=false and strict exit 1. This mode does not measure performance or grant complete model acceptance.
