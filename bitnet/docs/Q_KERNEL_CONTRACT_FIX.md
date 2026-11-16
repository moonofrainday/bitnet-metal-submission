# Q kernel launch and pipeline cache contract

The fix closes a real mismatch between optional host geometry and compiled Q shaders. It adds distinct versioned entries, a canonical descriptor and file-aware external-pipeline caching. Fresh Mac operator and two-history layer-0 Q4 regressions pass. New signed and unsigned iPhoneOS builds compile successfully; **this new binary has not been installed or run on the phone**. No new performance conclusion is available.

The baseline is `6fa8468ec1a28bb47aa22484125ef1337dbb898b`, with QVAC pinned at `2db8b046d555316019ae75d1be90484adc8b9330`. The source fix commit is `8ebcdb78680829c5453e0c7c72770c0b8e80225a`. The binaries were built from this matching dirty source snapshot before that commit; their embedded Git metadata still names baseline `6fa8468`. Compiled source hashes, rather than the old embedded label, bind the results. The public GGUF remains SHA-256 `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2`.

## Why the change is necessary

The previous low-level ActQuant-OFF route accepted rows 4 with one, two or four SIMD groups. The supplied stock4x1 shader fixed its internal group multiplier at 1. A host 4x2/grid 320 launch of that shader covered only 1,284 distinct outputs, left 1,276 unwritten and issued duplicate writes. The old Q4 and C8 products also exported the same symbol, so the function-name-only cache could select a prior library after a path change. A wrong C8 library with Q4 geometry could index beyond W/Y.

This did not describe the previously qualified ActQuant-ON run: its graph/runtime guards forced 4x1 and the supported launcher bound the correct source/product hashes. It nevertheless exposed an unsafe optional backend configuration and a cache identity defect, so the backend now rejects the combination directly.

## Canonical entries and dispatch

`ggml/src/ggml-metal/ggml-metal-bitnet-q-config.h` is the shared macro-only configuration. Host selection, shader geometry and build metadata derive from it.

| Variant | Versioned entry | Rows/SIMD | SIMD groups/threadgroup | Grid | Threads/threadgroup |
| --- | --- | ---: | ---: | --- | --- |
| `stock4x1` | `tq2_0_matvec_stock4x1_v1` | 4 | 1 | 640,1,1 | 32,1,1 |
| `c8x2` | `tq2_0_matvec_c8x2_v1` | 8 | 2 | 160,1,1 | 32,2,1 |

Both contracts require W TQ2_0 `[2560,2560]` and F32 X/Y `[2560,1]`. `GGML_METAL_BITNET_POC_VARIANT` optionally selects a variant; any supplied legacy rows/groups must agree. When the variant is unset, only exact legacy 4x1 or 8x2 is inferred. All-unset retains the old 8x2 selection. Unsupported 4x2/4x4, mixed variants and malformed values reject before custom encoding and preserve stock execution.

`ggml-metal-bitnet-q-contract.h` implements production selection, descriptor validation, launch checks and file/cache-key helpers. `ggml_metal_try_bitnet_poc` derives launch dimensions from the descriptor. The backend getter validates the descriptor again; a caller cannot pair a C8 entry with Q4 geometry merely by constructing a struct. Existing node/weight identity, N1/add/norm, ne/nb, view, alignment, bounds/disjointness and PSO resource checks remain. ActQuant ON additionally retains permission, layer 0 Q4-only and actual quantizer-producer/epsilon checks. Its unsupported startup combinations reject; they are not reported as successful runtime fallback.

The shader helper bodies remain unchanged. Offline AIR controls are equal after entry/source-filename metadata normalization; that is static evidence, not a hardware timing or resource claim. Old shared-name metallibs are **not runtime aliases** of the new entries.

## Pipeline cache identity

Before every optional external lookup, the backend resolves the current canonical path and reads its file identity. The cache key includes canonical path, unique entry, version/geometry and device/inode/size/nanosecond mtime/ctime. Missing files cannot reuse the previous PSO. Canonical aliases of the same file can reuse it; copied files and replacements get different identities.

On a miss, the loader reads immutable bytes, checks identity around the read, compiles from a dispatch-data copy and checks identity again before insertion. The same small cache change applies to optional FFN external loaders, while their arithmetic, ABI, entries and resource requirements remain unchanged. Default stock dispatch does not use this loader.

The hot path performs metadata queries, not a full-file hash. This adds unmeasured CPU filesystem/lock work to optional graph encoding. File metadata is cache invalidation, not cryptographic authentication or an atomic lock against concurrent/adversarial writers. Keep libraries frozen during a run and retain source/build/metallib hashes. Old identities remain allocated until backend teardown to protect in-flight users; a backend recreated between campaigns avoids accumulating reload history.

## Completed evidence and retained failure

| Check | Completed result | Scope |
| --- | --- | --- |
| Shared production CPU helpers | 10,347 checks pass | Selection, complete coverage, descriptor rejection, path/key identity and real filesystem changes |
| Offline compiler / schema controls | 20 wrong compiler configurations rejected; 12 schema tests pass | No GPU execution |
| Actual Mac getter/cache tests | 3 independent passing 18-case campaigns, plus 18/18 on the portable clean-build entry | Unique entries, wrong-library rejection, aliases, changed/deleted paths, restoration and forged descriptors; no target Q dispatch |
| Actual Mac GGML graph controls | 15 cases; all 38,400 outputs finite and within FP64 bound; canaries/inputs unchanged | Valid stock/Q4/C8, unsupported geometry, wrong libraries and same-path replacement/deletion; requested dispatch/fallback audited separately |
| Metal backend regressions | TQ2_0 MUL_MAT59/59; ACT_QUANT16/16 | M2 Pro MTL0 correctness |
| Real ActQuant-ON model graph | Two histories; 32/32 actual Q4 dispatches/completions each; nine scoped checks pass, exit 0 | Layer 0 Q-only; all 32 decode dots FP64, stock parity, prefill/non-target stock, probe neutrality and missing-library fallback |
| iPhoneOS builds | Fresh unsigned 0 and signed 0; source/product hashes recorded | Compilation/signing only; no phone installation or execution |

One failure is deliberately retained: the initial actual backend cache test reached 5 cases and failed `Q4_unique_entry_load` with native exit 1. Its generic warning did not identify which internal condition failed. Adding detailed warning fields was followed by 18/18 passes in the diagnostic run and two fresh repeats; the standalone load and CPU read/identity probes also pass. **The initial failure remains UNEXPLAINED. Logging is not claimed to have fixed its root cause.** Its original result/log and binary provenance remain local.

For each real-model history, 32 custom calls complete, 30 prefill Q calls and 928 non-target decode Q calls remain stock, and all 990 Q encodes complete. Q4 and same-build stock ON use identical W/X; all 32 N1 outputs per route pass `abs(error) <= 2e-4 + 2e-5*abs(reference)`. Prefill is parity-only. All 33 captured Q tensors and 33 complete logits match stock bitwise; the formal residual check samples steps 1/16/32. Independent archive checks are indexed in the derived analysis.
Independent CPU archive verification also checks all 33 captured `l_out-0` vectors, all 33 Q X/Y capture pairs, quantizer input/output pairs, full logits, and new stock ON against the preserved prior stock ON. They match bitwise on both histories. An independent absolute-K TQ2_0 unpack/FP64 evaluation reproduces zero failures over all 32 N1 dots per route/history. The Q-input quantizer oracle checks 33 captures/151,040 values per route on their own actual X. Actual packed-W payloads from new stock, Q4 and prior stock are hashed separately, not accepted only from metadata.

The first independent CPU oracle attempt is also preserved: casting rounded floating codes to int8 before dequantization discarded negative-zero signs. Correcting that CPU oracle restored the official floating-code dequantization behavior; no native code, shader or frozen tolerance changed. This is an analysis-tool correction, not a model or quantizer repair.


This is stock compatibility, not faithful Microsoft-model execution. The earlier trusted-reference full-model comparison remains strict exit 1; FFN/full-chain failures are not erased. C8 retains isolated arithmetic research status and does not acquire ActQuant-ON graph permission from these tests.

## Reproduce with fresh local destinations

Set `$PY` to the project Python environment, `$ARTIFACTS` to a new local directory outside the source checkout, `$BUILD` to a nonexistent Mac build directory, and `$DEVELOPER_DIR` to full Xcode Contents/Developer. The exact executed commands and immutable inputs are retained in the local experiment `q-contract-fix-2026-11-16_123607+0900`; no raw payload is embedded here.

```bash
"$PY" bitnet/tests/run_q_contract_cpu.py --output "$ARTIFACTS/cpu-contract"
"$PY" -m unittest bitnet.test_q_kernel_contract
"$PY" bitnet/scripts/build-shaders.py --sdk macosx \
  --output "$ARTIFACTS/shaders-macosx" --emit-air
"$PY" bitnet/tests/build_q_contract.py --build-dir "$BUILD" \
  --developer-dir "$DEVELOPER_DIR" --jobs 4
"$PY" bitnet/tests/run_q_pipeline_contract.py --run --build-dir "$BUILD" \
  --shaders "$ARTIFACTS/shaders-macosx" --output "$ARTIFACTS/cache-runtime"
"$BUILD/bin/test-backend-ops" test -b MTL0 -o MUL_MAT -p 'type_a=tq2_0'
"$BUILD/bin/test-backend-ops" test -b MTL0 -o ACT_QUANT
```

The clean builder refuses any existing build directory and records exact commands, source/product hashes and `FRESH_CORE_BUILD.json` only after a successful frozen-source build. Add `--cmake /absolute/path/to/cmake` if CMake is not on PATH; select full Xcode through `$DEVELOPER_DIR`. The recipe preserves the actual native default rather than claiming an unexecuted CPU compilation setting.

The cache runtime entry creates real Metal pipelines; `--run` is explicit. Without it, the wrapper performs build preparation only. A 15-case graph driver is in `bitnet/tests/q_graph_contract.cpp`, linked with `bitnet/validation/reference.cpp`; its independent saved-output entry is:

```bash
"$PY" bitnet/tests/analyze_q_graph_contract.py \
  --run "$GRAPH_RUN" --output "$ARTIFACTS/graph-analysis"
```

For a full replay, generate native config with `q4_library_identity` from the **new** stock4x1 source/build product, preserve the original 27+32 token file, and collect stock/Q4 capture/plain plus missing-library controls in fresh directories. `bitnet/actquant/analyze_q4.py` defaults to `--contract-profile canonical-v1`. Use the complete command recipe in [the joint qualification note](../analysis/actquant-q4-joint-2026-11-16/README.md), substituting the new signed build and entries. The native runner and phone launcher bind entry/version/geometry and source/build/product hashes, and clear inherited selectors.

Historical artifacts require an explicit analysis-only profile:

```bash
"$PY" bitnet/actquant/analyze_q4.py \
  --stock "$OLD_STOCK" --candidate "$OLD_Q4" \
  --stock-log "$OLD_STOCK_LOG" --candidate-log "$OLD_Q4_LOG" \
  --stock-plain "$OLD_STOCK_PLAIN" --candidate-plain "$OLD_Q4_PLAIN" \
  --missing-library "$OLD_FALLBACK" --missing-library-log "$OLD_FALLBACK_LOG" \
  --contract-profile legacy-shared-v0 --legacy-shader-build "$OLD_SHADER_BUILD_JSON" \
  --output "$ARTIFACTS/legacy-analysis"
```

That flag preserves auditable old evidence; it cannot load old entries into the new runtime or qualify a new binary. Missing required evidence remains null/exit 2, observed failure remains exit 1, and scoped pass is exit 0.

## Evening iPhone gate and performance boundary

Use a fresh source-bound signed build. The one-entry launcher plans 15 serialized stages: actual 18-case A16 cache/entry inspection, same-build 20-case ActQuant prerequisite with independent analysis, and two histories with stock/Q4 capture/plain/missing-library plus canonical scoped analysis. The saved daytime `--plan` completed without device calls.

```bash
"$PY" bitnet/tests/run_q_fix_phone.py --plan \
  --build-dir "$SIGNED_BUILD" --device "$IPHONE_DEVICE_ID" --model "$PUBLIC_2B_GGUF" \
  --operator-fixtures "$ACTQUANT_FIXTURES" \
  --tokens-primary "$TOKENS_PRIMARY" --tokens-independent "$TOKENS_INDEPENDENT" \
  --output "$ARTIFACTS/evening-plan"
```

When the evening phone session is intentionally started, replace `--plan` with `--run` and choose another fresh output directory. No timing stage is included. Only after correctness passes should phone operator and graph timing be repeated.

The previous 2.61%-2.68% A16 operator reduction belongs to the **earlier** joint binary. It is not a performance result for this fix. No Mac performance campaign, new phone performance or new full-decode throughput was measured here. New cache metadata work must be included in any later graph/CPU timing; isolated single-dispatch timing excludes it and quantizer preparation.

[Derived results and artifact hashes](../analysis/q-contract-fix-2026-11-16/README.md) retain the exact scope, failures and source snapshot. Raw arrays, logs, CSVs, captures, models, signing identities and applications remain local.

Additive patch: [`q-kernel-contract-fix.patch`](../patches/q-kernel-contract-fix.patch) (SHA-256 `c76dc42ff36c837614b27794e9ca9ca6848438a6a3393e03e3bec0f5dc0763d1`). It is already applied on this branch and applies to submission baseline `6fa8468ec1a28bb47aa22484125ef1337dbb898b`, not directly to pristine QVAC.
