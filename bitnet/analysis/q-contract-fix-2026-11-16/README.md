# Q launch contract and pipeline cache fix — 16 November 2026

The optional Q backend now binds a canonical variant, unique shader entry and launch geometry, and keys external pipelines by the current file identity. Fresh M2 Pro verification passes; new iPhoneOS builds are ready but have **not** been deployed. The initial unexplained loader failure and all prior numerical failures are retained.

The baseline is `6fa8468ec1a28bb47aa22484125ef1337dbb898b`, QVAC `2db8b046d555316019ae75d1be90484adc8b9330`. The source fix commit is `8ebcdb78680829c5453e0c7c72770c0b8e80225a`. Builds used its exact source snapshot before commit, so embedded Git metadata still names baseline `6fa8468`; source/build/product digests bind the tests. This directory contains derived analysis only. Local experiment: `q-contract-fix-2026-11-16_123607+0900`.

## Change and scope

The former legacy host accepted 4x2/4x4 while the stock4x1 shader internally assumed one group. The old Q4/C8 shared symbol and name-only cache also allowed wrong-library/stale-PSO ambiguity. The new immutable contract uses `tq2_0_matvec_stock4x1_v1` (4x1, grid 640) and `tq2_0_matvec_c8x2_v1` (8x2, grid 160). The host validates descriptor identity and derives launch from it; the backend getter revalidates. Cache keys bind canonical path, entry, contract and stat identity before lookup; immutable read snapshots are checked on load. Missing/changed files cannot simply reuse an earlier name-only PSO.

The existing shader arithmetic is preserved. ActQuant-ON still permits only layer 0 Q4 with its actual quantizer producer and original layout/buffer guards. Default stock remains stock. Optional FFN arithmetic/ABI remains unchanged, although its shared external loader gains file-aware invalidation.

## Measured results

| Evidence | Result | Interpretation |
| --- | --- | --- |
| Production CPU helper test | 10,347 checks pass | Exact tuples, coverage, false descriptor, path aliases and real file mutation/deletion |
| Offline negative compiles / schema | 20 expected compiler rejections; 12 schema tests pass | No GPU claim |
| Initial actual getter run | Exit 1 at case 5, valid Q4 load failed | **UNEXPLAINED**, preserved |
| Diagnostic getter +2 fresh repeats | 18/18 each, three passing campaigns; portable clean-build entry adds another 18/18 | Actual Mac PSO/cache checks, no target Q dispatch |
| Actual GGML graph | 15 cases; 38,400 outputs; 0 FP64/nonfinite failures; canaries/inputs unchanged | Valid Q4/C8 and unsafe legacy/wrong-library/same-path controls |
| Backend correctness | 59/59 TQ2_0; 16/16 ActQuant | M2 Pro MTL0 |
| Two real-model histories | Each 32/32 Q4 dispatch/completion; nine required scoped gates pass; exit 0 | ActQuant ON, layer 0 Q-only, not full trusted-reference acceptance |
| iPhoneOS arm64 builds | Unsigned 0; signed 0 | No phone installation/run for this fix |

The initial failure did not report its internal failed condition. The only subsequent loader source change added diagnostic WARN fields. Standalone loading and file-read/identity probes pass, and three diagnostic campaigns pass. These observations do **not** establish a root cause or a logger-induced repair. The failing result remains in `mac-pipeline-runtime/runtime/RESULT.json`.

For each history, all 81,920 decode values per route pass the frozen FP64 rule; Q4/stock use identical W/X. Maximum absolute error is 1.267946573e-5 primary and 1.267150037e-5 independent. All 33 Q captures and 33 full logits are stock-bitwise; the formal residual gate samples 1/16/32. Candidate counts are 32 custom, 30 stock prefill and 928 non-target stock decode, 990 completed Q calls total. Probe neutrality and 32 missing-library fallbacks pass. Prefill remains parity-only, not a new FP64 pass.


Independent CPU archive verification also checks all 33 captured `l_out-0` vectors, all 33 Q X/Y capture pairs, quantizer input/output pairs, full logits, and new stock ON against the preserved prior stock ON. They match bitwise on both histories. An independent absolute-K TQ2_0 unpack/FP64 evaluation reproduces zero failures over all 32 N1 dots per route/history. The Q-input quantizer oracle checks 33 captures/151,040 values per route on their own actual X. Actual packed-W payloads from new stock, Q4 and prior stock are hashed separately, not accepted only from metadata.

The first independent CPU oracle attempt is also preserved: casting rounded floating codes to int8 before dequantization discarded negative-zero signs. Correcting that CPU oracle restored the official floating-code dequantization behavior; no native code, shader or frozen tolerance changed. This is an analysis-tool correction, not a model or quantizer repair.

In the independent15-case operator graph, the valid C8 output differs from stock in 2,533 rows while passing local FP64 (max 5.761347e-6). Stock/Q4 max error is 8.837353e-6. This does not promote C8 to ActQuant-ON model compatibility. The original authored 4x2 PoC and all historical FFN outcomes remain distinct.

## Build identity and performance limits

New iPhoneOS framework SHA-256: `2895f30ad294b47f6eb42dfc419277afcd7738e1151c969af43cc002610495af`.
New iPhoneOS Q4 metallib: `abc05c6f28fa186dc9587d70dc7ec137903c385bcbd8203202ba95780f14cab6`.
Canonical config SHA-256: `4c36dda96c0be9a3425e3b26ce0566ed4352a450511012c5811625914b436308`.

Both signed/unsigned build manifests match the current backend source snapshot; compilation/signing is not physical-device acceptance. No new phone or Mac performance was measured. The earlier A16 paired 2.61%-2.68% reduction belongs to the older binary. This fix adds realpath/stat/lock work to optional graph encoding; old isolated GPU timings do not measure that cost. Stat identity is invalidation, not cryptographic authentication or protection against concurrent/adversarial library replacement. Freeze artifacts during inference; retained hashes remain required.

The full Microsoft trusted-reference comparison remains strict 1. Scoped Q FP64/stock compatibility does not establish full-model equivalence, quality, energy or throughput. Old FFN/full-chain failures are preserved.

## Evidence and reproduction

`SUMMARY.json` records selected derived values and SHA-256 for experiment-relative artifacts, without host paths, device/signing identifiers, raw arrays or samples. The complete local build and raw evidence remain on the Mac.

- `cpu-contract-tests/SUMMARY.json`: production-helper CPU test.
- `shader-contract-controls/SUMMARY.json`: 20 negative compiler cases, 12 schema cases and static AIR controls.
- `mac-pipeline-runtime/runtime/RESULT.json`: preserved initial failure.
- `mac-pipeline-runtime-diagnostic/`, `mac-pipeline-repeat-1/`, `mac-pipeline-repeat-2/`: three 18-case runtime passes.
- `mac-repro-build/`, `PORTABLE_BUILD_REPRO.json`, `mac-clean-repro-pipeline/`: actual portable fresh-build and additional 18-case runtime pass.
- `mac-graph-contract/`: actual GGML graph, per-case FP64/canary data and trace.
- `mac-real-regression/{primary,independent}-joint-analysis/`: both real-model scopes.
- `INDEPENDENT_MAC_AUDIT.json`: independent all 33 payload/parity/FP64 archive check.
- `MAC_GRAPH_CONTRACT_ANALYSIS.json`: 15-case actual dispatch/fallback/canary/FP64 audit.
- `INDEPENDENT_MAC_AUDIT_ATTEMPT1_EXPLANATION.json`: preserved CPU-oracle signed-zero correction.
- `night-launcher-plan/`: CPU-only 15-stage evening phone plan, no device execution.
- `legacy-analysis-replay/`: explicit preserved-old-profile analysis.
- `ios-{unsigned,signed}/BUILD.json`: new source-bound build completion, not runtime.

See [Q kernel contract fix](../../docs/Q_KERNEL_CONTRACT_FIX.md) for fresh-build, CPU, actual Mac runtime and evening-phone instructions. A safe first entry is:

```bash
"$PY" bitnet/tests/run_q_contract_cpu.py --output "$ARTIFACTS/new-cpu-contract"
```

Use new output directories. The actual cache runner requires explicit `--run`; it creates Metal pipelines. Historical evidence requires `--contract-profile legacy-shared-v0 --legacy-shader-build "$OLD_SHADER_BUILD_JSON"` in the analyzer; this is analysis-only, not a runtime alias.

The evening entry is `bitnet/tests/run_q_fix_phone.py`: default/explicit `--plan` prepares 15 stages without device calls; `--run` explicitly starts actual cache checks, quantizer qualification and two-history graph regression. Performance is not a stage and follows only after those gates pass. Use the placeholder-only command in the English fix note.

Additive patch: [`q-kernel-contract-fix.patch`](../../patches/q-kernel-contract-fix.patch), SHA-256 `c76dc42ff36c837614b27794e9ca9ca6848438a6a3393e03e3bec0f5dc0763d1`. Already applied; its base is the submission baseline above.
