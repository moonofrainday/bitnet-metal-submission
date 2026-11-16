# Limitations and remaining work

This is a source submission candidate, not a completed quality evaluation or production release.

| Item | Current boundary |
|---|---|
| Public versus team checkpoint | Only the converted public Microsoft 2B checkpoint is represented; the team's custom checkpoint is separate and has not been validated here. |
| Original PoC | The independently authored 4x2 passes host and physical-A16 sample-I/O FP64 on declared inputs. It does not establish its complete phone graph or stock-relative model-output acceptance. |
| Q4 integration | Layer 0, single-token Qcur-0 only; uses the stock helper. Legacy ActQuant-OFF two-history all-32-call Q FP64, sampled residual and full-logit checks pass. Its isolated gains did not yield whole-decode gains. Joint ActQuant-ON Q4 now passes its own two-history stock-ON Q-only qualification; no complete Transformer FP64 or multi-layer Q claim. |
| G | Observed stock logits match; broader G complete-FFN FP64/layer-output acceptance is unverified. Stock full-chain failures are not repaired by parity. |
| F8/K | Preserved FP64/model-output failures; performance research only. K's actual graph contract historically allowed 28 layers. |
| Model quality | Source-derived teacher forcing and a 128-target test-split PPL smoke are measured; representative held-out/task quality and a predeclared product budget remain unverified. |
| ActQuant repair | A16 20-case sample and 3,960 captured same-input quantizers pass official F32. Completed full-model residual/logit comparison fails, strict exit 1; faithful-model promotion remains unresolved. |
| CPU versus Metal | Public 2B inference succeeds on both; strict token parity fails at generated token 18. The original strict exit 1 remains. |
| iPhone timing | Historical ActQuant-OFF and earlier `edbd109` ON measurements remain separate. The current `14a9f103` package has isolated Q4 reductions of 2.51%-2.78% in Fair and 2.18%-2.44% in a separate post-model Serious cohort. Both histories' formal 32-token decode is slower: QB by 2.71%-3.74%, Q4 by 2.86%-3.76%, all Serious. The current joint corpus/PPL is unmeasured. Thermal endpoints do not prove a cold or fixed-frequency device; no hardware-counter cause or universal tail-latency claim. |
| Fresh packaging | Historical clean source/build checks are listed in PACKAGE_CHECKS.json. The current source-bound A16 package passes two independent 18/18 contract/cache runs, all 20 ActQuant cases, both histories' all-32-Q FP64 checks and all 33 stock-identical logits. Historical G/F8 results were not rerun on this binary. |
| Evidence publication | Derived analyses and digest index only. Raw tensors, logits, timing samples, traces, models and signed bundles remain local. |
| Upstream CI | The published baseline omits 63 upstream `.github/workflows` files; original blob identities are retained in BASELINE.json. No compute or local test source is omitted. |
| Final deliverables | The English technical PDF and source/harness package are included. Author review, team-checkpoint validation and any required external access or hand-off remain separate tasks. |

## Recommended next validation

1. Reproduce the exact original 4x2 sample-I/O and stock-helper Q4 scopes using the named clean-build entries; review the expanded change map rather than assuming the old eight-file patch is the complete current diff.
2. Preserve completed A16 ActQuant local PASS and full-model strict exit 1. Continue matched upstream precision/threshold controls; no additional phone-setup test can substitute for fidelity analysis.
3. Preserve the completed joint ActQuant/Q4 qualification: exact quantized W/X, all32 dispatch/FP64 checks, stock-ON parity, probe neutrality and missing-library fallback. Keep its strict scoped0 separate from the trusted-model strict1.
4. Preserve the current A16 result: local Q4 GPU time improves, but full-guard QB and Q4 both regress complete decode. Separately test the warm file-identity/key lifecycle or lock placement while retaining all 18 cache/contract controls, then requalify both histories before repeating the same-build comparison. Nested instrumented host elapsed cannot be subtracted from formal wall time as a complete causal decomposition. Existing pre-ActQuant and earlier `edbd109` results stay separate.
5. Measure critical-path shares and process/Metal memory before another large optimization campaign. Broader shape sweeps, FFN/G/F8, energy and physical Android work are optional after the core PoC/source/PDF handoff; no 5% improvement is required by the brief.

Stock remains the default and fallback. No automatic multi-layer Q enablement, threshold relaxation, upstream submission or sharing of the private repository is performed by this package.

The [current A16 analysis](../analysis/iphone-qcur0-attribution-2026-11-16/README.md) contains per-round results, input/build hashes and reproduction commands. Scoped Q correctness does not overwrite the full-model trusted-reference strict exit 1.
