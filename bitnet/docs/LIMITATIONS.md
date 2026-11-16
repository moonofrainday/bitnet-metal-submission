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
| iPhone timing | Historical ActQuant-OFF thermal cohorts stay separate. New ActQuant-ON input pairing shows 2.61%-2.68% isolated Q4 reduction in a short campaign; joint full-decode and corpus/PPL are unmeasured. Nominal endpoints do not prove a cold/fixed-frequency device. No hardware-counter bottleneck proof or universal tail-latency claim. |
| Fresh packaging | Fresh build/source checks are listed in PACKAGE_CHECKS.json. The 2026-11-15 signed-build Q4 campaign is documented separately; historical G/F8 results were not rerun on this binary. |
| Evidence publication | Derived analyses and digest index only. Raw tensors, logits, timing samples, traces, models and signed bundles remain local. |
| Upstream CI | The published baseline omits 63 upstream `.github/workflows` files; original blob identities are retained in BASELINE.json. No compute or local test source is omitted. |
| Final deliverables | Author review, final technical PDF, team-checkpoint validation and any required external access/hand-off remain separate tasks. |

## Recommended next validation

1. Reproduce the exact original 4x2 sample-I/O and stock-helper Q4 scopes using the named clean-build entries; review the expanded change map rather than assuming the old eight-file patch is the complete current diff.
2. Preserve completed A16 ActQuant local PASS and full-model strict exit 1. Continue matched upstream precision/threshold controls; no additional phone-setup test can substitute for fidelity analysis.
3. Preserve the completed joint ActQuant/Q4 qualification: exact quantized W/X, all32 dispatch/FP64 checks, stock-ON parity, probe neutrality and missing-library fallback. Keep its strict scoped0 separate from the trusted-model strict1.
4. The new same-build A16 operator comparison passes the measured scope, with A/A and balanced order. A future complete decode comparison must include quantizer preparation and separate guard/encoding diagnostics from GPU and normal wall time; no joint whole-model gain is claimed. Existing pre-ActQuant speedups stay separate.
5. Measure critical-path shares and process/Metal memory before another large optimization campaign. Broader shape sweeps, FFN/G/F8, energy and physical Android work are optional after the core PoC/source/PDF handoff; no 5% improvement is required by the brief.

Stock remains the default and fallback. No automatic multi-layer Q enablement, threshold relaxation, upstream submission or sharing of the private repository is performed by this package.
