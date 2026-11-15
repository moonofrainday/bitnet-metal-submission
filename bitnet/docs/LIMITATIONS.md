# Limitations and remaining work

This is a source submission candidate, not a completed quality evaluation or production release.

| Item | Current boundary |
|---|---|
| Public versus team checkpoint | Only the converted public Microsoft 2B checkpoint is represented; the team's custom checkpoint is separate and has not been validated here. |
| Original PoC | Standalone FP64 acceptance is demonstrated on the declared inputs. It does not establish stock-relative model-output compatibility. |
| Q4 integration | Layer 0, single-token Qcur-0 only; three sampled operator inputs and two fixed dispatch/logit histories. Complete 32-call pre-RoPE/residual verification and whole-model Q4 throughput are unmeasured. |
| G | Observed stock outputs match. Complete-FFN FP64 failures inherited from stock remain failures. |
| F8/K | Preserved FP64/model-output failures; performance research only. K's actual graph contract historically allowed 28 layers. |
| Model quality | No independent language-quality/perplexity/task evaluation is included. Local FP64 accuracy, fixed-token replay and greedy agreement do not replace it. |
| CPU versus Metal | Public 2B inference succeeds on both; strict token parity fails at generated token 18. The original strict exit 1 remains. |
| iPhone timing | Recorded hot/serious thermal campaigns, with strong wall-time drift. No cool-device generalization, hardware-counter bottleneck proof or universal tail-latency claim. |
| Fresh packaging | Fresh build/source checks are listed in PACKAGE_CHECKS.json. Historical phone data is not represented as rerun on the repackaged binary. |
| Evidence publication | Derived analyses and digest index only. Raw tensors, logits, timing samples, traces, models and signed bundles remain local. |
| Upstream CI | The published baseline omits 63 upstream `.github/workflows` files; original blob identities are retained in BASELINE.json. No compute or local test source is omitted. |
| Final deliverables | Author review, final technical PDF, team-checkpoint validation and any required external access/hand-off remain separate tasks. |

## Recommended next validation

1. Review the small standalone kernel/reference/harness and the eight-file upstream change map.
2. Build and run the repackaged source on the physical phone with an explicit signing identity. Revalidate dispatch and numerical scope before collecting new timing data; previous measurements do not validate a new binary automatically.
3. For Q4, add non-slicing readback of all 32 target outputs and residual boundaries before claiming full frozen model acceptance.
4. Retain strict stock/model failure records while localizing FFN projection/down arithmetic. Keep any proposed application-quality criterion separate from the frozen numerical rule.
5. If another phone performance campaign is needed, use stable thermal conditions and a direct same-build stock/G/F8 contrast; retain A/A calibration and all samples.

Stock remains the default and fallback. No automatic multi-layer Q enablement, threshold relaxation, upstream submission or sharing of the private repository is performed by this package.
