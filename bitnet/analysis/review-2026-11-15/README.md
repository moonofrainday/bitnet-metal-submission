# Proposal review response and measured controls

This revision addresses model semantics, exact authored-PoC qualification, optimization priority, framework attribution and numerical/product acceptance separately. All speed measurements concern the physical iPhone 14 Pro / A16. Mac work in this revision is source analysis and CPU reference computation, not a performance claim.

| Question | New evidence | Conclusion |
|---|---|---|
| Does current public-model inference preserve Microsoft semantics? | Official source, same captured Q W/X, two complete source-derived dtype profiles and 33 teacher-forced logits | No: required per-token INT8 activation quantize/dequantize is absent. Do not equate stock agreement with faithful W1.58A8 inference. |
| Did the original authored shader run on the phone? | Exact original 4x2 source, fresh metallib, five cases, 15 GPU completions and 38,400 independently checked values | Standalone sample-I/O passes. This is distinct from Q4 real-graph qualification and full-layer/model acceptance. |
| Is there a quality check? | Canonical test-split 128-target smoke, identical IDs, full logits and independent FP64 scoring | Phone stock PPL 19.088221; aligns closely with ActQuant-OFF. Too small to accept/rank language quality. |
| Does full guard + the same stock pipeline cost anything? | S/QB matched actual PSO, ABI, live bindings, output parity; short 3x7-pair steady decode screen | QB-minus-S medians +12.588, +13.329, +10.020 ms per 32 tokens. Additional path contributes in this screen; CPU/GPU cause and universal magnitude remain unresolved. |
| Is Q the best optimization target? | Audited geometry, tied-head storage, conditional Amdahl analysis | Q was a validation entry point. Operator-family critical-path shares and output-projection runtime remain unmeasured. |
| Do frozen FP64 failures prove poor model quality? | Reference ladder, cancellation/propagation analysis and old token-18 margins | No. Preserve V1 failures while separately defining semantic and product-level criteria. |

## Reading order

1. `REVIEW_SEMANTICS.md` / `.json`: public contract, exact-source qualification, Q controls, BF16/F32 ON/OFF and teacher forcing.
2. `ORIGINAL_POC_PHONE.md` / `original-poc-phone-summary.json`: original source/binary identity and all five phone cases, including a separate derived-ActQuant input control.
3. `PHONE_QUALITY_SMOKE.md` / `phone-ppl-summary.json`: independent raw-logit scoring and limits of the 128-target measurement.
4. `PROFILE_NUMERICS.md` / `guard-control-summary.json`: same-stock guard control, paired wall statistics, head/Amdahl and acceptance rationale.
5. `RERUN.md`: build, numerical/reference, phone and CPU reanalysis entry points.

The original strict CPU/Metal token comparison remains exit 1. Prior stock/candidate complete-FFN FP64 failures are not reclassified. The review controls use their declared scopes: original standalone local numerical pass; guard-only FP64 null/global V1 exit 2; quality measurement completed with semantic/product acceptance unverified. Analysis exit 0 is not a full-model acceptance pass.

Raw sources of evidence are local workspace `outputs/experiments/2026-11-15-review-221448/` and `outputs/review-response-2026-11-15/semantics/`; older traces are explicitly cited in the appendices. No raw corpus, tensor, logits, per-row/per-pair CSV, signed binary, personal device identifier or credentials are included here.

## Recommended next experiment

First add a default-off public-checkpoint ActQuant graph path matching the pinned source before all required BitLinear operations. Freeze rounding, absmax epsilon, dtype, scale placement and sharing rules. Validate same-input intermediates and teacher-forced logits before larger held-out evaluation and performance; include quantizer reduction/preparation in timings. Obtain the team's actual activation/weight contract independently. Do not promote the current stock path as faithful public W1.58A8 simply because it generates text.

For performance attribution after semantic alignment, independently time the read-only guard/plan path and native encoding, then profile actual output-projection/operator-family critical-path shares. Do not treat 54.56% file storage as 54.56% latency or translate the short guard screen into a pure CPU cost.
