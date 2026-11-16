# Validation and preserved failures

The frozen project tolerance is:

```text
abs(actual - reference) <= 2e-4 + 2e-5 * abs(reference)
```

All required rows must be finite, written and within the bound. This is our experimental acceptance criterion, not a threshold supplied by the assignment brief or an industry-wide model-quality standard.

## Separate questions and exit codes

| Outcome | Reference and scope |
|---|---|
| `parity_pass` | Agreement between the named same-input routes. Bitwise equality is reported separately from tolerance equality. |
| `fp64_pass` | The declared operator or complete FFN chain versus an independent FP64 computation on the same supplied inputs. |
| `model_output_pass` | Route-specific reference: legacy Q4 versus stock OFF and new joint Q4 versus same-build stock ON, for declared residuals/full logits; ActQuant residuals/full logits versus source-derived eager F32 ON. The reference profile, KV dtype and coverage must accompany the flag. |

An unmeasured item is `null`, never inferred from another pass. Analysis processing exit 0 means the evidence was processed; it does not turn a strict numerical failure into success. The numerical verifier returns 1 for a valid measured failure and 2 for missing/invalid evidence. The standalone evaluator's success is scoped to the standalone operator; `model_output_pass` remains null there.

## Fresh clean standalone evidence, 2026-11-15

The previous clean validation package was rebuilt from a pristine pinned QVAC checkout, without old dylibs. All three paths use the same resident W/X and separate guarded output storage: actual stock PSO, original PoC and stock4x1.

| Case | Outputs/path | Stock/Q4 max FP64 absolute error | Original PoC max FP64 absolute error | Failed rows, all paths |
|---|---:|---:|---:|---:|
| Hand-checkable | 2560 | 0 | 0 | 0 |
| Dense seed 42 | 2560 | 2.057916135527e-4 | 5.516947567230e-6 | 0 |
| Real step 1 | 2560 | 9.509993446954e-6 | 4.736043592857e-7 | 0 |
| Real step 16 | 2560 | 9.595010098451e-6 | 4.280494749764e-7 | 0 |
| Real step 32 | 2560 | 8.449031852820e-6 | 3.673589787923e-7 | 0 |

The dense absolute maximum exceeds the absolute part of the bound but passes its complete relative-plus-absolute row criterion. The three paths cover 38,400 GPU output values. Q4 equals stock bit for bit on all 12,800 compared rows. The original PoC differs in 10,122 F32 rows while satisfying the declared tolerance. Sentinel/padding, input immutability, actual dyld library origins, FP16 scale decoding and invalid-code rejection were checked. Original TQ2_0 regression: 59/59, zero selected failures/unsupported cases.

A separate true model-free run used system Python 3.9, two synthetic cases, a fresh pinned source/build and no models, captures, NumPy or old virtualenv. It passed 15,360 GPU values and the same 59/59 backend regression. The 17 CPU evaluator controls passed, including preservation of explicit FP64-fail/strict-exit semantics. These are evaluator controls, not a new FFN GPU failure reproduction.

These are numerical/build results on M2 Pro, **not Mac inference performance measurements**. See curated summaries under `../validation/analysis` and current packaging checks under `../analysis/PACKAGE_CHECKS.json` for the distinction between historical validation and this checkout's fresh build verification.

## Existing model-level limitations remain

- Public 2B CPU generation and stock Metal generation both succeeded. The strict comparison still fails at the eighteenth generated token: CPU 83699 (`computations`), Metal 14515 (`graphics`). Both inference subprocesses returned 0; strict comparison returned **1**. This is independent of custom-PoC dispatch.
- C8 can pass local same-input FP64 while failing the original stock-relative model-output criterion. Error propagation through nonlinear graph operations is a different question from a single dot product.
- Earlier Q4 phone checks covered only three sampled operator inputs. The [fresh 2026-11-15 Q-only campaign](IPHONE_Q4_COMPLETE_2026-11-15.md) now checks every pre-RoPE Q output in both 32-step histories, residuals at steps 1/16/32, and all 33 logits, including probe-off controls. Its declared Q-only strict result is 0. This does not establish complete Transformer/FFN FP64 or model quality.
- G preserves observed stock logits. Its broader complete-FFN FP64/layer-output acceptance is unverified (`null`); stock's independently measured full-chain failures remain failures and are not repaired by parity.
- F8/split-C8 observable activation, normalization/down input and down outputs are bit-exact in the complete captured replay scope. Both stock and F8 fail complete-chain FP64. F8's original stock-relative model-output failure remains. F8's internal unmaterialized gate/up vectors are null.

For the later independent-prompt all-layer replay, stock/F8 full-chain down failures were respectively 890/875 vectors on stock-X and 891/874 on F8-X (960 vectors per route/input stream). This does not license a model-quality ranking. Projection error and actual-input stock down arithmetic both contribute; conditional controls are not additive error shares. Activation-only rounding and normalization controls passed their stated scopes.

The full FFN reference starts at the captured normalized FFN input and includes gate/up, ReLU-squared multiplication, weighted RMSNorm with actual epsilon/gamma, and down. It is not an FP64 reference for the entire Transformer. Earlier 798/880 counts refer to a normalized activation diagnostic, not the number of weighted-RMSNorm output failures. Neither stock failures nor old candidate failures are erased by packaging or fusion parity.

## Minimal Down correction control, 2026-11-15

The fresh A16 run fixes the captured independent-history layer-0 step-16 normalization output. It checks 12 eight-row groups (96 rows per route), including the 12 known FP64 failures. Actual stock and the independently compiled stock wrapper are bit-identical on all selected rows; each retains the same 12 FP64 failures. Moving the correction after the packed-dot terms changes 48 F32 results, retains all 12 failures, and increases the maximum error/bound ratio from 1.566268 to 1.829765. All 288 outputs are finite. Native and independent strict numerical exits are **1**; CPU processing exits 0. This selected negative control is neither a full-matrix repair nor a performance result. No full-matrix expansion was justified.

## ActQuant qualification and separate gates, 2026-11-16

The signed A16 build from source `4c249e3` completes 20 quantizer cases / 386,567 values, every output bit equal to the source-derived eager F32 oracle; independent operator strict exit is 0. In the actual teacher graph, all 3,960 quantizers / 25,581,568 values likewise match official F32 on their own actual input, with verified 31/31 offload. The quantizer oracle is F32 execution, not a replacement FP64 matmul reference.

Completed A16 source-F32-ON model comparison remains **false, strict exit 1**: teacher logit RMSE 0.174339405 with 31/33 greedy agreement, and preserved residual/logit element failures. The ON corpus PPL 19.127701918 is close to reference 19.129529529 but is only a 128-target smoke, not model-fidelity or quality acceptance. Native collection exit 0 cannot override the analyzer. The reference-only CPU ranking flip and the older CPU/Metal generation split are historical controls, not proof that the repaired A16 graph restores the reference token-18 choice.

| Route and input | Scoped result | Excluded inference |
|---|---|---|
| Original authored 4x2, supplied W/F32 X | Host and A16 sample-I/O pass independent FP64 | Complete original-PoC phone graph, full-layer FP64 and quality |
| Legacy Q4, ActQuant OFF | Two-history Q-only FP64 and stock-F32 model-output acceptance pass | Microsoft activation semantics or performance after ActQuant repair |
| V2 ActQuant, same actual X | CPU/M2/A16 official-F32 sample and captured-input checks pass | Complete model reference acceptance, which is measured false |
| Joint ActQuant/Q4 | Both histories pass all32 Q FP64 checks, stock-ON Q/residual/logit parity and fallback; scoped exit0 | Trusted-reference model, full-Transformer FP64, quality and joint corpus/PPL are not established |

For the measured joint route, Q4 uses the actual quantized activation and passes independent FP64 and matching ActQuant-ON stock Q checks. Compare residuals/full logits against that same-build stock-ON route separately from the trusted-source F32 ON model gate. Every missing check is null. The new joint scope and fresh 2.61%-2.68% isolated-operator pairing are in `../analysis/actquant-q4-joint-2026-11-16/`; trusted-reference diagnostics stay in `../analysis/actquant-iphone-2026-11-16/`. Exact commands are in `BUILD_AND_RUN.md`; no frozen failure is revised.
