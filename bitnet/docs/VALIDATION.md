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
| `model_output_pass` | The explicitly captured layer outputs/full logits versus stock under the fixed replay and the frozen model-output rules. |

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
- Q4 historical phone checks cover three operator inputs and two 32-step dispatch/logit histories. They do **not** include all 32 pre-RoPE Q readbacks or residual-output checks, so no complete frozen V1 model acceptance is claimed.
- G preserves observed stock outputs but inherits stock's complete-FFN FP64 failures.
- F8/split-C8 observable activation, normalization/down input and down outputs are bit-exact in the complete captured replay scope. Both stock and F8 fail complete-chain FP64. F8's original stock-relative model-output failure remains. F8's internal unmaterialized gate/up vectors are null.

For the later independent-prompt all-layer replay, stock/F8 full-chain down failures were respectively 890/875 vectors on stock-X and 891/874 on F8-X (960 vectors per route/input stream). This does not license a model-quality ranking. Projection error and actual-input stock down arithmetic both contribute; conditional controls are not additive error shares. Activation-only rounding and normalization controls passed their stated scopes.

The full FFN reference starts at the captured normalized FFN input and includes gate/up, ReLU-squared multiplication, weighted RMSNorm with actual epsilon/gamma, and down. It is not an FP64 reference for the entire Transformer. Earlier 798/880 counts refer to a normalized activation diagnostic, not the number of weighted-RMSNorm output failures. Neither stock failures nor old candidate failures are erased by packaging or fusion parity.
