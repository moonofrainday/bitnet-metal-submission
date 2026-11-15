# Public-checkpoint semantic controls

The tested QVAC Metal graph omits Microsoft's prescribed per-token absmax INT8 activation quantize/dequantize. Functional generation, custom/stock parity, and FP64 correctness for supplied unquantized X do not establish faithful W1.58A8 execution. The team's own quantizer/scales/tokenizer contract remains a separate dependency.

## Reference contract and qualification

Public checkpoint revision: `04c3b9ad9361b824064a1f25ea60a8be9599b127`. The [official card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/blob/04c3b9ad9361b824064a1f25ea60a8be9599b127/README.md) points to Transformers `096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4`. Its [ActQuant and AutoBitLinear](https://github.com/huggingface/transformers/blob/096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4/src/transformers/integrations/bitnet.py#L216) implement:

```text
s = 127 / max(max_j(abs(x[j])), 1e-5)
xq[j] = clamp(round_ties_to_even(x[j].float() * s), -128, 127) / s
xq = cast_to_input_dtype(xq)
y = linear(xq, unpacked_ternary_weight) * checkpoint_weight_scale
```

An F32 tensor can contain dequantized INT8 values. Omission is confirmed here by the actual graph/kernel and same-input tests, not inferred from dtype. The stock kernel's floor expressions unpack weight fields; they do not implement ActQuant.

The CPU reference selects immutable official decoder/attention/RMSNorm/RoPE/ReLU-squared/AutoBitLinear arithmetic. Compile and inference-plumbing decorators are removed; one layer is streamed at a time. It is **source-derived eager execution, not an unmodified complete Transformers installation**. Separate BF16 (recommended profile) and F32 (same original BF16 parameter values losslessly lifted) controls use identical weights/backend/data within each ActQuant ON/OFF pair. Each workload executes 210 BitLinear sites. Full-sequence no-KV reference differs from QVAC F16 cached execution; cross-runtime comparisons retain that qualification. No CPU inference-performance claim is made.

## Same-input Q result

Original scaled Q weights equal decoded TQ2_0 effective weights exactly. Source-derived F32 AutoBitLinear agrees with independent quantized-input FP64 on all rows; stock agrees with unquantized-input FP64. Frozen tolerance remains `2e-4 + 2e-5 * abs(reference)`.

| Captured decode step | 1 | 16 | 32 |
|---|---:|---:|---:|
| X values changed / 2560 | 2559 | 2560 | 2559 |
| Stock vs unquantized FP64 failed rows | 0 | 0 | 0 |
| Stock vs official-ActQuant FP64 failed rows | 2498 | 2491 | 2502 |
| Maximum absolute difference vs quantized FP64 | 0.028416420 | 0.034888067 | 0.033746916 |

This identifies a missing semantic stage for this public checkpoint. It does not mean the supplied-input matvec is incorrect, or prove every conversion field correct.

## Fixed-history teacher forcing

The exact existing 27-token prompt and 32 continuation IDs produce 33 full logit vectors. The cached official chat template matches the prompt. At layer-0 step-16, QVAC and both F32 references have bitwise-identical pre-quantizer Q input. Pre-RoPE Q versus F32 OFF has 0/2560 tolerance failures; versus ON has 2492/2560 (the preceding table uses independent FP64, hence a nearby different count).

| Token-18 prediction | Source F32 ON | Source F32 OFF | QVAC stock Metal |
|---|---:|---:|---:|
| computations minus graphics | +0.094944 | -0.005909 | -0.005144 |
| Greedy token | computations | graphics | graphics |
| Greedy agreement with QVAC over 33 vectors | 31/33 | 33/33 | 33/33 |
| Full-logit RMSE vs QVAC | 0.251979 | 0.000724 | 0 |

The same-backend F32 ON/OFF control establishes that this quantizer change can alter a ranking. It does not replace the earlier blockwise Q8_K rounding explanation for the original CPU/Metal comparison. BF16 ON/OFF values near this decision are coarser (including a tie OFF); BF16 cannot resolve the old approximately 0.005 margin. All ranking summaries state argmax/tie handling.

## Tiny held-out-test-split smoke

Dataset: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), revision `b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-2-raw-v1`, test split. The first 128 ordinary IDs of the row-ordered text joined with two newlines are prefixed with one explicit BOS; all routes score the same 128 next-token targets.

| Reference profile | ActQuant | Mean NLL, FP64 scoring | Perplexity |
|---|---|---:|---:|
| BF16 | ON | 2.954328101 | 19.188825428 |
| BF16 | OFF | 2.949598624 | 19.098286597 |
| F32 | ON | 2.951233190 | 19.129529529 |
| F32 | OFF | 2.949089523 | 19.088566108 |

This is a tiny, nonrepresentative data-defined test-split smoke, not a standard benchmark, significance result, proof of no pretraining overlap, or license to omit quantization. Product acceptance needs a larger versioned dataset and predeclared NLL/task budgets. Dataset licensing and immutable source hashes are recorded; corpus/raw logits stay local.

Independent FP64 logsumexp and Torch FP64 cross-entropy agree; original Torch F32 scoring differs slightly and is retained. The table uses independent FP64 scoring. A JSON-only ranking serialization issue was repaired from unchanged raw logits with initial summaries preserved; no model arithmetic or frozen rule changed.

## Handoff and evidence

Keep three claims separate: supplied-input operator accuracy; custom/stock graph compatibility; trusted model semantics and language quality. The first two are not substitutes for the third. Both original CPU/Metal strict exit **1** and new semantic strict exit **1** remain; analysis processing exit **0** only means completion.

Next: add checkpoint-specific opt-in ActQuant before all required BitLinear projections, preserve reduction/round/clamp/scale/dtype, then align matched intermediate tensors and teacher-forced logits. Quantized X can be shared for identical Q/K/V and gate/up inputs, with preparation included in future timing. Do not change only custom Q and claim whole-model W1.58A8 fidelity. Obtain the team's inference contract before applying public-checkpoint assumptions.

Reproduction source: `q_activation_semantics.py`, `streamed_reference.py`, `analyze_reference.py`, `score_logits_fp64.py`, `repair_rankings.py`; exact commands/dependencies in `RERUN.md`. Scalar derived results and immutable IDs are in `REVIEW_SEMANTICS.json`. Official source files retain Apache-2.0 headers. Raw tensor/logit/corpus/per-row artifacts are local-only.
