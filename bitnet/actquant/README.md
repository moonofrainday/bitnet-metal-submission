# Public checkpoint ActQuant validation

These tools preserve the existing numerical bound:

`abs(actual-reference) <= 2e-4 + 2e-5 * abs(reference)`

They do not change any historical strict failure. Raw fixtures, intermediate tensors, full logits, masks and per-element CSV must remain outside the repository.

## Local operator oracle

`prepare_fixtures.py` checks the immutable official source SHA-256, AST-selects `ActQuant` and removes only its `torch.compile` decorator. The arithmetic is the official F32 implementation: per-token last-dimension absmax, denominator clamp at `1e-5`, `127/denominator`, ties-to-even round, clamp to `[-128,127]`, and division by the same scale. This is not a double-precision substitute for the F32 quantizer.

The complete fixture contains 18 cases: widths 2560/6912, 1/3/27 tokens, zero and signed zero, epsilon and adjacent F32 epsilon values, positive/negative exact half ties, seeded dense inputs with different token amplitudes, unchanged real Q inputs from steps 1/16/32, and one constructed seven-value floating boundary. The exact-half case has absmax 127 and scale exactly 1; expected integer codes are independently hand-checked. `--synthetic-only` emits 15 model-free cases and explicitly records `missing_real_capture=true`; it cannot silently claim real-input coverage.

Two optional complete input rows from the initial real-model quantizer failure can be added using `--boundary-capture` and `--boundary-failures`, producing 20 cases. Their data stay local. The seven-value constructed case is encoded by exact integer float bits in source and needs no model or private fixture. An optional `--counterexample` JSON must reproduce those bits exactly.

The algebraic source expression `127 / tensor` dispatches through the recorded Torch eager CPU profile as reciprocal followed by multiplication by 127. Direct hardware division can differ by one scale ULP and change a near-half integer code. The pinned profile and constructed counterexample are checked explicitly; this is an execution-profile qualification, not a claim that algebraic notation specifies every backend's instructions.

```sh
"$PY" bitnet/actquant/prepare_fixtures.py \
  --official-source bitnet/reference/official-bitnet-096f25ae.py \
  --real-fixture "$REAL_Q_FIXTURE" --output "$ARTIFACTS/fixtures"
"$PY" bitnet/actquant/analyze_op.py --fixture "$ARTIFACTS/fixtures" \
  --actual "$ARTIFACTS/native-op" --output "$ARTIFACTS/op-analysis"
```

For a source-only delivery without real captures, replace `--real-fixture "$REAL_Q_FIXTURE"` with `--synthetic-only`. This is an explicitly smaller correctness scope.

### Fresh model-free Mac build

Set `PY` to the isolated Python environment with the pinned reference dependencies, and set `ARTIFACTS` to a new local directory outside the source repository. No prior dylib or model is required for this scope.

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
"$PY" bitnet/actquant/prepare_fixtures.py \
  --official-source bitnet/reference/official-bitnet-096f25ae.py \
  --synthetic-only --output "$ARTIFACTS/fixtures"
"$PY" bitnet/actquant/run_mac.py --mode operator \
  --fixture "$ARTIFACTS/fixtures" --output "$ARTIFACTS/fresh-operator"
```

If CMake is not on PATH, add `--cmake /absolute/path/to/cmake`. The entry builds from the current source into a fresh build directory, strips inherited dylib and custom-kernel settings, runs the supplied samples on CPU and MTL0, and runs targeted ActQuant/TQ2_0 backend regressions. This is Mac numerical validation, not a Mac performance result. The 15-case synthetic scope explicitly lacks real model-input qualification.

`FIXTURES.json` records each case's GGML `ne`/`nb`, input, expected F32 output, per-token F32 scale, signed integer codes and hashes. A native runner writes `<case>/y.f32`. Optional `<case>/scale.f32` and `q.s8` are checked when exported. Optional runtime identity is supplied in `RUN.json` with fixture hash and per-case `name`, `ne`, `input_sha256`, `output_sha256` and `completed=true`; `--require-runtime` requires this evidence rather than inferring GPU execution from saved output files.

Passing requires all values finite, all values within the frozen rule and exact integer-grid code agreement. Codes reconstructed in F64 as `round(output * official_scale)` must match the official codes exactly. This is an output-grid check, not direct observation of an internal register. Native exported codes, if present, must also match. Scale/value ULP and bitwise differences are reported separately; missing native scale readback stays null.

This additional exact-code check matters for small activations: a wrong code can change the result by much less than the `2e-4` absolute term. The rejection tests explicitly cover that case. Exit 0 is a scoped operator pass, 1 an observed failure, 2 a missing file and 3 malformed evidence. A local quantizer pass is not full-model FP64 or language-quality acceptance.

## Same-token model comparison

```sh
"$PY" bitnet/actquant/compare_model.py --run "$NATIVE_MODEL_RUN" \
  --reference-f32 "$F32_REFERENCE" --reference-bf16 "$BF16_REFERENCE" \
  --output "$ARTIFACTS/model-analysis"
```

The native `RUN.json` declares the public GGUF hash, ActQuant switch, exact tokens, backend, KV/prefill settings, completed decode calls, ordered logits and checkpoint payloads. The teacher workload uses the original 27-token prompt and 32 continuation IDs, with 33 logits at input positions 26 through 58. The corpus workload uses the same BOS plus 128 test-split targets as the existing reference. There is no retokenization.

Each captured ActQuant input/output pair is checked on the same actual input using the official F32 source. Coverage requires all four stages in all 30 layers and every declared call. Final-layer FFN prefill cropping is handled explicitly; it does not excuse missing QKV/attention rows. The checker compares all declared residual layer outputs and layer-0 pre-RoPE Q with the separate F32/BF16 ON/OFF reference profiles. OFF Q input uses the unquantized saved X, not its hypothetical diagnostic `X-effective` file.

All full logits receive the frozen per-element comparison, rankings/margins and common stable-FP64 logsumexp scoring. F32 ON is the repair target; OFF is a diagnostic distance, and BF16 remains a separate precision profile. Greedy agreement, reduced RMSE and PPL alone cannot turn a frozen-tolerance failure into a pass. The 128-token corpus is a nonrepresentative quality smoke with no product-quality acceptance budget.

`SUMMARY.json` keeps same-input quantizer correctness, coverage, F32 ON model outputs, missing full-chain FP64 and quality acceptance separate. Default exit uses the scoped strict result; `--exit-mode processing` returns 0 after analysis while preserving an observed strict failure or incomplete result. Lossless packed failure masks retain every failed element; the readable CSV contains only the first 16 and worst 16 failures per comparison. Actual tensors and reference files retain the full values.

CPU stock TQ2_0 may additionally repack activations into Q8_K. F16 KV, matmul/reduction order and source no-cache execution also differ. These are reported as remaining execution differences, not silently attributed to an incorrect ActQuant kernel. No current or historical global V1 pass is inferred from the new F32-only semantic check.

### Fresh full Mac validation entry

Prepare the real fixture separately as above, and provide the existing checksum-verified GGUF plus both complete source-reference caches. The token file is the cached reference's exact `tokens.json`, containing both the 59-token teacher history and 129 corpus IDs.

```sh
"$PY" bitnet/actquant/run_mac.py --mode full \
  --fixture "$REAL_PREPARED_FIXTURE" --output "$ARTIFACTS/fresh-full" \
  --model "$PUBLIC_2B_GGUF" --tokens "$F32_REFERENCE/tokens.json" \
  --reference-f32 "$F32_REFERENCE" --reference-bf16 "$BF16_REFERENCE"
```

Full mode performs a fresh build and operator regressions, then records teacher OFF/plain, ON/captured and ON/plain controls, plus corpus OFF/plain and ON/captured. The captured ON teacher and corpus runs and probe-neutrality check are required candidate gates. A native process exit 0 is execution completion; the analyzer's strict numerical and missing-evidence verdicts remain separate. The current cross-reference model failures are retained and can make full mode return nonzero even when the quantizer is locally exact. No source/reference caches are silently regenerated, no tolerance is relaxed and no missing captures are replaced by synthetic data.

This round has no new physical-phone result or Mac performance claim. iPhone compilation/preparation, when present, is a separate deployment stage and does not establish device execution.

## Checker controls

```sh
"$PY" bitnet/actquant/test_analyze_op.py
"$PY" bitnet/actquant/test_model_contract.py
```

The tests reject hidden integer-code errors, nonfinite/missing/modified outputs, missing stage coverage and incorrect token mapping, and check exact tie behavior in reported greedy rankings. They execute no GPU or model.
