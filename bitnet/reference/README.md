# Trusted-source BitNet activation controls

CPU numerical/quality-smoke tools, not Mac performance benchmarks. Derived results are in `../analysis/review-2026-11-15/REVIEW_SEMANTICS.md` and its JSON companion. All raw checkpoint, corpus, fixtures, tensors, logits and row CSV must remain in external local directories.

The four bundled `official-*.py` files are immutable Transformers source at the model-card-recommended commit `096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4`, with Apache-2.0 headers and `TRANSFORMERS_LICENSE` retained. The driver AST-selects their arithmetic and removes compile/inference-plumbing decorators. It is source-derived eager execution, **not** an unmodified full recommended Transformers installation. Installed Transformers is not imported by this reference driver.

The public snapshot is `microsoft/bitnet-b1.58-2B-4T` revision `04c3b9ad9361b824064a1f25ea60a8be9599b127`. Original packed weights are unpacked by the official load hook. BF16 is the recommended compute profile; F32 is a separate control that losslessly lifts original BF16 parameter values. Within each profile the same weights/backend/data are reused for ActQuant ON/OFF. No MPS/Metal is used. One decoder layer at a time keeps memory bounded.

## Dependencies and external inputs

Use an isolated Python 3.11 environment, or reuse one with the pinned versions. Do not install into the system Python.

```sh
python3.11 -m venv "$REFERENCE_VENV"
"$REFERENCE_VENV/bin/python" -m pip install -r bitnet/reference/requirements.txt
export PY="$REFERENCE_VENV/bin/python"
export REF="$PWD/bitnet/reference"
export MODEL_DIR=/absolute/path/to/original-HF-snapshot
export REFERENCE_CACHE=/absolute/path/outside/repository/reference-cache
export RESULTS=/absolute/path/outside/repository/new-reference-results
"$PY" "$REF/prepare_sources.py" --cache "$REFERENCE_CACHE" --model-dir "$MODEL_DIR"
```

Preparation verifies the bundled sources, downloads checksum-verified immutable source copies and the WikiText test parquet into the external cache, and verifies the existing checkpoint/config/tokenizer. It does not download/copy a model. The original snapshot must include `model.safetensors`, `config.json`, `tokenizer.json` and `tokenizer_config.json` from that revision. `SOURCE_PROVENANCE.json` and preparation output retain source hashes. Raw sample inputs are separate and cannot be reconstructed from summary hashes alone.

Dataset: `Salesforce/wikitext`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-2-raw-v1`, test split. The driver joins rows with two newlines, takes the first 128 ordinary tokenizer IDs, prepends one explicit BOS, and scores 128 next-token targets. This is a nonrepresentative test-split quality smoke, not proof of no pretraining overlap. HF metadata lists CC-BY-SA-3.0/GFDL; card prose links CC-BY-SA-4.0. Keep the corpus local.

## Reproduce the controls

```sh
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6
mkdir -p "$RESULTS"

# Reference processing: BF16 recommended profile, then separate F32 control.
"$PY" "$REF/streamed_reference.py" --model-dir "$MODEL_DIR" \
  --source-dir "$REF" --dataset-parquet "$REFERENCE_CACHE/wikitext-2-raw-test.parquet" \
  --dtype bf16 --output "$RESULTS/bf16" > "$RESULTS/bf16.log" 2>&1
"$PY" "$REF/streamed_reference.py" --model-dir "$MODEL_DIR" \
  --source-dir "$REF" --dataset-parquet "$REFERENCE_CACHE/wikitext-2-raw-test.parquet" \
  --dtype f32 --output "$RESULTS/f32" > "$RESULTS/f32.log" 2>&1

# Independent FP64 scoring of unchanged model logits. Keep original FP32 scores.
"$PY" "$REF/score_logits_fp64.py" "$RESULTS/bf16"
"$PY" "$REF/score_logits_fp64.py" "$RESULTS/f32"

# Optional captured-input Q control: FIXTURE_DIR must contain step-1/16/32,
# each with w.tq2, x.f32 and y.f32 from the declared stock execution.
"$PY" "$REF/q_activation_semantics.py" --checkpoint "$MODEL_DIR/model.safetensors" \
  --fixture "$FIXTURE_DIR" --official-source "$REF/official-bitnet-096f25ae.py" \
  --output "$RESULTS/q-control" --strict

# Optional comparison to a local stock replay, exact IDs/model hash required.
"$PY" "$REF/analyze_reference.py" --reference "$RESULTS/f32" \
  --qvac-stock-run "$QVAC_STOCK_RUN" --output "$RESULTS/qvac-comparison"
```

Output directories must be new. Required variables and fixtures must exist; no silent synthetic substitution is allowed. Reference/analyzer processing exit 0 means computation completed. It is not a model-semantic pass. `q_activation_semantics.py --strict` returns **1** for the observed semantic mismatch. The old CPU/Metal strict comparison also remains **1**. Frozen tolerance is `2e-4 + 2e-5 * abs(reference)`.

The saved reference has no KV cache; QVAC F16 cached execution has additional precision/kernel/cache differences. Use same-backend ON/OFF and matched-input projection controls to isolate ActQuant, rather than attributing every full-logit difference to it. `score_logits_fp64.py` provides common FP64 stable-logsumexp scoring; original Torch F32 cross-entropy values remain recorded. An initial-run JSON ranking repair and its original summaries are preserved in the local research archive. Fresh canonical runs already contain that serialization fix and require no repair script. `SOURCE_PROVENANCE.json` retains the archived script's historical digest and labels it as excluded from this submission, rather than treating it as a required bundled file.

These tools do not add the absent ActQuant stage to the live graph and do not qualify the team's unavailable checkpoint. Parameter, activation, rounding, normalization, tokenizer and product-quality contracts must be confirmed separately.
