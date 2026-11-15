# Portable CPU reference rerun

Use a Python 3.11 virtual environment with `requirements.txt`. Set the following paths locally; all data and results stay outside the repository.

```sh
export PY=/absolute/path/to/reference-venv/bin/python
export REF="$PWD/bitnet/reference"
export MODEL_DIR=/absolute/path/to/original-HF-snapshot
export REFERENCE_CACHE=/absolute/path/outside/repository/reference-cache
export RESULTS=/absolute/path/outside/repository/new-reference-results
export OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6
"$PY" "$REF/prepare_sources.py" --cache "$REFERENCE_CACHE" --model-dir "$MODEL_DIR"
mkdir -p "$RESULTS"
"$PY" "$REF/streamed_reference.py" --model-dir "$MODEL_DIR" \
  --source-dir "$REF" --dataset-parquet "$REFERENCE_CACHE/wikitext-2-raw-test.parquet" \
  --dtype bf16 --output "$RESULTS/bf16" > "$RESULTS/bf16.log" 2>&1
"$PY" "$REF/streamed_reference.py" --model-dir "$MODEL_DIR" \
  --source-dir "$REF" --dataset-parquet "$REFERENCE_CACHE/wikitext-2-raw-test.parquet" \
  --dtype f32 --output "$RESULTS/f32" > "$RESULTS/f32.log" 2>&1
"$PY" "$REF/score_logits_fp64.py" "$RESULTS/bf16"
"$PY" "$REF/score_logits_fp64.py" "$RESULTS/f32"
```

The required original model revision is `04c3b9ad9361b824064a1f25ea60a8be9599b127`; expected checkpoint/tokenizer/config hashes are verified by `prepare_sources.py`. Four Apache-2.0 official source files are included and checksum-verified. The test parquet is fetched at its immutable dataset revision into the external cache.

Optional captured-input operator check:

```sh
export FIXTURE_DIR=/absolute/path/to/captured-step-1-16-32-Q-fixtures
"$PY" "$REF/q_activation_semantics.py" --checkpoint "$MODEL_DIR/model.safetensors" \
  --fixture "$FIXTURE_DIR" --official-source "$REF/official-bitnet-096f25ae.py" \
  --output "$RESULTS/q-control" --strict
```

Its observed strict exit is **1**, because stock does not apply the reference ActQuant to X; this is a measured semantic failure, not a setup failure. Other processing commands return 0 only for successful computation. Preserve nonzero stage exits and inspect logs; never relabel processing completion as numerical/semantic acceptance.

The original 27-token prompt, fixed 32 continuation IDs and exact corpus IDs are generated into each local run. No raw data are bundled. See `README.md` for reference transformations, BF16/F32 and cache qualifications, licensing, and optional comparison to cached QVAC logits. The packaged path options and configurable corpus-description labels do not change the reference arithmetic used in the recorded experiments.
