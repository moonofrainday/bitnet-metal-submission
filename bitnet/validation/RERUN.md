# Rebuild and rerun

Run these commands from the submission repository root. Use a **new** work/output directory for each attempt. Raw logs, vectors, per-row correctness files, compiler outputs, and negative-control copies remain local and should not be committed.

## Model-free entry

```bash
python3 bitnet/validation/bootstrap.py \
  --work-root /tmp/bitnet-validation-review-001 \
  --developer-dir /Applications/Xcode.app/Contents/Developer
```

Pass `--cmake /absolute/path/to/cmake` when CMake is absent from PATH. The default source cache is `~/Library/Caches/bitnet-metal-validation/qvac.git`; `--cache /another/outside/path/qvac.git` changes it. Only source objects are reused. Every attempt creates a pristine detached source worktree and a fresh build. The bootstrap records argv, actual exits, and logs under the work root. The full smoke result is `results/RUN.json`.

To avoid a network fetch when a local QVAC repository already contains the pin:

```bash
python3 bitnet/validation/bootstrap.py \
  --qvac-repo /absolute/path/to/existing-qvac-repository \
  --work-root /tmp/bitnet-validation-review-002
```

The submission HEAD is not the pristine baseline. Do not point `--source` at the modified submission tree. The bootstrap creates its own exact-pin worktree.

## Explicit smoke orchestration

If desired, call the lower-level entry directly. All three paths must be outside the pinned source tree, and output must not already exist:

```bash
python3 bitnet/validation/smoke.py \
  --qvac-repo /absolute/path/to/existing-qvac-repository \
  --source /tmp/bitnet-pristine-source-003 \
  --build /tmp/bitnet-pristine-build-003 \
  --output /tmp/bitnet-pristine-results-003
```

Without real inputs this explicitly selects the two synthetic cases. Backend regression remains enabled by default. `--skip-backend-regression` produces a narrower result and leaves its regression evidence absent; it is not the full documented qualification.

## Optional real public-model inputs

The full historical campaign additionally used local pre-RoPE Q fixtures at decode steps 1, 16, and 32. Their hashes and model identity are published in `analysis/INPUT_PROVENANCE.json`; the payloads are deliberately excluded.

Create a manifest from locally available captures, with a separate clean exact-pin QVAC checkout:

```bash
python3 bitnet/validation/prepare-real-inputs.py \
  --capture-root /absolute/path/to/local/stock-capture \
  --audit /absolute/path/to/local/q-input-audit.json \
  --qvac-source /absolute/path/to/pristine-pinned-qvac \
  --output /tmp/bitnet-local-real-inputs-004 \
  --model /absolute/path/to/microsoft-bitnet-b1.58-2B-4T-tq2_0.gguf

python3 bitnet/validation/bootstrap.py \
  --qvac-repo /absolute/path/to/existing-qvac-repository \
  --work-root /tmp/bitnet-fivecase-review-004 \
  --real-inputs /tmp/bitnet-local-real-inputs-004/real-inputs.json
```

The optional `--model` verification imports NumPy and `gguf-py` from that clean pinned checkout; omit the option when only previously audited local captures are available. The expected public GGUF SHA-256 is `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2`. This is a converted public checkpoint, not the team's custom checkpoint. Missing requested real cases cause failure rather than a silent model-free downgrade.

## CPU-only recomputation and evaluator controls

Given a completed native result directory:

```bash
python3 bitnet/validation/analyze.py \
  --input /absolute/path/to/results/gpu \
  --output /tmp/bitnet-reanalysis-005

python3 bitnet/validation/cpu-checks.py \
  --native /absolute/path/to/results/gpu \
  --output /tmp/bitnet-evaluator-controls-005
```

Add `--model-free` to `analyze.py` only when deliberately evaluating the two-case scope. Default analysis requires all five cases. CPU controls run the Python evaluator against copies of native output artifacts; they do not rerun the native harness or GPU. Analysis execution success and strict numerical acceptance remain distinct, as specified in `ACCEPTANCE.md`.

No command here is an inference benchmark. The generated local metadata may contain actual local absolute paths because it records executed commands; published source and curated summaries do not depend on those paths.
