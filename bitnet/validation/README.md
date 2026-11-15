# Standalone TQ2_0 Metal validation

This directory contains a readable authored matvec PoC, a stock-arithmetic 4x1 specialization, an independent FP64 reference, and a clean-build correctness entry. It is separate from the opt-in model-graph integration and iPhone app. Its Mac qualification covers numerical correctness, not inference performance.

## Run from this submission checkout

Requirements: native Apple Silicon macOS, complete Xcode with the Metal compiler, Git, CMake on PATH (or `--cmake`), and Python 3.9 or newer. The model-free run uses only the Python standard library. The first run needs network access to the public QVAC repository.

From the repository root, use a new directory outside the checkout:

```bash
python3 bitnet/validation/bootstrap.py \
  --work-root /tmp/bitnet-validation-review-001
```

This fetches immutable QVAC commit `2db8b046d555316019ae75d1be90484adc8b9330` into a local bare cache, creates a separate pristine worktree, rebuilds all required libraries and shaders, runs two generated fixtures, and runs the filtered TQ2_0 backend regression. It never builds against old project dylibs or metallibs. Existing result directories and nonempty builds are refused. See [RERUN.md](RERUN.md) for local real-input verification and explicit dependency paths.

The submission's main source tree contains compatibility and opt-in integration additions. The standalone entry deliberately builds a separate **unmodified upstream baseline**; its exact-pin and clean-tree checks remain enforced. The source checkout is not silently replaced with the submission HEAD.

## Code map

| File | Purpose |
|---|---|
| `bootstrap.py` | Fetches or reuses source objects, then starts a fresh pinned worktree/build/run outside the submission |
| `smoke.py`, `build.py` | Records and propagates build, reference, GPU, analysis, and backend-test exits |
| `main.mm` | Runs actual stock, authored PoC, and stock4x1 with identical W/X and separate guarded Y |
| `reference.cpp`, `reference.h` | Independent FP16/trit decoding and logical-K FP64 accumulation; generated fixtures |
| `shaders/matvec.metal` | Original authored explicit-unpack PoC, four rows/SIMD, two SIMD groups/threadgroup |
| `shaders/stock4x1.metal` | Pinned stock arithmetic specialized to four rows/SIMD, one SIMD group/threadgroup |
| `stock-runtime.h` | Adapter to the actual stock selector and native ABI of newly built QVAC libraries |
| `analyze.py` | Independent CPU analysis, pointwise comparisons, hashes, and selected-row `math.fsum` oracle |
| `cpu-checks.py` | Positive/negative evaluator controls; no GPU work |
| `prepare-real-inputs.py` | Verifies and copies optional local real W/X; optional GGUF identity check |
| `SOURCES.json` | Source origins and immutable checksums, without machine-specific runtime paths |

## Tensor and launch contract

W is logical `[2560,2560]` TQ2_0; X and Y are contiguous F32 vectors of 2560 values. Each 66-byte block stores 256 trits in 64 bytes, then one little-endian FP16 scale. The row stride is 660 bytes. Codes 0/1/2 map to -1/0/+1; code 3 is invalid. This is ternary decoding rather than binary XNOR/popcount.

The reference computes `Y[m] = sum(scale[block] * (code-1) * X[k])` in logical K order with FP64 accumulation. It does not copy the GPU coefficient transform or reduction. GGML dequantization is a separate cross-check.

| Route | Actual symbol | Binding | Grid / threads |
|---|---|---|---|
| S | `kernel_mul_mv_tq2_0_f32` | Stock args0, W1, X2, Y3, auxiliary bindings | `[320,1,1]` / `[32,2,1]` |
| PoC | `tq2_0_matvec_poc` | W0/X1/Y2 | `[320,1,1]` / `[32,2,1]` |
| Q4 | Historical symbol `tq2_0_matvec_poc_c8x2` | W0/X1/Y2 | `[640,1,1]` / `[32,1,1]` |

Q4 denotes the selected **4x1 launch**, not four-bit weights or C8x2. Its helper arithmetic derives from pinned QVAC stock; it is not a new dot algorithm. Actual S comes from QVAC's native selector, not a copied approximation. With Metal embedding enabled, the fresh dylib embeds upstream Metal source and runtime-compiles S; the two candidate metallibs are freshly compiled offline.

## Preserved correctness evidence

`analysis/` contains curated derived results from experiment `2026-11-15_170005+0900-clean-poc-stock4x1`, prior project commit `7fcf557fb593800df38fd217c02e07d3594dea4b`. These historical measurements are labeled separately from new executions of this relocated entry. No tensor payload, per-row raw CSV, full log, or built binary is included.

The original M2 Pro run completed 15 dispatches across five cases: hand-computable, dense seed42, and real public-2B layer-0 pre-RoPE Q inputs from steps 1/16/32. All 38,400 GPU values passed the frozen FP64 rule; both candidates passed same-input stock comparisons. Q4 was bitwise equal to stock across all 12,800 outputs. The fresh stock backend regression passed 59/59 with zero unsupported cases. A separate model-free clean rebuild also passed without GGUF/captures or a Python venv.

| Case | S/Q4 maximum FP64 absolute error | PoC maximum FP64 absolute error | Failed rows per route |
|---|---:|---:|---:|
| Hand-computable | 0 | 0 | 0/2560 |
| Dense seed42 | 2.057916135527e-4 | 5.516947567230e-6 | 0/2560 |
| Real step 1 | 9.509993446954e-6 | 4.736043592857e-7 | 0/2560 |
| Real step 16 | 9.595010098451e-6 | 4.280494749764e-7 | 0/2560 |
| Real step 32 | 8.449031852820e-6 | 3.673589787923e-7 | 0/2560 |

The full rule is `abs(actual-reference) <= 2e-4 + 2e-5*abs(reference)`. A maximum larger than its absolute term can still pass the row-dependent rule. Smaller local FP64 error does not establish model quality or compatibility with every intermediate stock output.

This standalone package does not load a GGUF, generate text, validate the complete FFN, or qualify the iPhone build. `model_output_pass=null` is intentional. Previous model/FFN failures remain separate. Physical iPhone correctness and performance claims belong to their own documented device experiments; this entry performs no inference timing.
