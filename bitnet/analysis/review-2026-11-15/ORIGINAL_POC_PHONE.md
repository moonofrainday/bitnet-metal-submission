# Original authored matvec: physical-iPhone sample-I/O qualification

The exact original authored **4×2** Metal PoC now passes five sample-I/O cases on **iPhone 14 Pro / Apple A16 GPU**, using a fresh iPhoneOS metallib. This closes the previous physical-phone sample-I/O evidence gap. It does not turn the original shader into a stock-derived Q4 implementation, establish whole-model semantic fidelity, or demonstrate a performance improvement.

The original-input run and a separate ActQuant-transformed-input control each completed 15 GPU dispatches and passed independent host recomputation of **all 38,400 GPU values**. The frozen tolerance remains:

`abs(actual − FP64_reference) <= 2e-4 + 2e-5 × abs(FP64_reference)`

## Implementations actually executed

| Route | Actual shader | Provenance | Grid | Threads per threadgroup |
|---|---|---|---|---|
| S | `kernel_mul_mv_tq2_0_f32` | Native QVAC selector from the hash-verified signed embedded framework | `(320,1,1)` | `(32,2,1)` |
| PoC | `tq2_0_matvec_poc` | Exact original authored 4×2 shader | `(320,1,1)` | `(32,2,1)` |
| Q4 | `tq2_0_matvec_poc_c8x2` | Stock-helper-derived four-row, one-SIMD specialization; symbol retains its historical name | `(640,1,1)` | `(32,1,1)` |

All use `W[2560,2560]` in TQ2_0, one F32 token vector and 2,560 F32 outputs. TQ2_0 is a packed ternary representation with block scales, not I2_S or a binary-sign-only encoding. The same resident W/X handles are shared across the three routes within a case; output buffers are separate. Candidate buffer slots are W/X/Y = 0/1/2. Actual S uses its native argument block and ABI.

The unchanged original shader source SHA-256 is `db8b1fd40120adee0bf2f8bc4db01ef7e7edf11692d285e143fa15f24c408b84`. Its validation copy has the same hash. Fresh `iphoneos` Metal compile/link commands returned 0, producing metallib SHA-256 `97c63c938aa38658a5a37a3a3a4a3803fb4048332323e3a165ccfbb2a70e24f9`, which matches the staged and device-recorded library. Twenty-one source/build/binary identity checks passed.

The relevant original-phone source is commit `7d680ad1b266ab652e20a27f12cbe876cec8bfff`, on pinned QVAC `2db8b046d555316019ae75d1be90484adc8b9330` plus the existing compatibility/integration patches. The app was built before that source commit; its recorded package hashes match the committed source. Source and binary identities, rather than a later commit label alone, bind this result to the executed implementation.

## Original captured-input results

Maximum absolute errors below use a fresh independent FP64 oracle, not stock as a numerical oracle. Every entry has **0 failing rows and 0 nonfinite values out of 2,560**.

| Case | S max absolute error | Original PoC max absolute error | Q4 max absolute error | PoC rows differing bitwise from S |
|---|---:|---:|---:|---:|
| Hand-computed known answer | 0 | 0 | 0 | 0 |
| Dense seed 42 | 2.0579161e−4 | 5.5169476e−6 | 2.0579161e−4 | 2,540 |
| Captured step 1 | 9.5099934e−6 | 4.7360436e−7 | 9.5099934e−6 | 2,532 |
| Captured step 16 | 9.5950101e−6 | 4.2804947e−7 | 9.5950101e−6 | 2,529 |
| Captured step 32 | 8.4490319e−6 | 3.6735898e−7 | 8.4490319e−6 | 2,521 |

The dense S/Q4 maximum exceeds the absolute-only term `2e-4` but passes the original combined absolute/relative rule. Q4 is stock-bitwise-identical in all five cases. The original PoC is generally **not** stock-bitwise-identical, while all original-PoC/stock pointwise comparisons pass the same tolerance. Smaller FP64 operator errors do not establish better model quality or whole-model output compatibility.

The hand-computed output is exactly 1.375 at row 0, −1.25 at row 2,559, and zero elsewhere, for all routes.

## Separate derived-ActQuant input control

The three real-input X vectors were replaced **offline** with the already-saved F32 outputs of the pinned official `ActQuant` source. An independent re-execution of that exact AST reproduced the saved X vectors bitwise. W, shaders, native binary, tolerance and launch stayed unchanged; known-answer and dense controls remained unchanged.

This fixture is labelled **`derived_quantized_X`**. It is not an authentic quantized-model graph capture. It tests ternary/F32 matvec on quantize/dequantize-grid activations; it does not implement integer arithmetic, repair the full model, or validate the team's undisclosed checkpoint rules.

Official source revision: `096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4`; source SHA-256: `a8b3170f1824f010701f4e033fbc33d199bc2ea344ddbf2554594b66fada8b1c`.

| Derived case | S max absolute error | Original PoC max absolute error | Q4 max absolute error | PoC rows differing bitwise from S |
|---|---:|---:|---:|---:|
| Step 1, official F32 ActQuant(X) | 1.0247051e−5 | 4.8233414e−7 | 1.0247051e−5 | 2,536 |
| Step 16, official F32 ActQuant(X) | 8.8373527e−6 | 4.3210093e−7 | 8.8373527e−6 | 2,537 |
| Step 32, official F32 ActQuant(X) | 1.1220904e−5 | 4.5182969e−7 | 1.1220904e−5 | 2,528 |

Again, all rows of all three routes pass FP64 and stock-relative tolerance, with no nonfinite values. Q4 remains bitwise-identical to S.

## Input and execution evidence

Public model SHA-256: `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2`.

The three captured Q matrices share W SHA-256 `b2dea7f42cacfd08ed780df979f6f28240026795b1f5c87ca8c16fc7da23dc58`. Exact original/derived X hashes, all numerical metrics, source hashes and relative evidence filenames are recorded in [original-poc-phone-summary.json](original-poc-phone-summary.json).

The independent reanalysis unpacks each logical K directly, decodes each FP16 block scale and uses FP64 products with `math.fsum` for **every output row**. It checks all output values, input hashes, legal trits/scales, native/reference fidelity, hand-computed values, expected launch geometry, shared W/X and separate Y handles, runtime pipeline names, signed embedded-framework identity, and 15 native plus 15 console GPU completions per run.

Coverage evidence consists of the expected row ownership map, NaN-initialized output becoming finite at all rows, and native prefix/suffix-canary checks after GPU completion. Raw guard bytes were not dumped, and no per-row hardware write counter was used. The host therefore cross-checks those native witnesses rather than independently re-reading guard bytes.

Both runs returned **local numerical exit 0**. Whole-model `model_output_pass` and `model_semantics_pass` remain **null**, and global strict V1 remains **exit 2 / unverified**. There is no new model-graph dispatch or performance claim for the original PoC in this test.

## Reproduction and local evidence

Raw tensors, logs, binaries and corpus data remain local. Set `RAW` to the local `2026-11-15-review-221448` artifact directory; no device identifier or private host path is published here.

Physical-phone test, using a signed build and the original fixture:

```sh
python3 bitnet/ios/run.py --mode poc-verify \
  --build-dir "$SIGNED_BUILD" --device "$IPHONE_DEVICE_ID" \
  --model "$PUBLIC_2B_GGUF" --fixture "$RAW/original-poc-fixture" \
  --output "$RAW/runs/original-poc-repeat" --install --reuse-model
```

Use `$RAW/original-poc-quantized-fixture` and a new output directory for the separate transformed-input control. These fixtures are local inputs, not tracked repository payloads.

CPU-only independent recomputation of saved original-phone results:

```sh
python3 bitnet/analysis/review-2026-11-15/reanalyze-phone-poc.py \
  --input "$RAW/runs/original-poc" \
  --analyzer bitnet/validation/analyze.py \
  --output "$RAW/original-poc-reanalysis-repeat"
```

Relative evidence index:

- `runs/original-poc/`: native results, console/build/launch identity and raw outputs.
- `runs/original-poc-actquant-derived/`: separate transformed-input execution and raw outputs.
- `phone-poc-independent-reanalysis/{SUMMARY.json,RESULTS.csv,SOURCE_BINARY_BINDING.json}`.
- `phone-poc-derived-independent-reanalysis/{SUMMARY.json,RESULTS.csv}`.
- `original-poc-quantized-fixture/MANIFEST.json`: original/derived X identities and exact official-source provenance.

The current recommendation is to present the original 4×2 shader as the authored matrix-multiply PoC with this new physical-phone sample-I/O qualification, and Q4 separately as a stock-helper-derived specialization. Public-model semantic repair and representative quality evaluation remain separate work.
