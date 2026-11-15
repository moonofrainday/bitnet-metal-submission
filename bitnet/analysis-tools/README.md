# Complete layer-0 Q readback analysis

`analyze-q4-complete.py` is CPU-only. It checks the fetched phone evidence and never launches Metal, Xcode or a physical device.

Requirements: Python 3.9 or newer and NumPy. Use the experiment's isolated Python environment. Run this analysis while formal phone performance sampling is idle.

```sh
python analyze-q4-complete.py \
  --input /path/to/fresh-probe-on-host-run \
  --probe-off /path/to/matched-probe-off-host-run \
  --runtime-log /path/to/fresh-probe-on-host-run/logs/launch.log \
  --output /path/to/fresh-local-analysis
```

The input directories can be host run directories containing `device/`, or their fetched device directory. Each host run must retain `BUILD_IDENTITY.json`. The captured model/checkpoint, selected signed build, Q4 library, 27 prompt IDs, 32 continuation IDs, context/batch settings and F16 KV must match. Probe-on may additionally load the diagnostic copy library; the actual Q4 library must be identical.

For both S and Q4 the analyzer requires all 68 manifest-bound files: one weight snapshot, all 32 X/Y pairs, and layer-0 residual outputs at steps 1/16/32. It independently checks actual copy encode/completion records, tensor shapes/strides/byte ranges, exact diagnostic phases, live weight binding consistency and stock/custom pipeline counts/launch. A saved file or completion marker alone is insufficient.

Every captured Q output is checked against an independent logical TQ2_0 unpack and FP64 CPU computation. Selected rows are independently recomputed from raw bytes with `struct`/`math.fsum`. Stock and Q4 outputs are compared only after their W/X hashes match. Layer outputs and all 33 logits vectors use the frozen stock-relative tolerance. Probe-on versus both the internal uninstrumented repeat and external probe-off run must match all 33 logits vectors bit for bit. No equality of uncaptured probe-off Q/residual tensors is inferred.

The criterion remains `abs(actual-reference) <= 2e-4 + 2e-5*abs(reference)`, with zero failed/nonfinite rows. `parity_pass`, `fp64_pass` and `model_output_pass` remain separate. This is the declared Q-only capture mode, not complete-FFN/Transformer FP64 or model-quality acceptance. Separate missing-library/invalid-geometry fallback controls belong to their own runner results.

Default process exit is strict: 0 for all required checks passing, 1 for an observed measured failure, and 2 for missing/invalid evidence or a reference-oracle inconsistency. `--exit-mode processing` returns 0 only when analysis completed, while preserving `strict_exit_code` and every failure in `SUMMARY.json`. It cannot turn a strict numerical failure into a pass.

The output includes `SUMMARY.json`, complete per-row `q-rows.csv`, the failure-only `failed-q-rows.csv`, `FP64_ORACLE.json`, `INPUT_HASHES.json`, `RUNTIME_EVIDENCE.json`, and full-vector logit summaries/margins in `LOGITS.json`. Zero-reference rows are explicitly marked; their relative error is not divided by zero. These outputs are local raw/derived evidence; publish only deliberately curated analysis summaries, not tensor payloads, per-row data or private host/device paths.

The packed weight requires 62 diagnostic copy dispatches because the frozen copy shader permits at most 6,912 uint values per dispatch. Its first 61 chunks have 27,648 bytes each and its final chunk has 3,072 bytes. The analyzer checks complete disjoint byte coverage and command identity. Each route still produces 68 payload files, using 129 actual diagnostic dispatches in total. A previous oversized single dispatch copied only the prefix; that dataset remains invalid and cannot be repaired by substituting a disk weight.

## Prepare the exact new operator input

After complete capture acceptance succeeds, prepare the existing phone Q harness's three-input format:

```sh
python prepare-q4-fixture.py \
  --input /path/to/accepted-primary-probe-on \
  --acceptance /path/to/complete-analysis/SUMMARY.json \
  --build-dir /path/to/the-same-signed-build \
  --output /path/to/fresh-local-fixture
```

The preparer accepts only a strict passing capture result bound to the same signed build. It checks the actual captured weight against the audited GGUF tensor, checks the original step-16 X hash, and copies actual S W/X/Y for steps 1/16/32. It computes fresh CPU FP64 vectors and writes the native `MANIFEST.json` with current C8/Q4 library hashes. No model/tensor substitution or GPU execution occurs. The fixture must remain outside the source checkout.

## Recompute actual paired phone timings

`analyze-q4-timing.py` uses only the Python standard library. It checks native CSV pair/order/count integrity, signed build identity and successful app/host exits. It preserves every sample and separates rounds, AB/BA order, chronological quartiles and observed thermal endpoint cohorts.

```sh
python analyze-q4-timing.py --mode operator \
  --input /path/to/q-benchmark-host-run \
  --accepted-q4 /path/to/complete-analysis/SUMMARY.json \
  --output /path/to/fresh-operator-analysis

python analyze-q4-timing.py --mode decode \
  --input /path/to/AA-SQ4-decode-host-run \
  --accepted-q4 /path/to/complete-analysis/SUMMARY.json \
  --output /path/to/fresh-decode-analysis
```

The operator mode verifies all 1,809 pairs (three rounds of AA/SQ/CQ, 201 pairs/group), unchanged three-input qualification and matching native p50/p95/paired-median/win summaries. The metric is one dispatch's GPU command-buffer elapsed time in microseconds. The decode mode verifies three AA/SQ4 rounds using the actual declared pair count, all 32 per-token intervals, chronological execution, same context/KV/prefix settings and clean timing flags. Its metric is synchronized 32-token host-wall milliseconds, including normal encoding/GPU completion while excluding reset and prefix restoration. The two metrics are never pooled.

Native operator quantiles use sorted linear interpolation at `(n-1)*p`; the independent analyzer uses the same definition. Paired B-A and median per-pair relative changes are reported separately from the ratio of route medians. A/A is calibration, never a value subtracted from candidate samples. Operator thermal observations are contrast-block endpoints (the first precedes warmup); per-pair thermal states are unavailable. Decode thermal observations are path-entry and after-decode endpoints, rather than continuous temperature/clock measurements. Nominal endpoints do not prove a cold or fixed-frequency device.

Timing outputs include `ROUND_SUMMARY.csv`, complete per-round/order/quartile/cohort statistics in `SUMMARY.json`, and a digest-bound `PROVENANCE.json`. Decode additionally emits `TOKEN_STEP_SUMMARY.csv` and `WALL_MINUS_TOKEN_INTERVALS.csv`. Only selected analysis tables should be published. Successful timing analysis establishes data integrity/statistics, not new numerical or model-quality acceptance. Without `--accepted-q4`, prior complete capture acceptance is explicitly null.

## Selected Down negative-control analysis

`analyze-down-selected.py` uses only the standard library. It independently unpacks each of 12 eight-row slices from the actual full TQ2_0 weight and checks all 96 outputs/path using the fixed captured 6912-element F32 norm input and logical-K `math.fsum`. The actual stock wrapper is compared with the embedded stock pipeline; correction-end remains a separate arithmetic experiment. It checks real GPU completion identities, canaries, input hashes and the native/host numerical flags.

```sh
python analyze-down-selected.py \
  --input /path/to/down-selected-host-run \
  --output /path/to/fresh-down-analysis --strict
```

The observed selected-row result is strict 1: 12 failed rows in each of stock, S-wrap and correction-end. S-wrap is stock-bit-identical; the variant changes 48 rows without repairing the known failures. Processing mode (omit `--strict`) returns 0 after valid evidence is analyzed while preserving `strict_exit_code=1`; missing/invalid evidence returns 2. The optional `--old-stock` compares the same 96 selected rows with a historical full stock capture. No full-matrix, model-quality or performance acceptance is inferred. Per-row/reference CSVs remain local.
