# Build and run

Run commands from the repository root. Keep models, build products and raw outputs outside the checkout. Every entry uses fresh output directories; an existing result must not be silently replaced.

## Tools

Apple Silicon macOS, Git, CMake, full Xcode/Metal and Python >=3.9 are sufficient for the standalone synthetic validation. The current package was prepared with macOS 27.2.1, Xcode 27.2 and SDK 27.2; these are recorded environment identities, not minimum requirements. The iPhone source package targets iOS 16.4 or later. Inspect `xcodebuild -version`, `xcodebuild -showsdks` and `xcrun -f metal` in your environment.

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
python3 bitnet/scripts/build-mac.py \
  --build ../bitnet-local/patched-build \
  --output ../bitnet-local/patched-checks
```

Use `--cmake /absolute/path/to/cmake` if CMake is not on PATH. This is a correctness/build entry, with no Mac performance campaign. It builds the patched core and native backend test. The pinned test's `--help` returns 1 after printing usage; the script verifies the usage and accepts that informational status only. A failed configure/build/test remains nonzero.

## Clean standalone stock / original PoC / stock4x1

```sh
python3 bitnet/validation/bootstrap.py \
  --work-root ../bitnet-local/standalone
```

This fetches the exact public upstream commit into its own cache, creates a pristine pinned checkout, performs a fresh build and runs the model-free cases. It never treats this modified submission tree as pristine stock. To reuse existing upstream Git objects without fetching, add `--qvac-repo /path/to/existing/qvac/repository`. The exact commit and clean-state checks remain mandatory.

For local historical steps 1/16/32 and optional GGUF verification, see `bitnet/validation/RERUN.md`. Tensor fixtures are intentionally not published. Hand-checkable and dense synthetic tests remain reproducible without those private local files.

## Candidate shader compilation

```sh
python3 bitnet/scripts/build-shaders.py --sdk macosx \
  --output ../bitnet-local/mac-shaders
python3 bitnet/scripts/build-shaders.py --sdk iphoneos \
  --output ../bitnet-local/phone-shaders
```

Source digests, exact compiler/link commands and output hashes are generated locally. Use `--plan` to inspect commands or `--emit-air` to retain textual AIR. Post/K retain safe math and contraction settings; F8/expand retain the historical macro/name mappings. Offline compilation is not device numerical validation.

## Public checkpoint conversion

Use the packed source checkpoint `microsoft/bitnet-b1.58-2B-4T`, revision `04c3b9ad9361b824064a1f25ea60a8be9599b127`. The official I2_S GGUF is incompatible with this TQ2_0 contract. Download checkpoint/tokenizer/config files to an external cache and retain their immutable revision/checksums.

Historical conversion dependencies were Python 3.11, torch 2.11.0, transformers 5.10.0, NumPy 2.2.6, tokenizers 0.22.2 and safetensors 0.8.0, with the in-tree `gguf-py`. The pinned base declares transformers 5.10.0; its historical package was yanked. That dependency identity is preserved for reproducibility, not presented as a new recommended global installation. Use an isolated environment; the standalone numerical tests do not require it.

```sh
MODEL_DIR=/absolute/path/to/packed-checkpoint
GGUF=/absolute/path/to/microsoft-bitnet-b1.58-2B-4T-tq2_0.gguf
OMP_NUM_THREADS=6 python convert_hf_to_gguf.py "$MODEL_DIR" \
  --outtype tq2_0 --no-lazy --outfile "$GGUF"
python bitnet/scripts/audit-conversion.py \
  --gguf "$GGUF" --checkpoint "$MODEL_DIR/model.safetensors" \
  --output /absolute/path/to/local-conversion-audit.json
shasum -a 256 "$GGUF"
```

The historical final GGUF is 1,203,563,808 bytes, SHA-256 `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2`. Architecture: 30 blocks, hidden size 2560, FFN size 6912, 20 Q heads/5 KV heads, vocabulary 128256, tied embeddings. Unique parameter count: 2,412,820,480. Tensor types: 210 TQ2_0, one F16 and 121 F32. The all-matrix audit checks trits and per-block FP16 direct scales; lazy conversion is disabled to avoid the previously observed meta/CPU device mismatch.

## iPhone build and launch

Follow `bitnet/ios/README.md`. It stages a copy of the upstream SwiftUI project, compiles matching helper objects and builds an iphoneos-arm64-only XCFramework from this source. No signed app, model, signing identity or device ID is included. Supply your team, bundle ID and connected device explicitly for deployment. Generic unsigned compilation is a separate build check and does not establish a physical-device run.

The following legacy Q4 controls describe ActQuant OFF. The new joint route additionally requires `LLAMA_BITNET_ACTQUANT=1` and default-off `LLAMA_BITNET_ACTQUANT_Q4=1`; the latter authorizes only layer-0 Q with rows=4 and one SIMD group and does not enable C8 or FFN. Joint phone stock-ON qualification passes both histories; only its new isolated-operator performance is measured, not joint whole decode or corpus/PPL. Q4 opt-in uses the current Q symbol with rows=4, SIMD groups=1, `GGML_METAL_BITNET_POC=1`, `GGML_METAL_BITNET_FFN_DECODE=1`, and the freshly compiled Q4 library. Unset FFN mechanism controls for a Q-only run. G/F8/K have separate controls documented in `bitnet/shaders/README.md`; they must never be silently enabled together. Use a fresh process for each library/control test.

Disable presence-tested diagnostic controls such as `GGML_METAL_BITNET_FFN_TIME`, `GGML_METAL_BITNET_LAYER_TIME` and `GGML_METAL_BITNET_FFN_TRACE` by **unsetting** them, rather than assigning `0`. Formal timing must also disable capture, diagnostic copies/readback and trace.

## Select the intended validation profile

| Entry | Scope and expected verdict |
|---|---|
| `bitnet/validation/bootstrap.py`, original `poc-verify` phone mode | Original authored 4x2, actual stock and Q4 supplied-input FP64 samples. Model-free synthetic scope is explicit; full-model gates remain null. |
| `bitnet/ios/run.py --mode graph-verify --routes S,Q4` | Historical ActQuant-OFF Q4 layer-0 compatibility. Its prior two-history strict exit is 0; performance numbers are legacy OFF results. |
| `bitnet/actquant/run_phone.py` plus independent analyzers | V2 quantizer and ActQuant-ON model diagnostics. Recorded A16 operator strict exit is 0; completed trusted-reference model comparison is strict exit 1. |
| New joint ActQuant/Q4 | Source `edbd109`; both histories pass its scoped stock-ON analyzer, strict0. Fresh ON operator timing is separate; joint decode/corpus remain unmeasured. Rebuild and qualify the intended binary. |

The historical reviewed/import snapshot is `bbf70e9cc70302534ff87b4f34d4ce52a9670baa`. The tested ActQuant A16 executable is from `4c249e3`, with evidence/docs consolidated at `e84732c`. Patches in this checkout are already applied. `pinned-to-iphone.patch`, `q4-readback-additive.patch`, `q-guard-control-2026-11-15.patch` and `public-2b-actquant.patch` describe successive scopes; they are not interchangeable clean-baseline inputs. See `CHANGES_FROM_QVAC.md` before replaying on any pristine tree.

## Original authored shader on iPhone

Use a fresh signed build and an explicitly prepared local PoC fixture, not an old installation's absolute Documents paths. Fixture preparation and schema are documented in `bitnet/validation/README.md` and `bitnet/ios/README.md`.

```sh
python3 bitnet/ios/run.py --build-dir "$SIGNED_BUILD" \
  --device "$IPHONE_DEVICE_ID" --model "$PUBLIC_2B_GGUF" \
  --mode poc-verify --fixture "$POC_FIXTURE" --install \
  --output "$ARTIFACTS/original-poc"
```

This route validates the exact original source alongside actual stock and Q4. It does not enable the original shader in a full Transformer graph. Use the matched independent CPU analyzer and preserve its scoped verdict; no broader model-output pass follows from sample-I/O.

## ActQuant dependencies and fresh fixture preparation

The source-derived oracle uses the isolated dependencies in `bitnet/reference/requirements.txt`. It does not require importing the historical/yanked Transformers package. Original checkpoint/tokenizer/config and the F32/BF16 reference caches remain external. To reproduce those caches, follow `bitnet/reference/README.md`; a summary hash alone cannot regenerate a missing real input.

```sh
python3.11 -m venv "$REFERENCE_VENV"
"$REFERENCE_VENV/bin/python" -m pip install -r bitnet/reference/requirements.txt
export PY="$REFERENCE_VENV/bin/python"
"$PY" bitnet/actquant/prepare_fixtures.py \
  --official-source bitnet/reference/official-bitnet-096f25ae.py \
  --synthetic-only --output "$ARTIFACTS/actq-synthetic"
"$PY" bitnet/actquant/run_mac.py --mode operator \
  --fixture "$ARTIFACTS/actq-synthetic" --output "$ARTIFACTS/actq-fresh-mac"
```

This is a fresh Mac numerical/build check only, with 15 synthetic cases and `missing_real_capture=true`; it makes no Mac performance or real-input claim. For the complete recorded 20-case suite, prepare the three real Q inputs and two boundary rows from the existing local captures:

```sh
"$PY" bitnet/actquant/prepare_fixtures.py \
  --official-source bitnet/reference/official-bitnet-096f25ae.py \
  --real-fixture "$REAL_Q_FIXTURE" \
  --boundary-capture "$V1_BOUNDARY_CAPTURE" \
  --boundary-failures "$V1_BOUNDARY_FAILURES" \
  --output "$ARTIFACTS/actq-real-fixtures"
```

The seven-value reciprocal-boundary case is source-defined. Raw captured rows stay local. Missing real inputs must be reported, not replaced silently by the synthetic suite. `bitnet/actquant/README.md` describes the 15/18/20-case scopes and exact data contracts.

## ActQuant physical-phone operator and model diagnostics

Build from the intended current source using `bitnet/ios/build.py`; no older framework supplies the new op. The phone launcher verifies every recorded framework/package/app/object hash and requires a successful signed build. Set local variables explicitly; no device or team identifier is stored here.

```sh
python3 bitnet/ios/build.py --team "$APPLE_TEAM_ID" \
  --bundle-id "$BUNDLE_ID" --allow-provisioning-updates \
  --output "$ARTIFACTS/actq-build"
"$PY" bitnet/actquant/run_phone.py --run --operation operator \
  --build-dir "$ARTIFACTS/actq-build" --device "$IPHONE_DEVICE_ID" \
  --model "$PUBLIC_2B_GGUF" --fixture "$ARTIFACTS/actq-real-fixtures" \
  --trace --output "$ARTIFACTS/actq-operator"
"$PY" bitnet/actquant/analyze_op.py --fixture "$ARTIFACTS/actq-real-fixtures" \
  --actual "$ARTIFACTS/actq-operator/device/native" \
  --runtime "$ARTIFACTS/actq-operator/device/native/RUN.json" --require-runtime \
  --output "$ARTIFACTS/actq-operator-analysis"
"$PY" bitnet/actquant/run_phone.py --run --operation model \
  --build-dir "$ARTIFACTS/actq-build" --device "$IPHONE_DEVICE_ID" \
  --model "$PUBLIC_2B_GGUF" --tokens "$F32_REFERENCE/tokens.json" \
  --actquant on --workload teacher --capture --trace \
  --operator-run "$ARTIFACTS/actq-operator" \
  --operator-analysis "$ARTIFACTS/actq-operator-analysis/SUMMARY.json" \
  --output "$ARTIFACTS/actq-teacher-on"
"$PY" bitnet/actquant/compare_model.py \
  --run "$ARTIFACTS/actq-teacher-on/device/native" \
  --reference-f32 "$F32_REFERENCE" --reference-bf16 "$BF16_REFERENCE" \
  --output "$ARTIFACTS/actq-teacher-on-analysis"
```

The token file is the reference cache's exact `tokens.json`. The recorded operator analyzer exits 0, but the current source-F32-ON model analyzer exits **1** for observed residual/logit failures. Capture versus ordinary execution, OFF controls and corpus ON/OFF use separate new output directories; follow the complete recipes in `bitnet/analysis/actquant-iphone-2026-11-16/README.md`. Native/collection exit 0 only proves execution and identity; never ignore a later strict nonzero result or report a processing-only exit as acceptance. The new same-build ActQuant/Q4 scope now passes strict0 on both histories; its stock-ON reference is separate from trusted-source model strict1. Follow `../analysis/actquant-q4-joint-2026-11-16/README.md` for the exact captured/plain/fallback runs and independent analyzer. Fresh ON operator pairing measures 2.61%-2.68% reduction; joint full decode and corpus/PPL remain unmeasured.
