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

Q4 opt-in uses the current Q symbol with rows=4, SIMD groups=1, `GGML_METAL_BITNET_POC=1`, `GGML_METAL_BITNET_FFN_DECODE=1`, and the freshly compiled Q4 library. Unset FFN mechanism controls for a Q-only run. G/F8/K have separate controls documented in `bitnet/shaders/README.md`; they must never be silently enabled together. Use a fresh process for each library/control test.

Disable presence-tested diagnostic controls such as `GGML_METAL_BITNET_FFN_TIME`, `GGML_METAL_BITNET_LAYER_TIME` and `GGML_METAL_BITNET_FFN_TRACE` by **unsetting** them, rather than assigning `0`. Formal timing must also disable capture, diagnostic copies/readback and trace.
