# Physical-iPhone current-stock quality smoke

This entry point scores the existing public BitNet 2B QVAC path. It does not repair or certify the public model's activation-quantization semantics, replace any kernel, or measure inference speed. The corpus slice has 128 scored targets and is too small for a representative quality claim.

The model's SHA-256 must be `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2`. The runner accepts exactly the saved corpus IDs used by the source-derived reference: one explicit BOS plus 128 ordinary token IDs. It does not tokenize or apply a chat template on the phone.

## Build and prepare

Use the existing signed physical-device build command in `bitnet/ios/build.py`, with the same matching clean-built iOS framework and a **new** output directory. The app must include `iphone-ppl.mm` in its recorded package sources; old app binaries are rejected. No framework-core source change is needed.

```sh
python3 bitnet/ios/ppl-prepare.py \
  --tokens-json /absolute/local/semantics/reference-full-f32/tokens.json \
  --reference-manifest /absolute/local/semantics/reference-full-f32/MANIFEST.json \
  --output /absolute/local/new-quality-fixture
```

## Run

```sh
python3 bitnet/ios/ppl-run.py --mode ppl-verify \
  --build-dir /absolute/local/new-signed-build \
  --device "$IPHONE_DEVICE_ID" \
  --model /absolute/local/microsoft-bitnet-b1.58-2B-4T-tq2_0.gguf \
  --fixture /absolute/local/new-quality-fixture \
  --output /absolute/local/new-phone-quality-run \
  --install --reuse-model
```

Omit `--reuse-model` to transfer the model. Reusing it still requires a full checksum verification in the app. Every run installs the selected signed app and records the exact launch command, binary identities, fixture identity, logs and process exit code.

## Exact execution and scoring scope

- Stock Metal only; all `GGML_METAL_BITNET_*` environment switches are cleared.
- Six CPU threads, context 512, batch/ubatch 128, F16 K/V, GPU KQV placement requested, flash attention disabled.
- 128 sequential N=1 decode calls. At position `i`, feed token `ids[i]`, wait with `llama_synchronize`, save the full 128,256-element F32 logit vector, and score `ids[i+1]`.
- The native helper uses stable F64 logsumexp for each NLL and long-double summation of the 128 NLL values. It retains all raw logits for independent host recomputation.
- Require the actual log to confirm 31/31 layers offloaded, the selected A16 GPU, all 128 synchronized completion markers, finite complete logits, and exact corpus/target counts.

`GPU_completion_evidence` specifically means successful decode plus `llama_synchronize` and complete readback; this helper does not instrument individual Metal command-buffer completion callbacks.

## Exit and evidence rules

CLI exit0 means this scoped measurement completed. It does not mean model acceptance passed. `parity_pass`, `fp64_pass`, `model_output_pass` and `model_semantics_pass` remain null, and global strict V1 remains exit2 because those checks were not performed here. `model_semantics_verified=false` explicitly retains the known public-reference ActQuant concern.

Raw `device/quality/logits.f32`, `nll.csv`, request/summary JSON and device logs remain local. Publish code, provenance and derived analysis only. Comparisons with the source-derived reference must disclose F32/BF16 arithmetic, sequential F16 KV versus no-cache full-sequence attention, and other execution differences; PPL similarity alone is not semantic proof.
