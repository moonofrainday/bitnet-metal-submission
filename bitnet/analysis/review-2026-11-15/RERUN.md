# Reproduction entry points

Run from the repository root. Use fresh output directories. Model files, real Q fixtures, raw histories/logits and signed artifacts are supplied locally; none are downloaded from this source repository.

## Fresh physical-phone application

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
python3 bitnet/ios/build.py --team "$APPLE_TEAM_ID" \
  --bundle-id "$BUNDLE_ID" --allow-provisioning-updates \
  --output "$ARTIFACTS/build"
```

This clean build compiles the matching framework and external shaders from source. To reuse a framework, the builder requires exact current native-source and framework hashes; it rejects an incompatible prebuilt binary.

## Exact original authored PoC sample-I/O

```sh
python3 bitnet/ios/prepare-original-poc.py \
  --real-fixture "$REAL_Q_FIXTURE" --output "$ARTIFACTS/original-fixture"
python3 bitnet/ios/run.py --mode poc-verify \
  --build-dir "$ARTIFACTS/build" --device "$IPHONE_DEVICE_ID" \
  --model "$PUBLIC_2B_GGUF" --fixture "$ARTIFACTS/original-fixture" \
  --install --output "$ARTIFACTS/original-phone"
python3 bitnet/analysis/review-2026-11-15/reanalyze-phone-poc.py \
  --input "$ARTIFACTS/original-phone" --analyzer bitnet/validation/analyze.py \
  --output "$ARTIFACTS/original-independent"
```

Use `--reuse-model` only after the same model is installed; the app still verifies its SHA-256. The fixture preparer requires the accepted real steps 1/16/32 W/X and adds hand-known/dense controls; a synthetic-only replacement cannot claim real-input coverage.

## Same stock pipeline after the full guard

```sh
python3 bitnet/ios/run.py --mode graph-verify --routes S,QB,Q4 --layers 0 \
  --history-dir bitnet/ios/histories/primary --q-readback --warmups 1 \
  --build-dir "$ARTIFACTS/build" --device "$IPHONE_DEVICE_ID" \
  --model "$PUBLIC_2B_GGUF" --install --output "$ARTIFACTS/guard-verify"
python3 bitnet/ios/run.py --mode decode-benchmark --routes S,QB,Q4 --layers 0 \
  --history-dir bitnet/ios/histories/primary --comparisons AA,SQB,SQ4 \
  --warmups 1 --rounds 3 --pairs 7 --build-dir "$ARTIFACTS/build" \
  --device "$IPHONE_DEVICE_ID" --model "$PUBLIC_2B_GGUF" \
  --install --output "$ARTIFACTS/guard-screen"
```

This is a short 32-token wall-time screen, not a 201-pair isolated GPU benchmark. It cannot split CPU versus GPU cost. Invalid/missing QB guards are rejected, even if the app safely falls back to stock. See `bitnet/docs/PHONE_GUARD_CONTROL_2026-11-15.md` and the portable analyzer's `--help`.

## Trusted-source CPU and exact-token phone quality smoke

Follow `bitnet/reference/RERUN.md` first: pinned official source, checkpoint, tokenizer and data hashes; BF16 and F32 controls; stable FP64 rescoring. This is numerical/semantic research, not a Mac speed test.

```sh
python3 bitnet/ios/ppl-prepare.py --tokens-json "$F32_REFERENCE/tokens.json" \
  --reference-manifest "$F32_REFERENCE/MANIFEST.json" \
  --output "$ARTIFACTS/quality-fixture"
python3 bitnet/ios/ppl-run.py --mode ppl-verify --build-dir "$ARTIFACTS/build" \
  --device "$IPHONE_DEVICE_ID" --model "$PUBLIC_2B_GGUF" \
  --fixture "$ARTIFACTS/quality-fixture" --install --output "$ARTIFACTS/phone-ppl"
python3 bitnet/analysis-tools/analyze-ppl-smoke.py \
  --input "$ARTIFACTS/phone-ppl" --fixture "$ARTIFACTS/quality-fixture" \
  --reference-f32 "$F32_REFERENCE" --reference-bf16 "$BF16_REFERENCE" \
  --output "$ARTIFACTS/phone-ppl-independent"
```

The phone helper disables every custom route. All 128 target positions, complete logits, actual 31/31 offload and synchronized completion are required. The PPL measurement does not add the missing quantizer, establish representative quality, or turn global V1 acceptance into a pass. The original strict generation comparison exit 1 remains unchanged.
