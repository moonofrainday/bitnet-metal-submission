# Physical-iPhone current-stock quality smoke

The existing public BitNet 2B QVAC path completed a 128-target corpus-scoring smoke on iPhone 14 Pro / Apple A16 GPU. Its independently recomputed mean NLL is **2.9490714489629446**, giving **PPL 19.08822110917837**. The actual log confirms **31/31 layers offloaded** and `MTL0 KV buffer size = 37.50 MiB`.

This result measures the current path, whose public-reference activation-quantization omission is separately demonstrated. It is not a repaired W1.58A8 path, representative benchmark, model-semantic acceptance, or evidence of better quality than the reference.

## Exact workload

- Public checkpoint experiment; no team checkpoint tested. Model SHA-256: `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2`.
- The same 128 ordinary token IDs from the canonical WikiText-2-raw-v1 test-split slice used by the source-derived reference, with one explicit BOS 128000. No retokenization or chat-template application on the phone.
- Six threads, context 512, batch/ubatch 128, F16 K/V, GPU KQV placement requested, flash attention disabled. All custom routes disabled.
- Sequential N=1 teacher forcing: for `i = 0..127`, feed `ids[i]`, synchronize, save the full 128,256-element F32 vector and score target `ids[i+1]`. The final target need not itself be decoded to score it.
- Stable F64 logsumexp over the saved F32 logits. The phone accumulates NLL in long double; the independent host uses `math.fsum`.

The source-derived reference uses full-sequence eager attention with no KV cache; both F32 and the recommended BF16 arithmetic are retained. Those execution differences remain a boundary on cross-backend interpretation.

## Independent readback and scoring checks

All **16,416,768** saved logits are finite. Exact token/fixture/model identities, all 128 CSV input/target positions, greedy IDs, individual NLL values, aggregate NLL/PPL, actual offload and ordered completion markers passed **22/22** integrity checks. There are zero CSV mismatches. Maximum phone/independent per-target NLL difference is `5.258016244624741e-13`.

GPU completion evidence is successful `llama_decode`, `llama_synchronize` and complete logits readback on each step. This helper does not instrument individual Metal command-buffer completion callbacks. No timing or speed claim is made.

Actual native source is recorded in the fresh combined build at `fd8234fcf5b2683f67d0c3148f2f3ae2e4fed454`, on pinned QVAC `2db8b046d555316019ae75d1be90484adc8b9330` plus the validated compatibility/integration patches. The helper hash is `10e0642c74d1648c8b9eb780bdebe6398881e2fdec9da9b516f4734eb435b73c`. Build/source hashes bind the result; later report edits do not imply a new binary measurement.

## Comparison with the same saved reference workload

All PPL values below use independently recomputed **F64 scoring** over saved F32 logits. Original Torch F32-loss CSVs remain preserved separately; their scoring-rounding differences are not mixed into this table.

| Execution | PPL, 128 targets | Greedy agreement with phone | Full-logit RMSE versus phone |
|---|---:|---:|---:|
| Physical A16, stock QVAC | **19.0882211092** | — | — |
| Source-derived F32, ActQuant OFF | 19.0885661082 | 128/128 | 0.0005368084 |
| Source-derived F32, ActQuant ON | 19.1295295294 | 126/128 | 0.2132423246 |
| Source-derived BF16, ActQuant OFF | 19.0982865966 | 125/128 | 0.0386638250 |
| Source-derived BF16, ActQuant ON | 19.1888254281 | 124/128 | 0.2156543834 |

The phone is much closer to the F32 ActQuant-OFF control on this corpus slice. Together with the independent same-W/X projection control and the ON/OFF reference intervention, this supports the diagnosis that the current phone path does not implement the public reference's activation quantizer. It does not isolate every remaining phone/reference difference or prove fidelity to all target-model semantics.

The tiny PPL ordering must **not** be interpreted as quality superiority of ActQuant OFF. The slice is nonrepresentative, has no significance estimate, and may have appeared during pretraining. The team's activation rules remain unknown.

Measurement integrity passes with CLI exit **0**. `parity_pass`, `fp64_pass`, `model_output_pass` and `model_semantics_pass` remain **null**, `model_semantics_verified=false`, and global strict V1 remains **2 / unverified**. FP64 **scoring arithmetic** does not establish FP64 acceptance of the model's internal operators.

## Reproduction

Use the signed build and fixture preparation described in [PHONE_STOCK_QUALITY_SMOKE.md](../../docs/PHONE_STOCK_QUALITY_SMOKE.md). Let `RAW` name the local `2026-11-15-review-221448` artifact directory and `SEMANTICS` the local source-derived reference output directory.

```sh
python3 bitnet/ios/ppl-run.py --mode ppl-verify \
  --build-dir "$SIGNED_BUILD" --device "$IPHONE_DEVICE_ID" \
  --model "$PUBLIC_2B_GGUF" --fixture "$RAW/phone-ppl-fixture" \
  --output "$RAW/runs/phone-ppl-repeat" --install --reuse-model
```

After the phone run, CPU-only integrity and reference comparison:

```sh
python3 bitnet/analysis-tools/analyze-ppl-smoke.py \
  --input "$RAW/runs/phone-ppl" --fixture "$RAW/phone-ppl-fixture" \
  --reference-f32 "$SEMANTICS/reference-full-f32" \
  --reference-bf16 "$SEMANTICS/reference-full" \
  --output "$RAW/phone-ppl-analysis-repeat"
```

The analyzer requires NumPy and runs no model or GPU. It produces a publishable aggregate `SUMMARY.json` and a **local-only** `per-target-local.csv`. Raw corpus IDs, complete logits, per-target CSV and device logs stay local. [phone-ppl-summary.json](phone-ppl-summary.json) contains the sanitized aggregate results and relative evidence index.
