# Clean source build and standalone operator acceptance

Declared before this campaign's GPU correctness runs, 15 November 2026 (Asia/Tokyo).

This campaign prepares a submission candidate. It rebuilds the pinned QVAC backend, the original authored PoC, and the selected stock4x1 wrapper. Its execution device is the MacBook Pro M2 Pro. Mac results cover build and numerical correctness only. Submission performance results must come from the physical iPhone 14 Pro/A16; this campaign performs no inference performance benchmark.

## Scope and references

- QVAC commit: `2db8b046d555316019ae75d1be90484adc8b9330`, clean worktree, no upstream patches for this standalone test.
- W is contiguous TQ2_0, logical 2560x2560, 256 elements/66 bytes per block, 10 blocks/row, 660-byte row stride. Only trits 0/1/2 are valid. FP16 scales and F32 X must be finite.
- X and Y are contiguous F32 vectors of 2560 elements. The three GPU routes receive the same resident W/X, with separate guarded Y storage.
- S is the real stock pipeline selected by newly compiled QVAC libraries. The build embeds pinned Metal source; the stock PSO is compiled by that fresh runtime. S is not a copied shader approximation.
- PoC and stock4x1 are newly compiled from byte-identical historical shader sources. PoC uses 4x2/grid320/64 threads; stock4x1 uses 4x1/grid640/32 threads. Their slots are W0/X1/Y2; stock retains its native arguments0/W1/X2/Y3 ABI.
- The independent FP64 reference decodes FP16 scales and packed trits in logical K order, without stock coefficient algebra or GPU reduction. The GGML dequantizer is a separate cross-check, not the FP64 reference implementation.

## Cases and criteria

Required model-free cases are the original hand-computable fixture and seed-42 dense fixture. The hand result is Y[0]=1.375, Y[2559]=-1.25, and zero for all other rows. The full campaign also requires independently copied real pre-RoPE Q inputs from steps 1, 16, and 32, with their source hashes verified. Missing requested real data is not a passing model-free result.

Every route checks all 2560 outputs for complete writes, finite values, and unchanged output canaries. All GPU-versus-FP64 comparisons use the existing frozen bound:

```text
abs(actual-reference) <= 2e-4 + 2e-5*abs(reference)
zero failed rows; zero nonfinite values
```

Both candidates are also compared with S using the same pointwise rule, with S as that comparison's reference. Bitwise difference counts are reported separately. Original PoC need not be bitwise equal to S. No relative-L2 or smaller maximum error overrides a failed row. W/X identity, binding/geometry checks and GPU command completion are mandatory prerequisites.

This is operator acceptance. `model_output_pass=null` remains explicit: no full-layer, whole-model, generation, FFN or iPhone acceptance is inferred. Existing C8/FFN/model failures and their original exit codes remain untouched.

## Exits and preservation

- Native full run: 0 only when all required standalone checks pass; 1 for observed numerical failure; nonzero for setup, missing data, GPU execution or configuration failure.
- CPU-only reference checks are a separately labeled mode and do not mark GPU flags true.
- The orchestrator records and propagates a failing stage's actual exit code. A successful statistics script cannot cover a failed native stage.
- Backend regression must execute a nonzero TQ2_0 MUL_MAT test count on MTL0, with passed/failed/unsupported counts recorded separately.
- Existing output/build directories containing prior results are refused. New attempts use new directories, retaining failed logs and data.
- Raw inputs, vectors, per-row CSVs and full logs remain local. GitHub receives an explicit code and derived-analysis allowlist only, with no raw Release.
