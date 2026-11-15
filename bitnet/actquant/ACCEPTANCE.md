# Public-checkpoint ActQuant validation scope

This experiment repairs a known missing stage in the public Microsoft checkpoint path. It does not change the team's unknown contract, previous stock/candidate results, or the frozen rule:

`abs(actual-reference) <= 2e-4 + 2e-5 * abs(reference)`.

## Gates fixed before GPU runs

1. **Direct operator:** exact fixture identity and shapes, actual selected backend, completed graph plus synchronization, every output finite/written, unchanged inputs/canaries, zero frozen-bound failures. Recovered INT8 codes must exactly match the pinned official source. Bitwise F32 differences, ULPs and actual versus hypothetical scale evidence are reported separately. Native execution exit 0 alone is insufficient; the independent analyzer must pass.
2. **Graph integration:** explicit public-model opt-in, all custom Q/FFN routes disabled, 120 shared-input ActQuant nodes covering 210 linears per complete graph, no quantization of the tied lm_head. Every observed node must be on the requested backend and match official ActQuant applied to its same captured input. Decode/prefill counts, row/token mapping and buffer contracts are checked. New process/context is required for each flag state because the graph cache does not key the environment flag.
3. **Propagation/reference:** use the same 27 prompt and 32 continuation IDs, and the same BOS+128 corpus targets. Retain full-vector frozen-bound failures for intermediate layers and logits against the source-derived F32 ON reference. BF16 ON, F32 OFF and BF16 OFF are separate controls. Report maximum absolute/RMSE/relative-L2, failed rows, finite values, rankings, token-18 margin and independently FP64-scored PPL. Closer logits or equal greedy choices do not make this gate pass.
4. **Legacy invariance:** flag-off graph remains unchanged; existing 59-case TQ2_0 regression and saved operator results are reused/rerun as affected. The original CPU/Metal token strict exit 1 and existing complete-FFN FP64 failures remain immutable. New missing checks are null. Processing exit 0 cannot overwrite any strict failure.
5. **Physical phone:** no device run is performed tonight. Build/SDK checks and a rerun entry can be prepared. The morning plan must distinguish locally qualified ActQuant sample-I/O from full-model diagnostic comparison and model promotion. A failed required Mac gate blocks claiming a completed phone-ready model fix; any diagnostic-only phone run must be explicitly labeled.

## Arithmetic and reference boundaries

The authoritative primitive is the pinned official F32 arithmetic: `127/max(absmax(X),1e-5)`, multiply, ties-to-even round, clamp to [-128,127], divide, and cast back to input dtype. The first implemented contract is contiguous finite F32 input/output; the source BF16 profile remains distinct. A scalar code is stored as dequantized F32, not as an I8 tensor.

CPU TQ2_0 dot currently repacks activation to blockwise Q8_K. Metal TQ2_0 uses its existing arithmetic and may use different prefill/decode paths. TQ2_0 folded scales and summation topology also differ from source post-dot scale; F16 KV and source no-KV execution differ. Existing OFF-versus-OFF full logits already have measurable numerical drift. These boundaries explain why exact local ActQuant is necessary but not sufficient for a complete trusted-model numerical pass. They are not permission to relax the frozen bound.

No Mac inference performance claim is made. No product-quality budget is invented after seeing results; the 128-target PPL measurement remains a small smoke test. Source, configuration, model/input/build hashes and every failed stage are preserved locally. Only code and derived analysis are eligible for private GitHub publication.
