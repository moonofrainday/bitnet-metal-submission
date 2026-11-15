# Mobile BitNet Inference Technical Proposal

**Metal proof of concept and Vulkan portability plan**

Prepared for the Tether technical assessment. Draft for author review. Evidence cutoff 15 November 2026, Asia/Tokyo.

## 1 Executive proposal

We propose a Metal-first implementation of mobile BitNet inference using the existing llama.cpp graph, loader, memory planner and GPU backend. A narrowly guarded matrix-vector kernel replaces one projection inside a Transformer layer, while all unsupported calls retain the stock path. A parallel file-level plan extends and validates QVAC's existing Vulkan support on selected Adreno and Mali devices.

The practical deliverable is a reproducible source package, an independent sample-I/O harness and this technical plan. The implementation uses a public Microsoft checkpoint to establish the development path; acceptance of the team's pretrained checkpoint remains a separate dependency. The source baseline is QVAC commit `2db8b046d555316019ae75d1be90484adc8b9330`. The reviewed implementation snapshot is `bbf70e9cc70302534ff87b4f34d4ce52a9670baa`. [E1][E2]

| Evidence at the cutoff | Result and decision |
|---|---|
| Public 2B stock inference on iPhone 14 Pro | Loads and generates 32 tokens with A16 Metal and 31/31 layers offloaded. Retain as the functional baseline. |
| Q4 specialization in the real layer 0 graph | Both fixed histories pass all 32 Q checks, sampled residuals and full-logit parity. Keep opt-in. |
| Mobile performance | Q4 saves about 6-7 us per isolated Q dispatch; no whole-decode benefit was observed. Do not claim model acceleration. |
| FFN research | G retains observed stock outputs; F8 has performance evidence and preserved numerical/model-output failures. Neither repairs complete-chain FP64. |

The independently authored original PoC, the C8 research shader and the stock-derived Q4 specialization are disclosed separately. The fresh complete iPhone Q acceptance belongs to Q4. The exact original 4x2 shader has clean host numerical tests and iPhoneOS compilation, but its fresh physical-phone sample-I/O test is still required. [E3][E4]

The brief's wording combines "1-bit" and "1.58-bit". We implement ternary BitNet weights, not a binary XOR/popcount kernel. A literal binary-only requirement would need a different representation and acceptance agreement.

<!-- page -->
## 2 Scope and assignment coverage

The selected operation is `Qcur-0`, `GGML_OP_MUL_MAT`, using `blk.0.attn_q.weight` before RoPE. The model executes the surrounding layer through llama.cpp. The custom shader is one linear operation inside that layer; it does not implement an entire Transformer block in one dispatch. Current Q integration permits only layer 0 and one-token decode. [E4]

![Model execution and guarded kernel selection](assets/execution-flow.png)

| Required topic | Report location |
|---|---|
| 1 Mobile low-bit kernel design | Sections 3 and 4 |
| 2 Framework changes and portability | Sections 5 and 6 |
| 3 Profiling method and optimization targets | Sections 9, 10 and 12 |
| 4 Frameworks libraries and SDKs | Section 9 |
| 5 Kernel optimization strategies | Section 11 |
| 6 Success criteria and benchmarks | Sections 7 and 12 |
| 7 Smartphone memory power and compute limits | Sections 11 and 12 |
| 8 Ten day milestones and checkpoints | Section 13 |
| 9 Risks and contingency plans | Section 14 |
| 10 Assumptions and critical dependencies | Sections 3 and 14 |
| 11 Other constraints and submission contents | Section 15 |

Metal is the implemented target. Android execution, energy measurement and team-checkpoint qualification remain planned work. The ten-day schedule is relative to assignment kickoff, not a claim that all archived research occurred within ten days. The required handoff contains source scripts and a PDF; editable sources are included for review.

<!-- page -->
## 3 Model and weight representation

The public compatibility experiment uses Microsoft `bitnet-b1.58-2B-4T`, source revision `04c3b9ad9361b824064a1f25ea60a8be9599b127`. Its 2,412,820,480 unique parameters include a tied embedding counted once. The converted file has 332 tensors: 210 TQ2_0 matrices, one F16 embedding and 121 F32 normalization tensors. It is not the team's checkpoint. [E2][R2]

| Property | Public checkpoint contract |
|---|---|
| Transformer blocks and width | 30 blocks; hidden width 2560; FFN width 6912 |
| Attention | 20 query heads, 5 KV heads, head width 128 |
| Tokenizer and activation | Vocabulary 128256; model tokenizer/template; ReLU(gate)^2 * up |
| GGUF bytes | 1,203,563,808 bytes; about 1.121 GiB |
| SHA 256 | `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2` |

TQ2_0 represents trits {-1,0,+1} with codes {0,1,2}; code 3 is rejected by the fixture validator. A block contains 256 logical weights, 64 packed bytes and an FP16 scale. Physical storage is 66*8/256 = 2.0625 bits per quantized weight, before other tensors and metadata. The ternary alphabet's ideal information content is log2(3), approximately 1.585 bits. I2_S packing is not interchangeable with TQ2_0. [E2][R1]

For row r and block b, compute the scaled ternary dot product:

```text
y[r] = sum_b d[r,b] * sum_j (q[r,b,j] - 1) * x[256*b+j]
byte = 32*half + lane
j = 128*half + 32*plane + lane
q = (packed[byte] >> (2*plane)) & 3
```

Here half is 0..1, plane 0..3 and lane 0..31. Four fields in one byte correspond to separated logical K positions. The FP64 reference decodes that logical order independently; it evaluates the supplied quantized W and F32 X, not the original unquantized model.

The compatibility patch preserves direct AutoBitLinear scales, avoids requantizing already reconstructed ternary weights and selects explicit ReLU-squared metadata. All 210 converted matrices were audited without trit or FP16-scale mismatches. Microsoft's reference uses INT8 activation quantization; equivalence of this TQ2_0/F32 path to that W1.58A8 execution is unproven. [E2][R2]

<!-- page -->
## 4 Kernel design and sample validation

For the Q projection, W has 2560 output rows and K=2560. Ten 66-byte blocks per row give 660 bytes per row and 1,689,600 bytes total. X and Y each contain 2560 contiguous F32 values, or 10,240 bytes. The kernel never expands the complete weight matrix to F32.

The original authored PoC loads eight X values per lane/block, decodes the packed trits, applies each output row's block scale and accumulates into four independent row sums. A SIMD reduction writes one value per row. X is reused across rows. The reference implementation uses independent FP64 accumulation. [E3]

| Path | Rows per SIMD group | Groups per threadgroup | Grid x | Attribution |
|---|---:|---:|---:|---|
| Actual stock | 4 | 2 | 320 | Existing QVAC implementation |
| Original PoC | 4 | 2 | 320 | Independently authored direct unpack/dot |
| Q4 stock4x1 | 4 | 1 | 640 | Reuses complete stock helper arithmetic |
| C8 research | 8 | 2 | 160 | Separate packed-mask algorithm; model compatibility fails |

All configurations use 32-lane SIMD groups on the measured Apple devices. Actual launch dimensions and pipeline thread limits are checked. Q4's historical function name still contains `c8x2`; actual host geometry and source, rather than the name, define its layout.

### Hand computable input and output

The fixture sets zero weights to code 1 and uses changing block scales. At row 0, block 0, byte 0x24 supplies codes [0,1,2,0] at columns [3,35,67,99]. X there is [1.25,4,2,-0.5], with scale 0.5. Its contribution is 0.625. A second-block +1 weight at column 385, scale 0.25 and X=3 contributes 0.75. [E3]

| Expected output | Value and purpose |
|---|---|
| y[0] | 1.375; signed/zero trits and multiple scales |
| y[1] | 0; nonzero trit with a zero block scale |
| y[2559] | -1.25; final-row coverage and negative weight |
| All other 2557 values | 0 |

Additional cases use dense seed 42 and actual pre-RoPE inputs from decode steps 1, 16 and 32. All 2560 rows are checked, including finite values, canaries, unchanged inputs and GPU completion. There are 2558 zero outputs in total, including y[1].

<!-- page -->
## 5 Metal integration and source changes

The existing loader, graph and backend remain responsible for allocation and scheduling. The opt-in route is evaluated inside `ggml_metal_op_mul_mat_mv_encode`. It resolves the actual W/X/Y buffers and offsets on each eligible invocation. Candidate slots are W/X/Y at 0/1/2; stock uses args/W/X/Y at 0/1/2/3. These ABIs must not be mixed. Backend filenames below share the prefix `ggml/src/ggml-metal/`. [E4]

| Modified upstream file or module | Responsibility |
|---|---|
| `conversion/base.py` | Direct public-checkpoint scale reconstruction; preserve legacy handling |
| `conversion/bitnet.py` | Offline ternary conversion and activation metadata |
| `src/models/bitnet.cpp` | Select existing ReLU-squared FFN graph from explicit metadata |
| `ggml/include/ggml-metal.h` | Replay and diagnostic interfaces |
| `ggml-metal-context.m` | Opt-in command-buffer timing and completion hooks |
| `ggml-metal-device.h` and `.m` | External pipeline declaration, load/cache and capability checks |
| `ggml-metal-ops.cpp` | Graph/shape guards, live bindings, dispatch, readback and stock fallback |

The last four rows live under the GGML include or Metal backend directories. Together these are eight changed upstream files. New kernels, harnesses, iOS staging and analyzers are under `bitnet/`. QVAC already supplied TQ2_0 kernels and the SwiftUI example; those are credited as existing infrastructure.

The Q guard requires the exact MUL_MAT node and layer-0 Q weight, N=1, TQ2_0/F32/F32, audited ne/nb, no view, add=0, norm=0, compatible pipeline and launch, aligned live offsets and complete byte-range bounds. X and Y may share a Metal buffer only when their byte intervals are disjoint. W/Y overlap is rejected. Other layers, prefill, incompatible contracts and failed library/geometry selection return to stock before custom writes.

Fresh phone controls prove missing-library and unsupported-geometry fallback. Normal-run strides are checked; a malformed real graph stride was not injected. The external pipeline cache uses a function-name key, so same-symbol library changes and missing-library tests use separate processes.

An unconditional residual-ADD binding correction selects the other operand by tensor identity. Thus "stock default" means stock shader/dispatch selection; it does not mean the CPU backend is byte-identical to pristine QVAC. The original compatibility patch and additive readback patch reproduce all eight changed upstream files exactly. [E1][E4]

<!-- page -->
## 6 Vulkan extension for Adreno and Mali

The pinned QVAC backend already contains TQ2_0 Vulkan matvec support. The proposed Android work audits and specializes that implementation, rather than starting a new runtime. Metal results do not establish Android correctness or speed. [R1][R6]

| Existing file under `ggml/src/ggml-vulkan/` | Proposed extension or validation |
|---|---|
| `ggml-vulkan.cpp` | TQ2_0 pipeline selection, device/shape variants and fallback |
| `vulkan-shaders/mul_mat_vec_tq2_0.comp` | Same-input ternary matvec and row/workgroup variants |
| `vulkan-shaders/types.glsl` | Confirm packed 256-value block and FP16 scale ABI |
| `vulkan-shaders/vulkan-shaders-gen.cpp` | Generate and register only required specializations |
| `ggml-vulkan-push-constants.h` | Preserve dimensions, strides and offsets in the native ABI |
| `ggml-vulkan-buffers.cpp` and `CMakeLists.txt` | Allocation/staging limits and reproducible NDK/SPIR-V build |

The first Android milestone is an unchanged stock Vulkan build, target feature inventory and public-model functional test. Then port the independent logical-K reference and the same sample cases. Only after stock passes should a shape-specific variant be enabled behind an explicit runtime guard. Add targeted shapes to `tests/test-backend-ops.cpp`; use CPU fallback if the device cannot implement the required operation safely.

Query compute queue/timestamp support, subgroup size and operations, workgroup limits, shared memory, descriptor/offset alignment, memory heaps and narrow-storage/FP16 features. Use subgroup reductions only when supported; otherwise use a conservative workgroup/shared-memory reduction with legal barriers. Do not assume a 32-lane subgroup on every Adreno or Mali. Integer dot instructions, subgroup-size control and cooperative matrices are optional device-specific variants. [R6]

For decode, prioritize packed streaming and activation reuse; for prefill, use matrix tiling and reuse weights across tokens. The current Q PoC is decode-only, so prefill stays stock. Build separate specializations instead of forcing GEMV geometry onto GEMM.

On Android, host-visible memory is not automatically cache-coherent. Validate staging, flush/invalidate rules, descriptor bounds and synchronization. Start with F32 activation/accumulation; treat reduced precision or activation quantization as a separate numerical change. Validation requires one physical Adreno and one physical Mali configuration before making cross-vendor claims. Devices and driver access are critical dependencies, not completed deliverables.

<!-- page -->
## 7 Correctness and acceptance protocol

Validation is hierarchical. The frozen experimental numerical rule remains unchanged:

```text
abs(actual - reference) <= 2e-4 + 2e-5 * abs(reference)
```

This is a project criterion, not a numerical tolerance specified in the brief. Each stage must declare its reference and required scope. A lower FP64 dot error does not establish stock-relative model compatibility. [E3][E5]

| Gate | Reference and required evidence |
|---|---|
| Storage and execution | Correct packing/scales, hashes, all writes, finite values, canaries, unchanged inputs and completed GPU work |
| parity_pass | Same captured W/X through named routes; bitwise differences and tolerance differences both retained |
| fp64_pass | Independently unpacked FP64 dot, or explicitly declared complete FFN chain |
| model_output_pass | Stock-relative residuals/full logits for the declared fixed replay; no inferred language-quality pass |
| Overall result | Every required gate for that mode must pass; missing checks stay null |

Numerical verifiers and CPU analyzers use strict exits 0 for pass, 1 for a completed measured failure and 2 for missing/invalid evidence. The device orchestration runner can also return 1 for setup/transport errors: inspect execution_pass, native completion, actual app exit and the saved summary before classifying that failure. Processing-only exit 0 cannot overwrite a strict numerical failure. Positive, nonfinite, missing-output and invalid-hash controls check the verifier semantics.

Clean builds from the pinned source, without old dylibs, tested actual stock, original PoC and Q4 on the hand case, dense seed 42 and real steps 1/16/32: 38,400 values passed, with no failing/nonfinite rows. Q4 equaled stock on all 12,800 compared values; original PoC differed in 10,122 F32 values but stayed within FP64 tolerance. TQ2_0 backend regression passed 59/59, with zero selected failures/unsupported cases. These are host numerical results, not Mac performance evidence. [E3]

Real-graph validation checks all 32 eligible Q calls, not only three samples. W is read once per route, X/Y at every step and residual output at steps 1/16/32. The independent analyzer rejects mismatched W/X hashes before comparing outputs. Probe-off and within-process repeats must preserve all 33 complete logits vectors. Timing uses a separate uninstrumented run.

For FFN, distinguish FP64 down on each route's actual input from a full FP64 gate/up -> ReLU-squared -> weighted RMSNorm -> down chain. Existing full-chain failures in stock and candidates remain failures; fusion parity cannot erase them.

<!-- page -->
## 8 Physical device functional and numerical results

Historical smoke runs used iPhone 14 Pro/A16, iOS 27.0.1, Xcode 27.2 and iPhoneOS SDK 27.2. Loading, generation, offload and KV placement are separate outcomes. [E2]

| Model and framework | Load | Generation | Metal placement | KV placement |
|---|---|---|---|---|
| Patched 1.27B regression | Pass | 32 tokens; exit 0 | Apple A16; 25/25 offload | MTL0; 96.00 MiB |
| Public 2B stock baseline | Pass | 32 tokens; exit 0 | Apple A16; 31/31 offload | MTL0; 37.50 MiB |

The transferred GGUF hash was checked on the phone. Public 2B text begins "A GPU, or Graphics Processing Unit, is a specialized hardware component..."; exact output is preserved locally. This smoke did not save phone token IDs, so text agreement is not token-ID parity. Its older thread/batch defaults are not retroactively replaced by the later replay configuration.

### Fresh Q4 layer zero acceptance

The 15 November campaign used iOS 27.2.1, Xcode 27.2, SDK 27.2 and a freshly rebuilt framework. Both histories use 27-token stock prefill, 32 fixed continuation IDs, context 512, batch/ubatch 128, six CPU threads, F16 K/V and flash attention disabled. Only Qcur-0 is replaced. [E5]

| History | Custom encode and completion | S plus Q4 values | Failed or nonfinite | Max FP64 absolute error |
|---|---|---:|---|---:|
| Primary | 32 / 32 | 163,840 | 0 / 0 | 1.576802e-5 |
| Independent | 32 / 32 | 163,840 | 0 / 0 | 1.324259e-5 |

All Q outputs are stock-bit-identical. Sampled residuals, every full-logit vector and probe-off controls also match bitwise. The declared Q-only parity/FP64/model-output gates pass with strict exit 0. This establishes a projection inside a running layer, not full-Transformer FP64 or model quality.

Mac stock CPU and Metal generation both succeeded, but strict token parity still fails at generated token 18: CPU 83699 (`computations`), Metal 14515 (`graphics`), strict exit 1. Keep that separate from custom routing. The exact original 4x2 shader's fresh phone sample-I/O remains unverified; the accepted Q4 result must not be reassigned to it. [E2][E3]

<!-- page -->
## 9 Profiling method and development stack

Use three distinct timing scopes: CPU selection/encoding, completed GPU command-buffer elapsed time, and synchronized application wall time. For an isolated operator, one command buffer contains one target dispatch; read `gpuEndTime-gpuStartTime` only after completion and convert seconds to microseconds. This is not pure shader cycles. [R4]

Each formal operator comparison uses identical resident W/X, 30 warmups per route, 201 alternating AB/BA pairs per round and three rounds. Save p50, p95, paired candidate-minus-stock median, wins, order groups and chronological blocks. A/A is calibration and is never subtracted. Separate nominal endpoints from sustained-load cohorts. Disable capture, readback and diagnostics during timing.

| Stack | Role and status |
|---|---|
| QVAC llama.cpp and GGML | Pinned graph, loader, CPU/Metal/Vulkan backends; implemented |
| Git CMake Clang and Python 3.9 or newer | Reproducible builds and analyzers; NumPy for full Q analysis |
| Isolated conversion environment | Python 3.11, PyTorch, Transformers, tokenizers, safetensors, NumPy and pinned in-tree gguf-py; versions in BUILD_AND_RUN.md |
| Full Xcode and Metal compiler | Metal/AIR/metallib and iPhoneOS arm64 framework; measured Xcode 27.2 |
| Apple SDK frameworks and devicectl | Metal, Foundation, Accelerate, UIKit and SwiftUI; signing, deployment and fixed-token replay |
| Xcode GPU Capture and Instruments | Pipeline/launch/dependency verification, Metal System Trace and CPU/memory profiling |
| Android SDK NDK CMake and adb | Proposed Android arm64 packaging and deployment |
| Vulkan loader headers shaderc or glslang and SPIRV Tools | Proposed shader build, validation and feature-specific pipelines |
| APA AGI and Perfetto | Proposed supported-device Android system/frame profiling; APA is current recommended system option |
| Qualcomm Snapdragon Profiler and Arm Performance Studio | Proposed vendor-specific counters; Mali Offline Compiler for static estimates |

Historical conversion used torch 2.11.0, transformers 5.10.0, NumPy 2.2.6, tokenizers 0.22.2 and safetensors 0.8.0. The transformers version was yanked; preserve it only in an isolated reproduction environment and validate replacements separately. Synthetic kernel tests do not require those packages.

Xcode counters can serialize passes, so profile/capture runs are separate from formal timing. AIR confirms intermediate-code changes, not hardware stalls, bandwidth or occupancy. Current project runtime counters remain unavailable. [R5]

Android tooling requires compatible OS/driver permissions and a debuggable application; APA is currently public beta. Validate exact tool versions on the selected device. Use AGI/Perfetto or vendor tools when supported. Deployment instructions and tool availability are not measured Android results. [R7][R9][R10]

<!-- page -->
## 10 Measured iPhone performance

Fresh Q4 operator tests use the accepted primary step-16 fixture. Values below are microseconds of single-dispatch GPU command-buffer elapsed time. All samples are retained; each thermal cohort has three direct stock/Q4 rounds of 201 pairs. [E5]

| Cohort and round | Stock p50 / p95 us | Q4 p50 / p95 us | Paired Q4 minus S us | Q4 wins |
|---|---|---|---:|---:|
| Nominal 1 | 224.542 / 228.500 | 218.500 / 222.000 | -6.250 | 198/201 |
| Nominal 2 | 224.542 / 228.333 | 218.375 / 222.625 | -6.042 | 195/201 |
| Nominal 3 | 224.125 / 228.500 | 218.125 / 221.000 | -6.000 | 198/201 |
| Serious 1 | 226.000 / 230.000 | 219.375 / 223.875 | -6.208 | 194/201 |
| Serious 2 | 226.125 / 229.875 | 219.167 / 223.042 | -7.000 | 198/201 |
| Serious 3 | 224.875 / 229.125 | 218.542 / 222.750 | -6.083 | 197/201 |

Median paired relative reductions are 2.681%-3.112%. All order and chronological-quartile direct medians favor Q4. A/A paired medians span -0.333 to +0.292 us. Thermal observations are endpoints, not continuous clocks/temperature; nominal does not prove a cold or fixed-frequency device. This compares complete operator paths, not pure threadgroup size alone.

The whole-model scope times 32 fixed decode tokens after KV clearing and restoration of the same prefix outside the timer. CPU encoding, GPU execution and completion are included. Both campaigns use only the primary history, A/A plus stock/Q4, three rounds and 21 pairs per contrast/round: 126 pairs and 8,064 token records per campaign.

| Campaign and round | S p50 ms | Q4 p50 ms | Paired Q4 minus S ms |
|---|---:|---:|---:|
| First 1 / 2 / 3 | 1523.294 / 1519.479 / 1672.692 | 1523.119 / 1528.410 / 1683.458 | +1.510 / +8.954 / +9.811 |
| Sustained 1 / 2 / 3 | 1838.651 / 2000.696 / 2418.593 | 1847.512 / 2018.455 / 2421.630 | +7.261 / +8.099 / +0.339 |

No whole-decode gain was observed. The last sustained differences overlap order/time/A-A variation, so a universal fixed penalty is not established. Full per-round p95 and order statistics are in the derived tables. Do not multiply isolated savings into an assumed model speedup.

Historical G/F8 phone timing favors F8 over G by 1.961%-3.568% in paired relative medians, but F8 retains strict numerical/model-output failure. That campaign has no formal direct stock contrast and changes arithmetic as well as fusion; it is not a stock speedup or pure fusion-cost result. [E6]

<!-- page -->
## 11 Optimization strategy and mobile constraints

Optimize one mechanism at a time, with a falsifiable hypothesis and matched computation scope. Preserve complete stock row arithmetic where model compatibility is required. The selected Q4 is a conservative specialization; no performance claim is transferred from M2 Pro to A16. [E4][E5]

| Mechanism | Expected benefit and required control |
|---|---|
| Packed streaming and activation reuse | Reduce repeated X work without full weight expansion; compare identical W/X and inspect actual generated loads |
| Threadgroup and row mapping | Balance parallel groups, reuse and live values; verify every row, pipeline limits and order effects |
| Coefficients and reduction topology | Reduce unpack/dot work; check FP64 and stock error separately because FP32 ordering changes propagation |
| Legal local fusion | Remove dispatch/intermediate traffic only when graph consumers, aliases and encoding boundaries allow it |
| Prefill tiling | Reuse weights over token columns; a separate GEMM route, not a decode-layout extrapolation |
| CPU planning and scheduling | Compare full guards followed by the same stock PSO before attributing model losses to shaders |

G combines two stock-arithmetic FFN projections and leaves compatible postprocessing separate. F8 combines C8 gate/up with ReLU(gate)^2*up. K additionally researches activation/normalization and encounters graph/alias limits. A fusion must preserve shared X, per-layer W/scales, consumer relationships and legal dependencies. A threadgroup cannot read another group's unfinished result without a valid cross-dispatch synchronization design. Weighted RMSNorm and down are included in full-FFN comparisons.

A16 has limited shared resources and a sustained thermal envelope. Keep packed weights resident; use bounded scratch buffers, asynchronous initialization and explicit lifetime management. A complete F32 Q matrix would be 26,214,400 bytes instead of 1,689,600 packed bytes. This storage comparison is not a measured bandwidth speedup. The 1.121 GiB GGUF is not the process memory footprint: embeddings, KV, scratch, executable and driver allocations also matter.

Tune context and batch to a measured memory budget. Prefer contiguous packed reads and reuse within a legal threadgroup over extra per-token copies. Additional lookup tables, reorders or coefficient preparation must be counted in the candidate's cost. Larger local arrays do not by themselves prove a spill; AIR does not provide measured occupancy.

Low-power operation should reduce memory traffic, unnecessary CPU polling and thermal overcommit while preserving correctness. Use thermal notifications/adaptation on iOS and supported ADPF hints/headroom on Android; neither guarantees clocks or energy savings. Energy must be measured separately. [R8]

<!-- page -->
## 12 Success criteria and evaluation benchmarks

The numerical threshold is frozen; performance targets below are proposed engineering goals, not requirements supplied by the brief or achieved claims. The stock baseline remains a valid functional fallback while optimization targets are pursued.

| Area | Proposed acceptance or target | Status at cutoff |
|---|---|---|
| Functional inference | Load and generate with verified mobile GPU offload; no unexpected fallback/crash | Public 2B A16 pass; team and Android pending |
| Operator correctness | All required outputs finite, covered and within frozen FP64 bound; stock differences reported | Original host samples and fresh phone Q4 pass |
| Layer compatibility | Required Q/residual/logit gates pass on two histories; preserve strict failures | Q4 declared layer-0 scope pass; broader quality unmeasured |
| Operator optimization | Seek at least 5% median paired reduction versus actual stock in 3 rounds with order checks | Q4's measured 2.68%-3.11% is below this future target |
| Model promotion | Direct steady decode improvement distinguishable from A/A and drift; preserve p95 | Q4 whole-decode benefit not observed |
| Memory and sustained operation | No OOM; measured peak footprint/scratch; at least 10-minute controlled thermal study | Peak/controlled sustained protocol pending |
| Power | Supported energy signal and matched work; seek at least 5% lower J/token without a correctness/latency loss | Unmeasured |

Use a benchmark matrix rather than one advertised tok/s number: hand-known and dense inputs; real steps 1/16/32; complete 32-call replay; two distinct histories; decode and prefill; nominal and sustained cohorts. Planned prefill lengths are 32, 128 and 256 tokens, with 32-token decode under context 512. Add other contexts only after validating memory and positional limits. Preserve exact token IDs, chat template, threads, batch/ubatch, KV dtype and flash-attention configuration.

Report loading separately from time to first token. Measure prefill latency/tok/s, steady decode wall latency/tok/s, per-token p50/p95, peak process/GPU allocations and observed offload. Isolated GPU timings, full FFN intervals, complete Transformer layers and application decode are distinct units. An instrumented layer slice may change scheduling; mark that diagnostic scope.

Power tests require a supported platform signal or external instrumentation synchronized to the same workload and a stated measurement boundary. Separate device energy from estimated GPU-only energy, control charging/display/thermal conditions and include idle/control uncertainty. Battery percentage or Instruments energy impact alone is not joules/token. No model-quality pass is inferred from arithmetic accuracy; a separate held-out evaluation requires agreement with the team.

<!-- page -->
## 13 Ten day phased schedule

This plan is relative to kickoff. The existing research provides reusable evidence, but it is not presented as a reconstructed ten-day work log. Metal is the required executed PoC route; Android device work is conditional on hardware and driver access.

| Day | Milestone | Reviewable deliverable and checkpoint |
|---:|---|---|
| 1 | Freeze model and baseline | Source/model hashes, tokenizer/template, scale and activation contract, target capability inventory |
| 2 | Establish stock inference | Clean backend build; CPU and physical-phone load/generation/offload results recorded separately |
| 3 | Establish the oracle | Hand-computable and seeded dense samples, logical-K FP64 reference, strict-exit controls |
| 4 | Implement the PoC | Original Metal matvec, complete output coverage, bounds/canaries and sample-I/O result |
| 5 | Integrate one real node | Opt-in Qcur-0, live buffer bindings, complete fixed replay and stock fallback |
| 6 | Profile and choose a mechanism | Capture/AIR/counter evidence as available; one hypothesis and controlled short A/B |
| 7 | Validate the phone candidate | Exact shader sample-I/O plus two-history all-call graph checks; no performance promotion on numerical failure |
| 8 | Evaluate performance and constraints | Direct A/A and stock/candidate pairs, prefill/decode, thermal and memory report |
| 9 | Close risks and portability design | Adreno/Mali feature/file matrix; execute a port only if devices are available; retain unresolved evidence |
| 10 | Freeze and defend | Clean source package, reviewed PDF, reproducibility entry, limitations and interview walkthrough |

A daily checkpoint records source/model/input hashes, actual exit codes, numerical flags, performance scope, the current blocker and the next discriminating experiment. Status is measured pass, measured fail or unverified. Keep raw evidence immutable and version derived summaries.

If a candidate fails, retain the failure and continue independent work such as reference checks, build portability or documentation. Do not spend the remaining schedule generating similar variants without a distinguishing hypothesis. Signing or physical-device access problems should produce a minimal reproducer while CPU/source tasks continue.

The current most valuable next performance experiment is stock A versus full Q4 guard followed by the same stock pipeline B, then Q4 C. Confirm B's actual guard hits and bitwise output before timing. B-A estimates guard/encoding effects; C-B still includes launch and execution differences. It is a proposed attribution experiment, not a measured conclusion.

<!-- page -->
## 14 Risks assumptions and contingencies

| Risk or dependency | Observable consequence | Mitigation and contingency |
|---|---|---|
| Team checkpoint differs from public model | Wrong scale, activation, tokenizer or graph | Freeze metadata/tensor contract first; audit conversion; keep public and team acceptance separate |
| Literal binary interpretation of the brief | Ternary implementation judged out of scope | Explicitly agree that 1.58-bit means ternary BitNet; design a binary specialization only if required |
| FP32 arithmetic propagation | Local dots pass while layer/logits fail | Independent FP64 plus stock-relative gates; retain stock arithmetic and all failed outcomes |
| Full FFN FP64 failure | Fusion parity mistaken for numerical repair | Compare actual-input down and complete chain separately; keep strict exit 1 |
| Fusion alias or dependency mismatch | Data hazard or corruption | Consumer/edge/buffer/encoding guards; decline before custom writes; stock fallback |
| Thermal drift and profiling perturbation | False speedup or incorrect bottleneck claim | Balanced pairs, A/A, time/order groups, separate capture and formal timing |
| Toolchain signing or profile expiration | Build/install blocked | Fresh manifests, user-managed credentials, trust/Developer Mode steps; host work continues |
| Unsupported Android features or missing phones | No defensible cross-platform result | Query features, use portable F32/reduction fallback; mark hardware-specific results unverified |

We assume legitimate access to the team's weights, tokenizer and inference specification, a working Apple signing setup, available physical targets and permission to use the relevant profiling tools. The public-model compatibility patch is not approval for every BitNet checkpoint. We also assume no unstated requirement to reproduce Microsoft's complete INT8 activation execution bit for bit.

The current complete-FFN failure is concrete. A new correction-placement Down control on 96 selected outputs/path retained the same 12 failures in stock, stock-wrapper and candidate. The candidate changed 48 rows and increased worst error/tolerance from 1.566268 to 1.829765. Native and independent strict exits remain 1; no full-matrix expansion or repair is claimed. [E5]

All safety and numerical gates remain active. Future application-quality criteria require their own reference and rationale; they do not overwrite these frozen experimental failures. The final team-checkpoint, Android, energy and exact original-PoC phone gaps must be resolved or explicitly accepted as delivery limitations.

<!-- page -->
## 15 Source delivery and reproducibility

The private source repository contains the pinned QVAC source, compatibility changes, original PoC, stock-derived Q4, guarded integration, iOS runners, independent references and derived analysis. It includes no model weights, raw tensors/logits, raw timing samples, signed app/framework or credentials. Raw evidence stays on the local Mac. [E1]

| Source location | Review purpose |
|---|---|
| `bitnet/validation/` | Original matvec, Q4 specialization, known/dense/real fixtures and independent FP64 |
| `bitnet/shaders/` | Mobile candidate shaders, including stock-derived and research variants |
| `bitnet/ios/` | Fresh arm64 framework/app build and fixed-token/device runner |
| `bitnet/analysis-tools/` | Complete-Q, paired-timing and selected-Down CPU analyzers |
| `bitnet/patches/` | Canonical compatibility/backend patch and additive default-off readback patch |
| `bitnet/analysis/2026-11-15-iphone-q4/` | Curated round summaries and digest-only local evidence index |

The published source import omits 63 upstream CI workflow files because of the available GitHub credential scope; runtime/local build/test source is retained, and exact exclusions are documented. Both patches are already applied here. A separate pristine upstream checkout is required for the clean standalone baseline comparison.

### Minimal numerical rebuild

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
python3 bitnet/validation/bootstrap.py \
  --qvac-repo /path/to/existing-qvac-repository \
  --work-root /path/to/fresh-clean-validation
```

This model-free entry builds the pristine pinned backend and runs the two synthetic cases plus TQ2_0 regression. Requested real cases require separate local captures and cannot silently downgrade to synthetic-only testing. It is a numerical test, not a Mac performance benchmark.

### Physical phone reproduction

```sh
python3 bitnet/ios/build.py --team "$APPLE_TEAM_ID" \
  --bundle-id "$BUNDLE_ID" --allow-provisioning-updates \
  --output "$ARTIFACTS/build"
python3 bitnet/ios/run.py --build-dir "$ARTIFACTS/build" \
  --device "$IPHONE_DEVICE_ID" --model "$BITNET_2B_GGUF" \
  --mode graph-verify --routes S,Q4 --layers 0 --install \
  --history-dir bitnet/ios/histories/primary --q-readback \
  --output "$ARTIFACTS/primary-on"
```

All output directories must be new; set absolute artifact/model paths locally. Run the matched probe-off and independent-history controls, then the CPU analyzer before formal timing. The complete recipe, exact scope and expected Down strict failure are in `bitnet/docs/IPHONE_Q4_COMPLETE_2026-11-15.md`. The original shader's phone test is a remaining explicit qualification, not supplied by this Q4 command.

<!-- page -->
## 16 Evidence index and references

Paths below are relative to the workspace or source package as stated. IDs distinguish local measured evidence from public design sources. The evidence cutoff is 15 November 2026; mutable vendor documentation was reviewed at that cutoff.

### Local evidence

- [E1] Source snapshot bbf70e9; `bitnet/provenance/BASELINE.json`, `SOURCE_MANIFEST.json`, `bitnet/docs/CHANGES_FROM_QVAC.md`. [Private source repository](https://github.com/moonofrainday/bitnet-metal-submission).
- [E2] Workspace `outputs/experiments/2026-11-07_133437-public-2b-compat/` for conversion/model identity and CPU/Metal strict comparison; `2026-11-07_152028-iphone-public-2b/` for patched 1.27B and public 2B phone smoke.
- [E3] Package `bitnet/validation/analysis/` and workspace `outputs/experiments/2026-11-15_170005+0900-clean-poc-stock4x1/`; clean source/build/known/dense/real operator evidence.
- [E4] Package `bitnet/docs/SOURCE_WALKTHROUGH.md`; original `bitnet/validation/shaders/matvec.metal`; Q4 `bitnet/shaders/stock4x1.metal`; backend `ggml/src/ggml-metal/ggml-metal-ops.cpp` at the reviewed snapshot.
- [E5] Workspace `outputs/experiments/2026-11-15_182149+0900-iphone-q4-complete/`; package `bitnet/docs/IPHONE_Q4_COMPLETE_2026-11-15.md` and `bitnet/analysis/2026-11-15-iphone-q4/`. Includes invalid preliminary captures, strict passes, timing and the Down negative control.
- [E6] Package `bitnet/docs/IPHONE_RESULTS.md` and `VALIDATION.md`; historical G/F8 two-history direct comparison and preserved failures. These were not rerun in the newest binary.

### Public primary sources

- [R1] [QVAC pinned Metal source](https://github.com/tetherto/qvac-fabric-llm.cpp/blob/2db8b046d555316019ae75d1be90484adc8b9330/ggml/src/ggml-metal/kernels/mul_mv.metal). Existing low-bit arithmetic and attribution baseline.
- [R2] [Microsoft BitNet b1.58 2B4T technical report](https://arxiv.org/html/2504.12285v2). Architecture, ternary weights and INT8 activation reference.
- [R3] [Microsoft W2A8 GPU GEMV](https://github.com/microsoft/BitNet/blob/main/gpu/README.md). Packed low-bit execution and separate operator/generation reporting; CUDA results are not A16 measurements.
- [R4] [Apple GPU start time](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime). Completed GPU command-buffer timing semantics.
- [R5] [Apple GPU counter statistics](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics). Runtime profiling and collection limitations.
- [R6] [Khronos Vulkan subgroups](https://docs.vulkan.org/guide/latest/subgroups.html). Capability queries and optional subgroup controls.
- [R7] [Android Performance Analyzer](https://developer.android.com/android-performance-analyzer) and [AGI quickstart](https://developer.android.com/agi/start). Current system profiling and supported-device alternatives.
- [R8] [Android Dynamic Performance Framework](https://developer.android.com/games/optimize/adpf). Thermal/performance information and hints.
- [R9] [Arm Performance Studio](https://developer.arm.com/tools-and-software/arm-performance-studio). Mali runtime tooling and static compiler analysis.
- [R10] [Qualcomm Snapdragon Profiler](https://www.qualcomm.com/developer/software/snapdragon-profiler). Target-dependent Adreno profiling.

Useful implementation precedents include [MLX Metal quantized kernels](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.metal), [BitBLAS BitNet integration](https://github.com/microsoft/BitBLAS/blob/main/integration/BitNet/README.md), and [Marlin](https://github.com/IST-DASLab/marlin). Their specialization, replacement tests and shape-sensitive reporting inform the plan; their hardware results are not mobile measurements here.
