# Submission scope

The deliverables are the English technical-plan PDF and source scripts for a Metal matrix-vector PoC with independent sample-I/O validation. The implementation runs one projection within a Transformer layer through llama.cpp; it does not implement a complete custom Transformer block. The public Microsoft checkpoint is a compatibility experiment, separate from the team's unavailable checkpoint.

## Primary implementations

| Implementation | Source and purpose | Attribution and acceptance boundary |
| --- | --- | --- |
| Original authored PoC | `bitnet/validation/shaders/matvec.metal`; explicit TQ2_0 unpack and F32 dot, four output rows per SIMD group and two groups per threadgroup | Independently written matvec. Host and physical-phone sample-I/O evidence covers hand-checkable, seeded dense and real-input cases. Its complete phone model-graph route is not qualified. |
| Q4 specialization | `bitnet/shaders/stock4x1.metal`, with the clean-validation copy under `bitnet/validation/shaders/` | Reuses the QVAC stock helper arithmetic, specialized to four rows and one SIMD group. Its qualified real-graph route is layer-0 `Qcur-0`, single-token decode, with stock fallback elsewhere. Q4 results do not establish the original PoC's validation scope. |

Both use a 2,560-by-2,560 TQ2_0 weight matrix with independent block scales, F32 activation and F32 output. TQ2_0 stores ternary weights; this is not a binary XOR/popcount implementation. The independent CPU unpack and FP64 reference are in `bitnet/validation/reference.cpp`. The original shader and Q4 are intentionally separate from the actual stock pipeline used as a reference.

## Build and reproducibility support

`bitnet/validation/bootstrap.py` obtains the pinned QVAC source and performs a clean numerical build without reusing an old dylib. `bitnet/ios/` provides the source-only arm64 framework/app build, physical-device runner and fixed-token histories. The canonical host/shader launch contract, actual cache tests and negative controls remain part of the submission. Models, signed products, credentials and raw experimental data are excluded.

Host numerical checks run on the Mac development machine. They support build reproducibility and calculation accuracy; they are not mobile performance results. Submitted mobile performance is labeled with the physical iPhone 14 Pro / Apple A16 device, measured binary and thermal cohort. Isolated GPU command-buffer elapsed time and complete decode wall time remain separate metrics. Q4's small isolated benefit does not establish an end-to-end improvement; the current guarded integration is slower in the measured complete-decode campaign.

## Supporting research

The FFN dual-projection, activation, normalization, F8, down-control and C8 sources are optional research. The current iOS builder compiles their shaders or native helpers, and the app router references those helpers. They are retained to keep the tested build dependency chain intact, inactive by default. Their presence does not recommend enabling them or imply that they passed Q4's acceptance criteria. The English report records their qualified results and retained numerical failures separately.

The public-checkpoint ActQuant and trusted-source reference tools are also retained. Same-input quantizer agreement, operator FP64 and stock-relative graph parity are distinct from faithful full-model execution. The existing complete trusted-reference comparison remains a measured failure with strict exit code 1. This cleanup does not alter reference arithmetic, frozen tolerances or failure records.

The acceptance threshold remains `abs(output-reference) <= 2e-4 + 2e-5 * abs(reference)`. Every result must name its reference, input and coverage. Missing evidence remains unverified; processing exit 0, token agreement or a local operator pass cannot replace a required numerical or model-output gate.
