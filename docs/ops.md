# GGML Operations

List of GGML operations and backend support status.

## Experimental BitNet extension

This fork adds `ACT_QUANT` (`ggml_act_quant`) for finite, contiguous F32 inputs
with one token per `ne[0]` row and a separate F32 output. CPU and Metal implement
per-token absmax INT8 fake quantization with ties-to-even rounding. The scale
evaluation follows the pinned Torch eager reference profile: F32 `1 / max(absmax,
eps)` followed by F32 multiplication by `127`; output is the rounded/clamped
integer code divided by that scale. Other backends have no implementation.
The targeted `test-backend-ops -o ACT_QUANT` cases check cross-backend shapes,
tail rows, contiguous views, zero rows and half ties; the independent
source-derived Microsoft fixtures in `bitnet/actquant` provide the separate
numerical oracle. Neither test establishes full-model semantic or quality parity.
The generated historical support matrix below has not been regenerated for this
experimental operation; no support CSV measurements are inferred from this note.

## How to add a backend to this table:

1. Run `test-backend-ops support --output csv` with your backend name and redirect output to a csv file in `docs/ops/` (e.g., `docs/ops/CUDA.csv`)
2. Regenerate `/docs/ops.md` via `./scripts/create_ops_docs.py`

Legend:
- ✅ Fully supported by this backend
- 🟡 Partially supported by this backend
- ❌ Not supported by this backend

| Operation | BLAS | CANN | CPU | CUDA | ET | HTP | MTL | OpenCL | SYCL | Vulkan | WebGPU | ZenDNN | zDNN |
|-----------|------|------|------|------|------|------|------|------|------|------|------|------|------|
|                              ABS | ❌ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              ACC | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              ADD | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             ADD1 | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                           ADD_ID | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                     AFFINE_PRELU | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                           ARANGE | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                           ARGMAX | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          ARGSORT | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        BIAS_GELU | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                             CEIL | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                  CHANNEL_SHUFFLE | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                            CLAMP | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        COL2IM_1D | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                           CONCAT | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                             CONT | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                       CONV_1D_DW | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
|                CONV_1D_DW_DIRECT | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                          CONV_2D | ❌ | ❌ | 🟡 | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ❌ | ❌ |
|                       CONV_2D_DW | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          CONV_3D | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                CONV_TRANSPOSE_1D | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                CONV_TRANSPOSE_2D | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              COS | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                      COUNT_EQUAL | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              CPY | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|               CROSS_ENTROPY_LOSS | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|          CROSS_ENTROPY_LOSS_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                           CUMSUM | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             DIAG | ❌ | ❌ | ✅ | ✅ | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                    DIAG_MASK_INF | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              DIV | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                     DSV4_HC_COMB | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                     DSV4_HC_POST | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                      DSV4_HC_PRE | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                              DUP | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              ELU | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              EXP | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            EXPM1 | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             FILL | ❌ | ❌ | ✅ | ✅ | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                   FLASH_ATTN_EXT | ❌ | 🟡 | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|                            FLOOR | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                  GATED_DELTA_NET | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | 🟡 | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                GATED_LINEAR_ATTN | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            GEGLU | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        GEGLU_ERF | ❌ | ✅ | ✅ | ✅ | 🟡 | ❌ | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                      GEGLU_QUICK | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             GELU | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         GELU_ERF | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                       GELU_QUICK | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         GET_ROWS | ❌ | 🟡 | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ❌ | ❌ |
|                    GET_ROWS_BACK | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ |
|                       GROUP_NORM | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                              GRU | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                      HARDSIGMOID | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        HARDSWISH | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                           IM2COL | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        IM2COL_3D | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                          L2_NORM | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                       LEAKY_RELU | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | 🟡 | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                LIGHTNING_INDEXER | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                              LOG | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        LSTM_CELL | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                             MEAN | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 🟡 | ✅ | ❌ | ❌ | ❌ |
|                              MUL | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          MUL_MAT | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
|                 MUL_MAT_HADAMARD | ❌ | ❌ | ✅ | ❌ | ✅ | 🟡 | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 |
|                       MUL_MAT_ID | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 🟡 | ❌ |
|                  MUL_MAT_ID_W4A4 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                  MUL_MAT_ID_W4A8 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                     MUL_MAT_W4A4 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                     MUL_MAT_W4A8 | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
|                              NEG | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             NORM | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                     NORM_CHANNEL | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                   OPT_STEP_ADAMW | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                     OPT_STEP_SGD | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                         OUT_PROD | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ |
|                              PAD | ❌ | 🟡 | ✅ | 🟡 | ❌ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                   PAD_REFLECT_1D | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                 PAD_REPLICATE_1D | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                          POOL_1D | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                          POOL_2D | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            REGLU | ❌ | ✅ | ✅ | ✅ | 🟡 | ❌ | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             RELU | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                           REPEAT | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ |
|                      REPEAT_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                         RMS_NORM | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                    RMS_NORM_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                             ROLL | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                             ROPE | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        ROPE_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            ROUND | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        RWKV_WKV6 | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                        RWKV_WKV7 | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            SCALE | ❌ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                   SCALE_RESIDUAL | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                              SET | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                         SET_ROWS | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|                              SGN | ❌ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            SIGLU | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                          SIGMOID | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             SILU | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                        SILU_BACK | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                              SIN | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            SNAKE | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
|                         SOFTPLUS | ❌ | ❌ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         SOFT_MAX | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                    SOFT_MAX_BACK | ❌ | ❌ | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | ✅ | ❌ | ❌ | ❌ |
|                        SOLVE_TRI | ❌ | ❌ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                              SQR | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             SQRT | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         SSM_CONV | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         SSM_SCAN | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | 🟡 | 🟡 | ✅ | ❌ | ❌ |
|                             STEP | ❌ | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              SUB | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                              SUM | ❌ | 🟡 | ✅ | 🟡 | ❌ | 🟡 | 🟡 | ❌ | 🟡 | 🟡 | 🟡 | ❌ | ❌ |
|                         SUM_ROWS | ❌ | ✅ | ✅ | 🟡 | ❌ | 🟡 | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ |
|                           SWIGLU | ❌ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                     SWIGLU_CLAMP | ❌ | ❌ | ✅ | ❌ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
|                       SWIGLU_OAI | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                             TANH | ❌ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                         TDT_STEP | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
|               TIMESTEP_EMBEDDING | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
|                            TOP_K | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | 🟡 | ✅ | ❌ | ❌ |
|                              TRI | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            TRUNC | ❌ | ❌ | ✅ | 🟡 | 🟡 | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                          UPSCALE | ❌ | 🟡 | ✅ | ✅ | ❌ | ❌ | ✅ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ |
|                            XIELU | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
|                    ZERO_UPSAMPLE | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | 🟡 | ❌ | ❌ | ❌ |
