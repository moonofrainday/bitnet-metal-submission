#include <metal_stdlib>
#include "ggml-metal-bitnet-q-config.h"
using namespace metal;
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)
#define N_SIMDWIDTH GGML_BITNET_Q_SIMD_WIDTH
#define QK_K GGML_BITNET_Q_BLOCK_ELEMENTS
#if defined(POC_ROWS_PER_SIMD) && POC_ROWS_PER_SIMD != GGML_BITNET_Q4_ROWS
#error "Q4 rows disagree with the canonical contract"
#endif
#if defined(POC_SIMD_PER_GROUP) && POC_SIMD_PER_GROUP != GGML_BITNET_Q4_NSG
#error "Q4 SIMD groups disagree with the canonical contract"
#endif
#ifdef POC_VARIANT
#error "Q4 uses stock helper arithmetic, not a POC_VARIANT"
#endif
static_assert(GGML_BITNET_Q_CONFIG_VERSION == 1, "Q contract version");
static_assert(GGML_BITNET_Q_NROWS % (GGML_BITNET_Q4_ROWS * GGML_BITNET_Q4_NSG) == 0, "Q4 row coverage");
constant short FC_mul_mv_nsg = GGML_BITNET_Q4_NSG;
constant short FC_mul_mv_ne12 = 1;
constant short FC_mul_mv_r2 = 1;
constant short FC_mul_mv_r3 = 1;
struct block_tq2_0 { uchar qs[64]; half d; };
static_assert(sizeof(block_tq2_0) == GGML_BITNET_Q_BLOCK_BYTES, "TQ2_0 block");
struct ggml_metal_kargs_mul_mv {
    int ne00, ne01, ne02;
    ulong nb00, nb01, nb02, nb03;
    int ne10, ne11, ne12;
    ulong nb10, nb11, nb12, nb13;
    int ne0, ne1, nr0;
    short r2, r3;
    float eps;
};

template<int nr0, typename args_t>
void kernel_mul_mv_tq2_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_tq2_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_tq2_0 *) ((device char *) src0 + offset0);
    }

    float sumf[nr0] = {0.f};

    // 8 threads per block, NBLOCK blocks per pass, 2 halves per block per pass
    constexpr short NBLOCK = 4;

    constexpr short NB = N_SIMDWIDTH/NBLOCK; // threads per block

    const short blk = tiisg / NB;    // 0..NBLOCK-1, block handled by this thread
    const short htg = tiisg % NB;    // 0..NB-1, thread within block (0..7)

    // byte and y base offsets within the block (32 elements per thread, 4 per byte)
    device const float4 * yb4 = (device const float4 *)(y + 4*htg + blk*QK_K);

    // hoisted per-byte coefficients (from y) and total y-sum, shared across rows
    // ref: https://github.com/ggml-org/llama.cpp/pull/26980
    float4 coef[4];

    for (int ib = blk; ib < nb; ib += NBLOCK) {
        FOR_UNROLL (short h0 = 0; h0 < 2; ++h0) {
            const float4 y0 = yb4[ 0 + 32*h0];
            const float4 y1 = yb4[ 8 + 32*h0];
            const float4 y2 = yb4[16 + 32*h0];
            const float4 y3 = yb4[24 + 32*h0];

            float sumy = 0.f;
            FOR_UNROLL (short j = 0; j < 4; ++j) {
                coef[j] = float4(
                        y0[j],
                        y1[j] - 4.0f*y0[j],
                        y2[j] - 4.0f*y1[j],
                        y3[j] - 4.0f*y2[j]);

                sumy += (y0[j] + y1[j]) + (y2[j] + y3[j]);
            }

            FOR_UNROLL (short row = 0; row < nr0; ++row) {
                device const block_tq2_0 & xb = ax[row][ib];
                device const uchar * qs = xb.qs + 4*htg + 32*h0;

                float sum = -sumy;
                FOR_UNROLL (short j = 0; j < 4; ++j) {
                    // express the 2-bit field shifts (v>>2, v>>4, v>>6) as float floor ops
                    const float v = (float)qs[j];

                    const float f0 = v;
                    const float f1 = floor(v*0.25f);    // v>>2
                    const float f2 = floor(v*0.0625);   // v>>4
                    const float f3 = floor(v*0.015625); // v>>6

                    sum += coef[j][0]*f0 + coef[j][1]*f1 + coef[j][2]*f2 + coef[j][3]*f3;
                }

                sumf[row] += xb.d * sum;
            }
        }

        yb4 += QK_K * NBLOCK / 4;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

kernel void GGML_BITNET_Q4_ENTRY(
    device const char *w [[buffer(0)]], device const char *x [[buffer(1)]], device char *y [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]], ushort simd [[simdgroup_index_in_threadgroup]]) {
    const ggml_metal_kargs_mul_mv args = {
        GGML_BITNET_Q_NCOLS,GGML_BITNET_Q_NROWS,1,
        GGML_BITNET_Q_BLOCK_BYTES,GGML_BITNET_Q_ROW_BYTES,GGML_BITNET_Q_WEIGHT_BYTES,GGML_BITNET_Q_WEIGHT_BYTES,
        GGML_BITNET_Q_NCOLS,1,1, sizeof(float),GGML_BITNET_Q_NCOLS*sizeof(float),GGML_BITNET_Q_NCOLS*sizeof(float),GGML_BITNET_Q_NCOLS*sizeof(float),
        GGML_BITNET_Q_NROWS,1,GGML_BITNET_Q4_ROWS, 1,1,0.0f
    };
    kernel_mul_mv_tq2_0_f32_impl<GGML_BITNET_Q4_ROWS>(args, w, x, y, nullptr, group, lane, simd);
}
