#include <metal_stdlib>
#include "contract.h"
using namespace metal;

struct TernaryBlock {
    uchar codes[TQ2_CODES_BYTES];
    half scale;
};
static_assert(sizeof(TernaryBlock) == TQ2_BLOCK_BYTES, "TQ2_0 block size");

inline float byte_dot(uchar packed, float4 x) {
    const uint4 codes = (uint4(packed) >> uint4(0, 2, 4, 6)) & 3u;
    return dot(float4(int4(codes) - 1), x);
}

kernel void tq2_0_matvec_poc(
    device const TernaryBlock *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]]) {
    const uint first_row = (group * TQ2_POC_SIMD_PER_GROUP + simd) * TQ2_POC_ROWS_PER_SIMD;
    float sums[TQ2_POC_ROWS_PER_SIMD] = {};

    for (uint block = 0; block < TQ2_BLOCKS_PER_ROW; ++block) {
        const uint i = block * TQ2_BLOCK_ELEMENTS + lane;
        const float4 first(x[i], x[i + 32], x[i + 64], x[i + 96]);
        const float4 second(x[i + 128], x[i + 160], x[i + 192], x[i + 224]);

        // A lane shares its eight activations across four output rows.
        #pragma clang loop unroll(full)
        for (uint row = 0; row < TQ2_POC_ROWS_PER_SIMD; ++row) {
            device const TernaryBlock &w = weights[(first_row + row) * TQ2_BLOCKS_PER_ROW + block];
            const float subtotal = byte_dot(w.codes[lane], first) + byte_dot(w.codes[lane + 32], second);
            sums[row] += float(w.scale) * subtotal;
        }
    }

    #pragma clang loop unroll(full)
    for (uint row = 0; row < TQ2_POC_ROWS_PER_SIMD; ++row) {
        const float sum = simd_sum(sums[row]);
        if (lane == 0) {
            y[first_row + row] = sum;
        }
    }
}
