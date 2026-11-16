#include <metal_stdlib>
#include "contract.h"
using namespace metal;

#if defined(BITNET_CANONICAL_Q_C8)
#include "ggml-metal-bitnet-q-config.h"
#if BITNET_CANONICAL_Q_C8 != 1
#error "Canonical Q C8 marker must be 1"
#endif
#ifndef POC_VARIANT
#define POC_VARIANT 2
#endif
#ifndef POC_ROWS_PER_SIMD
#define POC_ROWS_PER_SIMD GGML_BITNET_C8_ROWS
#endif
#ifndef POC_SIMD_PER_GROUP
#define POC_SIMD_PER_GROUP GGML_BITNET_C8_NSG
#endif
#define BITNET_C_EXPAND_ENTRY GGML_BITNET_C8_ENTRY
static_assert(GGML_BITNET_Q_CONFIG_VERSION == 1, "Q contract version");
static_assert(POC_VARIANT == 2 && POC_ROWS_PER_SIMD == GGML_BITNET_C8_ROWS &&
              POC_SIMD_PER_GROUP == GGML_BITNET_C8_NSG, "C8 arithmetic or layout differs from Q contract");
static_assert(TQ2_ROWS == GGML_BITNET_Q_NROWS && TQ2_COLUMNS == GGML_BITNET_Q_NCOLS &&
              TQ2_SIMD_WIDTH == GGML_BITNET_Q_SIMD_WIDTH && TQ2_BLOCK_ELEMENTS == GGML_BITNET_Q_BLOCK_ELEMENTS &&
              TQ2_BLOCK_BYTES == GGML_BITNET_Q_BLOCK_BYTES, "C8 packed shape differs from Q contract");
#else
#define BITNET_C_EXPAND_ENTRY tq2_0_matvec_poc
#endif

#ifndef POC_VARIANT
#define POC_VARIANT 0
#endif
#ifndef POC_ROWS_PER_SIMD
#define POC_ROWS_PER_SIMD TQ2_POC_ROWS_PER_SIMD
#endif
#ifndef POC_SIMD_PER_GROUP
#define POC_SIMD_PER_GROUP TQ2_POC_SIMD_PER_GROUP
#endif
#ifndef POC_ALIGNED_X
#define POC_ALIGNED_X 0
#endif

static_assert(POC_VARIANT >= 0 && POC_VARIANT <= 2, "Expected A, B or C");
static_assert(POC_ROWS_PER_SIMD == 2 || POC_ROWS_PER_SIMD == 4 || POC_ROWS_PER_SIMD == 8,
              "Unsupported rows per SIMD group");
static_assert(POC_SIMD_PER_GROUP == 1 || POC_SIMD_PER_GROUP == 2 || POC_SIMD_PER_GROUP == 4,
              "Unsupported SIMD groups per threadgroup");
static_assert(TQ2_ROWS % (POC_ROWS_PER_SIMD * POC_SIMD_PER_GROUP) == 0,
              "Threadgroups must cover complete rows");
static_assert(POC_ALIGNED_X == 0 || (POC_ALIGNED_X == 1 && POC_VARIANT == 0),
              "Aligned X is a separate A-only experiment");

struct TernaryBlock {
    uchar codes[TQ2_CODES_BYTES];
    half scale;
};
static_assert(sizeof(TernaryBlock) == TQ2_BLOCK_BYTES, "TQ2_0 block size");

#if POC_ALIGNED_X
inline float load_activation(device const float4 *x, uint index, ushort lane) {
    const ushort component = lane % 4;
    const ushort owner = lane - component;
    float4 values = float4(0.0f);
    if (lane == owner) {
        values = x[index];
    }
    return simd_shuffle(values, owner)[component];
}
#endif

#if POC_VARIANT == 0
inline float byte_dot(uchar packed, float4 x) {
    const uint4 codes = (uint4(packed) >> uint4(0, 2, 4, 6)) & 3u;
    return dot(float4(int4(codes) - 1), x);
}
#elif POC_VARIANT == 1
inline float unsigned_byte_dot(uchar packed, float4 x) {
    const uint4 codes = (uint4(packed) >> uint4(0, 2, 4, 6)) & 3u;
    return dot(float4(codes), x);
}
#else
inline float4 packed_coefficients(float4 x) {
    const float4 planes = x * float4(1.0f, 0.25f, 0.0625f, 0.015625f);
    return planes - float4(0.0f, planes.xyz);
}

inline float packed_byte_dot(uchar packed, float4 coefficients) {
    // Clear lower planes without shifting the remaining bits.
    const uint4 masked = uint4(packed) & uint4(0xffu, 0xfcu, 0xf0u, 0xc0u);
    return dot(float4(masked), coefficients);
}
#endif

kernel void BITNET_C_EXPAND_ENTRY(
    device const TernaryBlock *weights [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device float *y [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]]) {
    const uint first_row = (group * POC_SIMD_PER_GROUP + simd) * POC_ROWS_PER_SIMD;
    float sums[POC_ROWS_PER_SIMD] = {};

    for (uint block = 0; block < TQ2_BLOCKS_PER_ROW; ++block) {
        const uint i = block * TQ2_BLOCK_ELEMENTS + lane;
#if POC_ALIGNED_X
        // X must be 16-byte aligned. One lane per quartet loads and broadcasts each vector.
        device const float4 *x4 = reinterpret_cast<device const float4 *>(x);
        const uint j = i / 4;
        const float4 first(load_activation(x4, j, lane), load_activation(x4, j + 8, lane),
                           load_activation(x4, j + 16, lane), load_activation(x4, j + 24, lane));
        const float4 second(load_activation(x4, j + 32, lane), load_activation(x4, j + 40, lane),
                            load_activation(x4, j + 48, lane), load_activation(x4, j + 56, lane));
#else
        const float4 first(x[i], x[i + 32], x[i + 64], x[i + 96]);
        const float4 second(x[i + 128], x[i + 160], x[i + 192], x[i + 224]);
#endif

        // A lane shares its eight activations across the output rows.
#if POC_VARIANT != 0
        const float activation_sum = dot(first, float4(1.0f)) + dot(second, float4(1.0f));
#endif
#if POC_VARIANT == 2
        const float4 first_coefficients = packed_coefficients(first);
        const float4 second_coefficients = packed_coefficients(second);
#endif
        #pragma clang loop unroll(full)
        for (uint row = 0; row < POC_ROWS_PER_SIMD; ++row) {
            device const TernaryBlock &w = weights[(first_row + row) * TQ2_BLOCKS_PER_ROW + block];
#if POC_VARIANT == 0
            const float subtotal = byte_dot(w.codes[lane], first) + byte_dot(w.codes[lane + 32], second);
#elif POC_VARIANT == 1
            const float subtotal = unsigned_byte_dot(w.codes[lane], first) +
                                   unsigned_byte_dot(w.codes[lane + 32], second) - activation_sum;
#else
            const float subtotal = packed_byte_dot(w.codes[lane], first_coefficients) +
                                   packed_byte_dot(w.codes[lane + 32], second_coefficients) - activation_sum;
#endif
            sums[row] += float(w.scale) * subtotal;
        }
    }

    #pragma clang loop unroll(full)
    for (uint row = 0; row < POC_ROWS_PER_SIMD; ++row) {
        const float sum = simd_sum(sums[row]);
        if (lane == 0) {
            y[first_row + row] = sum;
        }
    }
}
