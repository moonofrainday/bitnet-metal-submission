#include "common.h"

// Finite F32 input; one token row per threadgroup. No in-place output.
kernel void kernel_act_quant_f32(
        constant ggml_metal_kargs_act_quant & args [[buffer(0)]],
        device const float * src [[buffer(1)]],
        device float * dst [[buffer(2)]],
        threadgroup float * maxima [[threadgroup(0)]],
        threadgroup uint * invalid [[threadgroup(1)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 size [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]],
        uint width [[threads_per_simdgroup]]) {
    const uint nth = size.x;
    const int64_t row = group.x;
    if (row >= args.nrows) {
        return;
    }
    const int64_t base = row*args.ncols;
    float amax = 0.0f;
    uint bad = 0;
    for (int64_t j = tid; j < args.ncols; j += nth) {
        const float x = src[base + j];
        bad |= !isfinite(x);
        amax = max(amax, abs(x));
    }
    amax = simd_max(amax);
    bad = simd_max(bad);
    if (lane == 0) {
        maxima[simd] = amax;
        invalid[simd] = bad;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        const uint groups = nth/width;
        amax = lane < groups ? maxima[lane] : 0.0f;
        bad = lane < groups ? invalid[lane] : 0;
        amax = simd_max(amax);
        bad = simd_max(bad);
        if (lane == 0) {
            // Preserve the pinned Torch eager reciprocal-then-multiply profile.
            const float reciprocal = precise::divide(1.0f, max(amax, args.eps));
            maxima[0] = reciprocal*127.0f;
            invalid[0] = bad;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = maxima[0];
    for (int64_t j = tid; j < args.ncols; j += nth) {
        const float scaled = src[base + j]*scale;
        const float rounded = rint(scaled);
        const float q = clamp(rounded, -128.0f, 127.0f);
        dst[base + j] = invalid[0] || isnan(rounded) ? NAN : precise::divide(q, scale);
    }
}
