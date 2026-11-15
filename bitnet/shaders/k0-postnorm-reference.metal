#include <metal_stdlib>
using namespace metal;

// Same two dependent F32 multiplies as the pre-existing safe post module.
inline float post_component(float gate, float up) {
    const float positive = fmax(0.0f, gate);
    const float squared = positive * positive;
    return squared * up;
}

// Exactly one threadgroup handles the full 6912-element vector. The host may
// use the actual stock RMSNorm thread count (32..1024, a multiple of 32),
// provided this pipeline/device reports support for it. Never launch >1 group.
kernel void ffn_post_weighted_norm(
    device const float4 *gate [[buffer(0)]],
    device const float4 *up [[buffer(1)]],
    device const float4 *gamma [[buffer(2)]],
    device float4 *activation [[buffer(3)]],
    device float4 *normalized [[buffer(4)]],
    uint index [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    uint3 ntg [[threads_per_threadgroup]]) {
    threadgroup float4 cached_activation[1728];
    threadgroup float partial[32];
    if (simd == 0) partial[lane] = 0.0f;

    float sum = 0.0f;
    for (uint i = index; i < 1728; i += ntg.x) {
        const float4 g = gate[i];
        const float4 u = up[i];
        const float4 a = float4(post_component(g.x, u.x),
                                post_component(g.y, u.y),
                                post_component(g.z, u.z),
                                post_component(g.w, u.w));
        cached_activation[i] = a;
        activation[i] = a;
        sum += dot(a, a);
    }
    sum = simd_sum(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = simd_sum(partial[lane]);
    const float mean = sum / 6912.0f;
    const float scale = 1.0f / sqrt(mean + 1.0e-5f);
    for (uint i = index; i < 1728; i += ntg.x) {
        normalized[i] = (cached_activation[i] * scale) * gamma[i];
    }
}
