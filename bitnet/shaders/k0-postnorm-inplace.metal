#include <metal_stdlib>
using namespace metal;

// Same two ordered F32 multiplies as the validated K0/J safe post module.
inline float post_component(float gate, float up) {
    const float positive = fmax(0.0f, gate);
    const float squared = positive * positive;
    return squared * up;
}

// Exactly one group of 1024 threads. Each RW argument is one alias pair:
// gate_and_activation holds G before this dispatch and A afterwards;
// up_and_normalized holds U before this dispatch and normalized A afterwards.
// The two full 27648-byte ranges and gamma must be mutually disjoint.
kernel void ffn_post_weighted_norm_inplace(
    device float4 *gate_and_activation [[buffer(0)]],
    device float4 *up_and_normalized [[buffer(1)]],
    device const float4 *gamma [[buffer(2)]],
    uint index [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    uint3 ntg [[threads_per_threadgroup]]) {
    threadgroup float4 cached_activation[1728];
    threadgroup float partial[32];
    if (simd == 0) partial[lane] = 0.0f;

    float sum = 0.0f;
    for (uint i = index; i < 1728; i += ntg.x) {
        const float4 g = gate_and_activation[i];
        const float4 u = up_and_normalized[i];
        const float4 a = float4(post_component(g.x, u.x),
                                post_component(g.y, u.y),
                                post_component(g.z, u.z),
                                post_component(g.w, u.w));
        cached_activation[i] = a;
        gate_and_activation[i] = a;
        sum += dot(a, a);
    }
    sum = simd_sum(sum);
    // All raw G/U reads and A writes are complete before any U range is
    // overwritten by normalized A. This is a group barrier, not global sync.
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (lane == 0) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = simd_sum(partial[lane]);
    const float mean = sum / 6912.0f;
    const float scale = 1.0f / sqrt(mean + 1.0e-5f);
    for (uint i = index; i < 1728; i += ntg.x) {
        up_and_normalized[i] = (cached_activation[i] * scale) * gamma[i];
    }
}
