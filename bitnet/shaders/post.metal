#include <metal_stdlib>
using namespace metal;

kernel void ffn_post_strict(
    device const float * gate [[buffer(0)]],
    device const float * up [[buffer(1)]],
    device float * output [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= 6912) { return; }
    const float g = gate[index];
    const float u = up[index];
    const float positive = fmax(0.0f, g);
    const float squared = positive * positive;
    const float result = squared * u;
    output[index] = result;
}

// Diagnostic only: bit-preserving shadows before exact in-place gate overwrite.
kernel void ffn_shadow_copy(device const uint *src [[buffer(0)]],
                            device uint *dst [[buffer(1)]],
                            uint index [[thread_position_in_grid]]) {
    if (index < 6912) dst[index] = src[index];
}
