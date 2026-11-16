#pragma once

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// One host clock domain. Metal GPU timestamps remain a separate domain.
static inline uint64_t ggml_metal_q_attribution_now(void) {
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

struct ggml_metal_q_cache_metrics {
    uint64_t lock_wait_ns, snapshot_key_ns, lookup_ns, load_ns, capability_ns;
    uint64_t cache_hits, cache_misses, invalid_paths, stable_load_failures;
};

void ggml_metal_q_attribution_cache_metrics(struct ggml_metal_q_cache_metrics * metrics);
int ggml_metal_q_attribution_device_reset(void);
int ggml_metal_q_attribution_device_write(FILE * file);
void ggml_metal_q_attribution_commit(void * command, const void * graph,
        int slice_start, int slice_end, uint64_t encode_start_ns, uint64_t encode_end_ns);

#ifdef __cplusplus
}
#endif
