#include "ggml-metal-ops.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-impl.h"
#include "ggml-metal-common.h"
#include "ggml-metal-device.h"
#include "ggml-metal-bitnet-q-contract.h"
#include "ggml-metal-fusion.h"
#include "ggml-metal-tuning.h"

#include <cassert>
#include <algorithm>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <set>

static int ggml_metal_bitnet_layer(const char * name, const char * prefix) {
    const size_t length = std::strlen(prefix);
    if (std::strncmp(name, prefix, length) != 0 || name[length] < '0' || name[length] > '9') {
        return -1;
    }
    char * end = nullptr;
    const long layer = std::strtol(name + length, &end, 10);
    if (*end || layer < 0 || layer >= 30) {
        return -1;
    }
    char expected[64];
    std::snprintf(expected, sizeof(expected), "%s%ld", prefix, layer);
    return std::strcmp(name, expected) == 0 ? (int) layer : -1;
}

static bool ggml_metal_bitnet_layer_name(const ggml_tensor * tensor, const char * prefix, int layer) {
    return tensor && layer >= 0 && ggml_metal_bitnet_layer(tensor->name, prefix) == layer;
}

static bool ggml_metal_bitnet_selected_layer(int layer) {
    const char * config = std::getenv("GGML_METAL_BITNET_LAYERS");
    if (!config || std::strcmp(config, "0") == 0) {
        return layer == 0;
    }
    if (std::strcmp(config, "0-3") == 0) {
        return layer >= 0 && layer <= 3;
    }
    if (std::strcmp(config, "0-29") == 0) {
        return layer >= 0 && layer <= 29;
    }
    return false;
}

static bool ggml_metal_bitnet_weight_name(const ggml_tensor * weight, int layer, const char * projection) {
    if (!weight || layer < 0) {
        return false;
    }
    char expected[96];
    std::snprintf(expected, sizeof(expected), "blk.%d.%s.weight", layer, projection);
    return std::strcmp(weight->name, expected) == 0;
}


#include <atomic>
#include <chrono>
#include <cinttypes>
#include <mutex>
#include <thread>
#include <vector>
#include <pthread.h>
#include <time.h>

// Experimental diagnostics are disabled unless explicitly requested. Durations
// are worker CPU work, not elapsed model critical-path time or GPU timestamps.
using bitnet_abc_clock = std::chrono::steady_clock;
static uint64_t bitnet_abc_now() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(bitnet_abc_clock::now().time_since_epoch()).count();
}
struct bitnet_abc_task {
    uint64_t task = 0, thread = 0, task_start_ns = 0, task_end_ns = 0;
    uintptr_t graph = 0, encoder = 0;
    char mode = 'A'; int step = 0;
    uint64_t encode_calls = 0, ffn_encode_calls = 0;
    uint64_t check_calls = 0, target_checks = 0, guard_passes = 0, commits = 0, stock_continuations = 0;
    uint64_t guard_ns = 0, target_guard_ns = 0, commit_ns = 0, ffn_encode_ns = 0;
    uint64_t names_layout_ns = 0, graph_check_ns = 0, buffer_check_ns = 0, pipeline_check_ns = 0;
    uint64_t guard_timers = 0, commit_timers = 0, encode_timers = 0, substage_timers = 0;
};
static std::atomic<uint64_t> bitnet_abc_next_task{0};
static std::mutex bitnet_abc_stats_mutex;
static std::vector<bitnet_abc_task> bitnet_abc_tasks;
extern "C" __attribute__((visibility("default"))) void ggml_backend_metal_ffn_ablation_diag_reset(void) {
    std::lock_guard<std::mutex> lock(bitnet_abc_stats_mutex);
    bitnet_abc_tasks.clear(); bitnet_abc_next_task.store(0);
}
extern "C" __attribute__((visibility("default"))) int ggml_backend_metal_ffn_ablation_diag_write(const char * path) {
    ggml_metal_bitnet_shadow_wait();
    std::vector<uint64_t> costs; costs.reserve(10000);
    for (int i = 0; i < 10000; ++i) { const auto a = bitnet_abc_now(); const auto b = bitnet_abc_now(); costs.push_back(b-a); }
    std::sort(costs.begin(), costs.end());
    const auto positive=std::upper_bound(costs.begin(),costs.end(),uint64_t(0));
    const uint64_t minimum_positive=positive==costs.end()?0:*positive;
    struct timespec resolution{}; const int resolution_status=clock_getres(CLOCK_MONOTONIC,&resolution);
    const uint64_t nominal_api_resolution_ns=uint64_t(resolution.tv_sec)*1000000000ULL+uint64_t(resolution.tv_nsec);
    FILE * f = std::fopen(path, "w"); if (!f) { return 1; }
    std::lock_guard<std::mutex> lock(bitnet_abc_stats_mutex);
    std::fprintf(f, "{\"schema\":1,\"units\":\"nanoseconds\",\"clock\":\"std::chrono::steady_clock\",\"clock_is_steady\":%s,\"nominal_duration_period_ns\":%.12g,\"nominal_api_resolution_ns\":%llu,\"clock_getres_status\":%d,\"observed_min_positive_clock_pair_ns\":%llu,\"calibration_pairs\":10000,\"clock_pair_p50_ns\":%llu,\"clock_pair_p95_ns\":%llu,\"clock_pair_min_ns\":%llu,\"interval_note\":\"Raw CPU worker work; nested/overlapping durations must not be added or called model critical-path time. No diagnostic clock or counter writes when disabled.\",\"tasks\":[",
        bitnet_abc_clock::is_steady ? "true" : "false", 1.e9*double(bitnet_abc_clock::period::num)/double(bitnet_abc_clock::period::den),
        (unsigned long long)nominal_api_resolution_ns,resolution_status,(unsigned long long)minimum_positive,
        (unsigned long long)costs[5000],(unsigned long long)costs[9500],(unsigned long long)costs.front());
    bool first = true;
    for (const auto & r : bitnet_abc_tasks) {
        std::fprintf(f, "%s{\"task\":%llu,\"thread\":%llu,\"graph\":%llu,\"encoder\":%llu,\"mode\":\"%c\",\"step\":%d",
            first?"":",",(unsigned long long)r.task,(unsigned long long)r.thread,(unsigned long long)r.graph,(unsigned long long)r.encoder,r.mode,r.step); first=false;
#define ABC_FIELD(x) std::fprintf(f, ",\"" #x "\":%llu", (unsigned long long)r.x)
        ABC_FIELD(task_start_ns); ABC_FIELD(task_end_ns); ABC_FIELD(encode_calls); ABC_FIELD(ffn_encode_calls); ABC_FIELD(check_calls); ABC_FIELD(target_checks); ABC_FIELD(guard_passes); ABC_FIELD(commits); ABC_FIELD(stock_continuations);
        ABC_FIELD(guard_ns); ABC_FIELD(target_guard_ns); ABC_FIELD(commit_ns); ABC_FIELD(ffn_encode_ns); ABC_FIELD(names_layout_ns); ABC_FIELD(graph_check_ns); ABC_FIELD(buffer_check_ns); ABC_FIELD(pipeline_check_ns);
        ABC_FIELD(guard_timers); ABC_FIELD(commit_timers); ABC_FIELD(encode_timers); ABC_FIELD(substage_timers);
#undef ABC_FIELD
        std::fputs("}",f);
    }
    std::fputs("]}\n", f); const bool bad = std::ferror(f); std::fclose(f); return bad ? 1 : 0;
}
static bool bitnet_abc_flag(const char * name) { const char * v=std::getenv(name); return v && std::strcmp(v,"0")!=0; }
static int bitnet_abc_ffn_node_layer(const ggml_tensor * node) {
    for (const char * p : {"ffn_gate-", "ffn_up-", "ffn_down-", "ffn_relu-", "ffn_sqr(relu)-", "ffn_sub_out-", "ffn_sub_norm-"}) {
        const int l=ggml_metal_bitnet_layer(node->name,p); if(l>=0) { return l; }
    }
    if (node->op==GGML_OP_RMS_NORM && node->src[0]) { return ggml_metal_bitnet_layer(node->src[0]->name,"ffn_sub_out-"); }
    return -1;
}

// threadgroups needed to fill the GPU
// TODO: dedup
static constexpr int64_t GGML_METAL_MIN_THREADGROUPS = 128;

// flash attention KV split for a few query rows: workgroups (one per lane of the vec reduce kernel), and the batch sizes,
// KV length, Q elements (rows x head size) and QK products (rows x KV x head size) per head where it was measured faster
static constexpr int     GGML_METAL_FA_KV_SPLIT_NWG          = 32;
static constexpr int64_t GGML_METAL_FA_KV_SPLIT_MIN_ROWS     = 4;
static constexpr int64_t GGML_METAL_FA_KV_SPLIT_MAX_ROWS     = 19;
static constexpr int64_t GGML_METAL_FA_KV_SPLIT_MIN_KV       = 256;
static constexpr int64_t GGML_METAL_FA_KV_SPLIT_MIN_Q_ELEMS  = 512;
static constexpr int64_t GGML_METAL_FA_KV_SPLIT_MIN_QK_ELEMS = 1024*1024;

static ggml_metal_buffer_id ggml_metal_get_buffer_id(const ggml_tensor * t) {
    if (!t) {
        return { nullptr, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t) buffer->context;

    return ggml_metal_buffer_get_id(ctx, t);
}

static bool ggml_metal_bitnet_numerics_active();
static bool ggml_metal_bitnet_q_readback_active();

struct ggml_metal_op {
    ggml_metal_op(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        ggml_metal_fusion_info * finfo,
        int  idx_start,
        int  idx_end,
        bool use_concurrency,
        bool use_capture,
        int  debug_graph) {
        this->dev             = dev;
        this->lib             = ggml_metal_device_get_library(dev);
        this->enc             = ggml_metal_encoder_init(cmd_buf, use_concurrency);
        this->mem_ranges      = use_concurrency ? ggml_mem_ranges_init(debug_graph) : nullptr;
        this->finfo           = finfo;
        this->idx_start       = idx_start;
        this->idx_end         = idx_end;
        this->use_concurrency = use_concurrency;
        this->use_capture     = use_capture;
        this->debug_graph     = debug_graph;
        this->numerics_capture = ggml_metal_bitnet_numerics_active();
        this->q_readback = ggml_metal_bitnet_q_readback_active();
        this->gf              = gf;
        const char * abc = std::getenv("GGML_METAL_BITNET_FFN_ABLATION");
        if (abc) {
            if (abc[1] || (abc[0]!='A' && abc[0]!='B' && abc[0]!='C')) { GGML_ABORT("invalid FFN ablation mode"); }
            ablation_mode = abc[0];
            if (std::getenv("GGML_METAL_BITNET_FFN_MECHANISM") || bitnet_abc_flag("GGML_METAL_BITNET_FFN") || bitnet_abc_flag("GGML_METAL_BITNET_POC")) {
                GGML_ABORT("FFN ablation modes must not be mixed with legacy custom switches");
            }
        }
        ablation_shadow_all = bitnet_abc_flag("GGML_METAL_BITNET_FFN_ABLATION_SHADOW_ALL");
        ablation_cpu = bitnet_abc_flag("GGML_METAL_BITNET_FFN_ABLATION_CPU");
        ablation_trace = bitnet_abc_flag("GGML_METAL_BITNET_FFN_ABLATION_TRACE");
        const bool ablation_gpu = bitnet_abc_flag("GGML_METAL_BITNET_FFN_ABLATION_GPU");
        ablation_count = ablation_cpu || ablation_trace || ablation_gpu || bitnet_abc_flag("GGML_METAL_BITNET_FFN_ABLATION_COUNT");
        if (ablation_count) {
            ablation_stats.task = bitnet_abc_next_task.fetch_add(1);
            pthread_threadid_np(nullptr, &ablation_stats.thread);
            ablation_stats.mode = ablation_mode ? ablation_mode : 'A';
            const char * st = std::getenv("GGML_METAL_BITNET_FFN_STEP");
            ablation_stats.step = st ? std::atoi(st) : -1;
            ablation_stats.graph = (uintptr_t)gf; ablation_stats.encoder = (uintptr_t)enc;
            if(ablation_cpu) { ablation_stats.task_start_ns=bitnet_abc_now(); }
        }
        ggml_metal_encoder_ffn_ablation_configure(enc, ablation_trace, ablation_stats.task,
                ablation_mode ? ablation_mode : 'A', ablation_stats.step, idx_start, idx_end, gf, ablation_gpu);


        idxs.reserve(gf->n_nodes);

        // filter empty nodes
        // TODO: this can be removed when the allocator starts filtering them earlier
        //       https://github.com/ggml-org/llama.cpp/pull/16130#issuecomment-3327905830
        for (int i = idx_start; i < idx_end; i++) {
            if (!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) {
                idxs.push_back(i);
            }
        }
    }

    ~ggml_metal_op() {
        if (ablation_count) {
            if(ablation_cpu) { ablation_stats.task_end_ns=bitnet_abc_now(); }
            std::lock_guard<std::mutex> lock(bitnet_abc_stats_mutex);
            bitnet_abc_tasks.push_back(ablation_stats);
        }
        ggml_metal_encoder_end_encoding(this->enc);
        ggml_metal_encoder_free(this->enc);
        ggml_mem_ranges_free(this->mem_ranges);
    }

    int n_nodes() const {
        return idxs.size();
    }

    ggml_tensor * node(int i) const {
        assert(i >= 0 && i < (int) idxs.size());
        return ggml_graph_node(gf, idxs[i]);
    }

    ggml_cgraph * graph() const {
        return gf;
    }

    int graph_idx(int i) const {
        assert(i >= 0 && i < (int) idxs.size());
        return idxs[i];
    }

    // consult the fusion table for the longest pattern starting at i0
    // returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes
    const ggml_metal_fusion * can_fuse(int i0, enum ggml_metal_fusion_mode mode, int * n_out) const {
        assert(use_fusion());
        assert(i0 >= 0 && i0 < n_nodes());

        return ggml_metal_fusion_next(gf, idxs.data(), (int) idxs.size(), i0, mode, n_out);
    }

    // whether the RMS_NORM_FWHT fusion at i0 also writes the weighted norm
    bool rms_norm_fwht_write_norm(int i0) const {
        assert(i0 >= 0 && i0 < n_nodes());

        return ggml_metal_fusion_rms_norm_fwht_write_norm(gf, idxs.data(), i0);
    }

    // whether to attempt fusion; the toggle lives in the shared fusion debugging context owned
    // by the device (initialized from GGML_METAL_FUSION_DISABLE, overridable by the test)
    bool use_fusion() const {
        return ggml_metal_fusion_info_enabled(finfo);
    }

    // record that a fusion fired, indexed by the matching table entry
    void count_fusions(const ggml_metal_fusion * fusion) const {
        ggml_metal_fusion_info_count_fusion(finfo, fusion);
    }

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;
    ggml_metal_encoder_t enc;
    ggml_mem_ranges_t    mem_ranges;

    // shared fusion debugging context
    ggml_metal_fusion_info * finfo;

    char ablation_mode = 0;
    bool ablation_count = false, ablation_cpu = false, ablation_trace = false, ablation_shadow_all = false;
    bitnet_abc_task ablation_stats;

    bool use_concurrency;
    bool use_capture;
    bool numerics_capture = false;
    bool q_readback = false;

    int debug_graph;

private:
    ggml_cgraph * gf;

    int idx_start;
    int idx_end;

    // non-empty node indices
    std::vector<int> idxs;
};

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        ggml_metal_fusion_info * finfo,
        int idx_start,
        int idx_end,
        bool use_concurrency,
        bool use_capture,
        int debug_graph) {
    ggml_metal_op_t res = new ggml_metal_op(
        dev,
        cmd_buf,
        gf,
        finfo,
        idx_start,
        idx_end,
        use_concurrency,
        use_capture,
        debug_graph);

    return res;
}

void ggml_metal_op_free(ggml_metal_op_t ctx) {
    delete ctx;
}

int ggml_metal_op_n_nodes(ggml_metal_op_t ctx) {
    return ctx->n_nodes();
}

static bool ggml_metal_op_concurrency_reset(ggml_metal_op_t ctx) {
    if (!ctx->mem_ranges) {
        return true;
    }

    ggml_metal_encoder_memory_barrier(ctx->enc);

    ggml_mem_ranges_reset(ctx->mem_ranges);

    return true;
}

static bool ggml_metal_op_concurrency_check(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return false;
    }

    return ggml_mem_ranges_check(ctx->mem_ranges, node);
}

static bool ggml_metal_op_concurrency_add(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return true;
    }

    return ggml_mem_ranges_add(ctx->mem_ranges, node);
}

// encode_impl checks only the first and the last of n_fuse fused nodes; the nodes between read their own sources,
// which a pending producer may still be writing
static void ggml_metal_op_fusion_concurrency(ggml_metal_op_t ctx, int idx, int n_fuse) {
    for (int i = 1; i < n_fuse - 1; ++i) {
        if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);
            break;
        }
    }
}

static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx) {
    struct ggml_tensor * node = ctx->node(idx);

    //GGML_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, idx, ggml_op_name(node->op));

    if (ggml_is_empty(node)) {
        return 1;
    }

    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            {
                // noop -> next node
                if (ctx->debug_graph > 0) {
                    GGML_LOG_DEBUG("%s: node[%5d] - %-12s %s\n", __func__, idx, ggml_op_name(node->op), "(noop)");
                }
            } return 1;
        default:
            {
            } break;
    }

    if (!ggml_metal_device_supports_op(ctx->dev, node)) {
        GGML_LOG_ERROR("%s: error: unsupported op '%s'\n", __func__, ggml_op_desc(node));
        GGML_ABORT("unsupported op");
    }

    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return 1;
    }

    int n_fuse = 1;

    // check if the current node can run concurrently with other nodes before it
    // the condition is that:
    //  - the current node cannot write to any previous src or dst ranges
    //  - the current node cannot read from any previous dst ranges
    //
    // if the condition is not satisfied, we put a memory barrier and clear all ranges
    // otherwise, we add the new ranges to the encoding context and process the node concurrently
    //
    {
        bool is_concurrent = ggml_metal_op_concurrency_check(ctx, node);

        if (is_concurrent && ctx->use_fusion()) {
            int n_fuse = 1;
            const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
            if (fusion) {
                // fused kernels write to the last node of the group, not necessarily to the first node's dst
                is_concurrent = ggml_mem_ranges_check(ctx->mem_ranges, ctx->node(idx + n_fuse - 1));
            }
        }

        if (!is_concurrent) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        if (ctx->debug_graph > 0) {
            GGML_LOG_DEBUG("%s: node[%5d] - %-12s %-12s %s\n", __func__, idx, ggml_op_name(node->op), ggml_get_name(node), is_concurrent ? "(concurrent)" : "");
        }
        if (ctx->debug_graph > 1) {
            GGML_TENSOR_LOCALS( int64_t, ne0, node->src[0], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb0, node->src[0], nb);
            GGML_TENSOR_LOCALS( int64_t, ne1, node->src[1], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb1, node->src[1], nb);
            GGML_TENSOR_LOCALS( int64_t, ne2, node->src[2], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb2, node->src[2], nb);
            GGML_TENSOR_LOCALS( int64_t, ne3, node->src[3], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb3, node->src[3], nb);
            GGML_TENSOR_LOCALS( int64_t, ne,  node,         ne);
            GGML_TENSOR_LOCALS(uint64_t, nb,  node,         nb);

            if (node->src[0]) {
                GGML_LOG_DEBUG("%s: src0 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[0]->type), ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
                        ggml_is_contiguous(node->src[0]), node->src[0]->name);
            }
            if (node->src[1]) {
                GGML_LOG_DEBUG("%s: src1 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[1]->type), ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
                        ggml_is_contiguous(node->src[1]), node->src[1]->name);
            }
            if (node->src[2]) {
                GGML_LOG_DEBUG("%s: src2 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[2]->type), ne20, ne21, ne22, ne23, nb20, nb21, nb22, nb23,
                        ggml_is_contiguous(node->src[2]), node->src[2]->name);
            }
            if (node->src[3]) {
                GGML_LOG_DEBUG("%s: src3 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[3]->type), ne30, ne31, ne32, ne33, nb30, nb31, nb32, nb33,
                        ggml_is_contiguous(node->src[3]), node->src[3]->name);
            }
            if (node) {
                GGML_LOG_DEBUG("%s: node - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], 1, %s\n", __func__, ggml_type_name(node->type), ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                        node->name);
            }
        }
    }

    switch (node->op) {
        case GGML_OP_CONCAT:
            {
                n_fuse = ggml_metal_op_concat(ctx, idx);
            } break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            {
                n_fuse = ggml_metal_op_bin(ctx, idx);
            } break;
        case GGML_OP_ADD_ID:
            {
                n_fuse = ggml_metal_op_add_id(ctx, idx);
            } break;
        case GGML_OP_REPEAT:
            {
                n_fuse = ggml_metal_op_repeat(ctx, idx);
            } break;
        case GGML_OP_OUT_PROD:
            {
                n_fuse = ggml_metal_op_out_prod(ctx, idx);
            } break;
        case GGML_OP_ACC:
            {
                n_fuse = ggml_metal_op_acc(ctx, idx);
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_FILL:
        case GGML_OP_CLAMP:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
            {
                n_fuse = ggml_metal_op_unary(ctx, idx);
            } break;
        case GGML_OP_SILU_BACK:
            {
                n_fuse = ggml_metal_op_silu_back(ctx, idx);
            } break;
        case GGML_OP_GLU:
            {
                n_fuse = ggml_metal_op_glu(ctx, idx);
            } break;
        case GGML_OP_SUM:
            {
                n_fuse = ggml_metal_op_sum(ctx, idx);
            } break;
        case GGML_OP_ACT_QUANT:
            {
                n_fuse = ggml_metal_op_act_quant(ctx, idx);
            } break;
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            {
                n_fuse = ggml_metal_op_sum_rows(ctx, idx);
            } break;
        case GGML_OP_CUMSUM:
            {
                n_fuse = ggml_metal_op_cumsum(ctx, idx);
            } break;
        case GGML_OP_LIGHTNING_INDEXER:
            {
                n_fuse = ggml_metal_op_lightning_indexer(ctx, idx);
            } break;
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_POST:
            {
                n_fuse = ggml_metal_op_dsv4_hc(ctx, idx);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_fuse = ggml_metal_op_soft_max(ctx, idx);
            } break;
        case GGML_OP_SOFT_MAX_BACK:
            {
                n_fuse = ggml_metal_op_soft_max_back(ctx, idx);
            } break;
        case GGML_OP_SSM_CONV:
            {
                n_fuse = ggml_metal_op_ssm_conv(ctx, idx);
            } break;
        case GGML_OP_SSM_CONV_BACK_SX:
            {
                n_fuse = ggml_metal_op_ssm_conv_back_sx(ctx, idx);
            } break;
        case GGML_OP_SSM_CONV_BACK_C:
            {
                n_fuse = ggml_metal_op_ssm_conv_back_c(ctx, idx);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                n_fuse = ggml_metal_op_ssm_scan(ctx, idx);
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            {
                n_fuse = ggml_metal_op_rwkv(ctx, idx);
            } break;
        case GGML_OP_GATED_DELTA_NET:
            {
                n_fuse = ggml_metal_op_gated_delta_net(ctx, idx);
            } break;
        case GGML_OP_GATED_DELTA_NET_DEFERRED:
            {
                n_fuse = ggml_metal_op_gated_delta_net_deferred(ctx, idx);
            } break;
        case GGML_OP_GDN_CONV_DEFERRED:
            {
                n_fuse = ggml_metal_op_gdn_conv_deferred(ctx, idx);
            } break;
        case GGML_OP_GATED_DELTA_NET_BACK:
            {
                n_fuse = ggml_metal_op_gated_delta_net_back(ctx, idx);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                n_fuse = ggml_metal_op_solve_tri(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT:
            {
                n_fuse = ggml_metal_op_mul_mat(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                n_fuse = ggml_metal_op_mul_mat_id(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT_ID_BACK_A:
            {
                n_fuse = ggml_metal_op_mul_mat_id_back_a(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT_ID_BACK_B:
            {
                n_fuse = ggml_metal_op_mul_mat_id_back_b(ctx, idx);
            } break;
        case GGML_OP_GET_ROWS:
            {
                n_fuse = ggml_metal_op_get_rows(ctx, idx);
            } break;
        case GGML_OP_SET_ROWS:
            {
                n_fuse = ggml_metal_op_set_rows(ctx, idx);
            } break;
        case GGML_OP_DIAG:
            {
                n_fuse = ggml_metal_op_diag(ctx, idx);
            } break;
        case GGML_OP_L2_NORM:
            {
                n_fuse = ggml_metal_op_l2_norm(ctx, idx);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                n_fuse = ggml_metal_op_group_norm(ctx, idx);
            } break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            {
                n_fuse = ggml_metal_op_norm(ctx, idx);
            } break;
        case GGML_OP_RMS_NORM_BACK:
            {
                n_fuse = ggml_metal_op_rms_norm_back(ctx, idx);
            } break;
        case GGML_OP_L2_NORM_BACK:
            {
                n_fuse = ggml_metal_op_l2_norm_back(ctx, idx);
            } break;
        case GGML_OP_GELU_BACK:
            {
                n_fuse = ggml_metal_op_gelu_back(ctx, idx);
            } break;
        case GGML_OP_SIGMOID_BACK:
            {
                n_fuse = ggml_metal_op_sigmoid_back(ctx, idx);
            } break;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            {
                n_fuse = ggml_metal_op_rope(ctx, idx);
            } break;
        case GGML_OP_IM2COL:
            {
                n_fuse = ggml_metal_op_im2col(ctx, idx);
            } break;
        case GGML_OP_CONV_2D:
            {
                n_fuse = ggml_metal_op_conv_2d(ctx, idx);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                n_fuse = ggml_metal_op_conv_2d_dw(ctx, idx);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                n_fuse = ggml_metal_op_conv_transpose_1d(ctx, idx);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_fuse = ggml_metal_op_conv_transpose_2d(ctx, idx);
            } break;
        case GGML_OP_COL2IM_1D:
            {
                n_fuse = ggml_metal_op_col2im_1d(ctx, idx);
            } break;
        case GGML_OP_CONV_3D:
            {
                n_fuse = ggml_metal_op_conv_3d(ctx, idx);
            } break;
        case GGML_OP_UPSCALE:
            {
                n_fuse = ggml_metal_op_upscale(ctx, idx);
            } break;
        case GGML_OP_PAD:
            {
                n_fuse = ggml_metal_op_pad(ctx, idx);
            } break;
        case GGML_OP_SNAKE:
            {
                n_fuse = ggml_metal_op_snake(ctx, idx);
            } break;
        case GGML_OP_LSTM_CELL:
            {
                n_fuse = ggml_metal_op_lstm_cell(ctx, idx);
            } break;
        case GGML_OP_TDT_STEP:
            {
                n_fuse = ggml_metal_op_tdt_step(ctx, idx);
            } break;
        case GGML_OP_CONV_1D_DW_DIRECT:
            {
                n_fuse = ggml_metal_op_conv_1d_dw_direct(ctx, idx);
            } break;
        case GGML_OP_NORM_CHANNEL:
            {
                n_fuse = ggml_metal_op_norm_channel(ctx, idx);
            } break;
        case GGML_OP_SCALE_RESIDUAL:
            {
                n_fuse = ggml_metal_op_scale_residual(ctx, idx);
            } break;
        case GGML_OP_BIAS_GELU:
            {
                n_fuse = ggml_metal_op_bias_gelu(ctx, idx);
            } break;
        case GGML_OP_PAD_REPLICATE_1D:
            {
                n_fuse = ggml_metal_op_pad_replicate_1d(ctx, idx);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                n_fuse = ggml_metal_op_pad_reflect_1d(ctx, idx);
            } break;
        case GGML_OP_DIAG_MASK_INF:
            {
                n_fuse = ggml_metal_op_diag_mask_inf(ctx, idx);
            } break;
        case GGML_OP_ROLL:
            {
                n_fuse = ggml_metal_op_roll(ctx, idx);
            } break;
        case GGML_OP_ARANGE:
            {
                n_fuse = ggml_metal_op_arange(ctx, idx);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                n_fuse = ggml_metal_op_timestep_embedding(ctx, idx);
            } break;
        case GGML_OP_ARGSORT:
            {
                n_fuse = ggml_metal_op_argsort(ctx, idx);
            } break;
        case GGML_OP_TOP_K:
            {
                n_fuse = ggml_metal_op_top_k(ctx, idx);
            } break;
        case GGML_OP_TRI:
            {
                n_fuse = ggml_metal_op_tri(ctx, idx);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                n_fuse = ggml_metal_op_flash_attn_ext(ctx, idx);
            } break;
        case GGML_OP_SET:
            {
                n_fuse = ggml_metal_op_set(ctx, idx);
            } break;
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            {
                n_fuse = ggml_metal_op_cpy(ctx, idx);
            } break;
        case GGML_OP_POOL_1D:
            {
                n_fuse = ggml_metal_op_pool_1d(ctx, idx);
            } break;
        case GGML_OP_POOL_2D:
            {
                n_fuse = ggml_metal_op_pool_2d(ctx, idx);
            } break;
        case GGML_OP_ARGMAX:
            {
                n_fuse = ggml_metal_op_argmax(ctx, idx);
            } break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                n_fuse = ggml_metal_op_opt_step_adamw(ctx, idx);
            } break;
        case GGML_OP_OPT_STEP_SGD:
            {
                n_fuse = ggml_metal_op_opt_step_sgd(ctx, idx);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                n_fuse = ggml_metal_op_count_equal(ctx, idx);
            } break;
        default:
            {
                GGML_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, idx, ggml_op_name(node->op));
                GGML_ABORT("fatal error");
            }
    }

    if (ctx->debug_graph > 0) {
        if (n_fuse > 1) {
            GGML_LOG_DEBUG("%s:               fuse %d ops\n", __func__, n_fuse);
        }
    }

    // update the mem ranges in the encoding context
    for (int i = 0; i < n_fuse; ++i) {
        if (!ggml_metal_op_concurrency_add(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);
        }
    }

    return n_fuse;
}

static void ggml_metal_bitnet_numerics_before(ggml_metal_op_t ctx, int idx);
static void ggml_metal_bitnet_numerics_after(ggml_metal_op_t ctx, int idx, int consumed);
static void ggml_metal_bitnet_q_readback_before(ggml_metal_op_t ctx, int idx);
static void ggml_metal_bitnet_q_readback_after(ggml_metal_op_t ctx, int idx, int consumed);

int ggml_metal_op_encode(ggml_metal_op_t ctx, int idx) {
    const auto * abc_node = ctx->node(idx);
    const int abc_layer = (ctx->ablation_count || ctx->ablation_trace) ? bitnet_abc_ffn_node_layer(abc_node) : -1;
    if (ctx->ablation_count) { ++ctx->ablation_stats.encode_calls; if (abc_layer>=0) { ++ctx->ablation_stats.ffn_encode_calls; } }
    if (ctx->ablation_trace) { ggml_metal_encoder_ffn_ablation_node(ctx->enc, abc_node, abc_layer, ctx->graph_idx(idx)); }
    const uint64_t abc_start = ctx->ablation_cpu && abc_layer>=0 ? bitnet_abc_now() : 0;
    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_push(ctx->enc, ggml_op_desc(ctx->node(idx)));
    }

    if (ctx->numerics_capture) {ggml_metal_bitnet_numerics_before(ctx, idx);}
    if (ctx->q_readback) {ggml_metal_bitnet_q_readback_before(ctx, idx);}
    int res = ggml_metal_op_encode_impl(ctx, idx);
    if (idx + res > ctx->n_nodes()) {
        GGML_ABORT("fusion error: nodes spanning multiple encoders have been fused. this indicates a bug in the fusion logic %s",
                "https://github.com/ggml-org/llama.cpp/pull/14849");
    }

    if (ctx->numerics_capture) {ggml_metal_bitnet_numerics_after(ctx, idx, res);}
    if (ctx->q_readback) {ggml_metal_bitnet_q_readback_after(ctx, idx, res);}

    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_pop(ctx->enc);
    }

    if (abc_start) { ctx->ablation_stats.ffn_encode_ns += bitnet_abc_now()-abc_start; ++ctx->ablation_stats.encode_timers; }
    if (ctx->ablation_trace) { ggml_metal_encoder_ffn_ablation_node(ctx->enc, nullptr, -1, -1); }
    return res;
}

int ggml_metal_op_concat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t dim = ((const int32_t *) op->op_params)[0];

    const bool is_q = ggml_is_quantized(op->type);

    // for quantized types, concat is done at the block level (nb0 == type_size == block size)
    int32_t ne00_arg = ne00;
    int32_t ne10_arg = ne10;
    int32_t ne0_arg  = ne0;
    if (is_q) {
        const int32_t blck = ggml_blck_size(op->type);
        GGML_ASSERT(ne00 % blck == 0);
        GGML_ASSERT(ne10 % blck == 0);
        GGML_ASSERT(ne0  % blck == 0);
        ne00_arg = ne00/blck;
        ne10_arg = ne10/blck;
        ne0_arg  = ne0/blck;
    }

    int nth = std::min(256, ne0_arg);

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;
    if (nth < 256) {
        nrptg = std::min((256 + nth - 1) / nth, ne1);
        if (nrptg * nth > 256) {
            nrptg = 256 / nth;
        }
    }

    const int nw0 = (ne1 + nrptg - 1) / nrptg;

    // split long rows across threadgroups when there are too few rows to fill the GPU
    const int64_t n_rows = (int64_t) nw0*ne2*ne3;

    int nc0 = 1;
    if (nrptg == 1) {
        nc0 = (int) std::max<int64_t>(1, std::min<int64_t>((ne0_arg + nth - 1)/nth, (GGML_METAL_MIN_THREADGROUPS + n_rows - 1)/n_rows));
    }

    ggml_metal_kargs_concat args = {
        /*.ne00 =*/ ne00_arg,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10_arg,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0_arg,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.dim  =*/ dim,
        /*.nc0  =*/ nc0,
    };

    auto pipeline = ggml_metal_library_get_pipeline_concat(lib, op->type);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*nc0, ne2, ne3, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_repeat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // TODO: optimize for degenerate cases such as ggml_nelements(op->src[0]) == 1 and others

    auto pipeline = ggml_metal_library_get_pipeline_repeat(lib, op->type);

    ggml_metal_kargs_repeat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_acc(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);
        const int nw0 = (ne00 + nth - 1)/nth;

        ggml_metal_encoder_dispatch_threadgroups(enc, nw0*ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ pnb1,
        /*.nb02 =*/ pnb2,
        /*.nb03 =*/ pnb3,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
        /*.offs =*/ offs,
        /*.o1   =*/ { 0 },
    };

    auto pipeline = ggml_metal_library_get_pipeline_bin_one(lib, GGML_OP_ADD);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    int nth = 1;

    while (2*nth < args.ne0 && nth < nth_max) {
        nth *= 2;
    }

    ggml_metal_encoder_dispatch_threadgroups(enc, ne11, ne12, ne13, nth, 1, 1);

    return 1;
}

// UNARY (silu/sigmoid/softplus) + MUL in one pass (see GGML_METAL_FUSION_UNARY_MUL)
static int ggml_metal_op_unary_mul(ggml_metal_op_t ctx, int idx, int n_fuse) {
    ggml_tensor * unary = ctx->node(idx);
    ggml_tensor * mul   = ctx->node(idx + n_fuse - 1);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * src0  = unary->src[0];
    const ggml_tensor * other = mul->src[0] == unary ? mul->src[1] : mul->src[0];

    auto pipeline = ggml_metal_library_get_pipeline_unary_mul(lib, unary, mul);

    GGML_TENSOR_LOCALS( int32_t, ne0, src0,  ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, src0,  nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, other, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, other, nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  mul,   ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  mul,   nb);

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.offs =*/ 0,
        /*.o1   =*/ { 0 },
    };

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne10 = ne10/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src0),  1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(other), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mul),   3);

    if (pipeline.cnt) {
        const int n = pipeline.c4 ? ggml_nelements(mul)/4 : ggml_nelements(mul);

        const int nth = MIN(256, n);

        ggml_metal_encoder_dispatch_threadgroups(enc, (n + nth - 1)/nth, 1, 1, nth, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        int nth = 1;
        while (2*nth < args.ne0 && nth < nth_max) {
            nth *= 2;
        }

        ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);
    }

    return n_fuse;
}

static int ggml_metal_op_unary_single(ggml_metal_op_t ctx, int idx, bool allow_fusion) {
    if (allow_fusion && ctx->use_fusion()) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_UNARY_MUL) {
            ctx->count_fusions(fusion);

            return ggml_metal_op_unary_mul(ctx, idx, n);
        }
    }

    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_unary args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.slope =*/ 0.0,
        /*.scale =*/ 0.0,
        /*.bias  =*/ 0.0,
        /*.val   =*/ 0.0,
        /*.min   =*/ 0.0,
        /*.max   =*/ 0.0,
    };

    if (op->op == GGML_OP_LEAKY_RELU) {
        args.slope = ggml_get_op_params_f32(op, 0);
    }

    if (op->op == GGML_OP_SCALE) {
        args.scale = ggml_get_op_params_f32(op, 0);
        args.bias  = ggml_get_op_params_f32(op, 1);
    }

    if (op->op == GGML_OP_FILL) {
        args.val = ggml_get_op_params_f32(op, 0);
    }

    if (op->op == GGML_OP_CLAMP) {
        args.min = ggml_get_op_params_f32(op, 0);
        args.max = ggml_get_op_params_f32(op, 1);
    }

    if (op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
        args.slope = ggml_get_op_params_f32(op, 1); // alpha_n
        args.scale = ggml_get_op_params_f32(op, 2); // alpha_p
        args.bias  = ggml_get_op_params_f32(op, 3); // beta
        args.val   = ggml_get_op_params_f32(op, 4); // eps
    }

    auto pipeline = ggml_metal_library_get_pipeline_unary(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    int bitnet_grid_x, bitnet_grid_y, bitnet_grid_z, bitnet_threads;
    if (pipeline.cnt) {
        const int n = pipeline.c4 ? ggml_nelements(op)/4 : ggml_nelements(op);

        const int nth = MIN(256, n);

        bitnet_grid_x = (n + nth - 1)/nth;
        bitnet_grid_y = bitnet_grid_z = 1;
        bitnet_threads = nth;
        ggml_metal_encoder_dispatch_threadgroups(enc, bitnet_grid_x, 1, 1, nth, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        const int nth = MIN(args.ne00, nth_max);
        const int nk0 = (args.ne00 + nth - 1)/nth;

        bitnet_grid_x = nk0*ne01;
        bitnet_grid_y = ne02;
        bitnet_grid_z = ne03;
        bitnet_threads = nth;
        ggml_metal_encoder_dispatch_threadgroups(enc, bitnet_grid_x, ne02, ne03, nth, 1, 1);
    }

    if (std::getenv("GGML_METAL_BITNET_FFN_TRACE")) {
        const int relu_layer = ggml_metal_bitnet_layer(op->name, "ffn_relu-");
        const int sqr_layer = ggml_metal_bitnet_layer(op->name, "ffn_sqr(relu)-");
        if (relu_layer >= 0) {
            ggml_metal_encoder_trace_bitnet_stage(enc, pipeline, "ffn-relu", op->ne[1]);
            ggml_metal_encoder_trace_bitnet_node(enc, pipeline, "ffn-relu", relu_layer, op, op,
                    bitnet_grid_x, bitnet_grid_y, bitnet_grid_z, bitnet_threads, 1);
        } else if (sqr_layer >= 0) {
            ggml_metal_encoder_trace_bitnet_stage(enc, pipeline, "ffn-sqr", op->ne[1]);
            ggml_metal_encoder_trace_bitnet_node(enc, pipeline, "ffn-sqr", sqr_layer, op, op,
                    bitnet_grid_x, bitnet_grid_y, bitnet_grid_z, bitnet_threads, 1);
        }
    }
    return 1;
}

int ggml_metal_op_unary(ggml_metal_op_t ctx, int idx) {
    return ggml_metal_op_unary_single(ctx, idx, true);
}

int ggml_metal_op_glu(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    if (op->src[1]) {
        GGML_ASSERT(ggml_are_same_shape(op->src[0], op->src[1]));
    }

    auto pipeline = ggml_metal_library_get_pipeline_glu(lib, op);

    const int32_t swp = ggml_get_op_params_i32(op, 1);
    const float alpha = ggml_get_op_params_f32(op, 2);
    const float limit = ggml_get_op_params_f32(op, 3);

    const int32_t i00 = swp ? ne0 : 0;
    const int32_t i10 = swp ? 0 : ne0;

    ggml_metal_kargs_glu args = {
        /*.ne00 =*/ ne00,
        /*.nb01 =*/ nb01,
        /*.ne10 =*/ op->src[1] ? ne10 : ne00,
        /*.nb11 =*/ op->src[1] ? nb11 : nb01,
        /*.ne0  =*/ ne0,
        /*.nb1  =*/ nb1,
        /*.i00  =*/ op->src[1] ? 0 : i00,
        /*.i10  =*/ op->src[1] ? 0 : i10,
        /*.alpha=*/ alpha,
        /*.limit=*/ limit
    };

    const int64_t nrows = ggml_nrows(op->src[0]);

    // split each row over several threadgroups: with a single row (decode) one threadgroup would run on one GPU core
    const int32_t nth = std::max(1, std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), std::min(256, ne0)));
    const int32_t nk  = (ne0 + nth - 1)/nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nk, nrows, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_sum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op  = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const uint64_t n = (uint64_t) ggml_nelements(op->src[0]);

    ggml_metal_kargs_sum args = {
        /*.np =*/ n,
    };

    auto pipeline = ggml_metal_library_get_pipeline_sum(lib, op);

    int nth = 32; // SIMD width

    while (nth < (int) n && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (int) n);

    const int nsg = (nth + 31) / 32;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, GGML_PAD(nsg * sizeof(float), 16), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_act_quant(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);
    const ggml_tensor * src = op->src[0];
    GGML_ASSERT(src && src->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(op));
    GGML_ASSERT(ggml_are_same_shape(src, op) && src->ne[0] > 0);

    const int64_t nrows = ggml_nrows(op);
    GGML_ASSERT(nrows >= 0 && nrows <= INT32_MAX);
    if (nrows == 0) {
        return 1;
    }
    float eps;
    memcpy(&eps, op->op_params, sizeof(eps));
    GGML_ASSERT(std::isfinite(eps) && eps > 0.0f);
    ggml_metal_kargs_act_quant args = { src->ne[0], nrows, eps };
    const auto source = ggml_metal_get_buffer_id(src);
    const auto output = ggml_metal_get_buffer_id(op);
    const size_t bytes = ggml_nbytes(op);
    GGML_ASSERT(source.metal && output.metal && source.offs % sizeof(float) == 0 && output.offs % sizeof(float) == 0);
    GGML_ASSERT(ggml_metal_buffer_contains(source, bytes) && ggml_metal_buffer_contains(output, bytes));
    if (source.metal == output.metal) {
        GGML_ASSERT(source.offs <= output.offs ? output.offs - source.offs >= bytes : source.offs - output.offs >= bytes);
    }

    const auto pipeline = ggml_metal_library_get_pipeline_base(ctx->lib, GGML_OP_ACT_QUANT);
    const int width = ggml_metal_pipeline_thread_execution_width(pipeline);
    const int limit = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    GGML_ASSERT(width > 0 && limit >= width);
    const int nth = std::max(width, std::min(256, limit)/width*width);
    const int groups = nth/width;
    GGML_ASSERT(nth >= width && groups <= width);
    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    ggml_metal_encoder_set_bytes(ctx->enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(ctx->enc, source, 1);
    ggml_metal_encoder_set_buffer(ctx->enc, output, 2);
    ggml_metal_encoder_set_threadgroup_memory_size(ctx->enc, GGML_PAD(width*sizeof(float), 16), 0);
    ggml_metal_encoder_set_threadgroup_memory_size(ctx->enc, GGML_PAD(width*sizeof(uint32_t), 16), 1);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, (int) nrows, 1, 1, nth, 1, 1);
    const char * trace = std::getenv("GGML_METAL_ACTQUANT_TRACE");
    if (trace && std::strcmp(trace, "1") == 0) {
        std::fprintf(stderr, "ACT_QUANT_ENCODE node=%s kernel=kernel_act_quant_f32 pipeline=%p ne=[%lld,%lld,%lld,%lld] src_nb=[%zu,%zu,%zu,%zu] dst_nb=[%zu,%zu,%zu,%zu] src_buffer=%p src_offset=%zu dst_buffer=%p dst_offset=%zu bytes=%zu eps=%.9g grid=[%lld,1,1] threads=[%d,1,1] precise=1 non_inplace=1\n",
            op->name, (void *) pipeline.pipeline,
            (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2], (long long) op->ne[3],
            src->nb[0], src->nb[1], src->nb[2], src->nb[3], op->nb[0], op->nb[1], op->nb[2], op->nb[3],
            source.metal, source.offs, output.metal, output.offs, bytes, eps, (long long) nrows, nth);
    }
    return 1;
}

int ggml_metal_op_sum_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_sum_rows args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_sum_rows(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    int nth = 32; // SIMD width

    while (nth < args.ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (int) args.ne00);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_cumsum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline_blk = ggml_metal_library_get_pipeline_cumsum_blk(lib, op);

    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_blk)) {
        nth *= 2;
    }

    GGML_ASSERT(ne00 <= nth*nth);

    const int64_t net0 = (ne00 + nth - 1) / nth;
    const int64_t net1 = ne01;
    const int64_t net2 = ne02;
    const int64_t net3 = ne03;

    const uint64_t nbt0 = sizeof(float);
    const uint64_t nbt1 = net0*nbt0;
    const uint64_t nbt2 = net1*nbt1;
    const uint64_t nbt3 = net2*nbt2;

    const size_t smem = GGML_PAD(32*sizeof(float), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    {
        ggml_metal_kargs_cumsum_blk args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.net0 =*/ net0,
            /*.net1 =*/ net1,
            /*.net2 =*/ net2,
            /*.net3 =*/ net3,
            /*.nbt0 =*/ nbt0,
            /*.nbt1 =*/ nbt1,
            /*.nbt2 =*/ nbt2,
            /*.nbt3 =*/ nbt3,
            /*.outb =*/ ne00 > nth,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
    }

    if (ne00 > nth) {
        ggml_metal_op_concurrency_reset(ctx);

        {
            ggml_metal_kargs_cumsum_blk args = {
                /*.ne00 =*/ net0,
                /*.ne01 =*/ net1,
                /*.ne02 =*/ net2,
                /*.ne03 =*/ net3,
                /*.nb00 =*/ nbt0,
                /*.nb01 =*/ nbt1,
                /*.nb02 =*/ nbt2,
                /*.nb03 =*/ nbt3,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
                /*.outb =*/ false,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, net1, net2, net3, nth, 1, 1);
        }

        ggml_metal_op_concurrency_reset(ctx);

        {
            auto pipeline_add = ggml_metal_library_get_pipeline_cumsum_add(lib, op);

            ggml_metal_kargs_cumsum_add args = {
                /*.ne00 =*/ ne00,
                /*.ne01 =*/ ne01,
                /*.ne02 =*/ ne02,
                /*.ne03 =*/ ne03,
                /*.nb00 =*/ nb00,
                /*.nb01 =*/ nb01,
                /*.nb02 =*/ nb02,
                /*.nb03 =*/ nb03,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline_add);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

            ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
        }
    }

    return 1;
}

// f32 rows copied as float4 need a row length and every row start 16-byte aligned
static bool ggml_metal_op_get_rows_use_vec4(const ggml_tensor * op) {
    constexpr size_t align = sizeof(float)*4;

    const ggml_tensor * src0 = op->src[0];

    return src0->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && src0->ne[0] % 4 == 0 &&
        src0->nb[1] % align == 0 && src0->nb[2] % align == 0 && src0->nb[3] % align == 0 &&
        op->nb[1] % align == 0 && op->nb[2] % align == 0 && op->nb[3] % align == 0 &&
        ggml_metal_get_buffer_id(src0).offs % align == 0 && ggml_metal_get_buffer_id(op).offs % align == 0;
}

int ggml_metal_op_get_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const bool vec4 = ggml_metal_op_get_rows_use_vec4(op);

    auto pipeline = ggml_metal_library_get_pipeline_get_rows(lib, op->src[0]->type, vec4);

    ggml_metal_kargs_get_rows args = {
        /*.ne00t =*/ vec4 ? ne00/4 : ggml_is_quantized(op->src[0]->type) ? ne00/16 : ne00,
        /*.ne00  =*/ ne00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne10  =*/ ne10,
        /*.nb10  =*/ nb10,
        /*.nb11  =*/ nb11,
        /*.nb12  =*/ nb12,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    const int nth = std::min(args.ne00t, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const int nw0 = (args.ne00t + nth - 1)/nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*ne10, ne11, ne12, nth, 1, 1);

    return 1;
}

int ggml_metal_op_set_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_set_rows(lib, op);

    const int32_t nk0 = ne0/ggml_blck_size(op->type);

    int nth = 32; // SIMD width

    while (nth < nk0 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    int nrptg = 1;
    if (nth > nk0) {
        nrptg = (nth + nk0 - 1)/nk0;
        nth   = nk0;

        if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nrptg--;
        }
    }

    nth = std::min(nth, nk0);

    ggml_metal_kargs_set_rows args = {
        /*.nk0  =*/ nk0,
        /*.ne01 =*/ ne01,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_diag(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(int32_t,  ne, op, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb, op, nb);

    ggml_metal_kargs_diag args = {
        /*.ne00 =*/ne00,
        /*.ne01 =*/ne01,
        /*.ne02 =*/ne02,
        /*.ne03 =*/ne03,
        /*.nb00 =*/nb00,
        /*.nb01 =*/nb01,
        /*.nb02 =*/nb02,
        /*.nb03 =*/nb03,
        /*.ne0  =*/ne0,
        /*.ne1  =*/ne1,
        /*.ne2  =*/ne2,
        /*.ne3  =*/ne3,
        /*.nb0  =*/nb0,
        /*.nb1  =*/nb1,
        /*.nb2  =*/nb2,
        /*.nb3  =*/nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_diag(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, 32, 1, 1);

    return 1;
}

int ggml_metal_op_lightning_indexer(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_LIGHTNING_INDEXER);

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * w = op->src[2];
    const ggml_tensor * m = op->src[3];

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32  ||
                k->type == GGML_TYPE_F16  ||
                k->type == GGML_TYPE_BF16 ||
                k->type == GGML_TYPE_Q4_0 ||
                k->type == GGML_TYPE_Q4_1 ||
                k->type == GGML_TYPE_Q5_0 ||
                k->type == GGML_TYPE_Q5_1 ||
                k->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(m->type == GGML_TYPE_F16);
    GGML_ASSERT(op->type == GGML_TYPE_F32);

    GGML_ASSERT(q->ne[0] == OP_LIGHTNING_INDEXER_DK);

    ggml_metal_kargs_lightning_indexer args = {
        /*.n_kv      =*/ (int32_t) k->ne[2],
        /*.n_batch   =*/ (int32_t) q->ne[2],
        /*.mask_ne3  =*/ (int32_t) m->ne[3],
        /*.nb1       =*/ op->nb[1],
        /*.nb3       =*/ op->nb[3],
        /*.nbq1      =*/ q->nb[1],
        /*.nbq2      =*/ q->nb[2],
        /*.nbq3      =*/ q->nb[3],
        /*.nbk2      =*/ k->nb[2],
        /*.nbk3      =*/ k->nb[3],
        /*.nbw1      =*/ w->nb[1],
        /*.nbw3      =*/ w->nb[3],
        /*.nbm1      =*/ m->nb[1],
        /*.nbm3      =*/ m->nb[3],
    };

    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(q),  1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(k),  2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(w),  3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(m),  4);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 5);

    const int nsg   = OP_LIGHTNING_INDEXER_NSG;
    const int nkptg = OP_LIGHTNING_INDEXER_NKPSG*nsg;
    const int nbptg = OP_LIGHTNING_INDEXER_NBPTG;

    auto pipeline = ggml_metal_library_get_pipeline_lightning_indexer(ctx->lib, op);
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_dispatch_threadgroups(enc,
            (k->ne[2] + nkptg - 1)/nkptg,
            (q->ne[2] + nbptg - 1)/nbptg,
            q->ne[3], 32, nsg, 1);

    return 1;
}

int ggml_metal_op_dsv4_hc(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_encoder_t enc = ctx->enc;
    auto pipeline = ggml_metal_library_get_pipeline_dsv4_hc(ctx->lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);

    switch (op->op) {
        case GGML_OP_DSV4_HC_COMB:
            {
                const ggml_tensor * mixes = op->src[0];
                const ggml_tensor * scale = op->src[1];
                const ggml_tensor * base  = op->src[2];

                GGML_ASSERT(mixes->type == GGML_TYPE_F32);
                GGML_ASSERT(scale->type == GGML_TYPE_F32);
                GGML_ASSERT(base->type  == GGML_TYPE_F32);
                GGML_ASSERT(op->type    == GGML_TYPE_F32);
                GGML_ASSERT(mixes->ne[0] == 24);
                GGML_ASSERT(op->ne[0] == 4 && op->ne[1] == 4);

                ggml_metal_kargs_dsv4_hc_comb args = {
                    /*.n_tokens =*/ (int32_t) mixes->ne[1],
                    /*.n_iter   =*/ ggml_get_op_params_i32(op, 1),
                    /*.nb_m0    =*/ mixes->nb[0],
                    /*.nb_m1    =*/ mixes->nb[1],
                    /*.nb_s0    =*/ scale->nb[0],
                    /*.nb_b0    =*/ base->nb[0],
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.nb_d2    =*/ op->nb[2],
                    /*.eps      =*/ ggml_get_op_params_f32(op, 0),
                };

                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(mixes), 1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(scale), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(base),  3);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),    4);

                // One SIMDgroup owns one 4x4 Sinkhorn matrix. Packing up to four
                // independent tokens per threadgroup keeps both decode and prompt
                // dispatches compact without any threadgroup-memory synchronization.
                const int nsg = std::min(4, args.n_tokens);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (args.n_tokens + nsg - 1)/nsg, 1, 1, 32, nsg, 1);
            } break;
        case GGML_OP_DSV4_HC_PRE:
            {
                const ggml_tensor * x       = op->src[0];
                const ggml_tensor * weights = op->src[1];

                GGML_ASSERT(x->type       == GGML_TYPE_F32);
                GGML_ASSERT(weights->type == GGML_TYPE_F32);
                GGML_ASSERT(op->type      == GGML_TYPE_F32);

                ggml_metal_kargs_dsv4_hc_pre args = {
                    /*.n_embd   =*/ (int32_t) x->ne[0],
                    /*.n_tokens =*/ (int32_t) x->ne[2],
                    /*.nb_x0    =*/ x->nb[0],
                    /*.nb_x1    =*/ x->nb[1],
                    /*.nb_x2    =*/ x->nb[2],
                    /*.nb_w0    =*/ weights->nb[0],
                    /*.nb_w1    =*/ weights->nb[1],
                    /*.nb_w2    =*/ weights->nb[2],
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.scale    =*/ ggml_get_op_params_f32(op, 0),
                };

                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),       1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weights), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),      3);

                const int n_tiles = (args.n_embd + 31)/32;
                const int nsg = std::min(4, n_tiles);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (n_tiles + nsg - 1)/nsg, args.n_tokens, 1, 32, nsg, 1);
            } break;
        case GGML_OP_DSV4_HC_POST:
            {
                const ggml_tensor * x        = op->src[0];
                const ggml_tensor * residual = op->src[1];
                const ggml_tensor * post     = op->src[2];
                const ggml_tensor * comb     = op->src[3];

                GGML_ASSERT(x->type        == GGML_TYPE_F32);
                GGML_ASSERT(residual->type == GGML_TYPE_F32);
                GGML_ASSERT(post->type     == GGML_TYPE_F32);
                GGML_ASSERT(op->type       == GGML_TYPE_F32);
                GGML_ASSERT(residual->ne[1] == 4);

                ggml_metal_kargs_dsv4_hc_post args = {
                    /*.n_embd   =*/ (int32_t) x->ne[0],
                    /*.n_tokens =*/ (int32_t) x->ne[1],
                    /*.nb_x0    =*/ x->nb[0],
                    /*.nb_x1    =*/ x->nb[1],
                    /*.nb_r0    =*/ residual->nb[0],
                    /*.nb_r1    =*/ residual->nb[1],
                    /*.nb_r2    =*/ residual->nb[2],
                    /*.nb_p0    =*/ post->nb[0],
                    /*.nb_p1    =*/ post->nb[1],
                    /*.nb_c0    =*/ comb ? comb->nb[0] : 0,
                    /*.nb_c1    =*/ comb ? comb->nb[1] : 0,
                    /*.nb_c2    =*/ comb ? comb->nb[2] : 0,
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.nb_d2    =*/ op->nb[2],
                };

                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),        1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(residual), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(post),     3);
                if (comb) {
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(comb), 4);
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),   5);
                } else {
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),   4);
                }

                const int n_tiles = (args.n_embd + 31)/32;
                const int nsg = std::min(4, n_tiles);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (n_tiles + nsg - 1)/nsg, args.n_tokens, 1, 32, nsg, 1);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    return 1;
}

static constexpr int64_t GGML_METAL_SOFT_MAX_GRID_THREADS = 8192;

int ggml_metal_op_soft_max(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    if (ctx->use_fusion()) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_TOPK_MOE) {
            return ggml_metal_op_topk_moe(ctx, idx);
        }
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float scale;
    float max_bias;

    memcpy(&scale,    ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) op->op_params) + 1, sizeof(max_bias));

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // softmax

    ggml_metal_kargs_soft_max args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne11        =*/ ne11,
        /*.ne12        =*/ ne12,
        /*.ne13        =*/ ne13,
        /*.nb11        =*/ nb11,
        /*.nb12        =*/ nb12,
        /*.nb13        =*/ nb13,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.scale       =*/ scale,
        /*.max_bias    =*/ max_bias,
        /*.m0          =*/ m0,
        /*.m1          =*/ m1,
        /*.n_head_log2 =*/ n_head_log2,
    };

    auto pipeline = ggml_metal_library_get_pipeline_soft_max(lib, op);

    int nth = 32; // SIMD width
    const int nth_max = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);

    if (ne00%4 == 0) {
        while (nth < ne00/4 && int64_t(nth)*ne01*ne02*ne03 < GGML_METAL_SOFT_MAX_GRID_THREADS && nth <= nth_max/2) {
            nth *= 2;
        }
    } else {
        while (nth < ne00 && int64_t(nth)*ne01*ne02*ne03 < GGML_METAL_SOFT_MAX_GRID_THREADS && nth <= nth_max/2) {
            nth *= 2;
        }
    }

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_ssm_conv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    int n_fuse = 1;
    bool use_silu = false;

    if (ctx->use_fusion()) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_SSM_CONV_SILU) {
            n_fuse = n;
            use_silu = true;

            ctx->count_fusions(fusion);
        }
    }

    ggml_metal_kargs_ssm_conv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne11 =*/ ne11,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
    };

    const ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(n_fuse > 1 ? ctx->node(idx + n_fuse - 1) : op);

    // Use batched kernel for prefill (ne1 > 1) to reduce threadgroup dispatch overhead
    const bool use_batched = (ne1 > 1);

    if (use_batched) {
        // Determine the smallest power of 2 that's >= ne1, but <= 256
        int BATCH_SIZE;
        if      (ne1 > 128) BATCH_SIZE = 256;
        else if (ne1 > 64 ) BATCH_SIZE = 128;
        else if (ne1 > 32 ) BATCH_SIZE = 64;
        else if (ne1 > 16 ) BATCH_SIZE = 32;
        else if (ne1 > 8  ) BATCH_SIZE = 16;
        else if (ne1 > 4  ) BATCH_SIZE = 8;
        else                BATCH_SIZE = 2;

        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv_batched(lib, op, BATCH_SIZE, (int32_t) ne10, use_silu);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst, 3);

        // Dispatch: ne01 rows, ceil(ne1/BATCH_SIZE) token batches, ne02 sequences
        // Each threadgroup has BATCH_SIZE threads, each handling one token
        const int n_token_batches = (ne1 + BATCH_SIZE - 1) / BATCH_SIZE;
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, n_token_batches, ne02, BATCH_SIZE, 1, 1);
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv(lib, op, (int32_t) ne10, use_silu);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst, 3);

        const int nth = 32;
        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nth - 1) / nth, ne1, ne02, nth, 1, 1);
    }

    if (n_fuse > 1 && ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: SSM_CONV + UNARY\n", __func__);
    }

    return n_fuse;
}

int ggml_metal_op_ssm_scan(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;
    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne4, op->src[4], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb4, op->src[4], nb);
    GGML_TENSOR_LOCALS( int32_t, ne5, op->src[5], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb5, op->src[5], nb);
    GGML_TENSOR_LOCALS( int32_t, ne6, op->src[6], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb6, op->src[6], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const ggml_tensor * src3 = op->src[3];
    const ggml_tensor * src4 = op->src[4];
    const ggml_tensor * src5 = op->src[5];
    const ggml_tensor * src6 = op->src[6];

    GGML_ASSERT(src3);
    GGML_ASSERT(src4);
    GGML_ASSERT(src5);
    GGML_ASSERT(src6);

    const int64_t d_state      = ne00;
    const int64_t d_inner      = ne01;
    const int64_t n_head       = ne02;
    const int64_t n_group      = ne41;
    const int64_t n_seq_tokens = ne12;
    const int64_t n_seqs       = ne13;
    const int64_t K            = ggml_get_op_params_i32(op, 0);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(op->src[1]) + K*d_state*d_inner*n_head*n_seqs == ggml_nelements(op));

    ggml_metal_kargs_ssm_scan args = {
        /*.d_state      =*/ d_state,
        /*.d_inner      =*/ d_inner,
        /*.n_head       =*/ n_head,
        /*.n_group      =*/ n_group,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seq_tokens_total =*/ n_seq_tokens,
        /*.token_offset =*/ 0,
        /*.n_seqs       =*/ n_seqs,
        /*.K            =*/ K,
        /*.s_off        =*/ ggml_nelements(op->src[1]) * sizeof(float),
        /*.nb00         =*/ nb00,
        /*.nb01         =*/ nb01,
        /*.nb02         =*/ nb02,
        /*.nb03         =*/ nb03,
        /*.nb10         =*/ nb10,
        /*.nb11         =*/ nb11,
        /*.nb12         =*/ nb12,
        /*.ns12         =*/ nb12/nb10,
        /*.nb13         =*/ nb13,
        /*.nb20         =*/ nb20,
        /*.nb21         =*/ nb21,
        /*.ns21         =*/ nb21/nb20,
        /*.nb22         =*/ nb22,
        /*.ne30         =*/ ne30,
        /*.nb31         =*/ nb31,
        /*.nb41         =*/ nb41,
        /*.nb42         =*/ nb42,
        /*.ns42         =*/ nb42/nb40,
        /*.nb43         =*/ nb43,
        /*.nb51         =*/ nb51,
        /*.nb52         =*/ nb52,
        /*.ns52         =*/ nb52/nb50,
        /*.nb53         =*/ nb53,
        /*.nb0          =*/ nb0,
    };

    constexpr int64_t CHUNK = OP_SSM_SCAN_SSD_CS;

    const int64_t snap_reserve = K > 1 ? K : 0; // tokens reserved for sequential kernel rollback snapshots
    const int64_t mma_tokens = ((n_seq_tokens - snap_reserve) / CHUNK) * CHUNK; // largest multiple of CHUNK that leaves snap_reserve for the tail
    const bool use_mma =
        mma_tokens > 0 &&
        ne30 == 1 && // checks that A tensor is set to scalar decay per head (A shape {1, n_head})
        props_dev->has_simdgroup_mm && // hardware check for M1 or newer
        d_state % 8 == 0 && // d_state must be multiple of 8 to align with simdgroup_float 8x8 tiles
        d_inner == OP_SSM_SCAN_SSD_HD; // mma kernel is specialized for the Mamba-2 head dim; this checks it

    const auto dispatch = [&](ggml_metal_pipeline_with_params pipeline, int64_t nth, int64_t n_tg_x) {
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        GGML_ASSERT(pipeline.smem <= props_dev->max_theadgroup_memory_size);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), 5);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), 6);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), 7);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         8);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, n_tg_x, n_head, n_seqs, nth, 1, 1);
    };

    if (!use_mma) {
        dispatch(ggml_metal_library_get_pipeline_ssm_scan(lib, op, false), d_state, d_inner);
        return 1;
    }

    args.n_seq_tokens = mma_tokens;
    dispatch(
        ggml_metal_library_get_pipeline_ssm_scan_ssd_mma(lib, op),
        OP_SSM_SCAN_SSD_NSG*32,
        1);

    if (mma_tokens < n_seq_tokens) {
        ggml_metal_op_concurrency_reset(ctx);

        args.n_seq_tokens = n_seq_tokens - mma_tokens;
        args.token_offset = mma_tokens;
        dispatch(ggml_metal_library_get_pipeline_ssm_scan(lib, op, true), d_state, d_inner);
    }

    return 1;
}

int ggml_metal_op_rwkv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int64_t B = op->op == GGML_OP_RWKV_WKV6 ? op->src[5]->ne[1] : op->src[6]->ne[1];
    const int64_t T = op->src[0]->ne[2];
    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    auto pipeline = ggml_metal_library_get_pipeline_rwkv(lib, op);

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++);
    if (op->op == GGML_OP_RWKV_WKV7) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), ida++);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &B, sizeof(B), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &T, sizeof(T), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &C, sizeof(C), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &H, sizeof(H), ida++);

    ggml_metal_encoder_dispatch_threadgroups(enc, B * H, 1, 1, C/H, 1, 1);

    return 1;
}

int ggml_metal_op_gated_delta_net(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion();
    const int  debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_gated_delta_net(lib, op);

    // when fused with the trailing cache cpy, the snapshots are written straight into the
    // recurrent cache and the cpy is skipped (see GGML_METAL_FUSION_GDN_CACHE)
    ggml_metal_buffer_id bid_out = ggml_metal_get_buffer_id(op);
    uint64_t nb_out = 0;
    int n_fuse = 1;

    if (use_fusion) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);

        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_GDN_CACHE) {
            const ggml_tensor * dst_cache = ctx->node(idx + 1)->src[1]; // cache view

            bid_out = ggml_metal_get_buffer_id(dst_cache);
            nb_out  = dst_cache->nb[2]/sizeof(float);
            n_fuse = 2;

            ctx->count_fusions(fusion);

            if (debug_fusion > 1) {
                GGML_LOG_DEBUG("%s: fuse: GATED_DELTA_NET + CPY\n", __func__);
            }
        }
    }

    int ida = 0;

    ggml_metal_kargs_gated_delta_net args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne20 =*/ ne20,
        /*.ne21 =*/ ne21,
        /*.ne22 =*/ ne22,
        /*.ne23 =*/ ne23,
        /*.nb20 =*/ nb20,
        /*.nb21 =*/ nb21,
        /*.nb22 =*/ nb22,
        /*.nb23 =*/ nb23,
        /*.ns02 =*/ (int32_t) (nb02/sizeof(float)),
        /*.ns12 =*/ (int32_t) (nb12/sizeof(float)),
        /*.ns22 =*/ (int32_t) (nb22/sizeof(float)),
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.nb_out =*/ nb_out,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                  ida++); // args
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++); // q
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++); // k
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++); // v
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++); // gate
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++); // beta
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++); // state
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++); // dst (attn)
    ggml_metal_encoder_set_buffer  (enc, bid_out,                              ida++); // state_out

    const int nsg = pipeline.nsg;

    if (pipeline.gdn_c8) {
        ggml_metal_encoder_dispatch_threadgroups(enc, (ne20 + 31)/32, ne21, ne23, 32, nsg, 1);
    } else {
        ggml_metal_encoder_dispatch_threadgroups(enc, (op->src[2]->ne[0] + nsg - 1)/nsg, op->src[2]->ne[1], op->src[2]->ne[3], 32, nsg, 1);
    }

    return n_fuse;
}

int ggml_metal_op_gated_delta_net_deferred(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion();
    const int  debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    const ggml_tensor * q   = op->src[0];
    const ggml_tensor * k   = op->src[1];
    const ggml_tensor * v   = op->src[2];
    const ggml_tensor * st  = op->src[5];
    const ggml_tensor * rec = op->src[6];

    const int64_t S_v = v->ne[0];
    const int64_t H   = v->ne[1];
    const int64_t G   = op->src[3]->ne[0];
    const int64_t K   = ggml_get_op_params_i32(op, 0);

    auto pipeline = ggml_metal_library_get_pipeline_gated_delta_net_deferred(lib, op);

    // when fused with the trailing cache cpy, the new base is written straight into the
    // recurrent cache and the cpy is skipped (see GGML_METAL_FUSION_GDN_DEFERRED_CACHE)
    ggml_metal_buffer_id bid_fuse = ggml_metal_get_buffer_id(op);
    int32_t fused  = 0;
    int     n_fuse = 1;

    if (use_fusion) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);

        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_GDN_DEFERRED_CACHE) {
            bid_fuse = ggml_metal_get_buffer_id(ctx->node(idx + 1)->src[1]); // cache view
            fused  = 1;
            n_fuse = 2;

            ctx->count_fusions(fusion);

            if (debug_fusion > 1) {
                GGML_LOG_DEBUG("%s: fuse: GATED_DELTA_NET_DEFERRED + CPY\n", __func__);
            }
        }
    }

    ggml_metal_kargs_gated_delta_net_deferred args = {
        /*.ne01  =*/ (int32_t) q->ne[1],
        /*.ne11  =*/ (int32_t) k->ne[1],
        /*.ne21  =*/ (int32_t) v->ne[1],
        /*.ne22  =*/ (int32_t) v->ne[2],
        /*.ne23  =*/ (int32_t) v->ne[3],
        /*.nb01  =*/ q->nb[1],
        /*.nb02  =*/ q->nb[2],
        /*.nb03  =*/ q->nb[3],
        /*.nb11  =*/ k->nb[1],
        /*.nb12  =*/ k->nb[2],
        /*.nb13  =*/ k->nb[3],
        /*.nb21  =*/ v->nb[1],
        /*.nb22  =*/ v->nb[2],
        /*.nb23  =*/ v->nb[3],
        /*.nb31  =*/ op->src[3]->nb[1],
        /*.nb32  =*/ op->src[3]->nb[2],
        /*.nb33  =*/ op->src[3]->nb[3],
        /*.nb41  =*/ op->src[4]->nb[1],
        /*.nb42  =*/ op->src[4]->nb[2],
        /*.nb43  =*/ op->src[4]->nb[3],
        /*.nb51  =*/ st->nb[1],
        /*.nb52  =*/ st->nb[2],
        /*.nb53  =*/ st->nb[3],
        /*.nb61  =*/ rec->nb[1],
        /*.rec_v =*/ (int32_t) (K*S_v*k->ne[1]),
        /*.rec_g =*/ (int32_t) (K*S_v*k->ne[1] + K*S_v*H),
        /*.rec_b =*/ (int32_t) (K*S_v*k->ne[1] + K*S_v*H + K*G*H),
        /*.fused =*/ fused,
        /*.gates =*/ ggml_get_op_params_i32(op, 1),
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                  ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++); // q
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++); // k
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++); // v
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++); // gate
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++); // beta
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++); // base state
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), ida++); // record (in/out)
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[7]), ida++); // record indices
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++); // dst
    ggml_metal_encoder_set_buffer  (enc, bid_fuse,                             ida++); // new base (fused)
    // the gate parameters, bound to dst when unused
    ggml_metal_encoder_set_buffer  (enc, op->src[8] ? ggml_metal_get_buffer_id(op->src[8]) : ggml_metal_get_buffer_id(op), ida++); // dt
    ggml_metal_encoder_set_buffer  (enc, op->src[9] ? ggml_metal_get_buffer_id(op->src[9]) : ggml_metal_get_buffer_id(op), ida++); // a

    const int nsg = pipeline.nsg;
    const int rps = pipeline.nr0; // state rows per simdgroup

    ggml_metal_encoder_dispatch_threadgroups(enc, (S_v/rps + nsg - 1)/nsg, H, v->ne[3], 32, nsg, 1);

    return n_fuse;
}

int ggml_metal_op_gdn_conv_deferred(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * x    = op->src[0];
    const ggml_tensor * hist = op->src[2];

    const int32_t head_dim = ggml_get_op_params_i32(op, 2);

    ggml_metal_kargs_gdn_conv_deferred args = {
        /*.C        =*/ (int32_t) x->ne[0],
        /*.n_tokens =*/ (int32_t) x->ne[1],
        /*.K        =*/ ggml_get_op_params_i32(op, 0),
        /*.n_norm   =*/ ggml_get_op_params_i32(op, 1),
        /*.eps      =*/ ggml_get_op_params_f32(op, 3),
        /*.nb_x1    =*/ x->nb[1],
        /*.nb_x2    =*/ x->nb[2],
        /*.nb1      =*/ op->nb[1],
        /*.nb2      =*/ op->nb[2],
        /*.nb_h1    =*/ hist->nb[1],
    };

    auto pipeline = ggml_metal_library_get_pipeline_gdn_conv_deferred(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1); // x
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2); // conv weights
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3); // history (in/out)
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4); // record indices
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         5); // dst

    ggml_metal_encoder_set_threadgroup_memory_size(enc, GGML_PAD((head_dim/32)*sizeof(float), 16), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, x->ne[0]/head_dim, x->ne[1], x->ne[2], head_dim, 1, 1);

    return 1;
}

int ggml_metal_op_solve_tri(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_solve_tri args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_solve_tri(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nsg = pipeline.nsg;

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne10 + nsg - 1)/nsg, ne02, ne03, 32, nsg, 1);

    return 1;
}

int ggml_metal_op_set(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);
        const int nw0 = (ne00 + nth - 1)/nth;

        ggml_metal_encoder_dispatch_threadgroups(enc, nw0*ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[1]->type, op->type);

    GGML_ASSERT(ne10 % ggml_blck_size(op->src[1]->type) == 0);

    int64_t nk0 = ne10;
    if (ggml_is_quantized(op->src[1]->type)) {
        nk0 = ne10/16;
    } else if (ggml_is_quantized(op->type)) {
        nk0 = ne10/ggml_blck_size(op->type);
    }

    int nth = std::min<int>(nk0*ne11, 256);

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(op->src[1]->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            if (nrptg*nth > 256) {
                nrptg--;
            }
        }
    }

    nth = std::min<int>(nth, nk0);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb10,
        /*.nb01 =*/ nb11,
        /*.nb02 =*/ nb12,
        /*.nb03 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ ggml_element_size(op),
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    bid_dst.offs += offs;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne11 + nrptg - 1)/nrptg, ne12, ne13, nth, nrptg, 1);

    return 1;
}

static int ggml_metal_op_cont_fwht(ggml_metal_op_t ctx, int idx);

int ggml_metal_op_cpy(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    if (op->op == GGML_OP_CONT && ctx->use_fusion()) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_CONT_FWHT) {
            const int n_fwht = ggml_metal_op_cont_fwht(ctx, idx);
            if (n_fwht > 0) {
                ctx->count_fusions(fusion);
                return n_fwht;
            }
        }
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type);

    GGML_ASSERT(ne00 % ggml_blck_size(op->src[0]->type) == 0);

    int64_t nk0 = ne00;
    if (ggml_is_quantized(op->src[0]->type)) {
        nk0 = ne00/16;
    } else if (ggml_is_quantized(op->type)) {
        nk0 = ne00/ggml_blck_size(op->type);
    }

    int nth = std::min<int>(nk0*ne01, 256);

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(op->src[0]->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            if (nrptg*nth > 256) {
                nrptg--;
            }
        }
    }

    nth = std::min<int>(nth, nk0);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_pool_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t s0 = opts[2];
    const int32_t p0 = opts[3];

    const int64_t IW = op->src[0]->ne[0];
    const int64_t OW = op->ne[0];

    const int64_t np = ggml_nelements(op);

    ggml_metal_kargs_pool_1d args_pool_1d = {
        /* .k0 = */  k0,
        /* .s0 = */  s0,
        /* .p0 = */  p0,
        /* .IW = */  IW,
        /* .OW = */  OW,
        /* .np = */  np
    };

    auto pipeline = ggml_metal_library_get_pipeline_pool_1d(lib, op, op_pool);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_1d, sizeof(args_pool_1d),  0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

static int ggml_metal_op_fwht_impl(ggml_metal_op_t ctx, ggml_tensor * op, const ggml_tensor * src, const ggml_tensor * signs) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const int64_t n = op->src[0]->ne[0];
    const int64_t nrows = ggml_nelements(src) / n;

    ggml_metal_kargs_fwht args = {
        /*.nrows = */ (int32_t) nrows,
        /*.n_blk  = */ signs ? (int32_t) (signs->ne[0] / n) : 0,
    };

    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(op->type == GGML_TYPE_F32);

    int nth = n >= GGML_METAL_FWHT_TG_MIN_N ? GGML_METAL_FWHT_TG_NT : 0;
    auto pipeline = ggml_metal_library_get_pipeline_fwht(lib, n, src->type, nth);
    if (nth != 0 && (!pipeline.pipeline || ggml_metal_pipeline_max_theads_per_threadgroup(pipeline) < nth)) {
        nth = GGML_METAL_FWHT_TG_NT_FALLBACK;
        pipeline = ggml_metal_library_get_pipeline_fwht(lib, n, src->type, nth);
    }
    if (!pipeline.pipeline || (nth != 0 && ggml_metal_pipeline_max_theads_per_threadgroup(pipeline) < nth)) {
        GGML_ABORT("FWHT pipeline unavailable: n=%lld, type=%s, nth=%d", (long long) n, ggml_type_name(src->type), nth);
    }

    const int th_max = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(src), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(signs ? signs : src), 3);

    if (nth != 0) {
        ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);
        return 1;
    }

    const int simd_size = 32;
    int sg_per_tg = 2;
    sg_per_tg = std::min(sg_per_tg, th_max/simd_size);
    sg_per_tg = std::max(sg_per_tg, 1);

    const int64_t n_tg = (nrows + sg_per_tg - 1) / sg_per_tg;
    ggml_metal_encoder_dispatch_threadgroups(enc, n_tg, 1, 1, 32*sg_per_tg, 1, 1);

    return 1;
}

int ggml_metal_op_fwht(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);
    return ggml_metal_op_fwht_impl(ctx, op, op->src[1], nullptr);
}

// the fusion table (ggml_metal_fusion_check_fwht_signed) matched MUL -> RESHAPE -> MUL_MAT and
// verified that the MUL/RESHAPE have no other consumer. the table has no device handle, so the
// threadgroup-memory bound of the wide FWHT kernels is enforced here: a declined fusion falls
// back to the unfused MUL + MUL_MAT
static bool ggml_metal_op_fwht_signed_elidable(ggml_metal_op_t ctx, int idx) {
    const ggml_tensor * mm = ctx->node(idx + 1);

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    return ggml_metal_op_mul_mat_use_fwht(mm, props_dev->max_theadgroup_memory_size);
}

static int ggml_metal_op_fwht_signed(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * mul = ctx->node(idx);
    ggml_tensor * mm  = ctx->node(idx + 1);

    const auto [x, signs] = ggml_metal_fwht_signed_operands(mul);

    // encode_impl checks only the first fused node (the MUL); the fused kernel writes the MUL_MAT
    if (!ggml_metal_op_concurrency_check(ctx, mm)) {
        ggml_metal_op_concurrency_reset(ctx);
    }

    ggml_metal_op_fwht_impl(ctx, mm, x, signs);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: MUL + MUL_MAT (signed FWHT)\n", __func__);
    }

    return 2;
}

int ggml_metal_op_pool_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t k1 = opts[2];
    const int32_t s0 = opts[3];
    const int32_t s1 = opts[4];
    const int32_t p0 = opts[5];
    const int32_t p1 = opts[6];

    const int64_t IH = op->src[0]->ne[1];
    const int64_t IW = op->src[0]->ne[0];

    const int64_t N  = op->ne[3];
    const int64_t OC = op->ne[2];
    const int64_t OH = op->ne[1];
    const int64_t OW = op->ne[0];

    const int64_t np = N * OC * OH * OW;

    ggml_metal_kargs_pool_2d args_pool_2d = {
        /* .k0 = */ k0,
        /* .k1 = */ k1,
        /* .s0 = */ s0,
        /* .s1 = */ s1,
        /* .p0 = */ p0,
        /* .p1 = */ p1,
        /* .IH = */ IH,
        /* .IW = */ IW,
        /* .OH = */ OH,
        /* .OW = */ OW,
        /* .np = */ np
    };

    auto pipeline = ggml_metal_library_get_pipeline_pool_2d(lib, op, op_pool);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_2d, sizeof(args_pool_2d), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

// the number of nodes from idx on that the fusion table fuses as pattern id into one dispatch, 1 if it fuses none
static int ggml_metal_op_try_fusion(ggml_metal_op_t ctx, int idx, ggml_metal_fusion_id id) {
    if (!ctx->use_fusion()) {
        return 1;
    }

    int n = 1;
    const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
    if (fusion == nullptr || ggml_metal_fusion_get_id(fusion) != id) {
        return 1;
    }

    ctx->count_fusions(fusion);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: %s to %s, %d nodes\n", __func__, ggml_op_name(ctx->node(idx)->op), ggml_op_name(ctx->node(idx + n - 1)->op), n);
    }

    return n;
}

static int ggml_metal_op_mul_mat_mma(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    // the MMA store adds the residual when the table fuses the ADD after this mat-mul
    const int  n_fuse   = ggml_metal_op_try_fusion(ctx, idx, GGML_METAL_FUSION_MUL_MAT_ADD);
    const bool fuse_add = n_fuse > 1;

    const ggml_tensor * dst = op;
    const ggml_tensor * res = dst;

    if (fuse_add) {
        dst = ctx->node(idx + n_fuse - 1);
        // the residual is the other operand of the ADD, by identity: it can itself be a MUL_MAT output
        res = dst->src[0] == op ? dst->src[1] : dst->src[0];
    }

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_mma_auto(lib, op, fuse_add);

    ggml_metal_kargs_mul_mv_ext args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne10  =*/ ne10,
        /*.ne11  =*/ ne11,
        /*.ne12  =*/ ne12,
        /*.nb10  =*/ nb10,
        /*.nb11  =*/ nb11,
        /*.nb12  =*/ nb12,
        /*.nb13  =*/ nb13,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.r2    =*/ (int16_t) (ne12/ne02),
        /*.r3    =*/ (int16_t) (ne13/ne03),
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst),        3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(res),        4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    const int rows0 = pipeline.nr0;
    const int rows1 = pipeline.nr1;

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + rows0 - 1)/rows0, (ne11 + rows1 - 1)/rows1, ne12*ne13, 32, pipeline.nsg, 1);

    return n_fuse;
}

// fused gate + up matrix-vector products followed by SWIGLU (see GGML_METAL_FUSION_MUL_MV_GLU)
// split:   up/gate MUL_MAT(_ID) + GLU
// stacked: ffn_gate_up_exps MUL_MAT_ID + GLU over its two halves (the VIEWs are elided)
static int ggml_metal_op_mul_mv_glu(ggml_metal_op_t ctx, int idx, int n_fuse) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    // encode_impl checks only the first and the last fused node; the second matmul reads its
    // own weights, which a pending producer may still be writing
    for (int i = 1; i < n_fuse; ++i) {
        if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);
            break;
        }
    }

    ggml_tensor * glu = ctx->node(idx + n_fuse - 1);

    const bool stacked = glu->src[0]->op == GGML_OP_VIEW;

    const ggml_tensor * up     = stacked ? ctx->node(idx) : glu->src[1];
    const ggml_tensor * gate_w = stacked ? up->src[0]     : glu->src[0]->src[0];

    const ggml_tensor * src0 = up->src[0];
    const ggml_tensor * src1 = up->src[1];
    const ggml_tensor * ids  = up->src[2];

    ggml_metal_buffer_id bid_up   = ggml_metal_get_buffer_id(src0);
    ggml_metal_buffer_id bid_gate = ggml_metal_get_buffer_id(gate_w);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(src1);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(glu);

    if (stacked) {
        bid_up.offs += (size_t) glu->ne[0] * src0->nb[1];
    }

    const int32_t ne00 = (int32_t) src0->ne[0];
    const int32_t ne01 = (int32_t) glu->ne[0];
    const int32_t ne02 = (int32_t) src0->ne[2];
    const int32_t ne10 = (int32_t) src1->ne[0];
    const int32_t ne11 = (int32_t) src1->ne[1];
    const int32_t ne12 = (int32_t) src1->ne[2];
    const int32_t ne13 = (int32_t) src1->ne[3];
    const int32_t ne0  = (int32_t) glu->ne[0];
    const int32_t ne1  = (int32_t) glu->ne[1];

    if (up->op == GGML_OP_MUL_MAT_ID) {
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_id_glu(lib, up);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;
        const size_t smem = pipeline.smem;

        const int32_t ne20 = (int32_t) ids->ne[0];
        const int32_t ne21 = (int32_t) ids->ne[1];

        ggml_metal_kargs_mul_mv_id args = {
            /*.nei0 =*/ ne20,
            /*.nei1 =*/ ne21,
            /*.nbi1 =*/ ids->nb[1],
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ src0->nb[0],
            /*.nb01 =*/ src0->nb[1],
            /*.nb02 =*/ src0->nb[2],
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.ne13 =*/ ne13,
            /*.nb10 =*/ src1->nb[0],
            /*.nb11 =*/ src1->nb[1],
            /*.nb12 =*/ src1->nb[2],
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nb1  =*/ glu->nb[1],
            /*.nr0  =*/ nr0,
            /*.scale_grouped =*/ 0,
            /*.nes0          =*/ 0,
            /*.nes1          =*/ 0,
            /*.nes2          =*/ 0,
            /*.nbs0          =*/ 0,
            /*.nbs1          =*/ 0,
            /*.nbs2          =*/ 0,
        };

        if (ggml_is_quantized(src0->type)) {
            GGML_ASSERT(ne00 >= nsg*nr0);
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_up,   1);
        ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ids), 4);
        ggml_metal_encoder_set_buffer  (enc, bid_gate, 5);

        const int64_t _ne1  = 1;
        const int64_t ne123 = (int64_t) ne20 * ne21;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 ||
                src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0 - 1)/(nr0), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0*nsg - 1)/(nr0*nsg), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        }
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_glu(lib, up);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;
        const size_t smem = pipeline.smem;

        const int32_t ne03 = (int32_t) src0->ne[3];
        const int16_t r2 = (int16_t) (ne12 / (int32_t) src0->ne[2]);
        const int16_t r3 = (int16_t) (ne13 / ne03);

        ggml_metal_kargs_mul_mv args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ src0->nb[0],
            /*.nb01 =*/ src0->nb[1],
            /*.nb02 =*/ src0->nb[2],
            /*.nb03 =*/ src0->nb[3],
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ src1->nb[0],
            /*.nb11 =*/ src1->nb[1],
            /*.nb12 =*/ src1->nb[2],
            /*.nb13 =*/ src1->nb[3],
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nr0  =*/ nr0,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
            /*.eps  =*/ 0.0f,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_up,   1);
        ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);
        ggml_metal_encoder_set_buffer  (enc, bid_gate, 4);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 ||
                src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0 - 1)/(nr0)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0*nsg - 1)/(nr0*nsg)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
        }
    }

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 0) {
        GGML_LOG_INFO("%s: fused %s + GLU (%s)\n",
                __func__, ggml_op_name(up->op), stacked ? "stacked" : "split");
    }

    return n_fuse;
}

// encode the mat-vec op into dst; the store adds res when dst is a fused ADD.
// with norm (a fused RMS_NORM + MUL) the kernel reads the norm input row and normalizes it with the weight w
static bool ggml_metal_bitnet_poc_target(const ggml_tensor * op) {
    return op->op == GGML_OP_MUL_MAT &&
        ggml_metal_bitnet_weight_name(op->src[0], ggml_metal_bitnet_layer(op->name, "Qcur-"), "attn_q");
}

static bool ggml_metal_bitnet_poc_flag(const char * name) {
    const char * value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

static bool ggml_metal_bitnet_poc_layout(const ggml_tensor * t, ggml_type type,
        const int64_t (&ne)[4], const size_t (&nb)[4]) {
    return t && !t->view_src && t->view_offs == 0 && t->buffer && t->type == type &&
        std::memcmp(t->ne, ne, sizeof(ne)) == 0 && std::memcmp(t->nb, nb, sizeof(nb)) == 0;
}

static bool ggml_metal_bitnet_actquant_q4_contract(const ggml_tensor * op,
        int rows, int groups, bool guard_only, bool trace) {
    const bool enabled = ggml_metal_bitnet_poc_flag("LLAMA_BITNET_ACTQUANT");
    const bool permission = ggml_metal_bitnet_poc_flag("LLAMA_BITNET_ACTQUANT_Q4");
    if (!enabled && !permission) {
        return true;
    }
    const ggml_tensor * input = op->src[1];
    const char * layers = std::getenv("GGML_METAL_BITNET_LAYERS");
    float eps = 0.0f;
    if (input && input->op == GGML_OP_ACT_QUANT) {
        std::memcpy(&eps, input->op_params, sizeof(eps));
    }
    bool valid = enabled && permission && !guard_only && rows == 4 && groups == 1 &&
        ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_POC") && layers && std::strcmp(layers, "0") == 0 &&
        op->op == GGML_OP_MUL_MAT && std::strcmp(op->name, "Qcur-0") == 0 &&
        ggml_metal_bitnet_weight_name(op->src[0], 0, "attn_q") && input && input->op == GGML_OP_ACT_QUANT &&
        std::strcmp(input->name, "actq_qkv_out-0") == 0 && input->type == GGML_TYPE_F32 &&
        input->src[0] && input->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(input->src[0]) &&
        ggml_are_same_shape(input, input->src[0]) && eps == 1e-5f;
    for (const char * name : { "GGML_METAL_BITNET_Q_GUARD_ONLY", "GGML_METAL_BITNET_FFN",
            "GGML_METAL_BITNET_FFN_MECHANISM", "GGML_METAL_BITNET_FFN_ABLATION" }) {
        const char * value = std::getenv(name);
        if (value && value[0] != '\0' && std::strcmp(value, "0") != 0) {
            valid = false;
        }
    }
    if (trace) {
        if (valid) {
            std::fprintf(stderr, "BITNET_ACTQUANT_Q4_GUARD node=%s source=%s source_op=ACT_QUANT eps=%.9g rows=%d simd=%d layer=0 checked=1\n",
                    op->name, input->name, eps, rows, groups);
        } else {
            std::fprintf(stderr, "BITNET_Q_FALLBACK node=%s layer=0 reason=actquant_contract permission=%d rows=%d simd=%d\n",
                    op->name, permission, rows, groups);
        }
    }
    return valid;
}

static bool ggml_metal_bitnet_poc_disjoint(ggml_metal_buffer_id a, size_t a_bytes,
        ggml_metal_buffer_id b, size_t b_bytes) {
    return a.metal != b.metal || a.offs + a_bytes <= b.offs || b.offs + b_bytes <= a.offs;
}

enum class bitnet_ffn_projection { none, gate, up, down };

static bitnet_ffn_projection ggml_metal_bitnet_ffn_projection(const ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT || !op->src[0]) {
        return bitnet_ffn_projection::none;
    }
    if (ggml_metal_bitnet_weight_name(op->src[0], ggml_metal_bitnet_layer(op->name, "ffn_gate-"), "ffn_gate")) {
        return bitnet_ffn_projection::gate;
    }
    if (ggml_metal_bitnet_weight_name(op->src[0], ggml_metal_bitnet_layer(op->name, "ffn_up-"), "ffn_up")) {
        return bitnet_ffn_projection::up;
    }
    if (ggml_metal_bitnet_weight_name(op->src[0], ggml_metal_bitnet_layer(op->name, "ffn_down-"), "ffn_down")) {
        return bitnet_ffn_projection::down;
    }
    return bitnet_ffn_projection::none;
}

static int ggml_metal_bitnet_ffn_layer(const ggml_tensor * op) {
    const auto projection = ggml_metal_bitnet_ffn_projection(op);
    switch (projection) {
        case bitnet_ffn_projection::gate: return ggml_metal_bitnet_layer(op->name, "ffn_gate-");
        case bitnet_ffn_projection::up:   return ggml_metal_bitnet_layer(op->name, "ffn_up-");
        case bitnet_ffn_projection::down: return ggml_metal_bitnet_layer(op->name, "ffn_down-");
        default: return -1;
    }
}

static const char * ggml_metal_bitnet_ffn_stock_route(bitnet_ffn_projection projection) {
    switch (projection) {
        case bitnet_ffn_projection::gate: return "ffn-stock-gate";
        case bitnet_ffn_projection::up:   return "ffn-stock-up";
        case bitnet_ffn_projection::down: return "ffn-stock-down";
        default: return nullptr;
    }
}

static bool ggml_metal_bitnet_ffn_mode(const char * expected) {
    const char * mode = std::getenv("GGML_METAL_BITNET_FFN");
    return mode && std::strcmp(mode, expected) == 0;
}

static bool ggml_metal_bitnet_ffn_reject(const ggml_tensor * node, const char * reason, bool trace) {
    if (trace) {
        std::fprintf(stderr, "BITNET_FFN_FALLBACK node=%s n=%lld reason=%s layer=%d\n",
                node->name, (long long) node->ne[1], reason, ggml_metal_bitnet_ffn_layer(node));
    }
    return false;
}

static bool ggml_metal_bitnet_ffn_vector(const ggml_tensor * tensor, int64_t elements) {
    const int64_t ne[] = { elements, 1, 1, 1 };
    const size_t bytes = (size_t) elements * sizeof(float);
    const size_t nb[] = { sizeof(float), bytes, bytes, bytes };
    return ggml_metal_bitnet_poc_layout(tensor, GGML_TYPE_F32, ne, nb);
}

static bool ggml_metal_bitnet_k_shadow_step(const char * step) {
    return step && std::atoi(step)>0 && (ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_K_SHADOW_ALL") ||
            std::strcmp(step,"1")==0 || std::strcmp(step,"16")==0 || std::strcmp(step,"32")==0);
}

static void ggml_metal_bitnet_k_save_weight(const char * directory, int layer, const char * role, const ggml_tensor * tensor) {
    static std::mutex mutex;
    static std::set<std::string> saved;
    std::lock_guard<std::mutex> lock(mutex);
    const auto folder=std::filesystem::path(directory)/"weights"/("layer-"+std::to_string(layer));
    const auto file=folder/role;
    if (saved.count(file.string())) {return;}
    std::filesystem::create_directories(folder);
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor,bytes.data(),0,bytes.size());
    FILE * output=std::fopen(file.c_str(),"wb");
    if (!output) {GGML_ABORT("K weight diagnostic open failed");}
    const bool wrote=std::fwrite(bytes.data(),1,bytes.size(),output)==bytes.size();
    const bool closed=std::fclose(output)==0;
    if (!wrote || !closed) {GGML_ABORT("K weight diagnostic write failed");}
    saved.insert(file.string());
    std::fprintf(stderr,"BITNET_FFN_K_WEIGHT path=%s node=%s bytes=%zu actual_tensor_readback=1\n",file.c_str(),tensor->name,bytes.size());
}

static bool ggml_metal_bitnet_ffn_expand_layout(const ggml_tensor * op) {
    const int64_t ne[] = { 2560, 6912, 1, 1 };
    const size_t nb[] = { 66, 660, 4561920, 4561920 };
    return ggml_metal_bitnet_poc_layout(op->src[0], GGML_TYPE_TQ2_0, ne, nb) &&
           ggml_metal_bitnet_ffn_vector(op->src[1], 2560) && ggml_metal_bitnet_ffn_vector(op, 6912) &&
           ggml_metal_bitnet_layer_name(op->src[1], "ffn_norm-", ggml_metal_bitnet_ffn_layer(op));
}

static bool ggml_metal_bitnet_q_readback_active() {
    const char * directory = std::getenv("GGML_METAL_BITNET_QREADBACK_DIR");
    const char * step_text = std::getenv("GGML_METAL_BITNET_QREADBACK_STEP");
    if (!directory || !directory[0] || !step_text ||
            !ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_DECODE")) {
        return false;
    }
    char * end = nullptr;
    const long step = std::strtol(step_text, &end, 10);
    if (end == step_text || *end || step < 0 || step > 32) {GGML_ABORT("Q readback step must be 0 through 32");}
    if (step == 0) {return false;}
    const char * route = std::getenv("GGML_METAL_BITNET_QREADBACK_ROUTE");
    if (!route || (std::strcmp(route, "S") && std::strcmp(route, "Q4") && std::strcmp(route, "QB"))) {
        GGML_ABORT("Q readback route must be S, Q4 or QB");
    }
    if (std::getenv("GGML_METAL_BITNET_FFN") || std::getenv("GGML_METAL_BITNET_FFN_MECHANISM")) {
        GGML_ABORT("Q readback is restricted to Q-only runs");
    }
    const bool custom = ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_POC");
    const bool guard_only = ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_Q_GUARD_ONLY");
    if (custom != (std::strcmp(route, "Q4") == 0) || guard_only != (std::strcmp(route, "QB") == 0)) {
        GGML_ABORT("Q readback route and actual custom/guard switches differ");
    }
    if (custom || guard_only) {
        const char * rows = std::getenv("GGML_METAL_BITNET_POC_ROWS_PER_SIMD");
        const char * groups = std::getenv("GGML_METAL_BITNET_POC_SIMD_PER_GROUP");
        if (!rows || std::strcmp(rows, "4") || !groups || std::strcmp(groups, "1")) {
            GGML_ABORT("Q4/QB readback requires the four-row one-SIMD configuration");
        }
    }
    return true;
}

static void ggml_metal_bitnet_q_readback_copy(ggml_metal_op_t ctx, const ggml_tensor * tensor,
        const char * role, const char * phase, int consumed, const std::filesystem::path & path) {
    const bool weight = std::strcmp(role, "w") == 0;
    const size_t bytes = weight ? 1689600 : 2560*sizeof(float);
    if (!tensor || ggml_nbytes(tensor) != bytes ||
            (weight ? tensor->type != GGML_TYPE_TQ2_0 : !ggml_metal_bitnet_ffn_vector(tensor, 2560))) {
        GGML_ABORT("Q readback tensor contract failed for %s", role);
    }
    const auto binding = ggml_metal_get_buffer_id(tensor);
    if (binding.offs % 4 || !ggml_metal_buffer_contains(binding, bytes)) {
        GGML_ABORT("Q readback bounds/alignment failed for %s", role);
    }
    const auto pipeline = ggml_metal_library_get_bitnet_ffn_mechanism_pipeline(ctx->lib, 3);
    if (!pipeline.pipeline) {GGML_ABORT("Q readback copy pipeline is unavailable");}
    const size_t copy_chunk_bytes = 6912*sizeof(float);
    const size_t copy_dispatches = (bytes + copy_chunk_bytes - 1)/copy_chunk_bytes;
    const size_t copy_last_bytes = bytes - (copy_dispatches - 1)*copy_chunk_bytes;
    std::fprintf(stderr,
            "BITNET_Q_READBACK_SHADOW route=%s step=%s role=%s phase=%s node=%s op=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] buffer=%p offset=%zu bytes=%zu bounds=1 consumed=%d copy_pipeline=%s copy_dispatches=%zu copy_grid=[%zu,1,1] copy_last_grid=[%zu,1,1] copy_threads=[256,1,1] path=%s\n",
            std::getenv("GGML_METAL_BITNET_QREADBACK_ROUTE"), std::getenv("GGML_METAL_BITNET_QREADBACK_STEP"), role, phase,
            tensor->name, ggml_op_name(tensor->op), ggml_type_name(tensor->type),
            (long long)tensor->ne[0], (long long)tensor->ne[1], (long long)tensor->ne[2], (long long)tensor->ne[3],
            tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3], binding.metal, binding.offs, bytes, consumed,
            ggml_metal_pipeline_name(pipeline), copy_dispatches,
            std::min(bytes, copy_chunk_bytes)/(256*sizeof(float)), copy_last_bytes/(256*sizeof(float)), path.c_str());
    // Copies remain in the original encoder/command buffer. Barriers protect values
    // before later in-place operations; diagnostic scheduling is not timing evidence.
    ggml_metal_op_concurrency_reset(ctx);
    if (!ctx->mem_ranges) {ggml_metal_encoder_memory_barrier(ctx->enc);}
    if (!ggml_metal_encoder_bitnet_shadow(ctx->enc, pipeline, binding, bytes, path.c_str())) {
        GGML_ABORT("Q readback GPU copy failed for %s", role);
    }
    ggml_metal_op_concurrency_reset(ctx);
    if (!ctx->mem_ranges) {ggml_metal_encoder_memory_barrier(ctx->enc);}
}

static void ggml_metal_bitnet_q_readback_before(ggml_metal_op_t ctx, int idx) {
    const auto * node = ctx->node(idx);
    if (!ggml_metal_bitnet_layer_name(node, "Qcur-", 0)) {return;}
    // Later graph transforms can retain the name; identify the actual producer by op.
    if (node->op != GGML_OP_MUL_MAT) {
        std::fprintf(stderr, "BITNET_Q_READBACK_SKIP route=%s step=%s node=%s op=%s ne=[%lld,%lld,%lld,%lld]\n",
                std::getenv("GGML_METAL_BITNET_QREADBACK_ROUTE"), std::getenv("GGML_METAL_BITNET_QREADBACK_STEP"),
                node->name, ggml_op_name(node->op), (long long)node->ne[0], (long long)node->ne[1],
                (long long)node->ne[2], (long long)node->ne[3]);
        return;
    }
    const int64_t weight_ne[] = {2560, 2560, 1, 1};
    const size_t weight_nb[] = {66, 660, 1689600, 1689600};
    const int64_t vector_ne[] = {2560, 1, 1, 1};
    const size_t vector_nb[] = {4, 10240, 10240, 10240};
    if (!ggml_metal_bitnet_weight_name(node->src[0], 0, "attn_q") ||
            !ggml_metal_bitnet_poc_layout(node->src[0], GGML_TYPE_TQ2_0, weight_ne, weight_nb) ||
            !ggml_metal_bitnet_poc_layout(node->src[1], GGML_TYPE_F32, vector_ne, vector_nb) ||
            !ggml_metal_bitnet_poc_layout(node, GGML_TYPE_F32, vector_ne, vector_nb)) {
        GGML_ABORT("Q readback target is not the audited Qcur-0 contract");
    }
    const auto root = std::filesystem::path(std::getenv("GGML_METAL_BITNET_QREADBACK_DIR"));
    const auto file = root/"weights"/"w.tq2";
    static std::mutex mutex;
    static std::set<std::string> queued;
    {
        std::lock_guard<std::mutex> lock(mutex);
        const bool fresh = queued.count(file.string()) == 0;
        if (fresh) {
            if (std::filesystem::exists(file)) {GGML_ABORT("Q readback refuses an existing weight file");}
            ggml_metal_bitnet_q_readback_copy(ctx, node->src[0], "w", "before-Q", 0, file);
            queued.insert(file.string());
        }
        const auto binding = ggml_metal_get_buffer_id(node->src[0]);
        std::fprintf(stderr, "BITNET_Q_READBACK_WEIGHT route=%s step=%s node=%s buffer=%p offset=%zu bytes=1689600 copied=%d cached=%d path=%s\n",
                std::getenv("GGML_METAL_BITNET_QREADBACK_ROUTE"), std::getenv("GGML_METAL_BITNET_QREADBACK_STEP"),
                node->src[0]->name, binding.metal, binding.offs, fresh, !fresh, file.c_str());
    }
    const auto folder = root/(std::string("step-")+std::getenv("GGML_METAL_BITNET_QREADBACK_STEP"));
    ggml_metal_bitnet_q_readback_copy(ctx, node->src[1], "x", "before-Q", 0, folder/"x.f32");
}

static void ggml_metal_bitnet_q_readback_after(ggml_metal_op_t ctx, int idx, int consumed) {
    const auto * node = ctx->node(idx);
    const auto * last = ctx->node(idx+consumed-1);
    const auto root = std::filesystem::path(std::getenv("GGML_METAL_BITNET_QREADBACK_DIR"));
    const char * step = std::getenv("GGML_METAL_BITNET_QREADBACK_STEP");
    const auto folder = root/(std::string("step-")+step);
    if (node->op == GGML_OP_MUL_MAT && ggml_metal_bitnet_layer_name(node, "Qcur-", 0)) {
        if (consumed != 1 || last != node) {GGML_ABORT("Q output is not independently materialized before RoPE");}
        ggml_metal_bitnet_q_readback_copy(ctx, node, "y", "after-Q-before-RoPE", consumed, folder/"y.f32");
    }
    if (last->op == GGML_OP_ADD && (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_ADD) &&
            ggml_metal_bitnet_layer_name(last, "l_out-", 0) &&
            (std::strcmp(step, "1") == 0 || std::strcmp(step, "16") == 0 || std::strcmp(step, "32") == 0)) {
        const auto path = folder/"l_out.f32";
        const auto binding = ggml_metal_get_buffer_id(last);
        struct captured_output {
            std::string path;
            const ggml_tensor * tensor;
            ggml_metal_buffer_id binding;
        };
        static std::mutex mutex;
        static std::vector<captured_output> captured;
        std::lock_guard<std::mutex> lock(mutex);
        const auto found = std::find_if(captured.begin(), captured.end(), [&](const captured_output & value) {
            return value.path == path.string();
        });
        if (found != captured.end()) {
            if (found->tensor != last || found->binding.metal != binding.metal || found->binding.offs != binding.offs) {
                GGML_ABORT("Q readback found different l_out-0 producers for one step");
            }
            std::fprintf(stderr, "BITNET_Q_READBACK_ALIAS_SKIP route=%s step=%s role=l_out node=%s duplicate=1 path=%s\n",
                    std::getenv("GGML_METAL_BITNET_QREADBACK_ROUTE"), step, last->name, path.c_str());
            return;
        }
        ggml_metal_bitnet_q_readback_copy(ctx, last, "l_out", "after-layer", consumed, path);
        captured.push_back({path.string(), last, binding});
    }
}

static bool ggml_metal_bitnet_numerics_active() {
    const char * directory=std::getenv("GGML_METAL_BITNET_FFN_NUMERICS_DIR");
    const char * route=std::getenv("GGML_METAL_BITNET_FFN_NUMERICS_ROUTE");
    const char * text=std::getenv("GGML_METAL_BITNET_FFN_STEP");
    if (!directory || !directory[0] || !ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_DECODE") || !text) {return false;}
    char * end=nullptr;
    const long step=std::strtol(text,&end,10);
    if (*end || step<1 || step>32) {return false;}
    if (!route || (std::strcmp(route,"S") && std::strcmp(route,"F8"))) {GGML_ABORT("FFN numerics route must be S or F8");}
    if (std::getenv("GGML_METAL_BITNET_FFN_MECHANISM") || ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_POC") ||
            (std::strcmp(route,"S")==0 && std::getenv("GGML_METAL_BITNET_FFN")) ||
            (std::strcmp(route,"F8")==0 && !ggml_metal_bitnet_ffn_mode("fused"))) {GGML_ABORT("FFN numerics route flags differ");}
    return ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_NUMERICS_ALL") || step==1 || step==16 || step==32;
}

static void ggml_metal_bitnet_numerics_copy(ggml_metal_op_t ctx,const ggml_tensor * tensor,int layer,const char * role,const char * phase,int consumed) {
    const int64_t n=(std::strcmp(role,"x")==0 || std::strcmp(role,"down")==0 || std::strcmp(role,"residual")==0 || std::strcmp(role,"l_out")==0) ? 2560 : 6912;
    if (!ggml_metal_bitnet_ffn_vector(tensor,n)) {GGML_ABORT("FFN numerics vector contract failed for %s",role);}
    const auto binding=ggml_metal_get_buffer_id(tensor);
    if (binding.offs%4 || !ggml_metal_buffer_contains(binding,ggml_nbytes(tensor))) {GGML_ABORT("FFN numerics bounds failed");}
    const auto pipeline=ggml_metal_library_get_bitnet_ffn_mechanism_pipeline(ctx->lib,3);
    if (!pipeline.pipeline) {GGML_ABORT("FFN numerics copy pipeline unavailable");}
    char path[4096];
    std::snprintf(path,sizeof(path),"%s/step-%s/layer-%d/%s.f32",std::getenv("GGML_METAL_BITNET_FFN_NUMERICS_DIR"),std::getenv("GGML_METAL_BITNET_FFN_STEP"),layer,role);
    std::fprintf(stderr,"BITNET_F8_NUMERIC_SHADOW route=%s layer=%d step=%s role=%s phase=%s node=%s op=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] buffer=%p offset=%zu bytes=%zu bounds=1 materialized=1 consumed=%d path=%s\n",
            std::getenv("GGML_METAL_BITNET_FFN_NUMERICS_ROUTE"),layer,std::getenv("GGML_METAL_BITNET_FFN_STEP"),role,phase,tensor->name,ggml_op_name(tensor->op),ggml_type_name(tensor->type),
            (long long)tensor->ne[0],(long long)tensor->ne[1],(long long)tensor->ne[2],(long long)tensor->ne[3],tensor->nb[0],tensor->nb[1],tensor->nb[2],tensor->nb[3],binding.metal,binding.offs,ggml_nbytes(tensor),consumed,path);
    ggml_metal_op_concurrency_reset(ctx);
    if (!ctx->mem_ranges) {ggml_metal_encoder_memory_barrier(ctx->enc);}
    if (!ggml_metal_encoder_bitnet_shadow(ctx->enc,pipeline,binding,ggml_nbytes(tensor),path)) {GGML_ABORT("FFN numerics copy failed");}
    ggml_metal_op_concurrency_reset(ctx);
    if (!ctx->mem_ranges) {ggml_metal_encoder_memory_barrier(ctx->enc);}
}

static void ggml_metal_bitnet_numerics_weight(const ggml_tensor * tensor,int layer,const char * role) {
    const bool gamma=std::strcmp(role,"subnorm.f32")==0;
    if (!tensor || (gamma ? !ggml_metal_bitnet_ffn_vector(tensor,6912) : tensor->type!=GGML_TYPE_TQ2_0 || ggml_nbytes(tensor)!=4561920)) {GGML_ABORT("FFN numerics weight contract failed");}
    const auto binding=ggml_metal_get_buffer_id(tensor);
    if (!ggml_metal_buffer_contains(binding,ggml_nbytes(tensor))) {GGML_ABORT("FFN numerics static bounds failed");}
    const char * shared=std::getenv("GGML_METAL_BITNET_FFN_NUMERICS_WEIGHTS_DIR");
    const auto root=shared && shared[0] ? std::filesystem::path(shared) : std::filesystem::path(std::getenv("GGML_METAL_BITNET_FFN_NUMERICS_DIR"))/"weights";
    const auto folder=root/("layer-"+std::to_string(layer));const auto file=folder/role;
    static std::mutex mutex;static std::set<std::string> saved;
    std::lock_guard<std::mutex> lock(mutex);
    const bool fresh=saved.count(file.string())==0;
    if (fresh) {
        std::filesystem::create_directories(folder);std::vector<uint8_t> bytes(ggml_nbytes(tensor));ggml_backend_tensor_get(tensor,bytes.data(),0,bytes.size());
        FILE * output=std::fopen(file.c_str(),"wb");if (!output) {GGML_ABORT("FFN numerics weight open failed");}
        const bool wrote=std::fwrite(bytes.data(),1,bytes.size(),output)==bytes.size();const bool closed=std::fclose(output)==0;
        if (!wrote || !closed) {GGML_ABORT("FFN numerics weight write failed");}saved.insert(file.string());
    }
    std::fprintf(stderr,"BITNET_F8_NUMERIC_WEIGHT route=%s layer=%d step=%s role=%s node=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] buffer=%p offset=%zu bytes=%zu bounds=1 actual_readback=%d cached_immutable_model=%d path=%s\n",
            std::getenv("GGML_METAL_BITNET_FFN_NUMERICS_ROUTE"),layer,std::getenv("GGML_METAL_BITNET_FFN_STEP"),role,tensor->name,ggml_type_name(tensor->type),
            (long long)tensor->ne[0],(long long)tensor->ne[1],(long long)tensor->ne[2],(long long)tensor->ne[3],tensor->nb[0],tensor->nb[1],tensor->nb[2],tensor->nb[3],binding.metal,binding.offs,ggml_nbytes(tensor),fresh,!fresh,file.c_str());
}

static void ggml_metal_bitnet_numerics_dispatch(ggml_metal_op_t ctx,const ggml_tensor * node,const ggml_tensor * output,ggml_metal_pipeline_with_params pipeline,int gx,int gy,int gz,int tx,int ty,int add,int norm,int consumed) {
    if (!ctx->numerics_capture) {return;}
    int layer=ggml_metal_bitnet_ffn_layer(node);
    if (layer<0) {for (const char * prefix : {"ffn_sub_out-","ffn_sub_norm-","l_out-"}) {layer=ggml_metal_bitnet_layer(node->name,prefix);if (layer>=0) {break;}}}
    if (layer<0 && node->op==GGML_OP_RMS_NORM && node->src[0]) {layer=ggml_metal_bitnet_layer(node->src[0]->name,"ffn_sub_out-");}
    if (layer<0) {return;}
    std::fprintf(stderr,"BITNET_F8_NUMERIC_DISPATCH route=%s layer=%d step=%s node=%s output=%s op=%s pipeline=%s grid=[%d,%d,%d] threads=[%d,%d,1] add=%d norm=%d consumed=%d src0=%s src1=%s\n",
            std::getenv("GGML_METAL_BITNET_FFN_NUMERICS_ROUTE"),layer,std::getenv("GGML_METAL_BITNET_FFN_STEP"),node->name,output->name,ggml_op_name(node->op),ggml_metal_pipeline_name(pipeline),gx,gy,gz,tx,ty,add,norm,consumed,node->src[0]?node->src[0]->name:"none",node->src[1]?node->src[1]->name:"none");
}

static void ggml_metal_bitnet_numerics_before(ggml_metal_op_t ctx,int idx) {
    if (!ctx->numerics_capture) {return;}
    const auto * node=ctx->node(idx);const auto projection=ggml_metal_bitnet_ffn_projection(node);const int layer=ggml_metal_bitnet_ffn_layer(node);
    if (projection==bitnet_ffn_projection::gate || projection==bitnet_ffn_projection::up) {
        if (!ggml_metal_bitnet_ffn_expand_layout(node)) {GGML_ABORT("FFN numerics projection layout failed");}
        ggml_metal_bitnet_numerics_weight(node->src[0],layer,projection==bitnet_ffn_projection::gate?"ffn_gate.tq2":"ffn_up.tq2");
        if (projection==bitnet_ffn_projection::gate) {ggml_metal_bitnet_numerics_copy(ctx,node->src[1],layer,"x","before-gate",0);}
    } else if (projection==bitnet_ffn_projection::down) {
        const auto * input=node->src[1];
        if (!ggml_metal_bitnet_ffn_vector(node,2560) || !ggml_metal_bitnet_layer_name(input,"ffn_sub_norm-",layer) || input->op!=GGML_OP_MUL || !input->src[0] || input->src[0]->op!=GGML_OP_RMS_NORM || !ggml_metal_bitnet_weight_name(input->src[1],layer,"ffn_sub_norm")) {GGML_ABORT("FFN numerics down/norm edge failed");}
        ggml_metal_bitnet_numerics_weight(node->src[0],layer,"ffn_down.tq2");ggml_metal_bitnet_numerics_weight(input->src[1],layer,"subnorm.f32");
        float eps;std::memcpy(&eps,input->src[0]->op_params,sizeof(eps));
        std::fprintf(stderr,"BITNET_F8_NUMERIC_NORM route=%s layer=%d step=%s input=%s weighted_output=%s gamma=%s epsilon=%.12g\n",std::getenv("GGML_METAL_BITNET_FFN_NUMERICS_ROUTE"),layer,std::getenv("GGML_METAL_BITNET_FFN_STEP"),input->src[0]->src[0]->name,input->name,input->src[1]->name,eps);
        ggml_metal_bitnet_numerics_copy(ctx,input,layer,"down-input","before-down",0);
    }
    const int residual_layer=ggml_metal_bitnet_layer(node->name,"l_out-");
    if (residual_layer>=0) {
        if (node->op!=GGML_OP_ADD) {GGML_ABORT("FFN numerics l_out is not residual ADD");}
        const ggml_tensor * down=node->src[0];const ggml_tensor * residual=node->src[1];
        if (ggml_metal_bitnet_ffn_projection(down)!=bitnet_ffn_projection::down) {std::swap(down,residual);}
        if (ggml_metal_bitnet_ffn_projection(down)!=bitnet_ffn_projection::down || ggml_metal_bitnet_ffn_layer(down)!=residual_layer || !ggml_metal_bitnet_layer_name(residual,"ffn_inp-",residual_layer)) {GGML_ABORT("FFN numerics residual graph edge failed");}
        std::fprintf(stderr,"BITNET_F8_NUMERIC_RESIDUAL route=%s layer=%d step=%s output=%s down=%s residual=%s\n",std::getenv("GGML_METAL_BITNET_FFN_NUMERICS_ROUTE"),residual_layer,std::getenv("GGML_METAL_BITNET_FFN_STEP"),node->name,down->name,residual->name);
        ggml_metal_bitnet_numerics_copy(ctx,residual,residual_layer,"residual","before-residual-add",0);
    }
}

static void ggml_metal_bitnet_numerics_after(ggml_metal_op_t ctx,int idx,int consumed) {
    if (!ctx->numerics_capture) {return;}
    const auto * node=ctx->node(idx);const auto * last=ctx->node(idx+consumed-1);const auto projection=ggml_metal_bitnet_ffn_projection(node);const int layer=ggml_metal_bitnet_ffn_layer(node);
    if (projection==bitnet_ffn_projection::gate && consumed==5 && ggml_metal_bitnet_layer_name(last,"ffn_sub_out-",layer)) {
        const auto * up=ctx->node(idx+2);ggml_metal_bitnet_numerics_weight(up->src[0],layer,"ffn_up.tq2");
        for (const char * role : {"gate","up"}) {std::fprintf(stderr,"BITNET_F8_NUMERIC_MATERIALIZATION route=%s layer=%d step=%s role=%s materialized=0 source=hidden-fused-projection consumed=5 replay_required=1\n",std::getenv("GGML_METAL_BITNET_FFN_NUMERICS_ROUTE"),layer,std::getenv("GGML_METAL_BITNET_FFN_STEP"),role);}
        ggml_metal_bitnet_numerics_copy(ctx,last,layer,"activation","after-fused-five",consumed);return;
    }
    if (projection==bitnet_ffn_projection::gate || projection==bitnet_ffn_projection::up || projection==bitnet_ffn_projection::down) {
        if (consumed!=1 || last!=node) {GGML_ABORT("FFN numerics projection not independently materialized");}
        ggml_metal_bitnet_numerics_copy(ctx,node,layer,projection==bitnet_ffn_projection::gate?"gate":projection==bitnet_ffn_projection::up?"up":"down","after-projection",consumed);return;
    }
    const int activation_layer=ggml_metal_bitnet_layer(last->name,"ffn_sub_out-");
    if (activation_layer>=0) {ggml_metal_bitnet_numerics_copy(ctx,last,activation_layer,"activation","after-activation",consumed);return;}
    const int residual_layer=ggml_metal_bitnet_layer(node->name,"l_out-");
    if (residual_layer>=0) {
        if (consumed!=1 || last!=node) {GGML_ABORT("FFN numerics l_out not independently materialized");}
        ggml_metal_bitnet_numerics_copy(ctx,node,residual_layer,"l_out","after-residual-add",consumed);
    }
}

static bool ggml_metal_bitnet_ffn_bindings(const ggml_tensor * const * tensors, ggml_metal_buffer_id * bindings, int count) {
    for (int i = 0; i < count; ++i) {
        bindings[i] = ggml_metal_get_buffer_id(tensors[i]);
        if (bindings[i].offs % 32 != 0 || !ggml_metal_buffer_contains(bindings[i], ggml_nbytes(tensors[i]))) {
            return false;
        }
        for (int j = 0; j < i; ++j) {
            if (!ggml_metal_bitnet_poc_disjoint(bindings[j], ggml_nbytes(tensors[j]), bindings[i], ggml_nbytes(tensors[i]))) {
                return false;
            }
        }
    }
    return true;
}

static bool ggml_metal_try_bitnet_ffn_split(ggml_metal_op_t ctx, const ggml_tensor * op,
        bitnet_ffn_projection projection, bool trace) {
    const int layer = ggml_metal_bitnet_ffn_layer(op);
    if (!ggml_metal_bitnet_selected_layer(layer)) {
        return ggml_metal_bitnet_ffn_reject(op, "layer-selection", trace);
    }
    if (!ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_DECODE") || !ggml_metal_bitnet_ffn_expand_layout(op)) {
        return ggml_metal_bitnet_ffn_reject(op, "decode-or-layout", trace);
    }
    const ggml_tensor * tensors[] = { op->src[0], op->src[1], op };
    ggml_metal_buffer_id bindings[3];
    if (!ggml_metal_bitnet_ffn_bindings(tensors, bindings, 3)) {
        return ggml_metal_bitnet_ffn_reject(op, "bindings", trace);
    }
    const auto pipeline = ggml_metal_library_get_bitnet_ffn_pipeline(ctx->lib, false);
    if (!pipeline.pipeline) {
        return ggml_metal_bitnet_ffn_reject(op, "pipeline", trace);
    }
    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    for (int i = 0; i < 3; ++i) {
        ggml_metal_encoder_set_buffer(ctx->enc, bindings[i], i);
    }
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, 432, 1, 1, 32, 2, 1);
    if (trace) {
        const char * route = projection == bitnet_ffn_projection::gate ? "ffn-c-gate" : "ffn-c-up";
        std::fprintf(stderr, "BITNET_MULTILAYER_BIND route=%s layer=%d node=%s weight=%s w=%p+%zu x=%p+%zu y=%p+%zu bounds=1 disjoint=1\n",
                route, layer, op->name, op->src[0]->name, bindings[0].metal, bindings[0].offs,
                bindings[1].metal, bindings[1].offs, bindings[2].metal, bindings[2].offs);
        ggml_metal_encoder_trace_bitnet(ctx->enc, pipeline,
                route, 1, 432, 1, 1, 2);
        ggml_metal_encoder_trace_bitnet_node(ctx->enc, pipeline, route, layer, op, op, 432, 1, 1, 32, 2);
    }
    return true;
}

static bool ggml_metal_try_bitnet_ffn_fused(ggml_metal_op_t ctx, int idx, bool trace) {
    const auto * gate = ctx->node(idx);
    const int layer = ggml_metal_bitnet_ffn_layer(gate);
    if (!ggml_metal_bitnet_selected_layer(layer)) {
        return ggml_metal_bitnet_ffn_reject(gate, "layer-selection", trace);
    }
    if (gate->ne[1] != 1 || !ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_DECODE")) {
        return ggml_metal_bitnet_ffn_reject(gate, "decode-or-token-count", trace);
    }
    if (ctx->n_nodes() - idx < 5) {
        return ggml_metal_bitnet_ffn_reject(gate, "encoder-slice", trace);
    }
    const auto * relu = ctx->node(idx + 1);
    const auto * up   = ctx->node(idx + 2);
    const auto * sqr  = ctx->node(idx + 3);
    const auto * out  = ctx->node(idx + 4);
    if (relu->op != GGML_OP_UNARY || ggml_get_unary_op(relu) != GGML_UNARY_OP_RELU || relu->src[0] != gate ||
        ggml_metal_bitnet_ffn_projection(up) != bitnet_ffn_projection::up ||
        ggml_metal_bitnet_ffn_layer(up) != layer || up->src[1] != gate->src[1] ||
        sqr->op != GGML_OP_SQR || sqr->src[0] != relu ||
        out->op != GGML_OP_MUL || out->src[0] != sqr || out->src[1] != up ||
        !ggml_metal_bitnet_layer_name(out, "ffn_sub_out-", layer) ||
        !ggml_metal_bitnet_layer_name(relu, "ffn_relu-", layer) ||
        !ggml_metal_bitnet_layer_name(sqr, "ffn_sqr(relu)-", layer)) {
        return ggml_metal_bitnet_ffn_reject(gate, "graph-edges", trace);
    }
    if (!ggml_metal_bitnet_ffn_expand_layout(gate) || !ggml_metal_bitnet_ffn_expand_layout(up) ||
        !ggml_metal_bitnet_ffn_vector(relu, 6912) || !ggml_metal_bitnet_ffn_vector(sqr, 6912) ||
        !ggml_metal_bitnet_ffn_vector(out, 6912)) {
        return ggml_metal_bitnet_ffn_reject(gate, "layout", trace);
    }
    const ggml_op ops[] = { GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL_MAT, GGML_OP_SQR, GGML_OP_MUL };
    const int nodes[] = { ctx->graph_idx(idx), ctx->graph_idx(idx + 1), ctx->graph_idx(idx + 2),
                          ctx->graph_idx(idx + 3), ctx->graph_idx(idx + 4) };
    if (!ggml_can_fuse_subgraph_ext(ctx->graph(), nodes, 5, ops, &nodes[4], 1)) {
        return ggml_metal_bitnet_ffn_reject(gate, "consumers", trace);
    }
    const ggml_tensor * tensors[] = { gate->src[0], up->src[0], gate->src[1], out };
    ggml_metal_buffer_id bindings[4];
    if (!ggml_metal_bitnet_ffn_bindings(tensors, bindings, 4)) {
        return ggml_metal_bitnet_ffn_reject(gate, "bindings", trace);
    }
    const auto pipeline = ggml_metal_library_get_bitnet_ffn_pipeline(ctx->lib, true);
    if (!pipeline.pipeline) {
        return ggml_metal_bitnet_ffn_reject(gate, "pipeline", trace);
    }

    // The first gate was checked by the caller. Check the other input and the true output.
    if (!ggml_metal_op_concurrency_check(ctx, up) || !ggml_metal_op_concurrency_check(ctx, out)) {
        ggml_metal_op_concurrency_reset(ctx);
    }
    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    for (int i = 0; i < 4; ++i) {
        ggml_metal_encoder_set_buffer(ctx->enc, bindings[i], i);
    }
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, 432, 1, 1, 32, 2, 1);
    if (ctx->numerics_capture) {ggml_metal_bitnet_numerics_dispatch(ctx,gate,out,pipeline,432,1,1,32,2,0,0,5);}
    if (trace) {
        std::fprintf(stderr, "BITNET_FFN_BIND x=%p+%zu out=%p+%zu node=%s consumers=checked skipped=4\n",
                bindings[2].metal, bindings[2].offs, bindings[3].metal, bindings[3].offs, out->name);
        ggml_metal_encoder_trace_bitnet(ctx->enc, pipeline, "ffn-fused", 1, 432, 1, 1, 2);
        std::fprintf(stderr, "BITNET_MULTILAYER_BIND route=ffn-fused layer=%d node=%s gate_weight=%s up_weight=%s gate_w=%p+%zu up_w=%p+%zu x=%p+%zu y=%p+%zu bounds=1 disjoint=1 consumers=checked\n",
                layer, gate->name, gate->src[0]->name, up->src[0]->name, bindings[0].metal, bindings[0].offs,
                bindings[1].metal, bindings[1].offs, bindings[2].metal, bindings[2].offs, bindings[3].metal, bindings[3].offs);
        ggml_metal_encoder_trace_bitnet_node(ctx->enc, pipeline, "ffn-fused", layer, gate, out, 432, 1, 1, 32, 2);
    }
    return true;
}

static bool ggml_metal_try_bitnet_poc(ggml_metal_op_t ctx, const ggml_tensor * op, bool trace, bool guard_only = false) {
    // This isolated experiment may only replace layer-0 Q, regardless of a broader layer selection.
    if (ggml_metal_bitnet_layer(op->name, "Qcur-") != 0) {
        return false;
    }
    const char * rows_env = std::getenv("GGML_METAL_BITNET_POC_ROWS_PER_SIMD");
    const char * simd_env = std::getenv("GGML_METAL_BITNET_POC_SIMD_PER_GROUP");
    const char * variant_env = std::getenv("GGML_METAL_BITNET_POC_VARIANT");
    struct ggml_metal_bitnet_q_contract contract;
    if (!ggml_metal_bitnet_q_select(variant_env, rows_env, simd_env, &contract)) {
        if (trace) {
            std::fprintf(stderr, "BITNET_Q_FALLBACK node=%s layer=0 reason=kernel_contract variant=%s rows=%s simd=%s\n",
                    op->name, variant_env ? variant_env : "legacy", rows_env ? rows_env : "unset", simd_env ? simd_env : "unset");
        }
        return false;
    }
    const int rows_per_simdgroup = contract.rows;
    const int simdgroups_per_threadgroup = contract.groups;
    const int64_t weight_ne[] = { 2560, 2560, 1, 1 };
    const size_t weight_nb[] = { 66, 660, 1689600, 1689600 };
    const int64_t vector_ne[] = { 2560, 1, 1, 1 };
    const size_t vector_nb[] = { 4, 10240, 10240, 10240 };
    if (!ggml_metal_bitnet_poc_layout(op->src[0], GGML_TYPE_TQ2_0, weight_ne, weight_nb) ||
        !ggml_metal_bitnet_poc_layout(op->src[1], GGML_TYPE_F32, vector_ne, vector_nb) ||
        !ggml_metal_bitnet_poc_layout(op, GGML_TYPE_F32, vector_ne, vector_nb)) {
        if (trace) {
            std::fprintf(stderr, "BITNET_Q_FALLBACK node=%s layer=%d reason=layout\n", op->name, ggml_metal_bitnet_layer(op->name, "Qcur-"));
        }
        return false;
    }
    if (!ggml_metal_bitnet_actquant_q4_contract(op, rows_per_simdgroup, simdgroups_per_threadgroup, guard_only, trace)) {
        return false;
    }

    const auto w = ggml_metal_get_buffer_id(op->src[0]);
    const auto x = ggml_metal_get_buffer_id(op->src[1]);
    const auto y = ggml_metal_get_buffer_id(op);
    const size_t w_bytes = ggml_nbytes(op->src[0]);
    const size_t x_bytes = ggml_nbytes(op->src[1]);
    const size_t y_bytes = ggml_nbytes(op);
    if ((w.offs | x.offs | y.offs) % 32 != 0 ||
        !ggml_metal_buffer_contains(w, w_bytes) ||
        !ggml_metal_buffer_contains(x, x_bytes) ||
        !ggml_metal_buffer_contains(y, y_bytes) ||
        !ggml_metal_bitnet_poc_disjoint(x, x_bytes, y, y_bytes) ||
        !ggml_metal_bitnet_poc_disjoint(w, w_bytes, y, y_bytes)) {
        if (trace) {
            std::fprintf(stderr, "BITNET_Q_FALLBACK node=%s layer=%d reason=bindings\n", op->name, ggml_metal_bitnet_layer(op->name, "Qcur-"));
        }
        return false;
    }

    const auto pipeline = ggml_metal_library_get_bitnet_poc_pipeline(ctx->lib, &contract);
    if (!pipeline.pipeline) {
        if (trace) {
            std::fprintf(stderr, "BITNET_Q_FALLBACK node=%s layer=%d reason=pipeline\n", op->name, ggml_metal_bitnet_layer(op->name, "Qcur-"));
        }
        return false;
    }
    const int execution_width = ggml_metal_pipeline_thread_execution_width(pipeline);
    const int max_threads = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    if (!ggml_metal_bitnet_q_launch_valid(&contract, op->ne[0], op->src[0]->ne[0], execution_width, max_threads)) {
        if (trace) {
            std::fprintf(stderr, "BITNET_Q_FALLBACK node=%s layer=0 reason=pipeline_limits width=%d max_threads=%d requested=%d\n",
                    op->name, execution_width, max_threads, 32*simdgroups_per_threadgroup);
        }
        return false;
    }

    if (guard_only) {
        if (trace) {
            std::fprintf(stderr, "BITNET_Q_GUARD_PASS node=%s layer=0 step=%s checked=1 candidate_pipeline=%s width=%d max_threads=%d rows=%d simd=%d w=%p+%zu x=%p+%zu y=%p+%zu bounds=1 disjoint=1 custom_encoded=0\n",
                    op->name, std::getenv("GGML_METAL_BITNET_FFN_STEP"), ggml_metal_pipeline_name(pipeline),
                    execution_width, max_threads, rows_per_simdgroup, simdgroups_per_threadgroup,
                    w.metal, w.offs, x.metal, x.offs, y.metal, y.offs);
        }
        return true;
    }

    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    ggml_metal_encoder_set_buffer(ctx->enc, w, 0);
    ggml_metal_encoder_set_buffer(ctx->enc, x, 1);
    ggml_metal_encoder_set_buffer(ctx->enc, y, 2);
    const int grid_x = contract.grid_x;
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, grid_x, 1, 1, contract.threads_x, contract.groups, 1);
    if (trace) {
        std::fprintf(stderr, "BITNET_Q_GEOMETRY node=%s variant=%s contract=%d entry=%s rows_per_simd=%d simd_per_group=%d grid=%d threads=%d width=%d max_threads=%d checked=1\n",
                op->name, contract.variant, contract.version, contract.entry, rows_per_simdgroup, simdgroups_per_threadgroup,
                grid_x, contract.threads_x*contract.groups, execution_width, max_threads);
        std::fprintf(stderr,
            "BITNET_BIND node=%s weight=%s w=%p+%zu x=%p+%zu y=%p+%zu xy_disjoint=1\n",
            op->name, op->src[0]->name, w.metal, w.offs, x.metal, x.offs, y.metal, y.offs);
        std::fprintf(stderr, "BITNET_MULTILAYER_BIND route=q-custom layer=%d node=%s weight=%s w=%p+%zu x=%p+%zu y=%p+%zu bounds=1 disjoint=1\n",
                ggml_metal_bitnet_layer(op->name, "Qcur-"), op->name, op->src[0]->name,
                w.metal, w.offs, x.metal, x.offs, y.metal, y.offs);
        ggml_metal_encoder_trace_bitnet(ctx->enc, pipeline, "poc", 1, grid_x, 1, 1, simdgroups_per_threadgroup);
        ggml_metal_encoder_trace_bitnet_node(ctx->enc, pipeline, "q-custom", ggml_metal_bitnet_layer(op->name, "Qcur-"),
                op, op, grid_x, 1, 1, 32, simdgroups_per_threadgroup);
    }
    return true;
}

static void ggml_metal_op_mul_mat_mv_encode(
        ggml_metal_op_t     ctx,
        const ggml_tensor * op,
        const ggml_tensor * dst,
        const ggml_tensor * res,
        const ggml_tensor * norm,
        const ggml_tensor * w,
        bool stock_only = false) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    const bool add = dst != op;

    const bool target = ggml_metal_bitnet_poc_target(op);
    const bool trace = target && ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_POC_TRACE");
    const bool guard_only = ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_Q_GUARD_ONLY");
    const bool custom_q = ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_POC");
    if (guard_only && custom_q) { GGML_ABORT("Q guard-only and custom routes are mutually exclusive"); }
    bool guarded_stock = false;
    if (!stock_only && !std::getenv("GGML_METAL_BITNET_FFN_MECHANISM") && target && ne11 == 1 && !add && !norm &&
        ggml_metal_bitnet_selected_layer(ggml_metal_bitnet_layer(op->name, "Qcur-")) &&
        ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_DECODE")) {
        if (guard_only) {
            guarded_stock = ggml_metal_try_bitnet_poc(ctx, op, trace, true);
        } else if (custom_q && ggml_metal_try_bitnet_poc(ctx, op, trace)) {
            return;
        }
    }

    const auto ffn = ggml_metal_bitnet_ffn_projection(op);
    const char * ffn_route = ggml_metal_bitnet_ffn_stock_route(ffn);
    const bool ffn_trace = ffn_route && ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_TRACE");
    if (!stock_only && !std::getenv("GGML_METAL_BITNET_FFN_MECHANISM") && (ffn == bitnet_ffn_projection::gate || ffn == bitnet_ffn_projection::up) && !add && !norm &&
        ggml_metal_bitnet_ffn_mode("split") && ggml_metal_try_bitnet_ffn_split(ctx, op, ffn, ffn_trace)) {
        return;
    }

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op, add, norm != nullptr);

    const int nr0 = pipeline.nr0;
    const int nr1 = pipeline.nr1;
    const int nsg = pipeline.nsg;

    const size_t smem = pipeline.smem;

    ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
        /*.eps  =*/ norm ? ggml_get_op_params_f32(norm, 0) : 0.0f,
    };

    const ggml_tensor * src1 = norm ? norm->src[0] : op->src[1];

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src1),       2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst),        3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(res),        4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(w ? w : src1), 5);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    if (op->src[0]->type == GGML_TYPE_F32 ||
        op->src[0]->type == GGML_TYPE_F16 ||
        op->src[0]->type == GGML_TYPE_BF16 ||
        op->src[0]->type == GGML_TYPE_Q8_0) {
        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0 - 1)/(nr0)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, 32, nsg, 1);
    } else {
        const int grid_x = (ne01 + nr0*nsg - 1)/(nr0*nsg);
        const int grid_y = (ne11 + nr1 - 1)/nr1;
        const int grid_z = ne12*ne13;
        ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, 32, nsg, 1);
        if (ctx->numerics_capture) {ggml_metal_bitnet_numerics_dispatch(ctx,op,dst,pipeline,grid_x,grid_y,grid_z,32,nsg,add,norm!=nullptr,1);}
        if (trace) {
            if (ggml_metal_bitnet_layer(op->name, "Qcur-") == 0 && ne11 == 1) {
                const auto weight_binding = ggml_metal_get_buffer_id(op->src[0]);
                const auto input_binding = ggml_metal_get_buffer_id(src1);
                const auto output_binding = ggml_metal_get_buffer_id(dst);
                const auto residual_binding = ggml_metal_get_buffer_id(res);
                const auto norm_binding = ggml_metal_get_buffer_id(w ? w : src1);
                std::fprintf(stderr, "BITNET_Q_STOCK_ABI node=%s step=%s guarded=%d pipeline=%s cached_pipeline=%p args_bytes=%zu add=%d norm=%d grid=[%d,%d,%d] threads=[32,%d,1] smem=%zu w=%p+%zu x=%p+%zu y=%p+%zu residual=%p+%zu norm_weight=%p+%zu slots=[0,1,2,3,4,5]\n",
                        op->name, std::getenv("GGML_METAL_BITNET_FFN_STEP"), guarded_stock,
                        ggml_metal_pipeline_name(pipeline), (void *)pipeline.pipeline, sizeof(args), add, norm != nullptr,
                        grid_x, grid_y, grid_z, nsg, smem, weight_binding.metal, weight_binding.offs,
                        input_binding.metal, input_binding.offs, output_binding.metal, output_binding.offs,
                        residual_binding.metal, residual_binding.offs, norm_binding.metal, norm_binding.offs);
            }
            ggml_metal_encoder_trace_bitnet(enc, pipeline, "stock-mv", ne11, grid_x, grid_y, grid_z, nsg);
            ggml_metal_encoder_trace_bitnet_node(enc, pipeline, "q-stock", ggml_metal_bitnet_layer(op->name, "Qcur-"),
                    op, dst, grid_x, grid_y, grid_z, 32, nsg);
        }
        if (ffn_trace) {
            if (ffn == bitnet_ffn_projection::down) {
                const auto input = ggml_metal_get_buffer_id(src1);
                std::fprintf(stderr, "BITNET_FFN_DOWN_BIND node=%s input=%s x=%p+%zu\n",
                        op->name, src1->name, input.metal, input.offs);
            }
            ggml_metal_encoder_trace_bitnet(enc, pipeline, ffn_route, ne11, grid_x, grid_y, grid_z, nsg);
            const auto weight = ggml_metal_get_buffer_id(op->src[0]);
            const auto input = ggml_metal_get_buffer_id(src1);
            const auto output = ggml_metal_get_buffer_id(dst);
            std::fprintf(stderr, "BITNET_MULTILAYER_BIND route=%s layer=%d node=%s weight=%s w=%p+%zu x=%p+%zu y=%p+%zu\n",
                    ffn_route, ggml_metal_bitnet_ffn_layer(op), op->name, op->src[0]->name,
                    weight.metal, weight.offs, input.metal, input.offs, output.metal, output.offs);
            ggml_metal_encoder_trace_bitnet_node(enc, pipeline, ffn_route, ggml_metal_bitnet_ffn_layer(op),
                    op, dst, grid_x, grid_y, grid_z, 32, nsg);
        }
    }
    if (ctx->ablation_shadow_all && (ffn == bitnet_ffn_projection::gate || ffn == bitnet_ffn_projection::up) &&
        ctx->ablation_mode!='C' && !std::getenv("GGML_METAL_BITNET_FFN_MECHANISM") &&
        ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_DECODE") && op->ne[1]==1) {
        const char * directory=std::getenv("GGML_METAL_BITNET_FFN_SHADOW_DIR");
        const char * step=std::getenv("GGML_METAL_BITNET_FFN_STEP");
        const int layer=ggml_metal_bitnet_ffn_layer(op);
        if(directory && step && std::atoi(step)>0 && ggml_metal_bitnet_selected_layer(layer)) {
            const auto shadow=ggml_metal_library_get_bitnet_ffn_mechanism_pipeline(ctx->lib,3);
            if(!shadow.pipeline) { GGML_ABORT("A/B FFN diagnostic shadow pipeline missing"); }
            char path[4096];std::snprintf(path,sizeof(path),"%s/step-%s/layer-%d/%s.f32",directory,step,layer,
                ffn==bitnet_ffn_projection::gate ? "gate" : "up");
            ggml_metal_op_concurrency_reset(ctx);
            if(!ggml_metal_encoder_bitnet_shadow(enc,shadow,ggml_metal_get_buffer_id(dst),27648,path)) { GGML_ABORT("A/B FFN diagnostic shadow failed"); }
            ggml_metal_op_concurrency_reset(ctx);
        }
    }
    if (ffn == bitnet_ffn_projection::down && op->ne[1]==1 &&
            ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_DECODE")) {
        const char * mode=std::getenv("GGML_METAL_BITNET_FFN_MECHANISM");
        const char * directory=std::getenv("GGML_METAL_BITNET_FFN_K_SHADOW_DIR");
        const char * step=std::getenv("GGML_METAL_BITNET_FFN_STEP");
        const int layer=ggml_metal_bitnet_ffn_layer(op);
        if (mode && std::strcmp(mode,"K")==0 && directory && ggml_metal_bitnet_k_shadow_step(step) && ggml_metal_bitnet_selected_layer(layer)) {
            ggml_metal_bitnet_k_save_weight(directory,layer,"ffn_down.tq2",op->src[0]);
            const auto shadow=ggml_metal_library_get_bitnet_ffn_mechanism_pipeline(ctx->lib,3);
            if (shadow.pipeline) {
                char path[4096];std::snprintf(path,sizeof(path),"%s/step-%s/layer-%d/down.f32",directory,step,layer);
                ggml_metal_op_concurrency_reset(ctx);
                if (!ctx->mem_ranges) {ggml_metal_encoder_memory_barrier(enc);}
                if (!ggml_metal_encoder_bitnet_shadow(enc,shadow,ggml_metal_get_buffer_id(dst),10240,path)) {GGML_ABORT("K down diagnostic shadow failed");}
                ggml_metal_op_concurrency_reset(ctx);
                if (!ctx->mem_ranges) {ggml_metal_encoder_memory_barrier(enc);}
            } else {GGML_LOG_WARN("K down diagnostic copy pipeline unavailable\n");}
        }
    }

}

// New, independent mechanism route. It is never enabled by the legacy FFN/Q flags.
static int ggml_metal_op_bin_single(ggml_metal_op_t ctx, int idx, bool allow_fusion);

static bool ggml_metal_bitnet_exact_alias(ggml_metal_buffer_id a, size_t an, ggml_metal_buffer_id b, size_t bn) {
    return a.metal == b.metal && a.offs == b.offs && an == bn;
}

struct bitnet_ffn_plan {
    int idx=0, layer=-1;
    const char * mode=nullptr;
    const ggml_tensor * gate=nullptr, * relu=nullptr, * up=nullptr, * sqr=nullptr, * out=nullptr;
    const ggml_tensor * ts[6]{};
    ggml_metal_buffer_id bids[6]{};
    ggml_metal_pipeline_with_params pipeline{};
    ggml_metal_pipeline_with_params post_pipeline{};
};

// Reads graph/tensor/Metal buffer state and may materialize a CPU-side pipeline
// cache. It never touches encoder/concurrency ranges or consumes graph nodes.
static bool ggml_metal_check_ffn_plan(ggml_metal_op_t ctx, int idx, const char * mode, bool trace, bitnet_ffn_plan & plan) {
    const uint64_t names_begin=ctx->ablation_cpu ? bitnet_abc_now() : 0;
    const auto * gate = ctx->node(idx);
    if (!mode) { return false; }
    const bool a = std::strcmp(mode, "A") == 0;
    const bool b = std::strcmp(mode, "B") == 0;
    const bool c = std::strcmp(mode, "C") == 0;
    const bool f = std::strcmp(mode, "F") == 0;
    const bool g = std::strcmp(mode, "G") == 0;
    const int layer = ggml_metal_bitnet_ffn_layer(gate);
    if (layer < 0 || (!(c || g) && layer != 0)) { return false; }
    if (!a && !b && !c && !f && !g) { return ggml_metal_bitnet_ffn_reject(gate, "mechanism-not-enabled", trace); }
    if ((c || g) && !ggml_metal_bitnet_selected_layer(layer)) {
        return ggml_metal_bitnet_ffn_reject(gate, "layer-selection", trace);
    }
    if (gate->ne[1] != 1 || !ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_DECODE")) {
        return ggml_metal_bitnet_ffn_reject(gate, "decode-or-token-count", trace);
    }
    if (ctx->n_nodes() - idx < 5) { return ggml_metal_bitnet_ffn_reject(gate, "encoder-slice", trace); }
    const auto * relu = ctx->node(idx + 1);
    const auto * up = ctx->node(idx + 2);
    const auto * sqr = ctx->node(idx + 3);
    const auto * out = ctx->node(idx + 4);
    if (relu->op != GGML_OP_UNARY || ggml_get_unary_op(relu) != GGML_UNARY_OP_RELU || relu->src[0] != gate ||
            ggml_metal_bitnet_ffn_projection(up) != bitnet_ffn_projection::up || ggml_metal_bitnet_ffn_layer(up) != layer || up->src[1] != gate->src[1] ||
            sqr->op != GGML_OP_SQR || sqr->src[0] != relu || out->op != GGML_OP_MUL || out->src[0] != sqr || out->src[1] != up ||
            !ggml_metal_bitnet_layer_name(relu, "ffn_relu-", layer) || !ggml_metal_bitnet_layer_name(sqr, "ffn_sqr(relu)-", layer) || !ggml_metal_bitnet_layer_name(out, "ffn_sub_out-", layer)) {
        return ggml_metal_bitnet_ffn_reject(gate, "graph-edges", trace);
    }
    if (!ggml_metal_bitnet_ffn_expand_layout(gate) || !ggml_metal_bitnet_ffn_expand_layout(up) ||
            !ggml_metal_bitnet_ffn_vector(relu, 6912) || !ggml_metal_bitnet_ffn_vector(sqr, 6912) || !ggml_metal_bitnet_ffn_vector(out, 6912)) {
        return ggml_metal_bitnet_ffn_reject(gate, "layout", trace);
    }
    const ggml_op ops[] = { GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL_MAT, GGML_OP_SQR, GGML_OP_MUL };
    const int nodes[] = { ctx->graph_idx(idx), ctx->graph_idx(idx+1), ctx->graph_idx(idx+2), ctx->graph_idx(idx+3), ctx->graph_idx(idx+4) };
    if(names_begin) { ctx->ablation_stats.names_layout_ns+=bitnet_abc_now()-names_begin; ++ctx->ablation_stats.substage_timers; }
    const uint64_t graph_begin = ctx->ablation_cpu ? bitnet_abc_now() : 0;
    const bool graph_ok=ggml_can_fuse_subgraph_ext(ctx->graph(), nodes, 5, ops, &nodes[4], 1);
    if (graph_begin) { ctx->ablation_stats.graph_check_ns += bitnet_abc_now()-graph_begin; ++ctx->ablation_stats.substage_timers; }
    if (!graph_ok) {
        return ggml_metal_bitnet_ffn_reject(gate, "consumers", trace);
    }
    // Gate/out or up/out may alias exactly: epilogues read only their own row before overwriting it.
    // Partial overlap is never allowed. All projection outputs must be disjoint from X and weights.
    const uint64_t buffer_begin = ctx->ablation_cpu ? bitnet_abc_now() : 0;
    const ggml_tensor * ts[] = { gate->src[0], up->src[0], gate->src[1], gate, up, out };
    ggml_metal_buffer_id bids[6];
    for (int i = 0; i < 6; ++i) {
        bids[i] = ggml_metal_get_buffer_id(ts[i]);
        if (bids[i].offs % 32 || !ggml_metal_buffer_contains(bids[i], ggml_nbytes(ts[i]))) {
            return ggml_metal_bitnet_ffn_reject(gate, "bindings", trace);
        }
        for (int j = 0; j < i; ++j) {
            const bool exact_inplace = i == 5 && j >= 3 && ggml_metal_bitnet_exact_alias(bids[i], ggml_nbytes(ts[i]), bids[j], ggml_nbytes(ts[j]));
            if (!exact_inplace && !ggml_metal_bitnet_poc_disjoint(bids[i], ggml_nbytes(ts[i]), bids[j], ggml_nbytes(ts[j]))) {
                return ggml_metal_bitnet_ffn_reject(gate, "partial-overlap", trace);
            }
        }
    }
    if (buffer_begin) { ctx->ablation_stats.buffer_check_ns += bitnet_abc_now()-buffer_begin; ++ctx->ablation_stats.substage_timers; }
    const uint64_t pipeline_begin = ctx->ablation_cpu ? bitnet_abc_now() : 0;
    const auto pipeline = f ? ggml_metal_pipeline_with_params{} : ggml_metal_library_get_bitnet_ffn_mechanism_pipeline(ctx->lib, a ? 0 : b ? 1 : 2);
    const int nth = a ? 256 : 64;
    if (!f && (!pipeline.pipeline || ggml_metal_pipeline_thread_execution_width(pipeline) != 32 || ggml_metal_pipeline_max_theads_per_threadgroup(pipeline) < nth)) {
        return ggml_metal_bitnet_ffn_reject(gate, "mechanism-pipeline", trace);
    }
    const auto post_pipeline = g ? ggml_metal_library_get_bitnet_ffn_mechanism_pipeline(ctx->lib, 0) : ggml_metal_pipeline_with_params{};
    if (g && (!post_pipeline.pipeline || ggml_metal_pipeline_thread_execution_width(post_pipeline) != 32 ||
              ggml_metal_pipeline_max_theads_per_threadgroup(post_pipeline) < 256)) {
        return ggml_metal_bitnet_ffn_reject(gate, "mechanism-post-pipeline", trace);
    }
    if (pipeline_begin) { ctx->ablation_stats.pipeline_check_ns += bitnet_abc_now()-pipeline_begin; ++ctx->ablation_stats.substage_timers; }
    plan.idx=idx; plan.layer=layer; plan.mode=mode; plan.gate=gate; plan.relu=relu; plan.up=up; plan.sqr=sqr; plan.out=out; plan.pipeline=pipeline; plan.post_pipeline=post_pipeline;
    for(int i=0;i<6;++i) { plan.ts[i]=ts[i]; plan.bids[i]=bids[i]; }
    return true;
}

static bool ggml_metal_commit_ffn_plan(ggml_metal_op_t ctx, const bitnet_ffn_plan & plan, bool trace) {
    const int idx=plan.idx, layer=plan.layer; const char * mode=plan.mode;
    const auto * gate=plan.gate; const auto * up=plan.up; const auto * out=plan.out;
    const bool a=std::strcmp(mode,"A")==0, b=std::strcmp(mode,"B")==0, c=std::strcmp(mode,"C")==0, f=std::strcmp(mode,"F")==0, g=std::strcmp(mode,"G")==0;
    const auto & bids=plan.bids; const auto & ts=plan.ts; const auto pipeline=plan.pipeline;
    const char * shadow_dir = std::getenv("GGML_METAL_BITNET_FFN_SHADOW_DIR");
    const char * shadow_step = std::getenv("GGML_METAL_BITNET_FFN_STEP");
    const bool shadow_requested = (a || c || f || g) && shadow_dir && shadow_step &&
            (ctx->ablation_shadow_all ||
             std::strcmp(shadow_step, "1") == 0 || std::strcmp(shadow_step, "16") == 0 || std::strcmp(shadow_step, "32") == 0);
    const auto shadow_pipeline = shadow_requested ? ggml_metal_library_get_bitnet_ffn_mechanism_pipeline(ctx->lib, 3) : ggml_metal_pipeline_with_params{};
    if (shadow_requested && !shadow_pipeline.pipeline) { return ggml_metal_bitnet_ffn_reject(gate, "shadow-pipeline", trace); }
    auto take_shadows = [&]() {
        if (!shadow_requested) { return; }
        char gate_path[4096], up_path[4096];
        if (c || g) {
            std::snprintf(gate_path, sizeof(gate_path), "%s/step-%s/layer-%d/gate.f32", shadow_dir, shadow_step, layer);
            std::snprintf(up_path, sizeof(up_path), "%s/step-%s/layer-%d/up.f32", shadow_dir, shadow_step, layer);
        } else {
            std::snprintf(gate_path, sizeof(gate_path), "%s/step-%s/gate.f32", shadow_dir, shadow_step);
            std::snprintf(up_path, sizeof(up_path), "%s/step-%s/up.f32", shadow_dir, shadow_step);
        }
        if (!ggml_metal_encoder_bitnet_shadow(ctx->enc, shadow_pipeline, bids[3], 27648, gate_path) ||
                !ggml_metal_encoder_bitnet_shadow(ctx->enc, shadow_pipeline, bids[4], 27648, up_path)) { GGML_ABORT("FFN diagnostic shadow allocation/encoding failed"); }
        ggml_metal_op_concurrency_reset(ctx);
    };
    // Validate all pending reads/writes before replacing the five-node sequence.
    for (int i = 0; i < 5; ++i) {
        if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx+i))) { ggml_metal_op_concurrency_reset(ctx); }
    }
    if (a || b || f) {
        // This is the real stock pipeline/argument ABI, explicitly bypassing all custom routes.
        ggml_metal_op_mul_mat_mv_encode(ctx, gate, gate, gate, nullptr, nullptr, true);
        if (a || f) { ggml_metal_op_mul_mat_mv_encode(ctx, up, up, up, nullptr, nullptr, true); }
        ggml_metal_op_concurrency_reset(ctx);
    }
    if (a || f) { take_shadows(); }
    if (!f) { ggml_metal_encoder_set_pipeline(ctx->enc, pipeline); }
    if (f) {
        ggml_metal_op_unary_single(ctx, idx+1, false);
        ggml_metal_op_concurrency_reset(ctx);
        ggml_metal_op_unary_single(ctx, idx+3, false);
        ggml_metal_op_concurrency_reset(ctx);
        ggml_metal_op_bin_single(ctx, idx+4, false);
    } else if (a) {
        ggml_metal_encoder_set_buffer(ctx->enc, bids[3], 0);
        ggml_metal_encoder_set_buffer(ctx->enc, bids[4], 1);
        ggml_metal_encoder_set_buffer(ctx->enc, bids[5], 2);
        ggml_metal_encoder_dispatch_threadgroups(ctx->enc, 27, 1, 1, 256, 1, 1);
        if (trace) { ggml_metal_encoder_trace_bitnet_node(ctx->enc, pipeline, "ffn-mech-A-post", layer, out, out, 27, 1, 1, 256, 1); }
    } else if (b) {
        ggml_metal_encoder_set_buffer(ctx->enc, bids[1], 0);
        ggml_metal_encoder_set_buffer(ctx->enc, bids[2], 1);
        ggml_metal_encoder_set_buffer(ctx->enc, bids[3], 2);
        ggml_metal_encoder_set_buffer(ctx->enc, bids[5], 3);
        ggml_metal_encoder_dispatch_threadgroups(ctx->enc, 864, 1, 1, 32, 2, 1);
        if (trace) { ggml_metal_encoder_trace_bitnet_node(ctx->enc, pipeline, "ffn-mech-B-up", layer, up, out, 864, 1, 1, 32, 2); }
    } else {
        ggml_metal_encoder_set_buffer(ctx->enc, bids[0], 0);
        ggml_metal_encoder_set_buffer(ctx->enc, bids[1], 1);
        ggml_metal_encoder_set_buffer(ctx->enc, bids[2], 2);
        ggml_metal_encoder_set_buffer(ctx->enc, bids[3], 3);
        ggml_metal_encoder_set_buffer(ctx->enc, bids[4], 4);
        ggml_metal_encoder_dispatch_threadgroups(ctx->enc, 864, 1, 2, 32, 2, 1);
        if (trace) { ggml_metal_encoder_trace_bitnet_node(ctx->enc, pipeline, g ? "ffn-mech-G-dual" : "ffn-mech-C-dual", layer, gate, up, 864, 1, 2, 32, 2); }
        ggml_metal_op_concurrency_reset(ctx);
        if (g && !ctx->mem_ranges) { ggml_metal_encoder_memory_barrier(ctx->enc); }
        take_shadows();
        if (g) {
            if (shadow_requested) { ggml_metal_encoder_memory_barrier(ctx->enc); }
            ggml_metal_encoder_set_pipeline(ctx->enc, plan.post_pipeline);
            ggml_metal_encoder_set_buffer(ctx->enc, bids[3], 0);
            ggml_metal_encoder_set_buffer(ctx->enc, bids[4], 1);
            ggml_metal_encoder_set_buffer(ctx->enc, bids[5], 2);
            ggml_metal_encoder_dispatch_threadgroups(ctx->enc, 27, 1, 1, 256, 1, 1);
            if (trace) {
                std::fprintf(stderr, "BITNET_FFN_G_BARRIER layer=%d step=%s scope=buffers projections_to_post=1\n", layer, shadow_step ? shadow_step : "unrecorded");
                ggml_metal_encoder_trace_bitnet_node(ctx->enc, plan.post_pipeline, "ffn-mech-G-post", layer, out, out, 27, 1, 1, 256, 1);
            }
        } else {
            // Retain the actual stock post pipelines and their three FP32 stores.
            ggml_metal_op_unary_single(ctx, idx+1, false);
            ggml_metal_op_concurrency_reset(ctx);
            ggml_metal_op_unary_single(ctx, idx+3, false);
            ggml_metal_op_concurrency_reset(ctx);
            ggml_metal_op_bin_single(ctx, idx+4, false);
        }
    }
    if (trace) {
        const char * step = shadow_step ? shadow_step : "unrecorded";
        const char * roles[] = { "gate_weight", "up_weight", "shared_x", "gate", "up", "activation" };
        for (int i = 0; i < 6; ++i) {
            const ggml_tensor * t = ts[i];
            std::fprintf(stderr, "BITNET_FFN_CONTRACT mode=%s layer=%d step=%s role=%s node=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] bytes=%zu buffer=%p offset=%zu bounds=1\n",
                    mode, layer, step, roles[i], t->name, ggml_type_name(t->type),
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                    t->nb[0], t->nb[1], t->nb[2], t->nb[3], ggml_nbytes(t), bids[i].metal, bids[i].offs);
        }
        std::fprintf(stderr, "BITNET_FFN_IDENTITY mode=%s layer=%d step=%s node=%s gate_weight=%s up_weight=%s shared_x=%s output=%s\n",
                mode, layer, step, gate->name, gate->src[0]->name, up->src[0]->name, gate->src[1]->name, out->name);
        std::fprintf(stderr, "BITNET_FFN_MECHANISM mode=%s layer=%d consumed=5 dispatches=%d shared_x=1 consumers=checked gate_out_alias=%d up_out_alias=%d wg=%p+%zu wu=%p+%zu x=%p+%zu gate=%p+%zu up=%p+%zu out=%p+%zu weighted_norm_down=stock\n", mode, layer, f ? 5 : a ? 3 : b ? 2 : g ? 2 : 4,
                ggml_metal_bitnet_exact_alias(bids[3],ggml_nbytes(gate),bids[5],ggml_nbytes(out)),ggml_metal_bitnet_exact_alias(bids[4],ggml_nbytes(up),bids[5],ggml_nbytes(out)),
                bids[0].metal,bids[0].offs,bids[1].metal,bids[1].offs,bids[2].metal,bids[2].offs,bids[3].metal,bids[3].offs,bids[4].metal,bids[4].offs,bids[5].metal,bids[5].offs);
    }
    return true;
}


struct bitnet_ffn_k_plan {
    int idx=0, layer=-1;
    const ggml_tensor * nodes[7]{};
    const ggml_tensor * ts[8]{};
    ggml_metal_buffer_id bids[8]{};
    ggml_metal_pipeline_with_params dual{}, postnorm{};
    float epsilon=0.0f;
};

static bool ggml_metal_check_ffn_k_plan(ggml_metal_op_t ctx, int idx, bool trace, bitnet_ffn_k_plan & plan) {
    const auto * gate = ctx->node(idx);
    const int layer = ggml_metal_bitnet_ffn_layer(gate);
    if (layer < 0) { return false; }
    if (!ggml_metal_bitnet_selected_layer(layer)) { return ggml_metal_bitnet_ffn_reject(gate, "k-layer-selection", trace); }
    if (gate->ne[1] != 1 || !ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_DECODE")) {
        return ggml_metal_bitnet_ffn_reject(gate, "k-decode-or-token-count", trace);
    }
    if (ctx->n_nodes() - idx < 7) { return ggml_metal_bitnet_ffn_reject(gate, "k-encoder-slice", trace); }
    const auto * relu = ctx->node(idx+1);
    const auto * up = ctx->node(idx+2);
    const auto * sqr = ctx->node(idx+3);
    const auto * act = ctx->node(idx+4);
    const auto * rms = ctx->node(idx+5);
    const auto * norm = ctx->node(idx+6);
    if (relu->op != GGML_OP_UNARY || ggml_get_unary_op(relu) != GGML_UNARY_OP_RELU || relu->src[0] != gate ||
            ggml_metal_bitnet_ffn_projection(up) != bitnet_ffn_projection::up || ggml_metal_bitnet_ffn_layer(up) != layer || up->src[1] != gate->src[1] ||
            sqr->op != GGML_OP_SQR || sqr->src[0] != relu || act->op != GGML_OP_MUL || act->src[0] != sqr || act->src[1] != up ||
            rms->op != GGML_OP_RMS_NORM || rms->src[0] != act || norm->op != GGML_OP_MUL || norm->src[0] != rms || !norm->src[1] ||
            !ggml_metal_bitnet_layer_name(relu,"ffn_relu-",layer) || !ggml_metal_bitnet_layer_name(sqr,"ffn_sqr(relu)-",layer) ||
            !ggml_metal_bitnet_layer_name(act,"ffn_sub_out-",layer) || !ggml_metal_bitnet_layer_name(rms,"norm-",layer) ||
            !ggml_metal_bitnet_layer_name(norm,"ffn_sub_norm-",layer) || !ggml_metal_bitnet_weight_name(norm->src[1],layer,"ffn_sub_norm")) {
        return ggml_metal_bitnet_ffn_reject(gate, "k-graph-edges", trace);
    }
    if (!ggml_metal_bitnet_ffn_expand_layout(gate) || !ggml_metal_bitnet_ffn_expand_layout(up) ||
            !ggml_metal_bitnet_ffn_vector(relu,6912) || !ggml_metal_bitnet_ffn_vector(sqr,6912) ||
            !ggml_metal_bitnet_ffn_vector(act,6912) || !ggml_metal_bitnet_ffn_vector(rms,6912) ||
            !ggml_metal_bitnet_ffn_vector(norm,6912) || !ggml_metal_bitnet_ffn_vector(norm->src[1],6912)) {
        return ggml_metal_bitnet_ffn_reject(gate, "k-layout", trace);
    }
    float epsilon;
    std::memcpy(&epsilon,rms->op_params,sizeof(epsilon));
    if (epsilon != 1.0e-5f) { return ggml_metal_bitnet_ffn_reject(gate,"k-epsilon",trace); }
    const int next = ctx->graph_idx(idx+6)+1;
    if (next >= ggml_graph_n_nodes(ctx->graph())) { return ggml_metal_bitnet_ffn_reject(gate,"k-down-edge",trace); }
    const auto * down = ggml_graph_node(ctx->graph(),next);
    const int64_t down_ne[] = {6912,2560,1,1};
    const size_t down_nb[] = {66,1782,4561920,4561920};
    if (ggml_metal_bitnet_ffn_projection(down) != bitnet_ffn_projection::down || ggml_metal_bitnet_ffn_layer(down) != layer || down->src[1] != norm ||
            !ggml_metal_bitnet_poc_layout(down->src[0],GGML_TYPE_TQ2_0,down_ne,down_nb) || !ggml_metal_bitnet_ffn_vector(down,2560)) {
        return ggml_metal_bitnet_ffn_reject(gate,"k-down-edge",trace);
    }
    const ggml_op ops[] = { GGML_OP_MUL_MAT, GGML_OP_UNARY, GGML_OP_MUL_MAT, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_RMS_NORM, GGML_OP_MUL };
    const int nodes[] = {ctx->graph_idx(idx),ctx->graph_idx(idx+1),ctx->graph_idx(idx+2),ctx->graph_idx(idx+3),ctx->graph_idx(idx+4),ctx->graph_idx(idx+5),ctx->graph_idx(idx+6)};
    if (!ggml_can_fuse_subgraph_ext(ctx->graph(),nodes,7,ops,&nodes[6],1)) {
        return ggml_metal_bitnet_ffn_reject(gate,"k-consumers",trace);
    }
    const ggml_tensor * ts[] = {gate->src[0],up->src[0],gate->src[1],gate,up,act,norm->src[1],norm};
    ggml_metal_buffer_id bids[8];
    for (int i=0;i<8;++i) {
        bids[i] = ggml_metal_get_buffer_id(ts[i]);
        if (bids[i].offs%32 || !ggml_metal_buffer_contains(bids[i],ggml_nbytes(ts[i]))) {
            return ggml_metal_bitnet_ffn_reject(gate,"k-bindings",trace);
        }
    }
    if (!ggml_metal_bitnet_exact_alias(bids[3],ggml_nbytes(ts[3]),bids[5],ggml_nbytes(ts[5])) ||
            !ggml_metal_bitnet_exact_alias(bids[4],ggml_nbytes(ts[4]),bids[7],ggml_nbytes(ts[7]))) {
        return ggml_metal_bitnet_ffn_reject(gate,"k-alias-pattern",trace);
    }
    for (int i=0;i<8;++i) {
        for (int j=0;j<i;++j) {
            if ((i==5 && j==3) || (i==7 && j==4)) { continue; }
            if (!ggml_metal_bitnet_poc_disjoint(bids[i],ggml_nbytes(ts[i]),bids[j],ggml_nbytes(ts[j]))) {
                return ggml_metal_bitnet_ffn_reject(gate,"k-overlap",trace);
            }
        }
    }
    const auto dual = ggml_metal_library_get_bitnet_ffn_mechanism_pipeline(ctx->lib,2);
    const auto postnorm = ggml_metal_library_get_bitnet_ffn_k_pipeline(ctx->lib);
    if (!dual.pipeline || ggml_metal_pipeline_thread_execution_width(dual)!=32 || ggml_metal_pipeline_max_theads_per_threadgroup(dual)<64) {
        return ggml_metal_bitnet_ffn_reject(gate,"k-dual-pipeline",trace);
    }
    if (!postnorm.pipeline || ggml_metal_pipeline_thread_execution_width(postnorm)!=32 || ggml_metal_pipeline_max_theads_per_threadgroup(postnorm)<1024) {
        return ggml_metal_bitnet_ffn_reject(gate,"k-postnorm-pipeline",trace);
    }
    plan.idx=idx; plan.layer=layer; plan.epsilon=epsilon; plan.dual=dual; plan.postnorm=postnorm;
    for (int i=0;i<7;++i) {plan.nodes[i]=ctx->node(idx+i);}
    for (int i=0;i<8;++i) {plan.ts[i]=ts[i];plan.bids[i]=bids[i];}
    return true;
}

static bool ggml_metal_try_ffn_k(ggml_metal_op_t ctx, int idx, bool trace) {
    bitnet_ffn_k_plan plan;
    if (!ggml_metal_check_ffn_k_plan(ctx,idx,trace,plan)) {return false;}
    const auto & b=plan.bids;
    const char * shadow_dir=std::getenv("GGML_METAL_BITNET_FFN_K_SHADOW_DIR");
    const char * shadow_step=std::getenv("GGML_METAL_BITNET_FFN_STEP");
    const bool take_shadows=shadow_dir && ggml_metal_bitnet_k_shadow_step(shadow_step);
    const auto shadow=take_shadows ? ggml_metal_library_get_bitnet_ffn_mechanism_pipeline(ctx->lib,3) : ggml_metal_pipeline_with_params{};
    if (take_shadows && !shadow.pipeline) {return ggml_metal_bitnet_ffn_reject(plan.nodes[0],"k-shadow-pipeline",trace);}
    if (take_shadows) {
        ggml_metal_bitnet_k_save_weight(shadow_dir,plan.layer,"ffn_gate.tq2",plan.ts[0]);
        ggml_metal_bitnet_k_save_weight(shadow_dir,plan.layer,"ffn_up.tq2",plan.ts[1]);
        ggml_metal_bitnet_k_save_weight(shadow_dir,plan.layer,"subnorm.f32",plan.ts[6]);
    }
    auto copy=[&](int slot,const char * role) {
        char path[4096];std::snprintf(path,sizeof(path),"%s/step-%s/layer-%d/%s.f32",shadow_dir,shadow_step,plan.layer,role);
        if (!ggml_metal_encoder_bitnet_shadow(ctx->enc,shadow,b[slot],ggml_nbytes(plan.ts[slot]),path)) {GGML_ABORT("K diagnostic shadow failed");}
    };
    for (int i=0;i<7;++i) {
        if (!ggml_metal_op_concurrency_check(ctx,plan.nodes[i])) {ggml_metal_op_concurrency_reset(ctx);}
    }
    ggml_metal_encoder_set_pipeline(ctx->enc,plan.dual);
    ggml_metal_encoder_set_buffer(ctx->enc,b[0],0);
    ggml_metal_encoder_set_buffer(ctx->enc,b[1],1);
    ggml_metal_encoder_set_buffer(ctx->enc,b[2],2);
    ggml_metal_encoder_set_buffer(ctx->enc,b[3],3);
    ggml_metal_encoder_set_buffer(ctx->enc,b[4],4);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc,864,1,2,32,2,1);
    if (trace) {ggml_metal_encoder_trace_bitnet_node(ctx->enc,plan.dual,"ffn-mech-K-dual",plan.layer,plan.nodes[0],plan.nodes[2],864,1,2,32,2);}
    ggml_metal_op_concurrency_reset(ctx);
    if (!ctx->mem_ranges) {ggml_metal_encoder_memory_barrier(ctx->enc);}
    if (take_shadows) {
        copy(2,"x");copy(3,"gate");copy(4,"up");
        ggml_metal_op_concurrency_reset(ctx);
        if (!ctx->mem_ranges) {ggml_metal_encoder_memory_barrier(ctx->enc);}
    }
    ggml_metal_encoder_set_pipeline(ctx->enc,plan.postnorm);
    ggml_metal_encoder_set_buffer(ctx->enc,b[3],0);
    ggml_metal_encoder_set_buffer(ctx->enc,b[4],1);
    ggml_metal_encoder_set_buffer(ctx->enc,b[6],2);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc,1,1,1,1024,1,1);
    if (trace) {
        const char * step=std::getenv("GGML_METAL_BITNET_FFN_STEP");
        if (!step) {step="unrecorded";}
        std::fprintf(stderr,"BITNET_FFN_K_BARRIER layer=%d step=%s scope=buffers projections_to_postnorm=1\n",plan.layer,step);
        ggml_metal_encoder_trace_bitnet_node(ctx->enc,plan.postnorm,"ffn-mech-K-postnorm",plan.layer,plan.nodes[6],plan.nodes[6],1,1,1,1024,1);
        const char * roles[]={"gate_weight","up_weight","shared_x","gate","up","activation","gamma","norm"};
        for (int i=0;i<8;++i) {
            const auto * t=plan.ts[i];
            std::fprintf(stderr,"BITNET_FFN_K_CONTRACT mode=K layer=%d step=%s role=%s node=%s op=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] bytes=%zu buffer=%p offset=%zu bounds=1\n",
                    plan.layer,step,roles[i],t->name,ggml_op_name(t->op),ggml_type_name(t->type),
                    (long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3],t->nb[0],t->nb[1],t->nb[2],t->nb[3],ggml_nbytes(t),b[i].metal,b[i].offs);
        }
        std::fprintf(stderr,"BITNET_FFN_K_IDENTITY mode=K layer=%d step=%s node=%s gate_weight=%s up_weight=%s shared_x=%s output=%s gamma=%s\n",
                plan.layer,step,plan.nodes[0]->name,plan.ts[0]->name,plan.ts[1]->name,plan.ts[2]->name,plan.nodes[6]->name,plan.ts[6]->name);
        std::fprintf(stderr,"BITNET_FFN_K_MECHANISM mode=K layer=%d step=%s consumed=7 dispatches=2 shared_x=1 consumers=checked gate_activation_alias=1 up_norm_alias=1 ranges_disjoint=1 epsilon=%.12g stock_down=1\n",plan.layer,step,plan.epsilon);
    }
    if (take_shadows) {
        ggml_metal_op_concurrency_reset(ctx);
        if (!ctx->mem_ranges) {ggml_metal_encoder_memory_barrier(ctx->enc);}
        copy(5,"activation");copy(7,"down-input");
        ggml_metal_op_concurrency_reset(ctx);
        if (!ctx->mem_ranges) {ggml_metal_encoder_memory_barrier(ctx->enc);}
    }
    return true;
}

static bool ggml_metal_try_ffn_mechanism(ggml_metal_op_t ctx, int idx, bool trace) {
    const char * mode=std::getenv("GGML_METAL_BITNET_FFN_MECHANISM");
    bitnet_ffn_plan plan;
    return mode && ggml_metal_check_ffn_plan(ctx,idx,mode,trace,plan) && ggml_metal_commit_ffn_plan(ctx,plan,trace);
}

// Called at the genuine mul-mat routing positions. C consumes up; B visits it
// normally, so real selector/check counts are intentionally not equalized.
static bool ggml_metal_try_ffn_ablation(ggml_metal_op_t ctx, int idx) {
    const uint64_t begin=ctx->ablation_cpu ? bitnet_abc_now() : 0;
    if (ctx->ablation_count) { ++ctx->ablation_stats.check_calls; }
    const auto * node=ctx->node(idx);
    const bool target=ggml_metal_bitnet_ffn_projection(node)==bitnet_ffn_projection::gate;
    if (target && ctx->ablation_count) { ++ctx->ablation_stats.target_checks; }
    bitnet_ffn_plan plan;
    const bool ok=target && ggml_metal_check_ffn_plan(ctx,idx,"C",ctx->ablation_trace,plan);
    if(begin) { const auto elapsed=bitnet_abc_now()-begin; ctx->ablation_stats.guard_ns+=elapsed; if(target)ctx->ablation_stats.target_guard_ns+=elapsed; ++ctx->ablation_stats.guard_timers; }
    if(!ok) { return false; }
    if(ctx->ablation_count) { ++ctx->ablation_stats.guard_passes; }
    if(ctx->ablation_trace) {
        std::fprintf(stderr,"FFN_ABLATION_PLAN_PASS mode=%c layer=%d step=%d node=%s graph=%p idx=%d state_mutations=0 pipeline_cache_only=1\n",ctx->ablation_mode,plan.layer,ctx->ablation_stats.step,node->name,(void*)ctx->graph(),ctx->graph_idx(idx));
        for(int i=0;i<6;++i) { const auto*t=plan.ts[i]; std::fprintf(stderr,"FFN_ABLATION_BIND mode=%c layer=%d step=%d slot=%d node=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] buffer=%p offset=%zu bytes=%zu bounds=1\n",ctx->ablation_mode,plan.layer,ctx->ablation_stats.step,i,t->name,ggml_type_name(t->type),(long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3],t->nb[0],t->nb[1],t->nb[2],t->nb[3],plan.bids[i].metal,plan.bids[i].offs,ggml_nbytes(t)); }
    }
    if(ctx->ablation_mode=='B') {
        if(ctx->ablation_count) { ++ctx->ablation_stats.stock_continuations; }
        if(ctx->ablation_trace) { std::fprintf(stderr,"FFN_ABLATION_STOCK_CONTINUE mode=B layer=%d step=%d consumed=0 custom_dispatches=0 state_mutations=0\n",plan.layer,ctx->ablation_stats.step); }
        return false;
    }
    const uint64_t commit_begin=ctx->ablation_cpu ? bitnet_abc_now() : 0;
    const bool committed=ggml_metal_commit_ffn_plan(ctx,plan,ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_TRACE"));
    if(commit_begin) { ctx->ablation_stats.commit_ns+=bitnet_abc_now()-commit_begin; ++ctx->ablation_stats.commit_timers; }
    if(committed && ctx->ablation_count) { ++ctx->ablation_stats.commits; }
    return committed;
}

// the generic mat-vec kernel, or its 2-row Q4_0 variant if nc
static int ggml_metal_op_mul_mat_mv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    // the mat-vec store adds the residual when the table fuses the ADD after this mat-mul
    const int  n_fuse   = ggml_metal_mul_mat_mv_use_add(op) ? ggml_metal_op_try_fusion(ctx, idx, GGML_METAL_FUSION_MUL_MAT_ADD) : 1;
    const bool fuse_add = n_fuse > 1;

    const ggml_tensor * dst = op;
    const ggml_tensor * res = dst;

    if (fuse_add) {
        dst = ctx->node(idx + n_fuse - 1);
        // the residual is the other operand of the ADD, by identity: it can itself be a MUL_MAT output
        res = dst->src[0] == op ? dst->src[1] : dst->src[0];
    }

    ggml_metal_op_mul_mat_mv_encode(ctx, op, dst, res, nullptr, nullptr);

    return n_fuse;
}

int ggml_metal_op_mul_mat(ggml_metal_op_t ctx, int idx) {
    if ((ctx->ablation_mode=='B' || ctx->ablation_mode=='C') && ggml_metal_try_ffn_ablation(ctx,idx)) { return 5; }
    if (ggml_metal_bitnet_ffn_projection(ctx->node(idx)) == bitnet_ffn_projection::gate) {
        const char * mechanism = std::getenv("GGML_METAL_BITNET_FFN_MECHANISM");
        if (mechanism) {
            const bool trace = ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_TRACE");
            if (std::strcmp(mechanism,"K")==0) {
                if (ggml_metal_try_ffn_k(ctx,idx,trace)) {return 7;}
            } else if (ggml_metal_try_ffn_mechanism(ctx,idx,trace)) {
                return 5;
            }
        }
    }
    if (ggml_metal_bitnet_ffn_projection(ctx->node(idx)) == bitnet_ffn_projection::gate &&
        !std::getenv("GGML_METAL_BITNET_FFN_MECHANISM") && ggml_metal_bitnet_ffn_mode("fused") &&
        ggml_metal_try_bitnet_ffn_fused(ctx, idx, ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_TRACE"))) {
        return 5;
    }
    if (ctx->use_fusion()) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_MUL_MV_GLU) {
            ctx->count_fusions(fusion);

            return ggml_metal_op_mul_mv_glu(ctx, idx, n);
        }
    }

    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    if (ggml_metal_op_mul_mat_use_fwht(op, props_dev->max_theadgroup_memory_size)) {
        return ggml_metal_op_fwht(ctx, idx);
    }

    if (props_dev->has_simdgroup_mm && ggml_metal_op_mul_mat_use_mma(op)) {
        return ggml_metal_op_mul_mat_mma(ctx, idx);
    }

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ne00 == ne10);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    if (op->src[1]->type == GGML_TYPE_F32 && (ne00%128 == 0) &&
        (
         (
          (
           op->src[0]->type == GGML_TYPE_F32  || // TODO: helper function
           op->src[0]->type == GGML_TYPE_F16  ||
           op->src[0]->type == GGML_TYPE_BF16 ||
           op->src[0]->type == GGML_TYPE_Q1_0 ||
           op->src[0]->type == GGML_TYPE_Q2_0 ||
           op->src[0]->type == GGML_TYPE_PQ2_0 ||
           op->src[0]->type == GGML_TYPE_PTQ1_0 ||
           op->src[0]->type == GGML_TYPE_Q4_0 ||
           op->src[0]->type == GGML_TYPE_Q4_1 ||
           op->src[0]->type == GGML_TYPE_Q5_0 ||
           op->src[0]->type == GGML_TYPE_Q5_1 ||
           op->src[0]->type == GGML_TYPE_Q8_0 ||
           op->src[0]->type == GGML_TYPE_MXFP4 ||
           op->src[0]->type == GGML_TYPE_IQ4_NL ||
           false) && (ne11 >= 2 && ne11 <= 8)
         ) ||
         (
          (
           op->src[0]->type == GGML_TYPE_Q4_K ||
           op->src[0]->type == GGML_TYPE_Q5_K ||
           op->src[0]->type == GGML_TYPE_Q6_K ||
           op->src[0]->type == GGML_TYPE_Q2_K ||
           op->src[0]->type == GGML_TYPE_Q3_K ||
           false) && (ne11 >= 4 && ne11 <= 8)
         )
        )
       ) {
        // TODO: determine the optimal parameters based on grid utilization
        //       I still don't know why we should not always use the maximum available threads:
        //
        //       nsg = pipeline.maxTotalThreadsPerThreadgroup / 32
        //
        //       my current hypothesis is that the work grid is not evenly divisible for different nsg
        //       values and there can be some tail effects when nsg is high. need to confirm this
        //
        const int nsg    = 2;                 // num simdgroups per threadgroup

        // num threads along row per simdgroup
        int16_t nxpsg = 0;
        if (ne00 % 256 == 0 && ne11 < 3) {
            nxpsg = 16;
        } else if (ne00 % 128 == 0) {
            nxpsg = 8;
        } else {
            nxpsg = 4;
        }

        const int16_t nypsg  = 32/nxpsg;          // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
        const int16_t r0ptg  = nypsg*nsg;         // num src0 rows per threadgroup
              int16_t r1ptg  = 4;                 // num src1 rows per threadgroup

        // note: not sure how optimal are those across all different hardware. there might be something cleverer
        switch (ne11) {
            case 2:
                r1ptg = 2; break;
            case 3:
            case 6:
                r1ptg = 3; break;
            case 4:
            case 7:
            case 8:
                r1ptg = 4; break;
            case 5:
                r1ptg = 5; break;
            default:
                GGML_ABORT("unsupported ne11");
        };

        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_ext(lib, op, nsg, nxpsg, r1ptg);

        ggml_metal_kargs_mul_mv_ext args = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne10  =*/ ne10,
            /*.ne11  =*/ ne11,
            /*.ne12  =*/ ne12,
            /*.nb10  =*/ nb10,
            /*.nb11  =*/ nb11,
            /*.nb12  =*/ nb12,
            /*.nb13  =*/ nb13,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.r2    =*/ r2,
            /*.r3    =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + r0ptg - 1)/r0ptg), ((ne11 + r1ptg - 1)/r1ptg), ne12*ne13, 32, nsg, 1);
    } else if (ggml_metal_op_mul_mat_use_mm(op, props_dev->has_simdgroup_mm)) {
        //GGML_LOG_INFO("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        auto pipeline = ggml_metal_library_get_pipeline_mul_mm(lib, op);

        ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ ne00,
            /*.ne02 =*/ ne02,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        const size_t smem = pipeline.smem;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const int grid_x = (ne11 + nr1 - 1) / nr1;
        const int grid_y = (ne01 + nr0 - 1) / nr0;
        const int grid_z = ne12 * ne13;
        ggml_metal_encoder_dispatch_threadgroups(enc, grid_x, grid_y, grid_z, 32, nsg, 1);
        if (ggml_metal_bitnet_poc_target(op) && ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_POC_TRACE")) {
            ggml_metal_encoder_trace_bitnet(enc, pipeline, "stock-mm", ne11, grid_x, grid_y, grid_z, nsg);
            ggml_metal_encoder_trace_bitnet_node(enc, pipeline, "q-stock", ggml_metal_bitnet_layer(op->name, "Qcur-"),
                    op, op, grid_x, grid_y, grid_z, 32, nsg);
        }
        if (ggml_metal_bitnet_poc_flag("GGML_METAL_BITNET_FFN_TRACE")) {
            const char * route = ggml_metal_bitnet_ffn_stock_route(ggml_metal_bitnet_ffn_projection(op));
            if (route) {
                ggml_metal_encoder_trace_bitnet(enc, pipeline, route, ne11, grid_x, grid_y, grid_z, nsg);
                ggml_metal_encoder_trace_bitnet_node(enc, pipeline, route, ggml_metal_bitnet_ffn_layer(op),
                        op, op, grid_x, grid_y, grid_z, 32, nsg);
            }
        }
    } else {
        return ggml_metal_op_mul_mat_mv(ctx, idx);
    }

    return 1;
}

// upper bound on the number of entry tiles emitted by kernel_mul_mm_id_map0: the full tiles
// over all entries plus at most one partial tile per expert
static int64_t ggml_metal_op_mul_mat_id_max_tiles(const ggml_tensor * op) {
    const int64_t ne02 = op->src[0]->ne[2]; // n_expert
    const int64_t ne20 = op->src[2]->ne[0]; // n_expert_used
    const int64_t ne21 = op->src[2]->ne[1]; // n_token

    const int64_t n_rows = ne20*ne21;
    const int64_t tiles  = (n_rows + 31)/32 + (ne02 < n_rows ? ne02 : n_rows) - 1;

    return n_rows < tiles ? n_rows : tiles;
}

size_t ggml_metal_op_mul_mat_id_extra_tpe(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    // row map of kernel_mul_mm_id: total tile count + one
    // (expert, first entry, valid rows) descriptor per tile
    const int64_t max_tiles = ggml_metal_op_mul_mat_id_max_tiles(op);

    return ggml_type_size(GGML_TYPE_I32)*(1 + 3*max_tiles);
}

size_t ggml_metal_op_mul_mat_id_extra_ids(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert
    const int64_t ne20 = op->src[2]->ne[0]; // n_expert_used
    const int64_t ne21 = op->src[2]->ne[1]; // n_token

    // one entry per (expert used, token) pair of ids plus one padding slot per expert
    // (see kernel_mul_mm_id_map0)
    return ggml_type_size(GGML_TYPE_I32)*(ne20*ne21 + ne02);
}

size_t ggml_metal_op_mul_mat_id_extra_amax(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    GGML_UNUSED(op);

    // 2 scaling factors (8 bytes) + N_MM_NPART_AMAX per-threadgroup scales for stage-1
    return 8 + N_MM_NPART_AMAX*sizeof(float);
}

// MUL_MAT_ID with the routing-weight MUL applied in the mat-vec epilogue (see GGML_METAL_FUSION_MUL_MV_ID_MUL)
static int ggml_metal_op_mul_mv_id_mul(ggml_metal_op_t ctx, int idx, int n_fuse) {
    ggml_tensor * mm  = ctx->node(idx);
    ggml_tensor * mul = ctx->node(idx + n_fuse - 1);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(!ggml_metal_op_mul_mat_id_use_mm(mm, ggml_metal_device_get_props(ctx->dev)->has_simdgroup_mm));

    const ggml_tensor * src0  = mm->src[0];
    const ggml_tensor * src1  = mm->src[1];
    const ggml_tensor * ids   = mm->src[2];
    const ggml_tensor * scale = mul->src[0] == mm ? mul->src[1] : mul->src[0];

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_id(lib, mm, 1);

    const int nr0 = pipeline.nr0;
    const int nr1 = pipeline.nr1;
    const int nsg = pipeline.nsg;
    const size_t smem = pipeline.smem;

    const enum ggml_type t = src0->type;
    const int32_t scale_grouped =
        !(t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 || t == GGML_TYPE_Q8_0);

    ggml_metal_kargs_mul_mv_id args = {
        /*.nei0          =*/ (int32_t) ids->ne[0],
        /*.nei1          =*/ (int32_t) ids->ne[1],
        /*.nbi1          =*/ ids->nb[1],
        /*.ne00          =*/ (int32_t) src0->ne[0],
        /*.ne01          =*/ (int32_t) src0->ne[1],
        /*.ne02          =*/ (int32_t) src0->ne[2],
        /*.nb00          =*/ src0->nb[0],
        /*.nb01          =*/ src0->nb[1],
        /*.nb02          =*/ src0->nb[2],
        /*.ne10          =*/ (int32_t) src1->ne[0],
        /*.ne11          =*/ (int32_t) src1->ne[1],
        /*.ne12          =*/ (int32_t) src1->ne[2],
        /*.ne13          =*/ (int32_t) src1->ne[3],
        /*.nb10          =*/ src1->nb[0],
        /*.nb11          =*/ src1->nb[1],
        /*.nb12          =*/ src1->nb[2],
        /*.ne0           =*/ (int32_t) mul->ne[0],
        /*.ne1           =*/ (int32_t) mul->ne[1],
        /*.nb1           =*/ mul->nb[1],
        /*.nr0           =*/ nr0,
        /*.scale_grouped =*/ scale_grouped,
        /*.nes0          =*/ (int32_t) scale->ne[0],
        /*.nes1          =*/ (int32_t) scale->ne[1],
        /*.nes2          =*/ (int32_t) scale->ne[2],
        /*.nbs0          =*/ scale->nb[0],
        /*.nbs1          =*/ scale->nb[1],
        /*.nbs2          =*/ scale->nb[2],
    };

    if (ggml_is_quantized(src0->type)) {
        GGML_ASSERT(src0->ne[0] >= (int64_t) nsg*nr0);
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src0),  1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src1),  2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mul),   3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ids),   4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(scale), 5);

    const int64_t _ne1  = 1;
    const int64_t ne01  = src0->ne[1];
    const int64_t ne123 = ids->ne[0] * ids->ne[1];

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    if (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 ||
            src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_Q8_0) {
        ggml_metal_encoder_dispatch_threadgroups(enc, (int) ((ne01 + nr0 - 1)/nr0), (int) ((_ne1 + nr1 - 1)/nr1), (int) ne123, 32, nsg, 1);
    } else {
        ggml_metal_encoder_dispatch_threadgroups(enc, (int) ((ne01 + nr0*nsg - 1)/(nr0*nsg)), (int) ((_ne1 + nr1 - 1)/nr1), (int) ne123, 32, nsg, 1);
    }

    return n_fuse;
}

int ggml_metal_op_mul_mat_id(ggml_metal_op_t ctx, int idx) {
    if (ctx->use_fusion()) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_MUL_MV_GLU) {
            ctx->count_fusions(fusion);

            return ggml_metal_op_mul_mv_glu(ctx, idx, n);
        }
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_MUL_MV_ID_MUL) {
            ctx->count_fusions(fusion);

            return ggml_metal_op_mul_mv_id_mul(ctx, idx, n);
        }
    }

    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // src2 = ids
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);

    GGML_ASSERT(!ggml_is_transposed(op->src[0]));
    GGML_ASSERT(!ggml_is_transposed(op->src[1]));

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const uint32_t r2 = 1;
    const uint32_t r3 = 1;

    if (ggml_metal_op_mul_mat_id_use_mm(op, props_dev->has_simdgroup_mm)) {
        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        // extra buffers for intermediate id mapping
        ggml_metal_buffer_id bid_tpe = bid_dst;
        bid_tpe.offs += ggml_nbytes(op);

        ggml_metal_buffer_id bid_ids = bid_tpe;
        bid_ids.offs += ggml_metal_op_mul_mat_id_extra_tpe(op);

        ggml_metal_buffer_id bid_amax = bid_ids;
        bid_amax.offs += ggml_metal_op_mul_mat_id_extra_ids(op);

        // src1 prec [TAG_GGML_PREC]
        const bool use_amax = ggml_get_op_params_i32(op, 3) == GGML_PREC_F32;

        // src1 rescale factors, computed before the matmul
        // ref: https://github.com/ggml-org/llama.cpp/pull/26223
        if (use_amax) {
            ggml_metal_kargs_mul_mm_id_amax args = {
                /*.ne00 =*/ ne10,
                /*.ne01 =*/ ne11,
                /*.ne02 =*/ ne12,
                /*.nb01 =*/ nb11,
                /*.nb02 =*/ nb12,
            };

            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_amax_part(lib);

            const size_t smem = pipeline.smem;

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_amax, 2);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, N_MM_NPART_AMAX, 1, 1, 256, 1, 1);
        }

        {
            ggml_metal_kargs_mul_mm_id_map0 args = {
                ne10,
                ne11, // n_expert_used (bcast)
                nb11,
                nb12,
                ne21, // n_tokens
                ne20, // n_expert_used
                nb21,
            };

            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_map0(lib, ne02, ne20);

            const size_t smem = pipeline.smem;

            GGML_ASSERT(ne02 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  2);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, ne02, 1, 1);
        }

        ggml_metal_op_concurrency_reset(ctx);

        if (use_amax) {
            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_amax(lib);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_buffer  (enc, bid_amax, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 32, 1, 1);

            // the next kernel has to wait for the amax data
            ggml_metal_op_concurrency_reset(ctx);
        }

        {
            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id(lib, op);

            ggml_metal_kargs_mul_mm_id args = {
                /*.ne00  =*/ ne00,
                /*.ne02  =*/ ne02,
                /*.nb01  =*/ nb01,
                /*.nb02  =*/ nb02,
                /*.nb03  =*/ nb03,
                /*.ne11  =*/ ne11, // n_expert_used (bcast)
                /*.nb10  =*/ nb10,
                /*.nb11  =*/ nb11,
                /*.nb12  =*/ nb12,
                /*.nb13  =*/ nb13,
                /*.ne20  =*/ ne20, // n_expert_used
                /*.ne0   =*/ ne0,
                /*.ne1   =*/ ne1,
                /*.r2    =*/ r2,
                /*.r3    =*/ r3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  3);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  4);
            ggml_metal_encoder_set_buffer  (enc, bid_dst,  5);
            ggml_metal_encoder_set_buffer  (enc, bid_amax, 6);

            const size_t smem = pipeline.smem;

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            const int max_tiles = (int) ggml_metal_op_mul_mat_id_max_tiles(op);

            ggml_metal_encoder_dispatch_threadgroups(enc, max_tiles, (ne01 + 63)/64, 1, 128, 1, 1);
        }
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_id(lib, op, 0);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const size_t smem = pipeline.smem;

        ggml_metal_kargs_mul_mv_id args = {
            /*.nei0 =*/ ne20,
            /*.nei1 =*/ ne21,
            /*.nbi1 =*/ nb21,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.ne13 =*/ ne13,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nb1  =*/ nb1,
            /*.nr0  =*/ nr0,
            /*.scale_grouped =*/ 0,
            /*.nes0          =*/ 0,
            /*.nes1          =*/ 0,
            /*.nes2          =*/ 0,
            /*.nbs0          =*/ 0,
            /*.nbs1          =*/ 0,
            /*.nbs2          =*/ 0,
        };

        if (ggml_is_quantized(op->src[0]->type)) {
            GGML_ASSERT(ne00 >= nsg*nr0);
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst,  3);
        ggml_metal_encoder_set_buffer(enc, bid_src2, 4);
        ggml_metal_encoder_set_buffer(enc, bid_dst,  5);

        const int64_t _ne1 = 1;
        const int64_t ne123 = ne20*ne21;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0 - 1)/(nr0), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0*nsg - 1)/(nr0*nsg), (_ne1 + nr1 - 1)/nr1, ne123, 32, nsg, 1);
        }
    }

    return 1;
}

int ggml_metal_op_add_id(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_kargs_add_id args = {
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb11 =*/ nb11,
        /*.nb21 =*/ nb21,
    };

    auto pipeline = ggml_metal_library_get_pipeline_base(lib, GGML_OP_ADD_ID);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, 1, nth, 1, 1);

    return 1;
}

// the non-vec kernel reads the K/V once per tile of 8 query rows, the vec kernel once per 1-2 rows; with the KV split,
// the few threadgroups of a small batch still fill the GPU. the two kernels sum in different orders, so a batch that
// crosses these bounds as the KV grows changes kernel and its results in the last bits
static bool ggml_metal_op_flash_attn_ext_use_kv_split(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const int64_t ne00 = op->src[0]->ne[0]; // head size
    const int64_t ne01 = op->src[0]->ne[1]; // batch size
    const int64_t ne11 = op->src[5] ? op->src[5]->ne[0] : op->src[1]->ne[1]; // KV size (the row lists with kv_rows)

    return ne01 >= GGML_METAL_FA_KV_SPLIT_MIN_ROWS &&
           ne01 <= GGML_METAL_FA_KV_SPLIT_MAX_ROWS &&
           ne11 >= GGML_METAL_FA_KV_SPLIT_MIN_KV &&
           ne01*ne00 >= GGML_METAL_FA_KV_SPLIT_MIN_Q_ELEMS &&
           ne01*ne11*ne00 >= GGML_METAL_FA_KV_SPLIT_MIN_QK_ELEMS &&
           ne00 % 32 == 0;
}

// true if the batch size is small and the head size is supported by the vec kernel
static bool ggml_metal_op_flash_attn_ext_vec_eligible(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const int64_t ne00 = op->src[0]->ne[0]; // head size
    const int64_t ne01 = op->src[0]->ne[1]; // batch size

    return (ne01 < 20) && (ne00 % 32 == 0);
}

bool ggml_metal_op_flash_attn_ext_use_vec(const ggml_tensor * op) {
    return ggml_metal_op_flash_attn_ext_vec_eligible(op) && !ggml_metal_op_flash_attn_ext_use_kv_split(op);
}

// ref: https://github.com/ggml-org/llama.cpp/pull/27390
// dequantize the quantized KV cache to F16 before running the F16 flash attention kernels
static bool ggml_metal_op_flash_attn_ext_use_kv_f16(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    // depending on compute/bandwidth ratio, dequant to f16 kv is not always beneficial
    // ref: https://github.com/ggml-org/llama.cpp/pull/27390#issuecomment-5355152767
    // TODO: tune per device
    // large heads need the upfront dequant to fit the non-vec threadgroup memory
    if (op->src[0]->ne[1] < 32 &&
        (op->src[0]->ne[0] < 512 || ggml_metal_op_flash_attn_ext_use_vec(op))) {
        return false;
    }

    switch (op->src[1]->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static bool ggml_metal_op_flash_attn_ext_use_tensor(const ggml_tensor * op, bool has_tensor) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    if (!has_tensor || ggml_metal_op_flash_attn_ext_use_vec(op)) {
        return false;
    }

    const int64_t ne01 = op->src[0]->ne[1];
    const int64_t ne02 = op->src[0]->ne[2];
    const int64_t ne03 = op->src[0]->ne[3];

    const int64_t dk = op->src[1]->ne[0];
    const int64_t dv = op->src[2]->ne[0];

    const bool dk_dv_ok = (dk == 64  && dv == 64)  ||
                          (dk == 128 && dv == 128) ||
                          (dk == 192 && dv == 128) ||
                          (dk == 256 && dv == 256) ||
                          (dk == 512 && dv == 512) ||
                          (dk == 576 && dv == 512);

    if (!dk_dv_ok) {
        return false;
    }

    // large heads use fewer queries per threadgroup, so that the queries fit in threadgroup memory
    const int64_t nqptg = dk >= 512 ? OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE : OP_FLASH_ATTN_EXT_TENSOR_NQPSG;

    // few heads and small batches do not fill the GPU - the half8x8 kernel is faster there
    // TODO: tune per device
    if (((ne01 + nqptg - 1)/nqptg)*ne02*ne03*dk < 8192) {
        return false;
    }

    if (op->src[1]->type != GGML_TYPE_F16 && op->src[1]->type != GGML_TYPE_BF16 && !ggml_metal_op_flash_attn_ext_use_kv_f16(op)) {
        return false;
    }

    // with kv_rows, the attention runs over the compacted row lists (padded to 256), not over the cache
    const int64_t ne11_attn = op->src[5] ? op->src[5]->ne[0] : op->src[1]->ne[1];

    return ne11_attn % OP_FLASH_ATTN_EXT_TENSOR_NCPSG == 0;
}

// upper bound on the n_kv_max hint for the sparse path, larger hints use the dense kernels
static constexpr int32_t OP_FLASH_ATTN_EXT_SPARSE_NKV_MAX = 4096;

static int32_t ggml_metal_op_flash_attn_ext_n_kv_max_hint(const ggml_tensor * op) {
    int32_t n_kv_max = 0;
    memcpy(&n_kv_max, ((const int32_t *) op->op_params) + 4, sizeof(n_kv_max));

    return n_kv_max;
}

// structural conditions for the sparse path: a positive hint, a mask and vec kernels for (type, dk, dv)
// independent of the hint magnitude, so the index scratch reserved for a graph stays valid when the hint changes
static bool ggml_metal_op_flash_attn_ext_sparse_eligible(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    if (ggml_metal_op_flash_attn_ext_n_kv_max_hint(op) <= 0) {
        return false;
    }

    // K/V row lists are handled by the dense kernels, not by the sparse path
    if (op->src[5] != nullptr) {
        return false;
    }

    // the sparse indices are gathered from the mask
    if (!op->src[3]) {
        return false;
    }

    // vec kernel instantiations exist for these (type, dk, dv) combinations only
    const int64_t dk = op->src[1]->ne[0];
    const int64_t dv = op->src[2]->ne[0];

    const bool dk_dv_ok = (dk == 32  && dv == 32)  ||
                          (dk == 64  && dv == 64)  ||
                          (dk == 96  && dv == 96)  ||
                          (dk == 96  && dv == 64)  ||
                          (dk == 128 && dv == 128) ||
                          (dk == 128 && dv == 96)  ||
                          (dk == 192 && dv == 128) ||
                          (dk == 192 && dv == 192) ||
                          (dk == 256 && dv == 256) ||
                          (dk == 320 && dv == 256) ||
                          (dk == 512 && dv == 512) ||
                          (dk == 576 && dv == 512);

    if (!dk_dv_ok) {
        return false;
    }

    switch (op->src[1]->type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F32:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            break;
        default:
            return false;
    }

    return true;
}

// returns the n_kv_max used by the sparse kernels for this op, or 0 for the dense path
// the mask (src[3]) remains the single source of truth, n_kv_max only bounds the finite entries per row;
// the cutoff applies to the raw hint, the result is clamped to the mask width (the hint may exceed it, e.g. qwen4exp)
static int ggml_metal_op_flash_attn_ext_n_kv_max_sparse(const ggml_tensor * op) {
    if (!ggml_metal_op_flash_attn_ext_sparse_eligible(op)) {
        return 0;
    }

    const int32_t n_kv_max = ggml_metal_op_flash_attn_ext_n_kv_max_hint(op);

    if (n_kv_max > OP_FLASH_ATTN_EXT_SPARSE_NKV_MAX) {
        return 0;
    }

    return std::min<int32_t>(n_kv_max, op->src[3]->ne[0]);
}

// in some models (e.g. MLA-based), V is a view of K (the first ne20 elements of each K row);
// the dequantized V is then a view of the dequantized K and does not need its own dequant or scratch
// - ref: https://github.com/ggml-org/llama.cpp/pull/13435
static bool ggml_metal_op_flash_attn_ext_v_is_view_of_k(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = op->src[1];
    const ggml_tensor * V = op->src[2];

    return V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
}

// size of the F16 dequantized K tensor; the dequantized V tensor follows it in the same scratch buffer
static size_t ggml_metal_op_flash_attn_ext_kv_f16_k_size(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    return GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne10*ne11*ne12*ne13, 16);
}

size_t ggml_metal_op_flash_attn_ext_extra_pad(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;
    const bool use_kv_f16 = ggml_metal_op_flash_attn_ext_use_kv_f16(op);

    // when the KV is dequantized to F16, the pad kernel copies the tail chunk from the F16 scratch buffer
    // note: when V is a view of K, the dequantized V is read from the dequantized K with K's row stride
    const bool v_is_view_of_k = use_kv_f16 && ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);
    uint64_t nb11_pad = nb11;
    uint64_t nb21_pad = nb21;

    if (use_kv_f16) {
        nb11_pad = sizeof(ggml_fp16_t)*ne10;
        nb21_pad = sizeof(ggml_fp16_t)*(v_is_view_of_k ? ne10 : ne20);
    }

    // note: the non-vec kernel requires more extra memory, so always reserve for it
    GGML_ASSERT(OP_FLASH_ATTN_EXT_NCPSG >= OP_FLASH_ATTN_EXT_VEC_NCPSG);

    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (false) {
        // note: always reserve the padding space to avoid graph reallocations
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_VEC_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_VEC_NCPSG*(
                nb11_pad*ne12*ne13 +
                nb21_pad*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    } else {
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_NCPSG*(
                nb11_pad*ne12*ne13 +
                nb21_pad*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    }

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_blk(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;

    if (!has_mask) {
        return res;
    }

    // the vec layout is the larger one: reserve it whenever the vec kernel may run, so the size does not change when
    // the KV split moves the batch to the non-vec kernel
    const bool is_vec = ggml_metal_op_flash_attn_ext_vec_eligible(op);

    // this optimization is not useful for the vector kernels
    // note: always reserve the blk buffer to avoid graph reallocations
    //if (is_vec) {
    //    return res;
    //}

    const int nqptg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NQPSG : OP_FLASH_ATTN_EXT_NQPSG;
    const int ncpsg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NCPSG : OP_FLASH_ATTN_EXT_NCPSG;

    const int64_t ne1 = (ne01 + nqptg - 1)/nqptg;
    const int64_t ne0 = (ne30 + ncpsg - 1)/ncpsg;

    res += GGML_PAD(ggml_type_size(GGML_TYPE_I8)*ne0*ne1*ne32*ne33, 32);

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_tmp(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (true) {
        const int64_t nwg = 32;
        const int64_t ne01_max = std::min(ne01, 32);

        // temp buffer for writing the results from each workgroup
        // - ne20: the size of the Value head
        // -  + 2: the S and M values for each intermediate result
        res += ggml_type_size(GGML_TYPE_F32)*(ne01_max*ne02*ne03*nwg*(ne20 + 2));
    }

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_kv_f16(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (!ggml_metal_op_flash_attn_ext_use_kv_f16(op)) {
    //    return 0;
    //}

    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);

    const size_t k_size = ggml_metal_op_flash_attn_ext_kv_f16_k_size(op);

    // when V is a view of K, the dequantized V is a view of the dequantized K
    const bool v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);
    if (v_is_view_of_k) {
        return k_size;
    }

    const size_t v_size = GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne20*ne21*ne22*ne23, 16);

    return k_size + v_size;
}

// size of the sparse index lists: one list of KV indices per mask row,
// padded with -1 up to a multiple of OP_FLASH_ATTN_EXT_VEC_NCPSG
// sized from the mask width and not from the hint: the hint can change between graphs of the same
// context (e.g. llama_set_causal_attn) and the buffer reserved for the first graph has to fit the others
size_t ggml_metal_op_flash_attn_ext_extra_idx(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    if (!ggml_metal_op_flash_attn_ext_sparse_eligible(op)) {
        return 0;
    }

    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);

    const int n_kv_max_padded = GGML_PAD(std::min(ne30, OP_FLASH_ATTN_EXT_SPARSE_NKV_MAX), OP_FLASH_ATTN_EXT_VEC_NCPSG);

    return GGML_PAD(sizeof(int32_t)*(size_t) n_kv_max_padded*ne31*ne32*ne33, 16);
}

// size of the contiguous F16 buffers holding the K/V rows referenced by the kv_rows lists
// the attention over these buffers is a standard dense op (no per-tile gather in the main kernel)
size_t ggml_metal_op_flash_attn_ext_extra_rows(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * kv_rows = op->src[5];

    if (!kv_rows) {
        return 0;
    }

    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);

    const int64_t n_rows   = kv_rows->ne[0];
    const int64_t n_slices = kv_rows->ne[1];

    // note: when V is a view of K, the gathered V is read from the gathered K, so only K is gathered
    const bool v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);

    const size_t k_size = GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne10*n_rows*ne12*n_slices, 16);
    const size_t v_size = v_is_view_of_k ? 0 : GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne20*n_rows*ne22*n_slices, 16);

    return k_size + v_size;
}

// grouping the heads pays off when the K/V reads dominate: a single tile of query rows over an F16 cache
static bool ggml_metal_op_flash_attn_ext_vec_can_group_heads(const ggml_tensor * op, bool use_sparse, int nqptg) {
    return !use_sparse && op->src[5] == nullptr &&
        op->src[0]->ne[1] <= nqptg &&
        op->src[1]->type == GGML_TYPE_F16 &&
        op->src[2]->type == GGML_TYPE_F16;
}

// largest divisor of the GQA ratio n_rep that is at most nhptg_max
static int ggml_metal_op_flash_attn_ext_vec_nhptg(int64_t n_rep, int64_t nhptg_max) {
    for (int64_t nhptg = std::min(n_rep, nhptg_max); nhptg > 1; --nhptg) {
        if (n_rep % nhptg == 0) {
            return nhptg;
        }
    }

    return 1;
}

// combine the unnormalized results that nwg workgroups wrote to the temp buffer into dst
static void ggml_metal_op_flash_attn_ext_reduce(ggml_metal_op_t ctx, const ggml_tensor * op, ggml_metal_buffer_id bid_tmp, ggml_metal_buffer_id bid_dst, int32_t nwg) {
    GGML_TENSOR_LOCALS(int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(int32_t, ne,  op,         ne);

    // sanity checks
    assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0);

    GGML_ASSERT(ne01*ne02*ne03 == ne1*ne2*ne3);
    GGML_ASSERT((uint64_t)ne1*ne2*ne3 <= (1u << 31));

    // sync with the kernel that wrote the temp buffer
    ggml_metal_op_concurrency_reset(ctx);

    const int32_t nrows = ne1*ne2*ne3;

    ggml_metal_kargs_flash_attn_ext_vec_reduce args = {
        nrows,
    };

    auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(ctx->lib, op, op->src[2]->ne[0], nwg);

    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    ggml_metal_encoder_set_bytes   (ctx->enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx->enc, bid_tmp, 1);
    ggml_metal_encoder_set_buffer  (ctx->enc, bid_dst, 2);

    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, nrows, 1, 1, 32*nwg, 1, 1);
}

int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS( int32_t, nb,  op,         nb);

    GGML_ASSERT(ne00 % 4 == 0);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == op->src[2]->type);

    //GGML_ASSERT(ggml_are_same_shape (src1, src2));
    GGML_ASSERT(ne11 == ne21);
    GGML_ASSERT(ne12 == ne22);

    GGML_ASSERT(!op->src[3] || op->src[3]->type == GGML_TYPE_F16);
    GGML_ASSERT(!op->src[3] || op->src[3]->ne[1] >= op->src[0]->ne[1] &&
            "the Flash-Attention Metal kernel requires the mask to be at least n_queries big");

    // K/V row lists (ggml_flash_attn_ext_rows): src[5] holds one list of cache rows per q slice
    // the attention loops then run over the mask columns and read K/V rows through the lists
    const ggml_tensor * kv_rows = op->src[5];
    const bool has_rows = kv_rows != nullptr;

    GGML_ASSERT(!has_rows || (kv_rows->type == GGML_TYPE_I32 && ggml_is_contiguous(kv_rows)));
    GGML_ASSERT(!has_rows || (kv_rows->ne[1] == ne03 && kv_rows->ne[2] == 1 && kv_rows->ne[3] == 1));
    GGML_ASSERT(!has_rows || op->src[3]);

    // the loop bound over the mask columns / K/V row lists
    const int64_t ne11_attn = has_rows ? kv_rows->ne[0] : ne11;

    // the row lists are padded by the KV cache to a multiple of 256 (supports_op rejects other lengths), so the pad path never triggers
    GGML_ASSERT(!has_rows || ne11_attn % OP_FLASH_ATTN_EXT_NCPSG == 0);
    GGML_ASSERT(!has_rows || ne11_attn % OP_FLASH_ATTN_EXT_VEC_NCPSG == 0);

    float scale;
    float max_bias;
    float logit_softcap;

    memcpy(&scale,         ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) op->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) op->op_params) + 2, sizeof(logit_softcap));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const bool has_mask  = op->src[3] != NULL;
    const bool has_sinks = op->src[4] != NULL;

    GGML_ASSERT(!has_rows || has_mask);
    const bool has_bias  = max_bias != 0.0f;
    const bool has_scap  = logit_softcap != 0.0f;

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    GGML_ASSERT(ne01 < 65536);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_src3 = has_mask  ? ggml_metal_get_buffer_id(op->src[3]) : bid_src0;
    ggml_metal_buffer_id bid_src4 = has_sinks ? ggml_metal_get_buffer_id(op->src[4]) : bid_src0;

    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_pad = bid_dst;
    bid_pad.offs += ggml_nbytes(op);

    ggml_metal_buffer_id bid_blk = bid_pad;
    bid_blk.offs += ggml_metal_op_flash_attn_ext_extra_pad(op);

    ggml_metal_buffer_id bid_tmp = bid_blk;
    bid_tmp.offs += ggml_metal_op_flash_attn_ext_extra_blk(op);

    ggml_metal_buffer_id bid_kv_f16 = bid_tmp;
    bid_kv_f16.offs += ggml_metal_op_flash_attn_ext_extra_tmp(op);

    // sparse path: gather the finite mask entries into index lists and run the vec kernels over them
    const int n_kv_max_sparse = ggml_metal_op_flash_attn_ext_n_kv_max_sparse(op);
    const bool use_sparse = n_kv_max_sparse > 0;
    const int n_kv_max_padded = use_sparse ? GGML_PAD(n_kv_max_sparse, OP_FLASH_ATTN_EXT_VEC_NCPSG) : 0;

    const bool use_vec = ggml_metal_op_flash_attn_ext_use_vec(op);

    // with kv_rows, the non-vec path gathers the referenced rows into a contiguous F16 buffer and
    // attends over it with the F16 kernel - quantized KV is first dequantized with the F16 pass
    const bool rows_gather = has_rows && !use_vec;

    // the vec kernels dequantize the KV inline; no need for the F16 dequant pass in the sparse path
    const bool dequant_kv = !use_sparse && (rows_gather ?
        ggml_is_quantized(op->src[1]->type) : ggml_metal_op_flash_attn_ext_use_kv_f16(op));

    const bool use_kv_f16 = !use_sparse && (rows_gather ? true : dequant_kv);

    ggml_metal_buffer_id bid_idx = bid_kv_f16;
    bid_idx.offs += ggml_metal_op_flash_attn_ext_extra_kv_f16(op);

    // contiguous K/V buffers holding the rows referenced by the kv_rows lists
    ggml_metal_buffer_id bid_rows = bid_idx;
    bid_rows.offs += ggml_metal_op_flash_attn_ext_extra_idx(op);

    ggml_metal_buffer_id bid_k = bid_src1;
    ggml_metal_buffer_id bid_v = bid_src2;

    uint64_t nb10_attn = nb10;
    uint64_t nb11_attn = nb11;
    uint64_t nb12_attn = nb12;
    uint64_t nb13_attn = nb13;
    uint64_t nb20_attn = nb20;
    uint64_t nb21_attn = nb21;
    uint64_t nb22_attn = nb22;
    uint64_t nb23_attn = nb23;

    if (dequant_kv) {
        assert(ggml_metal_op_flash_attn_ext_extra_kv_f16(op) != 0);

        const bool v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);

        const int64_t nblocks1_64 = (ne10/ggml_blck_size(op->src[1]->type))*(int64_t) ne11*ne12*ne13;
        GGML_ASSERT(nblocks1_64 <= INT32_MAX);
        const int32_t nblocks1 = nblocks1_64;

        ggml_metal_buffer_id bid_v_f16 = bid_kv_f16;
        bid_v_f16.offs += ggml_metal_op_flash_attn_ext_kv_f16_k_size(op);

        auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(lib, op);
        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline0), 256);

        // K
        ggml_metal_kargs_flash_attn_ext_kv_f16 args_k = {
            /*.ne0    =*/ ne10,
            /*.ne1    =*/ ne11,
            /*.ne2    =*/ ne12,
            /*.ne3    =*/ ne13,
            /*.nb0    =*/ nb10,
            /*.nb1    =*/ nb11,
            /*.nb2    =*/ nb12,
            /*.nb3    =*/ nb13,
            /*.nblocks =*/ nblocks1,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline0);
        ggml_metal_encoder_set_bytes   (enc, &args_k, sizeof(args_k), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src1,        1);
        ggml_metal_encoder_set_buffer  (enc, bid_kv_f16, 2);

        ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks1 + nth - 1)/nth, 1, 1, nth, 1, 1);

        // V (skip when V is a view of K: the dequantized V is a view of the dequantized K)
        if (!v_is_view_of_k) {
            const int64_t nblocks2_64 = (ne20/ggml_blck_size(op->src[2]->type))*(int64_t) ne21*ne22*ne23;
            GGML_ASSERT(nblocks2_64 <= INT32_MAX);
            const int32_t nblocks2 = nblocks2_64;

            ggml_metal_kargs_flash_attn_ext_kv_f16 args_v = {
                /*.ne0    =*/ ne20,
                /*.ne1    =*/ ne21,
                /*.ne2    =*/ ne22,
                /*.ne3    =*/ ne23,
                /*.nb0    =*/ nb20,
                /*.nb1    =*/ nb21,
                /*.nb2    =*/ nb22,
                /*.nb3    =*/ nb23,
                /*.nblocks =*/ nblocks2,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args_v, sizeof(args_v), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2,        1);
            ggml_metal_encoder_set_buffer  (enc, bid_v_f16,       2);

            ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks2 + nth - 1)/nth, 1, 1, nth, 1, 1);
        }

        // the pad and attention kernels read the dequantized KV
        ggml_metal_op_concurrency_reset(ctx);

        bid_k = bid_kv_f16;
        bid_v = v_is_view_of_k ? bid_k : bid_v_f16;

        // contiguous F16 layout of the dequantized K
        nb10_attn = sizeof(ggml_fp16_t);
        nb11_attn = nb10_attn*ne10;
        nb12_attn = nb11_attn*ne11;
        nb13_attn = nb12_attn*ne12;

        // if V is a view of K, the dequantized V is read from the dequantized K with K's strides
        if (v_is_view_of_k) {
            nb20_attn = nb10_attn;
            nb21_attn = nb11_attn;
            nb22_attn = nb12_attn;
            nb23_attn = nb13_attn;
        } else {
            // contiguous F16 layout of the dequantized V
            nb20_attn = sizeof(ggml_fp16_t);
            nb21_attn = nb20_attn*ne20;
            nb22_attn = nb21_attn*ne21;
            nb23_attn = nb22_attn*ne22;
        }
    }

    // with kv_rows, gather the cache rows referenced by the lists into contiguous F16 buffers
    // the attention then runs as a standard dense op over the gathered rows (tensor or half8x8 kernel)
    if (rows_gather) {
        assert(ggml_metal_op_flash_attn_ext_extra_rows(op) != 0);

        const bool v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);

        GGML_ASSERT(ne20 % 4 == 0);

        const size_t k_size = GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne10*ne11_attn*ne12*ne03, 16);

        ggml_metal_kargs_flash_attn_ext_rows args0 = {
            /*.ne11    =*/ (int32_t) ne11_attn,
            /*.ne_12_2 =*/ ne12,
            /*.ne10    =*/ ne10,
            /*.ne20    =*/ v_is_view_of_k ? 0 : ne20, // the gathered V is a view of the gathered K
            /*.v_off   =*/ k_size,
            /*.nb11    =*/ nb11_attn,
            /*.nb12    =*/ nb12_attn,
            /*.nb21    =*/ nb21_attn,
            /*.nb22    =*/ nb22_attn,
        };

        auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_rows(lib, op);
        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline0), 64);

        ggml_metal_encoder_set_pipeline(enc, pipeline0);
        ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_k, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_v, 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(kv_rows), 3);
        ggml_metal_encoder_set_buffer  (enc, bid_rows, 4);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne11_attn, ne03, 1, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);

        // contiguous F16 layout of the gathered rows: [DK, n_rows, n_head_kv, n_streams]
        bid_k = bid_rows;

        nb10_attn = sizeof(ggml_fp16_t);
        nb11_attn = nb10_attn*ne10;
        nb12_attn = nb11_attn*ne11_attn;
        nb13_attn = nb12_attn*ne12;

        // if V is a view of K, the gathered V is read from the gathered K with K's strides
        if (v_is_view_of_k) {
            bid_v = bid_k;

            nb20_attn = nb10_attn;
            nb21_attn = nb11_attn;
            nb22_attn = nb12_attn;
            nb23_attn = nb13_attn;
        } else {
            bid_v = bid_rows;
            bid_v.offs += k_size;

            nb20_attn = sizeof(ggml_fp16_t);
            nb21_attn = nb20_attn*ne20;
            nb22_attn = nb21_attn*ne11_attn;
            nb23_attn = nb22_attn*ne22;
        }
    }

    // prefill with F16 K/V and head size 256: mul_mm-like layout, 30% faster than the half8x8 kernel at long KV
    // (GGML_METAL_FA_TB=0 falls back to the half8x8 kernel; devices that use the tensor API kernel keep it)
    static const bool fa_tb_env = !getenv("GGML_METAL_FA_TB") || atoi(getenv("GGML_METAL_FA_TB")) != 0;
    const bool use_tb = fa_tb_env && !use_sparse && !ggml_metal_op_flash_attn_ext_use_vec(op) &&
        !ggml_metal_op_flash_attn_ext_use_tensor(op, props_dev->has_tensor) &&
        (use_kv_f16 || op->src[1]->type == GGML_TYPE_F16) &&
        ne00 == 256 && ne20 == 256 && ne11_attn % 64 == 0 && ne01 >= 32 && ne02 % ne12 == 0 && ne03 % ne13 == 0 &&
        nb10_attn == sizeof(ggml_fp16_t) && nb20_attn == sizeof(ggml_fp16_t) &&
        has_mask && !has_sinks && !has_bias && !has_scap;

    if (use_tb) {
        const char * name = "kernel_flash_attn_ext_tb_f16_dk256_bk64";

        ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(lib, name);
        if (!pipeline.pipeline) {
            ggml_metal_cv_t cv = ggml_metal_cv_init();
            pipeline = ggml_metal_library_compile_pipeline(lib, name, name, cv);
            ggml_metal_cv_free(cv);
        }

        ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ (int32_t) ne11_attn,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ has_rows ? ne03 : ne13, // the gathered K/V holds one copy per stream
            /*.ns10          =*/ (int32_t) (nb11_attn/nb10_attn),
            /*.nb11          =*/ nb11_attn,
            /*.nb12          =*/ nb12_attn,
            /*.nb13          =*/ nb13_attn,
            /*.ns20          =*/ (int32_t) (nb21_attn/nb20_attn),
            /*.nb21          =*/ nb21_attn,
            /*.nb22          =*/ nb22_attn,
            /*.nb23          =*/ nb23_attn,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        // staged Q (32 rows of half), S (32 x 64 floats) and the per-row rescale factors, padded as in the kernel
        const size_t smem = 32*(ne00 + 8)*sizeof(ggml_fp16_t) + (32*(64 + 4) + 32)*sizeof(float);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  5);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, GGML_PAD(smem, 16), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + 31)/32, ne02, ne03, 32, 8, 1);

        return 1;
    }

    if (!use_sparse && ggml_metal_op_flash_attn_ext_use_tensor(op, props_dev->has_tensor)) {
        // tensor API kernel
        const int nqptg = ne00 >= 512 ? OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE : OP_FLASH_ATTN_EXT_TENSOR_NQPSG; // queries per threadgroup
        const int ncpsg = OP_FLASH_ATTN_EXT_TENSOR_NCPSG; // cache values per threadgroup
        const int nsg   = OP_FLASH_ATTN_EXT_TENSOR_NSG;

        if (has_mask) {
            assert(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

            ggml_metal_kargs_flash_attn_ext_blk args0 = {
                /*.ne01 =*/ ne01,
                /*.ne30 =*/ ne30,
                /*.ne31 =*/ ne31,
                /*.ne32 =*/ ne32,
                /*.ne33 =*/ ne33,
                /*.nb31 =*/ nb31,
                /*.nb32 =*/ nb32,
                /*.nb33 =*/ nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);

            const int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
            const int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

            ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);

            ggml_metal_op_concurrency_reset(ctx);
        }

        const int32_t ns10 = nb11_attn/nb10_attn;
        const int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ (int32_t) ne11_attn,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ has_rows ? ne03 : ne13, // the gathered K/V holds one copy per stream
            /*.ns10          =*/ ns10,
            /*.nb11          =*/ nb11_attn,
            /*.nb12          =*/ nb12_attn,
            /*.nb13          =*/ nb13_attn,
            /*.ns20          =*/ ns20,
            /*.nb21          =*/ nb21_attn,
            /*.nb22          =*/ nb22_attn,
            /*.nb23          =*/ nb23_attn,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        // shared memory layout: queries (16-bit), scores (float), probabilities (16-bit), row scale (float), rescale flag (int)
        const size_t smem = GGML_PAD(nqptg*ne00*sizeof(ggml_fp16_t) + nqptg*ncpsg*(sizeof(float) + sizeof(ggml_fp16_t)) + nqptg*sizeof(float) + sizeof(int32_t), 16);

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_tensor(lib, op, has_mask, has_sinks, has_bias, has_scap);

        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_blk,  6);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  7);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
    } else if (!use_sparse && !ggml_metal_op_flash_attn_ext_use_vec(op)) {
        // half8x8 kernel
        const int nqptg = OP_FLASH_ATTN_EXT_NQPSG; // queries per threadgroup
        const int ncpsg = OP_FLASH_ATTN_EXT_NCPSG; // cache values per simdgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 8  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = ne11_attn % ncpsg != 0;

        if (has_kvpad) {
            assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11_attn,
                /*.nb12    =*/nb12_attn,
                /*.nb13    =*/nb13_attn,
                /*.nb21    =*/nb21_attn,
                /*.nb22    =*/nb22_attn,
                /*.nb23    =*/nb23_attn,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_k,    1);
            ggml_metal_encoder_set_buffer  (enc, bid_v,    2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (has_mask) {
            assert(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

            ggml_metal_kargs_flash_attn_ext_blk args0 = {
                /*.ne01 =*/ ne01,
                /*.ne30 =*/ ne30,
                /*.ne31 =*/ ne31,
                /*.ne32 =*/ ne32,
                /*.ne33 =*/ ne33,
                /*.nb31 =*/ nb31,
                /*.nb32 =*/ nb32,
                /*.nb33 =*/ nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);

            const int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
            const int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

            ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        const int is_q = !use_kv_f16 && ggml_is_quantized(op->src[1]->type) ? 1 : 0;

        // shared memory layout (halfs unless noted):
        //   queries/attn/result: Q*(DK + 2*PAD2(DV,64) + 4*C)
        //   quantized KV scratch: 16*32*NSG (only when is_q)
        const int64_t dv_pad = GGML_PAD(ne20, 64);

        auto fa_smem = [&](int32_t nsg) -> size_t {
            const size_t smem_half = nqptg*(ne00 + 2*dv_pad + 4*ncpsg) + is_q*(16*32*nsg);
            return GGML_PAD(smem_half*sizeof(ggml_fp16_t), 16);
        };

        // simdgroups per threadgroup (a.k.a. warps)
        int32_t nsg = ne00 >= 512 ? 8 : 4;

        const size_t smem = fa_smem(nsg);

        const int32_t ns10 = nb11_attn/nb10_attn;
        const int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ (int32_t) ne11_attn,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ has_rows ? ne03 : ne13, // the gathered K/V holds one copy per stream
            /*.ns10          =*/ ns10,
            /*.nb11          =*/ nb11_attn,
            /*.nb12          =*/ nb12_attn,
            /*.nb13          =*/ nb13_attn,
            /*.ns20          =*/ ns20,
            /*.nb21          =*/ nb21_attn,
            /*.nb22          =*/ nb22_attn,
            /*.nb23          =*/ nb23_attn,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        // workgroups: with the KV split each writes its unnormalized result to the temp buffer for the reduce kernel
        const int32_t nwg = ggml_metal_op_flash_attn_ext_use_kv_split(op) ? GGML_METAL_FA_KV_SPLIT_NWG : 1;

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg, nwg, use_kv_f16, ns10, ns20);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
        ggml_metal_encoder_set_buffer  (enc, bid_blk,  7);
        ggml_metal_encoder_set_buffer  (enc, nwg > 1 ? bid_tmp : bid_dst, 8);

        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03*nwg, 32, nsg, 1);

        if (nwg > 1) {
            ggml_metal_op_flash_attn_ext_reduce(ctx, op, bid_tmp, bid_dst, nwg);
        }
    } else {
        // half4x4 kernel
        // sparse: the index lists are per query row, so a threadgroup can share KV with Q == 1 only
        auto cfg = use_sparse
                ? ggml_metal_tuning::fa_vec_baseline_cfg((int) ne00, (int) ne20)
                : ggml_metal_tuning::fa_vec_pick(
                          props_dev->gpu_family,
                          (int) op->src[1]->type,
                          (int) ne00, (int) ne20,   // dk, dv (ne00 == dk for FA)
                          ne11_attn, ne01);

        int nqptg = cfg.Q; // queries per threadgroup

        const int ncpsg = OP_FLASH_ATTN_EXT_VEC_NCPSG; // cache values per simdgroup !! sync with kernel template arguments !!

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg == 1 || nqptg == 2 || nqptg == 4);  // only instantiated Q values
        GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = !use_sparse && ne11_attn % ncpsg != 0;

        if (use_sparse) {
            assert(ggml_metal_op_flash_attn_ext_extra_idx(op) != 0);

            GGML_ASSERT(ne30 == ne11);

            ggml_metal_kargs_flash_attn_ext_vec_idx args0 = {
                /*.ne30              =*/ ne30,
                /*.ne31              =*/ ne31,
                /*.ne32              =*/ ne32,
                /*.ne33              =*/ ne33,
                /*.nb31              =*/ nb31,
                /*.nb32              =*/ nb32,
                /*.nb33              =*/ nb33,
                /*.n_kv_max          =*/ n_kv_max_sparse,
                /*.n_kv_max_padded   =*/ n_kv_max_padded,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(lib, op);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_idx,  2);

            int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline0), 256);
            nth = std::max(32, (nth/32)*32);

            ggml_metal_encoder_dispatch_threadgroups(enc, ne31, ne32, ne33, nth, 1, 1);

            need_sync = true;
        }

        if (has_kvpad) {
            assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11_attn,
                /*.nb12    =*/nb12_attn,
                /*.nb13    =*/nb13_attn,
                /*.nb21    =*/nb21_attn,
                /*.nb22    =*/nb22_attn,
                /*.nb23    =*/nb23_attn,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_k,    1);
            ggml_metal_encoder_set_buffer  (enc, bid_v,    2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        // note: for simplicity assume the K is larger or equal than V
        GGML_ASSERT(ne10 >= ne20);

        // shared memory layout (halfs unless noted):
        //   queries:      Q*NSG*PAD2(ne00, 128)
        //   attn + mask:  NSG*4*Q*C
        //   results:      2*NSG*Q*PAD2(ne20, 128)
        //   sparse idx:   NSG*C ints (when use_sparse, or when gathering K/V rows)
        const int64_t dk_pad = GGML_PAD(ne00, 128);
        const int64_t dv_pad = GGML_PAD(ne20, 128);

        auto fa_vec_smem = [&](int64_t nsg, int32_t nqptg) -> size_t {
            const size_t smem_half = (size_t) (dk_pad + 4*ncpsg + 2*dv_pad)*nqptg*nsg;
            return GGML_PAD(smem_half*sizeof(ggml_fp16_t) + ((use_sparse || has_rows) ? (size_t) nsg*ncpsg*sizeof(int) : 0), 16);
        };

        int64_t nsg = 1;

        // workgroups
        // each workgroup handles nsg*nkpsg cache values
        int32_t nwg = 1;
        if (use_sparse) {
            if (ne01 > 32) {
                // large sparse batch
                nwg = 1;
                nsg = 1;
                if (n_kv_max_padded == 640) {
                    nsg = 4; // 640 % (4*32) == 0
                } else {
                    while (2*nwg*nsg*ncpsg < n_kv_max_padded && nsg < 4) {
                        nsg *= 2;
                    }
                }
            } else {
                // small sparse batch
                nwg = 32;
                nsg = 1;
                while (2*nwg*nsg*ncpsg < n_kv_max_padded && nsg < 4) {
                    nsg *= 2;
                }
            }
        } else {
            nwg = 32;
            nsg = 1;
            while (2*nwg*nsg*ncpsg < ne11_attn && nsg < 4) {
                nsg *= 2;
            }
        }

        // fall back to baseline (Q=1) if the tuned config exceeds threadgroup memory
        if (fa_vec_smem(nsg, nqptg) > props_dev->max_theadgroup_memory_size) {
            cfg   = ggml_metal_tuning::fa_vec_baseline_cfg((int) ne00, (int) ne20);
            nqptg = cfg.Q;  // = 1
        }

        // GQA: the query heads that share a KV head run in one threadgroup, one simdgroup each, so each K/V chunk
        // is fetched once per KV head while it is cached; the per-head math is unchanged
        int nhptg = 1; // heads per threadgroup
        if (ggml_metal_op_flash_attn_ext_vec_can_group_heads(op, use_sparse, nqptg)) {
            const int64_t nhptg_max = std::min<int64_t>(OP_FLASH_ATTN_EXT_VEC_NHPTG_MAX, props_dev->max_theadgroup_memory_size/fa_vec_smem(1, nqptg));

            nhptg = ggml_metal_op_flash_attn_ext_vec_nhptg(ne02/ne12, nhptg_max);
            if (nhptg > 1) {
                nsg = 1;
            }
        }

        const int32_t ns10 = nb11_attn/nb10_attn;
        const int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext_vec args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ use_sparse ? n_kv_max_padded : (int32_t) ne11_attn,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ ns10,
            /*.nb11          =*/ nb11_attn,
            /*.nb12          =*/ nb12_attn,
            /*.nb13          =*/ nb13_attn,
            /*.ns20          =*/ ns20,
            /*.nb21          =*/ nb21_attn,
            /*.nb22          =*/ nb22_attn,
            /*.nb23          =*/ nb23_attn,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
            /*.n_kv_max_padded =*/ n_kv_max_padded,
        };

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_vec(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, use_sparse, has_rows, nqptg, cfg.NE, nsg, nwg, nhptg, use_kv_f16, ns10, ns20);

        GGML_ASSERT(nhptg*nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, use_sparse ? bid_idx : (has_rows ? ggml_metal_get_buffer_id(kv_rows) : bid_src0), 8);

        const size_t smem = nhptg*fa_vec_smem(nsg, nqptg);

        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        if (nwg == 1) {
            // using 1 workgroup -> write the result directly into dst
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_dst, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02/nhptg, ne03*nwg, 32, nsg*nhptg, 1);
        } else {
            // write the results from each workgroup into a temp buffer
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_tmp, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02/nhptg, ne03*nwg, 32, nsg*nhptg, 1);

            ggml_metal_op_flash_attn_ext_reduce(ctx, op, bid_tmp, bid_dst, nwg);
        }
    }

    return 1;
}

static int ggml_metal_op_bin_single(ggml_metal_op_t ctx, int idx, bool allow_fusion) {
    int n_fuse = 1;
    const ggml_metal_fusion * fusion = nullptr;

    const bool use_fusion = allow_fusion && ctx->use_fusion();

    if (use_fusion) {
        int n = 1;
        fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        n_fuse = n;

        // snake activation autofuse: mul -> sin -> sqr -> mul -> add
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_SNAKE) {
            ctx->count_fusions(fusion);
            return ggml_metal_op_snake_fused(ctx, idx);
        }

        // MoE output reduction: experts * weights -> weighted sum
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_MOE_REDUCE) {
            ctx->count_fusions(fusion);
            return ggml_metal_op_moe_reduce(ctx, idx);
        }

        // sign vector + hadamard mul_mat: mul -> reshape -> mul_mat
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_FWHT_SIGNED) {
            if (ggml_metal_op_fwht_signed_elidable(ctx, idx)) {
                ctx->count_fusions(fusion);
                return ggml_metal_op_fwht_signed(ctx, idx);
            }

            fusion = nullptr;
            n_fuse = 1;
        }
    }

    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const int debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const bool f16_add =
        op->op == GGML_OP_ADD &&
        op->src[0]->type == GGML_TYPE_F16 &&
        op->src[1]->type == GGML_TYPE_F16 &&
        op->type == GGML_TYPE_F16;

    GGML_ASSERT(f16_add || ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(f16_add || ggml_is_contiguous_rows(op->src[1]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.offs =*/ 0,
        /*.o1   =*/ { bid_src1.offs },
    };

    // c[0] = add(a,    b[0])
    // c[1] = add(c[0], b[1])
    // c[2] = add(c[1], b[2])
    // ...
    if (use_fusion && fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_ADD_CHAIN) {
        // the offsets of the fused addends are relative to the start of the src1 buffer
        for (int i = 1; i < n_fuse; i++) {
            args.o1[i] = ggml_metal_get_buffer_id(ctx->node(idx + i)->src[1]).offs;
        }

        ctx->count_fusions(fusion);

        if (debug_fusion > 1) {
            GGML_LOG_DEBUG("%s: fuse: ADD x %d\n", __func__, n_fuse);
        }
    }

    // the offsets of src1 and all fused buffers are relative to the start of the src1 buffer
    bid_src1.offs = 0;

    struct ggml_metal_pipeline_with_params pipeline;

    pipeline = ggml_metal_library_get_pipeline_bin(lib, op, n_fuse);

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne10 = ne10/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

    int bitnet_grid_x, bitnet_grid_y, bitnet_grid_z, bitnet_threads;
    if (pipeline.cnt) {
        const int nth = MIN(256, args.ne0);

        bitnet_grid_x = (args.ne0 + nth - 1)/nth;
        bitnet_grid_y = ggml_nrows(op);
        bitnet_grid_z = 1;
        bitnet_threads = nth;
        ggml_metal_encoder_dispatch_threadgroups(enc, bitnet_grid_x, bitnet_grid_y, 1, nth, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        int nth = 1;

        while (2*nth < args.ne0 && nth < nth_max) {
            nth *= 2;
        }

        bitnet_grid_x = ne01;
        bitnet_grid_y = ne02;
        bitnet_grid_z = ne03;
        bitnet_threads = nth;
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    if (ctx->numerics_capture) {ggml_metal_bitnet_numerics_dispatch(ctx,op,ctx->node(idx+n_fuse-1),pipeline,bitnet_grid_x,bitnet_grid_y,bitnet_grid_z,bitnet_threads,1,0,0,n_fuse);}
    const int ffn_layer = ggml_metal_bitnet_layer(op->name, "ffn_sub_out-");
    if (ffn_layer >= 0 && std::getenv("GGML_METAL_BITNET_FFN_TRACE")) {
        ggml_metal_encoder_trace_bitnet_stage(enc, pipeline, "ffn-mul", op->ne[1]);
        ggml_metal_encoder_trace_bitnet_node(enc, pipeline, "ffn-mul", ffn_layer, op, ctx->node(idx + n_fuse - 1),
                bitnet_grid_x, bitnet_grid_y, bitnet_grid_z, bitnet_threads, 1);
    }
    return n_fuse;
}

int ggml_metal_op_bin(ggml_metal_op_t ctx, int idx) {
    return ggml_metal_op_bin_single(ctx, idx, true);
}

int ggml_metal_op_silu_back(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    auto pipeline = ggml_metal_library_get_pipeline_silu_back(lib, op);

    const int64_t ne = ggml_nelements(op);

    ggml_metal_kargs_silu_back args = {
        /*.ne =*/ ne,
    };

    int arg_idx{0};

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), arg_idx++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), arg_idx++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), arg_idx++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), arg_idx++);

    const int nth = std::min<int64_t>(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne);
    const int64_t n = (ne + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_l2_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_kargs_l2_norm args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.eps   =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_l2_norm(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_group_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t ngrp = ((const int32_t *) op->op_params)[0];

    float eps;
    memcpy(&eps, op->op_params + 1, sizeof(float));

    ggml_metal_kargs_group_norm args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ngrp =*/ ngrp,
        /*.eps  =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_group_norm(lib, op);

    int nth = 32; // SIMD width
    //while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
    //    nth *= 2;
    //}

    //nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    //nth = std::min(nth, ne00/4);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ngrp, 1, 1, nth, 1, 1);

    return 1;
}

// slot i (1 or 2) of the fused norm kernel reads t, broadcast over the norm rows
static void ggml_metal_op_norm_set_operand(ggml_metal_kargs_norm & args, ggml_metal_buffer_id * bid_fuse, int i, const ggml_tensor * t) {
    bid_fuse[i - 1] = ggml_metal_get_buffer_id(t);

    args.nef1[i] = t->ne[1];
    args.nef2[i] = t->ne[2];
    args.nef3[i] = t->ne[3];

    args.nbf1[i] = t->nb[1];
    args.nbf2[i] = t->nb[2];
    args.nbf3[i] = t->nb[3];
}

static constexpr int GGML_METAL_RMS_NORM_FWHT_N_OPS = 4;

using ggml_metal_fwht_tg_getter = ggml_metal_pipeline_with_params (*)(ggml_metal_library_t lib, int n, int nth);

// a fused threadgroup FWHT kernel for blocks of n, with *nth threads per threadgroup (the fallback width if the default does not fit)
static ggml_metal_pipeline_with_params ggml_metal_op_fwht_tg_pipeline(ggml_metal_library_t lib, ggml_metal_fwht_tg_getter get, int64_t n, int * nth) {
    *nth = GGML_METAL_FWHT_TG_NT;
    auto pipeline = get(lib, n, *nth);
    if (!pipeline.pipeline || ggml_metal_pipeline_max_theads_per_threadgroup(pipeline) < *nth) {
        *nth = GGML_METAL_FWHT_TG_NT_FALLBACK;
        pipeline = get(lib, n, *nth);
    }
    return pipeline;
}

// returns the number of fused nodes, or 0 when the group cannot fuse
static int ggml_metal_op_rms_norm_fwht(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * norm  = ctx->node(idx);
    ggml_tensor * mul   = ctx->node(idx + 1);
    ggml_tensor * mul_s = ctx->node(idx + 2);
    ggml_tensor * mm    = ctx->node(idx + 3);

    // the fusion table (ggml_metal_fusion_check_rms_norm_fwht) verified the consumers; the weighted norm is written
    // when it has consumers outside the group
    const bool write_norm = ctx->rms_norm_fwht_write_norm(idx);

    const int64_t n = mm->src[0]->ne[0];

    int nth = 0;
    auto pipeline = ggml_metal_op_fwht_tg_pipeline(ctx->lib, ggml_metal_library_get_pipeline_rms_norm_fwht, n, &nth);
    if (!pipeline.pipeline || ggml_metal_pipeline_max_theads_per_threadgroup(pipeline) < nth) {
        return 0;
    }

    ggml_metal_kargs_rms_norm_fwht args = {
        /*.ne00       =*/ (int32_t) norm->ne[0],
        /*.n_blk      =*/ (int32_t) (norm->ne[0] / n),
        /*.eps        =*/ ggml_get_op_params_f32(norm, 0),
        /*.write_norm =*/ write_norm ? 1 : 0,
    };

    ggml_metal_op_fusion_concurrency(ctx, idx, GGML_METAL_RMS_NORM_FWHT_N_OPS);

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(norm->src[0]),  1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mul->src[1]),   2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mul_s->src[1]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mm),            4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(write_norm ? mul : mm), 5);

    ggml_metal_encoder_dispatch_threadgroups(enc, ggml_nrows(norm)*args.n_blk, 1, 1, nth, 1, 1);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: RMS_NORM + MUL + MUL + MUL_MAT (signed FWHT), write_norm = %d\n", __func__, args.write_norm);
    }

    return GGML_METAL_RMS_NORM_FWHT_N_OPS;
}

static constexpr int GGML_METAL_CONT_FWHT_N_OPS = 3;

// returns the number of fused nodes, or 0 when the group cannot fuse
static int ggml_metal_op_cont_fwht(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * cont = ctx->node(idx);
    ggml_tensor * mul  = ctx->node(idx + 1);
    ggml_tensor * mm   = ctx->node(idx + 2);

    // the fusion table (ggml_metal_fusion_check_cont_fwht) verified that no intermediate has another consumer
    const ggml_tensor * signs = ggml_metal_fwht_signed_operands(mul).signs;
    const ggml_tensor * src   = cont->src[0];

    const int64_t n = mm->src[0]->ne[0];

    int nth = 0;
    auto pipeline = ggml_metal_op_fwht_tg_pipeline(ctx->lib, ggml_metal_library_get_pipeline_fwht_gather, n, &nth);
    if (!pipeline.pipeline || ggml_metal_pipeline_max_theads_per_threadgroup(pipeline) < nth) {
        return 0;
    }

    ggml_metal_kargs_fwht_gather args = {
        /*.nrows =*/ (int32_t) (ggml_nelements(cont) / n),
        /*.n_blk =*/ (int32_t) (signs->ne[0] / n),
        /*.ne0   =*/ (int32_t) src->ne[0],
        /*.ne1   =*/ (int32_t) src->ne[1],
        /*.ne2   =*/ (int32_t) src->ne[2],
        /*.nb0   =*/ src->nb[0],
        /*.nb1   =*/ src->nb[1],
        /*.nb2   =*/ src->nb[2],
        /*.nb3   =*/ src->nb[3],
    };

    ggml_metal_op_fusion_concurrency(ctx, idx, GGML_METAL_CONT_FWHT_N_OPS);

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src),   1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mm),    2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(signs), 3);

    ggml_metal_encoder_dispatch_threadgroups(enc, args.nrows, 1, 1, nth, 1, 1);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: CONT + MUL + MUL_MAT (signed FWHT gather)\n", __func__);
    }

    return GGML_METAL_CONT_FWHT_N_OPS;
}

// RMS_NORM + MUL + ROPE (+ SET_ROWS): one dispatch normalizes, weights and rotates the rows, and writes them to the ROPE
// output or, with SET_ROWS, straight into the indexed dst rows
static int ggml_metal_op_rms_norm_rope(ggml_metal_op_t ctx, int idx, int n_fuse) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * norm = ctx->node(idx);
    const ggml_tensor * mul  = ctx->node(idx + 1);
    const ggml_tensor * rope = ctx->node(idx + 2);
    const ggml_tensor * sr   = n_fuse == 4 ? ctx->node(idx + 3) : nullptr;

    const ggml_tensor * x   = norm->src[0];
    const ggml_tensor * dst = sr ? sr : rope;

    float eps;
    memcpy(&eps, norm->op_params, sizeof(float));

    const int32_t * rp = (const int32_t *) rope->op_params;

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   rp +  5, sizeof(float));
    memcpy(&freq_scale,  rp +  6, sizeof(float));
    memcpy(&ext_factor,  rp +  7, sizeof(float));
    memcpy(&attn_factor, rp +  8, sizeof(float));
    memcpy(&beta_fast,   rp +  9, sizeof(float));
    memcpy(&beta_slow,   rp + 10, sizeof(float));

    ggml_metal_op_fusion_concurrency(ctx, idx, n_fuse);

    ggml_metal_kargs_norm_rope args = {
        /*.ne00        =*/ (int32_t) x->ne[0],
        /*.ne01        =*/ (int32_t) x->ne[1],
        /*.ne02        =*/ (int32_t) x->ne[2],
        /*.nb01        =*/ x->nb[1],
        /*.nb02        =*/ x->nb[2],
        /*.nb1         =*/ rope->nb[1],
        /*.nb2         =*/ rope->nb[2],
        /*.nb_set      =*/ sr ? sr->nb[1] : 0,
        /*.eps         =*/ eps,
        /*.n_ctx_orig  =*/ rp[4],
        /*.freq_base   =*/ freq_base,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /*.src2        =*/ rope->src[2] != nullptr,
        /*.set_rows    =*/ sr != nullptr,
        /*.idx_i32     =*/ sr && sr->src[1]->type == GGML_TYPE_I32,
    };

    auto pipeline = ggml_metal_library_get_pipeline_norm_rope(lib, rp[2] == GGML_ROPE_TYPE_NEOX, dst->type);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(x),            1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mul->src[1]),  2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(rope->src[1]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(rope->src[2] ? rope->src[2] : x), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(sr ? sr->src[1] : x), 5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst),          6);

    // one simdgroup per row, nsg rows per threadgroup
    const int nsg = 4;

    ggml_metal_encoder_dispatch_threadgroups(enc, (args.ne01 + nsg - 1)/nsg, args.ne02, 1, 32, nsg, 1);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: RMS_NORM + MUL + ROPE%s\n", __func__, sr ? " + SET_ROWS" : "");
    }

    return n_fuse;
}

// RMS_NORM + MUL + MUL_MAT x N: encode only the N mat-vecs; each one normalizes the norm input with the norm weight
static int ggml_metal_op_rms_norm_mul_mv(ggml_metal_op_t ctx, int idx, int n_fuse) {
    const ggml_tensor * norm = ctx->node(idx);
    const ggml_tensor * mul  = ctx->node(idx + 1);

    ggml_metal_op_fusion_concurrency(ctx, idx, n_fuse);

    for (int i = 2; i < n_fuse; ++i) {
        const ggml_tensor * mm = ctx->node(idx + i);

        ggml_metal_op_mul_mat_mv_encode(ctx, mm, mm, mm, norm, mul->src[1]);
    }

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: RMS_NORM + MUL + %d MUL_MAT\n", __func__, n_fuse - 2);
    }

    return n_fuse;
}

int ggml_metal_op_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion();

    const int debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_norm args = {
        /*.ne00   =*/ ne00,
        /*.ne00_t =*/ ne00 % 4 == 0 ? ne00/4 : ne00,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.eps    =*/ eps,
        /*.nef1   =*/ { ne01 },
        /*.nef2   =*/ { ne02 },
        /*.nef3   =*/ { ne03 },
        /*.nbf1   =*/ { nb01 },
        /*.nbf2   =*/ { nb02 },
        /*.nbf3   =*/ { nb03 },
        /*.scale  =*/ 1.0f,
        /*.bias   =*/ 0.0f,
    };

    int n_fuse = 1;
    bool fused_norm_scale = false;

    int epilogue = GGML_METAL_NORM_EPI_NONE;

    ggml_metal_buffer_id bid_fuse[2] = { bid_src0, bid_src0 };

    // d[0] = norm(a)
    // d[1] = mul(d[0], b) or scale(d[0])
    // d[2] = add(d[1], c)
    if (use_fusion) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);

        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_RMS_NORM_MUL_MV) {
            ctx->count_fusions(fusion);
            return ggml_metal_op_rms_norm_mul_mv(ctx, idx, n);
        }

        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_RMS_NORM_ROPE) {
            ctx->count_fusions(fusion);
            return ggml_metal_op_rms_norm_rope(ctx, idx, n);
        }

        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_RMS_NORM_FWHT) {
            const int n_fwht = ggml_metal_op_rms_norm_fwht(ctx, idx);
            if (n_fwht > 0) {
                ctx->count_fusions(fusion);
                return n_fwht;
            }

            // no fused pipeline fits this device: run the norm alone
            fusion = nullptr;
        }

        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_RMS_NORM_GATE) {
            n_fuse   = n;
            epilogue = GGML_METAL_NORM_EPI_SILU_GATE;

            ctx->count_fusions(fusion);

            ggml_metal_op_norm_set_operand(args, bid_fuse, 1, ctx->node(idx + 1)->src[1]);
            ggml_metal_op_norm_set_operand(args, bid_fuse, 2, ctx->node(idx + 2)->src[0]);

            if (debug_fusion > 1) {
                GGML_LOG_DEBUG("%s: fuse: %s + MUL + UNARY(SILU) + MUL\n", __func__, ggml_op_name(op->op));
            }
        }

        if (fusion && (ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL || ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL_ADD)) {
            n_fuse = n;

            ctx->count_fusions(fusion);

            for (int i = 1; i < n_fuse; i++) {
                ggml_metal_op_norm_set_operand(args, bid_fuse, i, ctx->node(idx + i)->src[1]);
            }

            if (debug_fusion > 1) {
                if (n_fuse == 2) {
                    GGML_LOG_DEBUG("%s: fuse: %s + MUL\n", __func__, ggml_op_name(op->op));
                }
                if (n_fuse == 3) {
                    GGML_LOG_DEBUG("%s: fuse: %s + MUL + ADD\n", __func__, ggml_op_name(op->op));
                }
            }
        }

        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_SCALE) {
            n_fuse = n;
            fused_norm_scale = true;

            ctx->count_fusions(fusion);

            const ggml_tensor * scale_node = ctx->node(idx + 1);
            args.scale = ggml_get_op_params_f32(scale_node, 0);
            args.bias  = ggml_get_op_params_f32(scale_node, 1);

            if (debug_fusion > 1) {
                GGML_LOG_DEBUG("%s: fuse: %s + SCALE\n", __func__, ggml_op_name(op->op));
            }
        }
    }

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    auto pipeline = fused_norm_scale ?
        ggml_metal_library_get_pipeline_norm_scale(lib, op) :
        ggml_metal_library_get_pipeline_norm(lib, op, n_fuse, epilogue);

    int nth = 32; // SIMD width

    while (nth < args.ne00_t && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (args.ne00_t + 31)/32*32);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0,    1);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[0], 2);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[1], 3);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,     4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    if (ctx->numerics_capture) {ggml_metal_bitnet_numerics_dispatch(ctx,op,ctx->node(idx+n_fuse-1),pipeline,ne01,ne02,ne03,nth,1,0,1,n_fuse);}
    const int ffn_layer = ggml_metal_bitnet_layer(op->src[0]->name, "ffn_sub_out-");
    if (ffn_layer >= 0 && std::getenv("GGML_METAL_BITNET_FFN_TRACE")) {
        std::fprintf(stderr, "BITNET_FFN_NORM_BIND input=%s x=%p+%zu output=%s y=%p+%zu\n",
                op->src[0]->name, bid_src0.metal, bid_src0.offs, ctx->node(idx + n_fuse - 1)->name, bid_dst.metal, bid_dst.offs);
        ggml_metal_encoder_trace_bitnet_stage(enc, pipeline, "ffn-subnorm", op->ne[1]);
        ggml_metal_encoder_trace_bitnet_node(enc, pipeline, "ffn-subnorm", ffn_layer, op,
                ctx->node(idx + n_fuse - 1), ne01, ne02, ne03, nth, 1);
    }
    return n_fuse;
}

int ggml_metal_op_rope(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // make sure we have one or more position id(ne10) per token(ne02)
    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    const int nth = std::min(1024, ne00);

    const int n_past     = ((const int32_t *) op->op_params)[0];
    const int n_dims     = ((const int32_t *) op->op_params)[1];
  //const int mode       = ((const int32_t *) op->op_params)[2];
    // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
    const int n_ctx_orig = ((const int32_t *) op->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) op->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) op->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) op->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) op->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) op->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) op->op_params + 10, sizeof(float));

    // mrope
    const int sect_0 = ((const int32_t *) op->op_params)[11];
    const int sect_1 = ((const int32_t *) op->op_params)[12];
    const int sect_2 = ((const int32_t *) op->op_params)[13];
    const int sect_3 = ((const int32_t *) op->op_params)[14];

    const int n_offs = ((const int32_t *) op->op_params)[15];

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const bool inplace = op->data == op->src[0]->data;

    ggml_metal_kargs_rope args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.ne03        =*/ ne03,
        /*.nb00        =*/ nb00,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne0         =*/ ne0,
        /*.ne1         =*/ ne1,
        /*.ne2         =*/ ne2,
        /*.ne3         =*/ ne3,
        /*.nb0         =*/ nb0,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.n_past      =*/ n_past,
        /*.n_dims      =*/ n_dims,
        /*.n_offs      =*/ n_offs,
        /*.n_ctx_orig  =*/ n_ctx_orig,
        /*.freq_base   =*/ freq_base,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /* sect_0      =*/ sect_0,
        /* sect_1      =*/ sect_1,
        /* sect_2      =*/ sect_2,
        /* sect_3      =*/ sect_3,
        /* src2        =*/ op->src[2] != nullptr,
        /* inplace     =*/ inplace,
        /* sin_sign    =*/ 1.0f,
    };

    auto pipeline = ggml_metal_library_get_pipeline_rope(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_im2col(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t p0 = ((const int32_t *)(op->op_params))[2];
    const int32_t p1 = ((const int32_t *)(op->op_params))[3];
    const int32_t d0 = ((const int32_t *)(op->op_params))[4];
    const int32_t d1 = ((const int32_t *)(op->op_params))[5];

    const bool is_2D = ((const int32_t *)(op->op_params))[6] == 1;

    const int32_t N  = op->src[1]->ne[is_2D ? 3 : 2];
    const int32_t IC = op->src[1]->ne[is_2D ? 2 : 1];
    const int32_t IH = is_2D ? op->src[1]->ne[1] : 1;
    const int32_t IW =         op->src[1]->ne[0];

    const int32_t KH = is_2D ? op->src[0]->ne[1] : 1;
    const int32_t KW =         op->src[0]->ne[0];

    const int32_t OH = is_2D ? op->ne[2] : 1;
    const int32_t OW =         op->ne[1];

    const int32_t CHW = IC * KH * KW;

    const uint64_t ofs0 = op->src[1]->nb[is_2D ? 3 : 2] / 4;
    const uint64_t ofs1 = op->src[1]->nb[is_2D ? 2 : 1] / 4;

    ggml_metal_kargs_im2col args = {
        /*.ofs0 =*/ ofs0,
        /*.ofs1 =*/ ofs1,
        /*.IW   =*/ IW,
        /*.IH   =*/ IH,
        /*.CHW  =*/ CHW,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
        /*.N    =*/ N,
        /*.KH   =*/ KH,
        /*.KW   =*/ KW,
        /*.KHW  =*/ KH * KW,
    };

    auto pipeline = ggml_metal_library_get_pipeline_im2col(lib, op);

    if (KH*KW <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        const uint64_t ntptg0 = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)/(KH*KW), N);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, IC, OH, OW, ntptg0, KH, KW);
    } else {
        const uint64_t n_threads = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), N);
        const int64_t  quotient  = N / n_threads + (N % n_threads > 0 ? 1 : 0);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, quotient * CHW, OH, OW, n_threads, 1, 1);
    }

    return 1;
}

int ggml_metal_op_conv_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t *) op->op_params)[0];
    const int32_t s1 = ((const int32_t *) op->op_params)[1];
    const int32_t p0 = ((const int32_t *) op->op_params)[2];
    const int32_t p1 = ((const int32_t *) op->op_params)[3];
    const int32_t d0 = ((const int32_t *) op->op_params)[4];
    const int32_t d1 = ((const int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d args = {
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.IW   =*/ ne10,
        /*.IH   =*/ ne11,
        /*.KW   =*/ ne00,
        /*.KH   =*/ ne01,
        /*.IC   =*/ ne02,
        /*.OC   =*/ ne03,
        /*.OW   =*/ ne0,
        /*.OH   =*/ ne1,
        /*.N    =*/ ne3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_2d(lib, op);

    int nth = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    const uint64_t n_out = ggml_nelements(op);

    uint64_t tg = (n_out + nth - 1)/nth;
    tg = std::max<uint64_t>(tg, 1);
    tg = std::min<uint64_t>(tg, (uint64_t) std::numeric_limits<int>::max());

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_conv_2d_dw(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t *) op->op_params)[0];
    const int32_t s1 = ((const int32_t *) op->op_params)[1];
    const int32_t p0 = ((const int32_t *) op->op_params)[2];
    const int32_t p1 = ((const int32_t *) op->op_params)[3];
    const int32_t d0 = ((const int32_t *) op->op_params)[4];
    const int32_t d1 = ((const int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d_dw args = {
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb03,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.IW   =*/ ne10,
        /*.IH   =*/ ne11,
        /*.KW   =*/ ne00,
        /*.KH   =*/ ne01,
        /*.C    =*/ ne12,
        /*.OW   =*/ ne0,
        /*.OH   =*/ ne1,
        /*.N    =*/ ne13,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
    };

    const bool use_tiled = (nb12 < nb10);

    auto pipeline = ggml_metal_library_get_pipeline_conv_2d_dw(lib, op, use_tiled);

    int nth = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    const int32_t OW = ne0;
    const int32_t OH = ne1;
    const int32_t C  = ne12;
    const int32_t N  = ne13;

    const int tg_x = use_tiled ? (C + nth - 1) / nth : (OW + nth - 1) / nth;
    const int tg_y = OH;
    const int tg_z = use_tiled ? OW * N : C * N;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg_x, tg_y, tg_z, nth, 1, 1);

    return 1;
}

int ggml_metal_op_conv_3d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    // 1. Extract standard dimensions and byte strides
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // 2. Extract hyperparams from op_params
    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t s2 = ((const int32_t *)(op->op_params))[2];
    const int32_t p0 = ((const int32_t *)(op->op_params))[3];
    const int32_t p1 = ((const int32_t *)(op->op_params))[4];
    const int32_t p2 = ((const int32_t *)(op->op_params))[5];
    const int32_t d0 = ((const int32_t *)(op->op_params))[6];
    const int32_t d1 = ((const int32_t *)(op->op_params))[7];
    const int32_t d2 = ((const int32_t *)(op->op_params))[8];
    const int32_t IC = ((const int32_t *)(op->op_params))[9];
    const int32_t N  = ((const int32_t *)(op->op_params))[10];
    const int32_t OC = ((const int32_t *)(op->op_params))[11];

    // 3. Build the parameter struct using the macro-generated variables
    ggml_metal_kargs_conv_3d args = {
        /*.IW =*/ (int32_t)op->src[1]->ne[0],
        /*.IH =*/ (int32_t)op->src[1]->ne[1],
        /*.ID =*/ (int32_t)op->src[1]->ne[2],
        /*.OW =*/ (int32_t)op->ne[0],
        /*.OH =*/ (int32_t)op->ne[1],
        /*.OD =*/ (int32_t)op->ne[2],
        /*.KW =*/ (int32_t)op->src[0]->ne[0],
        /*.KH =*/ (int32_t)op->src[0]->ne[1],
        /*.KD =*/ (int32_t)op->src[0]->ne[2],
        s0, s1, s2,
        p0, p1, p2,
        d0, d1, d2,
        IC, N, OC,
        nb00, nb01, nb02, nb03, // Weight strides
        nb10, nb11, nb12, nb13, // Input strides
        nb0,  nb1,  nb2,  nb3   // Output strides
    };

    // 4. Fetch the JIT pipeline
    auto pipeline = ggml_metal_library_get_pipeline_conv_3d(lib, op);

    // 5. Grid mapping
    int nth0 = 32; // Standard SIMD width for Apple Silicon
    int nth1 = 1;
    int nth2 = 1;

    int64_t spatial_volume = args.OW * args.OH * args.OD;

    int ntg0 = (spatial_volume + nth0 - 1) / nth0;
    int ntg1 = args.OC;
    int ntg2 = args.N;

    // 6. Bind and Dispatch via the ggml C wrapper
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg0, ntg1, ntg2, nth0, nth1, nth2);

    return 1;
}

int ggml_metal_op_conv_transpose_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[1];
    const int32_t IL = op->src[1]->ne[0];

    const int32_t K  = op->src[0]->ne[0];

    const int32_t OL = op->ne[0];
    const int32_t OC = op->ne[1];

    ggml_metal_kargs_conv_transpose_1d args = {
        /*.IC  =*/ IC,
        /*.IL  =*/ IL,
        /*.K   =*/ K,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_1d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, OL, OC, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_col2im_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t OC = ((const int32_t *)(op->op_params))[1];
    const int32_t p0 = ((const int32_t *)(op->op_params))[2];

    const int32_t K_OC  = (int32_t) op->src[0]->ne[0];
    const int32_t T_in  = (int32_t) op->src[0]->ne[1];
    const int32_t K     = K_OC / OC;
    const int32_t T_out = (int32_t) op->ne[0];

    ggml_metal_kargs_col2im_1d args = {
        /*.T_in  =*/ T_in,
        /*.T_out =*/ T_out,
        /*.OC    =*/ OC,
        /*.K     =*/ K,
        /*.K_OC  =*/ K_OC,
        /*.s0    =*/ s0,
        /*.p0    =*/ p0,
    };

    auto pipeline = ggml_metal_library_get_pipeline_col2im_1d(lib, op);

    const int total = T_out * OC;
    const int nth   = 256;
    const int ntg   = (total + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

// Dispatch the fused snake kernel from the matched mul -> sin -> sqr -> mul -> add chain.
// idx points at the leading mul. The caller has validated the chain.
int ggml_metal_op_snake_fused(ggml_metal_op_t ctx, int idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * mul0 = ctx->node(idx + 0);
    const ggml_tensor * sqr  = ctx->node(idx + 2);
    const ggml_tensor * mul1 = ctx->node(idx + 3);
    ggml_tensor *       add  = ctx->node(idx + 4);

    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];
    const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    const int     T     = (int) x->ne[0];
    const int     C     = (int) x->ne[1];
    const int64_t total = (int64_t) T * C;

    // the encode loop pre-checked the leading mul only, check the rest of the chain
    for (int i = 1; i < 5; ++i) {
        if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);

            break;
        }
    }

    auto pipeline = ggml_metal_library_get_pipeline_snake(lib, x->type);

    ggml_metal_kargs_snake args = {
        /*.T  =*/ T,
        /*.C  =*/ C,
        /*.ne =*/ total,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(x),     1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(a),     2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(inv_b), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(add),   4);

    const int     nth = 256;
    const int64_t ntg = (total + nth - 1) / nth;
    GGML_ASSERT(ntg <= INT32_MAX);
    ggml_metal_encoder_dispatch_threadgroups(enc, (int) ntg, 1, 1, nth, 1, 1);

    return 5;
}

int ggml_metal_op_conv_transpose_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[2];
    const int32_t IH = op->src[1]->ne[1];
    const int32_t IW = op->src[1]->ne[0];

    const int32_t KH = op->src[0]->ne[1];
    const int32_t KW = op->src[0]->ne[0];

    const int32_t OW = op->ne[0];
    const int32_t OH = op->ne[1];
    const int32_t OC = op->ne[2];
    const int32_t N  = op->src[1]->ne[3];

    ggml_metal_kargs_conv_transpose_2d args = {
        /*.IC  =*/ IC,
        /*.IH  =*/ IH,
        /*.IW  =*/ IW,
        /*.KH  =*/ KH,
        /*.KW  =*/ KW,
        /*.OC  =*/ OC,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
        /*.nb2 =*/ nb2,
        /*.nb3 =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_2d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    // Metal requires buffer size to be multiple of 16 bytes
    const size_t smem = GGML_PAD(KW * KH * sizeof(float), 16);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, OW, OH, OC * N, KW, KH, 1);

    return 1;
}

int ggml_metal_op_upscale(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float sf0 = (float)ne0/op->src[0]->ne[0];
    float sf1 = (float)ne1/op->src[0]->ne[1];
    float sf2 = (float)ne2/op->src[0]->ne[2];
    float sf3 = (float)ne3/op->src[0]->ne[3];

    const int32_t mode_flags = ggml_get_op_params_i32(op, 0);

    float poffs = 0.5f;

    if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        poffs = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (float)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (float)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    ggml_metal_kargs_upscale args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.sf0   =*/ sf0,
        /*.sf1   =*/ sf1,
        /*.sf2   =*/ sf2,
        /*.sf3   =*/ sf3,
        /*.poffs =*/ poffs,
    };

    auto pipeline = ggml_metal_library_get_pipeline_upscale(lib, op);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_roll(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ggml_get_op_params_i32(op, 0);
    const int32_t s1 = ggml_get_op_params_i32(op, 1);
    const int32_t s2 = ggml_get_op_params_i32(op, 2);
    const int32_t s3 = ggml_get_op_params_i32(op, 3);

    ggml_metal_kargs_roll args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.s2   =*/ s2,
        /*.s3   =*/ s3
    };

    auto pipeline = ggml_metal_library_get_pipeline_roll(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.lp0  =*/ ggml_get_op_params_i32(op, 0),
        /*.lp1  =*/ ggml_get_op_params_i32(op, 2),
        /*.lp2  =*/ ggml_get_op_params_i32(op, 4),
        /*.lp3  =*/ ggml_get_op_params_i32(op, 6),
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad(lib, op);

    const int nth_max = MIN(64, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int nth = MIN(args.ne0, nth_max);
    const int nk0 = (args.ne0 + 1024 - 1)/1024; // note: 1024 is hardcoded in the kernel!

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad_reflect_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad_reflect_1d args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.p0 =*/ ((const int32_t *)(op->op_params))[0],
        /*.p1 =*/ ((const int32_t *)(op->op_params))[1]
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad_reflect_1d(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_arange(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float start;
    float step;

    memcpy(&start, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&step,  ((const int32_t *) op->op_params) + 2, sizeof(float));

    ggml_metal_kargs_arange args = {
        /*.ne0   =*/ ne0,
        /*.start =*/ start,
        /*.step  =*/ step
    };

    const int nth = std::min(1024, ne0);

    auto pipeline = ggml_metal_library_get_pipeline_arange(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 1);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_timestep_embedding(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int dim        = op->op_params[0];
    const int max_period = op->op_params[1];

    ggml_metal_kargs_timestep_embedding args = {
        /*.nb1 =*/ nb1,
        /*.dim =*/ dim,
        /*.max_period =*/ max_period,
    };

    auto pipeline = ggml_metal_library_get_pipeline_timestep_embedding(lib, op);

    const int nth = std::max(1, std::min(1024, dim/2));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne00, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_argmax(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_argmax args = {
        /*.ne00 = */ ne00,
        /*.nb01 = */ nb01,
    };

    auto pipeline = ggml_metal_library_get_pipeline_argmax(lib, op);

    const int64_t nrows = ggml_nrows(op->src[0]);

    int nth = 32; // SIMD width
    while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
        nth *= 2;
    }

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_argsort(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_argsort(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    const int npr = (ne00 + nth - 1)/nth;

    // Metal kernels require the buffer size to be multiple of 16 bytes
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
    const size_t smem = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    if ((int) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    ggml_metal_kargs_argsort args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ nth,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_argsort_merge(lib, op);

    int len = nth;

    while (len < ne00) {
        ggml_metal_op_concurrency_reset(ctx);

        ggml_metal_kargs_argsort_merge args_merge = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ ne00,
            /*.len   =*/ len,
        };

        // merges per row
        const int nm = (ne00 + 2*len - 1) / (2*len);

        const int nth = std::min(512, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge));

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, sizeof(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }

    return 1;
}

// bitonic-sort + merge fallback: efficient when k is small and there are few rows,
// where the single-workgroup-per-row radix-select cannot reach enough parallelism
static void ggml_metal_op_top_k_bitonic(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_top_k(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    // blocks per row
    const int npr = (ne00 + nth - 1)/nth;

    const size_t smem = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += sizeof(int32_t)*ggml_nelements(op->src[0]);

    if ((int) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    const int top_k = ne0;

    ggml_metal_kargs_argsort args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ std::min(nth, top_k), // for each block, keep just the top_k indices
    };

    if (npr > 1) {
        args.ne0 = (npr - 1)*args.top_k + std::min(ne00 - (npr - 1)*nth, args.top_k);
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_top_k_merge(lib, op);

    int len = args.top_k;

    while (len < args.ne0) {
        ggml_metal_op_concurrency_reset(ctx);

        // merges per row
        const int nm = (args.ne0 + 2*len - 1) / (2*len);

        const int nth = std::min(512, std::min(len, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge)));

        ggml_metal_kargs_argsort_merge args_merge = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ args.ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ nm == 1 ? top_k : args.ne0, // the final merge outputs top_k elements
            /*.len   =*/ len,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, sizeof(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }
}

// radix-select: one workgroup per row. Maps each float to an order-preserving unsigned
// key, finds the k-th largest via 4 radix-8 histogram passes, then compacts the top-k
// indices. Fast for large k and/or many rows.
static void ggml_metal_op_top_k_radix(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);

    auto pipeline = ggml_metal_library_get_pipeline_top_k_radix(lib, op);

    // one workgroup per row; radix-select the k-th largest value
    const int nth = std::min(1024, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ggml_metal_kargs_top_k args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.top_k =*/ (int32_t) op->ne[0],
    };

    // shared memory: 256-entry histogram + bucket/above scalars + output counter
    const size_t smem_histo  = GGML_PAD(256*sizeof(uint32_t), 16);
    const size_t smem_bucket = GGML_PAD(    sizeof(uint32_t), 16);
    const size_t smem_above  = GGML_PAD(    sizeof(uint32_t), 16);
    const size_t smem_out    = GGML_PAD(    sizeof(uint32_t), 16);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_histo,  0);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_bucket, 1);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_above,  2);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_out,    3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
}

int ggml_metal_op_topk_moe(ggml_metal_op_t ctx, int idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    int n_fuse = 1;
    const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
    if (!fusion || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_TOPK_MOE) {
        return 1;
    }

    ggml_tensor * softmax  = ctx->node(idx);
    ggml_tensor * logits   = softmax->src[0];
    ggml_tensor * get_rows = ctx->node(idx + 2);
    ggml_tensor * ids      = get_rows->src[1];
    ggml_tensor * weights  = ctx->node(idx + n_fuse - 1);

    const int64_t n_expert      = logits->ne[0];
    const int64_t n_tokens      = logits->ne[1];
    const int64_t n_expert_used = ids->ne[0];

    const bool with_norm  = n_fuse >= 6;
    const bool with_scale = n_fuse == 4 || n_fuse == 7;

    float clamp = -INFINITY;
    if (with_norm) {
        ggml_tensor * clamp_node = ctx->node(idx + 4);
        clamp = ggml_get_op_params_f32(clamp_node, 0);
    }

    float scale = 1.0f;
    if (with_scale) {
        ggml_tensor * scale_node = ctx->node(idx + n_fuse - 1);
        scale = ggml_get_op_params_f32(scale_node, 0);
    }

    ggml_metal_kargs_topk_moe args = {
        /*.ne01      =*/ (int32_t) n_tokens,
        /*.nb01      =*/ logits->nb[1],
        /*.nb1_ids   =*/ ids->nb[1],
        /*.clamp =*/ clamp,
        /*.scale =*/ scale,
    };

    auto pipeline = ggml_metal_library_get_pipeline_topk_moe(lib, (int32_t) n_expert, (int32_t) n_expert_used, with_norm);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(logits),  1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ids),     3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (uint32_t) n_tokens, 1, 1, 32, 1, 1);

    ctx->count_fusions(fusion);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: SOFT_MAX + ARGSORT + GET_ROWS\n", __func__);
    }

    return n_fuse;
}

int ggml_metal_op_moe_reduce(ggml_metal_op_t ctx, int idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    int n_fuse = 1;
    const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
    if (!fusion || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_MOE_REDUCE) {
        return 1;
    }

    ggml_tensor * mul     = ctx->node(idx);
    ggml_tensor * experts = mul->src[0];
    ggml_tensor * weights = mul->src[1];
    ggml_tensor * dst     = ctx->node(idx + n_fuse - 1);

    ggml_metal_kargs_moe_reduce args = {
        /*.ne00 =*/ (int32_t) experts->ne[0],
        /*.ne02 =*/ (int32_t) experts->ne[2],
    };

    auto pipeline = ggml_metal_library_get_pipeline_moe_reduce(lib, (int32_t) experts->ne[1]);

    const int nth = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int n_col_tiles = (args.ne00 + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(experts), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst),     3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (uint32_t) args.ne02, (uint32_t) n_col_tiles, 1, nth, 1, 1);

    ctx->count_fusions(fusion);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: MOE_REDUCE\n", __func__);
    }

    return n_fuse;
}

int ggml_metal_op_top_k(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    // radix-select has a fixed single-workgroup-per-row cost (~50-60us) that is only
    // amortized for long rows, many rows, or a large k; otherwise the bitonic path wins
    const int ncols = op->src[0]->ne[0];
    const int k     = op->ne[0];
    const int nrows = ggml_nrows(op->src[0]);

    const bool use_radix =
        ncols > 2048 && (k > 64 || (nrows > 4 && ncols >= 8192));

    if (use_radix) {
        ggml_metal_op_top_k_radix(ctx, idx);
    } else {
        ggml_metal_op_top_k_bitonic(ctx, idx);
    }

    return 1;
}

int ggml_metal_op_tri(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_tri args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_tri(lib, op);

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_out_prod(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_out_prod args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_out_prod(lib, op);

    const int threads = ne0 < 1 ? 1 : (int) ne0;
    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), threads);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_soft_max_back(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) op->op_params) + 1, sizeof(max_bias));

    GGML_ASSERT(max_bias == 0.0f);

    const bool use_vec4 = (ne00 % 4) == 0;

    ggml_metal_kargs_soft_max_back args = {
        /*.ne00   =*/ ne00,
        /*.ne00_4 =*/ ne00/4,
        /*.nb01   =*/ nb01,
        /*.nb02   =*/ nb02,
        /*.nb03   =*/ nb03,
        /*.nb11   =*/ nb11,
        /*.nb12   =*/ nb12,
        /*.nb13   =*/ nb13,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.scale  =*/ scale,
    };

    auto pipeline = ggml_metal_library_get_pipeline_soft_max_back(lib, op);

    int nth = 32; // SIMD width

    if (use_vec4) {
        const int ne00_4 = ne00/4;
        while (nth < ne00_4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nth *= 2;
        }
    } else {
        while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nth *= 2;
        }
    }

    nth = std::max(1, nth);
    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_rms_norm_back(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ne00 % 4 == 0);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_kargs_rms_norm_back args = {
        /*.ne00   =*/ ne00,
        /*.ne00_4 =*/ ne00/4,
        /*.nb01   =*/ nb01,
        /*.nb02   =*/ nb02,
        /*.nb03   =*/ nb03,
        /*.nb11   =*/ nb11,
        /*.nb12   =*/ nb12,
        /*.nb13   =*/ nb13,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.eps    =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_rms_norm_back(lib, op);

    int nth = 32; // SIMD width

    while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_gelu_back(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    auto pipeline = ggml_metal_library_get_pipeline_gelu_back(lib, op);

    int64_t n = ggml_nelements(op);

    if (n % 4 == 0) {
        n /= 4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_sigmoid_back(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    auto pipeline = ggml_metal_library_get_pipeline_sigmoid_back(lib, op);

    int64_t n = ggml_nelements(op);

    if (n % 4 == 0) {
        n /= 4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_l2_norm_back(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ne00 % 4 == 0);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_kargs_l2_norm_back args = {
        /*.ne00   =*/ ne00,
        /*.ne00_4 =*/ ne00/4,
        /*.nb01   =*/ nb01,
        /*.nb02   =*/ nb02,
        /*.nb03   =*/ nb03,
        /*.nb11   =*/ nb11,
        /*.nb12   =*/ nb12,
        /*.nb13   =*/ nb13,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.eps    =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_l2_norm_back(lib, op);

    int nth = 32; // SIMD width

    while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_mul_mat_id_back_a(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * grad_out = op->src[0];
    const ggml_tensor * b        = op->src[1];
    const ggml_tensor * ids      = op->src[2];

    const int64_t fsz = sizeof(float);
    const int64_t isz = sizeof(int32_t);

    ggml_metal_kargs_mul_mat_id_back_a args = {
        /*.K        =*/ (int32_t) op->ne[0],
        /*.N        =*/ (int32_t) op->ne[1],
        /*.n_expert =*/ (int32_t) op->ne[2],
        /*.n_used   =*/ (int32_t) ids->ne[0],
        /*.n_tok    =*/ (int32_t) ids->ne[1],
        /*.b_ne1    =*/ (int32_t) b->ne[1],
        /*.g_nb1    =*/ grad_out->nb[1]/fsz,
        /*.g_nb2    =*/ grad_out->nb[2]/fsz,
        /*.b_nb1    =*/ b->nb[1]/fsz,
        /*.b_nb2    =*/ b->nb[2]/fsz,
        /*.ids_nb1  =*/ ids->nb[1]/isz,
        /*.d_nb1    =*/ op->nb[1]/fsz,
        /*.d_nb2    =*/ op->nb[2]/fsz,
    };

    auto pipeline = ggml_metal_library_get_pipeline_mul_mat_id_back_a(lib, op);

    int nth = std::min<int>(op->ne[0], ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(grad_out), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(b),        2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ids),      3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),       4);

    ggml_metal_encoder_dispatch_threadgroups(enc, op->ne[1], op->ne[2], 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_mul_mat_id_back_b(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * as       = op->src[0];
    const ggml_tensor * grad_out = op->src[1];
    const ggml_tensor * ids      = op->src[2];

    const bool as_is_f32 = (as->type == GGML_TYPE_F32);

    const int64_t fsz = sizeof(float);
    const int64_t isz = sizeof(int32_t);

    ggml_metal_kargs_mul_mat_id_back_b args = {
        /*.K        =*/ (int32_t) op->ne[0],
        /*.N        =*/ (int32_t) grad_out->ne[0],
        /*.n_used   =*/ (int32_t) ids->ne[0],
        /*.n_tok    =*/ (int32_t) ids->ne[1],
        /*.n_expert =*/ (int32_t) as->ne[2],
        /*.dst_ne1  =*/ (int32_t) op->ne[1],
        /*.as_nb1   =*/ as_is_f32 ? as->nb[1]/fsz : 0,
        /*.as_nb2   =*/ as_is_f32 ? as->nb[2]/fsz : 0,
        /*.g_nb1    =*/ grad_out->nb[1]/fsz,
        /*.g_nb2    =*/ grad_out->nb[2]/fsz,
        /*.ids_nb1  =*/ ids->nb[1]/isz,
        /*.d_nb1    =*/ op->nb[1]/fsz,
        /*.d_nb2    =*/ op->nb[2]/fsz,
    };

    auto pipeline = ggml_metal_library_get_pipeline_mul_mat_id_back_b(lib, op);

    int nth = std::min<int>(op->ne[0], ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(as),       1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(grad_out), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ids),      3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),       4);

    ggml_metal_encoder_dispatch_threadgroups(enc, op->ne[1], op->ne[2], 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_ssm_conv_back_sx(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * grad_out = op->src[0];
    const ggml_tensor * c        = op->src[1];

    const int64_t fsz = sizeof(float);

    const int32_t nc  = (int32_t) c->ne[0];
    const int32_t ncs = (int32_t) op->ne[0];
    const int32_t nr  = (int32_t) op->ne[1];
    const int32_t n_t = (int32_t) grad_out->ne[1];
    const int32_t n_s = (int32_t) op->ne[2];

    ggml_metal_kargs_ssm_conv_back_sx args = {
        /*.nc       =*/ nc,
        /*.ncs      =*/ ncs,
        /*.nr       =*/ nr,
        /*.n_t      =*/ n_t,
        /*.n_s      =*/ n_s,
        /*.grad_nb0 =*/ grad_out->nb[0]/fsz,
        /*.grad_nb1 =*/ grad_out->nb[1]/fsz,
        /*.grad_nb2 =*/ grad_out->nb[2]/fsz,
        /*.c_nb1    =*/ c->nb[1]/fsz,
        /*.dst_nb0  =*/ op->nb[0]/fsz,
        /*.dst_nb1  =*/ op->nb[1]/fsz,
        /*.dst_nb2  =*/ op->nb[2]/fsz,
    };

    auto pipeline = ggml_metal_library_get_pipeline_ssm_conv_back_sx(lib, op);

    const int nth = std::min(32, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(grad_out), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(c),        2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),       3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ncs + nth - 1)/nth, nr, n_s, nth, 1, 1);

    return 1;
}

int ggml_metal_op_ssm_conv_back_c(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * grad_out = op->src[0];
    const ggml_tensor * sx       = op->src[1];

    const int64_t fsz = sizeof(float);

    const int32_t nc  = (int32_t) op->ne[0];
    const int32_t nr  = (int32_t) op->ne[1];
    const int32_t n_t = (int32_t) grad_out->ne[1];
    const int32_t n_s = (int32_t) grad_out->ne[2];

    ggml_metal_kargs_ssm_conv_back_c args = {
        /*.nc       =*/ nc,
        /*.ncs      =*/ (int32_t) sx->ne[0],
        /*.nr       =*/ nr,
        /*.n_t      =*/ n_t,
        /*.n_s      =*/ n_s,
        /*.grad_nb0 =*/ grad_out->nb[0]/fsz,
        /*.grad_nb1 =*/ grad_out->nb[1]/fsz,
        /*.grad_nb2 =*/ grad_out->nb[2]/fsz,
        /*.sx_nb0   =*/ sx->nb[0]/fsz,
        /*.sx_nb1   =*/ sx->nb[1]/fsz,
        /*.sx_nb2   =*/ sx->nb[2]/fsz,
        /*.dst_nb1  =*/ op->nb[1]/fsz,
    };

    auto pipeline = ggml_metal_library_get_pipeline_ssm_conv_back_c(lib, op);

    const int nth = std::min(32, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(grad_out), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(sx),       2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),       3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (nc + nth - 1)/nth, nr, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_gated_delta_net_back(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * q     = op->src[0];
    const ggml_tensor * k     = op->src[1];
    const ggml_tensor * v     = op->src[2];
    const ggml_tensor * g     = op->src[3];
    const ggml_tensor * beta  = op->src[4];
    const ggml_tensor * state = op->src[5];
    const ggml_tensor * d     = op->src[6];

    const int64_t fsz = sizeof(float);

    const int64_t S_v      = v->ne[0];
    const int64_t H        = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs   = v->ne[3];
    const int64_t neq1     = q->ne[1];
    const int64_t neq3     = q->ne[3];
    // K (snapshot slot count) is set as an op param by ggml_gated_delta_net_back().
    const int64_t K        = ggml_get_op_params_i32(op, 0);

    const int64_t pad_q = GGML_PAD(ggml_nelements(q)     * fsz, GGML_MEM_ALIGN) / fsz;
    const int64_t pad_k = GGML_PAD(ggml_nelements(k)     * fsz, GGML_MEM_ALIGN) / fsz;
    const int64_t pad_v = GGML_PAD(ggml_nelements(v)     * fsz, GGML_MEM_ALIGN) / fsz;
    const int64_t pad_g = GGML_PAD(ggml_nelements(g)     * fsz, GGML_MEM_ALIGN) / fsz;
    const int64_t pad_b = GGML_PAD(ggml_nelements(beta)  * fsz, GGML_MEM_ALIGN) / fsz;
    const int64_t pad_s = GGML_PAD(ggml_nelements(state) * fsz, GGML_MEM_ALIGN) / fsz;

    const int64_t off_dk      = pad_q;
    const int64_t off_dv      = off_dk + pad_k;
    const int64_t off_dg      = off_dv + pad_v;
    const int64_t off_db      = off_dg + pad_g;
    const int64_t off_ds      = off_db + pad_b;
    const int64_t off_scratch = off_ds + pad_s;
    const int64_t wg_stride   = n_tokens * (2*S_v*S_v + 2*S_v);

    GGML_UNUSED(d);

    ggml_metal_kargs_gated_delta_net_back args = {
        /*.H           =*/ (int32_t) H,
        /*.n_tokens    =*/ (int32_t) n_tokens,
        /*.n_seqs      =*/ (int32_t) n_seqs,
        /*.K           =*/ (int32_t) K,
        /*.neq1        =*/ (int32_t) neq1,
        /*.rq3         =*/ (int32_t) (n_seqs / neq3),
        /*.s_off       =*/ (uint64_t) (S_v * H * n_tokens * n_seqs),
        /*.sq1         =*/ q->nb[1]/fsz,
        /*.sq2         =*/ q->nb[2]/fsz,
        /*.sq3         =*/ q->nb[3]/fsz,
        /*.sv1         =*/ v->nb[1]/fsz,
        /*.sv2         =*/ v->nb[2]/fsz,
        /*.sv3         =*/ v->nb[3]/fsz,
        /*.sb1         =*/ beta->nb[1]/fsz,
        /*.sb2         =*/ beta->nb[2]/fsz,
        /*.sb3         =*/ beta->nb[3]/fsz,
        /*.off_dk      =*/ (uint64_t) off_dk,
        /*.off_dv      =*/ (uint64_t) off_dv,
        /*.off_dg      =*/ (uint64_t) off_dg,
        /*.off_db      =*/ (uint64_t) off_db,
        /*.off_ds      =*/ (uint64_t) off_ds,
        /*.off_scratch =*/ (uint64_t) off_scratch,
        /*.wg_stride   =*/ (uint64_t) wg_stride,
        /*.scale       =*/ 1.0f / sqrtf((float) S_v),
    };

    auto pipeline = ggml_metal_library_get_pipeline_gated_delta_net_back(lib, op);

    const int nth = pipeline.nth;

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(q),     1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(k),     2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(v),     3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(g),     4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(beta),  5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(state), 6);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(d),     7);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),    8);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, neq1, neq3, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_adamw(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_adamw(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_adamw args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_sgd(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_sgd(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_sgd args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_count_equal(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);

    {
        ggml_metal_kargs_memset args = { /*.val =*/ 0 };

        auto pipeline = ggml_metal_library_get_pipeline_memset(lib, op);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 1);

        ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1, 1, 1);
    }

    ggml_metal_op_concurrency_reset(ctx);

    {
        ggml_metal_kargs_count_equal args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
        };

        auto pipeline = ggml_metal_library_get_pipeline_count_equal(lib, op);

        const size_t smem = pipeline.smem;

        const int nth = 32*pipeline.nsg;

        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    return 1;
}

static bool ggml_metal_dw_ln_fusion_enabled() {
    static const bool enabled = getenv("GGML_METAL_FUSION_DW_LN_DISABLE") == nullptr;
    return enabled;
}

#define GGML_METAL_DW_LN_MAX_PER_THREAD 8
#define GGML_METAL_NORM_CHANNEL_MAX_SIMDGROUPS 8

static bool ggml_metal_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const ggml_metal_buffer_id ba = ggml_metal_get_buffer_id(a);
    const ggml_metal_buffer_id bb = ggml_metal_get_buffer_id(b);
    if (ba.metal != bb.metal) {
        return false;
    }
    return ba.offs < bb.offs + ggml_nbytes(b) && bb.offs < ba.offs + ggml_nbytes(a);
}

// Scratch for the channel layer norm reduction: one float per simdgroup.
static const size_t ggml_metal_norm_channel_shared_bytes = GGML_METAL_NORM_CHANNEL_MAX_SIMDGROUPS * sizeof(float);
static const int    ggml_metal_norm_channel_max_threads  = GGML_METAL_NORM_CHANNEL_MAX_SIMDGROUPS * 32;

// Threads per timestep for the channel layer norm: a multiple of 32 covering C, capped at 256.
static int ggml_metal_norm_channel_nth(int C) {
    const int max_threads = ggml_metal_norm_channel_max_threads;
    int nth = 32;
    while (nth < C && nth < max_threads) nth *= 2;
    if (nth > C) nth = ((C + 31) / 32) * 32;
    if (nth > max_threads) nth = max_threads;
    if (nth < 32) nth = 32;
    return nth;
}

// The layer norm consuming this depthwise output when both fit one dispatch: same layout, channels
// within the per-thread registers, and a result that does not alias the input the kernel keeps reading.
static const ggml_tensor * ggml_metal_dw_ln_match(ggml_metal_op_t ctx, int idx, int32_t layout, int C, int nth) {
    if (!ctx->use_fusion() || !ggml_metal_dw_ln_fusion_enabled() || idx + 1 >= ctx->n_nodes()) {
        return nullptr;
    }
    const ggml_op ops[2] = { GGML_OP_CONV_1D_DW_DIRECT, GGML_OP_NORM_CHANNEL };
    const int gi = ctx->graph_idx(idx);
    if (ctx->graph_idx(idx + 1) != gi + 1 || !ggml_can_fuse(ctx->graph(), gi, ops, 2)) {
        return nullptr;
    }
    const ggml_tensor * dw = ctx->node(idx);
    const ggml_tensor * ln = ctx->node(idx + 1);
    if (ln->src[0] != dw || ggml_get_op_params_i32(ln, 1) != layout) {
        return nullptr;
    }
    if (C > nth * GGML_METAL_DW_LN_MAX_PER_THREAD || ggml_metal_tensors_overlap(ln, dw->src[0])) {
        return nullptr;
    }
    return ln;
}

static int ggml_metal_op_conv_1d_dw_direct_norm_channel(ggml_metal_op_t ctx, int idx, const ggml_tensor * ln,
                                                            const ggml_metal_kargs_conv_1d_dw_direct & dw, int nth) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    float eps;
    memcpy(&eps, ln->op_params, sizeof(eps));

    ggml_metal_kargs_conv_1d_dw_direct_norm_channel args = {
        /*.L        =*/ dw.L,
        /*.C        =*/ dw.C,
        /*.B        =*/ dw.B,
        /*.K        =*/ dw.K,
        /*.dilation =*/ dw.dilation,
        /*.has_bias =*/ dw.has_bias,
        /*.causal   =*/ dw.causal,
        /*.seg_len  =*/ dw.seg_len,
        /*.sxt      =*/ dw.sxt,
        /*.sxc      =*/ dw.sxc,
        /*.syt      =*/ dw.syt,
        /*.syc      =*/ dw.syc,
        /*.eps      =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_1d_dw_direct_norm_channel(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1); // x
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2); // w
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2] ? op->src[2] : op->src[0]), 3); // bias
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ln->src[1]), 4); // g
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ln->src[2]), 5); // b
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ln),         6); // y
    ggml_metal_encoder_set_threadgroup_memory_size(enc, ggml_metal_norm_channel_shared_bytes, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, dw.L, dw.B, 1, nth, 1, 1);

    return 2;
}

int ggml_metal_op_snake(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const int     T     = (int) op->ne[0];
    const int     C     = (int) op->ne[1];
    const int64_t total = (int64_t) T * C;

    auto pipeline = ggml_metal_library_get_pipeline_snake(lib, op->src[0]->type);

    ggml_metal_kargs_snake args = {
        /*.T  =*/ T,
        /*.C  =*/ C,
        /*.ne =*/ total,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    const int     nth = 256;
    const int64_t ntg = (total + nth - 1) / nth;
    GGML_ASSERT(ntg <= INT32_MAX);
    ggml_metal_encoder_dispatch_threadgroups(enc, (int) ntg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_lstm_cell(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * mask = op->src[2];

    // Unmasked: src1 is c_prev [H, N]. Masked: src1 is hc_prev [2H, N] and c
    // starts half a row in.
    const int32_t H = (int32_t) (op->ne[0]/GGML_LSTM_N_OUTS);
    const int32_t N = (int32_t) op->src[1]->ne[1];

    ggml_metal_kargs_lstm_cell args = {
        /*.H           =*/ H,
        /*.N           =*/ N,
        /*.prev_row    =*/ (int32_t) op->src[1]->ne[0],
        /*.c_base      =*/ mask ? H : 0,
        /*.mask_stride =*/ (mask && ggml_nelements(mask) > 1) ? 1 : 0,
    };

    auto pipeline = ggml_metal_library_get_pipeline_lstm_cell(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1); // gates
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2); // c_prev / hc_prev
    if (mask) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(mask), 3);     // select mask
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),   4);     // h_new | c_new
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),   3);     // h_new | c_new
    }

    // Flat 1D grid over H*N elements with a grid-stride loop for high occupancy.
    const int64_t total = (int64_t) H * (int64_t) N;
    const int     nth   = 256;
    int64_t ntg = (total + nth - 1) / nth;
    if (ntg < 1)      ntg = 1;
    if (ntg > 65535)  ntg = 65535;  // grid-stride covers the remainder
    ggml_metal_encoder_dispatch_threadgroups(enc, (int) ntg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_tdt_step(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ggml_metal_kargs_tdt_step args = {
        /*.n_dur       =*/ (int32_t) ggml_nelements(op->src[3]),
        /*.blank_id    =*/ ggml_get_op_params_i32(op, 0),
        /*.max_symbols =*/ ggml_get_op_params_i32(op, 1),
        /*.rnnt        =*/ ggml_get_op_params_i32(op, 2),
    };

    auto pipeline = ggml_metal_library_get_pipeline_tdt_step(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1); // token
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2); // duration index
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3); // loop counters
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4); // duration table
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         5);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_conv_1d_dw_direct(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne, op, ne);

    const int32_t * opts = (const int32_t *) op->op_params;
    const int K        = opts[0];
    const int dilation = opts[1];
    // opts[2] layout (0 = [T, C], 1 = [C, T]), opts[3] causal (0 = edge clamp, 1 = left pad),
    // opts[4] segment length along T (0 = the whole row).
    const int32_t layout = opts[2];
    const int32_t causal = opts[3];
    const int32_t seg_len = opts[4];

    int L, C, sxt, sxc, syt, syc;
    if (layout == 0) {
        L = ne0; C = ne1;
        sxt = 1; sxc = L; syt = 1; syc = L;
    } else {
        C = ne0; L = ne1;
        sxt = C; sxc = 1; syt = C; syc = 1;
    }

    ggml_metal_kargs_conv_1d_dw_direct args = {
        /*.L        =*/ L,
        /*.C        =*/ C,
        /*.B        =*/ ne2,
        /*.K        =*/ K,
        /*.dilation =*/ dilation,
        /*.has_bias =*/ (op->src[2] != nullptr) ? 1 : 0,
        /*.causal   =*/ causal,
        /*.seg_len  =*/ seg_len,
        /*.sxt      =*/ sxt,
        /*.sxc      =*/ sxc,
        /*.syt      =*/ syt,
        /*.syc      =*/ syc,
    };

    const int ln_nth = ggml_metal_norm_channel_nth(C);
    if (const ggml_tensor * ln = ggml_metal_dw_ln_match(ctx, idx, layout, C, ln_nth)) {
        if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + 1))) {
            ggml_metal_op_concurrency_reset(ctx);
        }
        return ggml_metal_op_conv_1d_dw_direct_norm_channel(ctx, idx, ln, args, ln_nth);
    }

    auto pipeline = ggml_metal_library_get_pipeline_conv_1d_dw_direct(lib, op);

    // Cap threads-per-threadgroup at min(L, 1024).  One threadgroup per
    // channel; threads stride over time.
    int nth = L;
    if (nth > 1024) nth = 1024;
    if (nth < 1)    nth = 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1); // x
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2); // w
    if (op->src[2] != nullptr) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 3); // bias
    } else {
        // Bind src[0] as a harmless placeholder; the kernel won't read it.
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4); // y

    ggml_metal_encoder_dispatch_threadgroups(enc, C, ne2, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_norm_channel(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t, ne, op, ne);

    // op_params layout: [eps (f32), layout_flag (i32)].  layout_flag is
    // 0 (default) for [T, C] activations (T inner-most, the historical
    // contract), 1 for [C, T] (C inner-most — full B2 path).
    float eps;
    memcpy(&eps, op->op_params, sizeof(eps));
    const int32_t layout = ((const int32_t *) op->op_params)[1];

    int L, C;
    int sxt, sxc, syt, syc;
    if (layout == 0) {
        // [T, C]: ne0 = T, ne1 = C.  T inner-most (stride 1), C outer (stride L).
        L = ne0;
        C = ne1;
        sxt = 1;       sxc = L;
        syt = 1;       syc = L;
    } else {
        // [C, T]: ne0 = C, ne1 = T.  C inner-most (stride 1), T outer (stride C).
        C = ne0;
        L = ne1;
        sxt = C;       sxc = 1;
        syt = C;       syc = 1;
    }

    ggml_metal_kargs_norm_channel args = {
        /*.L   =*/ L,
        /*.C   =*/ C,
        /*.B   =*/ ne2,
        /*.eps =*/ eps,
        /*.sxt =*/ sxt,
        /*.sxc =*/ sxc,
        /*.syt =*/ syt,
        /*.syc =*/ syc,
    };

    auto pipeline = ggml_metal_library_get_pipeline_norm_channel(lib, op);

    const int nth = ggml_metal_norm_channel_nth(C);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1); // x
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2); // g
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3); // b
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4); // y
    ggml_metal_encoder_set_threadgroup_memory_size(enc, ggml_metal_norm_channel_shared_bytes, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, L, ne2, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_scale_residual(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t, ne, op, ne);

    // op_params[0]: 0 = [T, C], 1 = [C, T], 2 = [T, C] input to [C, T] output.
    const int32_t layout = ((const int32_t *) op->op_params)[0];

    int L, C, sxt, sxc, syt, syc, srt, src;
    if (layout == 0) {
        L = ne0; C = ne1;
        sxt = 1; sxc = L; syt = 1; syc = L; srt = 1; src = L;
    } else {
        C = ne0; L = ne1;
        sxt = layout == 1 ? C : 1;
        sxc = layout == 1 ? 1 : L;
        syt = C; syc = 1; srt = C; src = 1;
    }

    ggml_metal_kargs_scale_residual args = {
        /*.L   =*/ L,
        /*.C   =*/ C,
        /*.B   =*/ ne2,
        /*.sxt =*/ sxt,
        /*.sxc =*/ sxc,
        /*.syt =*/ syt,
        /*.syc =*/ syc,
        /*.srt =*/ srt,
        /*.src =*/ src,
    };

    auto pipeline = ggml_metal_library_get_pipeline_scale_residual(lib, op);

    int nth = L;
    if (nth > 256) nth = 256;
    if (nth < 1)   nth = 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1); // x
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2); // bias
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3); // gamma
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4); // residual
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         5); // y

    ggml_metal_encoder_dispatch_threadgroups(enc, C, ne2, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_bias_gelu(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t, ne, op, ne);

    // op_params[0]: 0 = [T, C], 1 = [C, T], 2 = [T, C] input to [C, T] output.
    const int32_t layout = ((const int32_t *) op->op_params)[0];

    int L, C, sxt, sxc, syt, syc;
    if (layout == 0) {
        L = ne0; C = ne1;
        sxt = 1; sxc = L; syt = 1; syc = L;
    } else {
        C = ne0; L = ne1;
        sxt = layout == 1 ? C : 1;
        sxc = layout == 1 ? 1 : L;
        syt = C; syc = 1;
    }

    ggml_metal_kargs_bias_gelu args = {
        /*.L   =*/ L,
        /*.C   =*/ C,
        /*.B   =*/ ne2,
        /*.sxt =*/ sxt,
        /*.sxc =*/ sxc,
        /*.syt =*/ syt,
        /*.syc =*/ syc,
    };

    auto pipeline = ggml_metal_library_get_pipeline_bias_gelu(lib, op);

    int nth = L;
    if (nth > 256) nth = 256;
    if (nth < 1)   nth = 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1); // x
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2); // bias
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3); // y

    ggml_metal_encoder_dispatch_threadgroups(enc, C, ne2, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad_replicate_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t, ne, op, ne);

    const int pad_left = ((const int32_t *) op->op_params)[0];
    // opts[2]: layout flag (opts[1] = pad_right).  0 = [T, C], 1 = [C, T].
    const int32_t layout = ((const int32_t *) op->op_params)[2];

    int L_in, L_out, C, sxt, sxc, syt, syc;
    if (layout == 0) {
        L_in  = (int) op->src[0]->ne[0];
        C     = (int) op->src[0]->ne[1];
        L_out = ne0;
        sxt = 1;     sxc = L_in;   syt = 1;     syc = L_out;
    } else {
        C     = (int) op->src[0]->ne[0];
        L_in  = (int) op->src[0]->ne[1];
        L_out = ne1;
        sxt = C;     sxc = 1;      syt = C;     syc = 1;
    }

    ggml_metal_kargs_pad_replicate_1d args = {
        /*.L_in     =*/ L_in,
        /*.L_out    =*/ L_out,
        /*.C        =*/ C,
        /*.pad_left =*/ pad_left,
        /*.sxt      =*/ sxt,
        /*.sxc      =*/ sxc,
        /*.syt      =*/ syt,
        /*.syc      =*/ syc,
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad_replicate_1d(lib, op);

    int nth = L_out;
    if (nth > 256) nth = 256;
    if (nth < 1)   nth = 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1); // x
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2); // y

    ggml_metal_encoder_dispatch_threadgroups(enc, C, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_diag_mask_inf(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const int32_t ncols = (int32_t) op->src[0]->ne[0];
    const int32_t nrows = (int32_t) ggml_nrows(op->src[0]);
    const int32_t rows_per_channel = (int32_t) op->src[0]->ne[1];
    const int32_t n_past = ((const int32_t *) op->op_params)[0];

    ggml_metal_kargs_diag_mask_inf args = {
        /*.ncols            =*/ ncols,
        /*.rows_per_channel =*/ rows_per_channel,
        /*.n_past           =*/ n_past,
    };

    auto pipeline = ggml_metal_library_get_pipeline_diag_mask_inf(lib, op);

    const int nth = std::min<int>(ncols, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int n_tiles = 1 + (ncols - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, n_tiles, 1, nth, 1, 1);

    return 1;
}
