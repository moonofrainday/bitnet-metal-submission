#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ggml.h"
#include "ggml-metal-device.h"
#include "ggml-metal-impl.h"

#include <cstddef>
#include <stdexcept>

// Uses the existing QVAC dylib's embedded library and real pipeline selector.
// No stock source copy, custom compile options, or opaque-object casts are used.
class StockRuntime {
    ggml_metal_device_t device_ = nullptr;
    ggml_context * context_ = nullptr;
    ggml_metal_pipeline_with_params pipeline_{};
    ggml_metal_kargs_mul_mv args_{};

    static void require(bool condition, const char * message) {
        if (!condition) throw std::runtime_error(message);
    }

    static ggml_metal_buffer_id binding(id<MTLBuffer> buffer, size_t offset, size_t bytes) {
        require(buffer != nil && offset <= buffer.length && bytes <= buffer.length-offset,
                "StockRuntime buffer range mismatch");
        return {(__bridge void *)buffer, offset};
    }

public:
    StockRuntime() {
        device_ = ggml_metal_device_init(0, 1);
        require(device_ != nullptr, "Actual QVAC Metal device unavailable");
        context_ = ggml_init({3*ggml_tensor_overhead(), nullptr, true});
        require(context_ != nullptr, "StockRuntime ggml_init failed");
        auto * w = ggml_new_tensor_2d(context_, GGML_TYPE_TQ2_0, 2560, 2560);
        auto * x = ggml_new_tensor_2d(context_, GGML_TYPE_F32, 2560, 1);
        auto * y = ggml_mul_mat(context_, w, x);
        ggml_set_name(w,"blk.0.attn_q.weight");
        ggml_set_name(x,"attn_norm-0");
        ggml_set_name(y,"Qcur-0");
        pipeline_ = ggml_metal_library_get_pipeline_mul_mv(ggml_metal_device_get_library(device_), y, false, false);
        require(pipeline_.pipeline != nullptr && pipeline_.nr0 == 4 && pipeline_.nr1 == 1 && pipeline_.nsg == 2,
                "Unexpected actual stock TQ2_0 pipeline specialization");
        require(ggml_metal_pipeline_thread_execution_width(pipeline_) == 32 &&
                ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_) >= 64,
                "Unsupported stock pipeline thread limits");
        args_ = {
            2560,2560,1, w->nb[0],w->nb[1],w->nb[2],w->nb[3],
            2560,1,1, x->nb[0],x->nb[1],x->nb[2],x->nb[3],
            2560,1,pipeline_.nr0,1,1,0.0f
        };
        static_assert(sizeof(ggml_metal_kargs_mul_mv) == 120, "Stock MUL_MAT ABI must be 120 bytes");
    }

    ~StockRuntime() {
        if (context_) ggml_free(context_);
        if (device_) ggml_metal_device_free(device_);
    }
    StockRuntime(const StockRuntime&) = delete;
    StockRuntime& operator=(const StockRuntime&) = delete;

    id<MTLDevice> device() const { return (__bridge id<MTLDevice>)ggml_metal_device_get_obj(device_); }
    id<MTLCommandQueue> queue() const { return (__bridge id<MTLCommandQueue>)ggml_metal_device_get_queue(device_); }
    ggml_metal_pipeline_with_params pipeline() const { return pipeline_; }
    const ggml_metal_kargs_mul_mv& args() const { return args_; }

    void encode(id<MTLCommandBuffer> command, id<MTLBuffer> weights, id<MTLBuffer> x, id<MTLBuffer> y,
                size_t w_offset=0, size_t x_offset=0, size_t y_offset=0) {
        require(command != nil, "StockRuntime command buffer missing");
        const auto wb=binding(weights,w_offset,1689600);
        const auto xb=binding(x,x_offset,10240);
        const auto yb=binding(y,y_offset,10240);
        auto encoder=ggml_metal_encoder_init((__bridge void*)command,false);
        require(encoder != nullptr, "StockRuntime encoder creation failed");
        ggml_metal_encoder_set_pipeline(encoder,pipeline_);
        ggml_metal_encoder_set_bytes(encoder,&args_,sizeof(args_),0);
        ggml_metal_encoder_set_buffer(encoder,wb,1);
        ggml_metal_encoder_set_buffer(encoder,xb,2);
        ggml_metal_encoder_set_buffer(encoder,yb,3);
        ggml_metal_encoder_set_buffer(encoder,yb,4);
        ggml_metal_encoder_set_buffer(encoder,xb,5);
        ggml_metal_encoder_set_threadgroup_memory_size(encoder,pipeline_.smem,0);
        ggml_metal_encoder_dispatch_threadgroups(encoder,320,1,1,32,pipeline_.nsg,1);
        ggml_metal_encoder_end_encoding(encoder);
        ggml_metal_encoder_free(encoder);
    }
};
