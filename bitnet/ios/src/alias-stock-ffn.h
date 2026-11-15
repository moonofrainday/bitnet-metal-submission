#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "ggml.h"
#include "ggml-metal-device.h"
#include "ggml-metal-impl.h"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace bitnet_iphone_operator {
// Selects the actual embedded QVAC pipelines. No stock shader copy is compiled.
// A single borrowed encoder is used for stock and custom work alike.
class StockFFN {
    ggml_metal_device_t dev_ = nullptr;
    ggml_context *ctx_ = nullptr;
    ggml_tensor *wg_, *wu_, *wd_, *x_, *gate_, *up_, *relu_, *sqr_, *act_, *rms_, *gamma_, *norm_, *down_;
    ggml_metal_pipeline_with_params gatep_{}, upp_{}, downp_{}, relup_{}, sqrp_{}, mulp_{}, normp_{};
    static void req(bool v,const char *m) {if(!v) throw std::runtime_error(m);}
    static ggml_metal_buffer_id bind(id<MTLBuffer> b,size_t n,size_t o=0) {
        req(b && o<=b.length && n<=b.length-o,"buffer bounds");return {(__bridge void*)b,o};
    }
    static ggml_metal_kargs_unary unaryArgs(const ggml_tensor *t) {
        const auto*s=t->src[0];ggml_metal_kargs_unary a{};
        a.ne00=s->ne[0];a.ne01=s->ne[1];a.ne02=s->ne[2];a.ne03=s->ne[3];
        a.nb00=s->nb[0];a.nb01=s->nb[1];a.nb02=s->nb[2];a.nb03=s->nb[3];
        a.ne0=t->ne[0];a.ne1=t->ne[1];a.ne2=t->ne[2];a.ne3=t->ne[3];
        a.nb0=t->nb[0];a.nb1=t->nb[1];a.nb2=t->nb[2];a.nb3=t->nb[3];return a;
    }
    void unary(ggml_metal_encoder_t e,ggml_tensor*t,ggml_metal_pipeline_with_params p,id<MTLBuffer>x,id<MTLBuffer>y) {
        auto a=unaryArgs(t);if(p.c4){a.ne00/=4;a.ne0/=4;}
        req(p.cnt,"unexpected noncontiguous unary pipeline");
        const int n=6912/(p.c4?4:1),nth=std::min(256,n);
        ggml_metal_encoder_set_pipeline(e,p);ggml_metal_encoder_set_bytes(e,&a,sizeof(a),0);
        ggml_metal_encoder_set_buffer(e,bind(x,27648),1);ggml_metal_encoder_set_buffer(e,bind(y,27648),2);
        ggml_metal_encoder_dispatch_threadgroups(e,(n+nth-1)/nth,1,1,nth,1,1);
    }
    void mv(ggml_metal_encoder_t e,ggml_tensor*t,ggml_metal_pipeline_with_params p,id<MTLBuffer>w,id<MTLBuffer>x,id<MTLBuffer>y) {
        const auto*a=t->src[0];const auto*b=t->src[1];
        ggml_metal_kargs_mul_mv args={
            (int32_t)a->ne[0],(int32_t)a->ne[1],1,a->nb[0],a->nb[1],a->nb[2],a->nb[3],
            (int32_t)b->ne[0],1,1,b->nb[0],b->nb[1],b->nb[2],b->nb[3],
            (int32_t)t->ne[0],1,p.nr0,1,1,0.0f};
        ggml_metal_encoder_set_pipeline(e,p);ggml_metal_encoder_set_bytes(e,&args,sizeof(args),0);
        ggml_metal_encoder_set_buffer(e,bind(w,ggml_nbytes(a)),1);
        ggml_metal_encoder_set_buffer(e,bind(x,ggml_nbytes(b)),2);
        ggml_metal_encoder_set_buffer(e,bind(y,ggml_nbytes(t)),3);
        ggml_metal_encoder_set_buffer(e,bind(y,ggml_nbytes(t)),4);
        ggml_metal_encoder_set_buffer(e,bind(x,ggml_nbytes(b)),5);
        ggml_metal_encoder_set_threadgroup_memory_size(e,p.smem,0);
        ggml_metal_encoder_dispatch_threadgroups(e,(t->ne[0]+p.nr0*p.nsg-1)/(p.nr0*p.nsg),1,1,32,p.nsg,1);
    }
public:
    StockFFN() {
        dev_=ggml_metal_device_init(0,1);req(dev_,"actual Metal device unavailable");
        ctx_=ggml_init({24*ggml_tensor_overhead(),nullptr,true});req(ctx_,"ggml init");
        wg_=ggml_new_tensor_2d(ctx_,GGML_TYPE_TQ2_0,2560,6912);
        wu_=ggml_new_tensor_2d(ctx_,GGML_TYPE_TQ2_0,2560,6912);
        wd_=ggml_new_tensor_2d(ctx_,GGML_TYPE_TQ2_0,6912,2560);
        x_=ggml_new_tensor_2d(ctx_,GGML_TYPE_F32,2560,1);
        gate_=ggml_mul_mat(ctx_,wg_,x_);up_=ggml_mul_mat(ctx_,wu_,x_);
        relu_=ggml_relu(ctx_,gate_);sqr_=ggml_sqr(ctx_,relu_);act_=ggml_mul(ctx_,sqr_,up_);
        rms_=ggml_rms_norm(ctx_,act_,1.e-5f);gamma_=ggml_new_tensor_1d(ctx_,GGML_TYPE_F32,6912);
        norm_=ggml_mul(ctx_,rms_,gamma_);down_=ggml_mul_mat(ctx_,wd_,norm_);
        auto lib=ggml_metal_device_get_library(dev_);
        gatep_=ggml_metal_library_get_pipeline_mul_mv(lib,gate_,false,false);
        upp_=ggml_metal_library_get_pipeline_mul_mv(lib,up_,false,false);
        downp_=ggml_metal_library_get_pipeline_mul_mv(lib,down_,false,false);
        relup_=ggml_metal_library_get_pipeline_unary(lib,relu_);
        sqrp_=ggml_metal_library_get_pipeline_unary(lib,sqr_);
        mulp_=ggml_metal_library_get_pipeline_bin(lib,act_,1);
        normp_=ggml_metal_library_get_pipeline_norm(lib,rms_,2,GGML_METAL_NORM_EPI_NONE);
        for(auto p:{gatep_,upp_,downp_})req(p.pipeline && p.nr0==4 && p.nsg==2,"actual stock matvec specialization changed");
        req(mulp_.c4 && mulp_.cnt,"actual MUL specialization changed");
        for(auto p:{gatep_,upp_,downp_,relup_,sqrp_,mulp_,normp_})req(p.pipeline,"missing actual stock pipeline");
    }
    ~StockFFN(){if(ctx_)ggml_free(ctx_);if(dev_)ggml_metal_device_free(dev_);}
    StockFFN(const StockFFN&)=delete;StockFFN&operator=(const StockFFN&)=delete;
    id<MTLDevice>device()const{return (__bridge id<MTLDevice>)ggml_metal_device_get_obj(dev_);}
    id<MTLCommandQueue>queue()const{return (__bridge id<MTLCommandQueue>)ggml_metal_device_get_queue(dev_);}
    void gate(ggml_metal_encoder_t e,id<MTLBuffer>w,id<MTLBuffer>x,id<MTLBuffer>y){mv(e,gate_,gatep_,w,x,y);}
    void up(ggml_metal_encoder_t e,id<MTLBuffer>w,id<MTLBuffer>x,id<MTLBuffer>y){mv(e,up_,upp_,w,x,y);}
    void down(ggml_metal_encoder_t e,id<MTLBuffer>w,id<MTLBuffer>x,id<MTLBuffer>y){mv(e,down_,downp_,w,x,y);}
    void relu(ggml_metal_encoder_t e,id<MTLBuffer>x,id<MTLBuffer>y){unary(e,relu_,relup_,x,y);}
    void square(ggml_metal_encoder_t e,id<MTLBuffer>x,id<MTLBuffer>y){unary(e,sqr_,sqrp_,x,y);}
    void multiply(ggml_metal_encoder_t e,id<MTLBuffer>a,id<MTLBuffer>b,id<MTLBuffer>y) {
        ggml_metal_kargs_bin k{};
        k.ne00=k.ne10=k.ne0=1728;k.ne01=k.ne02=k.ne03=k.ne11=k.ne12=k.ne13=k.ne1=k.ne2=k.ne3=1;
        k.nb00=k.nb10=k.nb0=4;k.nb01=k.nb02=k.nb03=k.nb11=k.nb12=k.nb13=k.nb1=k.nb2=k.nb3=27648;
        ggml_metal_encoder_set_pipeline(e,mulp_);ggml_metal_encoder_set_bytes(e,&k,sizeof(k),0);
        ggml_metal_encoder_set_buffer(e,bind(a,27648),1);ggml_metal_encoder_set_buffer(e,bind(b,27648),2);
        ggml_metal_encoder_set_buffer(e,bind(y,27648),3);
        ggml_metal_encoder_dispatch_threadgroups(e,7,1,1,256,1,1);
    }
    void norm(ggml_metal_encoder_t e,id<MTLBuffer>x,id<MTLBuffer>gamma,id<MTLBuffer>y) {
        ggml_metal_kargs_norm a{};a.ne00=6912;a.ne00_t=1728;a.nb1=a.nb2=a.nb3=27648;a.eps=1.e-5f;
        a.nef1[0]=a.nef2[0]=a.nef3[0]=a.nef1[1]=a.nef2[1]=a.nef3[1]=1;
        a.nbf1[0]=a.nbf2[0]=a.nbf3[0]=a.nbf1[1]=a.nbf2[1]=a.nbf3[1]=27648;a.scale=1.0f;
        int nth=32;const int maximum=ggml_metal_pipeline_max_theads_per_threadgroup(normp_);
        while(nth<1728 && nth<maximum)nth*=2;nth=std::min({nth,maximum,1728});
        ggml_metal_encoder_set_pipeline(e,normp_);ggml_metal_encoder_set_bytes(e,&a,sizeof(a),0);
        ggml_metal_encoder_set_buffer(e,bind(x,27648),1);ggml_metal_encoder_set_buffer(e,bind(gamma,27648),2);
        ggml_metal_encoder_set_buffer(e,bind(x,27648),3);ggml_metal_encoder_set_buffer(e,bind(y,27648),4);
        ggml_metal_encoder_set_threadgroup_memory_size(e,normp_.smem,0);
        ggml_metal_encoder_dispatch_threadgroups(e,1,1,1,nth,1,1);
    }
    int normThreads()const{int nth=32;const int maximum=ggml_metal_pipeline_max_theads_per_threadgroup(normp_);while(nth<1728 && nth<maximum)nth*=2;return std::min({nth,maximum,1728});}
    NSDictionary* metadata()const{
        return @{@"projection":@"kernel_mul_mv_tq2_0_f32",@"nr0":@4,@"simd_groups":@2,
            @"projection_grid":@864,@"down_grid":@320,@"unary_grid":@7,@"unary_threads":@256,
            @"post_dispatches":@3,@"pre_subnorm_dispatches":@5,@"whole_dispatches":@7,
            @"count_status":@"source-matched leaf sequence; normal graph trace required independently",
            @"weighted_norm":@"actual get_pipeline_norm(rms,2,NONE)",@"epsilon":@(1.e-5f),
            @"gate_activation_alias":@NO,@"up_norm_alias":@NO,@"norm_threads":@(normThreads()),
            @"stock_original_order":@"gate,relu,up,square,mul,norm,down"};
    }
};
} // namespace bitnet_iphone_operator
