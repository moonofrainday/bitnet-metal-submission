#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include "ggml.h"
#include "ggml-metal-device.h"
#include "ggml-metal-impl.h"
#include "iphone-down-controls.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bitnet_iphone_down {
namespace fs = std::filesystem;
constexpr int columns = 6912, rows = 2560, block_bytes = 66;
constexpr size_t row_bytes = 27*block_bytes, vector_bytes = columns*sizeof(float), guard = 128;
constexpr std::array<int,12> selected = {0,336,352,360,448,496,880,1144,1504,1824,2152,2504};
constexpr const char * model_sha = "52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2";
constexpr const char * weight_sha = "c032780f9245f4ecfe37fe117f0b6f16b5a76cfb72734c60b1f7e4beabe99888";
constexpr const char * input_sha = "c9e096fc21f1e6ab4dcc1c4329925942219f848b4ab082dc30b930fe2bf43e7a";

static void require(bool value, const std::string & message) {if (!value) throw std::runtime_error(message);}
static std::vector<uint8_t> read(const fs::path & path) {
    std::ifstream file(path, std::ios::binary); require(bool(file), "read " + path.string());
    return {std::istreambuf_iterator<char>(file), {}};
}
static std::string digest(const void * data, size_t bytes) {
    require(bytes <= std::numeric_limits<CC_LONG>::max(), "SHA extent");
    unsigned char hash[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(data, (CC_LONG)bytes, hash);
    char text[65]; for (size_t i=0;i<CC_SHA256_DIGEST_LENGTH;++i) std::snprintf(text+2*i,3,"%02x",hash[i]);
    return text;
}
static NSString * text(const std::string & value) {return [NSString stringWithUTF8String:value.c_str()];}
static id number(double value) {return std::isfinite(value) ? (id)@(value) : (id)[NSNull null];}
static NSDictionary * load(const fs::path & path) {
    NSData * data = [NSData dataWithContentsOfFile:@(path.c_str())]; require(data != nil, "read manifest");
    NSError * error = nil; id value = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    require([value isKindOfClass:NSDictionary.class], "manifest JSON"); return value;
}
static void json(const fs::path & path, id value) {
    NSError * error = nil; NSData * data = [NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingPrettyPrinted error:&error];
    require(data && [data writeToFile:@(path.c_str()) atomically:YES], "write JSON");
}
static void save(const fs::path & path, const void * data, size_t bytes) {
    std::ofstream file(path, std::ios::binary); require(bool(file.write((const char *)data, bytes)), "write " + path.string());
}

class ActualStock {
    ggml_metal_device_t device_ = nullptr;
    ggml_context * context_ = nullptr;
    ggml_metal_pipeline_with_params pipeline_{};
    ggml_metal_kargs_mul_mv args_{};
public:
    ActualStock() {
        static_assert(sizeof(ggml_metal_kargs_mul_mv)==120, "Native runtime ABI changed");
        device_ = ggml_metal_device_init(0,1); require(device_, "actual Metal device");
        context_ = ggml_init({8*ggml_tensor_overhead(),nullptr,true}); require(context_, "ggml context");
        auto full_w = ggml_new_tensor_2d(context_,GGML_TYPE_TQ2_0,columns,rows);
        auto small_w = ggml_new_tensor_2d(context_,GGML_TYPE_TQ2_0,columns,8);
        auto x = ggml_new_tensor_2d(context_,GGML_TYPE_F32,columns,1);
        auto full_y = ggml_mul_mat(context_,full_w,x), small_y = ggml_mul_mat(context_,small_w,x);
        auto library = ggml_metal_device_get_library(device_);
        auto full = ggml_metal_library_get_pipeline_mul_mv(library,full_y,false,false);
        pipeline_ = ggml_metal_library_get_pipeline_mul_mv(library,small_y,false,false);
        require(pipeline_.pipeline && pipeline_.pipeline==full.pipeline && pipeline_.nr0==4 && pipeline_.nsg==2,
                "Full/sliced stock PSO identity or four-row/two-SIMD specialization changed");
        require(ggml_metal_pipeline_thread_execution_width(pipeline_)==32 &&
                ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_)>=64, "Stock PSO limits");
        args_ = {columns,8,1,66,row_bytes,8*row_bytes,8*row_bytes,
                 columns,1,1,4,vector_bytes,vector_bytes,vector_bytes,8,1,4,1,1,0.0f};
    }
    ~ActualStock() {if(context_)ggml_free(context_); if(device_)ggml_metal_device_free(device_);}
    ActualStock(const ActualStock&)=delete;
    ActualStock&operator=(const ActualStock&)=delete;
    id<MTLDevice> device() const {return (__bridge id<MTLDevice>)ggml_metal_device_get_obj(device_);}
    id<MTLCommandQueue> queue() const {return (__bridge id<MTLCommandQueue>)ggml_metal_device_get_queue(device_);}
    const ggml_metal_kargs_mul_mv & args() const {return args_;}
    NSDictionary * metadata() const {
        return @{@"native_pipeline":text(ggml_metal_pipeline_name(pipeline_)), @"full_and_slice_same_PSO":@YES,
                 @"ABI_bytes":@(sizeof(args_)), @"rows_per_SIMD":@4, @"SIMD_groups":@2,
                 @"grid":@[@1,@1,@1], @"threads":@[@32,@2,@1], @"runtime_ne01":@8, @"runtime_ne0":@8,
                 @"runtime_K":@(columns), @"row_bytes":@(row_bytes), @"threadgroup_memory_bytes":@(pipeline_.smem)};
    }
    void encode(ggml_metal_encoder_t encoder, id<MTLComputePipelineState> custom,
                id<MTLBuffer> w, id<MTLBuffer> x, id<MTLBuffer> y) {
        if(custom) [(__bridge id<MTLComputeCommandEncoder>)ggml_metal_encoder_get_obj(encoder) setComputePipelineState:custom];
        else ggml_metal_encoder_set_pipeline(encoder,pipeline_);
        auto args=args_; ggml_metal_encoder_set_bytes(encoder,&args,sizeof(args),0);
        ggml_metal_encoder_set_buffer(encoder,{(__bridge void*)w,0},1);
        ggml_metal_encoder_set_buffer(encoder,{(__bridge void*)x,0},2);
        ggml_metal_encoder_set_buffer(encoder,{(__bridge void*)y,guard},3);
        ggml_metal_encoder_set_buffer(encoder,{(__bridge void*)y,guard},4);
        ggml_metal_encoder_set_buffer(encoder,{(__bridge void*)x,0},5);
        ggml_metal_encoder_set_threadgroup_memory_size(encoder,pipeline_.smem,0);
        ggml_metal_encoder_dispatch_threadgroups(encoder,1,1,1,32,2,1);
    }
};

static id<MTLComputePipelineState> candidate(id<MTLDevice> device, const fs::path & path) {
    NSError * error=nil; auto library=[device newLibraryWithURL:[NSURL fileURLWithPath:@(path.c_str())] error:&error];
    require(library != nil,"load candidate library");
    auto values=[MTLFunctionConstantValues new];
    const short two=2,one=1;
    [values setConstantValue:&two type:MTLDataTypeShort atIndex:600];
    for(NSUInteger index : {602u,603u,604u}) [values setConstantValue:&one type:MTLDataTypeShort atIndex:index];
    auto function=[library newFunctionWithName:@"stock_down_control" constantValues:values error:&error];
    require(function != nil,"specialize candidate function");
    auto pipeline=[device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline && pipeline.threadExecutionWidth==32 && pipeline.maxTotalThreadsPerThreadgroup>=64,"candidate PSO limits");
    return pipeline;
}

static std::array<double,8> reference(const uint8_t * weights, const float * x) {
    std::array<double,8> result{};
    for(int row=0;row<8;++row) for(int block=0;block<27;++block) {
        const auto * packed=weights+row*row_bytes+block*block_bytes;
        uint16_t bits=uint16_t(packed[64]) | (uint16_t(packed[65])<<8); _Float16 half; std::memcpy(&half,&bits,2);
        const double scale=double(half); require(std::isfinite(scale),"nonfinite block scale");
        for(int plane_half=0;plane_half<2;++plane_half) for(int plane=0;plane<4;++plane) for(int j=0;j<32;++j) {
            const int code=(packed[32*plane_half+j]>>(2*plane))&3; require(code!=3,"invalid TQ2_0 trit code");
            const int k=block*256+plane_half*128+plane*32+j;
            result[row]+=double(code-1)*scale*double(x[k]);
        }
    }
    return result;
}

static int run(const fs::path & fixture, const fs::path & libraries, const fs::path & output, const std::string & mode) {
    require(mode=="verify","Only the selected numerical verify mode exists; no timing mode");
    require(!fs::exists(output),"Refusing an existing output directory");
    fs::create_directories(output);
    auto manifest=load(fixture/"MANIFEST.json");
    require([manifest[@"model_sha256"]isEqual:@(model_sha)] && [manifest[@"history"]isEqual:@"independent"] &&
            [manifest[@"owner_route"]isEqual:@"S"] && [manifest[@"layer"]intValue]==0 && [manifest[@"step"]intValue]==16,
            "Expected the fixed public-model independent stock layer-0/step-16 fixture");
    auto weights=read(fixture/"w.tq2"), input=read(fixture/"x.f32");
    require(weights.size()==rows*row_bytes && digest(weights.data(),weights.size())==weight_sha &&
            [manifest[@"weight"][@"sha256"]isEqual:@(weight_sha)],"Actual down weight identity");
    require(input.size()==vector_bytes && digest(input.data(),input.size())==input_sha &&
            [manifest[@"input"][@"sha256"]isEqual:@(input_sha)],"Actual stock norm input identity");
    NSArray * groups=manifest[@"row_groups"]; require(groups.count==selected.size(),"Expected twelve selected eight-row groups");
    for(size_t i=0;i<selected.size();++i) require([groups[i]intValue]==selected[i],"Selected row group changed");
    require([manifest[@"tolerance"]isEqual:@{@"abs":@(2e-4),@"relative":@(2e-5)}],"Frozen tolerance changed");
    ActualStock stock; auto device=stock.device(); require([device.name isEqual:@"Apple A16 GPU"],"Requires Apple A16 GPU");
    std::array<std::string,3> names={"S","S-wrap","correction-end"};
    std::array<id<MTLComputePipelineState>,3> pipelines={nil,nil,nil};
    for(int route=1;route<3;++route) {
        NSDictionary * entry=manifest[@"libraries"][text(names[route])];
        require([entry isKindOfClass:NSDictionary.class] && [entry[@"file"]isKindOfClass:NSString.class],"Missing library identity");
        fs::path file=libraries/[(NSString*)entry[@"file"]UTF8String]; auto bytes=read(file);
        require([entry[@"sha256"]isEqual:text(digest(bytes.data(),bytes.size()))],"Candidate library digest");
        pipelines[route]=candidate(device,file);
    }
    auto x=[device newBufferWithBytes:input.data() length:input.size() options:MTLResourceStorageModeShared]; require(x!=nil,"X allocation");
    for(int k=0;k<columns;++k) require(std::isfinite(((const float*)x.contents)[k]),"Nonfinite X");
    save(output/"x.f32",input.data(),input.size());
    save(output/"args.bin",&stock.args(),sizeof(stock.args()));
    std::ofstream csv(output/"rows.csv"); csv<<std::setprecision(17)<<"group,route,row,reference,actual,abs_error,tolerance,error_over_tolerance,relative_error,failed,nonfinite,stock_bit_difference,stock_actual,abs_diff_from_stock\n";
    NSMutableArray * cases=[NSMutableArray array]; bool fp64_pass=true,stock_parity=true;
    for(int start : selected) {
        fs::path folder=output/("group-"+std::to_string(start)); fs::create_directories(folder);
        auto data=weights.data()+size_t(start)*row_bytes; const size_t bytes=8*row_bytes;
        auto w=[device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared]; require(w!=nil,"W allocation");
        auto ref=reference(data,(const float*)x.contents); save(folder/"w.tq2",data,bytes); save(folder/"reference.f64",ref.data(),sizeof(ref));
        std::array<float,8> baseline{};
        for(int route=0;route<3;++route) {
            @autoreleasepool {
                auto y=[device newBufferWithLength:2*guard+8*sizeof(float) options:MTLResourceStorageModeShared]; require(y!=nil,"Y allocation");
                std::memset(y.contents,0xa5,y.length); auto actual=(float*)((char*)y.contents+guard);
                for(int i=0;i<8;++i) actual[i]=NAN;
                auto command=[stock.queue() commandBuffer]; command.label=text("down-control-"+names[route]+"-group-"+std::to_string(start));
                auto encoder=ggml_metal_encoder_init((__bridge void*)command,false); require(encoder,"Encoder allocation");
                stock.encode(encoder,pipelines[route],w,x,y); ggml_metal_encoder_end_encoding(encoder); ggml_metal_encoder_free(encoder);
                [command commit]; [command waitUntilCompleted];
                require(command.status==MTLCommandBufferStatusCompleted && !command.error,"GPU command did not complete");
                const auto * raw=(const uint8_t*)y.contents;
                for(size_t i=0;i<guard;++i) require(raw[i]==0xa5 && raw[guard+8*sizeof(float)+i]==0xa5,"Output canary changed");
                require(digest(w.contents,w.length)==digest(data,bytes) && digest(x.contents,x.length)==input_sha,"GPU modified W/X");
                if(route==0)std::copy(actual,actual+8,baseline.begin());
                int failed=0,nonfinite=0,bits=0,stock_failed=0; double maxabs=0,maxratio=0,error2=0,ref2=0,max_stock_abs=0;
                NSMutableArray * failed_rows=[NSMutableArray array];
                for(int i=0;i<8;++i) {
                    double error=std::abs(double(actual[i])-ref[i]),tol=2e-4+2e-5*std::abs(ref[i]);
                    bool finite=std::isfinite(actual[i])&&std::isfinite(ref[i]),bad=!finite||error>tol;
                    bool different=std::memcmp(actual+i,baseline.data()+i,sizeof(float))!=0;
                    const double stock_error=std::abs(double(actual[i])-double(baseline[i]));
                    max_stock_abs=std::max(max_stock_abs,stock_error);
                    stock_failed+=!finite||stock_error>2e-4+2e-5*std::abs(double(baseline[i]));
                    failed+=bad; nonfinite+=!finite; bits+=different; maxabs=std::max(maxabs,error); maxratio=std::max(maxratio,error/tol);
                    error2+=error*error; ref2+=ref[i]*ref[i]; if(bad)[failed_rows addObject:@(start+i)];
                    csv<<start<<','<<names[route]<<','<<start+i<<','<<ref[i]<<','<<actual[i]<<','<<error<<','<<tol<<','<<error/tol<<',';
                    if(ref[i]==0)csv<<"zero-reference";else csv<<error/std::abs(ref[i]);
                    csv<<','<<bad<<','<<!finite<<','<<different<<','<<baseline[i]<<','<<stock_error<<'\n';
                }
                fp64_pass &= failed==0; if(route==1)stock_parity &= bits==0;
                double relative_l2=ref2>0?std::sqrt(error2/ref2):(error2==0?0:INFINITY);
                save(folder/(names[route]+".f32"),actual,8*sizeof(float));
                [cases addObject:@{@"group":@(start),@"route":text(names[route]),@"rows":@8,@"failed_rows":@(failed),
                    @"failed_indices":failed_rows,@"nonfinite_rows":@(nonfinite),@"stock_bit_different_rows":@(bits),
                    @"max_abs":number(maxabs),@"max_error_over_tolerance":number(maxratio),@"relative_L2":number(relative_l2),
                    @"stock_max_abs":number(max_stock_abs),@"stock_tolerance_failed_rows":@(stock_failed),
                    @"GPU_completed":@YES,@"GPU_status":@4,@"output_canaries_pass":@YES,@"inputs_unchanged":@YES,
                    @"W_group_sha256":text(digest(data,bytes)),@"X_sha256":@(input_sha)}];
                std::fprintf(stderr,"BITNET_DOWN_CONTROL_COMPLETE route=%s group=%d rows=8 grid=[1,1,1] threads=[32,2,1] status=4 completed=1 fp64_failed=%d stock_bits=%d\n",
                        names[route].c_str(),start,failed,bits);
            }
        }
    }
    csv.close(); require(bool(csv),"Write rows.csv failed");
    NSDictionary * result=@{@"analysis_completed":@YES,@"execution_pass":@YES,@"fp64_pass":@(fp64_pass),
        @"parity_pass":@(stock_parity),@"parity_scope":@"S-wrap versus actual S selected outputs, bitwise",
        @"model_output_pass":[NSNull null],@"strict_exit_code":@(fp64_pass?0:1),@"cases":cases,
        @"rows_per_route":@96,@"route_count":@3,@"GPU_command_buffers_completed":@36,@"device":device.name,
        @"model_sha256":@(model_sha),@"W_sha256":@(weight_sha),@"X_sha256":@(input_sha),
        @"tolerance":@{@"abs":@(2e-4),@"relative":@(2e-5)},@"stock":stock.metadata(),
        @"timing_measured":@NO,@"model_quality_tested":@NO,
        @"scope":@"12 selected eight-row groups; one real stock norm input; not model inference"};
    json(output/"summary.json",result); return fp64_pass?0:1;
}
}

extern "C" int bitnet_run_iphone_down(const char * fixture, const char * libraries, const char * output, const char * mode) {
    @autoreleasepool {
        if(!fixture || !libraries || !output || !mode) {
            std::fprintf(stderr,"BITNET_DOWN_CONTROL_ERROR Missing native argument\n"); return 2;
        }
        bool existed=true;
        try {existed=std::filesystem::exists(output);return bitnet_iphone_down::run(fixture,libraries,output,mode);}
        catch(const std::exception & error) {
            std::fprintf(stderr,"BITNET_DOWN_CONTROL_ERROR %s\n",error.what());
            try {
                if(!existed && std::filesystem::exists(output)) bitnet_iphone_down::json(std::filesystem::path(output)/"error.json",
                        @{@"analysis_completed":@NO,@"execution_pass":@NO,@"fp64_pass":[NSNull null],
                          @"strict_exit_code":@2,@"error":[NSString stringWithUTF8String:error.what()]});
            } catch(...) {}
            return 2;
        }
    }
}
