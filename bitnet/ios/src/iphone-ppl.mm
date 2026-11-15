#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <UIKit/UIKit.h>
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace bitnet_iphone_ppl {
namespace fs=std::filesystem;
static void require(bool ok,const std::string&message){if(!ok)throw std::runtime_error(message);}
static void write_json(const fs::path&p,id value){NSError*error=nil;NSData*d=[NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingPrettyPrinted error:&error];require(d&&[d writeToFile:@(p.c_str()) atomically:YES],"write JSON failed");}
static NSDictionary*read_json(const fs::path&p){NSData*d=[NSData dataWithContentsOfFile:@(p.c_str())];require(d!=nil,"quality fixture missing");id v=[NSJSONSerialization JSONObjectWithData:d options:0 error:nil];require([v isKindOfClass:NSDictionary.class],"quality fixture JSON malformed");return v;}
struct Log {int offloaded=-1,total=-1;std::string placement;};
static void logger(enum ggml_log_level,const char*text,void*opaque){
 std::fputs(text,stderr);fflush(stderr);Log*l=static_cast<Log*>(opaque);
 const char*p=std::strstr(text,"offloaded ");if(p){int a=-1,b=-1;if(std::sscanf(p,"offloaded %d/%d layers to GPU",&a,&b)==2){l->offloaded=a;l->total=b;}}
 if(std::strstr(text,"KV buffer")||std::strstr(text,"KV self size")||std::strstr(text,"KV cache")||std::strstr(text,"type_k"))l->placement+=text;
}
static void clear_custom(){
 for(NSString*key in NSProcessInfo.processInfo.environment)if([key hasPrefix:@"GGML_METAL_BITNET_"])unsetenv(key.UTF8String);
 for(const char*k:{"GGML_METAL_BITNET_POC","GGML_METAL_BITNET_FFN","GGML_METAL_BITNET_FFN_DECODE","GGML_METAL_BITNET_FFN_MECHANISM","GGML_METAL_BITNET_FFN_ABLATION","GGML_METAL_BITNET_FFN_ABLATION_CPU","GGML_METAL_BITNET_FFN_ABLATION_COUNT","GGML_METAL_BITNET_FFN_ABLATION_TRACE","GGML_METAL_BITNET_FFN_ABLATION_GPU","GGML_METAL_BITNET_FFN_ABLATION_REUSE_LAYER","GGML_METAL_BITNET_POC_TRACE","GGML_METAL_BITNET_FFN_TRACE","GGML_METAL_BITNET_QREADBACK_DIR","GGML_METAL_BITNET_FFN_SHADOW_DIR","GGML_METAL_BITNET_FFN_K_SHADOW_DIR","GGML_METAL_GRAPH_DEBUG","GGML_METAL_DEBUG_GRAPH","GGML_METAL_CAPTURE_COMPUTE","METAL_CAPTURE_ENABLED"})unsetenv(k);
}
static int run(const fs::path&model,const fs::path&fixture,const fs::path&out){
 require(!fs::exists(out),"refuse existing quality output");fs::create_directories(out);
 NSDictionary*request=read_json(fixture/"QUALITY.json");NSArray*array=request[@"tokens"];
 require([array isKindOfClass:NSArray.class]&&array.count==129,"quality requires BOS plus128 exact token IDs");
 require([array.firstObject intValue]==128000&&[request[@"scored_tokens"]intValue]==128,"quality BOS/scored-token contract");
 std::vector<llama_token>tokens;for(id value in array){require([value isKindOfClass:NSNumber.class],"token ID must be numeric");tokens.push_back([value intValue]);}
 write_json(out/"fixture.json",request);clear_custom();Log log;
 llama_log_set(logger,&log);llama_backend_init();
 struct BackendGuard{~BackendGuard(){llama_backend_free();llama_log_set(nullptr,nullptr);}}backend_guard;
 ggml_backend_dev_t devices[]={ggml_backend_dev_by_name("MTL0"),nullptr};require(devices[0]!=nullptr,"MTL0 unavailable");
 id<MTLDevice>gpu=MTLCreateSystemDefaultDevice();require(gpu!=nil,"Metal GPU missing");
 auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=999;mp.check_tensors=true;
 std::unique_ptr<llama_model,decltype(&llama_model_free)>m(llama_model_load_from_file(model.c_str(),mp),llama_model_free);
 require(bool(m)&&llama_model_n_layer(m.get())==30,"public2B model/block count invalid");
 const int vocab=llama_vocab_n_tokens(llama_model_get_vocab(m.get()));require(vocab==128256,"public2B vocabulary mismatch");
 for(auto token:tokens)require(token>=0&&token<vocab,"token ID outside vocabulary");
 require(log.offloaded==31&&log.total==31,"actual31/31 Metal offload not confirmed");
 auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=128;
 cp.n_threads=cp.n_threads_batch=6;cp.type_k=cp.type_v=GGML_TYPE_F16;
 cp.offload_kqv=true;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.cb_eval=nullptr;cp.cb_eval_user_data=nullptr;
 std::unique_ptr<llama_context,decltype(&llama_free)>ctx(llama_init_from_model(m.get(),cp),llama_free);require(bool(ctx),"context initialization failed");
 llama_memory_clear(llama_get_memory(ctx.get()),true);llama_synchronize(ctx.get());
 require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==-1,"initial KV not empty");
 std::ofstream binary(out/"logits.f32",std::ios::binary),csv(out/"nll.csv");require(bool(binary)&&bool(csv),"open quality outputs failed");
 csv<<"target_position,input_position,input_token,target_token,negative_log_likelihood,greedy_token,all_finite\n"<<std::setprecision(17);
 long double nll_sum=0;int completed=0;size_t finite=0;
 const auto thermal_before=NSProcessInfo.processInfo.thermalState;
 for(int i=0;i<128;++i){
  require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==i-1,"KV input position mismatch");
  llama_token token=tokens[i];require(llama_decode(ctx.get(),llama_batch_get_one(&token,1))==0,"quality llama_decode failed");
  llama_synchronize(ctx.get());require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()),0)==i,"KV decoded position mismatch");
  const float*y=llama_get_logits_ith(ctx.get(),-1);require(y!=nullptr,"missing quality logits");
  double maximum=-INFINITY;int greedy=0;for(int j=0;j<vocab;++j){require(std::isfinite(y[j]),"nonfinite quality logits");if(double(y[j])>maximum){maximum=y[j];greedy=j;}}
  double exp_sum=0;for(int j=0;j<vocab;++j)exp_sum+=std::exp(double(y[j])-maximum);
  const double nll=maximum+std::log(exp_sum)-double(y[tokens[i+1]]);require(std::isfinite(nll),"nonfinite quality NLL");
  nll_sum+=nll;finite+=vocab;binary.write(reinterpret_cast<const char*>(y),vocab*sizeof(float));require(bool(binary),"write quality logits failed");
  csv<<i+1<<','<<i<<','<<tokens[i]<<','<<tokens[i+1]<<','<<nll<<','<<greedy<<",1\n";require(bool(csv),"write quality NLL failed");++completed;
  std::printf("BITNET_QUALITY_GPU_COMPLETE input_position=%d target_position=%d backend=MTL0 finite_logits=%d synchronized=1\n",i,i+1,vocab);fflush(stdout);
 }
 binary.close();csv.close();const double mean=double(nll_sum/128);
 NSDictionary*result=@{@"schema":@1,@"analysis_completed":@YES,@"execution_pass":@YES,@"measurement_completed":@YES,
  @"model_quality_tested":@YES,@"model_quality_scope":@"128-token canonical WikiText test-split slice; nonrepresentative smoke only",
  @"model_semantics_pass":[NSNull null],@"model_semantics_verified":@NO,@"parity_pass":[NSNull null],@"fp64_pass":[NSNull null],@"model_output_pass":[NSNull null],
  @"current_path_semantics":@"Stock QVAC TQ2_0/F32; known public ActQuant omission is not repaired by this quality measurement",
  @"custom_routes_enabled":@NO,@"retokenized":@NO,@"template_applied":@NO,@"vocabulary":@(vocab),@"scored_tokens":@128,
  @"logit_vectors":@(completed),@"finite_logits":@(finite),@"nonfinite_logits":@0,
  @"sum_NLL":@(double(nll_sum)),@"mean_NLL":@(mean),@"perplexity":@(std::exp(mean)),
  @"scoring_arithmetic":@"F64 stable logsumexp over saved F32 logits; long-double accumulation of128 NLL values",
  @"GPU_completion_evidence":@"llama_synchronize after each successful one-token decode;128 complete logits readbacks; no individual Metal command-buffer callback instrumented",
  @"GPU_synchronized_decode_steps":@(completed),@"device":gpu.name,@"backend":@(ggml_backend_dev_description(devices[0])),@"iOS":UIDevice.currentDevice.systemVersion,
  @"offloaded_layers":@(log.offloaded),@"total_layers":@(log.total),@"kv_placement_log":@(log.placement.c_str()),
  @"threads":@6,@"n_ctx":@512,@"n_batch":@128,@"n_ubatch":@128,@"type_k":@"F16",@"type_v":@"F16",@"offload_kqv":@YES,@"flash_attention":@NO,
  @"prefill_execution":@"N1 sequential teacher forcing starting at explicit BOS; no batched prefill",
  @"thermal_before":@(thermal_before),@"thermal_after":@(NSProcessInfo.processInfo.thermalState),@"low_power":@(NSProcessInfo.processInfo.lowPowerModeEnabled),
  @"formal_performance_run":@NO,@"strict_exit_code":@0,@"strict_V1_exit_code":@2};
 write_json(out/"summary.json",result);return 0;
}
}
extern "C" int bitnet_run_iphone_ppl(const char*model,const char*fixture,const char*output){@autoreleasepool{
 try{return bitnet_iphone_ppl::run(model,fixture,output);}catch(const std::exception&e){std::fprintf(stderr,"BITNET_IPHONE_QUALITY_ERROR %s\n",e.what());return 2;}
}}
