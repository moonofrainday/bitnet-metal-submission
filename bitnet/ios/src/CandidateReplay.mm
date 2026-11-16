#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <UIKit/UIKit.h>
#import <CommonCrypto/CommonDigest.h>
#import <llama/llama.h>
#import <llama/ggml-backend.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int bitnet_run_iphone_operator(const char*,const char*,const char*,const char*);
extern "C" int bitnet_run_iphone_q(const char*,const char*,const char*,const char*);
extern "C" int bitnet_run_iphone_down(const char*,const char*,const char*,const char*);
extern "C" int bitnet_run_iphone_original_poc(const char*,const char*,const char*,const char*);
extern "C" int bitnet_run_iphone_ppl(const char*,const char*,const char*);
extern "C" int bitnet_actquant_run_config(const char*,const char*);
extern "C" void ggml_metal_bitnet_shadow_wait(void);
namespace bitnet_candidate {
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
static void require(bool v,const std::string&m){if(!v)throw std::runtime_error(m);}
static double elapsed(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
static double seconds(Clock::time_point a){return std::chrono::duration<double>(a.time_since_epoch()).count();}
static std::string thermal(){switch(NSProcessInfo.processInfo.thermalState){case NSProcessInfoThermalStateNominal:return "nominal";case NSProcessInfoThermalStateFair:return "fair";case NSProcessInfoThermalStateSerious:return "serious";case NSProcessInfoThermalStateCritical:return "critical";}return "unknown";}
static std::vector<llama_token> ids(const fs::path&p){std::ifstream f(p);std::vector<llama_token>v;llama_token t;while(f>>t)v.push_back(t);require(f.eof(),"cannot read IDs "+p.string());return v;}
static void save(const fs::path&p,const void*d,size_t n){fs::create_directories(p.parent_path());std::ofstream f(p,std::ios::binary);require(bool(f.write((const char*)d,n)),"write failed "+p.string());}
static void route(const std::string&r){require(r=="S"||r=="G"||r=="K"||r=="F8"||r=="Q4"||r=="QB","invalid route");for(const char*k:{"GGML_METAL_BITNET_Q_GUARD_ONLY","GGML_METAL_BITNET_POC","GGML_METAL_BITNET_FFN","GGML_METAL_BITNET_FFN_MECHANISM","GGML_METAL_BITNET_POC_ROWS_PER_SIMD","GGML_METAL_BITNET_POC_SIMD_PER_GROUP"})unsetenv(k);if(r=="Q4"||r=="QB"){setenv(r=="QB"?"GGML_METAL_BITNET_Q_GUARD_ONLY":"GGML_METAL_BITNET_POC","1",1);setenv("GGML_METAL_BITNET_POC_ROWS_PER_SIMD","4",1);setenv("GGML_METAL_BITNET_POC_SIMD_PER_GROUP","1",1);}else if(r=="F8")setenv("GGML_METAL_BITNET_FFN","fused",1);else if(r!="S")setenv("GGML_METAL_BITNET_FFN_MECHANISM",r.c_str(),1);}
static void trace(bool enabled){for(const char*k:{"GGML_METAL_BITNET_POC_TRACE","GGML_METAL_BITNET_FFN_TRACE"}){if(enabled)setenv(k,"1",1);else unsetenv(k);}}
struct Options{fs::path model,prompt,prompt_ids,continuation_ids,out,library,k_library,k_shadow,f8_library,q_library;std::string operation="verify",comparison="ALL",layers="0",q_fallback;int pairs=21,rounds=3,round_index=1,warmups=2;std::vector<std::string> routes;bool traced=false,allow_missing_k=false,k_shadow_all=false,q_readback=false;};
struct Replay{double wall=0,prefill=0,reset=0,begin_s=0,end_s=0;std::vector<double>token_ms,token_start,token_end;std::vector<float>logits;std::vector<int>greedy;std::string before,after;bool low_before=false,low_after=false;};
struct Diff{size_t bits=0,fail=0;double max_abs=0;};
static Diff difference(const Replay&a,const Replay&b){require(!a.logits.empty()&&a.logits.size()==b.logits.size(),"wrong logit extent");Diff d;for(size_t i=0;i<a.logits.size();++i){float x=a.logits[i],y=b.logits[i];require(std::isfinite(x)&&std::isfinite(y),"nonfinite logits");double e=std::abs(double(x)-y);d.bits+=std::memcmp(&x,&y,4)!=0;d.fail+=e>2e-4+2e-5*std::abs(double(x));d.max_abs=std::max(d.max_abs,e);}return d;}
static void emit(std::ostream&f,const Diff&d){f<<"{\"different_elements\":"<<d.bits<<",\"legacy_stock_tolerance_failed_elements\":"<<d.fail<<",\"max_abs\":"<<std::setprecision(17)<<d.max_abs<<"}";}
class Session{
 std::unique_ptr<llama_model,decltype(&llama_model_free)>model_{nullptr,llama_model_free};std::unique_ptr<llama_context,decltype(&llama_free)>ctx_{nullptr,llama_free};std::vector<llama_token>prompt_,continuation_;int vocabulary_=0;uint64_t path_=0;std::string o_layers_,q_fallback_;fs::path k_shadow_;bool k_shadow_all_,q_readback_requested_,q_readback_active_=false;
 void stop_q_readback(){q_readback_active_=false;for(const char*k:{"GGML_METAL_BITNET_QREADBACK_DIR","GGML_METAL_BITNET_QREADBACK_ROUTE","GGML_METAL_BITNET_QREADBACK_STEP"})unsetenv(k);}
 void decoded(const std::vector<llama_token>&tokens,int pos,int step){std::string v=std::to_string(step);setenv("GGML_METAL_BITNET_FFN_STEP",v.c_str(),1);if(q_readback_active_&&step>0)setenv("GGML_METAL_BITNET_QREADBACK_STEP",v.c_str(),1);else unsetenv("GGML_METAL_BITNET_QREADBACK_STEP");require(llama_memory_seq_pos_max(llama_get_memory(ctx_.get()),0)+1==pos,"wrong KV prefix");require(llama_decode(ctx_.get(),llama_batch_get_one(const_cast<llama_token*>(tokens.data()),tokens.size()))==0,"llama_decode failed");llama_synchronize(ctx_.get());if(q_readback_active_)ggml_metal_bitnet_shadow_wait();require(llama_memory_seq_pos_max(llama_get_memory(ctx_.get()),0)==pos+int(tokens.size())-1,"wrong KV end");}
 void snapshot(Replay&r,const fs::path&folder,int step){const float*p=llama_get_logits_ith(ctx_.get(),-1);require(p,"missing logits");require(std::all_of(p,p+vocabulary_,[](float x){return std::isfinite(x);}),"nonfinite logits");r.greedy.push_back(std::max_element(p,p+vocabulary_)-p);r.logits.insert(r.logits.end(),p,p+vocabulary_);if(!folder.empty())save(folder/"logits"/("step-"+std::to_string(step)+".f32"),p,vocabulary_*sizeof(float));}
 void reset(){llama_synchronize(ctx_.get());llama_memory_clear(llama_get_memory(ctx_.get()),true);llama_synchronize(ctx_.get());require(llama_memory_seq_pos_max(llama_get_memory(ctx_.get()),0)==-1,"KV reset failed");}
public:
 explicit Session(const Options&o):o_layers_(o.layers),q_fallback_(o.q_fallback),k_shadow_(o.k_shadow),k_shadow_all_(o.k_shadow_all),q_readback_requested_(o.q_readback){
  stop_q_readback();
  for(const char*k:{"GGML_METAL_BITNET_Q_GUARD_ONLY","GGML_METAL_BITNET_POC","GGML_METAL_BITNET_FFN","GGML_METAL_BITNET_FFN_DECODE","GGML_METAL_BITNET_FFN_MECHANISM","GGML_METAL_BITNET_FFN_SHADOW_DIR","GGML_METAL_BITNET_FFN_TIME","GGML_METAL_BITNET_LAYER_TIME","GGML_METAL_BITNET_FFN_ABLATION","GGML_METAL_BITNET_FFN_ABLATION_CPU","GGML_METAL_BITNET_FFN_ABLATION_COUNT","GGML_METAL_BITNET_FFN_ABLATION_TRACE","GGML_METAL_BITNET_FFN_ABLATION_GPU","GGML_METAL_BITNET_FFN_ABLATION_REUSE_LAYER","GGML_METAL_GRAPH_DEBUG","GGML_METAL_DEBUG_GRAPH","GGML_METAL_CAPTURE_COMPUTE","METAL_CAPTURE_ENABLED"})unsetenv(k);
  trace(false);setenv("GGML_METAL_BITNET_LAYERS",o.layers.c_str(),1);setenv("GGML_METAL_BITNET_FFN_MECHANISM_METALLIB",o.library.c_str(),1);setenv("GGML_METAL_BITNET_FFN_K_METALLIB",o.k_library.c_str(),1);setenv("GGML_METAL_BITNET_FFN_FUSED_METALLIB",o.f8_library.c_str(),1);setenv("GGML_METAL_BITNET_POC_METALLIB",o.q_library.c_str(),1);
  prompt_=ids(o.prompt_ids);continuation_=ids(o.continuation_ids);require(prompt_.size()==27&&continuation_.size()==32,"expected27/32 token IDs");llama_backend_init();ggml_backend_dev_t devices[]={ggml_backend_dev_by_name("MTL0"),nullptr};require(devices[0],"MTL0 unavailable");auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=999;mp.check_tensors=true;model_.reset(llama_model_load_from_file(o.model.c_str(),mp));require(bool(model_)&&llama_model_n_layer(model_.get())==30,"model/block count invalid");const auto*vocab=llama_model_get_vocab(model_.get());vocabulary_=llama_vocab_n_tokens(vocab);
  std::ifstream pf(o.prompt,std::ios::binary);std::string text((std::istreambuf_iterator<char>(pf)),{});std::vector<llama_token>actual(text.size()+2);int n=llama_tokenize(vocab,text.data(),text.size(),actual.data(),actual.size(),true,true);require(n==int(prompt_.size())&&std::equal(prompt_.begin(),prompt_.end(),actual.begin()),"prompt differs from pinned IDs");auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=128;cp.n_threads=cp.n_threads_batch=6;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.cb_eval=nullptr;cp.cb_eval_user_data=nullptr;ctx_.reset(llama_init_from_model(model_.get(),cp));require(bool(ctx_),"context init failed");
  std::ofstream f(o.out/"config.json");f<<"{\"driver_version\":1,\"operation\":"<<std::quoted(o.operation)<<",\"comparison\":"<<std::quoted(o.comparison)<<",\"layers\":"<<std::quoted(o.layers)<<",\"backend\":"<<std::quoted(ggml_backend_dev_description(devices[0]))<<",\"one_model_and_context\":true,\"kv_data_and_metadata_clear_each_path\":true,\"Q\":\"stock except optional layer0 Q4 route\",\"FFN\":\"S/G/K/F8-physical-device\",\"n_ctx\":512,\"n_batch\":128,\"n_ubatch\":128,\"threads\":6,\"kv_k\":\"f16\",\"kv_v\":\"f16\",\"flash_attention\":false,\"callbacks\":false,\"rows_per_simd\":4,\"simd_per_group\":2,\"dual_grid\":[864,1,2],\"G_post_grid\":[27,1,1],\"G_post_threads\":[256,1,1],\"K_postnorm_grid\":[1,1,1],\"K_postnorm_threads\":[1024,1,1],\"threads_per_threadgroup\":[32,2,1],\"rounds\":"<<o.rounds<<",\"round_index\":"<<o.round_index<<",\"pairs_per_comparison_per_round\":"<<o.pairs<<",\"warmups_per_route_per_phase\":"<<o.warmups<<",\"vocabulary\":"<<vocabulary_<<",\"nominal_host_clock_period_ns\":"<<double(Clock::period::num)*1e9/Clock::period::den<<",\"stock_logit_parity_required\":false,\"model_quality_tested\":false,\"prompt_ids\":[";for(size_t i=0;i<prompt_.size();++i)f<<(i?",":"")<<prompt_[i];f<<"],\"continuation_ids\":[";for(size_t i=0;i<continuation_.size();++i)f<<(i?",":"")<<continuation_[i];f<<"]}\n";
 }
 ~Session(){stop_q_readback();ctx_.reset();model_.reset();llama_backend_free();}
 Replay replay(const std::string&r,const fs::path&folder,bool capture,bool traced,const std::string&phase){
  stop_q_readback();
  uint64_t id=++path_;trace(false);unsetenv("GGML_METAL_BITNET_FFN_K_SHADOW_DIR");unsetenv("GGML_METAL_BITNET_FFN_K_SHADOW_ALL");route(r);unsetenv("GGML_METAL_BITNET_FFN_DECODE");Replay v;v.before=thermal();v.low_before=NSProcessInfo.processInfo.lowPowerModeEnabled;v.token_ms.reserve(32);v.token_start.reserve(32);v.token_end.reserve(32);auto a=Clock::now();reset();v.reset=elapsed(a);a=Clock::now();if(traced){trace(true);std::fprintf(stderr,"FFNK_PREFILL_BEGIN id=%llu route=%s\n",(unsigned long long)id,r.c_str());}decoded(prompt_,0,0);trace(false);if(traced)std::fprintf(stderr,"FFNK_PREFILL_END id=%llu route=%s\n",(unsigned long long)id,r.c_str());v.prefill=elapsed(a);if(capture)snapshot(v,folder,0);setenv("GGML_METAL_BITNET_FFN_DECODE","1",1);trace(traced);if(traced&&r=="K"&&!k_shadow_.empty()){setenv("GGML_METAL_BITNET_FFN_K_SHADOW_DIR",k_shadow_.c_str(),1);if(k_shadow_all_)setenv("GGML_METAL_BITNET_FFN_K_SHADOW_ALL","1",1);}
  const bool readback=q_readback_requested_&&capture&&phase=="verify";
  if(r=="Q4"&&q_fallback_=="invalid-geometry"){setenv("GGML_METAL_BITNET_POC_ROWS_PER_SIMD","3",1);}
  if(readback){require((r=="S"||r=="Q4"||r=="QB")&&!folder.empty(),"Q readback needs an S/Q4/QB correctness folder");const auto directory=folder/"q-readback";require(!fs::exists(directory),"refuse existing Q readback folder");fs::create_directories(directory);setenv("GGML_METAL_BITNET_QREADBACK_DIR",directory.c_str(),1);setenv("GGML_METAL_BITNET_QREADBACK_ROUTE",r.c_str(),1);q_readback_active_=true;}
  if(traced)std::fprintf(stderr,"FFNK_PATH_BEGIN id=%llu route=%s phase=%s layers=%s context=%p\n",(unsigned long long)id,r.c_str(),phase.c_str(),o_layers_.c_str(),(void*)ctx_.get());auto begin=Clock::now();v.begin_s=seconds(begin);
  for(int i=0;i<32;++i){auto b=Clock::now();decoded({continuation_[i]},prompt_.size()+i,i+1);auto e=Clock::now();v.token_ms.push_back(std::chrono::duration<double,std::milli>(e-b).count());v.token_start.push_back(seconds(b));v.token_end.push_back(seconds(e));if(capture)snapshot(v,folder,i+1);}
  auto end=Clock::now();v.end_s=seconds(end);v.wall=std::chrono::duration<double,std::milli>(end-begin).count();v.after=thermal();v.low_after=NSProcessInfo.processInfo.lowPowerModeEnabled;stop_q_readback();trace(false);if(traced)std::fprintf(stderr,"FFNK_PATH_END id=%llu route=%s phase=%s layers=%s context=%p\n",(unsigned long long)id,r.c_str(),phase.c_str(),o_layers_.c_str(),(void*)ctx_.get());if(capture){std::ofstream f(folder/"greedy.tsv");f<<"after_step\ttoken_id\n";for(size_t i=0;i<v.greedy.size();++i)f<<i<<'\t'<<v.greedy[i]<<'\n';std::ofstream p(folder/"path.json");p<<"{\"route\":"<<std::quoted(r)<<",\"path_id\":"<<id<<",\"context\":"<<std::quoted(std::to_string((uintptr_t)ctx_.get()))<<",\"phase\":"<<std::quoted(phase)<<",\"callbacks\":false,\"trace\":"<<(traced?"true":"false")<<",\"q_readback\":"<<(readback?"true":"false")<<",\"model_quality_tested\":false}\n";}return v;
 }
};

static NSString* hashFile(const fs::path&p){std::ifstream f(p,std::ios::binary);require(bool(f),"cannot hash "+p.string());CC_SHA256_CTX ctx;CC_SHA256_Init(&ctx);std::vector<char>b(1<<20);while(f){f.read(b.data(),b.size());const auto n=f.gcount();if(n)CC_SHA256_Update(&ctx,b.data(),(CC_LONG)n);}require(f.eof(),"hash read failed");unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256_Final(digest,&ctx);NSMutableString*s=[NSMutableString string];for(auto v:digest)[s appendFormat:@"%02x",v];return s;}
static std::string str(id value){require([value isKindOfClass:NSString.class],"string field missing");return [(NSString*)value UTF8String];}
static void json(const fs::path&p,id value){NSError*error=nil;auto d=[NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingPrettyPrinted error:&error];require(d&&[d writeToFile:@(p.c_str()) atomically:YES],"JSON write failed "+p.string());}
static NSDictionary* qReadbackManifest(const fs::path&directory,const std::string&route){
 NSMutableDictionary*files=[NSMutableDictionary dictionary];size_t finite_values=0;
 auto add=[&](const std::string&relative,size_t bytes,bool check_finite){
  const auto file=directory/relative;require(fs::is_regular_file(file)&&fs::file_size(file)==bytes,"Q readback missing/wrong-sized file "+file.string());
  if(check_finite){std::vector<float>values(bytes/sizeof(float));std::ifstream input(file,std::ios::binary);require(bool(input.read((char*)values.data(),bytes)),"Q readback read failed");require(std::all_of(values.begin(),values.end(),[](float x){return std::isfinite(x);}),"Q readback nonfinite values");finite_values+=values.size();}
  files[@(relative.c_str())]=@{@"bytes":@(bytes),@"sha256":hashFile(file)};
 };
 add("weights/w.tq2",1689600,false);
 for(int step=1;step<=32;++step){const auto prefix="step-"+std::to_string(step)+"/";add(prefix+"x.f32",10240,true);add(prefix+"y.f32",10240,true);if(step==1||step==16||step==32)add(prefix+"l_out.f32",10240,true);}
 require(files.count==68&&finite_values==171520,"Q readback coverage mismatch");
 NSDictionary*manifest=@{@"schema_version":@1,@"route":@(route.c_str()),@"node":@"Qcur-0",@"weight_tensor":@"blk.0.attn_q.weight",@"Q_output_stage":@"before RoPE",@"Q_shape":@[@2560,@2560],@"activation_type":@"F32",@"output_type":@"F32",@"weight_type":@"TQ2_0",@"block_bytes":@66,@"decode_steps":@32,@"l_out_steps":@[@1,@16,@32],@"file_count":@68,@"files":files,@"finite_values_checked":@(finite_values),@"nonfinite_values":@0,@"coverage_pass":@YES,@"copy_group_waited":@YES,@"fp64_pass":[NSNull null],@"model_quality_tested":@NO};
 json(directory/"manifest.json",manifest);return manifest;
}
static bool flag(NSDictionary*c,NSString*k){return [c[k] boolValue];}
static fs::path actquantDocumentPath(const fs::path&docs,id value,bool directory){
 const std::string name=str(value);
 require(!name.empty()&&name.size()<=160&&name.front()!='.'&&std::all_of(name.begin(),name.end(),[](unsigned char ch){return (ch>='a'&&ch<='z')||(ch>='A'&&ch<='Z')||(ch>='0'&&ch<='9')||ch=='.'||ch=='-'||ch=='_';}),"invalid ActQuant document basename");
 const auto path=fs::canonical(docs/name);
 require(path.parent_path()==fs::canonical(docs),"ActQuant document escapes app Documents");
 require(directory?fs::is_directory(path):fs::is_regular_file(path),"ActQuant document has wrong kind");return path;
}
static NSDictionary* actquantReadJSON(const fs::path&path,size_t limit=4*1024*1024){
 require(fs::file_size(path)<=limit,"ActQuant manifest too large");
 NSData*bytes=[NSData dataWithContentsOfFile:@(path.c_str())];NSError*error=nil;
 NSDictionary*result=bytes?[NSJSONSerialization JSONObjectWithData:bytes options:0 error:&error]:nil;
 require([result isKindOfClass:NSDictionary.class],"invalid ActQuant JSON");return result;
}
static void actquantVerifyFixture(const fs::path&fixture,NSDictionary*manifest){
 NSArray*cases=manifest[@"cases"];require([cases isKindOfClass:NSArray.class]&&cases.count>0&&cases.count<=128,"invalid ActQuant case count");
 std::set<std::string>names;
 for(NSDictionary*item in cases){
  require([item isKindOfClass:NSDictionary.class]&&names.insert(str(item[@"name"])).second,"duplicate/invalid ActQuant case");
  for(NSString*role in @[@"input",@"expected",@"scale",@"int8_codes"]){
   const std::string relative=str(item[role]);const fs::path part(relative);
   require(!relative.empty()&&!part.is_absolute()&&relative.find('\\')==std::string::npos,"invalid ActQuant fixture path");
   for(const auto&component:part)require(component!="."&&component!="..","invalid ActQuant fixture traversal");
   const auto file=fs::canonical(fixture/part);const auto rel=file.lexically_relative(fixture);
   require(!rel.empty()&&*rel.begin()!=".."&&fs::is_regular_file(file),"ActQuant fixture escapes directory");
   require([hashFile(file)isEqual:item[[role stringByAppendingString:@"_sha256"]]],"ActQuant fixture file checksum mismatch");
  }
 }
}
static int run(){
 const char* raw=std::getenv("BITNET_CANDIDATE_CONFIG");require(raw,"missing candidate config");NSError*error=nil;NSData*data=[@(raw) dataUsingEncoding:NSUTF8StringEncoding];NSDictionary*c=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];require([c isKindOfClass:NSDictionary.class],"invalid JSON config");
 const fs::path docs=[[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask].firstObject path].UTF8String;
 const std::string prefix=str(c[@"output"]);require(prefix.rfind("candidate-",0)==0&&prefix.find('/')==std::string::npos,"invalid output prefix");Options o;o.out=docs/prefix;require(!fs::exists(o.out),"refuse existing output");fs::create_directories(o.out);json(o.out/"request.json",c);
 const std::string model=str(c[@"model"]);require(model.find('/')==std::string::npos,"invalid model basename");o.model=docs/model;NSString*modelSHA=hashFile(o.model);require([modelSHA isEqual:c[@"model_sha256"]],"GGUF checksum mismatch");std::fprintf(stderr,"BITNET_PHONE_MODEL_SHA256 %s\n",modelSHA.UTF8String);fflush(stderr);

 if([c[@"operation"]isEqual:@"actquant"]){
  o.model=actquantDocumentPath(docs,c[@"model"],false);
  const std::string kind=str(c[@"actquant_operation"]);require(kind=="operator"||kind=="model","invalid ActQuant operation");
  require([c[@"backend"]isEqual:@"MTL0"]&&[c[@"kv"]isEqual:@"F16"]&&[c[@"prefill"]isEqual:@"batched"],"ActQuant phone requires MTL0/F16/batched");
  require([c[@"workload"]isEqual:@"teacher"]||[c[@"workload"]isEqual:@"corpus"],"invalid ActQuant workload");
  NSMutableDictionary*native=[@{@"operation":@(kind.c_str()),@"backend":@"MTL0",@"model":@(o.model.c_str()),@"kv":@"F16",@"prefill":@"batched",@"workload":c[@"workload"],@"actquant":@(flag(c,@"actquant")),@"capture":@(flag(c,@"capture")),@"trace":@(flag(c,@"trace"))}mutableCopy];
  NSMutableDictionary*identity=[NSMutableDictionary dictionary];
  if(kind=="operator"){
   const auto fixture=actquantDocumentPath(docs,c[@"fixture"],true);const auto manifest=fixture/"FIXTURES.json";
   NSString*digest=hashFile(manifest);require([digest isEqual:c[@"fixture_sha256"]],"ActQuant fixture manifest checksum mismatch");
   actquantVerifyFixture(fixture,actquantReadJSON(manifest));native[@"fixture"]=@(fixture.c_str());identity[@"fixture_sha256"]=digest;
  }else{
   const auto tokens=actquantDocumentPath(docs,c[@"tokens"],false);NSString*digest=hashFile(tokens);
   require([digest isEqual:c[@"tokens_sha256"]],"ActQuant token checksum mismatch");
   native[@"tokens"]=@(tokens.c_str());identity[@"tokens_sha256"]=digest;
   const std::string qr=c[@"q_route"]?str(c[@"q_route"]):"stock";
   const std::string qf=c[@"q_fallback"]?str(c[@"q_fallback"]):"none";
   const std::string qs=c[@"capture_scope"]?str(c[@"capture_scope"]):"all";
   require(qr=="stock"||qr=="Q4","invalid ActQuant Q route");
   require(qf=="none"||qf=="missing-library","invalid ActQuant Q fallback");
   require(qs=="all"||qs=="q4-layer0","invalid ActQuant capture scope");
   require(qr!="Q4"||flag(c,@"actquant"),"Q4 requires ActQuant ON");
   native[@"q_route"]=@(qr.c_str());native[@"q_fallback"]=@(qf.c_str());native[@"capture_scope"]=@(qs.c_str());
   if(c[@"q_library"]){
    const auto library=actquantDocumentPath(docs,c[@"q_library"],false);NSString*librarySHA=hashFile(library);
    require([librarySHA isEqual:c[@"q_library_sha256"]]&&[c[@"q_variant"]isEqual:@"stock4x1"],"Q4 library variant/hash mismatch");
    native[@"q_library"]=@(library.c_str());identity[@"q_library_sha256"]=librarySHA;
    for(NSString*key in @[@"q_library_sha256",@"q_variant",@"q_library_source_sha256",@"q_library_manifest_sha256",@"q_library_sdk"]){
     require([c[key]isKindOfClass:NSString.class],"missing Q4 library provenance");native[key]=c[key];
    }
   }
   require(qr!="Q4"||native[@"q_library"]!=nil,"Q4 library missing");
  }
  json(o.out/"native-config.json",native);
  NSData*encoded=[NSJSONSerialization dataWithJSONObject:native options:NSJSONWritingSortedKeys error:nil];require(encoded!=nil,"ActQuant config serialization failed");
  const int code=bitnet_actquant_run_config([[[NSString alloc]initWithData:encoded encoding:NSUTF8StringEncoding]UTF8String],(o.out/"native").c_str());
  NSMutableDictionary*record=[@{@"schema":@1,@"operation":@"actquant",@"actquant_operation":@(kind.c_str()),@"native_process_exit_code":@(code),@"process_exit_code":@(code),@"execution_pass":@NO,@"model_sha256":modelSHA,@"backend":@"MTL0",@"actquant":@(flag(c,@"actquant")),@"capture":@(flag(c,@"capture")),@"workload":c[@"workload"],@"parity_pass":[NSNull null],@"fp64_pass":[NSNull null],@"model_output_pass":[NSNull null],@"semantic_acceptance_pass":[NSNull null],@"strict_model_exit_code":@2,@"performance_measurement":@NO,@"iOS":UIDevice.currentDevice.systemVersion,@"thermal_after":@(thermal().c_str())}mutableCopy];
  [record addEntriesFromDictionary:identity];
  if(fs::is_regular_file(o.out/"native/RUN.json")){
   NSDictionary*runtime=actquantReadJSON(o.out/"native/RUN.json",64*1024*1024);record[@"native_run_sha256"]=hashFile(o.out/"native/RUN.json");record[@"device"]=runtime[@"device"]?:[NSNull null];
   record[@"execution_pass"]=@(code==0&&[runtime[@"execution_pass"]boolValue]);
  }
  json(o.out/"result.json",record);NSData*completion=[NSJSONSerialization dataWithJSONObject:record options:NSJSONWritingSortedKeys error:nil];
  std::printf("BITNET_PHONE_COMPLETE %s\n",[[[NSString alloc]initWithData:completion encoding:NSUTF8StringEncoding]UTF8String]);fflush(stdout);return code;
 }

 if([c[@"operation"] isEqual:@"quality"]){
  const std::string fixture=str(c[@"quality_fixture"]);require(fixture.rfind("candidate-",0)==0&&fixture.find('/')==std::string::npos,"invalid quality fixture basename");
  require([hashFile(docs/fixture/"QUALITY.json") isEqual:c[@"quality_fixture_sha256"]],"quality fixture checksum mismatch");
  const int code=bitnet_run_iphone_ppl(o.model.c_str(),(docs/fixture).c_str(),(o.out/"quality").c_str());
  require(code==0,"quality measurement failed with exit"+std::to_string(code));
  NSData*d=[NSData dataWithContentsOfFile:@((o.out/"quality/summary.json").c_str())];require(d!=nil,"missing quality completion");
  NSMutableDictionary*record=[[NSJSONSerialization JSONObjectWithData:d options:NSJSONReadingMutableContainers error:nil]mutableCopy];require([record isKindOfClass:NSMutableDictionary.class]&&[record[@"execution_pass"]boolValue],"invalid quality completion");
  record[@"operation"]=@"quality";record[@"model_sha256"]=modelSHA;record[@"quality_fixture_sha256"]=c[@"quality_fixture_sha256"];record[@"process_exit_code"]=@(code);
  json(o.out/"result.json",record);NSData*encoded=[NSJSONSerialization dataWithJSONObject:record options:NSJSONWritingSortedKeys error:nil];std::printf("BITNET_PHONE_COMPLETE %s\n",[[[NSString alloc]initWithData:encoded encoding:NSUTF8StringEncoding]UTF8String]);fflush(stdout);return code;
 }

 if([c[@"operation"] isEqual:@"operator"]){
  NSDictionary*operatorLibraries=c[@"libraries"]?:@{};
  for(NSString*key in operatorLibraries){NSDictionary*v=operatorLibraries[key];std::string name=str(v[@"file"]);require(name.find('/')==std::string::npos,"bad operator library basename");require([hashFile(docs/name) isEqual:v[@"sha256"]],"operator library checksum mismatch "+name);}

  std::string fixture=str(c[@"operator_fixture"]),dest=str(c[@"operator_output"]),mode=str(c[@"operator_mode"]);
  require(!fixture.empty()&&fixture[0]!='/'&&fixture.find("..") == std::string::npos&&!dest.empty()&&dest[0]!='/'&&dest.find("..") == std::string::npos,"invalid operator container paths");
  require(mode=="reference"||mode=="verify"||mode=="benchmark","bad operator mode");
  const bool isQ=[c[@"operator_kind"]isEqual:@"Q"],isDown=[c[@"operator_kind"]isEqual:@"DOWN"],isPoc=[c[@"operator_kind"]isEqual:@"POC"];
  require(isQ||isDown||isPoc||[c[@"operator_kind"]isEqual:@"FFN"],"bad operator kind");
  const int code=(isPoc?bitnet_run_iphone_original_poc:isDown?bitnet_run_iphone_down:isQ?bitnet_run_iphone_q:bitnet_run_iphone_operator)((docs/fixture).c_str(),docs.c_str(),(docs/dest).c_str(),mode.c_str());
  NSDictionary*nativeDown=nil;
  if((isDown||isPoc)&&(code==0||code==1)){const auto path=docs/dest/"summary.json";NSData*d=[NSData dataWithContentsOfFile:@(path.c_str())];require(d!=nil,"missing completed down summary");nativeDown=[NSJSONSerialization JSONObjectWithData:d options:0 error:nil];require([nativeDown isKindOfClass:NSDictionary.class]&&[nativeDown[@"analysis_completed"]boolValue]&&[nativeDown[@"execution_pass"]boolValue]&&[nativeDown[@"strict_exit_code"]intValue]==code,"down completion/exit mismatch");if(isDown)require([nativeDown[@"fp64_pass"]boolValue]==(code==0),"down FP64/exit mismatch");else require([nativeDown[@"overall_pass"]boolValue]==(code==0),"original PoC overall/exit mismatch");}
  NSMutableDictionary*record=[@{@"device":MTLCreateSystemDefaultDevice().name?:@"unavailable",@"iOS":UIDevice.currentDevice.systemVersion,@"operation":@"operator",@"operator_mode":@(mode.c_str()),@"model_sha256":modelSHA,@"operator_fixture":@(fixture.c_str()),@"operator_output":@(dest.c_str()),@"required_checks_pass":@(code==0),@"process_exit_code":@(code),@"model_quality_tested":@NO} mutableCopy];if(isDown||isPoc){record[@"operator_kind"]=isPoc?@"POC":@"DOWN";record[@"execution_pass"]=@((code==0||code==1)&&nativeDown!=nil);record[@"fp64_pass"]=nativeDown?nativeDown[@"fp64_pass"]:(id)[NSNull null];record[@"parity_pass"]=nativeDown?nativeDown[@"parity_pass"]:(id)[NSNull null];record[@"strict_exit_code"]=@(code);record[@"completed_numeric_failure"]=@(code==1&&nativeDown!=nil);record[@"numerical_scope"]=isPoc?@"Five 2560-row cases, original authored 4x2 PoC, actual stock and Q4; standalone only":@"96 selected down outputs/path; not whole FFN or model";}json(o.out/"result.json",record);auto encoded=[NSJSONSerialization dataWithJSONObject:record options:NSJSONWritingSortedKeys error:nil];std::printf("BITNET_PHONE_COMPLETE %s\n",[[[NSString alloc]initWithData:encoded encoding:NSUTF8StringEncoding]UTF8String]);fflush(stdout);return code;
 }
 o.operation=str(c[@"operation"]);o.layers=str(c[@"layers"]);o.traced=flag(c,@"trace");o.k_shadow_all=flag(c,@"shadow_all");o.q_readback=flag(c,@"q_readback");o.warmups=c[@"warmups"]?[c[@"warmups"] intValue]:1;o.pairs=c[@"pairs"]?[c[@"pairs"] intValue]:21;o.rounds=c[@"rounds"]?[c[@"rounds"] intValue]:3;
 require(o.operation=="verify"||o.operation=="benchmark","bad operation");require(o.layers=="0"||o.layers=="0-3"||o.layers=="0-29","bad layer scope");require(o.warmups>=0&&o.warmups<=30&&o.pairs>0&&o.rounds>0,"invalid repetition counts");require(!(o.operation=="benchmark"&&(o.traced||flag(c,@"shadow")||o.q_readback)),"diagnostics forbidden during timing");
 for(id r in c[@"routes"])o.routes.push_back(str(r));require(!o.routes.empty()&&o.routes[0]=="S"&&std::set<std::string>(o.routes.begin(),o.routes.end()).size()==o.routes.size(),"unique routes must start with S");for(auto&r:o.routes)require(r=="S"||r=="G"||r=="K"||r=="F8"||r=="Q4"||r=="QB","unknown route");
 auto has=[&](const char*r){return std::find(o.routes.begin(),o.routes.end(),r)!=o.routes.end();};
 o.q_fallback=c[@"q_fallback"]?str(c[@"q_fallback"]):"";
 require(o.q_fallback.empty()||o.q_fallback=="missing-library"||o.q_fallback=="invalid-geometry","bad Q fallback mode");
 require(o.q_fallback.empty()||(o.operation=="verify"&&o.layers=="0"&&has("Q4")&&!o.q_readback&&std::all_of(o.routes.begin(),o.routes.end(),[](const std::string&r){return r=="S"||r=="Q4"||r=="QB";})),"Q fallback is a separate layer-0 correctness control");
 require(!o.q_readback||(o.operation=="verify"&&o.layers=="0"&&std::all_of(o.routes.begin(),o.routes.end(),[](const std::string&r){return r=="S"||r=="Q4"||r=="QB";})),"Q readback is restricted to layer-0 S/Q4/QB correctness runs");
 NSDictionary*libs=c[@"libraries"]?:@{};NSMutableDictionary*librarySHAs=[NSMutableDictionary dictionary];
 for(NSString*key in libs){NSDictionary*v=libs[key];std::string name=str(v[@"file"]);require(name.find('/')==std::string::npos,"bad library basename");fs::path path=docs/name;NSString*hash=hashFile(path);require([hash isEqual:v[@"sha256"]],"library checksum mismatch "+name);librarySHAs[key]=hash;if([key isEqual:@"mechanisms"])o.library=path;if([key isEqual:@"K"])o.k_library=path;if([key isEqual:@"F8"])o.f8_library=path;if([key isEqual:@"Q4"])o.q_library=path;}
 require((!has("Q4")&&!has("QB"))||(!o.q_library.empty()&&o.layers=="0"),"Q4 requires its own library and layer0-only scope");
 require((!has("G")&&!has("K"))||!o.library.empty(),"G/K mechanism library required");require(!has("K")||!o.k_library.empty(),"K library required");require(!has("F8")||!o.f8_library.empty(),"F8 library required");if(flag(c,@"shadow")){require(o.traced&&has("K"),"K shadows require traced K route");o.k_shadow=o.out/"shadow-K";}
 require(!o.q_readback||!o.library.empty(),"Q readback requires the diagnostic copy library");

 if(flag(c,@"missing_k_library")){require(has("K"),"missing K test needs K route");o.k_library=docs/(prefix+".intentionally-missing.metallib");require(!fs::exists(o.k_library),"missing K sentinel exists");}
 if(o.q_fallback=="missing-library"){o.q_library=docs/(prefix+".intentionally-missing-q.metallib");require(!fs::exists(o.q_library),"missing Q sentinel exists");}
 o.prompt=o.out/"prompt.txt";o.prompt_ids=o.out/"prompt.ids";o.continuation_ids=o.out/"continuation.ids";
 {std::ofstream f(o.prompt);f<<str(c[@"prompt"]);}
 for(auto pair:{std::make_pair(o.prompt_ids,@"prompt_ids"),std::make_pair(o.continuation_ids,@"continuation_ids")}){std::ofstream f(pair.first);for(NSNumber*n in c[pair.second])f<<n.intValue<<'\n';}
 Session s(o);for(int i=0;i<o.warmups;++i)for(auto&r:o.routes)s.replay(r,{},false,false,"warmup");std::vector<Replay>first,second;for(auto&r:o.routes)first.push_back(s.replay(r,o.out/("route-"+r),true,o.traced,"verify"));for(auto&r:o.routes)second.push_back(s.replay(r,o.out/("repeat-"+r),true,false,"repeat"));NSMutableDictionary*diffs=[NSMutableDictionary dictionary];bool repeat=true,sg=true;
 for(size_t i=0;i<o.routes.size();++i){auto d=difference(first[i],second[i]);repeat&=d.bits==0;auto stock=difference(first[0],first[i]);if(o.routes[i]=="G")sg&=stock.bits==0;if(o.routes[i]=="QB")require(stock.bits==0,"QB stock-relative complete logits must be bitwise identical");if(o.routes[i]=="Q4")require(stock.fail==0,"Q4 stock-relative complete logits failed frozen tolerance");diffs[@(o.routes[i].c_str())]=@{@"repeat_bits":@(d.bits),@"stock_bits":@(stock.bits),@"stock_tolerance_failed":@(stock.fail),@"stock_max_abs":@(stock.max_abs)};}
 auto device=MTLCreateSystemDefaultDevice();NSMutableDictionary*result=[@{@"completion_marker":@"BITNET_PHONE_COMPLETE",@"operation":@(o.operation.c_str()),@"layers":@(o.layers.c_str()),@"routes":c[@"routes"],@"model_sha256":modelSHA,@"library_sha256":librarySHAs,@"device":device.name?:@"unavailable",@"iOS":UIDevice.currentDevice.systemVersion,@"phone":UIDevice.currentDevice.model,@"repeatability_pass":@(repeat),@"S_G_parity_pass":has("G")?(id)@(sg):(id)[NSNull null],@"local_fp64_pass":[NSNull null],@"fp64_pass":[NSNull null],@"model_output_pass":[NSNull null],@"model_quality_tested":@NO,@"prompt_ids":c[@"prompt_ids"],@"continuation_ids":c[@"continuation_ids"],@"logit_vectors_per_path":@33,@"logit_comparisons":diffs,@"callbacks":@NO,@"thermal":@(thermal().c_str()),@"low_power":@(NSProcessInfo.processInfo.lowPowerModeEnabled),@"output":@(prefix.c_str())} mutableCopy];
 result[@"missing_k_library"]=@(flag(c,@"missing_k_library"));result[@"bound_K_library"]=@(o.k_library.c_str());
 json(o.out/"preflight.json",result);require(repeat&&sg,"within-route or S/G parity failed");
 result[@"q_readback_enabled"]=@(o.q_readback);
 result[@"q_fallback_mode"]=@(o.q_fallback.c_str());
 result[@"probe_logit_nonperturbation_pass"]=o.q_readback?(id)@(repeat):(id)[NSNull null];
 NSMutableDictionary*readbacks=[NSMutableDictionary dictionary];
 if(o.q_readback){for(auto&r:o.routes){const auto relative="route-"+r+"/q-readback/manifest.json";auto manifest=qReadbackManifest(o.out/("route-"+r)/"q-readback",r);readbacks[@(r.c_str())]=@{@"file":@(relative.c_str()),@"sha256":hashFile(o.out/relative),@"file_count":manifest[@"file_count"],@"coverage_pass":manifest[@"coverage_pass"],@"decode_steps":@32};}}
 result[@"q_readback_manifests"]=readbacks;
 result[@"q_readback_coverage_pass"]=o.q_readback?(id)@YES:(id)[NSNull null];
 json(o.out/"preflight.json",result);

 if(o.operation=="benchmark"){
  first.clear();first.shrink_to_fit();second.clear();second.shrink_to_fit();
  std::vector<std::string> comparisons;
  NSArray*requested=c[@"comparisons"]?:@[@"AA",@"SG",@"SK",@"SF8"];
  for(NSString*value in requested)comparisons.emplace_back(value.UTF8String);
  require(!comparisons.empty()&&o.pairs<=100&&o.rounds<=12,"invalid benchmark bounds");
  auto pairRoutes=[&](const std::string&x){
   if(x=="AA")return std::make_pair(std::string("S"),std::string("S"));
   if(x=="SQB")return std::make_pair(std::string("S"),std::string("QB"));
   if(x=="SQ4")return std::make_pair(std::string("S"),std::string("Q4"));
   if(x=="SG")return std::make_pair(std::string("S"),std::string("G"));
   if(x=="SK")return std::make_pair(std::string("S"),std::string("K"));
   if(x=="SF8")return std::make_pair(std::string("S"),std::string("F8"));
   if(x=="GK")return std::make_pair(std::string("G"),std::string("K"));
   if(x=="KF8")return std::make_pair(std::string("K"),std::string("F8"));
   throw std::runtime_error("invalid benchmark comparison");
  };
  for(auto&x:comparisons){auto rs=pairRoutes(x);require(has(rs.first.c_str())&&has(rs.second.c_str()),"comparison route absent from preflight");}
  for(int i=0;i<o.warmups;++i)for(auto&r:o.routes)s.replay(r,{},false,false,"rewarm");
  std::ofstream samples(o.out/"samples.csv"),tokens(o.out/"token-times.csv");samples<<std::setprecision(17);tokens<<std::setprecision(17);
  samples<<"round,block_id,campaign_order,comparison,pair,order,a_route,b_route,a_wall_ms,b_wall_ms,b_minus_a_ms,a_prefill_ms,b_prefill_ms,a_reset_ms,b_reset_ms,a_thermal_before,a_thermal_after,b_thermal_before,b_thermal_after,a_low_power,b_low_power,a_start_s,a_end_s,b_start_s,b_end_s\n";
  tokens<<"round,block_id,campaign_order,comparison,pair,order,path,route,step,wall_ms,start_s,end_s\n";
  int block=0;NSMutableArray*schedule=[NSMutableArray array];NSMutableArray*roundStates=[NSMutableArray array];
  for(int rr=1;rr<=o.rounds;++rr){NSDictionary*before=@{@"thermal":@(thermal().c_str()),@"low_power":@(NSProcessInfo.processInfo.lowPowerModeEnabled),@"battery_level":@(UIDevice.currentDevice.batteryLevel),@"battery_state":@((int)UIDevice.currentDevice.batteryState),@"uptime_s":@(NSProcessInfo.processInfo.systemUptime)};auto cs=comparisons;std::rotate(cs.begin(),cs.begin()+((rr-1)%cs.size()),cs.end());
   for(size_t position=0;position<cs.size();++position,++block){auto contrast=cs[position];auto routes=pairRoutes(contrast);std::string ar=routes.first,br=routes.second;[schedule addObject:@((std::to_string(rr)+":"+std::to_string(position+1)+":"+contrast).c_str())];
    for(int i=0;i<o.pairs;++i){bool ab=i%4==0||i%4==3;if(rr%2==0)ab=!ab;Replay a,b;if(ab){a=s.replay(ar,{},false,false,"timing");b=s.replay(br,{},false,false,"timing");}else{b=s.replay(br,{},false,false,"timing");a=s.replay(ar,{},false,false,"timing");}
     samples<<rr<<','<<block<<','<<position+1<<','<<contrast<<','<<i<<','<<(ab?"AB":"BA")<<','<<ar<<','<<br<<','<<a.wall<<','<<b.wall<<','<<b.wall-a.wall<<','<<a.prefill<<','<<b.prefill<<','<<a.reset<<','<<b.reset<<','<<a.before<<','<<a.after<<','<<b.before<<','<<b.after<<','<<(a.low_before||a.low_after)<<','<<(b.low_before||b.low_after)<<','<<a.begin_s<<','<<a.end_s<<','<<b.begin_s<<','<<b.end_s<<'\n';
     for(int j=0;j<32;++j){tokens<<rr<<','<<block<<','<<position+1<<','<<contrast<<','<<i<<','<<(ab?"AB":"BA")<<",A,"<<ar<<','<<j+1<<','<<a.token_ms[j]<<','<<a.token_start[j]<<','<<a.token_end[j]<<'\n';tokens<<rr<<','<<block<<','<<position+1<<','<<contrast<<','<<i<<','<<(ab?"AB":"BA")<<",B,"<<br<<','<<j+1<<','<<b.token_ms[j]<<','<<b.token_start[j]<<','<<b.token_end[j]<<'\n';}
    }
   }
   [roundStates addObject:@{@"round":@(rr),@"before":before,@"after":@{@"thermal":@(thermal().c_str()),@"low_power":@(NSProcessInfo.processInfo.lowPowerModeEnabled),@"battery_level":@(UIDevice.currentDevice.batteryLevel),@"battery_state":@((int)UIDevice.currentDevice.batteryState),@"uptime_s":@(NSProcessInfo.processInfo.systemUptime)}}];
  }
  result[@"round_states"]=roundStates;
  require(bool(samples)&&bool(tokens),"timing CSV write failed");samples.close();tokens.close();
  json(o.out/"execution.json",@{@"execution_success":@YES,@"timing_clean":@YES,@"one_reused_model_and_context":@YES,@"callbacks":@NO,@"trace_during_timing":@NO,@"CPU_timers_during_timing":@NO,@"tensor_readback_during_timing":@NO,@"reset_prefill_excluded":@YES,@"rounds":@(o.rounds),@"round_index":@1,@"comparisons_per_round":@(comparisons.size()),@"pairs_per_comparison_per_round":@(o.pairs),@"tokens_per_path":@32,@"prefill_tokens":@27,@"schedule":schedule,@"model_quality_tested":@NO});
  result[@"timing_clean"]=@YES;result[@"sample_pairs"]=@(o.rounds*o.pairs*comparisons.size());result[@"token_rows"]=@(64*o.rounds*o.pairs*comparisons.size());result[@"timing_metric"]=@"32-token synchronized host wall, reset/prefill excluded";
  result[@"q_readback_during_timing"]=@NO;
 }

 result[@"process_runtime_pass"]=@YES;result[@"process_exit_code"]=@0;json(o.out/"result.json",result);auto encoded=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingSortedKeys error:nil];std::printf("BITNET_PHONE_COMPLETE %s\n",[[[NSString alloc]initWithData:encoded encoding:NSUTF8StringEncoding] UTF8String]);fflush(stdout);return 0;
}
} // namespace
extern "C" int bitnet_candidate_run(void){@autoreleasepool{try{return bitnet_candidate::run();}catch(const std::exception&e){std::fprintf(stderr,"BITNET_PHONE_ERROR %s\n",e.what());fflush(stderr);return 1;}}}
