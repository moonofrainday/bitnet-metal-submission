#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <mach-o/dyld.h>
#include "../validation/stock-runtime.h"
#include "../validation/reference.h"
#include "../../ggml/src/ggml-metal/ggml-metal-bitnet-q-config.h"
#include "../../vendor/nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs=std::filesystem;
using json=nlohmann::ordered_json;
constexpr size_t pad=256, weight_bytes=1689600, vector_bytes=10240;
static void require(bool value,const std::string & message) { if(!value)throw std::runtime_error(message); }
static std::vector<uint8_t> read(const fs::path & path) {
    require(fs::is_regular_file(path)&&fs::file_size(path)<=256*1024*1024,"input extent invalid");
    std::ifstream file(path,std::ios::binary);require(bool(file),"input missing");return {std::istreambuf_iterator<char>(file),{}};
}
static std::string hash(const fs::path & path) {
    auto data=read(path);unsigned char bytes[CC_SHA256_DIGEST_LENGTH];CC_SHA256(data.data(),CC_LONG(data.size()),bytes);
    static const char hex[]="0123456789abcdef";std::string result;
    for(unsigned char byte:bytes){result+=hex[byte>>4];result+=hex[byte&15];}return result;
}
static void write(const fs::path & path,const json & value) { std::ofstream file(path);file<<value.dump(2)<<'\n';require(bool(file),"JSON output failed"); }
static json load(const fs::path & path) { std::ifstream file(path);require(bool(file),"JSON missing");json value;file>>value;return value; }
static json state() {
    static const char * names[]={"nominal","fair","serious","critical"};const auto value=NSProcessInfo.processInfo.thermalState;
    return {{"thermal",value>=0&&value<4?names[value]:"unknown"},{"low_power",bool(NSProcessInfo.processInfo.lowPowerModeEnabled)}};
}
struct Buffer {
    id<MTLBuffer> value=nil;std::vector<uint8_t> original;size_t bytes=0;
    Buffer(id<MTLDevice> device,const uint8_t * data,size_t count,bool output=false):bytes(count) {
        original.assign(count+2*pad,0xa7);
        if(data)std::memcpy(original.data()+pad,data,count);
        if(output)for(size_t i=0;i<count/sizeof(float);++i){const float value=std::numeric_limits<float>::quiet_NaN();std::memcpy(original.data()+pad+i*4,&value,4);}
        value=[device newBufferWithBytes:original.data() length:original.size() options:MTLResourceStorageModeShared];require(value!=nil,"buffer allocation failed");
    }
    void check(bool readonly) const {
        const auto * p=static_cast<const uint8_t *>(value.contents);
        require(!std::memcmp(p,original.data(),pad)&&!std::memcmp(p+pad+bytes,original.data()+pad+bytes,pad),"prefix/suffix canary changed");
        if(readonly)require(!std::memcmp(p+pad,original.data()+pad,bytes),"input buffer changed");
    }
    const float * floats() const { return reinterpret_cast<const float *>(static_cast<const uint8_t *>(value.contents)+pad); }
};
struct Engine {
    StockRuntime stock;id<MTLComputePipelineState> q4=nil;
    explicit Engine(const fs::path & library) {
        require([stock.device().name isEqual:@"Apple M2 Pro"],"This operator experiment requires M2 Pro");
        require(std::string(ggml_metal_pipeline_name(stock.pipeline()))=="kernel_mul_mv_tq2_0_f32","Unexpected actual stock pipeline");
        NSError * error=nil;auto lib=[stock.device() newLibraryWithURL:[NSURL fileURLWithPath:@(library.c_str())] error:&error];
        require(lib!=nil,"Q4 library load failed");auto function=[lib newFunctionWithName:@GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_ENTRY)];
        require(function!=nil,"Unique Q4 function missing");q4=[stock.device() newComputePipelineStateWithFunction:function error:&error];
        require(q4&&q4.threadExecutionWidth==GGML_BITNET_Q_SIMD_WIDTH&&q4.maxTotalThreadsPerThreadgroup>=GGML_BITNET_Q_SIMD_WIDTH*GGML_BITNET_Q4_NSG,"Q4 pipeline resources invalid");
    }
    double dispatch(bool custom,const Buffer & w,const Buffer & x,const Buffer & y) {
        @autoreleasepool {
            auto command=[stock.queue() commandBuffer];command.label=custom?@"Mac-Qcur0-Q4-single-dispatch":@"Mac-Qcur0-stock-single-dispatch";
            if(!custom)stock.encode(command,w.value,x.value,y.value,pad,pad,pad);
            else {
                auto encoder=[command computeCommandEncoder];encoder.label=@"Qcur0-stock4x1-v1";
                [encoder setComputePipelineState:q4];[encoder setBuffer:w.value offset:pad atIndex:0];
                [encoder setBuffer:x.value offset:pad atIndex:1];[encoder setBuffer:y.value offset:pad atIndex:2];
                [encoder dispatchThreadgroups:MTLSizeMake(GGML_BITNET_Q_NROWS/(GGML_BITNET_Q4_ROWS*GGML_BITNET_Q4_NSG),1,1)
                           threadsPerThreadgroup:MTLSizeMake(GGML_BITNET_Q_SIMD_WIDTH,GGML_BITNET_Q4_NSG,1)];
                [encoder endEncoding];
            }
            [command commit];[command waitUntilCompleted];require(command.status==MTLCommandBufferStatusCompleted&&!command.error,"GPU dispatch failed");
            const double us=(command.GPUEndTime-command.GPUStartTime)*1e6;require(std::isfinite(us)&&us>0,"GPU elapsed unavailable");return us;
        }
    }
};
static json loaded(const fs::path & build) {
    const auto expected=fs::canonical(build/"bin");json result=json::array();bool base=false,metal=false;
    for(uint32_t i=0;i<_dyld_image_count();++i){const char * name=_dyld_get_image_name(i);if(!name)continue;
        const fs::path file(name);if(file.filename().string().rfind("libggml",0)!=0)continue;
        const auto real=fs::canonical(file);require(real.parent_path()==expected,"Non-frozen GGML dylib loaded");
        base|=real.filename().string().rfind("libggml-base",0)==0;metal|=real.filename().string().rfind("libggml-metal",0)==0;
        result.push_back({{"path",real.string()},{"sha256",hash(real)}});
    }
    require(base&&metal,"Actual GGML base/Metal images missing");return result;
}
static double quantile(std::vector<double> data,double fraction) {
    std::sort(data.begin(),data.end());const double at=(data.size()-1)*fraction;const size_t lo=size_t(at),hi=size_t(std::ceil(at));
    return data[lo]+(data[hi]-data[lo])*(at-lo);
}
static int run(const fs::path & config_file,const fs::path & out) {
    require(!fs::exists(out),"Refuse existing output");fs::create_directories(out);auto config=load(config_file);write(out/"REQUEST.json",config);
    const fs::path library=config.at("q_library").get<std::string>();require(hash(library)==config.at("q_library_sha256"),"Q4 library hash differs");
    require(config.at("entry")==GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_ENTRY)&&config.at("rows")==GGML_BITNET_Q4_ROWS&&
        config.at("groups")==GGML_BITNET_Q4_NSG,"Q4 canonical configuration differs");
    Engine engine(library);const auto images=loaded(config.at("build_dir").get<std::string>());
    require(config.at("cases").is_array()&&config.at("cases").size()==3,"Exactly three real cases required");
    for(size_t i=0;i<3;++i){const int expected[]={1,16,32};require(config.at("cases")[i].at("step").is_number_integer()&&config.at("cases")[i].at("step")==expected[i],"Real step mapping differs");}
    std::ofstream rows(out/"correctness.csv");rows<<std::setprecision(17)<<"step,route,row,reference,actual,absolute_error,tolerance,failed,stock_bitwise_different\n";
    json cases=json::array();bool pass=true;
    for(const auto & item:config.at("cases")){
        const int step=item.at("step");const auto w=read(item.at("weights").get<std::string>()),x=read(item.at("activation").get<std::string>()),s=read(item.at("stock_output").get<std::string>());
        require(w.size()==weight_bytes&&x.size()==vector_bytes&&s.size()==vector_bytes,"Real fixture extent differs");
        for(const char * role:{"weights","activation","stock_output"})require(hash(item.at(role).get<std::string>())==item.at(std::string(role)+"_sha256"),"Real fixture hash differs");
        Input input;input.weights=w;std::memcpy(input.activation.data(),x.data(),vector_bytes);
        for(float value:input.activation)require(std::isfinite(value),"Nonfinite X");
        for(size_t b=0;b<w.size();b+=66){for(int j=0;j<64;++j)for(int p=0;p<4;++p)require(((w[b+j]>>(2*p))&3)!=3,"Illegal trit code");uint16_t bits=w[b+64]|(uint16_t(w[b+65])<<8);_Float16 scale;std::memcpy(&scale,&bits,2);require(std::isfinite(float(scale)),"Nonfinite scale");}
        const auto reference=reference_matvec(input);Buffer wb(engine.stock.device(),w.data(),w.size()),xb(engine.stock.device(),x.data(),x.size());
        Buffer sy(engine.stock.device(),nullptr,vector_bytes,true),qy(engine.stock.device(),nullptr,vector_bytes,true);
        engine.dispatch(false,wb,xb,sy);engine.dispatch(true,wb,xb,qy);wb.check(true);xb.check(true);sy.check(false);qy.check(false);
        for(bool custom:{false,true}){
            const float * values=custom?qy.floats():sy.floats();size_t fails=0,bits=0,nonfinite=0;double maximum=0,error2=0,ref2=0;
            for(int row=0;row<TQ2_ROWS;++row){const double error=std::abs(double(values[row])-reference[row]),tolerance=2e-4+2e-5*std::abs(reference[row]);
                const bool finite=std::isfinite(values[row])&&std::isfinite(reference[row]);const bool bad=!finite||error>tolerance;
                const bool different=std::memcmp(values+row,s.data()+row*4,4)!=0;fails+=bad;bits+=different;nonfinite+=!finite;maximum=std::max(maximum,error);error2+=error*error;ref2+=reference[row]*reference[row];
                rows<<step<<','<<(custom?"Q4":"S")<<','<<row<<','<<reference[row]<<','<<values[row]<<','<<error<<','<<tolerance<<','<<bad<<','<<different<<'\n';}
            pass&=fails==0&&bits==0&&nonfinite==0;cases.push_back({{"step",step},{"route",custom?"Q4":"S"},{"rows",2560},{"FP64_failures",fails},
                {"stock_bitwise_differences",bits},{"nonfinite",nonfinite},{"max_abs",maximum},{"relative_L2",std::sqrt(error2/ref2)},
                {"canaries_unchanged",true},{"inputs_unchanged",true},{"GPU_completed",true}});
        }
    }
    rows.close();write(out/"qualification.json",{{"pass",pass},{"cases",cases},{"loaded_images",images},{"config_sha256",hash(config_file)}});
    if(!pass){std::fprintf(stderr,"MAC_Q_OPERATOR_NUMERICAL_FAILURE\n");return 1;}
    const auto item=*std::find_if(config.at("cases").begin(),config.at("cases").end(),[](const json & value){return value.at("step")==16;});
    const auto w=read(item.at("weights").get<std::string>()),x=read(item.at("activation").get<std::string>());
    Buffer wb(engine.stock.device(),w.data(),w.size()),xb(engine.stock.device(),x.data(),x.size()),ya(engine.stock.device(),nullptr,vector_bytes,true),yb(engine.stock.device(),nullptr,vector_bytes,true);
    std::ofstream csv(out/"samples.csv");csv<<std::setprecision(17)<<"round,contrast,pair,order,a_route,b_route,a_us,b_us,b_minus_a_us\n";json groups=json::array();
    for(int round=1;round<=3;++round){std::vector<std::string> contrasts=round%2?std::vector<std::string>{"AA","SQ4"}:std::vector<std::string>{"SQ4","AA"};
        for(const auto & contrast:contrasts){const bool candidate=contrast=="SQ4";const auto before=state();
            for(int i=0;i<30;++i){engine.dispatch(false,wb,xb,ya);engine.dispatch(candidate,wb,xb,yb);}
            std::vector<double> a,b,difference;int wins=0;
            for(int pair=0;pair<201;++pair){const bool ab=((pair%4==0||pair%4==3)!=(round%2==0));double av,bv;
                if(ab){av=engine.dispatch(false,wb,xb,ya);bv=engine.dispatch(candidate,wb,xb,yb);}else{bv=engine.dispatch(candidate,wb,xb,yb);av=engine.dispatch(false,wb,xb,ya);}
                a.push_back(av);b.push_back(bv);difference.push_back(bv-av);wins+=bv<av;csv<<round<<','<<contrast<<','<<pair<<','<<(ab?"AB":"BA")<<",S,"<<(candidate?"Q4":"S")<<','<<av<<','<<bv<<','<<bv-av<<'\n';}
            groups.push_back({{"round",round},{"contrast",contrast},{"pairs",201},{"a_p50_us",quantile(a,.5)},{"a_p95_us",quantile(a,.95)},
                {"b_p50_us",quantile(b,.5)},{"b_p95_us",quantile(b,.95)},{"paired_B_minus_A_median_us",quantile(difference,.5)},{"B_wins",wins},{"before",before},{"after",state()}});
        }
    }
    wb.check(true);xb.check(true);ya.check(false);yb.check(false);csv.close();require(bool(csv),"Timing CSV failed");
    write(out/"SUMMARY.json",{{"execution_pass",true},{"qualification_pass",true},{"device",engine.stock.device().name.UTF8String},{"groups",groups},
        {"stock_entry",ggml_metal_pipeline_name(engine.stock.pipeline())},{"Q4_entry",GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_ENTRY)},
        {"stock_grid",{320,1,1}},{"stock_threads",{32,2,1}},{"Q4_grid",{640,1,1}},{"Q4_threads",{32,1,1}},
        {"source_config_sha256",hash(config_file)},{"warmups_per_path",30},{"rounds",3},{"pairs_per_contrast_round",201},
        {"single_dispatch_per_command_buffer",true},{"capture_diagnostics_readback_during_timing",false},
        {"metric","(GPUEndTime-GPUStartTime)*1e6 after completion; GPU command-buffer elapsed microseconds"},
        {"scope","Isolated actual step16 Qcur0 W/X; excludes guard/cache checks, ActQuant preparation and full-model scheduling"},
        {"model_quality_pass",nullptr},{"official_source_model_output_pass",nullptr},{"historical_official_strict_exit_preserved",1}});
    std::printf("MAC_Q_OPERATOR_COMPLETE pairs=1206 dispatches=2412 qualification=1\n");return 0;
}
}
int main(int argc,char ** argv){@autoreleasepool{try{require(argc==3,"usage: mac-q-operator CONFIG NEW_OUTPUT");return run(argv[1],argv[2]);}
    catch(const std::exception & error){std::fprintf(stderr,"MAC_Q_OPERATOR_ERROR %s\n",error.what());return 2;}}}
