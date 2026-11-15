#!/usr/bin/env python3
"""Compile and exercise the actual backend readback-route validator without Metal."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
source = (ROOT / 'ggml/src/ggml-metal/ggml-metal-ops.cpp').read_text()
start = source.index('static bool ggml_metal_bitnet_q_readback_active() {')
end = source.index('\nstatic void ggml_metal_bitnet_q_readback_copy', start)
function = source[start:end]
preamble = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <initializer_list>
#define GGML_ABORT(...) throw std::runtime_error("validator rejected configuration")
static bool ggml_metal_bitnet_poc_flag(const char *name) {
    const char *value=std::getenv(name);
    return value && std::strcmp(value,"1")==0;
}
'''
main = r'''
static int cases = 0;
static void check(const char *route, bool custom, bool guard, bool geometry, bool decode, const char *step, int expected) {
    for (const char *key : {"GGML_METAL_BITNET_FFN", "GGML_METAL_BITNET_FFN_MECHANISM", "GGML_METAL_BITNET_POC", "GGML_METAL_BITNET_Q_GUARD_ONLY"}) unsetenv(key);
    setenv("GGML_METAL_BITNET_QREADBACK_DIR", "/tmp/route-validation-only", 1);
    setenv("GGML_METAL_BITNET_QREADBACK_STEP", step, 1);
    if(route) setenv("GGML_METAL_BITNET_QREADBACK_ROUTE",route,1); else unsetenv("GGML_METAL_BITNET_QREADBACK_ROUTE");
    setenv("GGML_METAL_BITNET_FFN_DECODE",decode?"1":"0",1);
    if(custom) setenv("GGML_METAL_BITNET_POC","1",1);
    if(guard) setenv("GGML_METAL_BITNET_Q_GUARD_ONLY","1",1);
    setenv("GGML_METAL_BITNET_POC_ROWS_PER_SIMD",geometry?"4":"8",1);
    setenv("GGML_METAL_BITNET_POC_SIMD_PER_GROUP","1",1);
    int result;
    try { result=ggml_metal_bitnet_q_readback_active()?1:0; }
    catch(const std::runtime_error&) { result=-1; }
    ++cases;
    if(result!=expected) {
        std::fprintf(stderr,"FAIL case=%d route=%s custom=%d guard=%d geometry=%d decode=%d step=%s actual=%d expected=%d\n",cases,route?route:"missing",custom,guard,geometry,decode,step,result,expected);
        std::exit(1);
    }
}
int main() {
    check("S",false,false,true,true,"1",1);
    check("Q4",true,false,true,true,"1",1);
    check("QB",false,true,true,true,"1",1);
    check("QB",false,true,true,true,"32",1);
    check("QB",false,false,true,true,"1",-1);
    check("QB",true,true,true,true,"1",-1);
    check("S",false,true,true,true,"1",-1);
    check("Q4",true,true,true,true,"1",-1);
    check("Q4",false,false,true,true,"1",-1);
    check("QB",false,true,false,true,"1",-1);
    check("Q4",true,false,false,true,"1",-1);
    check("QB",false,true,true,false,"1",0);
    check("QB",false,true,true,true,"0",0);
    check("QB",false,true,true,true,"33",-1);
    check("UNKNOWN",false,false,true,true,"1",-1);
    check(nullptr,false,false,true,true,"1",-1);
    std::printf("PASS %d actual-backend route/control cases; CPU-only, no GPU execution\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='bitnet-q-readback-route-') as folder:
    cpp = Path(folder) / 'routes.cpp'
    binary = Path(folder) / 'routes'
    cpp.write_text(preamble + function + main)
    subprocess.run(['c++', '-std=c++17', '-O0', str(cpp), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
