#!/usr/bin/env python3
"""CPU-only controls of the production ActQuant Q-route predicate.

The production functions are extracted verbatim and compiled with minimal tensor
mocks. This checks selector/producer policy, not GGML allocation or GPU execution.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ggml/src/ggml-metal/ggml-metal-ops.cpp"


def function(source: str, name: str) -> str:
    match = re.search(r"^static bool " + re.escape(name) + r"\s*\(", source, re.M)
    if match is None:
        raise ValueError(f"Production function not found: {name}")
    start = source.index("{", match.end())
    depth = 1
    end = start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]


class BackendActQuantQContract(unittest.TestCase):
    def test_actual_predicate_controls(self) -> None:
        compiler = shutil.which(os.environ.get("CXX", "clang++"))
        self.assertIsNotNone(compiler, "A local C++ compiler is required")
        source = SOURCE.read_text()
        functions = "\n".join(function(source, name) for name in (
            "ggml_metal_bitnet_weight_name", "ggml_metal_bitnet_poc_flag",
            "ggml_metal_bitnet_actquant_q4_contract"))
        prefix = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <initializer_list>
enum ggml_type { GGML_TYPE_F32, GGML_TYPE_F16 };
enum ggml_op { GGML_OP_NONE, GGML_OP_MUL_MAT, GGML_OP_ACT_QUANT };
struct ggml_tensor {
    char name[96] = {};
    ggml_type type = GGML_TYPE_F32;
    ggml_op op = GGML_OP_NONE;
    ggml_tensor * src[2] = {};
    int64_t ne[4] = {2560,1,1,1};
    unsigned char op_params[16] = {};
    bool contiguous = true;
};
static bool ggml_is_contiguous(const ggml_tensor * t) { return t->contiguous; }
static bool ggml_are_same_shape(const ggml_tensor * a, const ggml_tensor * b) {
    return std::memcmp(a->ne,b->ne,sizeof(a->ne))==0;
}
'''
        controls = r'''
static int total=0, failures=0;
static void check(const char * name, bool actual, bool expected) {
    ++total;
    if (actual!=expected) { ++failures; std::fprintf(stderr,"FAIL %s\n",name); }
}
static void env(const char * name, const char * value) {
    if (value) { setenv(name,value,1); } else { unsetenv(name); }
}
int main() {
    ggml_tensor original, quantized, weight, op;
    const char * const flags[] = {"LLAMA_BITNET_ACTQUANT","LLAMA_BITNET_ACTQUANT_Q4",
        "GGML_METAL_BITNET_POC","GGML_METAL_BITNET_Q_GUARD_ONLY","GGML_METAL_BITNET_LAYERS",
        "GGML_METAL_BITNET_FFN","GGML_METAL_BITNET_FFN_MECHANISM","GGML_METAL_BITNET_FFN_ABLATION"};
    auto reset = [&]() {
        for (auto name:flags) { unsetenv(name); }
        original=ggml_tensor{}; quantized=ggml_tensor{}; weight=ggml_tensor{}; op=ggml_tensor{};
        std::strcpy(weight.name,"blk.0.attn_q.weight"); std::strcpy(op.name,"Qcur-0");
        std::strcpy(quantized.name,"actq_qkv_out-0");
        op.op=GGML_OP_MUL_MAT; op.src[0]=&weight; op.src[1]=&quantized;
        quantized.op=GGML_OP_ACT_QUANT; quantized.src[0]=&original;
        const float eps=1e-5f; std::memcpy(quantized.op_params,&eps,sizeof(eps));
        env("LLAMA_BITNET_ACTQUANT","1"); env("LLAMA_BITNET_ACTQUANT_Q4","1");
        env("GGML_METAL_BITNET_POC","1"); env("GGML_METAL_BITNET_LAYERS","0");
    };
    auto run = [&](bool guard=false,int rows=4,int groups=1) {
        return ggml_metal_bitnet_actquant_q4_contract(&op,rows,groups,guard,false);
    };
    reset(); check("valid Q4",run(),true);
    reset(); env("GGML_METAL_BITNET_POC","0"); env("GGML_METAL_BITNET_Q_GUARD_ONLY","1");
    check("valid QB",run(true),true);
    check("QB argument mismatch",run(false),false);
    reset(); check("Q4 argument mismatch",run(true),false);
    reset(); env("GGML_METAL_BITNET_Q_GUARD_ONLY","1"); check("both routes",run(true),false);
    reset(); env("GGML_METAL_BITNET_POC","0"); check("no route",run(),false);
    reset(); env("LLAMA_BITNET_ACTQUANT_Q4",nullptr); check("missing permission",run(),false);
    reset(); env("LLAMA_BITNET_ACTQUANT","0"); check("permission without ActQuant",run(),false);
    reset(); env("LLAMA_BITNET_ACTQUANT","0"); env("LLAMA_BITNET_ACTQUANT_Q4","0");
    check("legacy scope unchanged",run(),true);
    for (auto bad:{"","2","true","-1"}) {
        reset(); env("GGML_METAL_BITNET_Q_GUARD_ONLY",bad); check("malformed QB selector",run(),false);
        reset(); env("GGML_METAL_BITNET_POC",bad); env("GGML_METAL_BITNET_Q_GUARD_ONLY","1");
        check("malformed custom selector",run(true),false);
    }
    for (auto layers:{"","1","0-3","0-29"}) {
        reset(); env("GGML_METAL_BITNET_LAYERS",layers); check("wrong layers",run(),false);
    }
    reset(); env("GGML_METAL_BITNET_LAYERS",nullptr); check("missing layers",run(),false);
    reset(); check("4x2 reject",run(false,4,2),false); check("4x4 reject",run(false,4,4),false);
    check("8x2 reject",run(false,8,2),false);
    reset(); std::strcpy(op.name,"Qcur-1"); check("wrong node",run(),false);
    reset(); std::strcpy(weight.name,"blk.1.attn_q.weight"); check("wrong weight",run(),false);
    reset(); op.op=GGML_OP_NONE; check("wrong operation",run(),false);
    reset(); op.src[1]=nullptr; check("missing input",run(),false);
    reset(); quantized.op=GGML_OP_NONE; check("missing quantizer",run(),false);
    reset(); std::strcpy(quantized.name,"actq_qkv_out-1"); check("wrong producer name",run(),false);
    reset(); quantized.type=GGML_TYPE_F16; check("wrong activation type",run(),false);
    reset(); quantized.src[0]=nullptr; check("missing quantizer input",run(),false);
    reset(); original.type=GGML_TYPE_F16; check("wrong producer input type",run(),false);
    reset(); original.contiguous=false; check("noncontiguous producer input",run(),false);
    reset(); original.ne[0]=1280; check("wrong producer input shape",run(),false);
    reset(); const float bad_eps=1e-4f; std::memcpy(quantized.op_params,&bad_eps,sizeof(bad_eps));
    check("wrong eps",run(),false);
    for (auto name:{"GGML_METAL_BITNET_FFN","GGML_METAL_BITNET_FFN_MECHANISM","GGML_METAL_BITNET_FFN_ABLATION"}) {
        for (auto active:{"1","C","2"}) { reset(); env(name,active); check("FFN route forbidden",run(),false); }
        reset(); env(name,"0"); check("inactive FFN selector",run(),true);
    }
    std::printf("actual_backend_predicate_cases=%d failures=%d GPU_executed=false\n",total,failures);
    return failures?1:0;
}
'''
        with tempfile.TemporaryDirectory(prefix="bitnet-actquant-q-backend-") as tmp:
            cpp = Path(tmp) / "test.cpp"
            binary = Path(tmp) / "test"
            cpp.write_text(prefix + functions + controls)
            build = subprocess.run([str(compiler), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                                    str(cpp), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
