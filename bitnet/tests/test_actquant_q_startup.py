#!/usr/bin/env python3
"""Exercise the actual model's Q4/QB startup predicate without Metal."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
source = (ROOT / 'src/models/bitnet.cpp').read_text()
start = source.index('static bool bitnet_actquant_q_route_contract(bool permission) {')
end = source.index('\nvoid llama_model_bitnet::load_arch_hparams', start)
function = source[start:end]
program = r'''
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
'''+function+r'''
int main() {
    const char * keys[] = {
        "GGML_METAL_BITNET_POC", "GGML_METAL_BITNET_Q_GUARD_ONLY",
        "GGML_METAL_BITNET_POC_VARIANT", "GGML_METAL_BITNET_POC_ROWS_PER_SIMD",
        "GGML_METAL_BITNET_POC_SIMD_PER_GROUP", "GGML_METAL_BITNET_LAYERS",
        "GGML_METAL_BITNET_FFN_ABLATION"
    };
    int cases = 0;
    auto check = [&](const char * custom, const char * guard, const char * variant,
                     const char * rows, const char * groups, const char * layers,
                     const char * ablation, bool permission, bool expected) {
        const char * values[] = {custom,guard,variant,rows,groups,layers,ablation};
        for (int i=0;i<7;++i) {
            if(values[i]) setenv(keys[i],values[i],1); else unsetenv(keys[i]);
        }
        ++cases;
        if(bitnet_actquant_q_route_contract(permission)!=expected) {
            std::fprintf(stderr,"FAIL actual startup predicate case=%d\n",cases);
            std::exit(1);
        }
    };
    check("1","0","stock4x1","4","1","0",nullptr,true,true);
    check("0","1","stock4x1","4","1","0",nullptr,true,true);
    check(nullptr,"1",nullptr,"4","1","0","0",true,true);
    check("1",nullptr,nullptr,"4","1","0",nullptr,true,true);
    check("1","1","stock4x1","4","1","0",nullptr,true,false);
    check("0","0","stock4x1","4","1","0",nullptr,true,false);
    check("0","1","stock4x1","4","1","0",nullptr,false,false);
    check("1","0","stock4x1","4","1","0",nullptr,false,false);
    check("0","1","c8x2","4","1","0",nullptr,true,false);
    check("0","1","stock4x1","4","2","0",nullptr,true,false);
    check("0","1","stock4x1","4","4","0",nullptr,true,false);
    check("0","1","stock4x1","8","2","0",nullptr,true,false);
    check("0","1","stock4x1","4","1","0-29",nullptr,true,false);
    check("0","1","stock4x1","4","1","0","1",true,false);
    check("0","1","stock4x1",nullptr,"1","0",nullptr,true,false);
    check("0","1","stock4x1","4",nullptr,"0",nullptr,true,false);
    check("0","1","stock4x1","4","1",nullptr,nullptr,true,false);
    check("0","2","stock4x1","4","1","0",nullptr,true,false);
    check("2","1","stock4x1","4","1","0",nullptr,true,false);
    check("0","true","stock4x1","4","1","0",nullptr,true,false);
    check("0","1","","4","1","0",nullptr,true,false);
    std::printf("PASS %d actual ActQuant-ON startup Q4/QB predicate cases; CPU-only\n",cases);
}
'''
with tempfile.TemporaryDirectory(prefix='bitnet-q-startup-') as folder:
    cpp = Path(folder) / 'startup.cpp'
    binary = Path(folder) / 'startup'
    cpp.write_text(program)
    subprocess.run(['c++', '-std=c++17', '-O0', '-Wall', '-Wextra', '-Werror', str(cpp), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
