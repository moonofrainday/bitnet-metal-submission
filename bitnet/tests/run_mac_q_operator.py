#!/usr/bin/env python3
"""Prepare/build/run the existing stock/Q4 Mac operator, separate from model wall.

Default --plan does not compile or access Metal. --build compiles only; --run
explicitly compiles and executes the isolated GPU experiment after qualification.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from run_mac_q_value import ROOT, read, require, save, sha, validate_gate


def fixtures(gate,run):
    native=read(run/'RUN.json');require(sha(run/'RUN.json')==gate['histories']['primary']['verification_native_sha256'],
                                      'Use the qualified primary real-model verification')
    candidates=[r for r in native['paths'] if r['route']=='S' and r['capture_enabled'] is True]
    require(len(candidates)==1,'Qualified stock capture missing');folder=(run/candidates[0]['subrun']).resolve()
    require(folder.is_relative_to(run.resolve()),'Capture outside verification')
    data=read(folder/'RUN.json');cases=[]
    for step in (1,16,32):
        records=[r for r in data['checkpoints'] if r['kind']=='Q_preRoPE' and r['node']=='Qcur-0' and r['decode_step']==step]
        require(len(records)==1,'Missing actual Q step');record=records[0]
        require(record['ne']==[2560,1,1,1] and record['activation']['name']=='actq_qkv_out-0' and record['activation']['op']=='ACT_QUANT',
                'Use actual post-ActQuant F32 Q input')
        case={'step':step}
        for role,field,digest in [('weights','weight_path','weight_sha256'),('activation','input_path','input_sha256'),('stock_output','path','output_sha256')]:
            file=(folder/record[field]).resolve();require(file.is_relative_to(folder) and sha(file)==record[digest],'Actual captured payload hash mismatch')
            case[role]=str(file);case[role+'_sha256']=sha(file)
        cases.append(case)
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
    mode.add_argument('--plan',action='store_true');mode.add_argument('--build',action='store_true');mode.add_argument('--run',action='store_true')
    for name in ('gate','verification-run','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--developer-dir',type=Path);args=p.parse_args();args.output=args.output.expanduser().resolve()
    if args.output.exists():p.error('Output must be fresh')
    args.output.mkdir(parents=True);result={'scope':'Isolated M2 Pro selected stock/Q4 matvec','GPU_executed':False,
        'qualification_pass':None,'model_quality_pass':None,'official_source_model_output_pass':None,'historical_official_strict_exit_preserved':1,'errors':[]}
    try:
        gate=validate_gate(args.gate);cases=fixtures(gate,args.verification_run.resolve());config={
            'model_sha256':gate['model_sha256'],'q_library':gate['Q4']['q_library'],'q_library_sha256':gate['Q4']['q_library_sha256'],
            'entry':gate['Q4']['q_entry'],'rows':gate['Q4']['q_rows_per_simd'],'groups':gate['Q4']['q_simd_per_group'],
            'build_dir':gate['build_dir'],'cases':cases,'Mac_gate_sha256':sha(args.gate)}
        config_file=args.output/'CONFIG.json';save(config_file,config);build=Path(gate['build_dir']);binary=args.output/'mac-q-operator'
        source=ROOT/'bitnet/tests/mac_q_operator.mm';reference=ROOT/'bitnet/validation/reference.cpp'
        argv=['xcrun','--sdk','macosx','clang++','-arch','arm64','-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-fobjc-arc',
              '-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-I',str(ROOT/'ggml/include'),'-I',str(ROOT/'ggml/src'),
              '-I',str(ROOT/'ggml/src/ggml-metal'),str(source),str(reference),'-L',str(build/'bin'),'-lggml-metal','-lggml','-lggml-base',
              '-Wl,-rpath,'+str(build/'bin'),'-framework','Foundation','-framework','Metal','-o',str(binary)]
        execution=[str(binary),str(config_file),str(args.output/'run')]
        provenance={str(file.relative_to(ROOT)):sha(file) for file in [source,reference,ROOT/'bitnet/validation/reference.h',
                    ROOT/'bitnet/validation/contract.h',ROOT/'bitnet/validation/stock-runtime.h',ROOT/'ggml/src/ggml-metal/ggml-metal-bitnet-q-config.h']}
        save(args.output/'PLAN.json',{'build':argv,'execution':execution,'sources_sha256':provenance,'GPU_executed':False,
                                     'scope':'Single-dispatch GPU command-buffer elapsed; excludes guards/cache/ActQuant and model scheduling'})
        if not (args.build or args.run):result.update(plan_only=True,exit_code=0);return 0
        env=os.environ.copy();env['DYLD_LIBRARY_PATH']=str(build/'bin')
        if args.developer_dir:env['DEVELOPER_DIR']=str(args.developer_dir)
        for name in ['GGML_METAL_BITNET_Q_ATTRIBUTION','GGML_METAL_BITNET_POC_TRACE','GGML_METAL_ACTQUANT_TRACE','METAL_CAPTURE_ENABLED']:env.pop(name,None)
        with (args.output/'build.log').open('w') as log:code=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        result['build_exit_code']=code
        if code:result['exit_code']=code;return code
        result['binary_sha256']=sha(binary);result['sources_sha256']=provenance
        save(args.output/'BUILD.json',result)
        if not args.run:result['exit_code']=0;return 0
        validate_gate(args.gate)
        with (args.output/'run.log').open('w') as log:code=subprocess.run(execution,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        result.update(native_exit_code=code,GPU_executed=True if code==0 else None,exit_code=code)
        qualification=args.output/'run/qualification.json'
        if qualification.is_file():
            evidence=read(qualification);result['qualification_pass']=evidence['pass']
            if evidence.get('cases') and all(v.get('GPU_completed') is True for v in evidence['cases']):result['GPU_executed']=True
        return code
    except (OSError,ValueError,KeyError,TypeError) as error:result['errors'].append(str(error));result['exit_code']=2;return 2
    finally:save(args.output/'SUMMARY.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':raise SystemExit(main())
