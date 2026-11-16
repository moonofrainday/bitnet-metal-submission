#!/usr/bin/env python3
"""Mac-only value experiment for the existing Qcur-0 stock4x1 candidate.

Default --plan validates frozen Mac evidence and writes configs/commands only.
Explicit --run invokes the existing native runner serially. This is not an A16
gate, a new kernel selection campaign, or trusted-reference model acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'bitnet/actquant'))
from bitnet.q_kernel_contract import load_contract, variant_contract
from run_phone import attribution_protocol, token_identity

MODEL_SHA='52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2'


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as file:
        for data in iter(lambda:file.read(8*1024*1024),b''):value.update(data)
    return value.hexdigest()


def relative(root,name):
    part=Path(name);require(not part.is_absolute() and '..' not in part.parts,'Unsafe provenance path')
    file=root/part;require(file.resolve().is_relative_to(root.resolve()) and file.is_file(),'Missing provenance file: '+name)
    return file


def build_gate(state):
    source=Path(state['source']).resolve();require(source==ROOT,'State source must be this checkout')
    prepared=Path(state['preparation_experiment']).resolve();old_source=Path(state['preparation_source']).resolve()
    build=Path(state['Mac_build']).resolve();shaders=Path(state['Mac_shaders']).resolve();model=Path(state['model']).resolve()
    manifest=read(build/'FRESH_CORE_BUILD.json')
    require(manifest['build_completed'] is True and manifest['old_dylib_reused'] is False and Path(manifest['source_root']).resolve()==old_source,
            'Expected the frozen clean prepared Mac build')
    needed={'bitnet/actquant/runner.cpp','bitnet/actquant/attribution.h','ggml/src/ggml-metal/ggml-metal-ops.cpp',
            'ggml/src/ggml-metal/ggml-metal-device.m','src/models/bitnet.cpp'}
    require(needed<=set(manifest['source_files_sha256']),'Incomplete native source identity')
    for name,digest in manifest['source_files_sha256'].items():
        require(sha(relative(source,name))==sha(relative(old_source,name))==digest,'Prepared/current native source differs: '+name)
    for name,digest in manifest['products_sha256'].items():require(sha(relative(build,name))==digest,'Frozen binary changed: '+name)
    runner=relative(build,'bin/bitnet-actquant-runner');shader=read(shaders/'BUILD.json');contract=variant_contract('stock4x1')
    require(shader['build_completed'] is True and shader['exit_code']==0 and shader['sdk']=='macosx' and
            shader['q_kernel_contract']==load_contract(),'Wrong Mac shader build/config')
    product=shader['products']['stock4x1.metallib'];library=relative(shaders,'stock4x1.metallib')
    require(product['q_contract']==contract and product['expected_symbols']==[contract['entry']] and
            product['bytes']==library.stat().st_size and product['sha256']==sha(library),'Wrong Q4 product')
    for name,digest in shader['sources'].items():require(sha(relative(source,name))==sha(relative(old_source,name))==digest,'Shader source changed: '+name)
    require(sha(model)==MODEL_SHA,'Wrong actual public GGUF')
    readiness=prepared/'mac-readiness';commands=read(readiness/'COMMANDS.json');overall=read(readiness/'SUMMARY.json')
    require(overall['Mac_correctness_readiness_pass'] is True and overall['actual18casecache_pass'] is True and
            overall['backend_TQ2_cases']==59 and overall['backend_ACT_QUANT_cases']==16,'Prepared correctness not complete')
    histories={};evidence={str(build/'FRESH_CORE_BUILD.json'):sha(build/'FRESH_CORE_BUILD.json'),
                            str(shaders/'BUILD.json'):sha(shaders/'BUILD.json'),str(library):sha(library)}
    for name in ('primary','independent'):
        folder=readiness/name;analysis=read(folder/'analysis/SUMMARY.json');runfile=folder/'verify/RUN.json';native=read(runfile)
        config=read(folder/'verify-config.json');request=read(folder/'verify/REQUEST.json');tokens=Path(config['tokens']).resolve();token_identity(tokens)
        require(analysis['analysis_completed'] is True and analysis['scoped_stock_compatibility_pass'] is True and analysis['strict_scoped_exit_code']==0 and
                analysis['checks'] and all(v is True for v in analysis['checks'].values()) and analysis['same_live_bindings_observed'] is True,
                'Prepared same-process S/QB/Q4 correctness missing: '+name)
        require(analysis['native_run_sha256']==sha(runfile) and native['operation']=='attribution' and native['attribution_kind']=='verify' and
                native['execution_pass'] is True and native['activation_quantization'] is True and native['one_model_and_context'] is True and
                native['device']=='Apple M2 Pro' and native['backend']=='MTL0','Analysis is not bound to the supplied real Mac verification')
        require(native['model_sha256']==analysis['model_sha256']==MODEL_SHA and native['tokens_sha256']==analysis['tokens_sha256']==sha(tokens) and
                native['q_library_sha256']==analysis['q_library_sha256']==sha(library) and config==request,
                'Prepared native model/token/library/request identity differs')
        executed=[v for v in commands if v['name']==name+'-verify']
        require(len(executed)==1 and type(executed[0]['exit_code']) is int and executed[0]['exit_code']==0 and
                [Path(v).resolve() for v in executed[0]['argv']]==[runner,(folder/'verify-config.json').resolve(),(folder/'verify').resolve()],
                'Actual verification execution did not use the frozen runner')
        histories[name]={'tokens':str(tokens),'tokens_sha256':sha(tokens),'same_process_scoped_pass':True,
                         'verification_native_sha256':sha(runfile),'verification_analysis_sha256':sha(folder/'analysis/SUMMARY.json')}
    for file in sorted(readiness.rglob('*')):
        if file.is_file() and file.suffix in ('.json','.log','.f32','.tq2'):evidence[str(file.resolve())]=sha(file)
    identity={'q_library':str(library),'q_library_sha256':sha(library),'q_variant':'stock4x1',
              'q_library_source_sha256':shader['sources']['bitnet/shaders/stock4x1.metal'],'q_library_manifest_sha256':sha(shaders/'BUILD.json'),
              'q_library_sdk':'macosx','q_kernel_contract_version':contract['version'],'q_kernel_config_sha256':contract['config_sha256'],
              'q_entry':contract['entry'],'q_grid':contract['grid'],'q_threads':contract['threads'],
              'q_rows_per_simd':contract['rows'],'q_simd_per_group':contract['groups']}
    return {'schema':1,'scope':'M2 Pro existing Qcur-0 candidate value; not A16 qualification','platform':'Mac','device':'Apple M2 Pro',
            'scoped_correctness_pass':True,'native_source_hash_compatible':True,'native_source_files':len(manifest['source_files_sha256']),
            'source_root':str(source),'prepared_source_root':str(old_source),'build_dir':str(build),'runner':str(runner),
            'runner_sha256':sha(runner),'products_sha256':manifest['products_sha256'],'model':str(model),'model_sha256':MODEL_SHA,
            'Q4':identity,'histories':histories,'evidence_sha256':evidence,
            'official_source_model_output_pass':None,'full_model_FP64_pass':None,'model_quality_pass':None,
            'historical_official_strict_exit_preserved':1,
            'thermal_qualification':'Native Mac thermal/low-power states unavailable; no stable-thermal claim'}


def validate_gate(path):
    gate=read(path)
    require(gate['platform']=='Mac' and gate['device']=='Apple M2 Pro' and gate['scoped_correctness_pass'] is True and
            gate['model_sha256']==MODEL_SHA and len(gate['histories'])==2 and all(v['same_process_scoped_pass'] is True for v in gate['histories'].values()),
            'Not a complete scoped M2 gate')
    require(sha(gate['runner'])==gate['runner_sha256'],'Runner changed after qualification')
    for name,digest in gate['products_sha256'].items():require(sha(relative(Path(gate['build_dir']),name))==digest,'Frozen binary changed')
    for name,digest in gate['evidence_sha256'].items():require(sha(name)==digest,'Prepared correctness evidence changed')
    for history in gate['histories'].values():require(sha(history['tokens'])==history['tokens_sha256'],'Token file changed')
    return gate


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
    mode.add_argument('--plan',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--through',choices=('diagnostic','overhead','formal'),default='formal')
    p.add_argument('--direct-qb-q4',action='store_true');args=p.parse_args();args.output=args.output.expanduser().resolve()
    if args.output.exists():p.error('Output must be fresh')
    args.output.mkdir(parents=True);result={'scope':'Mac existing Qcur-0 candidate value','plan_only':not args.run,'GPU_executed':False,
        'GPU_execution_attempted':False,
        'measurement_integrity_pass':None,'speedup_established':None,'official_source_model_output_pass':None,
        'model_quality_pass':None,'historical_official_strict_exit_preserved':1,'stages':[],'errors':[]}
    try:
        state=read(args.state);gate=build_gate(state);gatefile=args.output/'MAC_GATE.json';save(gatefile,gate)
        tasks=[];kinds=['diagnostic']+(['overhead'] if args.through in ('overhead','formal') else [])+(['formal'] if args.through=='formal' else [])
        for kind in kinds:
            for history,data in gate['histories'].items():
                name=kind+'-'+history;config={**gate['Q4'],'operation':'attribution','attribution_kind':kind,
                    'backend':'MTL0','actquant':True,'kv':'F16','prefill':'batched','workload':'teacher','capture':False,'trace':False,
                    'model':gate['model'],'tokens':data['tokens'],'correctness_gate_sha256':sha(gatefile),
                    **attribution_protocol(kind,args.direct_qb_q4 and kind!='overhead')}
                file=args.output/(name+'-config.json');save(file,config);native=args.output/name
                tasks.append((name,[gate['runner'],str(file),str(native)]))
                tasks.append((name+'-analysis',[sys.executable,str(ROOT/'bitnet/tests/analyze_mac_q_value.py'),
                    '--run',str(native),'--gate',str(gatefile),'--output',str(args.output/(name+'-analysis'))]))
        save(args.output/'PLAN.json',{'platform':'Mac','GPU_executed':False,'stages':[{'name':n,'argv':v} for n,v in tasks]})
        result['scoped_correctness_pass']=True
        if not args.run:result.update(planning_pass=True,exit_code=0);return 0
        logdir=args.output/'logs';logdir.mkdir();env=os.environ.copy();env['DYLD_LIBRARY_PATH']=str(Path(gate['build_dir'])/'bin')
        for name,argv in tasks:
            validate_gate(gatefile);record={'argv':argv,'cwd':str(ROOT),'DYLD_LIBRARY_PATH':env['DYLD_LIBRARY_PATH']};save(logdir/(name+'.command.json'),record)
            if not name.endswith('-analysis'):
                result['GPU_execution_attempted']=True
                if result['GPU_executed'] is not True:result['GPU_executed']=None
            with (logdir/(name+'.log')).open('w') as log:process=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            record['exit_code']=process.returncode;save(logdir/(name+'.command.json'),record)
            result['stages'].append({'name':name,'exit_code':process.returncode});print(name+': exit '+str(process.returncode),flush=True)
            if process.returncode:result.update(failed_stage=name,exit_code=process.returncode if process.returncode>0 else 1);return result['exit_code']
            if not name.endswith('-analysis'):result['GPU_executed']=True
        result.update(measurement_integrity_pass=True,exit_code=0);return 0
    except (OSError,ValueError,KeyError,TypeError) as error:result['errors'].append(str(error));result['exit_code']=2;return 2
    finally:save(args.output/'SUMMARY.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':raise SystemExit(main())
