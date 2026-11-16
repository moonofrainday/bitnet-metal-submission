#!/usr/bin/env python3
"""Bind completed same-build phone correctness evidence before attribution timing."""
import argparse
import json
from pathlib import Path
from run_phone import build_identity, token_identity
from common import PUBLIC_MODEL_SHA256, require, sha256


def verify_run_binding(directory, analysis, expected_identity, build, tokens_sha256):
    identity=json.loads((directory/'BUILD_IDENTITY.json').read_text())
    require(identity==expected_identity,'Verification run is not from the current signed build/source')
    summary=json.loads((directory/'summary.json').read_text())
    require(summary.get('execution_pass') is True and summary.get('GPU_executed') is True and
            all(type(summary.get(k)) is int and summary[k]==0 for k in ('native_process_exit_code','collection_exit_code','actual_process_exit_code','launch_command_exit_code')),
            'Verification host/native execution did not complete')
    native_file=directory/'device/native/RUN.json';native=json.loads(native_file.read_text());digest=sha256(native_file)
    require(analysis['native_run_sha256']==digest and summary['native_run_sha256']==digest,'Analysis and supplied native RUN differ')
    require(native.get('operation')=='attribution' and native.get('attribution_kind')=='verify' and native.get('execution_pass') is True and
            native.get('activation_quantization') is True and native.get('one_model_and_context') is True and
            native.get('device')=='Apple A16 GPU' and native.get('backend')=='MTL0' and
            native.get('model_sha256')==PUBLIC_MODEL_SHA256 and native.get('tokens_sha256')==tokens_sha256 and
            native.get('q_library_sha256')==build['shader_products_sha256']['q-stock4x1.metallib'],
            'Verification native model/history/library/backend identity differs')
    result=json.loads((directory/'device/result.json').read_text())
    require(result.get('operation')=='actquant' and result.get('actquant_operation')=='attribution' and
            result.get('process_exit_code')==result.get('native_process_exit_code')==0 and result.get('execution_pass') is True and
            result.get('native_run_sha256')==digest,'Verification app completion record does not bind supplied RUN')
    return digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['build-dir','correctness','repeat-contract','primary-analysis','independent-analysis','primary-run','independent-run','tokens-primary','tokens-independent','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Gate output must be fresh')
    result={'schema':1,'scoped_correctness_pass':None,'official_source_model_output_pass':None,
            'full_model_FP64_pass':None,'model_quality_pass':None,'historical_official_strict_exit_preserved':1}
    try:
        build,identity=build_identity(args.build_dir.resolve())
        base=json.loads((args.correctness/'SUMMARY.json').read_text())
        repeat=json.loads((args.repeat_contract/'summary.json').read_text())
        for value in [base.get('scoped_qualification_pass'),repeat.get('runtime_contract_pass')]:
            if value is False:result['scoped_correctness_pass']=False
            require(value is True,'Required preceding correctness/cache gate missing or failed')
        require(base['local_provenance']['build_manifest_sha256']==identity['build_manifest_sha256'] and
                repeat['selected_build_manifest_sha256']==identity['build_manifest_sha256'],'Preceding gates use another signed build')
        histories={}
        for name,analysis,tokens,run in [('primary',args.primary_analysis,args.tokens_primary,args.primary_run),
                                        ('independent',args.independent_analysis,args.tokens_independent,args.independent_run)]:
            token_identity(tokens);data=json.loads((analysis/'SUMMARY.json').read_text())
            if data.get('scoped_stock_compatibility_pass') is False:result['scoped_correctness_pass']=False
            require(data.get('scoped_stock_compatibility_pass') is True and data['strict_scoped_exit_code']==0 and
                    all(v is True for v in data['checks'].values()),'Same-process Q4/QB proof incomplete: '+name)
            require(data['tokens_sha256']==sha256(tokens) and data['model_sha256']==PUBLIC_MODEL_SHA256 and
                    data['q_library_sha256']==build['shader_products_sha256']['q-stock4x1.metallib'],'History/model/library identity mismatch')
            verify_run_binding(run,data,identity,build,sha256(tokens))
            histories[name]={'tokens_sha256':sha256(tokens),'same_process_scoped_pass':True,'analysis_sha256':sha256(analysis/'SUMMARY.json')}
        evidence={}
        for root in [args.correctness,args.repeat_contract,args.primary_analysis,args.independent_analysis,args.primary_run,args.independent_run]:
            for file in sorted(root.rglob('*')):
                if file.is_file() and file.suffix in ('.json','.log','.f32','.tq2'):
                    evidence[str(file.resolve())]=sha256(file)
        result.update(scoped_correctness_pass=True,build_manifest_sha256=identity['build_manifest_sha256'],
                      app_files_sha256=build['app_files_sha256'],framework_sha256=build['framework_sha256'],
                      model_sha256=PUBLIC_MODEL_SHA256,q_library_sha256=build['shader_products_sha256']['q-stock4x1.metallib'],
                      histories=histories,evidence_sha256=evidence)
        code=0
    except (OSError,RuntimeError,ValueError,KeyError,TypeError) as error:
        result['error']=str(error);code=1 if result['scoped_correctness_pass'] is False else 2
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('app_files_sha256','evidence_sha256')},indent=2));return code


if __name__=='__main__':raise SystemExit(main())
