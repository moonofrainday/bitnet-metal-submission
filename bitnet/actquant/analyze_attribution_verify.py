#!/usr/bin/env python3
"""CPU-only same-process S/QB/Q4 verification, including observed stock PSO identity."""
import argparse
import json
from pathlib import Path
import re
from types import SimpleNamespace
import analyze_q4 as q4
import analyze_qb as qb


def split_trace(text, output, expected_context):
    current=None;blocks={};contexts=[]
    for line in text.splitlines():
        fields=dict(re.findall(r'(\w+)=(\[[^\]]*\]|[^\s]+)',line))
        if line.startswith('ACTQUANT_ATTRIBUTION_PATH_BEGIN '):
            q4.require(current is None and fields['capture']=='1','Invalid same-process trace boundary')
            current=fields['route'];q4.require(current in ('stock','QB','Q4') and current not in blocks,'Duplicate route trace')
            q4.require(int(fields['context'],16)==int(expected_context),'Trace context differs from native context')
            contexts.append(fields['context']);blocks[current]=[]
        elif line.startswith('ACTQUANT_ATTRIBUTION_PATH_END '):
            q4.require(current==fields['route'] and fields['synchronized']=='1','Incomplete same-process route')
            current=None
        elif current is not None:
            blocks[current].append(line)
    q4.require(current is None and set(blocks)=={'stock','QB','Q4'} and len(set(contexts))==1,'Incomplete one-context routes')
    result={}
    for route,lines in blocks.items():
        file=output/(route+'.log');file.write_text('\n'.join(lines)+'\n');result[route]=file
    return result


def stock_abi(path, guarded):
    records=[]
    for line in path.read_text().splitlines():
        if not line.startswith('BITNET_Q_STOCK_ABI '):continue
        fields=dict(re.findall(r'(\w+)=(\[[^\]]*\]|[^\s]+)',line))
        if fields.get('node')!='Qcur-0':continue
        q4.require(fields['guarded']==str(int(guarded)) and fields['pipeline']=='kernel_mul_mv_tq2_0_f32','Wrong live stock ABI route')
        records.append(fields)
    q4.require([int(r['step']) for r in records]==list(range(1,33)),'Missing32 actual stock ABI records')
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','log','stock-null','missing-library','missing-library-log','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--official-source',type=Path,default=Path(__file__).resolve().parents[1]/'reference/official-bitnet-096f25ae.py')
    args=p.parse_args();q4.require(not args.output.exists(),'Refuse existing verification output');args.output.mkdir(parents=True)
    result={'scope':'Same-build ActQuant-ON one-context S/QB/Q4 verification','analysis_completed':False,
            'scoped_stock_compatibility_pass':None,'official_source_model_output_pass':None,'full_model_FP64_pass':None,
            'model_quality_pass':None,'performance_measured':False,'historical_official_strict_exit_preserved':1}
    try:
        root=q4.native_root(args.run);native=q4.load(root/'RUN.json')
        q4.require(native['operation']=='attribution' and native['attribution_kind']=='verify' and native['execution_pass'] is True and
                   native['one_model_and_context'] is True and native['activation_quantization'] is True and native['performance_measurement'] is False,
                   'Expected completed same-context ActQuant verification')
        records={(r['route'],r['capture_enabled']):r for r in native['paths']}
        q4.require(len(records)==len(native['paths'])==6 and set(records)=={(r,c) for r in ('S','QB','Q4') for c in (True,False)},'Incomplete verification paths')
        folders={key:root/record['subrun'] for key,record in records.items()}
        for folder in folders.values():
            q4.require(folder.resolve().is_relative_to(root.resolve()),'Subrun outside native root')
            sub=q4.load(folder/'RUN.json');q4.require(sub['context_id']==native['context_id'] and sub['callback_pointer_present'] is True,'Not the same callback-capable context')
        trace_dir=args.output/'trace';trace_dir.mkdir()
        traces=split_trace(args.log.read_text(),trace_dir,native['context_id'])
        shared=dict(stock=folders[('S',True)],stock_plain=folders[('S',False)],stock_log=traces['stock'],official_source=args.official_source)
        qb_code=qb.run(SimpleNamespace(**shared,candidate=folders[('QB',True)],candidate_plain=folders[('QB',False)],
                                      candidate_log=traces['QB'],output=args.output/'QB'))
        q4_code=q4.run(SimpleNamespace(**shared,candidate=folders[('Q4',True)],candidate_plain=folders[('Q4',False)],
                                      candidate_log=traces['Q4'],output=args.output/'Q4',contract_profile='canonical-v1',
                                      legacy_shader_build=None,missing_library=args.missing_library,missing_library_log=args.missing_library_log))
        q4.require(qb_code in (0,1) and q4_code in (0,1),'Required subanalysis has missing/invalid evidence')
        ss,bs=stock_abi(traces['stock'],False),stock_abi(traces['QB'],True)
        rows=[]
        abi_fields=('pipeline','cached_pipeline','args_bytes','add','norm','grid','threads','smem','slots')
        binding_fields=('w','x','y','residual','norm_weight')
        for step,(s,b) in enumerate(zip(ss,bs),1):
            rows.append({'step':step,'same_stock_PSO':s['cached_pipeline']==b['cached_pipeline'],
                         'same_ABI_launch':all(s[k]==b[k] for k in abi_fields),
                         'same_live_bindings':all(s[k]==b[k] for k in binding_fields),
                         'stock_PSO':s['cached_pipeline'],'QB_PSO':b['cached_pipeline'],
                         'differing_binding_fields':[k for k in binding_fields if s[k]!=b[k]]})
        null_root=q4.native_root(args.stock_null);null_run=q4.manifest(null_root,'stock',False)
        null_checks={}
        for (route,captured),folder in folders.items():
            run=q4.load(folder/'RUN.json');q4.require(run['token_ids']==null_run['token_ids'],'Callback-null history differs')
            comparison=q4.vector_comparison(q4.logits(folder,run),q4.logits(null_root,null_run))
            null_checks[route+('-capture' if captured else '-plain')]=comparison
        checks={'QB_scoped_pass':qb_code==0,'Q4_scoped_pass':q4_code==0,
                'same_process_stock_PSO_pass':all(row['same_stock_PSO'] for row in rows),
                'same_stock_ABI_launch_pass':all(row['same_ABI_launch'] for row in rows),
                'same_stock_live_bindings_pass':all(row['same_live_bindings'] for row in rows),
                'callback_null_stock_bitwise_pass':all(row['bitwise_pass'] for row in null_checks.values())}
        verdict,code=q4.scoped_verdict(checks)
        result.update(analysis_completed=True,checks=checks,scoped_stock_compatibility_pass=verdict,strict_scoped_exit_code=code,
                      same_live_bindings_observed=all(row['same_live_bindings'] for row in rows),PSO_ABI_rows=rows,
                      callback_null_comparisons=null_checks,QB_summary_sha256=q4.sha(args.output/'QB/SUMMARY.json'),
                      Q4_summary_sha256=q4.sha(args.output/'Q4/SUMMARY.json'),native_run_sha256=q4.sha(root/'RUN.json'),
                      model_sha256=native['model_sha256'],tokens_sha256=native['tokens_sha256'],q_library_sha256=native['q_library_sha256'],
                      callback_note='Verify plain retains a callback pointer returning false; all33 logits also compared with original15-stage callback-null stockplain',
                      binding_note='Live handle/offset equality is observed separately; identical captured W/X is enforced by same-input subanalyses')
    except (q4.EvidenceError,OSError,ValueError,KeyError,TypeError) as error:
        result.update(error=str(error),scoped_stock_compatibility_pass=None,strict_scoped_exit_code=2);code=2
    (args.output/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['PSO_ABI_rows','callback_null_comparisons']},indent=2));return code


if __name__=='__main__':raise SystemExit(main())
