#!/usr/bin/env python3
"""CPU-only full-guard/stock QB qualification; not official-model acceptance.

Complete observed failure returns 1. Missing/invalid evidence returns 2.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from analyze_q4 import (ATOL, RTOL, MODEL_SHA, WEIGHT_SHA, EvidenceError, analysis_contract,
                        decode_weight, logits, manifest, metrics, native_root, payload,
                        probe_control, q_records, quantizer_check, require, scoped_verdict,
                        trace_evidence, vector_comparison)


def run(args):
    require(not args.output.exists(), 'Refuse existing analysis')
    args.output.mkdir(parents=True)
    result = {'scope':'ActQuant-ON layer0 QB full guard followed by stock dispatch',
              'analysis_completed':False,'official_source_model_output_pass':None,'full_model_FP64_pass':None,
              'model_quality_pass':None,'performance_measured':False,'historical_official_strict_exit_preserved':1}
    try:
        contract=analysis_contract();sr,br=native_root(args.stock),native_root(args.candidate)
        s,b=manifest(sr,'stock',True,contract),manifest(br,'QB',True,contract)
        require(s['token_ids']==b['token_ids'] and s['device']==b['device'] and s['q_library_sha256']==b['q_library_sha256'],
                'Stock/QB identity differs')
        sq,bq=q_records(s),q_records(b)
        weight=payload(sr,sq[0]['weight_path'],1689600,WEIGHT_SHA,np.uint8)
        payload(br,bq[0]['weight_path'],1689600,WEIGHT_SHA,np.uint8)
        decoded=decode_weight(weight);rows=[];hashes=[];comparisons=[]
        with (args.output/'FP64_ROWS.csv').open('w',newline='') as file:
            writer=csv.writer(file);writer.writerow(['step','row','reference','stock','QB','tolerance','stock_fail','QB_fail'])
            for step,(a,c) in enumerate(zip(sq,bq)):
                n=len(a['input_positions']);require(a['input_sha256']==c['input_sha256'],'Actual stock/QB X differs')
                x=payload(sr,a['input_path'],n*2560,a['input_sha256']);payload(br,c['input_path'],n*2560,c['input_sha256'])
                y=payload(sr,a['path'],n*2560,a['output_sha256']);z=payload(br,c['path'],n*2560,c['output_sha256'])
                comparison,_=metrics(z,y);comparisons.append(comparison)
                hashes.append({'step':step,'W_sha256':WEIGHT_SHA,'X_sha256':a['input_sha256'],
                               'stock_Y_sha256':a['output_sha256'],'QB_Y_sha256':c['output_sha256']})
                if step==0:continue
                reference=np.einsum('rbk,bk->r',decoded,x.astype(np.float64).reshape(10,256),optimize=False)
                sm,sbad=metrics(y,reference);bm,bbad=metrics(z,reference);rows.append({'step':step,'stock':sm,'QB':bm})
                for i in range(2560):writer.writerow([step,i,reference[i],y[i],z[i],ATOL+RTOL*abs(reference[i]),int(sbad[i]),int(bbad[i])])
        layers=[]
        for root,run in [(sr,s),(br,b)]:
            records=[r for r in run['checkpoints'] if r['kind']=='l_out' and r['node']=='l_out-0']
            require([r['input_positions'] for r in records]==[list(range(27)),*[[i] for i in range(27,59)]],
                    'All33 layer outputs required')
            layers.append([payload(root,r['path'],2560*len(r['input_positions'])) for r in records])
        layer=vector_comparison(layers[1],layers[0]);logit=vector_comparison(logits(br,b),logits(sr,s))
        probes={'stock':probe_control(args.stock_plain,sr,s,'stock',contract),
                'QB':probe_control(args.candidate_plain,br,b,'QB',contract)}
        traces={'stock':trace_evidence(args.stock_log,'stock',contract=contract),
                'QB':trace_evidence(args.candidate_log,'QB',contract=contract)}
        quant={'stock':quantizer_check(sr,s,sq,args.official_source),'QB':quantizer_check(br,b,bq,args.official_source)}
        checks={'identical_real_inputs_pass':True,'all33_Q_bitwise_pass':all(r['bitwise_differences']==0 for r in comparisons),
                'fp64_pass':all(r['stock']['pass'] and r['QB']['pass'] for r in rows),
                'all33_layer_and_logit_bitwise_pass':layer['bitwise_pass'] and logit['bitwise_pass'],
                'local_actquant_pass':all(r['pass'] for r in quant.values()),'full_guard_and_stock_dispatch_pass':True,
                'probe_neutrality_pass':None if any(v is None for v in probes.values()) else all(v['bitwise_pass'] for v in probes.values())}
        verdict,code=scoped_verdict(checks)
        result.update(analysis_completed=True,checks=checks,scoped_stock_compatibility_pass=verdict,strict_scoped_exit_code=code,
                      model_sha256=MODEL_SHA,q_library_sha256=b['q_library_sha256'],input_hashes=hashes,Q_comparisons=comparisons,
                      FP64=rows,layers=layer,logits=logit,probe_controls=probes,actual_runtime=traces,actual_input_quantizers=quant,
                      numerical_scope='All32 same-input Q dots; all33 Q/layer/logit parity; prefill parity only',
                      callback_pointer_note='Same-process verify plain can retain an inactive callback pointer; compare separately with the15-stage stockplain callback-null run')
    except (EvidenceError,OSError,KeyError,TypeError,ValueError) as error:
        result.update(error=str(error),scoped_stock_compatibility_pass=None,strict_scoped_exit_code=2);code=2
    (args.output/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['input_hashes','Q_comparisons','FP64','layers','logits','probe_controls']},indent=2))
    return code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['stock','candidate','stock-log','candidate-log','stock-plain','candidate-plain','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--official-source',type=Path,default=Path(__file__).resolve().parents[1]/'reference/official-bitnet-096f25ae.py')
    return run(p.parse_args())


if __name__=='__main__':raise SystemExit(main())
