#!/usr/bin/env python3
"""Derived teacher-forced/activation comparisons; never updates old strict results."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(actual,reference):
 a=np.asarray(actual,dtype=np.float64);b=np.asarray(reference,dtype=np.float64);e=np.abs(a-b);t=2e-4+2e-5*np.abs(b)
 return {'values':int(a.size),'nonfinite':int(np.count_nonzero(~np.isfinite(a))),
 'max_abs':float(e.max()),'RMSE':float(np.sqrt(np.mean((a-b)**2))),
 'relative_L2':float(np.linalg.norm(a-b)/np.linalg.norm(b)),
 'frozen_tolerance_failures':int(np.count_nonzero(e>t)),'max_error_over_tolerance':float((e/t).max())}
def ranking(v):
 vmax=v.max();best=int(v.argmax());ids=np.lexsort((np.arange(v.size),-v))[:10]
 return {'greedy_token_id':best,'max_logit_tie_count':int(np.count_nonzero(v==vmax)),
  'tie_policy':'numpy/torch argmax first smallest token index; top10 sorted logit descending then tokenID ascending',
  'top10':[{'token_id':int(t),'logit':float(v[t])} for t in ids],
  'specific_tokens':{str(t):{'logit':float(v[t]),'rank_competition':int(np.count_nonzero(v>v[t]))+1} for t in (83699,14515)},
  'computations_minus_graphics':float(v[83699]-v[14515]),'top1_top2_margin':float(v[ids[0]]-v[ids[1]])}
def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--qvac-stock-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 tokens=json.loads((a.reference/'tokens.json').read_text());execution=json.loads((a.qvac_stock_run/'data/execution.json').read_text());version=json.loads((a.qvac_stock_run/'versions.json').read_text());ref_manifest=json.loads((a.reference/'MANIFEST.json').read_text())
 assert execution['prompt_ids']==tokens['prompt'] and execution['continuation_ids']==tokens['continuation']
 assert execution['mode']=='stock' and execution['continuation_tokens']==32
 assert version['model']['sha256']=='52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2' and version['model_sha256_verified']
 rawfiles=[a.qvac_stock_run/'data/logits'/('step-'+str(i)+'.f32') for i in range(33)]
 stock=np.stack([np.fromfile(p,dtype='<f4') for p in rawfiles]);assert stock.shape==(33,128256)
 refs={m:np.fromfile(a.reference/(m+'-teacher-forced-logits.f32'),dtype='<f4').reshape(33,128256) for m in ('quant_on','quant_off')}
 rows=[];positions=[]
 for i,y in enumerate(stock):
  item={'logit_index':i,'shared_continuation_tokens_already_fed':i,'input_position':26+i,'rankings':{'qvac_stock':ranking(y)}}
  for mode,x in refs.items():
   item['rankings'][mode]=ranking(x[i]);r=metrics(y,x[i]);r.update(position=i,comparison='qvac_stock_vs_'+mode,greedy_equal=int(y.argmax()==x[i].argmax()));rows.append(r)
  positions.append(item)
 intermediates=[]
 for step in (1,16,32):
  for role,filename in [('Q_input','q/x.f32'),('Q_preRoPE','q/y.f32'),('layer0_residual_output','l_out/y.f32')]:
   p=a.qvac_stock_run/'data'/('step-'+str(step))/'layer-0'/filename
   actual=np.fromfile(p,dtype='<f4');assert actual.size==2560
   for mode in refs:
    rf=a.reference/(mode+({'Q_input':'-layer0-q-X.f32','Q_preRoPE':'-layer0-q-preRoPE.f32','layer0_residual_output':'-layer-0-l_out.f32'}[role]))
    reference=np.fromfile(rf,dtype='<f4').reshape(59,2560)[26+step]
    intermediates.append({'step':step,'position':26+step,'role':role,'mode':mode,'QVAC_file':str(p.resolve()),'QVAC_sha256':sha(p),'reference_file':str(rf.resolve()),'reference_sha256':sha(rf),'comparison':metrics(actual,reference)})
 summary={'processing_completed':True,'existing_QVAC_run_reused':str(a.qvac_stock_run.resolve()),
 'QVAC_model_SHA256':version['model']['sha256'],'QVAC_commit':version['qvac_commit'],'exact_prompt_and_continuation_ID_match':True,
 'vectors':33,'vocabulary':128256,'QVAC_settings':{k:execution[k] for k in ('mode','n_ctx','n_batch','n_ubatch','threads','kv_k','kv_v','flash_attention')},
 'reference_settings':'CPU '+ref_manifest['dtype']+' source-derived official eager full-sequence; no cache; source immutable096f25ae; same originalcheckpoint04c3b9ad. QuantON uses official formula; OFF bypassesonlyActQuant.',
 'scope':'Cross-execution comparison combines activation quantization, BF16/F32 arithmetic, KV/no-cache, kernel order and backend differences; cannot isolate cause. ON/OFF samebackend control and sameinputF32Qcontrol isolatequantizer separately.',
 'comparisons':{mode:dict(metrics(stock,ref),greedy_agree_positions=int(np.count_nonzero(stock.argmax(1)==ref.argmax(1)))) for mode,ref in refs.items()},
 'token18_prediction':positions[17],'source_hashes':{str(p.resolve()):sha(p) for p in rawfiles},
 'frozen_original_strict_comparison_modified':False,'semantic_equivalence_verified':False,'quality_acceptance_pass':None,
 'processing_exit_code':0,'old_strict_exit_code':1}
 for name,data in [('SUMMARY.json',summary),('rankings.json',positions),('intermediates.json',intermediates)]:
  (a.output/name).write_text(json.dumps(data,indent=2)+'\n')
 with (a.output/'logit-comparisons.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
