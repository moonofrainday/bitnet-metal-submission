#!/usr/bin/env python3
"""Stable independent FP64 log-sum-exp scoring of preserved reference logits.

Torch CPU F32 cross_entropy and FP64 logsumexp differ slightly on this system;
retain original scoring, report both, use this scoring for cross-runtime PPL.
This does not rerun or change any model calculation.
"""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('reference',type=Path);a=p.parse_args()
 target=np.asarray(json.loads((a.reference/'tokens.json').read_text())['corpus'][1:])
 result={'scoring':'Independent NumPy FP64 max-subtracted logsumexp over saved F32 logits; original Torch F32 cross_entropy CSV/summary retained.',
  'scored_tokens':int(len(target)),'model_rerun':False,'models':{},'scope':'128-token test-split smoke only; not representative or significance evidence.'}
 for mode in ['quant_on','quant_off']:
  path=a.reference/(mode+'-corpus-logits.f32');logits=np.memmap(path,dtype='<f4',mode='r',shape=(len(target)+1,128256))
  old=np.loadtxt(a.reference/(mode+'-corpus-nll.csv'),delimiter=',',skiprows=1)[:,2]
  nll=[]
  for i,t in enumerate(target):
   row=logits[i].astype(np.float64);mx=row.max();nll.append(float(mx+np.log(np.exp(row-mx).sum())-row[t]))
  nll=np.asarray(nll);delta=nll-old
  result['models'][mode]={'sum_nll':float(nll.sum()),'mean_nll':float(nll.mean()),'perplexity':float(np.exp(nll.mean())),
   'original_FP32_sum_nll':float(old.sum()),'original_FP32_mean_nll':float(old.mean()),'original_FP32_perplexity':float(np.exp(old.mean())),
   'max_per_token_scoring_abs_difference':float(np.abs(delta).max()),'mean_scoring_difference_FP64_minus_FP32':float(delta.mean()),
   'raw_logits_sha256':sha(path),'tokens_sha256':sha(a.reference/'tokens.json')}
  out=a.reference/(mode+'-corpus-nll-fp64.csv')
  if out.exists():raise FileExistsError(out)
  with out.open('w',newline='') as f:
   writer=csv.writer(f);writer.writerow(['position','target_token','negative_log_likelihood_FP64','original_Torch_FP32_nll','FP64_minus_FP32'])
   writer.writerows((i+1,int(t),float(n),float(o),float(d)) for i,(t,n,o,d) in enumerate(zip(target,nll,old,delta)))
 (a.reference/'SCORING-FP64.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
