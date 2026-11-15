#!/usr/bin/env python3
"""Recompute rankings from preserved raw logits after JSON serialization fix.

The F32 initial run misplaced the scores[mode] assignment outside its loop,
omitting quant_on from top10 JSON only. Model arithmetic, raw tensors, logits,
PPL, errors and greedy agreement were unaffected. Keep initial JSON intact.
"""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_reference import ranking

def main():
 p=argparse.ArgumentParser();p.add_argument('reference',type=Path);a=p.parse_args()
 raw={m:np.fromfile(a.reference/(m+'-teacher-forced-logits.f32'),'<f4').reshape(33,128256) for m in ('quant_on','quant_off')}
 rows=[{'logit_index':i,'input_position':26+i,'shared_continuation_tokens_already_fed':i,'scores':{m:ranking(x[i]) for m,x in raw.items()}} for i in range(33)]
 initial=a.reference/'teacher-forced-top10.initial.json';summary_initial=a.reference/'SUMMARY.initial.json'
 if not initial.exists():initial.write_bytes((a.reference/'teacher-forced-top10.json').read_bytes())
 if not summary_initial.exists():summary_initial.write_bytes((a.reference/'SUMMARY.json').read_bytes())
 (a.reference/'teacher-forced-top10.json').write_text(json.dumps(rows,indent=2)+'\n')
 s=json.loads(summary_initial.read_text());s['token18_prediction']=rows[17]
 s['summary_serialization_note']='Rankings rederived from unchanged raw F32 logits, preserving *.initial.json. Fixed scores-loop serialization and made argmax/ties explicit; no model/tensor/PPL changes.'
 (a.reference/'SUMMARY.json').write_text(json.dumps(s,indent=2)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
