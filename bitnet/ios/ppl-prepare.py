#!/usr/bin/env python3
"""Prepare exact BOS+128 held-out IDs for a stock physical-phone quality smoke."""
import argparse,json
from pathlib import Path
from common import PUBLIC_MODEL_SHA256,require,sha256,write_json

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--tokens-json',type=Path,required=True);p.add_argument('--reference-manifest',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 t=json.loads(a.tokens_json.read_text());r=json.loads(a.reference_manifest.read_text());ids=t['corpus']
 require(len(ids)==129 and ids[0]==128000 and t['corpus_scored_tokens']==128,'Expected explicit BOS+128 ordinary token IDs')
 require(all(isinstance(i,int) and 0<=i<128256 for i in ids),'Bad corpus token ID')
 require(not a.output.exists(),'Refuse existing quality fixture');a.output.mkdir(parents=True)
 doc={'schema':1,'tokens':ids,'scored_tokens':128,'model_sha256':PUBLIC_MODEL_SHA256,
      'source_tokens_json_sha256':sha256(a.tokens_json),'source_reference_manifest_sha256':sha256(a.reference_manifest),
      'tokenizer_sha256':r['tokenizer_sha256'],'dataset_repository':t['dataset_repository'],
      'dataset_revision':t['dataset_revision'],'dataset_config':t['dataset_config'],'dataset_split':t['dataset_split'],
      'corpus_parquet_sha256':t['corpus_parquet_sha256'],'corpus_definition':t['corpus_definition'],
      'scope':'Current stock QVAC quality smoke on128 test-split targets; not a representative benchmark or public/team model semantic proof',
      'retokenized':False,'chat_template_applied':False,'BOS_count_explicitly_prepended':1,
      'teacher_forcing':'N1 sequential;decode token[i],score token[i+1],i=0..127',
      'raw_artifacts_stay_local':True}
 write_json(a.output/'QUALITY.json',doc);print(json.dumps({'fixture':str(a.output.resolve()),'sha256':sha256(a.output/'QUALITY.json'),'scored_tokens':128}));return 0
if __name__=='__main__':raise SystemExit(main())
