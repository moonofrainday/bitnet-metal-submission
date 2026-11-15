#!/usr/bin/env python3
"""Verify bundled reference source and prepare external immutable data cache.

Never download checkpoint or corpus into the source repository. Model download
is intentionally separate; --model-dir verifies an existing original snapshot.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = '096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4'
DATASET_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
SOURCES = {
 'official-bitnet-096f25ae.py': ('src/transformers/integrations/bitnet.py','a8b3170f1824f010701f4e033fbc33d199bc2ea344ddbf2554594b66fada8b1c'),
 'official-modeling-bitnet-096f25ae.py': ('src/transformers/models/bitnet/modeling_bitnet.py','e3965a8a9768519fdd22f376d216e665fe5f6d74a0a40c87da097b119c5d6e38'),
 'official-rope-utils-096f25ae.py': ('src/transformers/modeling_rope_utils.py','3a0bcdd0d4223c7645fc4bdcc53ff43d16689ead85e0751ed604dd173dba4358'),
 'official-activations-096f25ae.py': ('src/transformers/activations.py','284dc2ef1515e1c6aef4ceb35c9cf9c29d43ae97f77d802533783cfede009fef'),
}
MODEL_HASHES = {
 'model.safetensors':'8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b',
 'tokenizer.json':'e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7',
 'config.json':'2b43e80788972e6d53967b01ac3609d7df23cf0aabb0c866e51be694a59c1149',
}
DATASET_HASH = '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'

def sha(path):
 digest=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4<<20),b''):digest.update(block)
 return digest.hexdigest()

def verified_fetch(path,url,expected):
 if not path.exists():
  data=urlopen(url,timeout=60).read()
  actual=hashlib.sha256(data).hexdigest()
  if actual!=expected:raise ValueError(f'Download checksum mismatch: {url}: {actual}')
  path.write_bytes(data)
 actual=sha(path)
 if actual!=expected:raise ValueError(f'Existing file checksum mismatch; preserve and inspect: {path}: {actual}')
 return {'url':url,'sha256':actual}

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--cache',type=Path,required=True,help='External data/source cache, outside this repository')
 parser.add_argument('--model-dir',type=Path,help='Optional existing original HF snapshot to verify; never copied or downloaded')
 args=parser.parse_args();here=Path(__file__).resolve().parent;repo=here.parents[1];cache=args.cache.resolve()
 if cache.is_relative_to(repo):raise ValueError('Raw dataset cache must be outside the source repository')
 cache.mkdir(parents=True,exist_ok=True)
 result={'reference_revision':REVISION,'dataset_revision':DATASET_REVISION,'official_sources':{},'model_verified':None}
 for name,(relative,expected) in SOURCES.items():
  if sha(here/name)!=expected:raise ValueError(f'Bundled source changed: {name}')
  url=f'https://raw.githubusercontent.com/huggingface/transformers/{REVISION}/{relative}'
  result['official_sources'][name]=verified_fetch(cache/name,url,expected)
 url=f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{DATASET_REVISION}/wikitext-2-raw-v1/test-00000-of-00001.parquet'
 result['dataset']=verified_fetch(cache/'wikitext-2-raw-test.parquet',url,DATASET_HASH)
 if args.model_dir:
  checks={name:sha(args.model_dir/name)==expected for name,expected in MODEL_HASHES.items()}
  if not all(checks.values()):raise ValueError(f'Original model snapshot hash mismatch: {checks}')
  result['model_verified']=True
 (cache/'PREPARATION.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
