#!/usr/bin/env python3
"""Physical-iPhone current-stock PPL smoke; exact tokens and raw logits stay local.

This measures the existing public-checkpoint path, including its known ActQuant
omission. It does not claim semantic fidelity, representative model quality, or
inference speed. Exit0 means the scoped measurement completed; V1 remains2.
"""
import argparse,datetime,json,math,re,sys
from pathlib import Path
from common import PUBLIC_MODEL_SHA256,Commands,developer_environment,require,sha256,write_json

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--mode',choices=['ppl-verify'],default='ppl-verify')
 p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--device',required=True)
 p.add_argument('--model',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--developer-dir',type=Path)
 p.add_argument('--install',action='store_true',required=True);p.add_argument('--reuse-model',action='store_true')
 p.add_argument('--timeout',type=int,default=900);a=p.parse_args()
 require(not a.output.exists(),'Refuse existing local output');a.output.mkdir(parents=True)
 summary={'scope':'ppl-verify current stock QVAC128target quality smoke','execution_pass':False,
   'parity_pass':None,'fp64_pass':None,'model_output_pass':None,'model_semantics_pass':None,
   'model_semantics_verified':False,'strict_V1_exit_code':2,'errors':[]}
 try:
  model=a.model.expanduser().resolve();require(sha256(model)==PUBLIC_MODEL_SHA256,'Wrong public model hash')
  builddir=a.build_dir.expanduser().resolve();build=json.loads((builddir/'BUILD.json').read_text())
  require(build['build_completed'] and build['exit_code']==0 and not build['unsigned'],'Need successful signed physical-device build')
  require('bitnet/ios/src/iphone-ppl.mm' in build['package_sources_sha256'],'Selected app lacks new native quality helper')
  for f,h in build['app_files_sha256'].items():require(sha256(f)==h,'Built app changed')
  fixture=a.fixture.expanduser().resolve();qpath=fixture/'QUALITY.json';quality=json.loads(qpath.read_text())
  require(quality['model_sha256']==PUBLIC_MODEL_SHA256 and len(quality['tokens'])==129 and quality['tokens'][0]==128000 and quality['scored_tokens']==128,'Quality fixture contract')
  env=developer_environment(a.developer_dir);commands=Commands(a.output/'logs',env);bundle=build['bundle_id']
  prefix='candidate-'+datetime.datetime.now().astimezone().strftime('%Y%m%d-%H%M%S%f');remote_fixture=prefix+'-quality-fixture'
  cfg={'operation':'quality','output':prefix,'model':model.name,'model_sha256':PUBLIC_MODEL_SHA256,
       'quality_fixture':remote_fixture,'quality_fixture_sha256':sha256(qpath)}
  write_json(a.output/'config.json',cfg);write_json(a.output/'QUALITY.json',quality)
  write_json(a.output/'BUILD_IDENTITY.json',{'build_manifest_sha256':sha256(builddir/'BUILD.json'),'app_files_sha256':build['app_files_sha256'],'package_sources_sha256':build['package_sources_sha256'],'framework_sha256':build['framework_sha256']})
  def call(name,args,timeout=180,check=True):return commands.run(name,['xcrun','devicectl','--timeout',str(timeout),'device',*args,'--device',a.device],timeout=timeout+20,check=check)
  def copy(name,direction,source,dest,timeout=900):return commands.run(name,['xcrun','devicectl','--timeout',str(timeout),'device','copy',direction,'--device',a.device,'--domain-type','appDataContainer','--domain-identifier',bundle,'--source',source,'--destination',dest],timeout=timeout+20,check=False)
  call('device-details',['info','details'],timeout=60)
  call('install',['install','app',build['app']],timeout=180)
  if not a.reuse_model:require(copy('transfer-model','to',model,'Documents/'+model.name)[0]==0,'Model transfer failed')
  require(copy('transfer-quality-fixture','to',fixture,'Documents/'+remote_fixture)[0]==0,'Quality fixture transfer failed')
  code,log=commands.run('launch',['xcrun','devicectl','--timeout',str(a.timeout),'device','process','launch','--console','--terminate-existing','--device',a.device,'--environment-variables',json.dumps({'BITNET_CANDIDATE_CONFIG':json.dumps(cfg,separators=(',',':'))}),bundle],timeout=a.timeout+20,check=False)
  fetch,_=copy('fetch','from','Documents/'+prefix,a.output/'device')
  require(code==0 and 'The app terminated with the exit code 0.' in log,'Actual app completion exit0 missing')
  require(fetch==0,'Fetch failed');result=json.loads((a.output/'device/result.json').read_text())
  native=json.loads((a.output/'device/quality/summary.json').read_text())
  markers=[json.loads(v.split('BITNET_PHONE_COMPLETE ',1)[1]) for v in log.splitlines() if 'BITNET_PHONE_COMPLETE ' in v]
  require(len(markers)==1 and markers[0]==result,'Console/file completion mismatch')
  require(result['model_sha256']==PUBLIC_MODEL_SHA256 and result['quality_fixture_sha256']==sha256(qpath),'Model/fixture identity mismatch')
  require(result['operation']=='quality' and result['process_exit_code']==0 and result['execution_pass'] is True,'Quality app scope/completion mismatch')
  require(native['device']=='Apple A16 GPU' and native['offloaded_layers']==native['total_layers']==31,'Actual A16/31of31 offload missing')
  require(re.search(r'offloaded\s+31/31\s+layers to GPU',log) is not None,'Offload log missing')
  require(native['scored_tokens']==native['logit_vectors']==native['GPU_synchronized_decode_steps']==128 and native['nonfinite_logits']==0,'Incomplete scoring coverage')
  require(native['custom_routes_enabled'] is False and native['formal_performance_run'] is False,'Wrong custom/performance route')
  require(native['model_semantics_pass'] is None and native['model_semantics_verified'] is False,'Quality smoke cannot prove semantics')
  require((a.output/'device/quality/logits.f32').stat().st_size==128*128256*4,'Missing complete raw logits')
  completion=re.findall(r'BITNET_QUALITY_GPU_COMPLETE input_position=(\d+) target_position=(\d+) backend=MTL0 finite_logits=128256 synchronized=1',log)
  require([(int(i),int(j)) for i,j in completion]==[(i,i+1) for i in range(128)],'Actual128 synchronized completion markers missing')
  require(math.isfinite(native['perplexity']) and native['perplexity']>0,'Invalid PPL')
  summary.update(execution_pass=True,model_quality_tested=True,measurement_exit_code=0,
    completion=native,model_sha256=PUBLIC_MODEL_SHA256,quality_fixture_sha256=sha256(qpath),
    raw_logits_sha256=sha256(a.output/'device/quality/logits.f32'),
    native_summary_sha256=sha256(a.output/'device/quality/summary.json'))
  write_json(a.output/'summary.json',summary);print(json.dumps(summary,indent=2));return 0
 except (RuntimeError,ValueError,KeyError,OSError) as e:
  summary['errors'].append(str(e));summary['measurement_exit_code']=2;write_json(a.output/'summary.json',summary);print(json.dumps(summary,indent=2),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
