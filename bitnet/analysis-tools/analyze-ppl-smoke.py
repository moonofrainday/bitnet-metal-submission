#!/usr/bin/env python3
"""Independently score saved physical-phone logits; never runs a model or GPU.

Raw per-target CSV stays in the requested local output directory. SUMMARY.json
contains only aggregate analysis, hashes and execution settings, without token
payloads, private paths or device identifiers. Exit0 means measurement integrity
passed; it does not imply model semantic equivalence or quality acceptance.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math,re,sys
import numpy as np

VOCAB=128256
MODEL_SHA='52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2'
# Scoring-route verification precision; unrelated to frozen kernel tolerance.
SCORE_ATOL=1e-10
SCORE_RTOL=1e-11

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def load(path):return json.loads(path.read_text())
def near(a,b):return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=SCORE_ATOL+SCORE_RTOL*abs(b)
def logsumexp_nll(row,target):
 x=np.asarray(row,dtype=np.float64);maximum=float(x.max())
 return maximum+math.log(float(np.exp(x-maximum).sum(dtype=np.float64)))-float(x[target])
def memlogits(path,rows):
 if path.stat().st_size!=rows*VOCAB*4:raise ValueError('Wrong-sized raw logits: '+path.name)
 return np.memmap(path,mode='r',dtype='<f4',shape=(rows,VOCAB))

def analyze(a):
 fixture=load(a.fixture/'QUALITY.json');tokens=fixture['tokens']
 if len(tokens)!=129 or tokens[0]!=128000 or fixture['scored_tokens']!=128 or any(type(t)!=int or not 0<=t<VOCAB for t in tokens):
  raise ValueError('Expected explicit BOS+128 exact valid token IDs')
 host=load(a.input/'summary.json');native=load(a.input/'device/quality/summary.json')
 app=load(a.input/'device/result.json');cfg=load(a.input/'config.json')
 captured_fixture=load(a.input/'device/quality/fixture.json')
 rawpath=a.input/'device/quality/logits.f32';phone=memlogits(rawpath,128)
 rawcsv=list(csv.DictReader((a.input/'device/quality/nll.csv').open()))
 log=(a.input/'logs/launch.log').read_text(errors='replace')
 checks={
  'fixture_content_identity':captured_fixture==fixture and load(a.input/'QUALITY.json')==fixture,
  'fixture_hash_bound_to_launch_and_completion':cfg['quality_fixture_sha256']==app['quality_fixture_sha256']==sha(a.fixture/'QUALITY.json'),
  'model_hash_identity':cfg['model_sha256']==app['model_sha256']==fixture['model_sha256']==MODEL_SHA,
  'host_and_native_execution_completed':host['execution_pass'] is True and native['execution_pass'] is True and app['process_exit_code']==0,
  'actual_application_exit0':len(re.findall(r'The app terminated with the exit code 0\.',log))==1,
  'actual_A16_and31of31_offload':native['device']=='Apple A16 GPU' and native['offloaded_layers']==native['total_layers']==31 and re.search(r'offloaded\s+31/31\s+layers to GPU',log) is not None,
  'stock_configuration':native['custom_routes_enabled'] is False and native['threads']==6 and native['n_ctx']==512 and native['n_batch']==native['n_ubatch']==128 and native['type_k']==native['type_v']=='F16' and native['offload_kqv'] is True and native['flash_attention'] is False,
  'native128targets_vectors_and_completed_steps':native['scored_tokens']==native['logit_vectors']==native['GPU_synchronized_decode_steps']==128,
  'native_finite_witness':native['nonfinite_logits']==0 and native['finite_logits']==128*VOCAB,
  'native_semantic_acceptance_not_asserted':native['model_semantics_pass'] is None and native['model_semantics_verified'] is False,
  'performance_not_measured':native['formal_performance_run'] is False,
  'native_CSV_128rows':len(rawcsv)==128,
 }
 completion=re.findall(r'BITNET_QUALITY_GPU_COMPLETE input_position=(\d+) target_position=(\d+) backend=MTL0 finite_logits=128256 synchronized=1',log)
 checks['128ordered_API_synchronize_completion_markers']=[(int(i),int(j)) for i,j in completion]==[(i,i+1) for i in range(128)]
 rows=[];nlls=[];phone_nonfinite=0;csv_mismatches=[];max_nll_diff=0.0
 for i,row in enumerate(phone):
  finite=bool(np.isfinite(row).all());phone_nonfinite+=int(np.count_nonzero(~np.isfinite(row)))
  nll=logsumexp_nll(row,tokens[i+1]) if finite else None
  if nll is not None:nlls.append(nll)
  original=rawcsv[i] if i<len(rawcsv) else {}
  identity=(original.get('target_position')==str(i+1) and original.get('input_position')==str(i)
   and original.get('input_token')==str(tokens[i]) and original.get('target_token')==str(tokens[i+1]) and original.get('all_finite')=='1')
  recorded=float(original['negative_log_likelihood']) if original.get('negative_log_likelihood') is not None else None
  agrees=identity and nll is not None and recorded is not None and near(recorded,nll)
  greedy=int(np.argmax(row)) if finite else None
  if original.get('greedy_token')!=str(greedy):agrees=False
  if not agrees:csv_mismatches.append(i+1)
  delta=abs(recorded-nll) if recorded is not None and nll is not None else None
  if delta is not None:max_nll_diff=max(max_nll_diff,delta)
  rows.append({'target_position':i+1,'input_position':i,'input_token':tokens[i],'target_token':tokens[i+1],
   'phone_NLL_FP64':nll,'native_NLL_FP64':recorded,'native_NLL_abs_difference':delta,
   'native_CSV_identity_and_score_pass':agrees,'phone_greedy_token':greedy})
 checks['all_raw_phone_logits_finite']=phone_nonfinite==0
 checks['all_native_CSV_targets_greedy_and_NLL_match']=not csv_mismatches
 total=math.fsum(nlls) if len(nlls)==128 else None
 mean=total/128 if total is not None else None;ppl=math.exp(mean) if mean is not None else None
 checks['native_sum_mean_and_PPL_recomputed']=total is not None and near(float(native['sum_NLL']),total) and near(float(native['mean_NLL']),mean) and near(float(native['perplexity']),ppl)
 result={'schema':1,'CPU_only':True,'model_or_GPU_executed':False,
  'scope':'Current-stock phone128-target quality smoke and saved-reference comparison; nonrepresentative, no semantic-fidelity claim',
  'tolerance_scope':{'scoring_verification_abs':SCORE_ATOL,'scoring_verification_relative':SCORE_RTOL,'not_frozen_kernel_acceptance':True},
  'checks':checks,'phone':{'scored_tokens':128,'vocabulary':VOCAB,'logit_vectors':128,'nonfinite_logits':phone_nonfinite,
   'sum_NLL_FP64':total,'mean_NLL_FP64':mean,'PPL_FP64':ppl,'native_CSV_mismatch_count':len(csv_mismatches),
   'native_CSV_first_mismatch_position':csv_mismatches[0] if csv_mismatches else None,'max_native_NLL_abs_difference':max_nll_diff,
   'raw_logits_sha256':sha(rawpath),'native_summary_sha256':sha(a.input/'device/quality/summary.json'),
   'device':'iPhone14Pro / AppleA16GPU','iOS':native['iOS'],'offloaded_layers':31,'total_layers':31,
   'configuration':{k:native[k] for k in ['threads','n_ctx','n_batch','n_ubatch','type_k','type_v','offload_kqv','flash_attention']},
   'completion_evidence':'128 successful N1 decodes followed by llama_synchronize and full finite logits; no per-command-buffer callback'},
  'fixture_sha256':sha(a.fixture/'QUALITY.json'),'model_sha256':MODEL_SHA,'references':{},
  'parity_pass':None,'fp64_pass':None,'model_output_pass':None,'model_semantics_pass':None,
  'model_semantics_verified':False,'global_strict_V1_exit_code':2,'formal_performance_run':False,
  'comparison_limits':['Source-derived eager full-sequence/no-cache reference differs from sequential phone F16 KV execution.',
    'ActQuant ON/OFF isolates activation semantics only inside the reference; phone/reference differences also include arithmetic and execution differences.',
    '128 test-split targets are not representative and may have appeared in pretraining; no significance or quality superiority conclusion.']}
 for precision,directory in [('F32',a.reference_f32),('BF16',a.reference_bf16)]:
  rtokens=load(directory/'tokens.json');manifest=load(directory/'MANIFEST.json')
  exact_ids=rtokens['corpus']==tokens and rtokens['corpus_scored_tokens']==128
  checks[precision+'_reference_exact_tokenIDs']=exact_ids
  fp64scores=load(directory/'SCORING-FP64.json')
  for mode in ['quant_on','quant_off']:
   name=precision+'_'+mode;rp=directory/(mode+'-corpus-logits.f32');reference=memlogits(rp,129)
   nonfinite=0;ref_nlls=[];squared=0.0;ref_squared=0.0;maximum=0.0;equal=0;max_native_diff=0.0
   for i,ref in enumerate(reference[:128]):
    finite=bool(np.isfinite(ref).all());nonfinite+=int(np.count_nonzero(~np.isfinite(ref)))
    ref_nll=logsumexp_nll(ref,tokens[i+1]) if finite else None
    if ref_nll is not None:ref_nlls.append(ref_nll)
    rows[i][name+'_NLL_FP64']=ref_nll
    rows[i][name+'_greedy_token']=int(np.argmax(ref)) if finite else None
    if finite and bool(np.isfinite(phone[i]).all()):
     r64=np.asarray(ref,dtype=np.float64);delta=np.asarray(phone[i],dtype=np.float64)-r64
     squared+=float(np.dot(delta,delta));ref_squared+=float(np.dot(r64,r64));maximum=max(maximum,float(np.abs(delta).max()))
     same=int(np.argmax(phone[i]))==int(np.argmax(ref));equal+=int(same);rows[i][name+'_phone_greedy_equal']=same
    else:rows[i][name+'_phone_greedy_equal']=None
   refmean=math.fsum(ref_nlls)/128 if len(ref_nlls)==128 else None
   refppl=math.exp(refmean) if refmean is not None else None
   saved=fp64scores['models'][mode];score_ok=refmean is not None and near(saved['mean_nll'],refmean) and near(saved['perplexity'],refppl)
   checks[name+'_finite_and_FP64_scoring_reproduced']=nonfinite==0 and score_ok and sha(rp)==saved['raw_logits_sha256']
   result['references'][name]={'inference_dtype':precision,'activation_quantization':mode,
     'scored_tokens':128,'raw_logit_vectors':129,'prediction_vectors_compared':128,'nonfinite_logits_in_compared_vectors':nonfinite,
     'mean_NLL_FP64':refmean,'PPL_FP64':refppl,'phone_minus_reference_mean_NLL':mean-refmean if mean is not None and refmean is not None else None,
     'phone_minus_reference_PPL':ppl-refppl if ppl is not None and refppl is not None else None,
     'phone_vs_reference_greedy_equal_positions':equal,'greedy_positions':128,
     'phone_vs_reference_logit_RMSE':math.sqrt(squared/(128*VOCAB)) if nonfinite==0 and phone_nonfinite==0 else None,
     'phone_vs_reference_relative_L2':math.sqrt(squared/ref_squared) if nonfinite==0 and phone_nonfinite==0 and ref_squared>0 else None,
     'phone_vs_reference_max_abs_logit_difference':maximum if nonfinite==0 and phone_nonfinite==0 else None,
     'raw_logits_sha256':sha(rp),'source_manifest_sha256':sha(directory/'MANIFEST.json'),
     'checkpoint_sha256':manifest['checkpoint_sha256'],'tokenizer_sha256':manifest['tokenizer_sha256']}
 result['measurement_integrity_pass']=all(checks.values());result['analysis_exit_code']=0 if result['measurement_integrity_pass'] else 1
 a.output.mkdir(parents=True,exist_ok=False)
 (a.output/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 with (a.output/'per-target-local.csv').open('w',newline='') as f:
  fields=list(dict.fromkeys(k for row in rows for k in row));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',type=Path,required=True,help='New phone ppl-run.py host output')
 p.add_argument('--fixture',type=Path,required=True,help='Directory containing exact QUALITY.json')
 p.add_argument('--reference-f32',type=Path,required=True)
 p.add_argument('--reference-bf16',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True,help='Fresh LOCAL analysis directory; CSV is raw and must not be published')
 a=p.parse_args()
 if a.output.exists():print('Refusing existing output',file=sys.stderr);return 3
 try:
  result=analyze(a);print(json.dumps({'measurement_integrity_pass':result['measurement_integrity_pass'],'analysis_exit_code':result['analysis_exit_code'],'global_strict_V1_exit_code':2,'phone':result['phone'],'failed_checks':[k for k,v in result['checks'].items() if not v]},indent=2));return result['analysis_exit_code']
 except FileNotFoundError as e:print(json.dumps({'analysis_completed':False,'missing_file':e.filename,'analysis_exit_code':2}),file=sys.stderr);return 2
 except (ValueError,KeyError,TypeError,OverflowError) as e:print(json.dumps({'analysis_completed':False,'malformed':str(e),'analysis_exit_code':3}),file=sys.stderr);return 3
if __name__=='__main__':raise SystemExit(main())
