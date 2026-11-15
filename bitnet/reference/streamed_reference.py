#!/usr/bin/env python3
"""CPU-only source-derived official BitNet eager reference; one BF16 layer resident.

This is a controlled semantics/quality smoke, not an inference-speed benchmark.
ActQuant ON/OFF reuse identical checkpoint weights, eager decoder source, data,
and BF16 compute. Raw corpus, tensors, logits and CSV stay in local output only.
"""
import argparse, ast, csv, gc, hashlib, json, logging, math, resource, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

PROMPT=[128000,2374,25,1472,527,264,11190,15592,18328,13,128009,1502,25,83017,1148,264,23501,1587,304,1403,2875,23719,13,128009,72803,25,220]
CONT=[32,23501,11,477,20914,29225,8113,11,374,264,28175,12035,3777,6319,311,3790,6485,14515,21568,9256,13,1102,596,1511,304,8522,430,1397,1579,55580,2410,11]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def jwrite(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def selected(source,names,env,changes):
 tree=ast.parse(source.read_text());nodes=[]
 for n in tree.body:
  if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names:
   for item in ast.walk(n):
    if isinstance(item,(ast.ClassDef,ast.FunctionDef)):
     keep=[]
     for d in item.decorator_list:
      text=ast.unparse(d)
      if text=='torch.compile' or text=='dynamic_rope_update' or text.startswith('use_kernel_forward_from_hub'):
       changes.append({'source':source.name,'symbol':item.name,'removed_decorator':text})
      else:keep.append(d)
     item.decorator_list=keep
   nodes.append(n)
 assert {n.name for n in nodes}==set(names),(source,names)
 body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes
 exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(source),'exec'),env)

def make_env(base):
 env={'torch':torch,'nn':nn,'F':F,'VALUES_PER_ITEM':4,'GradientCheckpointingLayer':nn.Module,
      'logger':logging.getLogger(__name__),'ALL_ATTENTION_FUNCTIONS':{}}
 changes=[]
 selected(base/'official-bitnet-096f25ae.py',{'unpack_weights','WeightQuant','ActQuant','AutoBitLinear'},env,changes)
 selected(base/'official-activations-096f25ae.py',{'ReLUSquaredActivation'},env,changes)
 env['ACT2FN']={'relu2':env['ReLUSquaredActivation']()}
 selected(base/'official-rope-utils-096f25ae.py',{'_compute_default_rope_parameters'},env,changes)
 env['ROPE_INIT_FUNCTIONS']={'default':env['_compute_default_rope_parameters']}
 selected(base/'official-modeling-bitnet-096f25ae.py',{'BitNetRMSNorm','BitNetMLP','rotate_half',
  'apply_rotary_pos_emb','repeat_kv','eager_attention_forward','BitNetAttention','BitNetDecoderLayer','BitNetRotaryEmbedding'},env,changes)
 return env,changes

def prepare(a):
 a.output.mkdir(parents=True,exist_ok=False)
 if a.dependency_dir:sys.path.insert(0,str(a.dependency_dir))
 import pyarrow.parquet as pq
 rows=pq.read_table(a.dataset_parquet,columns=['text'])['text'].to_pylist()
 text='\n\n'.join(rows)
 tokenizer=Tokenizer.from_file(str(a.model_dir/'tokenizer.json'))
 corpus_ids=tokenizer.encode(text,add_special_tokens=False).ids[:a.corpus_tokens]
 tokens={'teacher_forced':PROMPT+CONT,'prompt':PROMPT,'continuation':CONT,
  'corpus':[128000]+corpus_ids,'corpus_scored_tokens':len(corpus_ids),
  'corpus_definition':f'First {a.corpus_tokens} ordinary token IDs of newline-newline joined text rows in original WikiText-2-raw-v1 test parquet order; prepend BOS128000; score positions1..{a.corpus_tokens}',
  'dataset_repository':'Salesforce/wikitext','dataset_revision':'b08601e04326c79dfdd32d625aee71d232d685c3',
  'dataset_config':'wikitext-2-raw-v1','dataset_split':'test','corpus_parquet_sha256':sha(a.dataset_parquet),
  'dataset_license_note':'HF metadata lists cc-by-sa-3.0 and gfdl; dataset card prose links CC-BY-SA4.0. No full corpus redistribution.',
  'scope':'Tiny data-defined held-out-test-split quality smoke, nonrepresentative; not a standard benchmark and not evidence data were unseen during pretraining.'}
 jwrite(a.output/'tokens.json',tokens)
 (a.output/'corpus-preview.txt').write_text(tokenizer.decode(corpus_ids))
 tokcfg=json.loads((a.model_dir/'tokenizer_config.json').read_text())
 template=tokcfg.get('chat_template')
 jwrite(a.output/'prompt-provenance.json',{'fixed_history':'Existing exact IDs retained; no retemplating',
  'decoded_prompt':tokenizer.decode(PROMPT,skip_special_tokens=False),'official_chat_template':template,
  'note':'Fixed replay prompt used for numerical alignment; plain held-out text uses ordinary tokenizer encoding with explicit BOS.'})
 return tokens,tokenizer

def replace_linears(module,env,dtype):
 for name,child in list(module.named_children()):
  if isinstance(child,nn.Linear):
   with torch.device('meta'):
    replacement=env['AutoBitLinear'](child.in_features,child.out_features,bias=child.bias is not None,dtype=dtype,online_quant=False).to(dtype)
   setattr(module,name,replacement)
  else:replace_linears(child,env,dtype)

def metrics(a,b):
 a=a.double();b=b.double();err=(a-b).abs();tol=2e-4+2e-5*b.abs()
 return {'values':a.numel(),'nonfinite':int((~torch.isfinite(a)).sum()),'max_abs':err.max().item(),
  'frozen_tolerance_failures':int((err>tol).sum()),'max_error_over_tolerance':(err/tol).max().item(),
  'relative_L2':((a-b).norm()/b.norm()).item(),'RMSE':(a-b).square().mean().sqrt().item()}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--source-dir',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--dataset-parquet',type=Path,required=True);p.add_argument('--dependency-dir',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--corpus-tokens',type=int,default=128);p.add_argument('--layers',type=int,default=30);p.add_argument('--dtype',choices=['bf16','f32'],default='bf16');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 dtype={'bf16':torch.bfloat16,'f32':torch.float32}[a.dtype]
 torch.set_num_threads(6);torch.set_num_interop_threads(1);torch.manual_seed(1)
 tokens,tokenizer=prepare(a)
 if a.prepare_only:print(a.output/'tokens.json');return 0
 env,changes=make_env(a.source_dir)
 cfg=json.loads((a.model_dir/'config.json').read_text());cfg.update(attention_bias=False,attention_dropout=0.0,_attn_implementation='eager',rope_scaling=None,head_dim=128)
 config=SimpleNamespace(**cfg)
 assert config.num_hidden_layers==30 and config.hidden_act=='relu2' and config.tie_word_embeddings
 quant_forward=env['ActQuant'].forward;active={'mode':None,'count':0}
 def quant_control(ctx,x):
  active['count']+=1
  return quant_forward(ctx,x) if active['mode']=='quant_on' else x
 env['ActQuant'].forward=staticmethod(quant_control)
 sourcepaths=list(a.source_dir.glob('official-*-096f25ae.py'))
 manifest={'official_transformers_commit':'096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4','model_revision':'04c3b9ad9361b824064a1f25ea60a8be9599b127',
  'checkpoint_sha256':sha(a.model_dir/'model.safetensors'),'config':cfg,'torch':torch.__version__,'numpy':np.__version__,
  'device':'CPU','dtype':str(dtype),'CPU_threads':6,'attention':'official-source eager, causal full-sequence, no KV cache',
  'source_sha256':{p.name:sha(p) for p in sourcepaths},'tokenizer_sha256':sha(a.model_dir/'tokenizer.json'),
  'source_transformations':changes,'streaming':'One official decoder layer loaded from original packed checkpoint, ternary unpack into selected compute dtype; same loaded weights reused by all ON/OFF workloads; one tied embedding resident.',
  'precision_profile':'BF16 is official recommended profile; F32 is separate controlled profile, losslessly lifting original BF16 parameter values and using selected dtype for all eager arithmetic. No weight retraining or changed quantizer.',
  'driver_changes':'AST selects original classes; no hub RMSNorm override; eager no torch.compile; base GradientCheckpointingLayer->nn.Module inference; default dynamic-RoPE update wrapper removed (not used by default RoPE); explicit causal mask; no cache.',
  'comparisons':'ON vs OFF isolates ActQuant bypass in the same reference backend; neither is a backend-parity test with QVAC. No speed claim.',
  'full_reference_status':'source-derived official decoder; not an unmodified end-to-end Transformers package run'}
 jwrite(a.output/'MANIFEST.json',manifest)
 inputs={'teacher_forced':torch.tensor(tokens['teacher_forced']).reshape(1,-1),'corpus':torch.tensor(tokens['corpus']).reshape(1,-1)}
 with torch.inference_mode(),safe_open(a.model_dir/'model.safetensors',framework='pt',device='cpu') as ckpt:
  embedding=ckpt.get_tensor('model.embed_tokens.weight');assert embedding.dtype==torch.bfloat16
  embedding=embedding.to(dtype)
  tracks={name:{mode:F.embedding(ids,embedding).clone() for mode in ('quant_on','quant_off')} for name,ids in inputs.items()}
  rotary=env['BitNetRotaryEmbedding'](config)
  masks={};positions={};rotaries={}
  for name,ids in inputs.items():
   length=ids.shape[1];positions[name]=torch.arange(length).unsqueeze(0)
   mask=torch.full((length,length),torch.finfo(dtype).min,dtype=dtype).triu(1)
   masks[name]=mask.reshape(1,1,length,length)
   rotaries[name]=rotary(tracks[name]['quant_on'],positions[name])
  counts=[]
  for idx in range(a.layers):
   start=time.time()
   with torch.device('meta'):layer=env['BitNetDecoderLayer'](config,idx).to(dtype)
   replace_linears(layer,env,dtype)
   prefix='model.layers.'+str(idx)+'.'
   state={k[len(prefix):]:ckpt.get_tensor(k) for k in ckpt.keys() if k.startswith(prefix)}
   state={k:(v.to(dtype) if v.is_floating_point() else v) for k,v in state.items()}
   layer.load_state_dict(state,strict=True,assign=True);layer.eval();del state
   for name,modes in tracks.items():
    for mode,h in modes.items():
     active.update(mode=mode,count=0)
     hooks=[]
     if idx==0 and name=='teacher_forced':
      def save_x(_module,args):
       x=args[0].float().contiguous().numpy();x.tofile(a.output/(mode+'-layer0-q-X.f32'))
       env['ActQuant'].apply(args[0]).float().contiguous().numpy().tofile(a.output/(mode+'-layer0-q-X-effective.f32'))
      def save_y(_module,args,y):y.float().contiguous().numpy().tofile(a.output/(mode+'-layer0-q-preRoPE.f32'))
      hooks=[layer.self_attn.q_proj.register_forward_pre_hook(save_x),layer.self_attn.q_proj.register_forward_hook(save_y)]
     modes[mode]=layer(h,attention_mask=masks[name],position_ids=positions[name],position_embeddings=rotaries[name])[0]
     for hook in hooks:hook.remove()
     assert bool(torch.isfinite(modes[mode]).all()),(idx,name,mode)
     counts.append({'layer':idx,'workload':name,'mode':mode,'quantizer_calls':active['count'],'hook_extra_quantizer_call':int(bool(hooks))})
     if name=='teacher_forced':modes[mode].float().numpy().tofile(a.output/(mode+'-layer-'+str(idx)+'-l_out.f32'))
   del layer;gc.collect()
   print(json.dumps({'layer_completed':idx,'seconds_orchestration_only':round(time.time()-start,2),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)
  jwrite(a.output/'quantizer-calls.json',counts)
  if a.layers!=30:
   jwrite(a.output/'PARTIAL.json',{'layers':a.layers,'completed':True,'full_model_evaluated':False});return 0
  norm=env['BitNetRMSNorm'](config.hidden_size,eps=config.rms_norm_eps).to(dtype)
  norm.load_state_dict({'weight':ckpt.get_tensor('model.norm.weight').to(dtype)},strict=True)
  logits={};nlls={}
  for name,modes in tracks.items():
   logits[name]={}
   for mode,h in modes.items():
    h=norm(h);parts=[]
    for block in h.split(16,dim=1):parts.append(F.linear(block,embedding).float())
    full=torch.cat(parts,dim=1)[0];assert bool(torch.isfinite(full).all())
    if name=='teacher_forced':
     full=full[26:];full.numpy().tofile(a.output/(mode+'-teacher-forced-logits.f32'))
    else:
     nll=F.cross_entropy(full[:-1],inputs[name][0,1:],reduction='none')
     nlls[mode]={'scored_tokens':nll.numel(),'sum_nll':nll.double().sum().item(),'mean_nll':nll.double().mean().item(),'perplexity':math.exp(nll.double().mean().item())}
     with (a.output/(mode+'-corpus-nll.csv')).open('w',newline='') as f:
      writer=csv.writer(f);writer.writerow(['position','target_token','negative_log_likelihood']);writer.writerows((i+1,int(t),float(n)) for i,(t,n) in enumerate(zip(inputs[name][0,1:],nll)))
     full.numpy().tofile(a.output/(mode+'-corpus-logits.f32'))
    logits[name][mode]=full
  on=logits['teacher_forced']['quant_on'];off=logits['teacher_forced']['quant_off'];toprows=[];diffrows=[]
  for i,(v_on,v_off) in enumerate(zip(on,off)):
   scores={}
   for mode,v in [('quant_on',v_on),('quant_off',v_off)]:
    vals,ids=v.topk(10);specific={str(t):{'logit':v[t].item(),'rank':int((v>v[t]).sum())+1} for t in (83699,14515)}
    scores[mode]={'top10':[{'token_id':int(t),'token_text':tokenizer.decode([int(t)]),'logit':float(x)} for t,x in zip(ids,vals)],'specific_tokens':specific,'computations_minus_graphics':(v[83699]-v[14515]).item(),'top1_top2_margin':(vals[0]-vals[1]).item(), 'greedy_token_id':int(v.argmax()),'max_logit_tie_count':int((v==v.max()).sum()),'tie_policy':'argmax first smallest tokenID; torch.topk order among ties is not authoritative'}
   toprows.append({'logit_index':i,'input_position':i+26,'shared_continuation_tokens_already_fed':i,'scores':scores})
   lp_on=v_on.log_softmax(0).double();lp_off=v_off.log_softmax(0).double();d=metrics(v_off,v_on)
   d.update(logit_index=i,greedy_equal=bool(v_on.argmax()==v_off.argmax()),KL_on_to_off=(lp_on.exp()*(lp_on-lp_off)).sum().item());diffrows.append(d)
  jwrite(a.output/'teacher-forced-top10.json',toprows);jwrite(a.output/'teacher-forced-differences.json',diffrows)
  summary={'processing_completed':True,'full_model_source_derived_reference_completed':True,'dtype':str(dtype),'layers':30,
   'teacher_forced_logit_vectors':len(on),'vocabulary':on.shape[-1],'ActQuant_ON_OFF_comparison':metrics(off,on),
   'teacher_forced_greedy_agree_positions':sum(r['greedy_equal'] for r in diffrows),'teacher_forced_greedy_positions':len(diffrows),
   'token18_prediction':toprows[17],'tiny_test_split_PPL':nlls,'peak_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
   'processing_exit_code':0,'semantic_equivalence_proven':False,'QVAC_backend_parity_evaluated':False,
   'quality_scope':f'{a.corpus_tokens}-token canonical test-split slice only; not representative benchmark, not assurance unseen in pretraining, no significance claim; ON/OFF is controlled reference ablation.',
   'performance_scope':'No inference-speed claim; streaming/CPU runtime is orchestration only.'}
  jwrite(a.output/'SUMMARY.json',summary);print(json.dumps(summary,indent=2),flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
