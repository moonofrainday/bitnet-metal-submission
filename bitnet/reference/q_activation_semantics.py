#!/usr/bin/env python3
"""CPU-only reference quantizer control on exact captured public BitNet Q inputs."""
import argparse,ast,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(a,b):
 a=np.asarray(a,dtype=np.float64).reshape(-1);b=np.asarray(b,dtype=np.float64).reshape(-1)
 e=np.abs(a-b);t=2e-4+2e-5*np.abs(b)
 return {'values':int(a.size),'failed_rows':int(np.count_nonzero(e>t)),'nonfinite':int(np.count_nonzero(~np.isfinite(a))),
 'max_abs_error':float(e.max()),'max_error_over_tolerance':float((e/t).max()),
 'relative_L2':float(np.linalg.norm(a-b)/np.linalg.norm(b)),'RMSE':float(np.sqrt(np.mean((a-b)**2)))}

def source_classes(source):
 tree=ast.parse(source.read_text());wanted={'unpack_weights','WeightQuant','ActQuant','AutoBitLinear'}
 nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted]
 removed=[]
 for node in nodes:
  for n in ast.walk(node):
   if isinstance(n,ast.FunctionDef):
    keep=[]
    for deco in n.decorator_list:
     if ast.unparse(deco)=='torch.compile':removed.append(n.name)
     else:keep.append(deco)
    n.decorator_list=keep
 env={'torch':torch,'nn':nn,'F':F,'VALUES_PER_ITEM':4}
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),env)
 return env,removed

def decode_tq2(data):
 rows=2560;blocks=10
 d=np.frombuffer(data,dtype=np.uint8).reshape(rows,blocks,66)
 packed=d[:,:,:64].reshape(rows,blocks,2,32)
 codes=np.stack([(packed>>(2*p))&3 for p in range(4)],axis=3)
 assert codes.shape==(rows,blocks,2,4,32) and not np.any(codes==3)
 scale=np.frombuffer(d[:,:,64:].copy().tobytes(),dtype='<f2').reshape(rows,blocks).astype(np.float64)
 return ((codes.astype(np.float64)-1)*scale[:,:,None,None,None]).reshape(rows,2560)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--official-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--strict',action='store_true');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(6);torch.set_num_interop_threads(1)
 env,removed=source_classes(a.official_source)
 with safe_open(a.checkpoint,framework='pt',device='cpu') as f:
  packed=f.get_tensor('model.layers.0.self_attn.q_proj.weight');scale=f.get_tensor('model.layers.0.self_attn.q_proj.weight_scale')
 m=env['AutoBitLinear'](2560,2560,bias=False,dtype=torch.float32,online_quant=False)
 m.load_state_dict({'weight':packed,'weight_scale':scale.to(torch.float32)},strict=True);m.eval()
 raw_w=m.weight.detach().numpy().astype(np.float64)*m.weight_scale.item()
 cases=[];rowdata=[]
 with torch.inference_mode():
  for step in (1,16,32):
   folder=a.fixture/('step-'+str(step));wfile=folder/'w.tq2';xfile=folder/'x.f32';yfile=folder/'y.f32'
   converted_w=decode_tq2(wfile.read_bytes());x=np.fromfile(xfile,dtype='<f4');stock=np.fromfile(yfile,dtype='<f4');assert x.size==stock.size==2560
   xt=torch.from_numpy(x.copy()).reshape(1,-1);quant=env['ActQuant'].apply(xt).numpy().reshape(-1)
   trusted=m(xt).numpy().reshape(-1)
   unquant=converted_w@x.astype(np.float64);quant_conv=converted_w@quant.astype(np.float64);quant_raw=raw_w@quant.astype(np.float64)
   np.asarray(quant,dtype='<f4').tofile(a.output/('step-'+str(step)+'-official-quantized-x.f32'))
   for name,v in [('stock',stock),('official-f32-AutoBitLinear',trusted),('converted-quant-FP64',quant_conv),('converted-unquant-FP64',unquant),('raw-quant-FP64',quant_raw)]:np.asarray(v,dtype='<f8').tofile(a.output/('step-'+str(step)+'-'+name+'.f64'))
   comparisons={'stock_vs_converted_unquant_FP64':metrics(stock,unquant),'stock_vs_converted_quant_FP64':metrics(stock,quant_conv),
    'official_f32_vs_raw_quant_FP64':metrics(trusted,quant_raw),'raw_vs_converted_quant_FP64':metrics(quant_raw,quant_conv),'stock_vs_official_f32_AutoBitLinear':metrics(stock,trusted)}
   item={'step':step,'X_sha256':sha(xfile),'captured_stock_Y_sha256':sha(yfile),'converted_W_sha256':sha(wfile),
    'activation_values_changed_by_official_quantizer':int(np.count_nonzero(x!=quant)),
    'activation_quant_max_abs_delta':float(np.max(np.abs(x.astype(np.float64)-quant))),
    'activation_absmax':float(np.abs(x).max()),'quantizer_scale':float(127/max(float(np.abs(x).max()),1e-5)),
    'raw_checkpoint_vs_TQ2_weight_max_abs':float(np.abs(raw_w-converted_w).max()),'comparisons':comparisons}
   cases.append(item)
   for name,reference in [('converted-unquant-FP64',unquant),('converted-quant-FP64',quant_conv),('official-f32-AutoBitLinear',trusted)]:
    for row,(actual,ref) in enumerate(zip(stock,reference)):
     e=abs(float(actual)-float(ref));t=2e-4+2e-5*abs(float(ref));rowdata.append({'step':step,'comparison':'stock-vs-'+name,'row':row,'stock':float(actual),'reference':float(ref),'abs_error':e,'tolerance':t,'failed':int(e>t)})
 independent_source_verified=all(c['comparisons']['official_f32_vs_raw_quant_FP64']['failed_rows']==0 for c in cases)
 old_path_matches_unquant=all(c['comparisons']['stock_vs_converted_unquant_FP64']['failed_rows']==0 for c in cases)
 matches_quant=all(c['comparisons']['stock_vs_converted_quant_FP64']['failed_rows']==0 for c in cases)
 result={'processing_completed':True,'GPU_executed':False,'CPU_threads':6,'device':'CPU','official_transformers_commit':'096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4',
 'official_source_sha256':sha(a.official_source),'source_transform':'AST-select exact official classes/functions; strip torch.compile decorators only; preserve staticmethod and arithmetic',
 'removed_compile_decorators':removed,'torch':torch.__version__,'reference_dtype':'F32 activation/ternary and F32 output; FP64 independent dot for isolation, not full recommended BF16 model',
 'reference_activation_rule':'x.float();s=127/max(abs(x),1e-5) per token;round ties-to-even;clamp[-128,127];divide bys;cast to inputdtype',
 'checkpoint_q_scale_dtype':str(scale.dtype),'checkpoint_q_scale':float(scale.item()),
 'reference_execution_control_pass':independent_source_verified,'stock_matches_unquantized_dot':old_path_matches_unquant,'stock_matches_official_activation_semantics_on_captured_inputs':matches_quant,
 'full_model_semantics_verified':False,'teacher_forced_logits_evaluated':False,'perplexity_evaluated':False,'semantic_strict_exit_code':0 if matches_quant else 1,'processing_exit_code':0,'cases':cases}
 (a.output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
 with (a.output/'rows.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rowdata[0]));w.writeheader();w.writerows(rowdata)
 print(json.dumps(result,indent=2));return result['semantic_strict_exit_code'] if a.strict else 0
if __name__=='__main__':raise SystemExit(main())
