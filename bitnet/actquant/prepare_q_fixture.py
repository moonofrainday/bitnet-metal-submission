#!/usr/bin/env python3
"""CPU-only legacy Q operator fixtures from accepted fresh ActQuant-ON A16 captures.

No model, Metal, device, conversion or timing workload is executed. Raw fixture
files must remain outside the source checkout. The legacy Q runner separately
qualifies all three input cases before timing; C8 remains an extra research route.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from analyze_q4 import MODEL_SHA, WEIGHT_SHA, EvidenceError, decode_weight, load, manifest, metrics, native_root, payload, q_records, require, sha
from q4_contract import q4_library_identity

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True,help='Fresh completed host phone collection directory, stock-ON or Q4-ON')
    p.add_argument('--acceptance',type=Path,required=True,help='Its passing complete scoped joint-analysis SUMMARY.json')
    p.add_argument('--build-dir',type=Path,required=True,help='Same successful signed-build directory')
    p.add_argument('--output',type=Path,required=True,help='New local fixture directory outside the source checkout')
    args=p.parse_args(argv)
    output=args.output.expanduser().resolve();require(not output.exists(),'Refuse existing output')
    require(output!=ROOT and ROOT not in output.parents,'Raw fixtures must stay outside the source checkout')
    root=native_root(args.run);host=root.parent.parent
    require((host/'device/native').resolve()==root,'Physical-phone host collection is required; Mac captures are not substituted')
    run=load(root/'RUN.json');route=run['q_route'];require(route in ('stock','Q4'),'Unexpected source route')
    run=manifest(root,route);require(run['device']=='Apple A16 GPU' and run['activation_quantization'] is True,
                                  'Fresh actual A16 post-ActQuant inputs required')
    acceptance_path=args.acceptance.expanduser().resolve();acceptance=load(acceptance_path)
    require(acceptance['analysis_completed'] is True and acceptance['scoped_stock_compatibility_pass'] is True and
            acceptance['strict_scoped_exit_code']==0 and all(v is True for v in acceptance['checks'].values()),
            'Every required joint numerical, probe, runtime and fallback gate must pass first')
    require(acceptance['model_sha256']==MODEL_SHA and acceptance['weight_sha256']==WEIGHT_SHA and
            acceptance['q_library_sha256']==run['q_library_sha256'],'Accepted graph/library identity mismatch')
    collection=load(host/'summary.json');identity=load(host/'BUILD_IDENTITY.json');build_dir=args.build_dir.expanduser().resolve();build=load(build_dir/'BUILD.json')
    require(collection['execution_pass'] is True and collection['GPU_executed'] is True and
            collection['actual_process_exit_code']==0 and collection['native_run_sha256']==sha(root/'RUN.json'),
            'Actual phone collection/exit identity missing')
    require(build['build_completed'] is True and build['exit_code']==0 and build['unsigned'] is False and
            identity['build_manifest_sha256']==sha(build_dir/'BUILD.json') and identity['framework_sha256']==build['framework_sha256'] and
            identity['app_files_sha256']==build['app_files_sha256'],'Capture must use the selected signed app/framework')
    for file,digest in build['app_files_sha256'].items():require(sha(file)==digest,'Signed app binary changed')
    libraries={};shaders=Path(build['shaders']);qidentity=q4_library_identity(shaders/'q-stock4x1.metallib',shaders/'BUILD.json','iphoneos')
    require(qidentity['q_library_sha256']==run['q_library_sha256'] and build['shader_build_manifest_sha256']==sha(shaders/'BUILD.json'),
            'Q4 library differs from the accepted graph build')
    for key,filename in [('C8','q-c8x2.metallib'),('Q4','q-stock4x1.metallib')]:
        digest=sha(shaders/filename);require(build['shader_products_sha256'][filename]==digest,'Signed-build shader product changed')
        libraries[key]={'file':filename,'sha256':digest}
    records=q_records(run);by_step={r['decode_step']:r for r in records};accepted={r['step']:r for r in acceptance['source_input_hashes']}
    w=payload(root,records[0]['weight_path'],1689600,WEIGHT_SHA,np.uint8);decoded=decode_weight(w)
    generated={};validation=[]
    quant_records={tuple(r['input_positions']):r for r in run['checkpoints'] if r['kind']=='actquant' and r['node']=='actq_qkv_out-0'}
    for step in [1,16,32]:
        r=by_step[step];a=accepted[step];actual_hash=a['stock_output_sha256' if route=='stock' else 'Q4_output_sha256']
        require(r['weight_sha256']==a['weight_sha256']==WEIGHT_SHA and r['input_sha256']==a['input_sha256'] and
                r['output_sha256']==actual_hash,'Source W/X/Y changed after accepted analysis')
        x=payload(root,r['input_path'],2560,r['input_sha256']);y=payload(root,r['path'],2560,r['output_sha256'])
        quant=quant_records[tuple(r['input_positions'])]
        require(sha(root/quant['output_path'])==r['input_sha256'],'Actual Q input is not the captured ActQuant output')
        ref=np.einsum('rbk,bk->r',decoded,x.astype(np.float64).reshape(10,256),optimize=False);metric,_=metrics(y,ref)
        require(metric['pass'],'Actual Q output fails the unchanged independent FP64 bound')
        validation.append({'step':step,'input_position':r['input_positions'][0],**metric})
        generated[step]={'w.tq2':w.tobytes(),'x.f32':x.astype('<f4').tobytes(),'y.f32':y.astype('<f4').tobytes(),'reference.f64':ref.astype('<f8').tobytes()}
    data={'schema':1,'phase':'CPU preparation of accepted physical A16 ActQuant-ON Q0 fixtures; no new GPU execution',
          'GPU_executed_by_preparer':False,'actual_phone_capture_source':True,'activation_quantization':True,
          'activation_source':'Actual Qcur-0 src[1] = actq_qkv_out-0 / GGML_OP_ACT_QUANT; dequantized F32',
          'model_sha256':MODEL_SHA,'node':'Qcur-0','weight':'blk.0.attn_q.weight','W_shape':[2560,2560],
          'X_shape':[2560,1],'activation':'F32','output':'F32','weight_type':'TQ2_0',
          'prompt_ids':run['prompt_ids'],'continuation_ids':run['continuation_ids'],'fixtures':{},'libraries':libraries,
          'source_native_run_sha256':sha(root/'RUN.json'),'source_tokens_sha256':run['tokens_sha256'],
          'source_route':route,'source_device':'Apple A16 GPU','acceptance_sha256':sha(acceptance_path),
          'selected_build_manifest_sha256':sha(build_dir/'BUILD.json'),'source_weight_sha256':WEIGHT_SHA,
          'timing_step16_activation_sha256':by_step[16]['input_sha256'],'tolerance':{'abs':2e-4,'relative':2e-5},
          'source_validation':validation,'source_joint_scoped_pass':True,'whole_model_quality_acceptance_inferred':False,
          'whole_model_source_reference_acceptance_inferred':False,'timing_includes_ActQuant_preparation':False,
          'legacy_C8_scope':'Extra isolated arithmetic research only; not the accepted Q4 graph route'}
    output.mkdir(parents=True)
    for step,files in generated.items():
        folder=output/f'step-{step}';folder.mkdir();data['fixtures'][str(step)]={}
        for name,content in files.items():
            (folder/name).write_bytes(content);data['fixtures'][str(step)][name]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),
                'source_role':'Independent logical-K FP64 reference' if name=='reference.f64' else 'Actual fresh A16 model-graph W/X/Y'}
    (output/'MANIFEST.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'fixture_prepared':True,'GPU_executed_by_preparer':False,'model_sha256':MODEL_SHA,'weight_sha256':WEIGHT_SHA,
                      'step16_X_sha256':by_step[16]['input_sha256'],'all3_samples_pass_FP64':True,'output':str(output)},indent=2))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (EvidenceError,OSError,KeyError,ValueError,TypeError) as error:
        print(str(error),file=sys.stderr);raise SystemExit(2)
