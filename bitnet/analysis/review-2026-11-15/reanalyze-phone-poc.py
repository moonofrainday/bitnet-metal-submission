#!/usr/bin/env python3
"""CPU-only independent reanalysis of frozen physical-phone binary readbacks.

All raw data stay local. All 38,400 GPU values are compared with a newly
decoded FP64 logical-K/math.fsum oracle. Canary and GPU-completion facts are
cross-checked against native records/console, not inferred from float values.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, math, sys

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--analyzer',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    spec=importlib.util.spec_from_file_location('operator_analysis',args.analyzer)
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    native=json.loads((args.input/'operator/summary.json').read_text())
    manifest=json.loads((args.input/'staged-fixture/MANIFEST.json').read_text())
    build=json.loads((args.input/'BUILD_IDENTITY.json').read_text())
    launch=(args.input/'logs/launch.log').read_text()
    manifest_cases={c['id']:c for c in manifest['cases']}
    summary={'CPU_only':True,'GPU_executed_by_analyzer':False,
        'scope':'Five standalone M2560 K2560 N1 cases; all 38400 GPU values',
        'reference':'Independently logical-K packed decode, struct FP16 scale, math.fsum of FP64 products for every row',
        'tolerance':{'abs':a.ATOL,'relative':a.RTOL},'cases':[],'failures':[],
        'model_output_pass':None,'model_semantics_pass':None,
        'formal_performance_run':False,'canary_reanalysis_scope':'Native GPU-completion/NaN-write/canary witness and console cross-check; raw guard bytes are not dumped'}
    rows=[]
    for n in native['cases']:
        ident=n['id'];p=args.input/'operator'/ident
        weight=(p/'W.tq2_0').read_bytes();_,x=a.vector(p/'X.f32','f')
        _,oldref=a.vector(p/'reference.f64','d')
        oracle=tuple(a.oracle_row(weight,x,i) for i in range(a.M))
        outputs={r:a.vector(p/(r+'.f32'),'f') for r in a.ROUTES}
        checks={'weight_contract':a.validate_weights(weight)['pass'],
            'finite_x':all(map(math.isfinite,x)),
            'recorded_input_hashes':sha(p/'W.tq2_0')==n['weights_sha256'] and sha(p/'X.f32')==n['activation_sha256'],
            'native_reference_vs_fsum':all(abs(v-r)<=a.ORACLE_ATOL+a.ORACLE_RTOL*abs(r) for v,r in zip(oldref,oracle)),
            'immutable_inputs_witness':n['input_immutable_pass'] is True,
            'console_completed_routes':all(f'ORIGINAL_POC_GPU_COMPLETE case={ident} route={r}' in launch for r in a.ROUTES)}
        if ident.startswith('real-step-'):
            checks['prepared_input_hashes']=all(sha(p/f)==manifest_cases[ident][key] for f,key in [('W.tq2_0','weights_sha256'),('X.f32','activation_sha256')])
        records={d['route']:d for d in n['dispatches']}
        checks['same_w_x_handles']=len({d['weight_handle'] for d in records.values()})==1 and len({d['activation_handle'] for d in records.values()})==1
        checks['separate_outputs']=len({d['output_handle'] for d in records.values()})==3
        checks['dispatch_route_count']=len(n['dispatches'])==3 and set(records)==set(a.ROUTES)
        native_refdiff=max(abs(r-o) for r,o in zip(oldref,oracle))
        case={'id':ident,'W_sha256':sha(p/'W.tq2_0'),'X_sha256':sha(p/'X.f32'),
            'native_reference_vs_new_oracle_max_abs':native_refdiff,'checks':checks,'routes':{}}
        for route in a.ROUTES:
            d=records[route];g=1 if route=='Q4' else 2
            dsymbol={'S':'kernel_mul_mv_tq2_0_f32','PoC':'tq2_0_matvec_poc','Q4':'tq2_0_matvec_poc_c8x2'}[route]
            checks[route+'_dispatch_witness']=(d['command_status']==4 and d['completed'] is True
                and d['canary_pass'] is True and d['canary_corrupt_bytes']==0 and d['coverage_pass'] is True
                and d['finite_rows']==2560 and d['input_immutable_pass'] is True
                and d['kernel_symbol']==dsymbol and d['encoded_pipeline_name']==dsymbol
                and d['grid']==[2560//(4*g),1,1] and d['threads']==[32,g,1]
                and d['buffer_offsets']==[0,0,128] and d['timing_measured'] is False)
            if route!='S': checks[route+'_library_identity']=d['library_sha256']==manifest['libraries'][route]['sha256']
            m=a.metrics(outputs[route][1],oracle)
            bitdiff=sum(outputs[route][0][i*4:i*4+4]!=outputs['S'][0][i*4:i*4+4] for i in range(a.M))
            stockm=a.metrics(outputs[route][1],outputs['S'][1])
            m['stock_bitwise_diff_rows']=bitdiff;m['stock_max_abs_diff']=stockm['max_abs_error'];m['stock_tolerance_failing_rows']=stockm['failing_rows']
            case['routes'][route]=m
            rows.append({'case':ident,'route':route,**{k:m[k] for k in ['max_abs_error','failing_rows','nonfinite_rows','relative_l2','stock_bitwise_diff_rows','stock_max_abs_diff','stock_tolerance_failing_rows']}})
            if not m['pass'] or not stockm['pass']:summary['failures'].append(ident+':'+route+':numerics')
        if ident=='known-answer':
            expected=tuple(1.375 if i==0 else -1.25 if i==2559 else 0.0 for i in range(2560))
            checks['exact_hand_known']=oracle==expected and all(outputs[r][1]==expected for r in a.ROUTES)
        summary['failures'] += [ident+':'+k for k,v in checks.items() if v is not True]
        summary['cases'].append(case)
        print(ident,'recomputed all 7680 GPU values',flush=True)
    images=native['loaded_ggml_images']
    framework=manifest['embedded_framework_sha256']
    summary['runtime_framework_verified']=len(images)==1 and images[0]['actual_embedded_framework_verified'] is True and images[0]['sha256']==framework and framework in build['app_files_sha256'].values()
    summary['command_completion_count']=native['GPU_command_buffers_completed']
    summary['console_completion_count']=launch.count('ORIGINAL_POC_GPU_COMPLETE case=')
    if not summary['runtime_framework_verified']:summary['failures'].append('signed_embedded_framework_identity')
    if summary['command_completion_count']!=15 or summary['console_completion_count']!=15:summary['failures'].append('completion_count')
    summary['local_numerical_exit_code']=1 if summary['failures'] else 0
    summary['global_strict_v1_exit_code']=2
    summary['global_strict_v1_reason']='Whole-model output and model-semantics acceptance not measured by standalone sample-I/O test'
    summary['overall_pass']=not summary['failures']
    (args.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (args.output/'RESULTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:summary[k] for k in ['local_numerical_exit_code','global_strict_v1_exit_code','overall_pass','failures']}))
    return summary['local_numerical_exit_code']

if __name__=='__main__':sys.exit(main())
