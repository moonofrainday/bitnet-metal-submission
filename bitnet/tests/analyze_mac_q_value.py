#!/usr/bin/env python3
"""CPU-only Mac-scoped analysis of the existing Qcur-0 candidate.

Uses the shared raw-integrity verifier in telemetry-only mode. It never creates
an A16 gate or changes the phone analyzer. Mac statistics with unavailable native
thermal state are explicitly unqualified, not evidence of stable thermal benefit.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'bitnet/actquant'));sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_attribution import audit, stats, thermal_groups
from run_mac_q_value import read, save, sha, require, validate_gate


def union(intervals):
    if not intervals:return 0
    ordered=sorted(intervals);left,right=ordered[0];total=0
    for start,end in ordered[1:]:
        if start>right:total+=right-left;left,right=start,end
        else:right=max(right,end)
    return total+right-left


def mac_statistics(rows,paths):
    unavailable=[];other=[]
    for row in rows:
        key=(int(row['round']),row['contrast'],int(row['pair']))
        values=[paths[(*key,side)][end] for side in ('A','B') for end in ('before','after')]
        if all(v.get('thermal')=='unavailable' and v.get('low_power') is None for v in values):unavailable.append(row)
        else:other.append(row)
    result={'stable_thermal_benefit_established':False,'thermally_unqualified_statistics':None,
            'known_thermal_strata':thermal_groups(other,paths),'unavailable_state_pairs':len(unavailable)}
    if unavailable:
        order={key:stats([r for r in unavailable if r['order']==key]) for key in ('AB','BA') if any(r['order']==key for r in unavailable)}
        blocks=[]
        for i in range(3):
            chosen=unavailable[len(unavailable)*i//3:len(unavailable)*(i+1)//3]
            if chosen:blocks.append({'first_pair':int(chosen[0]['pair']),'last_pair':int(chosen[-1]['pair']),'statistics':stats(chosen)})
        result['thermally_unqualified_statistics']={'statistics':stats(unavailable),'order_groups':order,'time_blocks':blocks,
            'qualification':'Actual wall samples; native Mac thermal and low-power state unavailable. No stable-thermal or A16 conclusion.'}
    return result


def analyze(args):
    gate=validate_gate(args.gate);root=args.run.resolve();native=read(root/'RUN.json')
    require(native['device']=='Apple M2 Pro' and native['backend']=='MTL0' and native['model_sha256']==gate['model_sha256'] and
            native['q_library_sha256']==gate['Q4']['q_library_sha256'] and native['q_entry']==gate['Q4']['q_entry'] and
            native['q_kernel_contract_version']==gate['Q4']['q_kernel_contract_version'] and
            native['correctness_gate_sha256']==sha(args.gate),'Mac run/gate identity mismatch')
    require(native['tokens_sha256'] in {v['tokens_sha256'] for v in gate['histories'].values()},'Unqualified history')
    request=read(root/'REQUEST.json')
    require(Path(request['model']).resolve()==Path(gate['model']) and Path(request['q_library']).resolve()==Path(gate['Q4']['q_library']) and
            sha(request['tokens'])==native['tokens_sha256'] and request['correctness_gate_sha256']==sha(args.gate),'Actual native request is not gate-bound')
    checked=audit(SimpleNamespace(run=root,telemetry_only=True,gate=None))
    with (root/'samples.csv').open(newline='') as file:rows=list(csv.DictReader(file))
    paths={(v['round'],v['contrast'],v['pair'],v['side']):v for v in native['paths']}
    results=[]
    for round in range(1,native['rounds']+1):
        for contrast in native['comparisons']:
            chosen=[r for r in rows if int(r['round'])==round and r['contrast']==contrast]
            results.append({'round':round,'contrast':contrast,**mac_statistics(chosen,paths)})
    cpu_gpu=[]
    for path in native['paths']:
        if path['diagnostic_file'] is None:continue
        diag=read(root/path['diagnostic_file']);steps=[]
        for step in range(1,33):
            buffers=[r for r in diag['command_buffers'] if r['step']==step]
            cpu=[(r['encode_start_ns'],r['encode_end_ns']) for r in buffers]
            gpu=[(r['gpu_start_s'],r['gpu_end_s']) for r in buffers]
            host_work=union(cpu);gpu_work=union(gpu)
            steps.append({'step':step,'existing_command_buffers':len(buffers),'CPU_existing_CB_encoding_union_us':host_work/1000,
                'GPU_existing_CB_elapsed_sum_us':1e6*sum(b-a for a,b in gpu),'GPU_existing_CB_interval_union_us':1e6*gpu_work,
                'GPU_CB_overlap_us':1e6*(sum(b-a for a,b in gpu)-gpu_work),
                'CPU_CB_encoding_overlap_us':(sum(b-a for a,b in cpu)-host_work)/1000,
                'host_encode_begin_to_completion_callback_span_us':(max(r['completed_host_ns'] for r in buffers)-min(a for a,b in cpu))/1000})
        cpu_gpu.append({'route':path['route'],'round':path['round'],'contrast':path['contrast'],'pair':path['pair'],'side':path['side'],
                        'per_token':steps,'scope':'Instrumented existing CB windows. CPU and GPU durations/overlaps remain in separate clock domains.'})
    return {**checked,'scope':'Mac M2 Pro value of the existing layer0 Qcur-0 stock4x1 candidate',
        'platform':'Mac','phone_qualification':None,'telemetry_only':False,'performance_statistics':results,
        'CPU_GPU_overlap_diagnostics':cpu_gpu,'thermal_qualification':gate['thermal_qualification'],
        'stable_thermal_speedup_established':False,'speedup_established':None,
        'same_process_correctness_gate_sha256':sha(args.gate),'native_runner_sha256':gate['runner_sha256'],
        'target_coverage':{'target_Q_decode_calls_per_path':32,'all_layer_Q_decode_calls_per_path':960,
                         'time_fraction_of_Qcur0':None,'count_fraction_is_not_runtime_fraction':True},
        'interpretation':'AA calibrates repeated-run variation; SQB estimates full guard/lookup with the same stock PSO; SQ4 includes guard, custom binding and launch/shader/schedule. No other kernel ranking.',
        'critical_path_boundary':'Nested worker durations are not additive model latency; command-buffer elapsed is not isolated Q GPU time; no host/GPU absolute timestamp alignment is assumed.',
        'warmup_note':'Formal/diagnostic2 full replays per route. Overhead2 per route per metrics setting (OFF and ON).',
        'official_source_model_output_pass':None,'full_model_FP64_pass':None,'model_quality_pass':None,
        'historical_official_strict_exit_preserved':1}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','gate','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Analysis output must be fresh')
    args.output.mkdir(parents=True)
    try:result=analyze(args);code=0
    except (OSError,ValueError,KeyError,TypeError) as error:
        result={'measurement_integrity_pass':None,'error':str(error),'platform':'Mac','official_source_model_output_pass':None,
                'historical_official_strict_exit_preserved':1};code=2
    save(args.output/'SUMMARY.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('performance_statistics','diagnostics','CPU_GPU_overlap_diagnostics')},indent=2));return code


if __name__=='__main__':raise SystemExit(main())
