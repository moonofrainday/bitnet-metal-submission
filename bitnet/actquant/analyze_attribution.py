#!/usr/bin/env python3
"""CPU-only attribution archive validation and thermally separated statistics.

Mac diagnostic archives require --telemetry-only and never produce performance
statistics. Exit0 means valid measurement evidence, not a speedup or quality pass.
Missing/invalid evidence returns2; an observed numerical gate failure is separate.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import numpy as np
from analyze_q4 import EvidenceError, MODEL_SHA, native_root, require, sha
from run_phone import attribution_protocol


def number(row, name):
    value=float(row[name]);require(math.isfinite(value),'Nonfinite CSV '+name);return value


def stats(rows):
    a=np.array([number(r,'a_wall_ms') for r in rows]);b=np.array([number(r,'b_wall_ms') for r in rows])
    difference=b-a
    return {'pairs':len(rows),'a_p50_ms':float(np.percentile(a,50)),'a_p95_ms':float(np.percentile(a,95)),
            'b_p50_ms':float(np.percentile(b,50)),'b_p95_ms':float(np.percentile(b,95)),
            'paired_B_minus_A_median_ms':float(np.median(difference)),
            'paired_relative_median_percent':float(np.median(100*difference/a)),
            'B_wins':int(np.count_nonzero(difference<0)),'ties':int(np.count_nonzero(difference==0)),
            'AB_pairs':sum(r['order']=='AB' for r in rows),'BA_pairs':sum(r['order']=='BA' for r in rows)}


def thermal_groups(rows, paths):
    groups=defaultdict(list);transitions=[]
    for row in rows:
        key=(int(row['round']),row['contrast'],int(row['pair']))
        a,b=paths[(*key,'A')],paths[(*key,'B')]
        thermals=[a['before']['thermal'],a['after']['thermal'],b['before']['thermal'],b['after']['thermal']]
        power=[a['before']['low_power'],a['after']['low_power'],b['before']['low_power'],b['after']['low_power']]
        if any(v not in ('nominal','fair','serious','critical') for v in thermals) or len(set(thermals))!=1 or any(type(v) is not bool for v in power) or len(set(power))!=1:
            transitions.append({'round':key[0],'contrast':key[1],'pair':key[2],'thermal':thermals,'low_power':power})
        else:
            groups[(thermals[0],power[0])].append(row)
    results=[]
    for (thermal,power),data in groups.items():
        parts={'thermal':thermal,'low_power':power,'statistics':stats(data),'order_groups':{},'time_blocks':[]}
        for order in ('AB','BA'):
            chosen=[r for r in data if r['order']==order]
            if chosen:parts['order_groups'][order]=stats(chosen)
        for indices in np.array_split(np.arange(len(data)),3):
            chosen=[data[int(i)] for i in indices]
            if chosen:parts['time_blocks'].append({'first_pair':int(chosen[0]['pair']),'last_pair':int(chosen[-1]['pair']),'statistics':stats(chosen)})
        results.append(parts)
    return {'stable_strata':results,'transition_or_unavailable_pairs':transitions,
            'cross_thermal_pooled_statistics':None}


def diagnostics(path, route):
    record=json.loads(path.read_text())
    require(record['schema']==1 and record['complete'] is True and record['dropped_records']==0 and
            record['command_buffer_drops']==0 and record['pending_callback_handlers']==0 and record['callback_wait_timed_out'] is False,
            'Incomplete diagnostic collection')
    require(record['clock']=='CLOCK_UPTIME_RAW' and record['units']=='nanoseconds' and record['calibration']['pairs']==10000,
            'Unknown CPU diagnostic clock/protocol')
    tasks=record['tasks'];calls=record['q_calls'];buffers=record['command_buffers']
    require(tasks and buffers and len(calls)==32 and [int(r['step']) for r in calls]==list(range(1,33)),
            'Missing all32 actual layer0 Q calls')
    require(all(t['mode']==route and 1<=t['step']<=32 for t in tasks),'Diagnostic route/step mismatch')
    fields=('native_encode_calls','native_encode_ns','matvec_calls','q_identity_calls','outer_eligible','guard_attempts','guard_passes',
            'guard_rejects','custom_dispatches','stock_target_dispatches','guard_ns','guard_tensor_buffer_ns','getter_ns','native_target_encode_ns')
    total={key:sum(t[key] for t in tasks) for key in fields}
    attempts=32 if route!='S' else 0;custom=32 if route=='Q4' else 0
    require(total['q_identity_calls']==960 and total['outer_eligible']==32 and total['guard_attempts']==total['guard_passes']==attempts and
            total['guard_rejects']==0 and total['custom_dispatches']==custom and total['stock_target_dispatches']==32-custom,
            'Actual diagnostic guard/dispatch counts differ')
    require(sum(v['guard_attempt'] for v in calls)==sum(v['guard_pass'] for v in calls)==attempts and
            sum(v['custom'] for v in calls)==custom and sum(v['stock'] for v in calls)==32-custom,
            'Per-call and aggregate counts differ')
    families=defaultdict(lambda:{'encode_calls':0,'raw_worker_encode_ns':0})
    for task in tasks:
        require(task['slice_start']<=task['slice_end'] and task['native_encode_ns']>=0,'Bad encode interval')
        for family in task['operator_family_cpu']:
            families[family['family']]['encode_calls']+=family['encode_calls'];families[family['family']]['raw_worker_encode_ns']+=family['encode_ns']
    token_buffers=[]
    for step in range(1,33):
        selected=[r for r in buffers if r['step']==step];require(selected,'Missing per-token existing command buffers')
        gpu=[]
        for row in selected:
            require(row['completed'] is True and row['status']==4 and
                    row['encode_start_ns']<=row['encode_end_ns'] and row['commit_start_ns']<=row['commit_end_ns'],
                    'Incomplete/reversed command buffer interval')
            start,end=row['gpu_start_s'],row['gpu_end_s']
            require(type(start) in (int,float) and type(end) in (int,float) and math.isfinite(start) and math.isfinite(end) and 0<start<=end,
                    'Unavailable/invalid GPU command-buffer timing')
            gpu.append((start,end))
        gpu.sort();union=0.0;left,right=gpu[0]
        for start,end in gpu[1:]:
            if start>right:union+=right-left;left,right=start,end
            else:right=max(right,end)
        union+=right-left
        token_buffers.append({'step':step,'existing_command_buffers':len(selected),
                              'GPU_command_buffer_elapsed_us':[1e6*(r['gpu_end_s']-r['gpu_start_s']) for r in selected],
                              'GPU_interval_union_us':1e6*union,
                              'host_encode_begin_to_completion_callback_span_us':
                              (max(r['completed_host_ns'] for r in selected)-min(r['encode_start_ns'] for r in selected))/1000})
    return {'pass':True,'route':route,'tasks':len(tasks),'actual_layer0_Q_calls':len(calls),'totals':total,
            'operator_family_CPU_work':dict(families),'clock_calibration':record['calibration'],'per_token_command_buffers':token_buffers,
            'raw_sha256':sha(path),'scope':'Existing command-buffer intervals, diagnostic instrumentation enabled; no per-shader GPU attribution',
            'nonadditive_warning':'Nested/concurrent worker times are not additive critical-path time; GPU/host timestamps have distinct domains',
            'unavailable':['GPU ALU counters','GPU occupancy','GPU bandwidth','individual operator GPU elapsed']}


def audit(args):
    root=native_root(args.run);native=json.loads((root/'RUN.json').read_text());kind=native['attribution_kind']
    require(kind in ('formal','diagnostic','overhead') and native['operation']=='attribution' and native['execution_pass'] is True and
            native['activation_quantization'] is True and native['one_model_and_context'] is True and native['model_sha256']==MODEL_SHA,
            'Not a complete ActQuant attribution archive')
    require(args.telemetry_only or native['device']=='Apple A16 GPU','Mac data requires --telemetry-only; no Mac performance report')
    if not args.telemetry_only:
        require(args.gate is not None,'Performance analysis requires the same-build correctness gate')
        gate=json.loads(args.gate.read_text())
        require(sha(args.gate)==native['correctness_gate_sha256'] and gate['scoped_correctness_pass'] is True and
                gate['model_sha256']==MODEL_SHA and gate['q_library_sha256']==native['q_library_sha256'] and
                native['tokens_sha256'] in {r['tokens_sha256'] for r in gate['histories'].values()},'Performance gate identity differs')
        for name,digest in gate['evidence_sha256'].items():require(sha(name)==digest,'Correctness evidence changed after measurement')
    protocol=attribution_protocol(kind,len(native['comparisons'])==4)
    expected_contrasts=['S_METRICS','QB_METRICS','Q4_METRICS'] if kind=='overhead' else protocol['comparisons']
    require(native['comparisons']==expected_contrasts,'Unexpected comparison scope')
    require(native['rounds']==protocol['rounds'] and native['pairs']==protocol['pairs'] and native['pipeline_warmup_replays_per_route']==2 and
            native['new_synchronization_points']==0 and native['capture_enabled'] is False and native['trace_enabled'] is False,
            'Changed timing protocol/synchronization/capture')
    with (root/'samples.csv').open(newline='') as file:samples=list(csv.DictReader(file))
    with (root/'token-times.csv').open(newline='') as file:tokens=list(csv.DictReader(file))
    require(len(samples)==native['sample_pairs']==protocol['rounds']*protocol['pairs']*len(native['comparisons']) and
            len(tokens)==native['token_rows']==len(samples)*64 and len(native['paths'])==len(samples)*2,'Raw coverage differs')
    path_map={(r['round'],r['contrast'],r['pair'],r['side']):r for r in native['paths']}
    require(len(path_map)==len(native['paths']) and len({r['final_logits_sha256'] for r in native['paths']})==1,'Path identity/output neutrality differs')
    token_map=defaultdict(list)
    for row in tokens:token_map[(int(row['round']),row['contrast'],int(row['pair']),row['path'])].append(row)
    seen=set();expected_schedule=[]
    for round in range(1,protocol['rounds']+1):
        contrasts=native['comparisons'];shift=(round-1)%len(contrasts)
        for position,contrast in enumerate(contrasts[shift:]+contrasts[:shift],1):expected_schedule.append((round,(round-1)*len(contrasts)+position,contrast))
    require([(r['round'],r['block'],r['contrast']) for r in native['schedule']]==expected_schedule,'Contrast rotation differs')
    for row in samples:
        round,pair,block=int(row['round']),int(row['pair']),int(row['block']);key=(round,row['contrast'],pair)
        require(key not in seen and 0<=pair<protocol['pairs'] and (round,block,row['contrast']) in expected_schedule,'Duplicate/unexpected sample')
        seen.add(key);ab=((pair%4 in (0,3)) != (round%2==0));require(row['order']==('AB' if ab else 'BA'),'Unbalanced order schedule')
        am=kind=='diagnostic';bm=kind!='formal'
        require(int(row['a_metrics'])==int(am) and int(row['b_metrics'])==int(bm),'Instrumentation mixed into wrong campaign')
        routes={'AA':('S','S'),'SQB':('S','QB'),'SQ4':('S','Q4'),'QBQ4':('QB','Q4'),
                'S_METRICS':('S','S'),'QB_METRICS':('QB','QB'),'Q4_METRICS':('Q4','Q4')}
        require((row['a_route'],row['b_route'])==routes[row['contrast']],'Wrong route in comparison')
        for side in ('A','B'):
            prefix=side.lower();begin,end=int(row[prefix+'_begin_ns']),int(row[prefix+'_end_ns']);wall=number(row,prefix+'_wall_ms')
            require(0<begin<end and abs((end-begin)/1e6-wall)<=1e-9*max(1,wall),'Wall boundary mismatch')
            path=path_map[(*key,side)];require(path['route']==row[prefix+'_route'] and path['metrics']==(am if side=='A' else bm),'Path metadata differs')
            records=token_map[(*key,side)];require([int(r['step']) for r in records]==list(range(1,33)),'Incomplete token history')
            previous=begin
            for token in records:
                require(token['route']==path['route'] and int(token['metrics'])==int(path['metrics']),'Token route/instrumentation differs')
                tb,te=int(token['begin_ns']),int(token['end_ns']);require(previous<=tb<te<=end,'Token boundary mismatch');previous=te
                require(abs((te-tb)/1e6-number(token,'wall_ms'))<=1e-9,'Token elapsed mismatch')
        require(int(row['a_end_ns'])<=int(row['b_begin_ns']) if ab else int(row['b_end_ns'])<=int(row['a_begin_ns']),'Recorded execution order differs')
        require(abs(number(row,'b_minus_a_ms')-(number(row,'b_wall_ms')-number(row,'a_wall_ms')))<=1e-9,'Paired difference mismatch')
    metrics=[]
    for path in native['paths']:
        if path['diagnostic_file'] is not None:
            file=(root/path['diagnostic_file']).resolve();require(file.is_relative_to(root.resolve()),'Diagnostic path escapes run')
            data=diagnostics(file,path['route']);data.update(round=path['round'],contrast=path['contrast'],pair=path['pair'],side=path['side']);metrics.append(data)
    require(len(metrics)==(0 if kind=='formal' else len(samples)*2 if kind=='diagnostic' else len(samples)), 'Missing diagnostic path records')
    comparison=[]
    if not args.telemetry_only:
        for round,_,contrast in expected_schedule:
            chosen=[r for r in samples if int(r['round'])==round and r['contrast']==contrast]
            comparison.append({'round':round,'contrast':contrast,**thermal_groups(chosen,path_map)})
    return {'measurement_integrity_pass':True,'analysis_completed':True,'scope':'Fixed32-token decode wall; reset/prefill excluded',
            'attribution_kind':kind,'device':native['device'],'pairs':len(samples),'token_rows':len(tokens),'formal_wall_campaign':kind=='formal',
            'telemetry_only':args.telemetry_only,'performance_statistics':None if args.telemetry_only else comparison,
            'diagnostics':metrics,'speedup_established':None,'model_quality_pass':None,'official_source_model_output_pass':None,
            'historical_official_strict_exit_preserved':1,'new_synchronization_points':0,
            'guard_attribution_boundary':'SQB includes full guard and candidate lookup before stock; SQ4/QBQ4 include encoding, binding, shader and scheduling changes',
            'diagnostic_boundary':'CPU/GPU command-buffer intervals are instrumented observations; overhead campaign is separate; no hardware bottleneck attribution',
            'raw_hashes':{'RUN.json':sha(root/'RUN.json'),'samples.csv':sha(root/'samples.csv'),'token-times.csv':sha(root/'token-times.csv')}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--telemetry-only',action='store_true');p.add_argument('--gate',type=Path);args=p.parse_args()
    if args.output.exists():p.error('Output must be fresh')
    args.output.mkdir(parents=True)
    try:result=audit(args);code=0
    except (EvidenceError,OSError,ValueError,KeyError,TypeError) as error:
        result={'analysis_completed':False,'measurement_integrity_pass':None,'error':str(error),'model_quality_pass':None,
                'official_source_model_output_pass':None,'historical_official_strict_exit_preserved':1};code=2
    (args.output/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('diagnostics','performance_statistics')},indent=2));return code


if __name__=='__main__':raise SystemExit(main())
