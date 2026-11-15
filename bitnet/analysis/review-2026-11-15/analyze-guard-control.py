#!/usr/bin/env python3
"""CPU reanalysis of local A16 guard artifacts. Publishes aggregate analysis only."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse, csv, hashlib, json, math, re, statistics, struct


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def hash_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load(path):
    return json.loads(path.read_text())


def percentile(values, p):
    values=sorted(values)
    at=(len(values)-1)*p
    lo=int(at)
    hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(at-lo)


def verify(root):
    device=root/'device'
    log=(root/'logs/launch.log').read_text()
    cfg=load(root/'config.json')
    completion=load(device/'result.json')
    require(cfg['routes']==['S','QB','Q4'] and cfg['layers']=='0' and cfg['q_readback'], 'Wrong verification scope')
    require(cfg['model_sha256']=='52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2', 'Wrong model')
    require(completion['device']=='Apple A16 GPU' and completion['iOS']=='27.2.1', 'Unexpected device/OS')
    require(completion['process_exit_code']==0, 'App did not complete')
    require('offloaded 31/31 layers to GPU' in log,'Missing real offload')
    require((root/'logs/launch.exit.txt').read_text().strip()=='0','Launch command failed')
    enc=defaultdict(Counter); done=defaultdict(Counter); guards=defaultdict(list); abis=defaultdict(list); psos=defaultdict(list)
    current=None;prefill=False; prefill_custom=0;prefill_guard=0
    for line in log.splitlines():
        f=dict(re.findall(r'(\w+)=(\[[^\]]*\]|[^\s]+)',line))
        if line.startswith('FFNK_PREFILL_BEGIN '): prefill=True
        if line.startswith('FFNK_PREFILL_END '): prefill=False
        if line.startswith('FFNK_PATH_BEGIN '):current=f['route']
        if line.startswith('FFNK_PATH_END '):current=None
        if line.startswith('BITNET_Q_GUARD_PASS '):
            prefill_guard+=prefill
            if current:guards[current].append(f)
        if prefill and line.startswith('BITNET_MULTILAYER_ENCODE ') and f.get('route')=='q-custom':prefill_custom+=1
        if not current:continue
        if line.startswith('BITNET_MULTILAYER_ENCODE '):
            enc[current][f['route']]+=1
            if f.get('node')=='Qcur-0' and f.get('route')=='q-stock' and f.get('n')=='1':psos[current].append(f['pso'])
        if line.startswith('BITNET_MULTILAYER_COMPLETE '):
            done[current][f['route']]+=1
            require(f['status']=='4' and f['completed']=='1','GPU completion failed')
        if line.startswith('BITNET_Q_STOCK_ABI '):abis[current].append(f)
    for route in cfg['routes']:
        require(enc[route]==done[route], 'Encoded/completed dispatch mismatch '+route)
        require(enc[route]['q-stock']==(928 if route=='Q4' else 960),'Stock Q count '+route)
        require(enc[route]['q-custom']==(32 if route=='Q4' else 0),'Custom Q count '+route)
        require(enc[route]['ffn-stock-down']==960,'Stock down count '+route)
    require(prefill_custom==prefill_guard==0,'Prefill changed')
    require(len(guards['S'])==len(guards['Q4'])==0 and len(guards['QB'])==32,'Wrong guard count')
    steps=list(map(str,range(1,33)))
    require([v['step'] for v in guards['QB']]==steps,'Missing guard steps')
    for f in guards['QB']:
        require(all(f[k]==v for k,v in {'checked':'1','custom_encoded':'0','bounds':'1','disjoint':'1','width':'32','rows':'4','simd':'1'}.items()),'Incomplete full guard evidence')
    keys=['node','step','pipeline','cached_pipeline','args_bytes','add','norm','grid','threads','smem','w','x','y','residual','norm_weight','slots']
    for route,flag in [('S','0'),('QB','1')]:
        require(len(abis[route])==len(psos[route])==32,'Missing actual stock ABI/PSO')
        require([v['step'] for v in abis[route]]==steps,'ABI steps missing')
        for f in abis[route]:
            require(all(f[k]==v for k,v in {'guarded':flag,'args_bytes':'120','add':'0','norm':'0','grid':'[320,1,1]','threads':'[32,2,1]','slots':'[0,1,2,3,4,5]'}.items()),'ABI changed')
    require([[f[k] for k in keys]for f in abis['S']]==[[f[k]for k in keys]for f in abis['QB']], 'Stock native ABI/bindings/offset/launch not identical')
    require(psos['S']==psos['QB'] and len(set(psos['S']))==1,'Actual compiled stock PSO not identical')
    require(set(psos['S'])-{'0x0','(nil)'},'Null pipeline')
    reference_files=None; captured_checks=0;captured_values=0
    for route in cfg['routes']:
        folder=device/('route-'+route)/'q-readback';m=load(folder/'manifest.json')
        expected={'weights/w.tq2':1689600}
        for step in range(1,33):
            expected.update({f'step-{step}/{role}.f32':10240 for role in ('x','y')})
            if step in (1,16,32):expected[f'step-{step}/l_out.f32']=10240
        require(set(m['files'])==set(expected) and len(expected)==68,'Bad captured coverage')
        values={}
        for relative,size in expected.items():
            data=(folder/relative).read_bytes()
            require(len(data)==size and hash_bytes(data)==m['files'][relative]['sha256'],'Raw readback/manifest mismatch')
            if relative.endswith('.f32'):
                require(all(math.isfinite(x[0])for x in struct.iter_unpack('<f',data)),'Nonfinite captured values')
                captured_values+=len(data)//4
            values[relative]=data
        if reference_files is None:reference_files=values
        else:
            for relative in expected:require(values[relative]==reference_files[relative], 'Raw S/'+route+' captured bytes differ '+relative)
            captured_checks+=len(expected)
    logit_checks=0
    for step in range(33):
        base=(device/'route-S/logits'/f'step-{step}.f32').read_bytes()
        require(len(base)==128256*4,'Wrong full-vocabulary extent')
        for route in cfg['routes']:
            data=(device/('route-'+route)/'logits'/f'step-{step}.f32').read_bytes()
            again=(device/('repeat-'+route)/'logits'/f'step-{step}.f32').read_bytes()
            require(data==base and data==again,'Logits/diagnostic neutrality changed')
            require(all(math.isfinite(x[0])for x in struct.iter_unpack('<f',data)),'Nonfinite logits')
            logit_checks+=1
    state=abis['S'][15]
    woffset=int(state['w'].split('+')[1]);xoffset=int(state['x'].split('+')[1]);yoffset=int(state['y'].split('+')[1])
    require(xoffset+10240<=yoffset or yoffset+10240<=xoffset,'X/Y offsets overlap')
    return {'execution_pass':True,'device':'iPhone 14 Pro / Apple A16 GPU','iOS':'27.2.1','model_sha256':cfg['model_sha256'],'history':'primary fixed27-token prompt /32 continuation IDs','guard_only_successes':32,'stock_Q_completed':{'S':960,'QB':960,'Q4':928},'custom_Q_completed':{'S':0,'QB':0,'Q4':32},'stock_down_completed_per_route':960,'prefill_guard_successes':prefill_guard,'prefill_custom_dispatches':prefill_custom,'same_actual_stock_PSO':True,'same_native_stock_ABI_bindings_offsets_launch_all32':True,'native_stock_ABI_bytes':120,'stock_grid':[320,1,1],'stock_threads':[32,2,1],'readback_files_per_route':68,'S_to_QB_and_Q4_file_comparisons':captured_checks,'finite_readback_F32_values_total':captured_values,'all32_pre_RoPE_W_X_Y_bitwise_equal':True,'steps1_16_32_residuals_bitwise_equal':True,'all33_full_logits_each_route_equal_to_stock_and_probe_off_repeat':True,'logit_vectors_checked':logit_checks,'readback_weight_sha256':hash_bytes(reference_files['weights/w.tq2']),'step16_X_sha256':hash_bytes(reference_files['step-16/x.f32']),'step16_Y_sha256':hash_bytes(reference_files['step-16/y.f32']),'numeric_fp64_pass':None,'model_semantic_fidelity_pass':None,'model_quality_tested':False,'strict_v1_exit_code':2,'note':'Matched guard-control acceptance; missing new FP64 checks retain strict V1 exit2, not full numerical acceptance.'}


def timing(root):
    device=root/'device';cfg=load(root/'config.json');execution=load(device/'execution.json');completion=load(device/'result.json')
    require(cfg['comparisons']==['AA','SQB','SQ4'] and cfg['rounds']==3 and cfg['pairs']==7 and cfg['warmups']==1,'Wrong short-screen protocol')
    require(completion['process_exit_code']==0 and execution['timing_clean'] is True,'Timing did not complete cleanly')
    require(execution['trace_during_timing'] is False and execution['tensor_readback_during_timing'] is False and execution['CPU_timers_during_timing'] is False,'Diagnostics during timing')
    rows=list(csv.DictReader((device/'samples.csv').open()));tokens=list(csv.DictReader((device/'token-times.csv').open()))
    require(len(rows)==63 and len(tokens)==4032,'Actual sample count mismatch')
    groups=[]
    for rnd in (1,2,3):
        for contrast in ('AA','SQB','SQ4'):
            rs=[r for r in rows if int(r['round'])==rnd and r['comparison']==contrast];require(len(rs)==7,'Missing direct pairs')
            a=[float(r['a_wall_ms'])for r in rs];b=[float(r['b_wall_ms'])for r in rs];d=[y-x for x,y in zip(a,b)]
            require(all(math.isfinite(x)and x>0 for x in a+b),'Nonfinite elapsed')
            for row,value in zip(rs,d):require(abs(float(row['b_minus_a_ms'])-value)<1e-6,'Paired delta mismatch')
            orders={}
            for order in ('AB','BA'):
                rr=[float(r['b_wall_ms'])-float(r['a_wall_ms'])for r in rs if r['order']==order]
                orders[order]={'pairs':len(rr),'paired_B_minus_A_p50_ms':statistics.median(rr)}
            groups.append({'round':rnd,'comparison':contrast,'pairs':7,'a_p50_ms':statistics.median(a),'a_p95_ms':percentile(a,.95),'b_p50_ms':statistics.median(b),'b_p95_ms':percentile(b,.95),'paired_B_minus_A_p50_ms':statistics.median(d),'paired_relative_B_minus_A_p50_percent':statistics.median([(y/x-1)*100 for x,y in zip(a,b)]),'B_wins':sum(y<x for x,y in zip(a,b)),'order_split':orders,'first_half_pair_delta_median_ms':statistics.median(d[:3]),'last_half_pair_delta_median_ms':statistics.median(d[-3:]),'thermal_states':sorted(set(r[k]for r in rs for k in ('a_thermal_before','a_thermal_after','b_thermal_before','b_thermal_after'))),'low_power_seen':any(r[k]=='1'for r in rs for k in ('a_low_power','b_low_power'))})
    token_groups=[]
    for rnd in (1,2,3):
        for contrast in ('AA','SQB','SQ4'):
            selected=[r for r in tokens if int(r['round'])==rnd and r['comparison']==contrast]
            grouped=defaultdict(dict)
            for row in selected:grouped[(int(row['pair']),int(row['step']))][row['path']]=float(row['wall_ms'])
            require(len(grouped)==7*32 and all(set(v)=={'A','B'}for v in grouped.values()),'Missing paired token samples')
            step_deltas={step:[grouped[(pair,step)]['B']-grouped[(pair,step)]['A']for pair in range(7)]for step in range(1,33)}
            medians=[statistics.median(step_deltas[step])for step in range(1,33)]
            token_groups.append({'round':rnd,'comparison':contrast,'per_step_median_B_minus_A_ms':medians,'steps_with_positive_paired_median':sum(x>0 for x in medians),'step1_paired_median_ms':medians[0],'steps2_to32_median_of_step_medians_ms':statistics.median(medians[1:]),'note':'Per-token synchronized wall intervals, not CPU instruction cost or isolated GPU timing.'})
    return {'metric':'32-token synchronized host decode wall, milliseconds; reset/prefill excluded','scope':'current QVAC public-checkpoint compatibility path, not restored Microsoft INT8 semantics','protocol':'short screening only: 1 warmup/route/phase;3 rounds;7 direct balanced pairs/contrast;AA/SQB/SQ4','sample_pairs':len(rows),'token_rows':len(tokens),'diagnostics_during_timing':False,'CPU_selection_or_encode_time_measured':False,'GPU_interval_measured':False,'rounds':groups,'per_token_summary':token_groups,'power_and_thermal_endpoints':[{**r,'before':{k:v for k,v in r['before'].items()if k!='uptime_s'},'after':{k:v for k,v in r['after'].items()if k!='uptime_s'}}for r in completion.get('round_states',[])],'caution':'Do not call this pure CPU guard cost, GPU kernel time, formal201-pair performance confirmation or model-quality result.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',type=Path,required=True)
    parser.add_argument('--timing',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result={'verification':verify(args.verify)}
    if args.timing:result['timing']=timing(args.timing)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
