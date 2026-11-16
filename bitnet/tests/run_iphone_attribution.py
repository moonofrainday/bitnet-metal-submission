#!/usr/bin/env python3
"""Plan staged A16 Q guard attribution. Device execution requires explicit --run.

Start with --through correctness; reuse its immutable --qualified-root for later
diagnostic/formal stages. Every intended command is written to PLAN.json for
manual one-stage-at-a-time execution. Raw data stays in the selected local output.
No isolated-kernel benchmark, new algorithm, official-model or quality gate is
implied. Existing isolated Q timing remains a separate optional experiment.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'bitnet/actquant'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_q_fix_phone import prepare, failed_stage_gate
from run_phone import attribution_gate, build_identity


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def plan(args):
    py=sys.executable;out=args.output;qualified=args.qualified_root or out
    common=['--build-dir',str(args.build_dir),'--device',args.device,'--model',str(args.model)]
    developer=['--developer-dir',str(args.developer_dir)] if args.developer_dir else []
    stage=[];correctness=qualified/'correctness';gate=qualified/'TIMING_GATE.json'
    if args.qualified_root is None:
        stage.append(('correctness',[py,str(ROOT/'bitnet/tests/run_q_fix_phone.py'),'--run',*common,
                     '--operator-fixtures',str(args.operator_fixtures),'--tokens-primary',str(args.tokens_primary),
                     '--tokens-independent',str(args.tokens_independent),'--output',str(correctness),*developer]))
        stage.append(('cache-repeat',[py,str(ROOT/'bitnet/ios/run.py'),'--mode','q-contract','--install',
                     '--build-dir',str(args.build_dir),'--device',args.device,'--output',str(out/'cache-repeat'),*developer]))
    native=[py,str(ROOT/'bitnet/actquant/run_phone.py'),'--run','--operation','attribution',*common,
            '--actquant','on','--workload','teacher','--capture-scope','q4-layer0','--reuse-model',
            '--operator-run',str(correctness/'operator'),'--operator-analysis',str(correctness/'operator-analysis/SUMMARY.json'),*developer]
    if args.qualified_root is None:
        for history,tokens in [('primary',args.tokens_primary),('independent',args.tokens_independent)]:
            run=out/('verify-'+history)
            stage.append(('verify-'+history,[*native,'--tokens',str(tokens),'--attribution-kind','verify',
                          '--capture','--trace','--output',str(run)]))
            stage.append(('verify-'+history+'-analysis',[py,str(ROOT/'bitnet/actquant/analyze_attribution_verify.py'),
                          '--run',str(run),'--log',str(run/'logs/launch.log'),
                          '--stock-null',str(correctness/(history+'-stock-plain')),
                          '--missing-library',str(correctness/(history+'-Q4-missing-library')),
                          '--missing-library-log',str(correctness/(history+'-Q4-missing-library/logs/launch.log')),
                          '--output',str(out/('verify-'+history+'-analysis'))]))
        stage.append(('timing-gate',[py,str(ROOT/'bitnet/actquant/make_attribution_gate.py'),
                      '--build-dir',str(args.build_dir),'--correctness',str(correctness),'--repeat-contract',str(out/'cache-repeat'),
                      '--primary-analysis',str(out/'verify-primary-analysis'),'--independent-analysis',str(out/'verify-independent-analysis'),
                      '--primary-run',str(out/'verify-primary'),'--independent-run',str(out/'verify-independent'),
                      '--tokens-primary',str(args.tokens_primary),'--tokens-independent',str(args.tokens_independent),'--output',str(gate)]))
    kinds=[] if args.through=='correctness' else ['diagnostic','overhead'] if args.through=='diagnostic' else ['diagnostic','overhead','formal']
    for kind in kinds:
        for history,tokens in [('primary',args.tokens_primary),('independent',args.tokens_independent)]:
            run=out/(kind+'-'+history);direct=['--direct-qb-q4'] if args.direct_qb_q4 and kind!='overhead' else []
            stage.append((kind+'-'+history,[*native,'--tokens',str(tokens),'--attribution-kind',kind,
                          '--attribution-gate',str(gate),'--output',str(run),*direct]))
            stage.append((kind+'-'+history+'-analysis',[py,str(ROOT/'bitnet/actquant/analyze_attribution.py'),
                          '--run',str(run),'--output',str(out/(kind+'-'+history+'-analysis')),
                          *(['--telemetry-only'] if kind=='diagnostic' else ['--gate',str(gate)])]))
    return stage,qualified


def classify(args,name):
    if name=='correctness':
        path=args.output/'correctness/SUMMARY.json';field='scoped_qualification_pass'
    elif name=='cache-repeat':
        path=args.output/'cache-repeat/summary.json';field='runtime_contract_pass'
    elif name.startswith('verify-') and name.endswith('-analysis'):
        path=args.output/name/'SUMMARY.json';field='scoped_stock_compatibility_pass'
    elif name=='timing-gate':path=args.output/'TIMING_GATE.json';field='scoped_correctness_pass'
    else:return None
    try:
        value=json.loads(path.read_text()).get(field)
        return False if value is False else None
    except (OSError,ValueError,AttributeError):return None


def classify_measurement(args,name):
    if not name.endswith('-analysis') or not name.startswith(('diagnostic-','overhead-','formal-')):return None
    try:
        value=json.loads((args.output/name/'SUMMARY.json').read_text()).get('measurement_integrity_pass')
        return False if value is False else None
    except (OSError,ValueError,AttributeError):return None


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
    mode.add_argument('--run',action='store_true');mode.add_argument('--plan',action='store_true')
    for name in ['build-dir','model','operator-fixtures','tokens-primary','tokens-independent','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',required=True);p.add_argument('--developer-dir',type=Path)
    p.add_argument('--through',choices=('correctness','diagnostic','formal'),default='correctness')
    p.add_argument('--qualified-root',type=Path,help='Reuse an already passed same-build TIMING_GATE.json and correctness/operator under this root')
    p.add_argument('--direct-qb-q4',action='store_true',help='Additional mixed encoding/shader/scheduling comparison')
    args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.expanduser().resolve())
    if args.output.exists():p.error('--output must be fresh')
    args.output.mkdir(parents=True)
    result={'scope':'Staged current ActQuant-ON A16 Q attribution','plan_only':not args.run,'through':args.through,
            'scoped_correctness_pass':None,'measurement_integrity_pass':None,'speedup_established':None,
            'official_source_model_output_pass':None,'full_model_FP64_pass':None,'model_quality_pass':None,
            'historical_official_strict_exit_preserved':1,'all_stages_completed':False,
            'device_commands_requested':args.run,'device_commands_executed':False,'collection_command_started':False,
            'errors':[],'stages':[]}
    try:
        result['local_provenance']=prepare(args)
        tasks,qualified=plan(args)
        if args.qualified_root is not None:
            build,_=build_identity(args.build_dir)
            for tokens in [args.tokens_primary,args.tokens_independent]:
                from run_phone import sha256
                attribution_gate(qualified/'TIMING_GATE.json',build,sha256(tokens))
            result.update(scoped_correctness_pass=True,reused_correctness=True)
        save(args.output/'PLAN.json',{'device_commands_executed':False,'stage_count':len(tasks),
                                     'stages':[{'name':name,'argv':command} for name,command in tasks],
                                     'timing_gate':str(qualified/'TIMING_GATE.json'),
                                     'optional_isolated_operator':'Not automated here; use existing q-verify/q-benchmark after new same-build correctness'})
        if not args.run:
            result.update(planning_pass=True,exit_code=0);return 0
        logdir=args.output/'logs';logdir.mkdir()
        for name,command in tasks:
            record={'name':name,'argv':command,'cwd':str(ROOT)};save(logdir/(name+'.command.json'),record)
            with (logdir/(name+'.log')).open('w') as log:
                process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if not name.endswith('-analysis') and name!='timing-gate':
                result.update(collection_command_started=True,device_commands_executed=None)
            record['exit_code']=process.returncode;save(logdir/(name+'.command.json'),record)
            (logdir/(name+'.exit-code')).write_text(str(process.returncode)+'\n')
            result['stages'].append({'name':name,'exit_code':process.returncode})
            save(args.output/'SUMMARY.json',result);print(name+': exit '+str(process.returncode),flush=True)
            if process.returncode:
                observed=classify(args,name)
                if observed is False:result['scoped_correctness_pass']=False
                if classify_measurement(args,name) is False:result['measurement_integrity_pass']=False
                result.update(failed_stage=name,execution_failed=True,exit_code=process.returncode if process.returncode>0 else 1)
                return result['exit_code']
            if name=='timing-gate':
                if json.loads((qualified/'TIMING_GATE.json').read_text()).get('scoped_correctness_pass') is not True:
                    raise ValueError('Timing gate child did not produce verified correctness')
                result['scoped_correctness_pass']=True
        gate=json.loads((qualified/'TIMING_GATE.json').read_text())
        if gate.get('scoped_correctness_pass') is not True:raise ValueError('Completed stages lack a correctness gate')
        result.update(scoped_correctness_pass=True,all_stages_completed=True,execution_failed=False,exit_code=0)
        if tasks:result['device_commands_executed']=True
        if args.through!='correctness':
            for kind in ['diagnostic','overhead']+(['formal'] if args.through=='formal' else []):
                for history in ['primary','independent']:
                    data=json.loads((args.output/(kind+'-'+history+'-analysis/SUMMARY.json')).read_text())
                    if data.get('measurement_integrity_pass') is not True:raise ValueError('Measurement evidence incomplete')
            result['measurement_integrity_pass']=True
        return 0
    except (OSError,RuntimeError,ValueError,KeyError,TypeError) as error:
        result['errors'].append(str(error));result.update(exit_code=2,execution_failed=args.run,all_stages_completed=False)
        return 2
    finally:
        save(args.output/'SUMMARY.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':raise SystemExit(main())
