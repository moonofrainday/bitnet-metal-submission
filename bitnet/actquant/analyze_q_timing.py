#!/usr/bin/env python3
"""CPU-only audit of the legacy A16 Q operator timing protocol.

Consumes a host q-benchmark collection, its fresh-phone fixture, and the actual
ActQuant-ON Q4 graph capture. Does not execute a model, GPU, device or compiler.
Output contains derived statistics and relative evidence labels with hashes;
raw CSV, captures and private device paths stay local.

Exit 0: every audit check passes; 1: an observed audit failure; 2: missing,
malformed or invalid evidence/arguments. Existing output is never overwritten.
"""
import argparse
from collections import OrderedDict
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

from analyze_q4 import EvidenceError, load, manifest, native_root, payload, q_records, quantizer_check, sha
from prepare_fixtures import SOURCE_SHA
from q4_contract import analysis_contract, operator_q_contract

ROUNDS = (1, 2, 3)
COMPARISONS = ('AA', 'SQ', 'CQ')
PAIRS = 201
WARMUPS = 30
ROUTES = {'AA': ('S', 'S'), 'SQ': ('S', 'Q4'), 'CQ': ('C8', 'Q4')}


class AuditFailure(RuntimeError):
    pass


def require(value, text):
    if not value:
        raise AuditFailure(text)


def read_fixture_file(root, step, name, entry):
    path = root / ('step-' + str(step)) / name
    require(path.is_file() and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
            'Fixture file extent/hash mismatch: step-' + str(step) + '/' + name)
    return path


def quantile(values, p):
    return float(np.quantile(values, p, method='linear'))


def origin_control(fixture_root, fixture, candidate_root, candidate):
    require(fixture.get('actual_phone_capture_source') is True and fixture.get('activation_quantization') is True,
            'Fixture must identify actual ActQuant-ON physical-phone captures')
    require(fixture['model_sha256'] == candidate['model_sha256'] and candidate['device'] == 'Apple A16 GPU',
            'Fixture/candidate model or actual-device identity differs')
    require(fixture['prompt_ids'] == candidate['prompt_ids'] and fixture['continuation_ids'] == candidate['continuation_ids'],
            'Fixture/candidate teacher history differs')
    require(fixture['W_shape'] == [2560,2560] and fixture['X_shape'] == [2560,1] and
            fixture['weight_type'] == 'TQ2_0' and fixture['activation'] == fixture['output'] == 'F32',
            'Unexpected Q operator contract')
    records = q_records(candidate)
    quant = quantizer_check(candidate_root, candidate, records,
                            Path(__file__).resolve().parents[1] / 'reference/official-bitnet-096f25ae.py')
    require(quant['pass'] is True, 'Actual candidate quantizer differs from the pinned official F32 oracle')
    selected = {x['decode_step']: x for x in records}
    data_hashes = {}
    for step in (1,16,32):
        r = selected[step]
        require(r['activation']['name'] == 'actq_qkv_out-0' and r['activation']['op'] == 'ACT_QUANT',
                'Actual Q input is not the declared quantizer output')
        entries = fixture['fixtures'][str(step)]
        paths = {name: read_fixture_file(fixture_root,step,name,entries[name]) for name in ('w.tq2','x.f32','y.f32')}
        require(sha(paths['w.tq2']) == r['weight_sha256'] == fixture['source_weight_sha256'] and
                sha(paths['x.f32']) == r['input_sha256'] and sha(paths['y.f32']) == r['output_sha256'],
                'Fixture values differ from the fresh candidate actual W/X/Y')
        payload(candidate_root,r['weight_path'],1689600,r['weight_sha256'],np.uint8)
        payload(candidate_root,r['input_path'],2560,r['input_sha256'])
        payload(candidate_root,r['path'],2560,r['output_sha256'])
        for name,path in paths.items(): data_hashes['step-'+str(step)+'/'+name] = sha(path)
    require(data_hashes['step-16/x.f32'] == fixture['timing_step16_activation_sha256'],
            'Timing X differs from the fixture declared step16 input')
    return quant,data_hashes


def audit(args):
    run = args.performance_run.expanduser().resolve()
    fixture_root = args.fixture.expanduser().resolve()
    candidate_root = native_root(args.joint_candidate)
    perf = run / 'operator/performance'
    native = load(perf/'summary.json')
    host = load(run/'summary.json')
    fixture = load(fixture_root/'MANIFEST.json')
    contract = analysis_contract(args.contract_profile,args.legacy_shader_build)
    candidate = manifest(candidate_root,'Q4',contract=contract)
    quant,data_hashes = origin_control(fixture_root,fixture,candidate_root,candidate)
    with (perf/'samples.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == ['round','comparison','pair','order','first_route','second_route',
                                     'first_us','second_us','second_minus_first_us'], 'Unexpected CSV schema')
        rows = list(reader)
    groups = OrderedDict()
    for row in rows:
        key = (int(row['round']),row['comparison'])
        groups.setdefault(key,[]).append(row)
    expected_order = [(1,'AA'),(1,'SQ'),(1,'CQ'),(2,'SQ'),(2,'CQ'),(2,'AA'),(3,'CQ'),(3,'AA'),(3,'SQ')]
    checks = {
        'exactly1809pairs_3618timeddispatches': len(rows) == len(ROUNDS)*len(COMPARISONS)*PAIRS,
        'ninegroups_three_rounds_all_comparisons': set(groups) == {(r,c) for r in ROUNDS for c in COMPARISONS},
        'rotated_comparison_order': list(groups) == expected_order,
        'official_post_ActQuant_origin': quant['pass'],
        'native_current_kernel_contract_matches': contract.get('analysis_only_legacy',False) or native.get('q_kernel_contract') == operator_q_contract(),
        'all9_input_hashes_match_fixture': len(native['inputs']) == 9 and
            {str(Path(k).parts[-2]+'/'+Path(k).name):v for k,v in native['inputs'].items()} == data_hashes,
        'same_Q4_library_as_joint_graph': native['libraries']['Q4'] == candidate['q_library_sha256'] == fixture['libraries']['Q4']['sha256'],
        'all_candidate_library_hashes_match_fixture': native['libraries'] == {k:fixture['libraries'][k]['sha256'] for k in ('C8','Q4')},
        'single_dispatch_no_capture_readback': native['single_dispatch_per_command_buffer'] is True and
            native['capture'] is False and native['diagnostic_readback'] is False,
        'native_execution_and_fresh_qualification': host['execution_pass'] is True and host['completion']['process_exit_code'] == 0 and
            host['completion']['device'] == 'Apple A16 GPU' and host['q_benchmark_orchestration']['same_install_fresh_qualification'] is True and
            host['q_benchmark_orchestration']['native_gate_unchanged'] is True and host['qualified_case_count'] == 9,
    }
    require(all(checks.values()), 'Timing protocol, qualification or identity check failed')
    lookup = {(x['round'],x['comparison']):x for x in native['groups']}
    require(len(lookup) == len(native['groups']) == 9 and set(lookup) == set(groups), 'Native timing group coverage differs')
    records = []
    for (rr,comparison),entries in groups.items():
        require(len(entries) == PAIRS and [int(x['pair']) for x in entries] == list(range(PAIRS)),
                'Missing, duplicate or reordered timing pairs')
        require(all((x['first_route'],x['second_route']) == ROUTES[comparison] for x in entries), 'CSV route labels differ')
        a = np.array([float(x['first_us']) for x in entries]); b = np.array([float(x['second_us']) for x in entries])
        d = b-a; raw_d = np.array([float(x['second_minus_first_us']) for x in entries])
        require(np.isfinite(a).all() and np.isfinite(b).all() and (a>0).all() and (b>0).all(), 'Nonfinite/nonpositive GPU timing')
        require(np.allclose(d,raw_d,rtol=0,atol=1e-10), 'CSV paired differences differ from their actual samples')
        expected_orders = [('AB' if ((i%4 in (0,3)) != (rr%2==0)) else 'BA') for i in range(PAIRS)]
        require([x['order'] for x in entries] == expected_orders, 'Actual ABBA/round inversion order differs')
        calculated = {'first_p50_us':quantile(a,.5),'first_p95_us':quantile(a,.95),
                      'second_p50_us':quantile(b,.5),'second_p95_us':quantile(b,.95),
                      'paired_second_minus_first_p50_us':quantile(d,.5),'second_wins':int(np.count_nonzero(d<0))}
        n = lookup[(rr,comparison)]
        require(n['pairs'] == PAIRS and (n['first_route'],n['second_route']) == ROUTES[comparison] and
                all(abs(calculated[k]-n[k])<1e-8 for k in calculated), 'Native timing summary does not reproduce from raw CSV')
        order = {}
        for name in ('AB','BA'):
            mask = np.array([x['order']==name for x in entries])
            require(int(mask.sum()) in (100,101), 'Pair execution order is unbalanced')
            order[name] = {'pairs':int(mask.sum()),'first_p50_us':quantile(a[mask],.5),'second_p50_us':quantile(b[mask],.5),
                           'paired_second_minus_first_p50_us':quantile(d[mask],.5),'second_wins':int(np.count_nonzero(d[mask]<0)),
                           'median_paired_percent_change':quantile(100*d[mask]/a[mask],.5)}
        blocks = []
        for start in (0,67,134):
            sl=slice(start,start+67)
            blocks.append({'pairs':[start,start+66],'first_p50_us':quantile(a[sl],.5),'second_p50_us':quantile(b[sl],.5),
                           'paired_second_minus_first_p50_us':quantile(d[sl],.5),'second_wins':int(np.count_nonzero(d[sl]<0))})
        records.append({'round':rr,'comparison':comparison,'routes':list(ROUTES[comparison]),'pairs':PAIRS,**calculated,
                        'equal_pairs':int(np.count_nonzero(d==0)),'median_paired_percent_change':quantile(100*d/a,.5),
                        'ratio_of_marginal_p50_percent_change':100*(quantile(b,.5)/quantile(a,.5)-1),
                        'AB_BA':order,'time_blocks':blocks,'thermal_before':n['before'],'thermal_after':n['after']})
    checks['all_csv_native_statistics_reproduced'] = True
    sq=[x for x in records if x['comparison']=='SQ']; aa=[x for x in records if x['comparison']=='AA']; cq=[x for x in records if x['comparison']=='CQ']
    evidence=[('performance-run/operator/performance/samples.csv',perf/'samples.csv'),
              ('performance-run/operator/performance/summary.json',perf/'summary.json'),
              ('performance-run/summary.json',run/'summary.json'),('fixture/MANIFEST.json',fixture_root/'MANIFEST.json'),
              ('joint-candidate/device/native/RUN.json',candidate_root/'RUN.json')]
    return {'contract_profile':args.contract_profile,'Q4_entry':contract['entry'],'Q4_contract_version':contract['version'],
            'analysis_completed':True,'strict_exit_code':0,'audit_pass':True,'CPU_only_audit':True,'new_GPU_device_or_model_run':False,
            'checks':checks,'device':host['completion']['device'],'iOS':host['completion']['iOS'],'metric':native['metric'],
            'scope':'One Qcur-0 matvec dispatch on actual physical-phone post-ActQuant step16 W/X; quantizer preparation excluded',
            'protocol':{'rounds':3,'comparisons_per_round':list(COMPARISONS),'warmups_per_path_per_group':WARMUPS,
                        'warmup_evidence':'Fixed native benchmark protocol; CSV stores measured pairs only',
                        'pairs_per_group':PAIRS,'timed_pairs':len(rows),'timed_dispatches':len(rows)*2,
                        'single_dispatch_per_command_buffer':True,'capture':False,'tensor_readback':False,
                        'pair_execution_order':'ABBA repeating; invert in round2','comparison_order':'Rotate AA,SQ,CQ each round',
                        'first_second_route_columns':'Logical comparison routes; physical execution order is in order=AB/BA'},
            'inputs':{'model_sha256':fixture['model_sha256'],'W_sha256':fixture['source_weight_sha256'],
                      'X_step16_sha256':data_hashes['step-16/x.f32'],'selected_fixture_hashes':data_hashes,
                      'libraries':native['libraries'],'actual_input_quantizer_oracle':quant,'official_quantizer_source_sha256':SOURCE_SHA},
            'evidence':[{'file':label,'sha256':sha(path)} for label,path in evidence],
            'groups':records,'interpretation':{
                'Q4_vs_stock_paired_us':[x['paired_second_minus_first_p50_us'] for x in sq],
                'Q4_vs_stock_median_paired_percent_change':[x['median_paired_percent_change'] for x in sq],
                'Q4_vs_stock_wins':[x['second_wins'] for x in sq],
                'AA_paired_us':[x['paired_second_minus_first_p50_us'] for x in aa],
                'Q4_vs_C8_paired_us':[x['paired_second_minus_first_p50_us'] for x in cq],
                'Q4_vs_stock_negative_in_all_AB_BA_and_time_blocks':all(v['paired_second_minus_first_p50_us']<0 for x in sq for v in list(x['AB_BA'].values())+x['time_blocks']),
                'all_groups_nominal_lowpower_off':all(x['thermal_before']['thermal']==x['thermal_after']['thermal']==0 and
                    x['thermal_before']['low_power'] is False and x['thermal_after']['low_power'] is False for x in records),
                'C8_status':'Extra isolated research comparison; no new ActQuant real-graph/model acceptance',
                'semantic_full_model_acceptance':None,'end_to_end_performance_measured':False,'energy_measured':False,
                'sustained_thermal_study':False,'campaign_duration_from_group_uptime_seconds':records[-1]['thermal_after']['uptime_s']-records[0]['thermal_before']['uptime_s'],
                'limits':['GPU command-buffer elapsed includes dispatch overhead, not GPU instruction-only time.',
                          'No CPU encode or ActQuant preprocessing time is measured.',
                          'Three short rounds in one campaign are not independent thermal/cohort runs.',
                          'Median per-pair percentages and ratios of marginal medians are distinct estimators.',
                          'A/A is reported as a control, not subtracted as a universal correction.',
                          'No end-to-end throughput, model quality, energy or sustained-load conclusion.']}}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('performance-run','fixture','joint-candidate','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--contract-profile',choices=('canonical-v1','legacy-shared-v0'),default='canonical-v1')
    p.add_argument('--legacy-shader-build',type=Path)
    args=p.parse_args(argv);output=args.output.expanduser().resolve()
    if output.exists():
        print('Refuse existing analysis output',file=sys.stderr);return 2
    try:
        result=audit(args);code=0
    except AuditFailure as error:
        result={'analysis_completed':False,'audit_pass':False,'strict_exit_code':1,'CPU_only_audit':True,'error':str(error)};code=1
    except (OSError,KeyError,TypeError,ValueError,EvidenceError) as error:
        result={'analysis_completed':False,'audit_pass':False,'strict_exit_code':2,'CPU_only_audit':True,
                'error':type(error).__name__+': required evidence is missing or malformed'};code=2
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('analysis_completed','audit_pass','strict_exit_code','error','interpretation')},indent=2))
    return code


if __name__=='__main__':
    raise SystemExit(main())
