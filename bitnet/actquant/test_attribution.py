#!/usr/bin/env python3
"""Synthetic CPU-only attribution controls; no native/device/GPU execution."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from analyze_q4 import EvidenceError, trace_evidence
from test_q4_contract import trace
from analyze_attribution import thermal_groups
from analyze_attribution_verify import split_trace, stock_abi
from run_phone import attribution_gate, attribution_protocol, PUBLIC_MODEL_SHA256, sha256
from run_iphone_attribution import plan, classify, classify_measurement
from make_attribution_gate import verify_run_binding


def qb_trace():
    text=trace().replace('q_route=Q4','q_route=QB').replace('route=q-custom','route=q-stock')
    from q4_contract import analysis_contract
    text=text.replace(analysis_contract()['entry'],'kernel_mul_mv_tq2_0_f32')
    lines=[]
    for line in text.splitlines():
        lines.append(line)
        if line.startswith('BITNET_ACTQUANT_Q4_GUARD '):
            lines.append('BITNET_Q_GUARD_PASS node=Qcur-0 checked=1 custom_encoded=0 bounds=1 disjoint=1 candidate_pipeline='+analysis_contract()['entry']+' rows=4 simd=1')
            lines.append('BITNET_Q_STOCK_ABI node=Qcur-0 guarded=1 pipeline=kernel_mul_mv_tq2_0_f32')
    return '\n'.join(lines)


class AttributionTests(unittest.TestCase):
    def test_QB_requires_full_guard_and_stock(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'trace.log';file.write_text(qb_trace());result=trace_evidence(file,'QB')
            self.assertEqual(result['guard_hits'],32);self.assertEqual(result['custom_dispatches'],0)
            self.assertEqual(result['QB_guard_passes'],32);self.assertEqual(result['QB_guarded_stock_calls'],32)

    def test_QB_missing_full_guard_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'trace.log';file.write_text(qb_trace().replace('custom_encoded=0','custom_encoded=1',1))
            with self.assertRaises(EvidenceError):trace_evidence(file,'QB')

    def test_one_context_trace_required(self):
        text='\n'.join('ACTQUANT_ATTRIBUTION_PATH_BEGIN id=1 route='+r+' capture=1 context=0x123\nbody\nACTQUANT_ATTRIBUTION_PATH_END id=1 route='+r+' synchronized=1' for r in ['stock','QB','Q4'])
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(set(split_trace(text,Path(temp),'291')),{'stock','QB','Q4'})
            with self.assertRaises(EvidenceError):split_trace(text.replace('context=0x123','context=0x124',1),Path(temp),'291')

    def test_stock_ABI_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'abi.log'
            file.write_text('\n'.join(f'BITNET_Q_STOCK_ABI node=Qcur-0 step={i} guarded=1 pipeline=kernel_mul_mv_tq2_0_f32' for i in range(1,33)))
            self.assertEqual(len(stock_abi(file,True)),32)
            file.write_text(file.read_text().replace('step=32','step=31'))
            with self.assertRaises(EvidenceError):stock_abi(file,True)

    def test_frozen_protocol(self):
        self.assertEqual((attribution_protocol('formal')['rounds'],attribution_protocol('formal')['pairs']),(3,21))
        self.assertEqual((attribution_protocol('diagnostic')['rounds'],attribution_protocol('diagnostic')['pairs']),(1,1))
        self.assertEqual((attribution_protocol('overhead')['rounds'],attribution_protocol('overhead')['pairs']),(3,7))
        self.assertEqual(attribution_protocol('formal',True)['comparisons'][-1],'QBQ4')
        with self.assertRaises(RuntimeError):attribution_protocol('unknown')

    def thermal_fixture(self):
        rows=[];paths={}
        for pair,state in enumerate(['nominal','serious']):
            rows.append({'round':'1','contrast':'SQB','pair':str(pair),'order':'AB','a_wall_ms':'10','b_wall_ms':'11'})
            for side in ['A','B']:paths[(1,'SQB',pair,side)]={'before':{'thermal':state,'low_power':False},'after':{'thermal':state,'low_power':False}}
        return rows,paths

    def test_thermal_states_never_pooled(self):
        rows,paths=self.thermal_fixture();result=thermal_groups(rows,paths)
        self.assertEqual(len(result['stable_strata']),2);self.assertIsNone(result['cross_thermal_pooled_statistics'])
        self.assertTrue(all(r['statistics']['pairs']==1 for r in result['stable_strata']))

    def test_transition_has_no_performance_statistics(self):
        rows,paths=self.thermal_fixture();paths[(1,'SQB',0,'A')]['after']['thermal']='fair'
        result=thermal_groups(rows[:1],paths)
        self.assertEqual(result['stable_strata'],[]);self.assertEqual(len(result['transition_or_unavailable_pairs']),1)

    def test_unknown_thermal_states_are_not_stable(self):
        for state in ['unknown','unavailable',None]:
            rows,paths=self.thermal_fixture()
            for side in ['A','B']:
                paths[(1,'SQB',0,side)]['before']['thermal']=state;paths[(1,'SQB',0,side)]['after']['thermal']=state
            result=thermal_groups(rows[:1],paths)
            self.assertEqual(result['stable_strata'],[]);self.assertEqual(len(result['transition_or_unavailable_pairs']),1)

    def test_gate_binds_actual_evidence_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);proof=root/'proof.json';proof.write_text('{}')
            gate=root/'gate.json';build={'framework_sha256':'framework','app_files_sha256':{},'shader_products_sha256':{'q-stock4x1.metallib':'library'}}
            gate.write_text(json.dumps({'schema':1,'scoped_correctness_pass':True,'official_source_model_output_pass':None,
                'historical_official_strict_exit_preserved':1,'framework_sha256':'framework','app_files_sha256':{},
                'model_sha256':PUBLIC_MODEL_SHA256,'q_library_sha256':'library','histories':{
                    'primary':{'same_process_scoped_pass':True,'tokens_sha256':'tokens'},
                    'independent':{'same_process_scoped_pass':True,'tokens_sha256':'different'}},'evidence_sha256':{str(proof):sha256(proof)}}))
            self.assertIs(attribution_gate(gate,build,'tokens')['scoped_correctness_gate_pass'],True)
            proof.write_text('{"modified":true}')
            with self.assertRaises(RuntimeError):attribution_gate(gate,build,'tokens')

    def test_verification_analysis_is_bound_to_current_build_and_actual_RUN(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'device/native').mkdir(parents=True)
            identity={'build_manifest_sha256':'current-build','framework_sha256':'framework','app_files_sha256':{'app':'current'},
                      'source_files_sha256':{'source':'current'},'package_sources_sha256':{'native':'current'},'object_sha256':{'object':'current'}}
            (root/'BUILD_IDENTITY.json').write_text(json.dumps(identity))
            native={'operation':'attribution','attribution_kind':'verify','execution_pass':True,'activation_quantization':True,
                    'one_model_and_context':True,'device':'Apple A16 GPU','backend':'MTL0','model_sha256':PUBLIC_MODEL_SHA256,
                    'tokens_sha256':'tokens','q_library_sha256':'library'}
            file=root/'device/native/RUN.json';file.write_text(json.dumps(native));digest=sha256(file)
            host={'execution_pass':True,'GPU_executed':True,'native_run_sha256':digest,**{k:0 for k in
                  ['native_process_exit_code','collection_exit_code','actual_process_exit_code','launch_command_exit_code']}}
            (root/'summary.json').write_text(json.dumps(host))
            (root/'device/result.json').write_text(json.dumps({'operation':'actquant','actquant_operation':'attribution',
                'process_exit_code':0,'native_process_exit_code':0,'execution_pass':True,'native_run_sha256':digest}))
            analysis={'native_run_sha256':digest};build={'shader_products_sha256':{'q-stock4x1.metallib':'library'}}
            self.assertEqual(verify_run_binding(root,analysis,identity,build,'tokens'),digest)
            stale={**identity,'build_manifest_sha256':'old-build'};(root/'BUILD_IDENTITY.json').write_text(json.dumps(stale))
            with self.assertRaises(RuntimeError):verify_run_binding(root,analysis,identity,build,'tokens')
            (root/'BUILD_IDENTITY.json').write_text(json.dumps(identity))
            with self.assertRaises(RuntimeError):verify_run_binding(root,{'native_run_sha256':'old-analysis'},identity,build,'tokens')
            file.write_text(json.dumps({**native,'tokens_sha256':'different'}))
            with self.assertRaises(RuntimeError):verify_run_binding(root,analysis,identity,build,'tokens')

    def plan_args(self,through='correctness',qualified=None):
        return SimpleNamespace(output=Path('/synthetic/output'),qualified_root=qualified,build_dir=Path('/synthetic/build'),device='synthetic',
            model=Path('/synthetic/model'),developer_dir=None,operator_fixtures=Path('/synthetic/fixtures'),
            tokens_primary=Path('/synthetic/primary'),tokens_independent=Path('/synthetic/independent'),through=through,direct_qb_q4=False)

    def test_stages_expose_correctness_before_timing(self):
        stages,_=plan(self.plan_args('formal'));names=[r[0] for r in stages]
        self.assertEqual(len(stages),19);self.assertEqual(names[:2],['correctness','cache-repeat'])
        self.assertLess(names.index('timing-gate'),names.index('diagnostic-primary'))
        self.assertLess(names.index('overhead-independent-analysis'),names.index('formal-primary'))

    def test_qualified_resume_skips_expensive_correctness(self):
        stages,_=plan(self.plan_args('formal',Path('/synthetic/qualified')))
        self.assertEqual(len(stages),12);self.assertEqual(stages[0][0],'diagnostic-primary')

    def test_missing_failed_stage_evidence_is_null(self):
        with tempfile.TemporaryDirectory() as temp:
            args=SimpleNamespace(output=Path(temp))
            self.assertIsNone(classify(args,'correctness'))
            (Path(temp)/'correctness').mkdir();(Path(temp)/'correctness/SUMMARY.json').write_text('{"scoped_qualification_pass":false}')
            self.assertIs(classify(args,'correctness'),False)

    def test_measurement_failure_is_distinct_from_correctness(self):
        with tempfile.TemporaryDirectory() as temp:
            args=SimpleNamespace(output=Path(temp));name='formal-primary-analysis'
            self.assertIsNone(classify_measurement(args,name))
            folder=Path(temp)/name;folder.mkdir();path=folder/'SUMMARY.json'
            path.write_text('{"measurement_integrity_pass":false}')
            self.assertIs(classify_measurement(args,name),False)
            self.assertIsNone(classify(args,name))
            path.write_text('{"measurement_integrity_pass":null}')
            self.assertIsNone(classify_measurement(args,name))
            path.write_text('{invalid')
            self.assertIsNone(classify_measurement(args,name))


if __name__=='__main__':unittest.main()
