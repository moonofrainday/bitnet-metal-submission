#!/usr/bin/env python3
"""Small synthetic CPU-only controls for Mac-specific provenance and statistics."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_mac_q_value import MODEL_SHA, relative, sha, validate_gate
from analyze_mac_q_value import mac_statistics, union
import run_mac_q_value


class MacValueTests(unittest.TestCase):
    def test_first_native_failure_preserves_unknown_GPU_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);state=root/'state.json';state.write_text('{}');output=root/'new-output'
            gate={'Q4':{},'histories':{'primary':{'tokens':'synthetic-tokens'}},'build_dir':str(root),
                  'runner':'synthetic-runner-never-executed','model':'synthetic-model'}
            argv=['run_mac_q_value.py','--run','--through','diagnostic','--state',str(state),'--output',str(output)]
            with patch.object(sys,'argv',argv),patch.object(run_mac_q_value,'build_gate',return_value=gate),\
                 patch.object(run_mac_q_value,'validate_gate',return_value=gate),\
                 patch.object(run_mac_q_value.subprocess,'run',return_value=SimpleNamespace(returncode=23)) as native:
                self.assertEqual(run_mac_q_value.main(),23)
            result=json.loads((output/'SUMMARY.json').read_text())
            self.assertEqual(native.call_count,1)
            self.assertTrue(result['GPU_execution_attempted'])
            self.assertIsNone(result['GPU_executed'])
            self.assertFalse(result['plan_only'])
            self.assertEqual(result['exit_code'],23)

    def test_interval_union_does_not_double_count_overlap(self):
        self.assertEqual(union([(1,3),(2,4),(5,6)]),4)
        self.assertEqual(union([]),0)

    def test_unavailable_Mac_state_is_explicitly_unqualified(self):
        row={'round':'1','contrast':'SQB','pair':'0','order':'AB','a_wall_ms':'10','b_wall_ms':'11'}
        paths={(1,'SQB',0,side):{'before':{'thermal':'unavailable','low_power':None},
                               'after':{'thermal':'unavailable','low_power':None}} for side in ('A','B')}
        result=mac_statistics([row],paths)
        self.assertIs(result['stable_thermal_benefit_established'],False)
        self.assertEqual(result['unavailable_state_pairs'],1)
        self.assertEqual(result['thermally_unqualified_statistics']['statistics']['paired_B_minus_A_median_ms'],1)
        self.assertEqual(result['known_thermal_strata']['stable_strata'],[])

    def test_known_nominal_and_serious_states_are_separate(self):
        rows=[];paths={}
        for pair,state in enumerate(('nominal','serious')):
            rows.append({'round':'1','contrast':'SQB','pair':str(pair),'order':'AB','a_wall_ms':'10','b_wall_ms':'9'})
            for side in ('A','B'):paths[(1,'SQB',pair,side)]={'before':{'thermal':state,'low_power':False},'after':{'thermal':state,'low_power':False}}
        result=mac_statistics(rows,paths)
        self.assertIsNone(result['thermally_unqualified_statistics'])
        self.assertEqual(len(result['known_thermal_strata']['stable_strata']),2)
        self.assertIsNone(result['known_thermal_strata']['cross_thermal_pooled_statistics'])

    def test_gate_rejects_changed_products_and_wrong_platform(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'bin').mkdir();runner=root/'bin/runner';runner.write_bytes(b'synthetic-not-executable')
            proof=root/'proof.json';proof.write_text('{}');tokens=[]
            for name in ('a','b'):
                file=root/(name+'.json');file.write_text('{}');tokens.append({'tokens':str(file),'tokens_sha256':sha(file),'same_process_scoped_pass':True})
            gate={'platform':'Mac','device':'Apple M2 Pro','scoped_correctness_pass':True,'model_sha256':MODEL_SHA,
                  'histories':dict(zip(('primary','independent'),tokens)),'runner':str(runner),'runner_sha256':sha(runner),
                  'build_dir':str(root),'products_sha256':{'bin/runner':sha(runner)},'evidence_sha256':{str(proof):sha(proof)}}
            file=root/'gate.json';file.write_text(json.dumps(gate));self.assertEqual(validate_gate(file)['platform'],'Mac')
            runner.write_bytes(b'changed')
            with self.assertRaises(ValueError):validate_gate(file)
            gate['platform']='A16';file.write_text(json.dumps(gate))
            with self.assertRaises(ValueError):validate_gate(file)

    def test_provenance_paths_do_not_escape_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'x').write_text('x')
            self.assertEqual(relative(root,'x'),root/'x')
            with self.assertRaises(ValueError):relative(root,'../outside')
            with self.assertRaises(ValueError):relative(root,str(root/'x'))


if __name__=='__main__':unittest.main()
