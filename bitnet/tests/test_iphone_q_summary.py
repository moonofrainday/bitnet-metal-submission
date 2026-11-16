"""Synthetic CPU-only regression controls; SUMMARIZER_SOURCE optionally overrides the adjacent source."""
from pathlib import Path
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest

SOURCE=Path(os.environ.get("SUMMARIZER_SOURCE", Path(__file__).with_name("summarize_iphone_q_attribution.py")))
spec=importlib.util.spec_from_file_location('portable_summary_under_test',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class SummaryControls(unittest.TestCase):
    def test_observed_false_is_correctness_rejection(self):
        with self.assertRaises(module.RequiredCorrectnessFailure):module.correct(False,'synthetic gate')

    def test_missing_is_not_numerical_false(self):
        try:module.correct(None,'synthetic missing')
        except module.RequiredCorrectnessFailure:self.fail('Missing evidence misclassified')
        except ValueError:pass
        else:self.fail('Missing evidence accepted')

    def test_thermal_mapping_and_unknown(self):
        self.assertEqual(module.stable_power({'thermal':1})['thermal'],'fair')
        self.assertEqual(module.stable_power({'thermal':2})['thermal'],'serious')
        self.assertEqual(module.stable_power({'thermal':None})['thermal'],'unavailable')
        self.assertEqual(module.stable_power({'thermal':True})['thermal'],'unavailable')

    def test_unknown_cannot_be_stable_stratum(self):
        with self.assertRaises(ValueError):
            module.compact_thermal_round({'round':1,'contrast':'SQ4','stable_strata':[{'thermal':'unknown','low_power':False}],'transition_or_unavailable_pairs':[]})

    def test_unknown_pairs_not_pooled(self):
        d=module.compact_thermal_round({'round':1,'contrast':'SQ4','stable_strata':[],'transition_or_unavailable_pairs':[{},{}]})
        self.assertEqual(d['transition_or_unavailable_pairs'],2)
        self.assertIsNone(d['cross_thermal_pooled_statistics'])

    def call_main(self,root,out):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            code=module.main(['--experiment',str(root),'--output',str(out)])
        return code,json.loads((out/'SUMMARY.json').read_text())

    def test_missing_complete_marker_exit2_private_paths_absent(self):
        with tempfile.TemporaryDirectory() as text:
            root=Path(text)/'missing';root.mkdir();code,d=self.call_main(root,Path(text)/'out')
            self.assertEqual(code,2);self.assertIsNone(d['scoped_correctness_pass'])
            self.assertNotIn(str(root),json.dumps(d));self.assertEqual(d['historical_official_strict_exit_preserved'],1)

    def test_complete_marker_but_observed_false_exit1(self):
        with tempfile.TemporaryDirectory() as text:
            root=Path(text)/'synthetic';q=root/'qualification-v2';q.mkdir(parents=True)
            names=['prepare-phone-fixture','phone-operator','phone-operator-analysis','measurement-plan']+[k+'-'+h+s for k in ['diagnostic','overhead','formal'] for h in ['primary','independent'] for s in ['','-analysis']]
            (root/'MEASUREMENT_COMPLETE.json').write_text(json.dumps({'all_commands_completed':True,'performance_data_collected':True,'historical_official_strict_exit_preserved':1,'raw_data_uploaded':False,'commands':16,'numerical_gate':str(q/'TIMING_GATE.json')}))
            (root/'MEASUREMENT_COMMANDS.json').write_text(json.dumps([{'name':n,'status':'completed','exit_code':0} for n in names]))
            (q/'SUMMARY.json').write_text(json.dumps({'scoped_correctness_pass':False}))
            code,d=self.call_main(root,Path(text)/'out');self.assertEqual(code,1);self.assertIs(d['scoped_correctness_pass'],False)

    def test_later_missing_preserves_verified_correctness(self):
        original=module.run_summary
        def synthetic(root,progress):progress['scoped_correctness_pass']=True;raise ValueError('synthetic later missing')
        try:
            module.run_summary=synthetic
            with tempfile.TemporaryDirectory() as text:
                code,d=self.call_main(Path(text),Path(text)/'out')
                self.assertEqual(code,2);self.assertIs(d['scoped_correctness_pass'],True);self.assertIsNone(d['analysis_integrity_pass'])
        finally:module.run_summary=original

    def test_six_slower_rounds_bounded_false_speedup(self):
        def history():return {'rounds':[{'contrast':'SQ4','transition_or_unavailable_pairs':0,'stable_strata':[{'statistics':{'pairs':21,'paired_B_minus_A_median_ms':2.0}}]} for _ in range(3)]}
        d=module.wall_observation([history(),history()]);self.assertIs(d['observed_regression_in_this_campaign'],True);self.assertIs(d['end_to_end_speedup_established'],False)

    def test_incomplete_wall_does_not_promote_a_verdict(self):
        d=module.wall_observation([{'rounds':[]}]);self.assertIsNone(d['observed_regression_in_this_campaign']);self.assertIsNone(d['end_to_end_speedup_established'])


if __name__=='__main__':unittest.main()
