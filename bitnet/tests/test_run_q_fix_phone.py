#!/usr/bin/env python3
"""CPU-only synthetic launcher controls. Never calls a device or native runner."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_q_fix_phone as entry


class ClassificationTests(unittest.TestCase):
    def classify(self, stage, filename=None, value=None, field='runtime_contract_pass'):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            if filename:
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({field: value}))
            return entry.failed_stage_gate(root, stage)

    def test_actual_contract_failure_is_false(self):
        result = self.classify('contract', 'contract/summary.json', False)
        self.assertIs(result['required_gate'], False)
        self.assertIs(result['scoped_qualification_pass'], False)

    def test_blocked_contract_is_null(self):
        self.assertIsNone(self.classify('contract', 'contract/summary.json', None)['scoped_qualification_pass'])

    def test_missing_contract_is_null(self):
        self.assertIsNone(self.classify('contract')['scoped_qualification_pass'])

    def test_passed_contract_does_not_complete_other_gates(self):
        result = self.classify('contract', 'contract/summary.json', True)
        self.assertIs(result['required_gate'], True)
        self.assertIsNone(result['scoped_qualification_pass'])

    def test_invalid_boolean_is_not_a_failure(self):
        self.assertIsNone(self.classify('contract', 'contract/summary.json', 0)['scoped_qualification_pass'])

    def test_malformed_contract_is_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'contract').mkdir()
            (root / 'contract/summary.json').write_text('{malformed')
            self.assertIsNone(entry.failed_stage_gate(root, 'contract')['scoped_qualification_pass'])

    def test_operator_numerical_failure_is_false(self):
        result = self.classify('operator-analysis', 'operator-analysis/SUMMARY.json', False, 'numerical_pass')
        self.assertIs(result['scoped_qualification_pass'], False)

    def test_history_required_failure_is_false(self):
        for history in ('primary', 'independent'):
            result = self.classify(history + '-analysis', history + '-analysis/SUMMARY.json',
                                   False, 'scoped_stock_compatibility_pass')
            self.assertIs(result['scoped_qualification_pass'], False)

    def test_collection_failure_is_not_numerical_failure(self):
        result = self.classify('primary-Q4-capture', 'primary-Q4-capture/summary.json', False, 'execution_pass')
        self.assertIsNone(result['scoped_qualification_pass'])


class ControlFlowTests(unittest.TestCase):
    def argv(self, output):
        return ['run_q_fix_phone.py', '--run', '--build-dir', 'unused-build', '--device', 'synthetic-device',
                '--model', 'unused-model', '--operator-fixtures', 'unused-fixtures', '--tokens-primary',
                'unused-primary', '--tokens-independent', 'unused-independent', '--output', str(output)]

    def execute(self, gate, child_code):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'fresh'

            def fake_child(*unused, **kwargs):
                (output / 'contract').mkdir()
                (output / 'contract/summary.json').write_text(json.dumps({'runtime_contract_pass': gate}))
                return mock.Mock(returncode=child_code)

            with mock.patch.object(sys, 'argv', self.argv(output)), mock.patch.object(entry, 'prepare', return_value={}), \
                 mock.patch.object(entry, 'stages', return_value=[('contract', ['synthetic-command'])]), \
                 mock.patch.object(entry.subprocess, 'run', side_effect=fake_child) as child, \
                 contextlib.redirect_stdout(io.StringIO()):
                code = entry.main()
            self.assertEqual(child.call_count, 1)
            return code, json.loads((output / 'SUMMARY.json').read_text())

    def test_failed_child_gate_preserves_failure_and_exit(self):
        code, result = self.execute(False, 1)
        self.assertEqual(code, 1)
        self.assertIs(result['scoped_qualification_pass'], False)
        self.assertIs(result['execution_failed'], True)
        self.assertIs(result['all_stages_completed'], False)

    def test_blocked_child_preserves_null_and_exit(self):
        code, result = self.execute(None, 2)
        self.assertEqual(code, 2)
        self.assertIsNone(result['scoped_qualification_pass'])
        self.assertEqual(result['failed_stage'], 'contract')
        self.assertIs(result['execution_failed'], True)
        self.assertIs(result['all_stages_completed'], False)
        self.assertIs(result['device_commands_requested'], True)
        self.assertIs(result['collection_command_started'], True)
        self.assertIsNone(result['device_commands_executed'])

    def test_setup_failure_is_unverified_and_never_launches(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'fresh'
            with mock.patch.object(sys, 'argv', self.argv(output)), \
                 mock.patch.object(entry, 'prepare', side_effect=ValueError('synthetic unavailable setup')), \
                 mock.patch.object(entry.subprocess, 'run') as child, contextlib.redirect_stdout(io.StringIO()):
                code = entry.main()
            result = json.loads((output / 'SUMMARY.json').read_text())
            child.assert_not_called()
            self.assertEqual(code, 2)
            self.assertIsNone(result['scoped_qualification_pass'])
            self.assertIs(result['device_commands_executed'], False)
            self.assertIs(result['device_commands_requested'], True)
            self.assertIs(result['collection_command_started'], False)

    def test_complete_gate_failure_survives_verdict_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'contract').mkdir()
            (root / 'contract/summary.json').write_text(json.dumps({'runtime_contract_pass': False}))
            summary = {'scoped_qualification_pass': None}
            with self.assertRaises(ValueError):
                entry.collect_verdict(mock.Mock(output=root), summary)
            self.assertIs(summary['scoped_qualification_pass'], False)


if __name__ == '__main__':
    unittest.main()
