#!/usr/bin/env python3
"""Reject guard-only controls that never traversed the full live guard."""
from pathlib import Path
import importlib.util
import sys
import unittest

IOS = Path(__file__).resolve().parents[1] / 'ios'
sys.path.insert(0, str(IOS))
spec = importlib.util.spec_from_file_location('phone_runner', IOS / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def valid_log():
    lines = []
    for route in ('S', 'QB'):
        lines.append('FFNK_PATH_BEGIN route=' + route)
        for step in range(1, 33):
            if route == 'QB':
                lines.append(f'BITNET_Q_GUARD_PASS node=Qcur-0 layer=0 step={step} checked=1 candidate_pipeline=stock4x1 width=32 max_threads=1024 rows=4 simd=1 w=0xW+0 x=0xXY+0 y=0xXY+10240 bounds=1 disjoint=1 custom_encoded=0')
            lines.append(f'BITNET_Q_STOCK_ABI node=Qcur-0 step={step} guarded={int(route == "QB")} pipeline=kernel_mul_mv_tq2_0_f32 cached_pipeline=0xC args_bytes=120 add=0 norm=0 grid=[320,1,1] threads=[32,2,1] smem=0 w=0xW+0 x=0xXY+0 y=0xXY+10240 residual=0xXY+10240 norm_weight=0xXY+0 slots=[0,1,2,3,4,5]')
            lines.append('BITNET_MULTILAYER_ENCODE route=q-stock layer=0 node=Qcur-0 n=1 pso=0xP')
        lines.append('FFNK_PATH_END route=' + route)
    return '\n'.join(lines)


class GuardEvidence(unittest.TestCase):
    def errors(self, log):
        return runner.validate_q_guard_control(log, ['S', 'QB'])[1]

    def test_valid_control(self):
        self.assertEqual(self.errors(valid_log()), [])

    def test_rejected_guard_is_not_success(self):
        log = '\n'.join(line for line in valid_log().splitlines() if not line.startswith('BITNET_Q_GUARD_PASS '))
        self.assertTrue(self.errors(log))

    def test_one_missing_decode_cannot_pass(self):
        log = valid_log().replace('BITNET_Q_GUARD_PASS node=Qcur-0 layer=0 step=16 ', 'UNRECOGNIZED step=16 ', 1)
        self.assertTrue(self.errors(log))

    def test_changed_live_offset_cannot_pass(self):
        log = valid_log()
        pivot = log.index('FFNK_PATH_BEGIN route=QB')
        log = log[:pivot] + log[pivot:].replace('y=0xXY+10240', 'y=0xXY+20480')
        self.assertTrue(self.errors(log))

    def test_different_actual_pipeline_cannot_pass(self):
        log = valid_log()
        pivot = log.index('FFNK_PATH_BEGIN route=QB')
        self.assertTrue(self.errors(log[:pivot] + log[pivot:].replace('pso=0xP', 'pso=0xDIFFERENT')))

    def test_custom_encoding_cannot_pass(self):
        self.assertTrue(self.errors(valid_log().replace('custom_encoded=0', 'custom_encoded=1', 1)))


if __name__ == '__main__':
    unittest.main()
