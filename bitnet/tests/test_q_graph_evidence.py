"""Synthetic CPU-only rejection tests; these do not establish GPU execution."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bitnet.q_kernel_contract import variant_contract
from bitnet.tests.analyze_q_graph_contract import audit, ObservedFailure


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        q4, c8 = (variant_contract(x) for x in ('stock4x1', 'c8x2'))
        names = [('stock', None, None), ('Q4-valid', q4, None),
                 ('Q4-wrong-4x2', None, 'kernel_contract'), ('Q4-wrong-4x4', None, 'kernel_contract'),
                 ('legacy-wrong-4x2', None, 'kernel_contract'), ('legacy-wrong-4x4', None, 'kernel_contract'),
                 ('C8-with-Q4-geometry', None, 'kernel_contract'), ('Q4-descriptor-C8-library', None, 'pipeline'),
                 ('C8-valid', c8, None), ('C8-descriptor-Q4-library', None, 'pipeline'),
                 ('same-path-Q4-initial', q4, None), ('same-path-replace-C8', None, 'pipeline'),
                 ('same-path-restore-Q4', q4, None), ('same-path-missing-after-valid', None, 'pipeline'),
                 ('same-path-restore-after-missing', q4, None)]
        reference = struct.pack('<2560d', *([0.0] * 2560)); output = struct.pack('<2560f', *([0.0] * 2560))
        (self.root / 'reference.f64').write_bytes(reference)
        self.data = {'native_exit_code': 0, 'native_numerical_pass': True, 'scope': 'SYNTHETIC CPU evidence test',
                     'graph_and_buffers_reused': True, 'same_calling_thread': True,
                     'reference_sha256': hashlib.sha256(reference).hexdigest(), 'W_sha256': 'w', 'X_sha256': 'x', 'cases': []}
        lines = []
        for index, (name, descriptor, fallback) in enumerate(names):
            path = self.root / name; path.mkdir(); (path / 'y.f32').write_bytes(output)
            for side in ('before', 'after'): (path / ('canary-' + side + '.bin')).write_bytes(bytes([0xa5]) * 512)
            self.data['cases'].append({'name': name, 'synchronized': True, 'graph_status': 0,
                                      'output_sha256': hashlib.sha256(output).hexdigest(), 'FP64_failures': 0,
                                      'W_after_sha256': 'w', 'X_after_sha256': 'x', 'stock_bitwise_differences': 0})
            route = 'q-custom' if descriptor else 'q-stock'
            entry = descriptor['entry'] if descriptor else 'kernel_mul_mv_tq2_0_f32'
            grid = descriptor['grid'] if descriptor else [320, 1, 1]
            threads = descriptor['threads'] if descriptor else [32, 2, 1]
            lines.append('Q_GRAPH_CASE_BEGIN ' + json.dumps({'name': name}))
            if descriptor: lines.append('BITNET_Q_GEOMETRY checked=1')
            if fallback: lines.append('BITNET_Q_FALLBACK node=Qcur-0 layer=0 reason=' + fallback)
            lines.append(f'BITNET_MULTILAYER_ENCODE route={route} layer=0 node=Qcur-0 n=1 pipeline={entry} grid=[{",".join(map(str, grid))}] threads=[{",".join(map(str, threads))}] command=c{index}')
            lines.append(f'BITNET_MULTILAYER_COMPLETE route={route} node=Qcur-0 command=c{index} completed=1')
            lines.append('Q_GRAPH_CASE_END ' + json.dumps({'name': name}))
        self.trace = '\n'.join(lines)
        self.save()

    def save(self):
        (self.root / 'RUN.json').write_text(json.dumps(self.data))
        (self.root / 'backend-trace.log').write_text(self.trace)

    def test_complete_synthetic_evidence(self):
        self.assertEqual(audit(self.root)['strict_exit_code'], 0)

    def test_guard_hit_without_custom_dispatch_rejected(self):
        self.trace = self.trace.replace('route=q-custom', 'route=q-stock', 1); self.save()
        with self.assertRaisesRegex(ObservedFailure, 'wrong actual dispatch'):
            audit(self.root)

    def test_missing_completion_rejected(self):
        self.trace = self.trace.replace('completed=1', 'completed=0', 1); self.save()
        with self.assertRaisesRegex(ObservedFailure, 'dispatch/completion'):
            audit(self.root)

    def test_wrong_pipeline_rejected(self):
        self.trace = self.trace.replace('pipeline=tq2_0_matvec_stock4x1_v1', 'pipeline=tq2_0_matvec_c8x2_v1', 1); self.save()
        with self.assertRaisesRegex(ObservedFailure, 'pipeline/node'):
            audit(self.root)

    def test_canary_corruption_rejected(self):
        (self.root / 'Q4-valid/canary-after.bin').write_bytes(bytes(512))
        with self.assertRaisesRegex(ObservedFailure, 'canary changed'):
            audit(self.root)

    def test_setup_failure_is_not_numerical_fail(self):
        self.data['native_exit_code'] = 2; self.save()
        with self.assertRaisesRegex(ValueError, 'setup or evidence'):
            audit(self.root)


if __name__ == '__main__':
    unittest.main()
