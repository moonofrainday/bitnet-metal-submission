#!/usr/bin/env python3
"""CPU-only rejection/coverage controls for the joint Q4 verifier."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from analyze_q4 import EvidenceError, POSITIONS, WEIGHT_NB, WEIGHT_NE, WEIGHT_SHA, payload, q_records, scoped_verdict, trace_evidence


def record(position):
    positions=list(range(27)) if position==0 else [position]
    n=len(positions);ne=[2560,n,1,1];nb=[4,10240,10240*n,10240*n]
    return {'kind':'Q_preRoPE','node':'Qcur-0','op':'MUL_MAT','type':'f32','layer':0,'input_positions':positions,
            'ne':ne,'nb':nb,'weight_path':'weights/w.tq2','weight_sha256':WEIGHT_SHA,
            'decode_step':0 if position==0 else position-26,
            'weight':{'name':'blk.0.attn_q.weight','type':'tq2_0','ne':WEIGHT_NE,'nb':WEIGHT_NB,'bytes':1689600},
            'activation':{'name':'actq_qkv_out-0','op':'ACT_QUANT','type':'f32','ne':ne,'nb':nb}}


def trace():
    lines=[]
    for step in range(33):
        lines.append(f'ACTQUANT_Q_PHASE_BEGIN q_route=Q4 decode_step={step}')
        if step:
            lines.append('BITNET_ACTQUANT_Q4_GUARD node=Qcur-0 source=actq_qkv_out-0 source_op=ACT_QUANT checked=1')
        for layer in range(30):
            custom=step>0 and layer==0;r='q-custom' if custom else 'q-stock';n=1 if step else 27
            pipeline='tq2_0_matvec_poc_c8x2' if custom else 'kernel_mul_mv_tq2_0_f32' if step else 'kernel_mul_mm_tq2_0_f32'
            fields=f'route={r} layer={layer} node=Qcur-{layer} weight=blk.{layer}.attn_q.weight n={n} command=0x{step:02x}{layer:02x}'
            lines.append(f'BITNET_MULTILAYER_ENCODE {fields} pipeline={pipeline} grid=[640,1,1] threads=[32,1,1]')
            lines.append(f'BITNET_MULTILAYER_COMPLETE {fields} status=4 completed=1')
        lines.append(f'ACTQUANT_Q_PHASE_COMPLETE q_route=Q4 decode_step={step} synchronized=1')
    lines.append('ACTQUANT_MODEL_COMPLETE enabled=1 backend=MTL0 workload=teacher decode_calls=33 logits=33 actquant_callbacks=33 synchronized=1')
    return '\n'.join(lines)


class Q4Contract(unittest.TestCase):
    def test_full_coverage(self):
        self.assertEqual(len(q_records({'checkpoints':[record(0),*[record(p) for p in POSITIONS]]})),33)

    def test_missing_last_call_rejected(self):
        with self.assertRaises(EvidenceError):q_records({'checkpoints':[record(0),*[record(p) for p in POSITIONS[:-1]]]})

    def test_duplicate_call_rejected(self):
        rows=[record(0),*[record(p) for p in POSITIONS]];rows[-1]=copy.deepcopy(rows[-2])
        with self.assertRaises(EvidenceError):q_records({'checkpoints':rows})

    def test_wrong_weight_stride_rejected(self):
        rows=[record(0),*[record(p) for p in POSITIONS]];rows[1]=copy.deepcopy(rows[1]);rows[1]['weight']['nb']=[66,660,0,0]
        with self.assertRaises(EvidenceError):q_records({'checkpoints':rows})

    def test_unquantized_actual_X_rejected(self):
        rows=[record(0),*[record(p) for p in POSITIONS]];rows[1]=copy.deepcopy(rows[1]);rows[1]['activation']['op']='MUL'
        with self.assertRaises(EvidenceError):q_records({'checkpoints':rows})

    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'x.f32').write_bytes(b'\x00'*4)
            with self.assertRaises(EvidenceError):payload(root,'x.f32',1,'0'*64)

    def test_escaping_path_rejected(self):
        with self.assertRaises(EvidenceError):payload(Path('/tmp'),'../secret',1)

    def test_missing_probe_cannot_pass(self):
        self.assertEqual(scoped_verdict({'fp64':True,'probe':None}),(None,2))
        self.assertEqual(scoped_verdict({'fp64':False,'probe':None}),(False,1))
        self.assertEqual(scoped_verdict({'fp64':True,'probe':True}),(True,0))

    def test_complete_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.log';path.write_text(trace())
            self.assertEqual(trace_evidence(path,'Q4')['custom_dispatches'],32)

    def test_missing_gpu_completion_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.log';path.write_text(trace().replace('status=4 completed=1','status=5 completed=0',1))
            with self.assertRaises(EvidenceError):trace_evidence(path,'Q4')

    def test_incorrect_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.log';path.write_text(trace().replace('pipeline=tq2_0_matvec_poc_c8x2 grid=[640,1,1]','pipeline=tq2_0_matvec_poc_c8x2 grid=[320,1,1]',1))
            with self.assertRaises(EvidenceError):trace_evidence(path,'Q4')


if __name__=='__main__':unittest.main()
