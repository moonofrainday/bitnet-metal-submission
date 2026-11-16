#!/usr/bin/env python3
"""CPU-only rejection/coverage controls for the joint Q4 verifier."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from q4_contract import ROOT, SOURCE, LEGACY_SOURCE_SHA, LEGACY_ENTRY, analysis_contract, q4_library_identity
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
            pipeline=analysis_contract()['entry'] if custom else 'kernel_mul_mv_tq2_0_f32' if step else 'kernel_mul_mm_tq2_0_f32'
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
            path=Path(tmp)/'trace.log';path.write_text(trace().replace('pipeline='+analysis_contract()['entry']+' grid=[640,1,1]','pipeline='+analysis_contract()['entry']+' grid=[320,1,1]',1))
            with self.assertRaises(EvidenceError):trace_evidence(path,'Q4')


    def test_legacy_name_rejected_by_default_trace_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.log';path.write_text(trace().replace(analysis_contract()['entry'],LEGACY_ENTRY))
            with self.assertRaises(EvidenceError):trace_evidence(path,'Q4')

    def test_legacy_analysis_profile_requires_build_source_identity(self):
        with self.assertRaises(ValueError):analysis_contract('legacy-shared-v0')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'BUILD.json'
            build={'build_completed':True,'exit_code':0,'sdk':'iphoneos','sources':{SOURCE:LEGACY_SOURCE_SHA},
                   'products':{'stock4x1.metallib':{'source':'stock4x1.metal','expected_symbols':[LEGACY_ENTRY],'sha256':'a'*64}},
                   'commands':[{'name':'stock4x1-compile','exit_code':0,'argv':[str(ROOT/SOURCE)]}]}
            path.write_text(json.dumps(build));contract=analysis_contract('legacy-shared-v0',path)
            self.assertTrue(contract['analysis_only_legacy']);self.assertEqual(contract['entry'],LEGACY_ENTRY)
            build['sources'][SOURCE]='0'*64;path.write_text(json.dumps(build))
            with self.assertRaises(ValueError):analysis_contract('legacy-shared-v0',path)

    def test_legacy_build_is_never_a_current_runtime_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);library=root/'q.metallib';library.write_bytes(b'unit-only-not-a-real-library')
            path=root/'BUILD.json';path.write_text(json.dumps({'build_completed':True,'exit_code':0,'sdk':'iphoneos',
                'sources':{SOURCE:LEGACY_SOURCE_SHA},'products':{'stock4x1.metallib':{'source':'stock4x1.metal',
                'expected_symbols':[LEGACY_ENTRY],'sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'bytes':library.stat().st_size}}}))
            with self.assertRaises(ValueError):q4_library_identity(library,path,'iphoneos')


if __name__=='__main__':unittest.main()
