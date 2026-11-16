#!/usr/bin/env python3
"""CPU-only scoped stock-ON/Q4-ON audit. Never executes a model or Metal.

Exit 0: every required scoped gate passes; 1: complete evidence but a required
gate fails; 2: missing/malformed/invalid evidence or an EvidenceError contract rejection.
Argparse also returns 2 for malformed command-line arguments.
This does not replace historical official-source full-model failures.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

from q4_contract import analysis_contract

MODEL_SHA = '52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2'
WEIGHT_SHA = 'b2dea7f42cacfd08ed780df979f6f28240026795b1f5c87ca8c16fc7da23dc58'
ATOL, RTOL = 2e-4, 2e-5
POSITIONS = list(range(27, 59))
VECTOR_NB = [4, 10240, 10240, 10240]
WEIGHT_NE = [2560, 2560, 1, 1]
WEIGHT_NB = [66, 660, 1689600, 1689600]


class EvidenceError(ValueError):
    pass


def require(test, text):
    if not test:
        raise EvidenceError(text)


def scoped_verdict(checks):
    verdict = False if any(v is False for v in checks.values()) else None if any(v is None for v in checks.values()) else True
    return verdict, 1 if verdict is False else 2 if verdict is None else 0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def native_root(path):
    path = Path(path).expanduser().resolve()
    return path / 'device/native' if (path / 'device/native/RUN.json').is_file() else path


def payload(root, relative, count, digest=None, dtype='<f4'):
    part = Path(relative)
    require(not part.is_absolute() and '..' not in part.parts, 'Unsafe capture path')
    path = (root / part).resolve()
    require(path.is_relative_to(root) and path.is_file(), 'Missing capture payload: ' + relative)
    require(path.stat().st_size == count * np.dtype(dtype).itemsize, 'Wrong capture byte extent: ' + relative)
    require(digest is None or sha(path) == digest, 'Capture SHA256 mismatch: ' + relative)
    return np.fromfile(path, dtype=dtype)


def metrics(actual, reference):
    a, b = np.asarray(actual, np.float64).reshape(-1), np.asarray(reference, np.float64).reshape(-1)
    require(a.size == b.size and a.size > 0, 'Comparison extent mismatch')
    finite = np.isfinite(a) & np.isfinite(b)
    err = np.abs(a - b)
    tol = ATOL + RTOL * np.abs(b)
    bad = ~finite | (err > tol)
    return {'values': int(a.size), 'failing_values': int(np.count_nonzero(bad)),
            'nonfinite_values': int(np.count_nonzero(~finite)),
            'max_abs': float(err[finite].max()) if finite.any() else None,
            'max_error_over_tolerance': float((err[finite] / tol[finite]).max()) if finite.any() else None,
            'relative_L2': float(np.linalg.norm(a-b) / np.linalg.norm(b)) if finite.all() and np.linalg.norm(b) else None,
            'bitwise_differences': int(np.count_nonzero(np.asarray(actual, np.float32).reshape(-1).view(np.uint32) !=
                                                       np.asarray(reference, np.float32).reshape(-1).view(np.uint32))),
            'pass': not bool(bad.any())}, bad


def manifest(root, route, captured=True, contract=None):
    contract = contract or analysis_contract()
    r = load(root / 'RUN.json')
    require(r['schema'] == 1 and r['operation'] == 'model' and r['workload'] == 'teacher', 'Expected teacher model RUN')
    require(r['model_sha256'] == MODEL_SHA and r['backend'] == 'MTL0' and r['activation_quantization'] is True,
            'Expected same public GGUF, Metal and ActQuant ON')
    require(r['q_route'] == route and r['q_fallback'] == 'none', 'Wrong Q route/fallback')
    require(r['prefill'] == 'batched' and r['kv'] == 'F16' and r['settings'] ==
            {'n_ctx': 512, 'batch': 128, 'ubatch': 128, 'threads': 6, 'flash_attention': False}, 'Replay settings differ')
    require(r['offloaded_layers'] == r['total_offload_layers'] == 31 and r['execution_pass'] is True and
            r['finite_logits'] is True and r['completed_decode_calls'] == 33, 'Incomplete Metal collection')
    require(r['capture_enabled'] is captured, 'Wrong probe mode')
    require(len(r['prompt_ids']) == 27 and len(r['continuation_ids']) == 32 and
            r['token_ids'] == r['prompt_ids'] + r['continuation_ids'], 'Wrong teacher history')
    require([x['input_position'] for x in r['logits']] == list(range(26, 59)), 'Incomplete logit coverage')
    if contract.get('analysis_only_legacy'):
        require(r['q_library_sha256'] == contract['legacy_library_sha256'] and
                r['q_library_source_sha256'] == contract['legacy_source_sha256'], 'Legacy run is not bound to the supplied audited shader build')
    else:
        require(r['q_kernel_contract_version'] == contract['version'] and r['q_entry'] == contract['entry'] and
                r['q_kernel_config_sha256'] == contract['config_sha256'], 'RUN canonical entry/config provenance mismatch')
    if route == 'Q4':
        require(r['q_rows_per_simd'] == contract['rows'] and r['q_simd_per_group'] == contract['groups'] and r['q_variant'] == 'stock4x1',
                'Wrong Q4 algorithm/layout')
        for key in ['q_library_sha256', 'q_library_source_sha256', 'q_library_manifest_sha256']:
            require(isinstance(r[key], str) and len(r[key]) == 64, 'Missing Q4 library provenance')
    return r


def q_records(run):
    records = [x for x in run['checkpoints'] if x['kind'] == 'Q_preRoPE' and x['node'] == 'Qcur-0']
    keys = [tuple(x['input_positions']) for x in records]
    require(keys == [tuple(range(27)), *[(p,) for p in POSITIONS]], 'Missing, duplicate or reordered prefill/32 Q0 outputs')
    for record in records:
        n = len(record['input_positions'])
        require(record['type'] == 'f32' and record['op'] == 'MUL_MAT' and record['layer'] == 0 and
                record['ne'] == [2560,n,1,1] and record['nb'] == [4,10240,10240*n,10240*n], 'Q0 output tensor contract')
        w, x = record['weight'], record['activation']
        require(w['name'] == 'blk.0.attn_q.weight' and w['type'] == 'tq2_0' and
                w['ne'] == WEIGHT_NE and w['nb'] == WEIGHT_NB and w['bytes'] == 1689600, 'Q0 weight tensor contract')
        require(x['name'] == 'actq_qkv_out-0' and x['op'] == 'ACT_QUANT' and x['type'] == 'f32' and
                x['ne'] == record['ne'] and x['nb'] == record['nb'], 'Q4 actual X is not the declared ActQuant result')
        require(record['decode_step'] == (0 if n == 27 else record['input_positions'][0]-26), 'Q decode position mapping')
    require(len({x['weight_path'] for x in records}) == 1 and {x['weight_sha256'] for x in records} == {WEIGHT_SHA},
            'Static Q0 weight file/hash identity changed')
    return records


def decode_weight(raw):
    require(raw.size == 1689600, 'Weight size')
    blocks = raw.reshape(2560,10,66)
    codes = ((blocks[:,:,:64].reshape(2560,10,2,1,32) >>
              np.array([0,2,4,6], np.uint8).reshape(1,1,1,4,1)) & 3).reshape(2560,10,256)
    require(not np.any(codes == 3), 'Illegal TQ2_0 code 3')
    scales = blocks[:,:,64:].copy().view('<f2').reshape(2560,10).astype(np.float64)
    require(np.isfinite(scales).all(), 'Nonfinite block scale')
    return (codes.astype(np.float64)-1) * scales[:,:,None]


def trace_evidence(path, route, fallback='none', contract=None):
    contract = contract or analysis_contract()
    text = Path(path).read_text(errors='replace')
    phase, begins, ends = None, [], []
    encoded, completed = Counter(), Counter()
    phase_encoded, phase_completed = Counter(), Counter()
    counts = Counter(); guard_steps = []; fallback_count = 0
    for line in text.splitlines():
        f = dict(re.findall(r'(\w+)=(\[[^\]]*\]|[^\s]+)', line))
        if line.startswith('ACTQUANT_Q_PHASE_BEGIN '):
            require(phase is None and f['q_route'] == route, 'Overlapping/wrong Q trace phase')
            phase = int(f['decode_step']); begins.append(phase); phase_encoded.clear(); phase_completed.clear()
        elif line.startswith('ACTQUANT_Q_PHASE_COMPLETE '):
            require(phase == int(f['decode_step']) and f['synchronized'] == '1', 'Missing synchronized phase completion')
            require(phase_encoded == phase_completed and sum(phase_encoded.values()) == 30,
                    'Every replay phase must complete all 30 actual Q dispatches before its synchronized marker')
            ends.append(phase); phase = None
        elif line.startswith('BITNET_ACTQUANT_Q4_GUARD '):
            require(phase is not None and phase > 0 and route == 'Q4' and f['node'] == 'Qcur-0' and
                    f['source'] == 'actq_qkv_out-0' and f['source_op'] == 'ACT_QUANT' and f['checked'] == '1',
                    'Wrong ActQuant-Q4 guard identity/phase')
            guard_steps.append(phase)
        elif line.startswith('BITNET_Q_FALLBACK '):
            require(fallback == 'missing-library' and f['reason'] == 'pipeline' and f['node'] == 'Qcur-0',
                    'Unexpected runtime fallback')
            fallback_count += 1
        elif line.startswith(('BITNET_MULTILAYER_ENCODE ', 'BITNET_MULTILAYER_COMPLETE ')) and f.get('route','').startswith('q-'):
            key = (f['route'], f['layer'], f['node'], f['command'])
            if line.startswith('BITNET_MULTILAYER_COMPLETE '):
                require(f['status'] == '4' and f['completed'] == '1', 'GPU did not complete Q dispatch')
                require(phase is not None, 'Q completion outside its replay call')
                completed[key] += 1; phase_completed[key] += 1; continue
            require(phase is not None, 'Q encode outside its replay call')
            encoded[key] += 1; phase_encoded[key] += 1
            layer = int(f['layer']); custom = f['route'] == 'q-custom'
            require(f['node'] == f'Qcur-{layer}' and f['weight'] == f'blk.{layer}.attn_q.weight', 'Wrong live Q identity')
            require(not custom or (route == 'Q4' and fallback == 'none' and phase > 0 and layer == 0 and f['n'] == '1'),
                    'Custom prefill/non-target/stock dispatch')
            if custom:
                require(f['pipeline'] == contract['entry'] and f['grid'] == json.dumps(contract['grid'],separators=(',',':')) and
                        f['threads'] == json.dumps(contract['threads'],separators=(',',':')), 'Wrong actual Q4 pipeline/launch')
            else:
                require(f['pipeline'] == ('kernel_mul_mm_tq2_0_f32' if phase == 0 else 'kernel_mul_mv_tq2_0_f32'),
                        'Wrong actual stock Q pipeline')
            bucket = 'prefill_stock' if phase == 0 else 'decode_target_custom' if custom else 'decode_target_stock' if layer == 0 else 'decode_nontarget_stock'
            counts[bucket] += 1
    require(phase is None and begins == ends == list(range(33)), 'Expected one prefill plus 32 complete decode phases')
    require(encoded == completed, 'Q pipeline encode/completion identities or counts differ')
    expect_custom = 32 if route == 'Q4' and fallback == 'none' else 0
    require(counts['prefill_stock'] == 30 and counts['decode_nontarget_stock'] == 928 and
            counts['decode_target_custom'] == expect_custom and counts['decode_target_stock'] == 32-expect_custom,
            'Wrong actual stock/custom Q dispatch counts')
    require(guard_steps == (list(range(1,33)) if route == 'Q4' else []), 'ActQuant-specific guard did not hit every eligible call')
    require(fallback_count == (32 if fallback == 'missing-library' else 0), 'Missing-library fallback count')
    require('ACTQUANT_MODEL_COMPLETE enabled=1 backend=MTL0 workload=teacher decode_calls=33 logits=33 ' in text,
            'Native complete marker missing')
    return {'pass': True, 'counts': dict(counts), 'eligible_calls': 32, 'custom_dispatches': expect_custom,
            'guard_hits': len(guard_steps), 'fallbacks': fallback_count, 'all_Q_encodes_completed': sum(encoded.values()),
            'log_sha256': sha(path)}


def logits(root, run):
    return [payload(root, v['path'], 128256) for v in run['logits']]


def vector_comparison(a, b):
    rows = [metrics(x,y)[0] for x,y in zip(a,b)]
    return {'vectors': len(rows), 'pass': all(x['pass'] for x in rows),
            'bitwise_pass': all(x['bitwise_differences'] == 0 for x in rows),
            'failing_values': sum(x['failing_values'] for x in rows),
            'bitwise_differences': sum(x['bitwise_differences'] for x in rows),
            'max_abs': max(x['max_abs'] for x in rows), 'rows': rows}


def probe_control(path, reference_root, reference_run, route, contract=None):
    if path is None:
        return None
    root = native_root(path); r = manifest(root, route, False, contract)
    require(r['token_ids'] == reference_run['token_ids'] and r['q_library_sha256'] == reference_run['q_library_sha256'],
            'Probe history/library differs')
    return vector_comparison(logits(root,r), logits(reference_root,reference_run))


def quantizer_check(root, run, q, source):
    from prepare_fixtures import official_quantizer
    import torch
    torch.set_num_threads(1)
    oracle = official_quantizer(source)
    records = [x for x in run['checkpoints'] if x['kind'] == 'actquant' and x['node'] == 'actq_qkv_out-0']
    require([x['input_positions'] for x in records] == [list(range(27)), *[[p] for p in POSITIONS]], 'Missing Q0 quantizer coverage')
    values = bits = codes = 0
    for r, projection in zip(records,q):
        n = len(r['input_positions']); x = payload(root,r['input_path'],n*2560).reshape(n,2560)
        y = payload(root,r['output_path'],n*2560).reshape(n,2560)
        xt = torch.from_numpy(x.copy()); expected = oracle.apply(xt).numpy()
        scale = (127 / xt.abs().max(-1,keepdim=True).values.clamp(min=1e-5)).numpy()
        expected_codes = np.rint((xt * torch.from_numpy(scale)).numpy()).clip(-128,127)
        bits += int(np.count_nonzero(y.view(np.uint32) != expected.view(np.uint32)))
        codes += int(np.count_nonzero(np.rint(y.astype(np.float64)*scale.astype(np.float64)) != expected_codes))
        require(sha(root/r['output_path']) == projection['input_sha256'], 'Captured Q X does not equal actual quantizer Y')
        values += y.size
    return {'captures':33,'values':int(values),'bitwise_differences':bits,'integer_code_mismatches':codes,'pass':bits==codes==0}


def run(args):
    out = args.output.resolve(); require(not out.exists(), 'Refuse existing analysis directory'); out.mkdir(parents=True)
    result = {'analysis_completed':False,'scope':'ActQuant-ON layer0 Q4 vs stock-ON; 27+32 fixed-token replay',
              'frozen_tolerance':{'absolute':ATOL,'relative':RTOL},'official_source_model_output_pass':None,
              'model_quality_pass':None,'performance_measured':False,'historical_strict_failures_overwritten':False}
    try:
        contract = analysis_contract(args.contract_profile, args.legacy_shader_build)
        result['contract_profile'] = args.contract_profile
        result['Q4_entry'] = contract['entry']; result['Q4_contract_version'] = contract['version']
        if contract.get('analysis_only_legacy'): result['legacy_shader_build_sha256'] = contract['legacy_build_sha256']
        sr, cr = native_root(args.stock), native_root(args.candidate)
        s,c = manifest(sr,'stock',True,contract),manifest(cr,'Q4',True,contract); sq,cq = q_records(s),q_records(c)
        require(s['token_ids'] == c['token_ids'] and s['device'] == c['device'] and s['q_library_sha256'] == c['q_library_sha256'],
                'Routes must have identical history, device and Q4 library identity')
        w = payload(sr,sq[0]['weight_path'],1689600,WEIGHT_SHA,np.uint8); payload(cr,cq[0]['weight_path'],1689600,WEIGHT_SHA,np.uint8)
        decoded = decode_weight(w); comparisons=[]; input_hashes=[]; fp_rows=[]
        with (out/'FP64_ROWS.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['step','input_position','row','FP64_reference','stock','Q4','stock_abs_error','Q4_abs_error','tolerance','stock_fail','Q4_fail'])
            for step,(a,b) in enumerate(zip(sq,cq)):
                n=len(a['input_positions']); require(a['input_sha256'] == b['input_sha256'], 'Stock/Q4 actual X hashes differ')
                x=payload(sr,a['input_path'],2560*n,a['input_sha256']);payload(cr,b['input_path'],2560*n,b['input_sha256'])
                y=payload(sr,a['path'],2560*n,a['output_sha256']);z=payload(cr,b['path'],2560*n,b['output_sha256'])
                comparison,_=metrics(z,y);comparison.update(step=step,input_positions=a['input_positions']);comparisons.append(comparison)
                input_hashes.append({'step':step,'weight_sha256':WEIGHT_SHA,'input_sha256':a['input_sha256'],'stock_output_sha256':a['output_sha256'],'Q4_output_sha256':b['output_sha256']})
                if step==0: continue  # Batched prefill's stock half staging is a separate existing failure.
                ref=np.einsum('rbk,bk->r',decoded,x.astype(np.float64).reshape(10,256),optimize=False)
                sm,sbad=metrics(y,ref);cm,cbad=metrics(z,ref);fp_rows.append({'step':step,'stock':sm,'Q4':cm})
                for i in range(2560): writer.writerow([step,a['input_positions'][0],i,ref[i],y[i],z[i],abs(float(y[i])-ref[i]),abs(float(z[i])-ref[i]),ATOL+RTOL*abs(ref[i]),int(sbad[i]),int(cbad[i])])
        layer_s=[x for x in s['checkpoints'] if x['kind']=='l_out' and x['node']=='l_out-0' and x['input_positions'] in [[27],[42],[58]]]
        layer_c=[x for x in c['checkpoints'] if x['kind']=='l_out' and x['node']=='l_out-0' and x['input_positions'] in [[27],[42],[58]]]
        require([x['input_positions'] for x in layer_s] == [x['input_positions'] for x in layer_c] == [[27],[42],[58]], 'Missing sampled residual steps1/16/32')
        residual=vector_comparison([payload(cr,x['path'],2560) for x in layer_c],[payload(sr,x['path'],2560) for x in layer_s])
        logit=vector_comparison(logits(cr,c),logits(sr,s));probe_s=probe_control(args.stock_plain,sr,s,'stock',contract);probe_c=probe_control(args.candidate_plain,cr,c,'Q4',contract)
        traces={'stock':trace_evidence(args.stock_log,'stock',contract=contract),'Q4':trace_evidence(args.candidate_log,'Q4',contract=contract)}
        quant={'stock':quantizer_check(sr,s,sq,args.official_source),'Q4':quantizer_check(cr,c,cq,args.official_source)}
        fallback=None
        if args.missing_library is not None and args.missing_library_log is not None:
            fr=native_root(args.missing_library);f=load(fr/'RUN.json');require(f['q_route']=='Q4' and f['q_fallback']=='missing-library' and f['token_ids']==s['token_ids'] and f['activation_quantization'] is True,'Wrong missing-library control')
            fb=vector_comparison(logits(fr,f),logits(sr,s));trace=trace_evidence(args.missing_library_log,'Q4','missing-library',contract);fallback={'pass':fb['bitwise_pass'] and trace['pass'],'logits':fb,'trace':trace}
        checks={'identical_real_inputs_pass':True,'parity_pass':all(v['pass'] for v in comparisons[1:]),
                'fp64_pass':all(v['stock']['pass'] and v['Q4']['pass'] for v in fp_rows),
                'model_output_pass':residual['pass'] and logit['pass'],
                'local_actquant_pass':all(v['pass'] for v in quant.values()),'dispatch_pass':True,
                'prefill_and_nontarget_stock_pass':comparisons[0]['bitwise_differences']==0,
                'probe_neutrality_pass':None if probe_s is None or probe_c is None else probe_s['bitwise_pass'] and probe_c['bitwise_pass'],
                'missing_library_stock_fallback_pass':None if fallback is None else fallback['pass']}
        verdict,code=scoped_verdict(checks)
        result.update(analysis_completed=True,checks=checks,scoped_stock_compatibility_pass=verdict,strict_scoped_exit_code=code,
                      numerical_scope='All32 N1 Q0 dots; prefill parity only; residual1/16/32; all33 logits versus stock-ON',
                      model_sha256=MODEL_SHA,weight_sha256=WEIGHT_SHA,q_library_sha256=c['q_library_sha256'],
                      source_input_hashes=input_hashes,Q_comparisons=comparisons,FP64=fp_rows,residuals=residual,logits=logit,
                      probe_controls={'stock':probe_s,'Q4':probe_c},actual_runtime=traces,actual_input_quantizers=quant,missing_library=fallback,
                      full_model_FP64_pass=None,official_source_logit_verdict='Not recomputed by this scoped analyzer; prior strict1 remains')
    except (EvidenceError,OSError,KeyError,TypeError,ValueError) as e:
        result.update(analysis_completed=False,error=str(e),scoped_stock_compatibility_pass=None,strict_scoped_exit_code=2);code=2
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['Q_comparisons','FP64','source_input_hashes','residuals','logits','probe_controls']},indent=2))
    return code


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['stock','candidate','stock-log','candidate-log','output']:
        p.add_argument('--'+name,type=Path,required=True)
    for name in ['stock-plain','candidate-plain','missing-library','missing-library-log']:
        p.add_argument('--'+name,type=Path)
    p.add_argument('--contract-profile',choices=('canonical-v1','legacy-shared-v0'),default='canonical-v1')
    p.add_argument('--legacy-shader-build',type=Path,help='Analysis-only preserved shader BUILD.json; required for legacy-shared-v0')
    p.add_argument('--official-source',type=Path,default=Path(__file__).resolve().parents[1]/'reference/official-bitnet-096f25ae.py')
    return run(p.parse_args(argv))


if __name__=='__main__':
    raise SystemExit(main())
