#!/usr/bin/env python3
"""Serial evening qualification of the repaired Q4 contract on a physical iPhone.

Default --plan performs CPU-only local provenance checks and writes intended
commands. Only explicit --run installs or launches an app. This entry delegates
to the existing provenance, native collection and numerical-analysis CLIs; it
does not measure performance or grant official-model/quality acceptance.

Exit 0 means a plan was saved, or every required scoped runtime/numerical gate
passed during --run. A failed child stage preserves its nonzero exit code;
invalid/missing local evidence returns 2. Raw artifacts remain in --output.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bitnet/actquant'))
sys.path.insert(0, str(ROOT / 'bitnet/ios'))


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    # These helpers perform local file checks only. No Xcode/device/GPU calls.
    from bitnet.actquant.run_phone import build_identity, fixture_identity, token_identity
    from bitnet.actquant.q4_contract import q4_library_identity
    build, identity = build_identity(args.build_dir)
    fixture, _ = fixture_identity(args.operator_fixtures)
    require(len(fixture['cases']) == 20, 'This qualification requires the frozen 20-case ActQuant fixture pack')
    require(sum(v['elements'] for v in fixture['cases']) == 386567,
            'Frozen ActQuant fixture element count differs')
    primary = token_identity(args.tokens_primary)
    independent = token_identity(args.tokens_independent)
    require(primary['prompt'] != independent['prompt'], 'Independent prompt must differ from the primary prompt')
    require(args.model.is_file(), 'Model file is missing; collection CLI verifies its exact SHA-256')
    shader_root = Path(build['shaders'])
    q4 = q4_library_identity(shader_root / 'q-stock4x1.metallib', shader_root / 'BUILD.json', 'iphoneos')
    return {'build_manifest_sha256': identity['build_manifest_sha256'],
            'model_sha256_expected': '52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2',
            'operator_fixture_sha256': digest(args.operator_fixtures / 'FIXTURES.json'),
            'operator_cases': 20, 'operator_values': 386567,
            'tokens_sha256': {'primary': digest(args.tokens_primary), 'independent': digest(args.tokens_independent)},
            'Q4': {k: v for k, v in q4.items() if k != 'q_library'}}


def stages(args):
    python = sys.executable
    shared = ['--build-dir', str(args.build_dir), '--device', args.device]
    developer = ['--developer-dir', str(args.developer_dir)] if args.developer_dir else []
    native = [python, str(ROOT / 'bitnet/actquant/run_phone.py'), '--run', *shared,
              '--model', str(args.model), *developer]
    reuse = [] if args.copy_model else ['--reuse-model']
    op = args.output / 'operator'
    op_analysis = args.output / 'operator-analysis'
    tasks = [('contract', [python, str(ROOT / 'bitnet/ios/run.py'), *shared,
                          '--mode', 'q-contract', '--install', '--output', str(args.output / 'contract'), *developer]),
             ('operator', [*native, '--operation', 'operator', '--fixture', str(args.operator_fixtures),
                           '--output', str(op), *reuse]),
             ('operator-analysis', [python, str(ROOT / 'bitnet/actquant/analyze_op.py'),
                                    '--fixture', str(args.operator_fixtures), '--actual', str(op / 'device/native'),
                                    '--runtime', str(op / 'device/native/RUN.json'), '--require-runtime',
                                    '--output', str(op_analysis)])]
    for history, tokens in [('primary', args.tokens_primary), ('independent', args.tokens_independent)]:
        model_base = [*native, '--operation', 'model', '--tokens', str(tokens), '--actquant', 'on',
                      '--workload', 'teacher', '--capture-scope', 'q4-layer0',
                      '--operator-run', str(op), '--operator-analysis', str(op_analysis / 'SUMMARY.json'), *reuse]
        for name, route, capture, missing in [('stock-capture', 'stock', True, False),
                                               ('Q4-capture', 'Q4', True, False),
                                               ('stock-plain', 'stock', False, False),
                                               ('Q4-plain', 'Q4', False, False),
                                               ('Q4-missing-library', 'Q4', False, True)]:
            flags = ['--capture', '--trace'] if capture else ['--trace'] if missing else []
            tasks.append((history + '-' + name, [*model_base, '--q-route', route,
                          '--q-fallback', 'missing-library' if missing else 'none',
                          '--output', str(args.output / (history + '-' + name)), *flags]))
        names = {name: args.output / (history + '-' + name) for name in
                 ['stock-capture', 'Q4-capture', 'stock-plain', 'Q4-plain', 'Q4-missing-library']}
        tasks.append((history + '-analysis', [python, str(ROOT / 'bitnet/actquant/analyze_q4.py'),
                      '--contract-profile', 'canonical-v1', '--stock', str(names['stock-capture']),
                      '--candidate', str(names['Q4-capture']), '--stock-log', str(names['stock-capture'] / 'logs/launch.log'),
                      '--candidate-log', str(names['Q4-capture'] / 'logs/launch.log'),
                      '--stock-plain', str(names['stock-plain']), '--candidate-plain', str(names['Q4-plain']),
                      '--missing-library', str(names['Q4-missing-library']),
                      '--missing-library-log', str(names['Q4-missing-library'] / 'logs/launch.log'),
                      '--output', str(args.output / (history + '-analysis'))]))
    return tasks


def failed_stage_gate(output, stage):
    """A collection/process failure alone is not a numerical rejection."""
    sources = {'contract': ('contract/summary.json', 'runtime_contract_pass'),
               'operator-analysis': ('operator-analysis/SUMMARY.json', 'numerical_pass'),
               'primary-analysis': ('primary-analysis/SUMMARY.json', 'scoped_stock_compatibility_pass'),
               'independent-analysis': ('independent-analysis/SUMMARY.json', 'scoped_stock_compatibility_pass')}
    if stage not in sources:
        return {'required_gate': None, 'scoped_qualification_pass': None,
                'evidence_status': 'collection/process stage; no independent acceptance gate'}
    filename, field = sources[stage]
    result = {'summary': filename, 'field': field, 'required_gate': None,
              'scoped_qualification_pass': None, 'evidence_status': 'missing or invalid evidence'}
    try:
        value = load(output / filename).get(field)
        if type(value) is bool:
            result.update(required_gate=value, evidence_status='observed required gate',
                          scoped_qualification_pass=False if value is False else None)
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    return result


def collect_verdict(args, summary):
    contract = load(args.output / 'contract/summary.json')
    if contract.get('runtime_contract_pass') is False:
        summary['scoped_qualification_pass'] = False
    require(contract['runtime_contract_pass'] is True and contract['native_exit_code'] == 0,
            'Completed contract stage lacks runtime acceptance')
    operator = load(args.output / 'operator-analysis/SUMMARY.json')
    if operator.get('numerical_pass') is False:
        summary['scoped_qualification_pass'] = False
    require(operator['numerical_pass'] is True and operator['strict_operator_exit_code'] == 0 and
            len(operator['cases']) == 20 and operator['values'] == 386567,
            'Completed operator stage lacks the full independent numerical gate')
    histories = {}
    for history in ['primary', 'independent']:
        result = load(args.output / (history + '-analysis/SUMMARY.json'))
        if result.get('scoped_stock_compatibility_pass') is False:
            summary['scoped_qualification_pass'] = False
        require(result['contract_profile'] == 'canonical-v1' and result['strict_scoped_exit_code'] == 0 and
                result['scoped_stock_compatibility_pass'] is True and result['checks'] and
                all(v is True for v in result['checks'].values()), 'Incomplete scoped checks: ' + history)
        require(result['official_source_model_output_pass'] is None and result['full_model_FP64_pass'] is None,
                'Scoped analysis must not grant full-model/reference acceptance')
        histories[history] = {'scoped_stock_compatibility_pass': True, 'strict_scoped_exit_code': 0,
                              'checks': result['checks'], 'summary_sha256': digest(args.output / (history + '-analysis/SUMMARY.json'))}
    summary.update(runtime_contract_pass=True, ActQuant_operator_pass=True, histories=histories,
                   scoped_qualification_pass=True, all_stages_completed=True, execution_pass=True,
                   execution_failed=False, device_commands_executed=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--run', action='store_true', help='Explicitly install and launch the physical iPhone stages')
    mode.add_argument('--plan', action='store_true', help='Default: CPU-only provenance and command plan')
    for name in ['build-dir', 'model', 'operator-fixtures', 'tokens-primary', 'tokens-independent', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', required=True, help='Physical devicectl device identifier; no stored default')
    p.add_argument('--developer-dir', type=Path)
    p.add_argument('--copy-model', action='store_true', help='Transfer model for each collection instead of reusing the checksum-verified device cache')
    args = p.parse_args()
    for name in ['build_dir', 'model', 'operator_fixtures', 'tokens_primary', 'tokens_independent', 'output', 'developer_dir']:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.output.exists():
        p.error('--output must be a fresh directory')
    args.output.mkdir(parents=True)
    summary = {'scope': 'Canonical Q4 runtime contract + ActQuant operator + two-history layer0 Q4/stock-ON qualification',
               'mode': 'run' if args.run else 'plan', 'device_commands_executed': False, 'all_stages_completed': False,
               'device_commands_requested': args.run, 'collection_command_started': False,
               'execution_pass': None, 'execution_failed': False,
               'runtime_contract_pass': None, 'ActQuant_operator_pass': None, 'scoped_qualification_pass': None,
               'official_full_model_pass': None, 'full_model_FP64_pass': None, 'model_quality_pass': None,
               'historical_official_strict_exit_code': 1, 'historical_strict_results_recomputed': False,
               'performance_measured': False, 'raw_data_remains_local': True, 'stages': [], 'errors': []}
    try:
        summary['local_provenance'] = prepare(args)
        tasks = stages(args)
        write(args.output / 'PLAN.json', {'device_commands_executed': False, 'stage_count': len(tasks),
                                         'stages': [{'name': name, 'argv': command} for name, command in tasks]})
        if not args.run:
            summary.update(planning_pass=True, exit_code=0)
            return 0
        logdir = args.output / 'logs'
        logdir.mkdir()
        for name, command in tasks:
            record = {'name': name, 'argv': command, 'cwd': str(ROOT),
                      'started_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            write(logdir / (name + '.command.json'), record)
            with (logdir / (name + '.log')).open('w') as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if not name.endswith('-analysis'):
                summary.update(collection_command_started=True, device_commands_executed=None)
            code = completed.returncode
            record['exit_code'] = code
            write(logdir / (name + '.command.json'), record)
            (logdir / (name + '.exit-code')).write_text(str(code) + '\n')
            summary['stages'].append({'name': name, 'exit_code': code})
            write(args.output / 'SUMMARY.json', summary)
            print(name + ': exit ' + str(code), flush=True)
            if code:
                gate = failed_stage_gate(args.output, name)
                summary.update(failed_stage=name, exit_code=code if code > 0 else 1,
                               execution_pass=False, execution_failed=True, failed_stage_gate=gate,
                               scoped_qualification_pass=gate['scoped_qualification_pass'])
                return summary['exit_code']
        collect_verdict(args, summary)
        summary['exit_code'] = 0
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        summary['errors'].append(str(error))
        summary['exit_code'] = 2
        if summary['scoped_qualification_pass'] is not False:
            summary['scoped_qualification_pass'] = None
        if args.run:
            summary.update(execution_pass=False, execution_failed=True)
        return 2
    finally:
        write(args.output / 'SUMMARY.json', summary)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    sys.exit(main())
