"""CPU-only canonical Q4 provenance and explicit historical-analysis profiles."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bitnet.q_kernel_contract import load_contract, variant_contract

SOURCE = 'bitnet/shaders/stock4x1.metal'
LEGACY_SOURCE_SHA = '99a6f27566340ff465d1ca6242a4faf5a1e08dda9b676de2622e8ff99e5fa028'
LEGACY_ENTRY = 'tq2_0_matvec_poc_c8x2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, text):
    if not condition:
        raise ValueError(text)


def q4_library_identity(library, manifest, sdk=None):
    """Runtime preparation accepts only the current canonical Q4 product."""
    library, manifest = Path(library).resolve(), Path(manifest).resolve()
    build = json.loads(manifest.read_text())
    desc = variant_contract('stock4x1')
    canonical = load_contract()
    require(build.get('build_completed') is True and build.get('exit_code') == 0, 'Q4 shader build incomplete')
    require(sdk is None or build.get('sdk') == sdk, 'Q4 shader SDK mismatch')
    require(build.get('q_kernel_contract') == canonical, 'Q4 build canonical contract differs from current source')
    product = build['products']['stock4x1.metallib']
    require(product['source'] == 'stock4x1.metal' and product['expected_symbols'] == [desc['entry']] and
            product.get('q_contract') == desc, 'Q4 product entry/geometry/contract mismatch')
    require(build['sources'].get(SOURCE) == sha(ROOT / SOURCE), 'Q4 shader source differs from current worktree')
    require(library.stat().st_size == product['bytes'] and sha(library) == product['sha256'], 'Q4 library differs from stock4x1 product')
    commands = [x for x in build['commands'] if x['name'] == 'stock4x1-compile']
    require(len(commands) == 1 and commands[0]['exit_code'] == 0 and
            str(ROOT / SOURCE) in commands[0]['argv'], 'Q4 compile provenance missing')
    return {'q_library': str(library), 'q_library_sha256': sha(library),
            'q_variant': desc['variant'], 'q_library_source_sha256': build['sources'][SOURCE],
            'q_library_manifest_sha256': sha(manifest), 'q_library_sdk': build['sdk'],
            'q_kernel_contract_version': desc['version'], 'q_kernel_config_sha256': desc['config_sha256'],
            'q_entry': desc['entry'], 'q_grid': desc['grid'], 'q_threads': desc['threads'],
            'q_rows_per_simd': desc['rows'], 'q_simd_per_group': desc['groups']}


def analysis_contract(profile='canonical-v1', legacy_shader_build=None):
    """Legacy names are analysis-only and require a preserved source-bound build."""
    if profile == 'canonical-v1':
        require(legacy_shader_build is None, 'Legacy build cannot qualify a canonical profile')
        return variant_contract('stock4x1')
    require(profile == 'legacy-shared-v0' and legacy_shader_build is not None,
            'Legacy replay requires an explicit preserved shader BUILD.json')
    path = Path(legacy_shader_build).resolve(); build = json.loads(path.read_text())
    product = build['products']['stock4x1.metallib']
    require(build.get('build_completed') is True and build.get('exit_code') == 0 and
            build['sources'].get(SOURCE) == LEGACY_SOURCE_SHA and product['source'] == 'stock4x1.metal' and
            product['expected_symbols'] == [LEGACY_ENTRY], 'Legacy shader source/entry provenance mismatch')
    commands = [x for x in build['commands'] if x['name'] == 'stock4x1-compile']
    require(len(commands) == 1 and commands[0]['exit_code'] == 0 and
            any(Path(x).name == 'stock4x1.metal' for x in commands[0]['argv']), 'Legacy compile provenance missing')
    desc = dict(variant_contract('stock4x1'))
    desc.update(version=0, entry=LEGACY_ENTRY, config_sha256=None, analysis_only_legacy=True,
                legacy_library_sha256=product['sha256'], legacy_source_sha256=LEGACY_SOURCE_SHA,
                legacy_build_sha256=sha(path), legacy_sdk=build['sdk'])
    return desc


def operator_q_contract():
    """Compact native fixture/qualification schema, derived from canonical data."""
    full = load_contract()
    keys = ('variant', 'entry', 'rows', 'groups', 'grid', 'threads')
    return {'version': full['version'], **{route: {key: full['variants'][variant][key] for key in keys}
            for route, variant in [('Q4', 'stock4x1'), ('C8', 'c8x2')]}}
