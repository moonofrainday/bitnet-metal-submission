"""CPU-only Q4 shader provenance and replay-contract checks."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYMBOL = 'tq2_0_matvec_poc_c8x2'
SOURCE = 'bitnet/shaders/stock4x1.metal'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, text):
    if not condition:
        raise ValueError(text)


def q4_library_identity(library, manifest, sdk=None):
    """Bind even an alias library to freshly compiled stock4x1, not C8's same symbol."""
    library, manifest = Path(library).resolve(), Path(manifest).resolve()
    build = json.loads(manifest.read_text())
    require(build.get('build_completed') is True and build.get('exit_code') == 0, 'Q4 shader build incomplete')
    require(sdk is None or build.get('sdk') == sdk, 'Q4 shader SDK mismatch')
    product = build['products']['stock4x1.metallib']
    require(product['source'] == 'stock4x1.metal' and product['expected_symbols'] == [SYMBOL], 'Q4 variant mismatch')
    require(build['sources'].get(SOURCE) == sha(ROOT / SOURCE), 'Q4 shader source differs from current worktree')
    require(library.stat().st_size == product['bytes'] and sha(library) == product['sha256'], 'Q4 library differs from stock4x1 product')
    commands = [x for x in build['commands'] if x['name'] == 'stock4x1-compile']
    require(len(commands) == 1 and commands[0]['exit_code'] == 0 and
            str(ROOT / SOURCE) in commands[0]['argv'], 'Q4 compile provenance missing')
    return {'q_library': str(library), 'q_library_sha256': sha(library),
            'q_variant': 'stock4x1', 'q_library_source_sha256': build['sources'][SOURCE],
            'q_library_manifest_sha256': sha(manifest), 'q_library_sdk': build['sdk']}
