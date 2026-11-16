"""Read the versioned Q-only Metal contract without evaluating C expressions."""
import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = 'ggml/src/ggml-metal/ggml-metal-bitnet-q-config.h'
INTEGER_FIELDS = ('Q_CONFIG_VERSION', 'Q_SIMD_WIDTH', 'Q_NROWS', 'Q_NCOLS',
                  'Q_BLOCK_ELEMENTS', 'Q_BLOCK_BYTES', 'Q4_ROWS', 'Q4_NSG',
                  'C8_ROWS', 'C8_NSG')
EXPRESSIONS = {
    'Q_ROW_BYTES': '(GGML_BITNET_Q_NCOLS/GGML_BITNET_Q_BLOCK_ELEMENTS*GGML_BITNET_Q_BLOCK_BYTES)',
    'Q_WEIGHT_BYTES': '(GGML_BITNET_Q_NROWS*GGML_BITNET_Q_ROW_BYTES)',
    'Q_STRINGIFY_IMPL(value)': '#value',
    'Q_STRINGIFY(value)': 'GGML_BITNET_Q_STRINGIFY_IMPL(value)',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_contract(header=None):
    """Reject unknown, duplicate, conditional or incompatible schema definitions."""
    path = Path(header) if header is not None else ROOT / CONFIG_PATH
    content = path.read_bytes()
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', content.decode('ascii'), flags=re.S)
    macros = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line == '#pragma once':
            continue
        match = re.fullmatch(r'#define\s+GGML_BITNET_(\w+(?:\(value\))?)\s+(.+)', line)
        require(match is not None, 'Unknown Q contract directive: ' + line)
        name, value = match.groups()
        require(name not in macros, 'Duplicate Q contract macro: ' + name)
        macros[name] = value.strip()
    expected = set(INTEGER_FIELDS) | set(EXPRESSIONS) | {'Q4_ENTRY', 'C8_ENTRY'}
    require(set(macros) == expected, 'Q contract schema fields differ')
    values = {}
    for name in INTEGER_FIELDS:
        require(re.fullmatch(r'[1-9][0-9]*', macros[name]) is not None,
                'Expected a positive literal in ' + name)
        values[name] = int(macros[name])
    for name, expression in EXPRESSIONS.items():
        require(re.sub(r'\s+', '', macros[name]) == expression,
                'Unknown Q contract expression: ' + name)
    require(values['Q_CONFIG_VERSION'] == 1, 'Unsupported Q contract version')
    require(values['Q_SIMD_WIDTH'] == 32 and values['Q_BLOCK_ELEMENTS'] == 256 and
            values['Q_BLOCK_BYTES'] == 66, 'Unsupported SIMD or TQ2_0 representation')
    require(values['Q_NCOLS'] % values['Q_BLOCK_ELEMENTS'] == 0,
            'Q columns must contain complete packed blocks')
    config_sha = hashlib.sha256(content).hexdigest()
    variants = {}
    for variant, prefix in [('stock4x1', 'Q4'), ('c8x2', 'C8')]:
        rows, groups = values[prefix + '_ROWS'], values[prefix + '_NSG']
        entry = macros[prefix + '_ENTRY']
        require(re.fullmatch(r'tq2_0_matvec_' + variant + r'_v1', entry) is not None,
                'Entry does not identify the versioned Q variant')
        named_layout = re.search(r'(\d+)x(\d+)$', variant)
        require((rows, groups) == tuple(map(int, named_layout.groups())),
                'Q layout does not match its variant identity')
        require(rows <= values['Q_SIMD_WIDTH'] and groups in (1, 2, 4) and
                values['Q_NROWS'] % (rows * groups) == 0, 'Incomplete Q row coverage')
        row_bytes = values['Q_NCOLS'] // values['Q_BLOCK_ELEMENTS'] * values['Q_BLOCK_BYTES']
        variants[variant] = {
            'version': values['Q_CONFIG_VERSION'], 'variant': variant, 'entry': entry,
            'rows': rows, 'groups': groups, 'width': values['Q_SIMD_WIDTH'],
            'simd_width': values['Q_SIMD_WIDTH'],
            'grid': [values['Q_NROWS'] // (rows * groups), 1, 1],
            'threads': [values['Q_SIMD_WIDTH'], groups, 1],
            'shape': [values['Q_NROWS'], values['Q_NCOLS']],
            'nrows': values['Q_NROWS'], 'ncols': values['Q_NCOLS'],
            'block_elements': values['Q_BLOCK_ELEMENTS'], 'block_bytes': values['Q_BLOCK_BYTES'],
            'row_bytes': row_bytes, 'weight_bytes': values['Q_NROWS'] * row_bytes,
            'config_path': CONFIG_PATH, 'config_sha256': config_sha,
        }
    require(variants['stock4x1']['entry'] != variants['c8x2']['entry'], 'Q entries must differ')
    return {'version': values['Q_CONFIG_VERSION'], 'config_path': CONFIG_PATH,
            'config_sha256': config_sha, 'variants': variants}


def variant_contract(variant, header=None):
    contract = load_contract(header)
    require(variant in contract['variants'], 'Unknown Q variant: ' + str(variant))
    return contract['variants'][variant]
