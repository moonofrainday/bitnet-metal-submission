"""CPU-only schema and coverage controls for the shared Q shader contract."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bitnet.q_kernel_contract import CONFIG_PATH, ROOT, load_contract, variant_contract


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / CONFIG_PATH).read_text()

    def changed(self, old, new):
        self.assertEqual(self.source.count(old), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'q-config.h'
            path.write_text(self.source.replace(old, new))
            return load_contract(path)

    def test_distinct_entries_and_complete_row_ownership(self):
        contract = load_contract()
        self.assertNotEqual(contract['variants']['stock4x1']['entry'], contract['variants']['c8x2']['entry'])
        for variant in contract['variants'].values():
            owners = [0] * variant['nrows']
            for group in range(variant['grid'][0]):
                for simd in range(variant['groups']):
                    first = (group * variant['groups'] + simd) * variant['rows']
                    for row in range(variant['rows']):
                        owners[first + row] += 1
            self.assertTrue(all(count == 1 for count in owners))
            self.assertEqual(variant['weight_bytes'], variant['nrows'] * variant['row_bytes'])

    def test_unknown_variant(self):
        with self.assertRaisesRegex(ValueError, 'Unknown Q variant'):
            variant_contract('legacy-shared-symbol')

    def test_unknown_version(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported Q contract version'):
            self.changed('Q_CONFIG_VERSION 1', 'Q_CONFIG_VERSION 2')

    def test_duplicate_macro(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.changed('#pragma once', '#pragma once\n#define GGML_BITNET_Q4_ROWS 4')

    def test_missing_field(self):
        with self.assertRaisesRegex(ValueError, 'schema fields'):
            self.changed('#define GGML_BITNET_Q4_ROWS 4', '')

    def test_conditional_injection(self):
        with self.assertRaisesRegex(ValueError, 'Unknown Q contract directive'):
            self.changed('#pragma once', '#pragma once\n#if 1\n#endif')

    def test_include_injection(self):
        with self.assertRaisesRegex(ValueError, 'Unknown Q contract directive'):
            self.changed('#pragma once', '#pragma once\n#include "other.h"')

    def test_expression_is_not_evaluated(self):
        with self.assertRaisesRegex(ValueError, 'Unknown Q contract expression'):
            self.changed('(GGML_BITNET_Q_NROWS * GGML_BITNET_Q_ROW_BYTES)', '(1 << 32)')

    def test_signed_or_nonliteral_value(self):
        for old, new in [('Q4_ROWS 4', 'Q4_ROWS -4'), ('Q4_ROWS 4', 'Q4_ROWS (2+2)')]:
            with self.subTest(new=new), self.assertRaisesRegex(ValueError, 'positive literal'):
                self.changed(old, new)

    def test_entry_cannot_alias_other_variant(self):
        with self.assertRaisesRegex(ValueError, 'Entry does not identify'):
            self.changed('Q4_ENTRY tq2_0_matvec_stock4x1_v1', 'Q4_ENTRY tq2_0_matvec_c8x2_v1')

    def test_layout_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, 'layout does not match'):
            self.changed('Q4_ROWS 4', 'Q4_ROWS 8')

    def test_packed_column_tail_rejected(self):
        with self.assertRaisesRegex(ValueError, 'complete packed blocks'):
            self.changed('Q_NCOLS 2560', 'Q_NCOLS 2561')


if __name__ == '__main__':
    unittest.main()
