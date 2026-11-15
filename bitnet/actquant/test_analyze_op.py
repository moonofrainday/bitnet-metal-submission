#!/usr/bin/env python3
"""Reject an incorrect INT8 code even when the frozen value tolerance passes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE_SHA = "a8b3170f1824f010701f4e033fbc33d199bc2ea344ddbf2554594b66fada8b1c"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExitContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fixture = self.root / "fixture"
        self.actual = self.root / "actual"
        self.case = "tiny-epsilon"
        folder = self.fixture / self.case
        folder.mkdir(parents=True)
        output = self.actual / self.case
        output.mkdir(parents=True)
        scale = np.array([127 / np.float32(1e-5)], dtype="<f4")
        values = {"input": ("x.f32", np.zeros(4, dtype="<f4")),
                  "expected": ("expected.f32", np.zeros(4, dtype="<f4")),
                  "scale": ("scale.f32", scale), "int8_codes": ("q.s8", np.zeros(4, dtype="i1"))}
        case = {"name": self.case, "elements": 4, "width": 4, "rows": 1, "ne": [4, 1, 1, 1]}
        for role, (name, a) in values.items():
            path = folder / name
            a.tofile(path)
            case[role] = f"{self.case}/{name}"
            case[role + "_sha256"] = sha(path)
        manifest = {"schema": 1, "epsilon": 1e-5, "official_source_sha256": SOURCE_SHA, "cases": [case]}
        (self.fixture / "FIXTURES.json").write_text(json.dumps(manifest))
        np.zeros(4, dtype="<f4").tofile(output / "y.f32")

    def tearDown(self):
        self.temp.cleanup()

    def run_check(self, name):
        path = self.root / name
        result = subprocess.run([sys.executable, str(HERE / "analyze_op.py"), "--fixture", str(self.fixture),
                                 "--actual", str(self.actual), "--output", str(path)], capture_output=True, text=True)
        return result, json.loads((path / "SUMMARY.json").read_text()) if path.is_dir() else None

    def test_all_pass(self):
        result, summary = self.run_check("pass")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(summary["numerical_pass"])
        self.assertIsNone(summary["execution_pass"])
        self.assertIsNone(summary["model_semantics_pass"])

    def test_wrong_grid_code_cannot_hide_in_absolute_tolerance(self):
        y = np.zeros(4, dtype="<f4")
        y[1] = np.float32(1 / np.float32(127 / np.float32(1e-5)))
        y.tofile(self.actual / self.case / "y.f32")
        result, summary = self.run_check("wrong-code")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(summary["cases"][0]["frozen_tolerance_failures"], 0)
        self.assertEqual(summary["cases"][0]["recovered_integer_code_mismatches"], 1)
        self.assertFalse(summary["numerical_pass"])

    def test_nonfinite_fails(self):
        y = np.zeros(4, dtype="<f4")
        y[2] = np.nan
        y.tofile(self.actual / self.case / "y.f32")
        result, summary = self.run_check("nonfinite")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(summary["cases"][0]["nonfinite"], 1)

    def test_missing_output_is_not_a_pass(self):
        (self.actual / self.case / "y.f32").unlink()
        result, summary = self.run_check("missing")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIsNone(summary)

    def test_changed_input_hash_is_not_a_pass(self):
        np.ones(4, dtype="<f4").tofile(self.fixture / self.case / "x.f32")
        result, summary = self.run_check("changed")
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIsNone(summary)


if __name__ == "__main__":
    unittest.main()
