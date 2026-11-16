"""Fresh-process host import controls; no device, build or GPU calls."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class PhoneHostImports(unittest.TestCase):
    def fresh(self, argv):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as directory:
            return subprocess.run(argv, cwd=directory, env=env, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=30)

    def test_package_import_without_pythonpath(self):
        code = ("import importlib,sys;sys.path.insert(0,sys.argv[1]);"
                "m=importlib.import_module('bitnet.actquant.run_phone');"
                "print(m.__name__);print(m.q4_library_identity.__module__)")
        result = self.fresh([sys.executable, "-c", code, str(ROOT)])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(),
                         ["bitnet.actquant.run_phone", "bitnet.actquant.q4_contract"])

    def test_direct_script_help_without_pythonpath(self):
        result = self.fresh([sys.executable, str(ROOT / "bitnet/actquant/run_phone.py"),
                             "--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--operation", result.stdout)
        self.assertIn("--attribution-kind", result.stdout)


if __name__ == "__main__":
    unittest.main()
