#!/usr/bin/env python3
"""Build the preserved experimental Metal shaders without running GPU work.

Requires macOS, Python 3.9+, and full Xcode with the Metal compiler installed.
Compiler output, exact commands, source hashes, and product hashes stay in the
selected local output directory. This script does not build the QVAC framework.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bitnet.q_kernel_contract import load_contract

SHADERS = ROOT / "bitnet" / "shaders"
SAFE = ["-fmetal-math-mode=safe", "-ffp-contract=off"]
C8 = ["-DPOC_VARIANT=2", "-DPOC_ALIGNED_X=0",
      "-DPOC_SIMD_PER_GROUP=2", "-DPOC_ROWS_PER_SIMD=8"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def units(contract=None):
    contract = contract or load_contract()
    q4, c8 = (contract['variants'][name] for name in ('stock4x1', 'c8x2'))
    q_flags = ['-I', str(ROOT / Path(contract['config_path']).parent), '-Werror=macro-redefined']
    return [
        ("original-poc", "original/matvec.metal", [], "tq2_0_matvec_poc"),
        ("stock4x1", "stock4x1.metal", q_flags, q4['entry']),
        ("qcur0-c8x2", "c-expand.metal", q_flags + [
            '-DBITNET_CANONICAL_Q_C8=1', '-DPOC_VARIANT=2', '-DPOC_ALIGNED_X=0',
            "-I", str(SHADERS / "original")], c8['entry']),
        ("dual", "dual.metal", [], "ffn_dual_projection"),
        ("post", "post.metal", SAFE, "ffn_post_strict,ffn_shadow_copy"),
        ("up-epilogue", "up-epilogue.metal", [], "ffn_up_epilogue"),
        ("k0-postnorm-inplace", "k0-postnorm-inplace.metal", SAFE,
         "ffn_post_weighted_norm_inplace"),
        ("k0-postnorm-reference", "k0-postnorm-reference.metal", SAFE,
         "ffn_post_weighted_norm"),
        ("ffn-gate-up-8x2", "f8-gate-up.metal", C8 + [
            "-I", str(SHADERS / "include-expand"),
            "-Dtq2_0_matvec_poc=tq2_0_matvec_ffn_gate_up_8x2"],
         "tq2_0_matvec_ffn_gate_up_8x2,ffn_relu2_mul"),
        ("ffn-expand", "c-expand.metal", C8 + [
            "-I", str(SHADERS / "include-expand"),
            "-Dtq2_0_matvec_poc=tq2_0_matvec_ffn_expand"], "tq2_0_matvec_ffn_expand"),
    ]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", choices=["macosx", "iphoneos"], required=True)
    parser.add_argument("--output", type=Path,
                        help="Empty local output directory; default is outside this repository under ../bitnet-local")
    parser.add_argument("--developer-dir", type=Path,
                        help="Full Xcode Contents/Developer directory; otherwise use DEVELOPER_DIR or xcode-select")
    parser.add_argument("--deployment-target",
                        help="Defaults to 16.4 for iPhone and 13.3 for Mac")
    parser.add_argument("--emit-air", action="store_true",
                        help="Also save textual compiler AIR/LLVM; this is static compiler evidence")
    parser.add_argument("--plan", action="store_true",
                        help="Print commands only; do not invoke tools or write files")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S%z")
    output = (args.output or ROOT.parent / "bitnet-local" /
              ("shaders-" + timestamp + "-" + args.sdk)).expanduser().resolve()
    target = args.deployment_target or ("16.4" if args.sdk == "iphoneos" else "13.3")
    target_flag = ("-mios-version-min=" if args.sdk == "iphoneos" else
                   "-mmacosx-version-min=") + target
    compiler = ["xcrun", "--sdk", args.sdk, "metal"]
    linker = ["xcrun", "--sdk", args.sdk, "metallib"]
    common = ["-O3", target_flag, "-fmodules-cache-path=" + str(output / "modules")]
    plan = []
    products = {}
    contract = load_contract()
    q_units = {'stock4x1': contract['variants']['stock4x1'],
               'qcur0-c8x2': contract['variants']['c8x2']}
    for name, source, flags, symbols in units(contract):
        air = output / (name + ".air")
        lib = output / (name + ".metallib")
        plan.append((name + "-compile", compiler + common + flags +
                     [str(SHADERS / source), "-c", "-o", str(air)]))
        if args.emit_air or name in q_units:
            plan.append((name + "-llvm", compiler + common + flags +
                         [str(SHADERS / source), "-S", "-emit-llvm", "-o",
                          str(output / (name + ".ll"))]))
        plan.append((name + "-link", linker + [str(air), "-o", str(lib)]))
        products[lib.name] = {"source": source, "expected_symbols": symbols.split(",")}
        if name in q_units:
            products[lib.name]['q_contract'] = q_units[name]
    combined = output / "ffn-mechanisms.metallib"
    plan.append(("ffn-mechanisms-link", linker + [str(output / (name + ".air"))
                 for name in ["post", "up-epilogue", "dual"]] + ["-o", str(combined)]))
    products[combined.name] = {"sources": ["post.metal", "up-epilogue.metal", "dual.metal"],
                               "expected_symbols": ["ffn_post_strict", "ffn_shadow_copy",
                                                    "ffn_up_epilogue", "ffn_dual_projection"]}
    if args.plan:
        print(json.dumps({"sdk": args.sdk, "deployment_target": target,
                          "output": str(output), 'q_kernel_contract': contract, "commands": [
                              {"name": name, "argv": argv} for name, argv in plan]}, indent=2))
        return 0
    if platform.system() != "Darwin":
        raise RuntimeError("Metal compilation requires macOS and full Xcode")
    if output == ROOT or output == SHADERS or SHADERS in output.parents:
        raise RuntimeError("Select an artifact directory outside the shader source directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError("Refusing a nonempty output directory; select a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.developer_dir:
        developer = args.developer_dir.expanduser().resolve()
        if not (developer / "usr" / "bin" / "xcodebuild").is_file():
            raise RuntimeError("--developer-dir must identify a full Xcode Contents/Developer directory")
        env["DEVELOPER_DIR"] = str(developer)
    manifest = {
        "created": datetime.datetime.now().astimezone().isoformat(),
        "sdk": args.sdk, "deployment_target": target,
        "DEVELOPER_DIR": env.get("DEVELOPER_DIR"),
        "macOS": platform.mac_ver()[0], "machine": platform.machine(),
        "build_completed": False, "GPU_executed": False,
        'q_kernel_contract': contract,
        "sources": {str(p.relative_to(ROOT)): digest(p)
                    for p in sorted(SHADERS.rglob("*")) if p.is_file()},
        "commands": [], "products": {},
    }
    manifest['sources'][contract['config_path']] = contract['config_sha256']

    def save():
        (output / "BUILD.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run(name, argv):
        logfile = output / ("%03d-%s.log" % (len(manifest["commands"]), name))
        print("+ " + shlex.join(argv), flush=True)
        started = time.monotonic()
        with logfile.open("w") as stream:
            result = subprocess.run(argv, env=env, cwd=ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        record = {"name": name, "argv": argv, "exit_code": result.returncode,
                  "elapsed_seconds": time.monotonic() - started,
                  "log": str(logfile)}
        manifest["commands"].append(record)
        with (output / "COMMANDS.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        save()
        if result.returncode:
            print(logfile.read_text(errors="replace")[-5000:], file=sys.stderr)
            raise RuntimeError("%s failed with exit code %d; see %s" %
                               (name, result.returncode, logfile))
        return logfile.read_text(errors="replace").strip()

    save()
    try:
        if not manifest["DEVELOPER_DIR"]:
            manifest["DEVELOPER_DIR"] = run("developer-selection", ["xcode-select", "-p"])
        env["DEVELOPER_DIR"] = manifest["DEVELOPER_DIR"]
        manifest["Xcode"] = run("xcode-version", ["xcodebuild", "-version"])
        manifest["SDK_version"] = run("sdk-version", ["xcrun", "--sdk", args.sdk, "--show-sdk-version"])
        manifest["SDK_path"] = run("sdk-path", ["xcrun", "--sdk", args.sdk, "--show-sdk-path"])
        manifest["compiler"] = run("metal-version", compiler + ["--version"])
        for name, argv in plan:
            run(name, argv)
        for filename, description in products.items():
            path = output / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError("Missing or empty library: " + str(path))
            manifest["products"][filename] = dict(description, sha256=digest(path),
                                                    bytes=path.stat().st_size)
            if 'q_contract' in description:
                descriptor = description['q_contract']
                llvm = output / (path.stem + '.ll')
                text = llvm.read_text()
                defined = re.findall(r'^define\s+[^@\n]*@([A-Za-z_][A-Za-z_0-9]*)\(', text, re.M)
                entry = descriptor['entry']
                require(entry in defined and '!air.kernel' in text, 'Expected Q entry missing from compiler AIR')
                binary = path.read_bytes()
                require(entry.encode('ascii') in binary, 'Expected Q entry missing from metallib name table')
                forbidden = {'tq2_0_matvec_poc_c8x2'} | {d['entry'] for d in q_units.values() if d['entry'] != entry}
                require(all(name not in defined and name.encode('ascii') not in binary for name in forbidden),
                        'Q metallib contains a shared or mismatched entry')
                manifest['products'][filename]['symbol_validation'] = {
                    'AIR_defined_functions': defined, 'AIR_sha256': digest(llvm),
                    'expected_entry_in_AIR_and_metallib': True, 'forbidden_entries_absent': True,
                    'scope': 'Offline compiler AIR and metallib name table; no GPU pipeline creation',
                }
        if digest(ROOT / contract['config_path']) != contract['config_sha256']:
            raise RuntimeError('Canonical Q contract changed during shader build')
        manifest["build_completed"] = True
        manifest["exit_code"] = 0
        print("Build completed. No GPU correctness or performance test was run.")
        print(str(output / "BUILD.json"))
        return 0
    except Exception as error:
        manifest["error"] = str(error)
        manifest["exit_code"] = 1
        print(str(error), file=sys.stderr)
        return 1
    finally:
        save()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
