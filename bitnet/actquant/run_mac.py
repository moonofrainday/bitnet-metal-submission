#!/usr/bin/env python3
"""Fresh Mac numerical build/run; this script makes no performance claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def source_identity():
    files = {ROOT / "CMakeLists.txt", ROOT / "vendor/nlohmann/json.hpp"}
    for name in ("ggml", "src", "include", "cmake", "bitnet/actquant"):
        for path in (ROOT / name).rglob("*"):
            if path.is_file() and (path.name == "CMakeLists.txt" or path.suffix in
                                   {".c", ".cpp", ".m", ".mm", ".h", ".hpp", ".metal", ".py", ".cmake", ".S", ".s"}):
                files.add(path)
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--mode", choices=("operator", "full"), default="operator")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--reference-f32", type=Path)
    parser.add_argument("--reference-bf16", type=Path)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--developer-dir", default="/Applications/Xcode.app/Contents/Developer")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("This entry requires a physical Mac with full Xcode and Metal")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.mode == "full" and not all((args.model, args.tokens, args.reference_f32, args.reference_bf16)):
        parser.error("full mode requires --model, --tokens and both --reference-* paths")
    out = args.output.expanduser().resolve()
    if out.exists():
        parser.error("Use a new local output directory")
    if ROOT == out or ROOT in out.parents:
        parser.error("Raw numerical artifacts must be outside the source repository")
    fixture = args.fixture.expanduser().resolve()
    if not (fixture / "FIXTURES.json").is_file():
        parser.error("Missing prepared FIXTURES.json")
    out.mkdir(parents=True)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GGML_METAL_BITNET_") and key not in
           {"DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "SDKROOT"}}
    env["DEVELOPER_DIR"] = args.developer_dir
    commands = []
    failures = []

    def run(name, argv, allowed=(0,)):
        argv = list(map(str, argv))
        record = {"name": name, "argv": argv, "cwd": str(ROOT),
                  "DEVELOPER_DIR": env["DEVELOPER_DIR"]}
        commands.append(record)
        write(out / "COMMANDS.json", commands)
        with (out / (name + ".log")).open("w") as handle:
            code = subprocess.run(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                  stdout=handle, stderr=subprocess.STDOUT).returncode
        record["exit_code"] = code
        write(out / "COMMANDS.json", commands)
        (out / (name + ".exit-code")).write_text(str(code) + "\n")
        print(name, code, flush=True)
        if code not in allowed:
            raise RuntimeError(name + " failed; see " + str(out / (name + ".log")))
        return code

    try:
        run("host", ["sw_vers"])
        run("xcode", ["xcodebuild", "-version"])
        run("sdk", ["xcrun", "--sdk", "macosx", "--show-sdk-version"])
        run("git-head", ["git", "rev-parse", "HEAD"])
        run("git-status", ["git", "status", "--short"])
        run("git-patch", ["git", "diff", "--binary"])
        source = source_identity()
        write(out / "SOURCE.json", {"files": source, "fixture_manifest_sha256": sha(fixture / "FIXTURES.json")})
        build = out / "mac-build"
        run("configure", [args.cmake, "-S", ROOT / "bitnet/actquant", "-B", build,
                          "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF", "-DGGML_METAL=ON",
                          "-DLLAMA_BUILD_TESTS=ON"])
        run("build", [args.cmake, "--build", build, "--target", "bitnet-actquant-runner",
                      "test-backend-ops", "--parallel", str(args.jobs)])
        if source_identity() != source:
            raise RuntimeError("Source changed during the build; preserve logs and rerun from a frozen snapshot")
        binary = build / "bin/bitnet-actquant-runner"
        write(out / "BINARIES.json", {path.name: sha(path) for path in sorted((build / "bin").iterdir())
                                      if path.is_file() and not path.is_symlink()})
        for backend in ("CPU", "MTL0"):
            name = "operator-" + backend
            config = out / (name + ".json")
            write(config, {"operation": "operator", "backend": backend, "fixture": str(fixture), "trace": True})
            run(name, [binary, config, out / name])
            code = run(name + "-analysis", [sys.executable, ROOT / "bitnet/actquant/analyze_op.py",
                       "--fixture", fixture, "--actual", out / name, "--runtime", out / name / "RUN.json",
                       "--require-runtime", "--output", out / (name + "-analysis")], (0, 1, 2, 3))
            if code:
                failures.append({"stage": name, "strict_exit_code": code})
        test = build / "bin/test-backend-ops"
        run("backend-actquant", [test, "test", "-b", "MTL0", "-o", "ACT_QUANT"])
        run("backend-tq2", [test, "test", "-b", "MTL0", "-o", "MUL_MAT", "-p", "type_a=tq2_0"])
        diagnostics = []
        if args.mode == "full" and not failures:
            for workload, enabled, capture in (("teacher", False, False), ("teacher", True, True),
                                               ("teacher", True, False), ("corpus", False, False),
                                               ("corpus", True, True)):
                name = workload + ("-ON" if enabled else "-OFF") + ("-capture" if capture else "-plain")
                config = out / (name + ".json")
                write(config, {"operation": "model", "backend": "MTL0", "workload": workload,
                               "actquant": enabled, "capture": capture, "trace": capture,
                               "prefill": "batched" if workload == "teacher" else "sequential", "kv": "F16",
                               "model": str(args.model.expanduser().resolve()), "tokens": str(args.tokens.expanduser().resolve())})
                run(name, [binary, config, out / name])
                code = run(name + "-analysis", [sys.executable, ROOT / "bitnet/actquant/compare_model.py",
                           "--run", out / name, "--reference-f32", args.reference_f32.expanduser().resolve(),
                           "--reference-bf16", args.reference_bf16.expanduser().resolve(),
                           "--output", out / (name + "-analysis")], (0, 1, 2, 3))
                required = enabled and capture
                diagnostics.append({"stage": name, "strict_exit_code": code, "required_candidate_gate": required})
                if code and required:
                    failures.append({"stage": name, "strict_exit_code": code})
            off = json.loads((out / "teacher-OFF-plain/RUN.json").read_text())
            on = json.loads((out / "teacher-ON-plain/RUN.json").read_text())
            captured = json.loads((out / "teacher-ON-capture/RUN.json").read_text())
            neutral = len(on["logits"]) == len(captured["logits"]) == 33 and all(
                a["input_position"] == b["input_position"] and
                sha(out / "teacher-ON-plain" / a["path"]) == sha(out / "teacher-ON-capture" / b["path"])
                for a, b in zip(on["logits"], captured["logits"]))
            write(out / "PROBE_CONTROL.json", {"ON_capture_plain_logits_bitwise": neutral,
                                               "OFF_logit_vectors": len(off["logits"])})
            if not neutral:
                failures.append({"stage": "probe-control", "strict_exit_code": 1})
        if source_identity() != source:
            raise RuntimeError("Source changed during validation; results cannot qualify this snapshot")
        result = {"processing_completed": True, "scope": args.mode, "failures": failures,
                  "diagnostic_verdicts": diagnostics,
                  "required_model_gates": ["teacher-ON-capture", "corpus-ON-capture", "probe-control"] if args.mode == "full" else [],
                  "numerical_acceptance_pass": not failures, "phone_verified": None,
                  "legacy_V1_verdict_overwritten": False, "performance_measurement": False}
        write(out / "RESULT.json", result)
        return max([item["strict_exit_code"] for item in failures], default=0)
    except (OSError, RuntimeError, ValueError) as error:
        write(out / "RESULT.json", {"processing_completed": False, "error": str(error),
                                   "numerical_acceptance_pass": None, "exit_code": 2})
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
