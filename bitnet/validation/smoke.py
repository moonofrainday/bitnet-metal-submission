#!/usr/bin/env python3
"""Clean build, CPU reference checks, and standalone Metal correctness; no benchmark."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PIN = "2db8b046d555316019ae75d1be90484adc8b9330"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--qvac-repo", type=Path, help="Existing repository to create a missing pinned worktree from")
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-inputs", type=Path)
    parser.add_argument("--cmake", type=Path)
    parser.add_argument("--developer-dir", default="/Applications/Xcode.app/Contents/Developer")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--skip-backend-regression", action="store_true")
    args = parser.parse_args()
    source, build, output = args.source.resolve(), args.build.resolve(), args.output.resolve()
    require(not output.exists(), "Refuse existing output: " + str(output))
    require(not build.exists() or not any(build.iterdir()), "Refuse nonempty build; old binaries cannot be reused")
    require(output != source and source not in output.parents, "Keep output outside the QVAC source")
    if args.real_inputs:
        require(args.real_inputs.is_file(), "Missing requested real input manifest")
    output.mkdir(parents=True)
    (output / "stages").mkdir()
    env = os.environ.copy()
    removed = [key for key in env if key.startswith(("DYLD_", "MTL_", "GGML_", "CCACHE_"))
               or key in {"CFLAGS", "CXXFLAGS", "LDFLAGS", "CMAKE_PREFIX_PATH"}]
    for key in removed:
        env.pop(key, None)
    env.update(DEVELOPER_DIR=args.developer_dir, CCACHE_DISABLE="1", PYTHONDONTWRITEBYTECODE="1")
    report = {
        "scope": "Mac clean-build standalone operator correctness; no inference performance claim",
        "source": str(source), "source_commit_required": PIN, "build": str(build), "output": str(output),
        "started_UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "raw_publication_allowed": False, "performance_benchmark_performed": False,
        "model_output_pass": None, "old_project_binary_reuse_allowed": False,
        "sanitized_environment_keys": sorted(removed), "stages": [], "overall_pass": False,
    }
    workflow_names = ("smoke.py", "build.py", "analyze.py", "prepare-real-inputs.py")
    report["workflow_source_sha256"] = {name: sha(HERE / name) for name in workflow_names}
    if args.real_inputs:
        report["real_input_manifest"] = {"path": str(args.real_inputs.resolve()), "sha256": sha(args.real_inputs)}

    def save():
        (output / "RUN.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, command):
        command = list(map(str, command))
        log = output / "stages" / (label + ".log")
        record = {"stage": label, "argv": command, "log": str(log)}
        (output / "stages" / (label + ".command.json")).write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"stage_start": label, "log": str(log)}), flush=True)
        start = time.monotonic()
        with log.open("xb") as stream:
            process = subprocess.run(command, cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record.update(exit_code=process.returncode, elapsed_seconds=time.monotonic() - start)
        report["stages"].append(record)
        (output / "stages" / (label + ".exit.json")).write_text(json.dumps(record, indent=2) + "\n")
        save()
        print(json.dumps({"stage_end": label, "actual_exit_code": process.returncode}), flush=True)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        return log

    try:
        if not source.exists():
            require(args.qvac_repo and args.qvac_repo.is_dir(), "Missing source requires --qvac-repo")
            run("create-worktree", ["git", "-C", args.qvac_repo, "worktree", "add", "--detach", source, PIN])
        command = [sys.executable, HERE / "build.py", "--source", source, "--build", build,
                   "--output", output / "build", "--developer-dir", args.developer_dir, "--jobs", args.jobs]
        if args.cmake:
            command += ["--cmake", args.cmake]
        run("build", command)
        manifest_path = output / "build/BUILD.json"
        manifest = json.loads(manifest_path.read_text())
        require(manifest.get("build_completed") is True and manifest.get("source_commit") == PIN,
                "Fresh build identity/completion is unverified")
        report["build_manifest"] = {"path": str(manifest_path), "sha256": sha(manifest_path)}
        harness = Path(manifest["harness"]["path"])
        native = [harness, "--poc", manifest["shaders"]["poc"]["metallib"]["path"],
                  "--stock4x1", manifest["shaders"]["stock4x1"]["metallib"]["path"],
                  "--expected-build", build]
        if args.real_inputs:
            native += ["--inputs", args.real_inputs.resolve()]
        run("cpu-reference", native + ["--cpu-only", "--output", output / "cpu"])
        run("gpu-correctness", native + ["--output", output / "gpu"])
        result = json.loads((output / "gpu/summary.json").read_text())
        for flag in ("execution_pass", "parity_pass", "fp64_pass", "knownanswer_pass", "coverage_pass", "overall_pass"):
            require(result.get(flag) is True, "Missing/failing native required flag: " + flag)
        expected = {"known-answer", "dense-seed42"}
        if args.real_inputs:
            expected |= {"real-step-1", "real-step-16", "real-step-32"}
        actual = {row["id"] for row in result["cases"]}
        require(actual == expected, "Requested case coverage mismatch: " + str(actual))
        require(result.get("model_output_pass") is None, "Standalone run cannot claim model acceptance")
        report["native_required_flags"] = {key: result[key] for key in
            ("execution_pass", "parity_pass", "fp64_pass", "knownanswer_pass", "coverage_pass", "overall_pass")}
        analysis_command = [sys.executable, HERE / "analyze.py", "--input", output / "gpu", "--output", output / "analysis"]
        if not args.real_inputs:
            analysis_command += ["--model-free"]
        run("independent-analysis", analysis_command)
        if not args.skip_backend_regression:
            log = run("backend-TQ2-MUL_MAT", [manifest["backend_test"]["path"], "test", "-b", "MTL0", "-o", "MUL_MAT", "-p", "type_a=tq2_0"])
            text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
            counts = re.findall(r"(\d+)/(\d+) tests passed", text)
            require(len(counts) == 1, "Missing/ambiguous backend test count")
            passed, total = map(int, counts[0])
            unsupported = len(re.findall(r"MUL_MAT[^\n]*not supported", text))
            require(total > 0 and passed == total and "Backend MTL0: OK" in text,
                    "Backend filter executed zero tests or did not pass on MTL0")
            report["backend_regression"] = {"backend": "MTL0", "passed": passed, "executed": total,
                "failed": total - passed, "unsupported": unsupported,
                "unselected_cases": None, "note": "Nonselected operations/backends are outside this filtered count"}
        else:
            report["backend_regression"] = None
        require(report["workflow_source_sha256"] == {name: sha(HERE / name) for name in workflow_names},
                "Workflow scripts changed during execution; preserve this run and use a fresh attempt")
        report.update(overall_pass=True, exit_code=0, finished_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        print(json.dumps({"overall_pass": True, "case_count": len(actual), "model_output_pass": None,
                          "performance_benchmark_performed": False, "result": str(output / "RUN.json")}, indent=2))
        return 0
    except Exception as error:
        code = error.returncode if isinstance(error, subprocess.CalledProcessError) else 2
        report.update(overall_pass=False, exit_code=code, error=str(error),
                      finished_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        print(json.dumps({"overall_pass": False, "actual_exit_code": code, "error": str(error),
                          "result": str(output / "RUN.json")}), file=sys.stderr)
        return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as error:
        print("SMOKE_BLOCKED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
