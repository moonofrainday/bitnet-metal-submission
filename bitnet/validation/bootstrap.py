#!/usr/bin/env python3
"""Obtain pristine pinned QVAC source and run the standalone correctness smoke.

Build products and raw results are written outside the submission checkout. No
previous dylib or metallib is reused. This entry performs no speed benchmark.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
PIN = "2db8b046d555316019ae75d1be90484adc8b9330"
UPSTREAM = "https://github.com/tetherto/qvac-fabric-llm.cpp.git"


class StageFailure(Exception):
    def __init__(self, code, name):
        self.code = code if code > 0 else 2
        super().__init__("Stage failed: " + name)


def outside_checkout(path):
    return path != REPOSITORY and REPOSITORY not in path.parents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True,
                        help="New directory outside this checkout; source, build and raw results go here")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--qvac-repo", type=Path,
                       help="Existing local repository containing the pinned commit; never uses its build products")
    group.add_argument("--cache", type=Path,
                       help="Bare upstream cache, default: ~/Library/Caches/bitnet-metal-validation/qvac.git")
    parser.add_argument("--cmake", type=Path, help="CMake executable, otherwise use PATH")
    parser.add_argument("--developer-dir", default="/Applications/Xcode.app/Contents/Developer")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--real-inputs", type=Path,
                        help="Optional local input manifest; absent means explicitly model-free")
    parser.add_argument("--skip-backend-regression", action="store_true")
    args = parser.parse_args()
    work = args.work_root.expanduser().resolve()
    if work.exists() or not outside_checkout(work) or args.jobs < 1:
        parser.error("--work-root must be new and outside this checkout; --jobs must be positive")
    cache = (args.cache or Path.home() / "Library/Caches/bitnet-metal-validation/qvac.git").expanduser().resolve()
    if not args.qvac_repo and (not outside_checkout(cache) or cache == work or work in cache.parents or cache in work.parents):
        parser.error("upstream cache must be outside both this checkout and the new work root")
    work.mkdir(parents=True)
    report = {"schema_version": 1, "pinned_qvac_commit": PIN, "upstream_url": UPSTREAM,
              "started_UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "work_root": str(work), "performance_measured": False,
              "model_free_requested": args.real_inputs is None, "commands": [],
              "completed": False, "exit_code": None}

    def save():
        (work / "BOOTSTRAP.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, command):
        argv = list(map(str, command))
        log = work / ("bootstrap-" + name + ".log")
        entry = {"name": name, "argv": argv, "log": str(log)}
        report["commands"].append(entry)
        save()
        print("Running " + name, flush=True)
        with log.open("xb") as stream:
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
        entry["exit_code"] = result.returncode
        save()
        if result.returncode:
            raise StageFailure(result.returncode, name)
        return log

    try:
        if args.qvac_repo:
            repository = args.qvac_repo.expanduser().resolve(strict=True)
            run("verify-existing-pin", ["git", "-C", repository, "cat-file", "-e", PIN + "^{commit}"])
        else:
            repository = cache
            if not repository.exists():
                repository.parent.mkdir(parents=True, exist_ok=True)
                run("init-cache", ["git", "init", "--bare", repository])
            bare_log = run("verify-bare-cache", ["git", "-C", repository, "rev-parse", "--is-bare-repository"])
            if bare_log.read_text().strip() != "true":
                raise ValueError("--cache must identify a bare repository; use --qvac-repo for a normal checkout")
            # A named immutable commit is the trust anchor. Fetch directly from
            # the official URL, without depending on any mutable remote config.
            present = subprocess.run(["git", "-C", str(repository), "cat-file", "-e", PIN + "^{commit}"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            report["pin_already_cached"] = present
            if not present:
                run("fetch-pin", ["git", "-C", repository, "fetch", "--no-tags", "--depth=1", UPSTREAM, PIN])
            run("verify-fetched-pin", ["git", "-C", repository, "cat-file", "-e", PIN + "^{commit}"])
        command = [sys.executable, HERE / "smoke.py", "--qvac-repo", repository,
                   "--source", work / "upstream", "--build", work / "build",
                   "--output", work / "results", "--developer-dir", args.developer_dir,
                   "--jobs", args.jobs]
        if args.cmake:
            command += ["--cmake", args.cmake.expanduser().resolve()]
        if args.real_inputs:
            command += ["--real-inputs", args.real_inputs.expanduser().resolve()]
        if args.skip_backend_regression:
            command += ["--skip-backend-regression"]
        run("smoke", command)
        report.update(completed=True, exit_code=0)
        save()
        print("Standalone correctness passed; results: " + str(work / "results"))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        code = error.code if isinstance(error, StageFailure) else 2
        report.update(completed=False, exit_code=code, error=str(error))
        save()
        print("BOOTSTRAP_FAILED: " + str(error) + "; records: " + str(work), file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
