#!/usr/bin/env python3
"""Build the clean fixed-shape PoC from pinned source; never execute GPU work."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PIN = "2db8b046d555316019ae75d1be90484adc8b9330"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def identity(path):
    p = Path(path)
    return {"path": str(p), "resolved": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha(p)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cmake", type=Path, help="CMake executable; defaults to CMake on PATH")
    parser.add_argument("--developer-dir", type=Path, default=Path("/Applications/Xcode.app/Contents/Developer"))
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    a = parser.parse_args()
    if a.cmake is None:
        found = shutil.which("cmake")
        a.cmake = Path(found) if found else None
    source, build, output = a.source.resolve(), a.build.resolve(), a.output.resolve()
    require(platform.system() == "Darwin", "Build requires the native Darwin host")
    require(a.jobs > 0, "--jobs must be positive")
    require(source.is_dir(), "Missing pinned QVAC source directory")
    require(a.cmake is not None and a.cmake.is_file(), "CMake unavailable: install it on PATH or pass --cmake /absolute/path/to/cmake")
    a.cmake = a.cmake.resolve()
    require(not output.exists(), "Refuse existing output: " + str(output))
    require(not build.exists() or not any(build.iterdir()), "Refuse nonempty build; no binary reuse: " + str(build))
    require(source != build and source not in build.parents, "Use a build directory outside upstream source")
    require(source != output and source not in output.parents, "Use output outside upstream source")
    output.mkdir(parents=True)
    (output / "commands").mkdir()
    (output / "shaders").mkdir()
    (output / "bin").mkdir()
    env = os.environ.copy()
    removed = {}
    for key in ("DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES"):
        if key in env:
            removed[key] = True
            del env[key]
    env.update(DEVELOPER_DIR=str(a.developer_dir.resolve()), CCACHE_DISABLE="1")
    report = {"build_completed": False, "GPU_executed": False, "timing_performed": False,
              "started_UTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "package": str(HERE), "source": str(source), "build": str(build), "output": str(output),
              "source_pin": PIN, "commands": [], "removed_dyld_environment_keys": sorted(removed),
              "stock_compilation": "Fresh dylib embeds pinned Metal SOURCE chunks. Native stock PSO compilation happens only when the separate GPU harness initializes its fresh QVAC runtime.",
              "old_dylib_or_metallib_reuse_allowed": False}
    stage = "preflight"

    def record():
        (output / "BUILD.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(name, argv, cwd=None):
        nonlocal stage
        stage = name
        argv = list(map(str, argv))
        stem = f"{len(report['commands']):02d}-{name}"
        log = output / "commands" / (stem + ".log")
        entry = {"name": name, "argv": argv, "cwd": str(cwd or HERE), "log": str(log),
                 "DEVELOPER_DIR": env["DEVELOPER_DIR"], "CCACHE_DISABLE": "1"}
        (output / "commands" / (stem + ".argv.json")).write_text(json.dumps(entry, indent=2) + "\n")
        start = time.monotonic()
        with log.open("xb") as stream:
            stream.write((shlex.join(argv) + "\n").encode())
            stream.flush()
            result = subprocess.run(argv, cwd=cwd or HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
        entry.update(exit_code=result.returncode, elapsed_seconds=time.monotonic() - start)
        report["commands"].append(entry)
        (output / "commands" / (stem + ".exit.json")).write_text(json.dumps(entry, indent=2) + "\n")
        record()
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, argv)
        return log

    def capture(name, argv):
        log = run(name, argv)
        return log.read_text().partition("\n")[2]

    try:
        for name, argv in (("uname", ["uname", "-s"]), ("macos", ["sw_vers"]),
                           ("xcode", ["xcodebuild", "-version"]),
                           ("sdk", ["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
                           ("metal-version", ["xcrun", "--sdk", "macosx", "metal", "--version"]),
                           ("cmake-version", [a.cmake, "--version"])):
            run(name, argv)
        head = capture("source-head", ["git", "-C", source, "rev-parse", "HEAD"]).strip()
        status = capture("source-clean-before", ["git", "-C", source, "status", "--porcelain=v1", "--untracked-files=all"])
        report.update(source_commit=head, source_status_before=status)
        require(head == PIN and not status, "Upstream must be the exact clean pinned commit")
        source_manifest = json.loads((HERE / "SOURCES.json").read_text())
        for item in source_manifest["copies"]:
            require(sha(HERE / item["path"]) == item["sha256"], "Package source identity changed: " + item["path"])
        report["upstream_source_files"] = {str(p.relative_to(source)): identity(p) for p in
            (source / "ggml/src/ggml-metal/kernels/mul_mv.metal", source / "ggml/src/ggml-common.h",
             source / "ggml/src/ggml-metal/ggml-metal-impl.h", source / "ggml/src/ggml-metal/ggml-metal-device.cpp",
             source / "ggml/src/ggml-metal/ggml-metal-device.m", source / "ggml/src/ggml-metal/CMakeLists.txt")}
        flags = {
            "CMAKE_BUILD_TYPE": "Release", "CMAKE_OSX_ARCHITECTURES": "arm64",
            "BUILD_SHARED_LIBS": "ON", "LLAMA_USE_SYSTEM_GGML": "OFF",
            "GGML_METAL": "ON", "GGML_METAL_EMBED_LIBRARY": "ON", "GGML_METAL_SHADER_DEBUG": "OFF",
            "GGML_CCACHE": "OFF", "GGML_BACKEND_DL": "OFF", "GGML_OPENMP": "OFF",
            "GGML_CPU_KLEIDIAI": "OFF", "LLAMA_BUILD_COMMON": "ON", "LLAMA_BUILD_TESTS": "ON",
            "LLAMA_BUILD_TOOLS": "OFF", "LLAMA_BUILD_EXAMPLES": "OFF", "LLAMA_BUILD_SERVER": "OFF",
            "LLAMA_BUILD_APP": "OFF", "LLAMA_BUILD_UI": "OFF", "LLAMA_OPENSSL": "OFF",
        }
        report["cmake_flags"] = flags
        run("configure", [a.cmake, "-S", source, "-B", build] + ["-D" + k + "=" + v for k, v in flags.items()])
        run("backend-build", [a.cmake, "--build", build, "--config", "Release", "--target",
                               "test-backend-ops", "ggml-metal", "ggml-base", "--parallel", str(a.jobs)])
        cache = build / "CMakeCache.txt"
        report["cmake_cache"] = identity(cache)
        report["backend_compile_commands"] = identity(build / "compile_commands.json")
        effective = {}
        for line in cache.read_text().splitlines():
            if line and not line.startswith(("#", "//")) and "=" in line and ":" in line.split("=", 1)[0]:
                key = line.split(":", 1)[0]
                if key in flags:
                    effective[key] = line.split("=", 1)[1]
        report["effective_cmake_flags"] = effective
        require(all(effective.get(k) == v for k, v in flags.items()), "Effective CMake flags differ from explicit recipe")
        report["stock_embedded_source"] = identity(build / "ggml/src/ggml-metal/autogenerated/ggml-metal-embed-mul_mv.metal")
        custom = {}
        for label, filename, symbol, grid, threads in (
                ("poc", "matvec.metal", "tq2_0_matvec_poc", [320, 1, 1], [32, 2, 1]),
                ("stock4x1", "stock4x1.metal", "tq2_0_matvec_poc_c8x2", [640, 1, 1], [32, 1, 1])):
            src = HERE / "shaders" / filename
            air, library = output / "shaders" / (label + ".air"), output / "shaders" / (label + ".metallib")
            run(label + "-metal", ["xcrun", "--sdk", "macosx", "metal", "-O3",
                 "-fmodules-cache-path=" + str(output / "metal-modules"), "-I", HERE, "-c", src, "-o", air])
            run(label + "-metallib", ["xcrun", "--sdk", "macosx", "metallib", air, "-o", library])
            custom[label] = {"source": identity(src), "air": identity(air), "metallib": identity(library),
                             "symbol": symbol, "grid": grid, "threads": threads}
        report["shaders"] = custom
        stage = "native-source-check"
        for filename in ("main.mm", "reference.cpp", "reference.h"):
            require((HERE / filename).is_file(), "Missing native harness source: " + filename)
        library_dir = build / "bin"
        native = output / "bin" / "clean-poc-harness"
        command = ["xcrun", "--sdk", "macosx", "clang++", "-std=c++17", "-O2", "-fno-fast-math",
                   "-ffp-contract=off", "-fobjc-arc", "-Wall", "-Wextra", "-Werror",
                   "-I", HERE, "-I", source / "ggml/include", "-I", source / "ggml/src",
                   "-I", source / "ggml/src/ggml-metal", HERE / "main.mm", HERE / "reference.cpp",
                   "-L", library_dir, "-Wl,-rpath," + str(library_dir),
                   "-lggml-metal", "-lggml", "-lggml-base", "-framework", "Foundation", "-framework", "Metal",
                   "-o", native]
        run("native-build", command)
        dylibs = sorted({p.resolve() for p in library_dir.glob("*.dylib")})
        report["dylibs"] = [identity(p) for p in dylibs]
        report["harness"] = identity(native)
        report["backend_test"] = identity(library_dir / "test-backend-ops")
        linkage = {}
        forbidden = []
        for p in [native, library_dir / "test-backend-ops"] + dylibs:
            text = capture("otool-L-" + p.name, ["otool", "-L", p])
            linkage[str(p)] = text
            for line in text.splitlines()[1:]:
                dependency = line.strip().split(" (", 1)[0]
                if dependency.startswith("/") and not dependency.startswith(("/usr/lib/", "/System/Library/")):
                    if not Path(dependency).is_relative_to(library_dir):
                        forbidden.append({"binary": str(p), "dependency": dependency})
        report["linkage"] = linkage
        report["harness_load_commands"] = capture("harness-load-commands", ["otool", "-l", native])
        rpaths = []
        lines = report["harness_load_commands"].splitlines()
        for i, line in enumerate(lines):
            if "cmd LC_RPATH" in line:
                for following in lines[i + 1:i + 4]:
                    if following.strip().startswith("path "):
                        rpaths.append(following.strip()[5:].rsplit(" (offset ", 1)[0])
        require(rpaths == [str(library_dir)], "Unexpected harness rpath(s): " + str(rpaths))
        require(not forbidden, "Non-system dependency outside fresh build: " + str(forbidden))
        report["fresh_library_link_audit"] = {"pass": True, "rpaths": rpaths, "forbidden_dependencies": forbidden,
            "scope": "Static load commands only. The GPU run must separately record loaded ggml image paths."}
        report["package_source_files"] = {name: identity(HERE / name) for name in
             ("build.py", "stock-runtime.h", "contract.h", "main.mm", "reference.cpp", "reference.h", "SOURCES.json")}
        after = capture("source-clean-after", ["git", "-C", source, "status", "--porcelain=v1", "--untracked-files=all"])
        report["source_status_after"] = after
        require(not after, "Build changed upstream source")
        report.update(build_completed=True, exit_code=0, finished_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        record()
        print(json.dumps({"build_completed": True, "GPU_executed": False, "manifest": str(output / "BUILD.json"),
                          "harness": str(native), "shaders": {k: v["metallib"]["path"] for k, v in custom.items()}}, indent=2))
        return 0
    except Exception as error:
        code = error.returncode if isinstance(error, subprocess.CalledProcessError) else 2
        report.update(build_completed=False, failed_stage=stage, error=str(error), exit_code=code,
                      finished_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        record()
        print(json.dumps({"build_completed": False, "failed_stage": stage, "error": str(error),
                          "exit_code": code, "manifest": str(output / "BUILD.json")}), file=sys.stderr)
        return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError) as error:
        print("BUILD_BLOCKED: " + str(error), file=sys.stderr)
        raise SystemExit(2)
