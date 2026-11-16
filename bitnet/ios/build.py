#!/usr/bin/env python3
"""Build a fresh, source-only iPhoneOS package. Does not install or launch an app."""
from pathlib import Path
import argparse
import datetime
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys

from common import ROOT, Commands, developer_environment, require, sha256, write_json
sys.path.insert(0, str(ROOT))
from bitnet.q_kernel_contract import load_contract

HERE = Path(__file__).resolve().parent


def source_identity():
    files = {ROOT / "CMakeLists.txt"}
    for directory in ("cmake", "include", "src", "ggml/include", "ggml/src", "vendor/hash"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and (path.name == "CMakeLists.txt" or path.suffix in
                                   {".h", ".hpp", ".c", ".cpp", ".m", ".mm", ".metal", ".cmake", ".s", ".S"}):
                files.add(path)
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(files)}


def package_identity():
    files = {p for p in HERE.rglob("*") if p.is_file() and p.suffix in {".py", ".mm", ".h", ".swift"}}
    files.update(ROOT / "bitnet/validation" / name for name in ("reference.cpp", "reference.h", "contract.h", "stock-runtime.h"))
    files.update(p for p in (ROOT / "bitnet/actquant").glob("*") if p.is_file() and p.suffix in {".cpp", ".h"})
    files.add(ROOT / "vendor/nlohmann/json.hpp")
    files.add(ROOT / "ggml/src/ggml-metal/ggml-metal-bitnet-q-config.h")
    files.add(ROOT / "bitnet/q_kernel_contract.py")
    files.add(ROOT / "bitnet/tests/q_pipeline_contract.cpp")
    files.add(ROOT / "ggml/src/ggml-metal/ggml-metal-bitnet-q-contract.h")
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(files)}


def build_framework(output, commands, minimum, jobs, identity):
    cmake = output / "cmake-iphoneos"
    flags = ["-DBUILD_SHARED_LIBS=OFF", "-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON",
             "-DGGML_METAL_TARGET_OS=ios", "-DGGML_NATIVE=OFF", "-DGGML_OPENMP=OFF", "-DGGML_BLAS=ON",
             "-DLLAMA_BUILD_APP=OFF", "-DLLAMA_BUILD_COMMON=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TOOLS=OFF",
             "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_MTMD=OFF",
             "-DLLAMA_OPENSSL=OFF", "-DCMAKE_SYSTEM_NAME=iOS", "-DCMAKE_OSX_SYSROOT=iphoneos",
             "-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_OSX_DEPLOYMENT_TARGET=" + minimum,
             "-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO",
             "-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO",
             "-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY=", "-DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM="]
    commands.run("framework-configure", ["cmake", "-S", ROOT, "-B", cmake, "-G", "Xcode", *flags])
    commands.run("framework-build", ["cmake", "--build", cmake, "--config", "Release", "--target", "llama",
                                      "--parallel", str(jobs)])
    libraries = []
    for name in ("llama", "ggml", "ggml-base", "ggml-cpu", "ggml-metal", "ggml-blas"):
        matches = list(cmake.rglob("lib" + name + ".a"))
        require(len(matches) == 1, "Expected exactly one fresh archive for " + name + ": " + str(matches))
        libraries.extend(matches)
    artifacts = output / "framework"
    framework = artifacts / "llama.framework"
    (framework / "Headers").mkdir(parents=True)
    (framework / "Modules").mkdir()
    archive = artifacts / "combined.a"
    commands.run("framework-archive", ["xcrun", "libtool", "-static", "-o", archive, *libraries])
    _, sdk = commands.run("framework-sdk-path", ["xcrun", "--sdk", "iphoneos", "--show-sdk-path"])
    commands.run("framework-link", ["xcrun", "--sdk", "iphoneos", "clang++", "-dynamiclib", "-arch", "arm64",
                                   "-isysroot", sdk.strip(), "-mios-version-min=" + minimum,
                                   "-Wl,-force_load," + str(archive), "-framework", "Foundation", "-framework", "Metal",
                                   "-framework", "Accelerate", "-install_name", "@rpath/llama.framework/llama",
                                   "-o", framework / "llama"])
    headers = [ROOT / "include/llama.h"] + [ROOT / "ggml/include" / name for name in
               ("ggml.h", "ggml-opt.h", "ggml-alloc.h", "ggml-backend.h", "ggml-metal.h", "ggml-cpu.h", "ggml-blas.h", "gguf.h")]
    for header in headers:
        shutil.copy2(header, framework / "Headers" / header.name)
    (framework / "Modules/module.modulemap").write_text(
        'framework module llama {\n  umbrella "Headers"\n  link "c++"\n'
        '  link framework "Accelerate"\n  link framework "Metal"\n  link framework "Foundation"\n  export *\n}\n')
    with (framework / "Info.plist").open("wb") as handle:
        plistlib.dump({"CFBundleDevelopmentRegion": "en", "CFBundleExecutable": "llama",
                      "CFBundleIdentifier": "org.ggml.llama", "CFBundleInfoDictionaryVersion": "6.0",
                      "CFBundleName": "llama", "CFBundlePackageType": "FMWK", "CFBundleShortVersionString": "1.0",
                      "CFBundleVersion": "1", "MinimumOSVersion": minimum, "CFBundleSupportedPlatforms": ["iPhoneOS"],
                      "DTPlatformName": "iphoneos", "UIDeviceFamily": [1, 2]}, handle)
    xcframework = artifacts / "llama.xcframework"
    commands.run("framework-xcframework", ["xcodebuild", "-create-xcframework", "-framework", framework, "-output", xcframework])
    binary = xcframework / "ios-arm64/llama.framework/llama"
    require(binary.is_file(), "iPhoneOS arm64 framework is missing")
    write_json(artifacts / "framework-provenance.json", {
        "source_files_sha256": identity, "framework_sha256": sha256(binary),
        "linkage": "static CMake libraries merged into a dynamic, embedded llama.framework",
        "sdk": "iphoneos", "architecture": "arm64", "minimum_iOS": minimum,
        "GPU_executed": False})
    return xcframework


def stage_project(output, xcframework):
    stage = output / "staging"
    project = stage / "examples/llama.swiftui"
    shutil.copytree(ROOT / "examples/llama.swiftui", project,
                    ignore=shutil.ignore_patterns("xcuserdata", "DerivedData", ".DS_Store"))
    destination = stage / "build-apple/llama.xcframework"
    destination.parent.mkdir(parents=True)
    shutil.copytree(xcframework, destination, symlinks=True)
    bridge = project / "llama.swiftui/Bridging"
    shutil.copy2(HERE / "src/CandidateReplay.mm", bridge / "CandidateReplay.mm")
    shutil.copy2(ROOT / "ggml/src/ggml-metal/ggml-metal-bitnet-q-config.h", bridge / "ggml-metal-bitnet-q-config.h")
    with (bridge / "FinetuneBridge.h").open("a") as handle:
        handle.write('\n#ifdef __cplusplus\nextern "C" {\n#endif\nint bitnet_candidate_run(void);\n#ifdef __cplusplus\n}\n#endif\n')
    with (bridge / "FinetuneBridge.mm").open("a") as handle:
        handle.write('\n#include "CandidateReplay.mm"\n')
    shutil.copy2(HERE / "src/llama_swiftuiApp.swift", project / "llama.swiftui/llama_swiftuiApp.swift")
    pbx = project / "llama.swiftui.xcodeproj/project.pbxproj"
    text = re.sub(r"DEVELOPMENT_TEAM = [^;]*;", 'DEVELOPMENT_TEAM = "";', pbx.read_text())
    pbx.write_text(text)
    return project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh local build directory")
    parser.add_argument("--developer-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--minimum-ios", default="16.4")
    parser.add_argument("--framework", type=Path, help="Matching XCFramework produced by this builder")
    parser.add_argument("--framework-manifest", type=Path)
    parser.add_argument("--unsigned", action="store_true", help="Build without a signing identity; cannot install")
    parser.add_argument("--team", default=os.environ.get("APPLE_TEAM_ID"))
    parser.add_argument("--bundle-id", default=os.environ.get("BITNET_BUNDLE_ID", "local.bitnet.submission.smoke"))
    parser.add_argument("--configuration", choices=("Debug", "Release"), default="Debug")
    parser.add_argument("--allow-provisioning-updates", action="store_true")
    args = parser.parse_args()
    require(sys.platform == "darwin", "iPhoneOS builds require macOS with full Xcode")
    require(args.jobs > 0, "jobs must be positive")
    require(args.unsigned or args.team, "Signed builds require --team or APPLE_TEAM_ID; use --unsigned for compilation only")
    require(not args.unsigned or not args.allow_provisioning_updates, "Unsigned builds do not request provisioning")
    require(re.fullmatch(r"[A-Za-z0-9.-]+", args.bundle_id), "Invalid bundle identifier")
    output = args.output.expanduser().resolve()
    require(not output.exists(), "Refuse to reuse an existing build directory: " + str(output))
    output.mkdir(parents=True)
    env = developer_environment(args.developer_dir)
    commands = Commands(output / "logs", env)
    record = {"created": datetime.datetime.now().astimezone().isoformat(), "build_completed": False,
              "GPU_executed": False, "installed": False, "unsigned": args.unsigned,
              "bundle_id": args.bundle_id, "source_root": str(ROOT), "DEVELOPER_DIR": env["DEVELOPER_DIR"]}
    try:
        identity = source_identity()
        record["source_files_sha256"] = identity
        record["package_sources_sha256"] = package_identity()
        for name, argv in (("host", ["sw_vers"]), ("xcode", ["xcodebuild", "-version"]),
                           ("sdk", ["xcrun", "--sdk", "iphoneos", "--show-sdk-version"]),
                           ("git-commit", ["git", "rev-parse", "HEAD"]),
                           ("git-status", ["git", "status", "--short"])):
            record[name] = commands.run(name, argv)[1].strip()
        if args.framework:
            xcframework = args.framework.expanduser().resolve()
            provenance_path = args.framework_manifest or xcframework.parent / "framework-provenance.json"
            provenance = json.loads(provenance_path.read_text())
            require(provenance["source_files_sha256"] == identity, "External framework does not match current source")
            require(provenance["sdk"] == "iphoneos" and provenance["architecture"] == "arm64", "Wrong framework platform")
            require(sha256(xcframework / "ios-arm64/llama.framework/llama") == provenance["framework_sha256"], "Framework checksum mismatch")
        else:
            xcframework = build_framework(output, commands, args.minimum_ios, args.jobs, identity)
        shaders = output / "shaders"
        commands.run("shaders", [sys.executable, ROOT / "bitnet/scripts/build-shaders.py", "--sdk", "iphoneos",
                                 "--output", shaders, "--developer-dir", env["DEVELOPER_DIR"]])
        for original, alias in (("qcur0-c8x2.metallib", "q-c8x2.metallib"), ("stock4x1.metallib", "q-stock4x1.metallib")):
            shutil.copy2(shaders / original, shaders / alias)
        record["q_kernel_contract"] = load_contract()
        require(json.loads((shaders / "BUILD.json").read_text())["q_kernel_contract"] == record["q_kernel_contract"], "Shader/header contract differs")
        record["shader_build_manifest_sha256"] = sha256(shaders / "BUILD.json")
        record["shader_products_sha256"] = {path.name: sha256(path) for path in shaders.glob("*.metallib")}
        objects = []
        for source in ("iphone-q.mm", "iphone-operator.mm", "iphone-down-controls.mm", "iphone-original-poc.mm", "iphone-ppl.mm", "iphone-actquant.mm"):
            obj = output / "objects" / (Path(source).stem + ".o")
            obj.parent.mkdir(exist_ok=True)
            command = ["xcrun", "--sdk", "iphoneos", "clang++", "-arch", "arm64", "-mios-version-min=" + args.minimum_ios,
                       "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off", "-fobjc-arc", "-Wall", "-Wextra", "-Werror",
                       "-Wno-deprecated-declarations"]
            for include in (HERE / "src", ROOT / "include", ROOT / "ggml/include", ROOT / "ggml/src", ROOT / "ggml/src/ggml-metal"):
                command += ["-I", include]
            commands.run("compile-" + Path(source).stem, command + ["-c", HERE / "src" / source, "-o", obj])
            objects.append(obj)
        contract_obj = output / "objects/q-pipeline-contract.o"
        contract_command = ["xcrun", "--sdk", "iphoneos", "clang++", "-arch", "arm64", "-mios-version-min=" + args.minimum_ios,
                            "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"]
        for include in (ROOT / "include", ROOT / "ggml/include", ROOT / "ggml/src", ROOT / "ggml/src/ggml-metal", ROOT / "vendor"):
            contract_command += ["-I", include]
        commands.run("compile-q-pipeline-contract", contract_command + ["-c", ROOT / "bitnet/tests/q_pipeline_contract.cpp", "-o", contract_obj])
        objects.append(contract_obj)
        project = stage_project(output, xcframework)
        command = ["xcodebuild", "-project", project / "llama.swiftui.xcodeproj", "-scheme", "llama.swiftui",
                   "-configuration", args.configuration, "-destination", "generic/platform=iOS",
                   "-derivedDataPath", output / "derived", "IPHONEOS_DEPLOYMENT_TARGET=" + args.minimum_ios,
                   "PRODUCT_BUNDLE_IDENTIFIER=" + args.bundle_id, "DEVELOPMENT_TEAM=" + (args.team or ""),
                   "CODE_SIGN_STYLE=Automatic", "OTHER_LDFLAGS=$(inherited) " + " ".join('"' + str(obj) + '"' for obj in objects)]
        if args.unsigned:
            command += ["CODE_SIGNING_ALLOWED=NO", "CODE_SIGNING_REQUIRED=NO", "CODE_SIGN_IDENTITY="]
        if args.allow_provisioning_updates:
            command += ["-allowProvisioningUpdates"]
        commands.run("app-build", command + ["build"])
        app = output / "derived/Build/Products" / (args.configuration + "-iphoneos") / "llama.swiftui.app"
        require(app.is_dir(), "App bundle missing after build")
        with (app / "Info.plist").open("rb") as handle:
            executable = plistlib.load(handle)["CFBundleExecutable"]
        paths = [app / executable, app / "Frameworks/llama.framework/llama"]
        debug_library = app / "llama.swiftui.debug.dylib"
        if debug_library.exists():
            paths.append(debug_library)
        if not args.unsigned:
            commands.run("codesign-verify", ["codesign", "--verify", "--deep", "--strict", app])
        require(source_identity() == identity, "Source changed during build; use a frozen checkout")
        require(package_identity() == record["package_sources_sha256"], "Package source changed during build; use a frozen checkout")
        record.update({"build_completed": True, "exit_code": 0, "app": str(app), "shaders": str(shaders),
                       "app_files_sha256": {str(p): sha256(p) for p in paths},
                       "framework": str(xcframework), "framework_sha256": sha256(xcframework / "ios-arm64/llama.framework/llama"),
                       "object_sha256": {str(p): sha256(p) for p in objects}})
        print(json.dumps({"build_completed": True, "app": str(app), "unsigned": args.unsigned, "GPU_executed": False}, indent=2))
        return 0
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        record.update({"error": str(error), "exit_code": 1})
        print(str(error), file=sys.stderr)
        return 1
    finally:
        write_json(output / "BUILD.json", record)


if __name__ == "__main__":
    sys.exit(main())
