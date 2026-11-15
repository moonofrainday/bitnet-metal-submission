#!/usr/bin/env python3
"""Explicit physical-iPhone replay and operator runner. Raw artifacts remain local."""
from pathlib import Path
import argparse
import collections
import datetime
import json
import math
import os
import re
import shutil
import struct
import sys

from common import PUBLIC_MODEL_SHA256, Commands, developer_environment, require, sha256, write_json


def safe_component(value):
    require(bool(re.fullmatch(r"[A-Za-z0-9_.-]+", value)) and value not in (".", ".."), "Invalid container basename")
    return value


def custom_route(name):
    return name == "q-custom" or name == "ffn-fused" or name.startswith(("ffn-mech-", "ffn-c-"))


def runtime_counts(log, routes, layers, expected_fallback, q_fallback=""):
    encoded = collections.defaultdict(collections.Counter)
    completed = collections.defaultdict(collections.Counter)
    eligible = collections.defaultdict(collections.Counter)
    fallback = collections.defaultdict(collections.Counter)
    q_fallback_counts = collections.defaultdict(collections.Counter)
    errors = []
    current = None
    prefill = False
    prefill_custom = 0
    prefill_custom_completions = 0
    for line in log.splitlines():
        fields = dict(re.findall(r"(\w+)=(\[[^\]]*\]|[^\s]+)", line))
        if line.startswith("FFNK_PREFILL_BEGIN "):
            prefill = True
        if line.startswith("FFNK_PREFILL_END "):
            prefill = False
        if line.startswith("FFNK_PATH_BEGIN "):
            current = fields["route"]
        if line.startswith("FFNK_PATH_END "):
            current = None
        if prefill and line.startswith("BITNET_MULTILAYER_ENCODE "):
            if custom_route(fields.get("route", "")):
                prefill_custom += 1
        if prefill and line.startswith("BITNET_MULTILAYER_COMPLETE ") and custom_route(fields.get("route", "")):
            prefill_custom_completions += 1
        if current and line.startswith("BITNET_MULTILAYER_ENCODE "):
            encoded[current][fields["route"]] += 1
            if fields["route"] in ("q-custom", "ffn-mech-G-dual", "ffn-mech-K-dual", "ffn-fused"):
                eligible[current][int(fields["layer"])] += 1
        if current and line.startswith("BITNET_MULTILAYER_COMPLETE "):
            completed[current][fields["route"]] += 1
            if fields.get("status") != "4" or fields.get("completed") != "1":
                errors.append("GPU completion failed: " + line)
        if current and line.startswith("BITNET_FFN_FALLBACK "):
            fallback[current][fields.get("reason", "unknown")] += 1
        if current and line.startswith("BITNET_Q_FALLBACK "):
            if fields.get("node") != "Qcur-0" or fields.get("layer") != "0":
                errors.append("Unexpected Q fallback identity: " + line)
            q_fallback_counts[current][fields.get("reason", "unknown")] += 1
    wanted = {0} if layers == "0" else set(range(4 if layers == "0-3" else 30))
    allowed_custom = {"S": set(), "Q4": {"q-custom"},
                      "G": {"ffn-mech-G-dual", "ffn-mech-G-post"},
                      "K": {"ffn-mech-K-dual", "ffn-mech-K-postnorm"}, "F8": {"ffn-fused"}}
    if q_fallback:
        allowed_custom["Q4"] = set()
    for route in routes:
        for label, counts in (("encoded", encoded[route]), ("completed", completed[route])):
            unexpected = {name: count for name, count in counts.items()
                          if count and custom_route(name) and name not in allowed_custom[route]}
            if unexpected:
                errors.append(route + " has unexpected custom " + label + " work: " + str(unexpected))
        if encoded[route] != completed[route]:
            errors.append(route + " encode/completion counts differ")
        if encoded[route]["q-stock"] != (928 if route == "Q4" and not q_fallback else 960):
            errors.append(route + " stock Q coverage mismatch")
        if encoded[route]["ffn-stock-down"] != 960:
            errors.append(route + " stock down coverage mismatch")
        if route != "S":
            expected = ({0} if not q_fallback else set()) if route == "Q4" else wanted - (expected_fallback if route == "K" else set())
            if dict(eligible[route]) != {index: 32 for index in expected}:
                errors.append(route + " target layer/32-step coverage mismatch")
        if route == "Q4":
            expected_reasons = {"pipeline" if q_fallback == "missing-library" else "geometry": 32} if q_fallback else {}
            if dict(q_fallback_counts[route]) != expected_reasons:
                errors.append("Q4 fallback reasons/counts do not match the explicit control")
    if prefill_custom or prefill_custom_completions:
        errors.append("Custom GPU work occurred during prefill")
    return {"encodes": {k: dict(v) for k, v in encoded.items()},
            "completions": {k: dict(v) for k, v in completed.items()},
            "eligible_layer_counts": {k: dict(v) for k, v in eligible.items()},
            "fallbacks": {k: dict(v) for k, v in fallback.items()},
            "Q_fallbacks": {k: dict(v) for k, v in q_fallback_counts.items()},
            "prefill_custom_dispatches": prefill_custom,
            "prefill_custom_completions": prefill_custom_completions}, errors


def read_q_qualification(path):
    require(path.is_file(), "Missing mandatory fetched Q qualification.json")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "Q qualification must be a JSON object")
    return value


def validate_q_qualification(qualification, libraries):
    require(qualification.get("local_fp64_pass") is True, "Q local FP64 qualification failed")
    expected_libraries = {key: libraries[key]["sha256"] for key in ("C8", "Q4")}
    require(qualification.get("libraries") == expected_libraries, "Q qualification shader identity mismatch")
    require(qualification.get("device") == "Apple A16 GPU", "Q qualification is not from the expected A16 GPU")
    require(qualification.get("tolerance") == {"abs": 2e-4, "relative": 2e-5}, "Q qualification tolerance changed")
    cases = qualification.get("cases")
    require(isinstance(cases, list) and len(cases) == 9, "Q qualification needs all 9 step/route cases")
    expected = {(step, route) for step in (1, 16, 32) for route in ("S", "C8", "Q4")}
    found = []
    for case in cases:
        key = (case.get("step"), case.get("route"))
        require(key in expected and key not in found, "Q qualification has missing/duplicate/unexpected cases")
        found.append(key)
        require(case.get("rows") == 2560 and case.get("failed_rows") == 0
                and case.get("nonfinite_rows") == 0 and case.get("GPU_completed") is True,
                "Q qualification has failed/nonfinite/incomplete output coverage")
        if case.get("route") in ("S", "Q4"):
            require(case.get("stock_bit_different_rows") == 0, "Stock/Q4 output is not stock-bit-identical")
        require(isinstance(case.get("max_abs"), (int, float)) and math.isfinite(case["max_abs"])
                and isinstance(case.get("relative_L2"), (int, float)) and math.isfinite(case["relative_L2"]),
                "Q qualification contains missing/nonfinite error metrics")
    require(set(found) == expected, "Q qualification coverage mismatch")
    inputs = qualification.get("inputs")
    require(isinstance(inputs, dict) and len(inputs) == 9
            and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in inputs.values()),
            "Q qualification must bind all 9 fixture files by SHA-256")
    expected_inputs = {("step-" + str(step), name) for step in (1, 16, 32)
                       for name in ("w.tq2", "x.f32", "y.f32")}
    require({tuple(Path(name).parts[-2:]) for name in inputs} == expected_inputs,
            "Q qualification fixture step/file coverage mismatch")
    return qualification


def validate_q_readback(directory, route):
    """Check actual fetched coverage/hash evidence; numerical FP64 remains separate."""
    path = directory / "manifest.json"
    require(path.is_file(), "Missing Q readback manifest")
    manifest = json.loads(path.read_text())
    expected = {"weights/w.tq2": 1689600}
    for step in range(1, 33):
        expected.update({"step-" + str(step) + "/" + role + ".f32": 10240 for role in ("x", "y")})
        if step in (1, 16, 32):
            expected["step-" + str(step) + "/l_out.f32"] = 10240
    require(manifest.get("route") == route and manifest.get("coverage_pass") is True
            and manifest.get("decode_steps") == 32 and manifest.get("file_count") == 68
            and manifest.get("copy_group_waited") is True, "Q readback identity/coverage failed")
    require(set(manifest.get("files", {})) == set(expected), "Q readback file coverage differs")
    for name, size in expected.items():
        item = manifest["files"][name]
        file = directory / name
        require(file.is_file() and file.stat().st_size == item.get("bytes") == size,
                "Missing or wrong-sized Q readback: " + name)
        require(sha256(file) == item.get("sha256"), "Q readback checksum mismatch: " + name)
        if name.endswith(".f32"):
            require(all(math.isfinite(value[0]) for value in struct.iter_unpack("<f", file.read_bytes())),
                    "Nonfinite actual Q readback: " + name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--device", default=os.environ.get("IPHONE_DEVICE_ID"), help="devicectl identifier; no stored device default")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", default=PUBLIC_MODEL_SHA256)
    parser.add_argument("--mode", choices=("graph-verify", "decode-benchmark", "q-verify", "q-benchmark",
                                        "ffn-reference", "ffn-verify", "ffn-benchmark", "down-verify"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh local host artifact directory")
    parser.add_argument("--developer-dir", type=Path)
    parser.add_argument("--install", action="store_true", help="Required: install the selected signed app before every launch")
    parser.add_argument("--reuse-model", action="store_true", help="Reuse the container copy; the app still checks its SHA-256")
    parser.add_argument("--history-dir", type=Path, help="Graph input directory with prompt.txt, prompt.ids, continuation.ids")
    parser.add_argument("--routes", default="S,Q4")
    parser.add_argument("--layers", choices=("0", "0-3", "0-29"), default="0")
    parser.add_argument("--fixture", type=Path, help="Operator fixture directory; raw tensors are not shipped")
    parser.add_argument("--down-libraries", type=Path, help="Explicit locally compiled down-control metallib directory; down-verify only")
    parser.add_argument("--operator-fixture", help="Existing device fixture basename for operator timing reuse")
    parser.add_argument("--operator-output", help="Fresh device verify output, or existing qualified output for timing")
    parser.add_argument("--comparisons", default="AA,SG", help="Decode benchmark contrasts supported by the preserved harness")
    parser.add_argument("--warmups", type=int, default=1, help="Decode warmups; operator protocol fixes 30/path")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=21, help="Decode pairs/round; operator protocol fixes 201")
    parser.add_argument("--expect-fallback-layers", default="", help="Explicit K fallback layer list, e.g. 2,29")
    parser.add_argument("--q-readback", action="store_true", help="Read all 32 real Q pre-RoPE X/Y plus layer-0 residual checkpoints; correctness only")
    parser.add_argument("--q-fallback", choices=("missing-library", "invalid-geometry"), help="Separate Q4 stock-fallback control; correctness only")
    parser.add_argument("--research-gate", type=Path, help="Required for F8/K decode timing; preserves numerical failures")
    parser.add_argument("--strict-v1", action="store_true", help="Also require full frozen numerical acceptance; missing checks return 2")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    require(args.device, "Supply --device or IPHONE_DEVICE_ID")
    require(args.install, "Every run requires --install to select the signed app from this BUILD.json")
    require(args.timeout > 0, "Timeout must be positive")
    output = args.output.expanduser().resolve()
    require(not output.exists(), "Refuse existing host output directory")
    output.mkdir(parents=True)
    env = developer_environment(args.developer_dir)
    commands = Commands(output / "logs", env)
    summary = {"execution_pass": False, "parity_pass": None, "fp64_pass": None,
               "model_output_pass": None, "strict_v1_exit_code": 2, "model_quality_tested": False,
               "scope": args.mode, "errors": [], "host_output": str(output)}
    try:
        model = args.model.expanduser().resolve()
        require(sha256(model) == args.model_sha256 == PUBLIC_MODEL_SHA256, "Public 2B GGUF SHA-256 mismatch")
        build_dir = args.build_dir.expanduser().resolve()
        build = json.loads((build_dir / "BUILD.json").read_text())
        require(build.get("build_completed") and build.get("exit_code") == 0, "Build did not complete")
        require(not build.get("unsigned"), "Unsigned app cannot run on a physical device; rebuild with --team")
        for name, digest in build["app_files_sha256"].items():
            require(sha256(name) == digest, "App binary changed: " + name)
        bundle = build["bundle_id"]
        write_json(output / "BUILD_IDENTITY.json", {"build_manifest_sha256": sha256(build_dir / "BUILD.json"),
                                                   "framework_sha256": build["framework_sha256"],
                                                   "app_files_sha256": build["app_files_sha256"],
                                                   "bundle_id": bundle, "unsigned": build["unsigned"]})
        stamp = datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%f")
        prefix = "candidate-" + stamp
        graph = args.mode in ("graph-verify", "decode-benchmark")
        timing = args.mode.endswith("benchmark")
        routes = args.routes.split(",") if graph else ["S"]
        require(routes and routes[0] == "S" and len(set(routes)) == len(routes)
                and set(routes) <= {"S", "G", "K", "F8", "Q4"}, "Unique routes must begin with S")
        require("Q4" not in routes or args.layers == "0", "Q4 real-graph route is layer-0 only")
        require(not args.q_readback or (args.mode == "graph-verify" and args.layers == "0"
                and set(routes) <= {"S", "Q4"} and not args.q_fallback), "Q readback is restricted to layer-0 S/Q4 correctness")
        require(not args.q_fallback or (args.mode == "graph-verify" and args.layers == "0"
                and "Q4" in routes and set(routes) <= {"S", "Q4"} and not args.q_readback),
                "Q fallback must be a separate layer-0 S/Q4 correctness control")
        cfg = {"output": prefix, "model": safe_component(model.name), "model_sha256": args.model_sha256,
               "operation": "benchmark" if graph and timing else "verify" if graph else "operator",
               "layers": args.layers, "routes": routes, "trace": graph and not timing,
               "shadow": False, "shadow_all": False, "warmups": args.warmups, "rounds": args.rounds,
               "pairs": args.pairs, "missing_k_library": False, "comparisons": args.comparisons.split(","),
               "q_readback": args.q_readback, "q_fallback": args.q_fallback or "",
               "prompt": "", "prompt_ids": [], "continuation_ids": [], "libraries": {}}
        shader_root = Path(build["shaders"])
        library_root = shader_root
        libraries = {}
        if graph:
            require(args.history_dir is not None, "Graph runs require --history-dir")
            history = args.history_dir.expanduser().resolve()
            cfg["prompt"] = (history / "prompt.txt").read_text()
            cfg["prompt_ids"] = list(map(int, (history / "prompt.ids").read_text().split()))
            cfg["continuation_ids"] = list(map(int, (history / "continuation.ids").read_text().split()))
            require(len(cfg["prompt_ids"]) == 27 and len(cfg["continuation_ids"]) == 32, "Expected fixed 27/32 token configuration")
            if set(routes) & {"G", "K"} or args.q_readback:
                libraries["mechanisms"] = "ffn-mechanisms.metallib"
            for route, filename in (("K", "k0-postnorm-inplace.metallib"), ("F8", "ffn-gate-up-8x2.metallib"),
                                    ("Q4", "q-stock4x1.metallib")):
                if route in routes:
                    libraries[route] = filename
            if timing and set(routes) & {"K", "F8"}:
                require(args.research_gate is not None, "K/F8 timing requires an explicit research qualification gate")
                gate = json.loads(args.research_gate.read_text())
                require(gate.get("research_timing_pass") is True and gate.get("model_sha256") == args.model_sha256,
                        "Research gate must identify this model and authorize scoped timing")
                require(gate.get("fp64_pass") is False and gate.get("strict_exit_code") == 1,
                        "K/F8 research gate must preserve known frozen numerical failure")
                require(gate.get("app_files_sha256") == build["app_files_sha256"], "Research gate app identity mismatch")
                require(gate.get("layers") == args.layers and set(gate.get("routes", [])) == set(routes), "Research gate scope mismatch")
                for filename in libraries.values():
                    require(gate.get("libraries_sha256", {}).get(filename) == sha256(shader_root / filename), "Research gate shader mismatch")
                for name, digest in gate.get("evidence_sha256", {}).items():
                    require(sha256(name) == digest, "Research evidence changed")
                summary["research_gate_sha256"] = sha256(args.research_gate)
        else:
            kind = "DOWN" if args.mode == "down-verify" else "Q" if args.mode.startswith("q-") else "FFN"
            op_mode = args.mode.split("-", 1)[1]
            require(kind != "Q" or op_mode in ("verify", "benchmark"), "Q mode invalid")
            cfg.update({"operator_kind": kind, "operator_mode": op_mode,
                        "operator_fixture": safe_component(args.operator_fixture or prefix + "-fixture"),
                        "operator_output": safe_component(args.operator_output or prefix + "-operator")})
            if args.mode == "q-benchmark":
                require(args.operator_fixture or args.fixture,
                        "Q timing needs an existing device --operator-fixture or local --fixture")
                cfg["operator_output"] = prefix + "-operator"
                summary["q_benchmark_orchestration"] = {
                    "same_install_fresh_qualification": True,
                    "requested_operator_output": args.operator_output,
                    "actual_operator_output": cfg["operator_output"],
                    "operator_fixture": cfg["operator_fixture"],
                    "qualification_completed": False,
                    "installation_count": 0,
                    "native_gate_unchanged": True}
            else:
                require(not timing or (args.operator_fixture and args.operator_output),
                        "Operator timing must reuse explicit qualified --operator-fixture and --operator-output")
            if kind == "Q":
                libraries = {"C8": "q-c8x2.metallib", "Q4": "q-stock4x1.metallib"}
            elif kind == "DOWN":
                require(args.down_libraries is not None and args.fixture is not None,
                        "down-verify requires --down-libraries and --fixture")
                library_root = args.down_libraries.expanduser().resolve()
                down_manifest = json.loads((args.fixture.expanduser().resolve() / "MANIFEST.json").read_text())
                require(set(down_manifest.get("libraries", {})) == {"S-wrap", "correction-end"},
                        "Expected exactly the two down control libraries")
                libraries = {key: safe_component(entry["file"]) for key, entry in down_manifest["libraries"].items()}
            else:
                libraries = {"mechanisms": "ffn-mechanisms.metallib", "K": "k0-postnorm-inplace.metallib",
                             "Kref": "k0-postnorm-reference.metallib", "F8": "ffn-gate-up-8x2.metallib",
                             "split": "ffn-expand.metallib"}
        for key, filename in libraries.items():
            cfg["libraries"][key] = {"file": filename, "sha256": sha256(library_root / filename)}
        write_json(output / "config.json", cfg)
        write_json(output / "model-identity.json", {"path": str(model), "bytes": model.stat().st_size, "sha256": args.model_sha256})

        def device(name, argv, timeout=180, check=True):
            return commands.run(name, ["xcrun", "devicectl", "--timeout", str(timeout), "device", *argv,
                                       "--device", args.device], timeout=timeout + 20, check=check)

        def copy(name, direction, source, destination, timeout=3600, check=True):
            return commands.run(name, ["xcrun", "devicectl", "--timeout", str(timeout), "device", "copy", direction,
                                       "--device", args.device, "--domain-type", "appDataContainer", "--domain-identifier", bundle,
                                       "--source", source, "--destination", destination], timeout=timeout + 20, check=check)

        device("device-details", ["info", "details"], timeout=60)
        if args.install:
            device("install", ["install", "app", build["app"]], timeout=240)
            if args.mode == "q-benchmark":
                summary["q_benchmark_orchestration"]["installation_count"] = 1
        if not args.reuse_model:
            copy("transfer-model", "to", model, "Documents/" + model.name)
        for key, value in cfg["libraries"].items():
            copy("transfer-" + key, "to", library_root / value["file"], "Documents/" + value["file"], timeout=180)
        if not graph and (not timing or (args.mode == "q-benchmark" and args.fixture is not None)):
            require(args.fixture is not None, "Operator correctness requires --fixture")
            fixture = output / "staged-fixture"
            shutil.copytree(args.fixture.expanduser().resolve(), fixture)
            if cfg["operator_kind"] == "Q":
                manifest_path = fixture / "MANIFEST.json"
                manifest = json.loads(manifest_path.read_text())
                require(manifest.get("model_sha256") == args.model_sha256, "Fixture belongs to a different model")
                manifest["libraries"] = {key: cfg["libraries"][key] for key in ("C8", "Q4")}
                write_json(manifest_path, manifest)
            elif cfg["operator_kind"] == "DOWN":
                manifest_path = fixture / "MANIFEST.json"
                manifest = json.loads(manifest_path.read_text())
                require(manifest.get("model_sha256") == args.model_sha256, "Down fixture model mismatch")
                for key, filename in libraries.items():
                    require(manifest["libraries"][key]["sha256"] == cfg["libraries"][key]["sha256"],
                            "Down fixture/library checksum mismatch")
                write_json(manifest_path, manifest)
            write_json(output / "fixture-files-sha256.json", {str(p.relative_to(fixture)): sha256(p)
                                                               for p in sorted(fixture.rglob("*")) if p.is_file()})
            copy("transfer-fixture", "to", fixture, "Documents/" + cfg["operator_fixture"])
        if args.mode == "q-benchmark":
            # Qualification stores absolute app-container paths. Reinstalling between
            # qualification and timing invalidates those paths, even with equal bytes.
            qualification_cfg = dict(cfg)
            qualification_cfg["output"] = prefix + "-qualification"
            qualification_cfg["operator_mode"] = "verify"
            write_json(output / "q-preflight-config.json", qualification_cfg)
            preflight_code, preflight_log = commands.run(
                "q-preflight-launch", ["xcrun", "devicectl", "--timeout", str(args.timeout), "device", "process", "launch",
                                       "--console", "--terminate-existing", "--device", args.device,
                                       "--environment-variables", json.dumps({"BITNET_CANDIDATE_CONFIG": json.dumps(qualification_cfg, separators=(",", ":"))}),
                                       bundle], timeout=args.timeout + 20, check=False)
            preflight_fetch_code, _ = copy("q-preflight-fetch", "from", "Documents/" + qualification_cfg["output"],
                                            output / "qualification-device", check=False)
            qualification_fetch_code, _ = copy("q-preflight-qualification-fetch", "from",
                                                "Documents/" + cfg["operator_output"] + "/qualification.json",
                                                output / "q-preflight-qualification.json", check=False)
            protocol = summary["q_benchmark_orchestration"]
            protocol.update({"qualification_launch_exit_code": preflight_code,
                             "qualification_result_fetch_exit_code": preflight_fetch_code,
                             "qualification_file_fetch_exit_code": qualification_fetch_code,
                             "qualification_output": qualification_cfg["output"],
                             "installations_between_qualification_and_timing": 0})
            write_json(output / "q-benchmark-protocol.json", protocol)
            require(preflight_fetch_code == 0 and qualification_fetch_code == 0,
                    "Fresh Q qualification artifacts were not fetched; timing not started")
            fresh = read_q_qualification(output / "q-preflight-qualification.json")
            if fresh.get("local_fp64_pass") is False:
                summary["fp64_pass"] = False
                summary["strict_v1_exit_code"] = 1
            validate_q_qualification(fresh, cfg["libraries"])
            preflight_result_path = output / "qualification-device/result.json"
            require(preflight_result_path.is_file(), "Missing fresh Q application completion; timing not started")
            preflight_result = json.loads(preflight_result_path.read_text())
            require(preflight_code == 0 and "The app terminated with the exit code 0." in preflight_log,
                    "Fresh Q verification app did not exit 0; timing not started")
            require(preflight_result.get("operator_mode") == "verify"
                    and preflight_result.get("operator_fixture") == cfg["operator_fixture"]
                    and preflight_result.get("operator_output") == cfg["operator_output"]
                    and preflight_result.get("required_checks_pass") is True
                    and preflight_result.get("process_exit_code") == 0
                    and preflight_result.get("device") == "Apple A16 GPU"
                    and preflight_result.get("model_sha256") == args.model_sha256,
                    "Fresh Q verification completion identity mismatch; timing not started")
            for name, digest in build["app_files_sha256"].items():
                require(sha256(name) == digest, "App binary changed between qualification and timing")
            for value in cfg["libraries"].values():
                require(sha256(library_root / value["file"]) == value["sha256"],
                        "Shader changed between qualification and timing")
            protocol.update({"qualification_completed": True,
                             "qualification_sha256": sha256(output / "q-preflight-qualification.json"),
                             "qualified_case_count": len(fresh["cases"]),
                             "qualified_inputs": fresh["inputs"],
                             "qualified_libraries": fresh["libraries"],
                             "same_build_identity_sha256": sha256(output / "BUILD_IDENTITY.json")})
            write_json(output / "q-benchmark-protocol.json", protocol)
        launch_code, log = commands.run("launch", ["xcrun", "devicectl", "--timeout", str(args.timeout), "device", "process", "launch",
                                                    "--console", "--terminate-existing", "--device", args.device,
                                                    "--environment-variables", json.dumps({"BITNET_CANDIDATE_CONFIG": json.dumps(cfg, separators=(",", ":"))}),
                                                    bundle], timeout=args.timeout + 20, check=False)
        fetch_code, _ = copy("fetch", "from", "Documents/" + prefix, output / "device", check=False)
        operator_fetch_code = 0
        if not graph:
            operator_fetch_code, _ = copy("fetch-operator", "from", "Documents/" + cfg["operator_output"], output / "operator", check=False)
        errors = summary["errors"]
        if args.mode != "down-verify":
            if launch_code != 0:
                errors.append("devicectl launch failed with exit " + str(launch_code))
            if "The app terminated with the exit code 0." not in log:
                errors.append("Actual application exit 0 is missing")
        if fetch_code != 0 or operator_fetch_code != 0:
            errors.append("Artifact fetch failed")
        result_path = output / "device/result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        marker_lines = [line.split("BITNET_PHONE_COMPLETE ", 1)[1] for line in log.splitlines() if "BITNET_PHONE_COMPLETE " in line]
        console_result = None
        if len(marker_lines) == 1:
            try:
                console_result = json.loads(marker_lines[0])
            except json.JSONDecodeError:
                pass
        if not result:
            result = console_result or {}
        if not result:
            errors.append("Missing complete file/console result")
        if console_result and result != console_result:
            errors.append("Console and file completion records differ")
        summary["completion"] = result
        write_json(output / "completion-source.json", {"file": str(result_path), "console_marker_count": len(marker_lines),
                                                        "authoritative": "fetched result.json" if result_path.exists() else "console",
                                                        "note": "Console streams can interleave; application exit and required checks remain mandatory."})
        if result.get("model_sha256") != args.model_sha256:
            errors.append("Model completion identity mismatch")
        if result.get("device") != "Apple A16 GPU":
            errors.append("Expected physical Apple A16 GPU")
        if graph:
            if result.get("routes") != routes or result.get("repeatability_pass") is not True:
                errors.append("Route list or within-route repeat failed")
            if "offloaded 31/31 layers to GPU" not in log:
                errors.append("31/31 Metal offload not confirmed")
            if "G" in routes and result.get("S_G_parity_pass") is not True:
                errors.append("G does not reproduce stock logits")
            if not timing:
                expected_fallback = set(map(int, args.expect_fallback_layers.split(","))) if args.expect_fallback_layers else set()
                counts, count_errors = runtime_counts(log, routes, args.layers, expected_fallback, args.q_fallback or "")
                summary["runtime"] = counts
                errors.extend(count_errors)
            if result.get("q_readback_enabled") is not args.q_readback:
                errors.append("Q readback completion does not match the requested mode")
            if result.get("q_fallback_mode", "") != (args.q_fallback or ""):
                errors.append("Q fallback completion does not match the requested control")
            summary["q_fallback_control"] = args.q_fallback
            if args.q_readback:
                require(result.get("q_readback_coverage_pass") is True
                        and result.get("probe_logit_nonperturbation_pass") is True,
                        "Q readback coverage or probe/logit nonperturbation failed")
                records = result.get("q_readback_manifests", {})
                require(set(records) == set(routes), "Q readback completion misses route manifests")
                manifests = {}
                for route in routes:
                    relative = "route-" + route + "/q-readback/manifest.json"
                    require(records[route].get("file") == relative, "Q readback manifest identity differs")
                    directory = output / "device" / ("route-" + route) / "q-readback"
                    require(sha256(directory / "manifest.json") == records[route].get("sha256"), "Fetched Q manifest checksum differs")
                    manifests[route] = validate_q_readback(directory, route)
                    for step in range(33):
                        first = output / "device" / ("route-" + route) / "logits" / ("step-" + str(step) + ".f32")
                        repeat = output / "device" / ("repeat-" + route) / "logits" / first.name
                        require(first.is_file() and repeat.is_file() and first.stat().st_size == repeat.stat().st_size == 128256 * 4,
                                "Missing probe/non-probe full logits")
                        require(first.read_bytes() == repeat.read_bytes(), "Q probe changed observed full logits")
                summary["q_readback_coverage_pass"] = True
                summary["probe_logit_nonperturbation_pass"] = True
                summary["q_readback_manifests"] = records
                summary["fp64_pass"] = None
                summary["fp64_scope"] = "Raw all-32-call Q data captured; independent CPU FP64 analysis is a separate required stage"
            comparisons = result.get("logit_comparisons", {})
            summary["parity_pass"] = all(value.get("stock_bits") == 0 for value in comparisons.values()) if comparisons else None
            summary["parity_scope"] = "33 full logits vectors relative to stock; not FFN fusion/split parity"
            summary["model_output_pass"] = all(value.get("stock_tolerance_failed") == 0 for value in comparisons.values()) if comparisons else None
            summary["model_output_scope"] = "33 full stock-relative logits vectors and fixed-token replay; no layer-output or model-quality assessment"
            if timing and result.get("timing_clean") is not True:
                errors.append("Clean timing completion missing")
        else:
            if cfg["operator_kind"] == "DOWN":
                native_path = output / "operator/summary.json"
                require(native_path.is_file(), "Missing completed native down summary")
                native = json.loads(native_path.read_text())
                require(native.get("analysis_completed") is True and native.get("execution_pass") is True,
                        "Down operator did not finish its execution checks")
                expected_exit = native.get("strict_exit_code")
                require(expected_exit in (0, 1) and native.get("fp64_pass") is (expected_exit == 0),
                        "Down native numerical status/exit mismatch")
                require(result.get("operator_kind") == "DOWN" and result.get("operator_mode") == "verify"
                        and result.get("execution_pass") is True and result.get("process_exit_code") == expected_exit
                        and result.get("fp64_pass") is native["fp64_pass"], "Down app/native completion mismatch")
                actual_exits = re.findall(r"The app terminated with the exit code (\d+)\.", log)
                require(len(actual_exits) == 1 and int(actual_exits[0]) == expected_exit,
                        "Missing/mismatched actual down application exit")
                require(launch_code in (0, expected_exit), "devicectl failed beyond the completed numeric result")
                require(native.get("rows_per_route") == 96 and native.get("route_count") == 3
                        and native.get("GPU_command_buffers_completed") == 36, "Down total coverage mismatch")
                cases = native.get("cases", [])
                require(len(cases) == 36, "Missing down route/group cases")
                wanted = {(group, route) for group in (0, 336, 352, 360, 448, 496, 880, 1144, 1504, 1824, 2152, 2504)
                          for route in ("S", "S-wrap", "correction-end")}
                require({(case.get("group"), case.get("route")) for case in cases} == wanted, "Down group identity differs")
                for case in cases:
                    require(case.get("rows") == 8 and isinstance(case.get("nonfinite_rows"), int)
                            and 0 <= case["nonfinite_rows"] <= 8 and isinstance(case.get("failed_rows"), int)
                            and case["nonfinite_rows"] <= case["failed_rows"] <= 8
                            and case.get("GPU_completed") is True and case.get("GPU_status") == 4
                            and case.get("output_canaries_pass") is True and case.get("inputs_unchanged") is True,
                            "Down output/input/GPU checks failed")
                require(native["fp64_pass"] is (sum(case["failed_rows"] for case in cases) == 0)
                        and isinstance(native.get("parity_pass"), bool), "Down failure counts/flags disagree")
                require(native.get("model_sha256") == args.model_sha256
                        and native.get("tolerance") == {"abs": 2e-4, "relative": 2e-5},
                        "Down model/tolerance mismatch")
                summary["parity_pass"] = native.get("parity_pass")
                summary["parity_scope"] = native.get("parity_scope")
                summary["fp64_pass"] = native["fp64_pass"]
                summary["fp64_scope"] = native.get("scope")
                summary["local_numerical_exit_code"] = expected_exit
                summary["completed_numeric_failure"] = expected_exit == 1
                summary["numerically_failed_rows"] = sum(case["failed_rows"] for case in cases)
                summary["nonfinite_rows"] = sum(case["nonfinite_rows"] for case in cases)
                summary["native_summary_sha256"] = sha256(native_path)
                summary["overall_pass"] = not errors and native["fp64_pass"] and native.get("parity_pass") is True
            elif result.get("required_checks_pass") is not True or result.get("operator_mode") != cfg["operator_mode"]:
                errors.append("Operator required checks failed")
            if cfg["operator_kind"] == "Q":
                qualification_path = output / "operator/qualification.json"
                qualification = read_q_qualification(qualification_path)
                if qualification.get("local_fp64_pass") is False:
                    summary["fp64_pass"] = False
                    summary["strict_v1_exit_code"] = 1
                qualification = validate_q_qualification(qualification, cfg["libraries"])
                summary["fp64_pass"] = True
                summary["fp64_scope"] = "All 2560 Q outputs on steps 1/16/32 using identical captured W/X; not full-model FP64"
                summary["qualification_sha256"] = sha256(qualification_path)
                summary["qualified_case_count"] = len(qualification["cases"])
            elif cfg["operator_kind"] == "FFN":
                summary["note"] = "FFN helper checks local research equivalence; it does not grant full-chain FP64 acceptance"
        summary["execution_pass"] = not errors
        if summary["fp64_pass"] is False or summary["model_output_pass"] is False:
            summary["strict_v1_exit_code"] = 1
        if args.strict_v1:
            return summary["strict_v1_exit_code"] if not errors else 1
        if args.mode == "down-verify":
            return 0 if summary.get("overall_pass") is True else 1
        return 0 if not errors else 1
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        summary["errors"].append(str(error))
        print(str(error), file=sys.stderr)
        return 1
    finally:
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
