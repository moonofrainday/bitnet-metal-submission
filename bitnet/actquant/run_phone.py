#!/usr/bin/env python3
"""Prepare an ActQuant phone run; device commands require explicit --run.

Exit 0 after --run means native collection and its identity checks completed.
It does not mean numerical, reference-model or model-quality acceptance passed.
Use analyze_op.py or compare_model.py on the fetched local artifacts separately.
Raw tensors, tokens, signed binaries and device logs must remain local.
"""
import argparse
import csv
import datetime
import json
import math
from pathlib import Path
import re
import struct
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "bitnet/ios"))
from common import PUBLIC_MODEL_SHA256, Commands, developer_environment, require, sha256, write_json
from q4_contract import q4_library_identity

OFFICIAL_ACTQUANT_SHA256 = "a8b3170f1824f010701f4e033fbc33d199bc2ea344ddbf2554594b66fada8b1c"


def safe_relative(root, text):
    part = Path(text)
    require(text and not part.is_absolute() and "\\" not in text and all(v not in (".", "..") for v in text.split("/")),
            "Unsafe manifest path")
    path = (root / part).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), "Manifest file missing or outside fixture")
    return path


def fixture_identity(root):
    manifest_file = root / "FIXTURES.json"
    require(manifest_file.stat().st_size <= 4 * 1024 * 1024, "Fixture manifest too large")
    manifest = json.loads(manifest_file.read_text())
    require(manifest["schema"] == 1 and manifest["epsilon"] == 1e-5 and
            manifest["official_source_sha256"] == OFFICIAL_ACTQUANT_SHA256 and 1 <= len(manifest["cases"]) <= 128,
            "Invalid ActQuant fixture manifest")
    files = {"FIXTURES.json": sha256(manifest_file)}
    names = set()
    for case in manifest["cases"]:
        name, width, rows = case["name"], case["width"], case["rows"]
        require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", name) and name not in names, "Invalid/duplicate case")
        names.add(name)
        require(type(width) is int and 1 <= width <= 16384 and type(rows) is int and 1 <= rows <= 128 and
                case["ne"] == [width, rows, 1, 1] and case["elements"] == width * rows and case["dtype"] == "F32",
                "Fixture shape contract")
        sizes = {"input": width * rows * 4, "expected": width * rows * 4, "scale": rows * 4, "int8_codes": width * rows}
        for role, size in sizes.items():
            path = safe_relative(root, case[role])
            require(path.stat().st_size == size and sha256(path) == case[role + "_sha256"], "Fixture file identity mismatch")
            files[case[role]] = case[role + "_sha256"]
    return manifest, files


def token_identity(path):
    require(path.stat().st_size <= 4 * 1024 * 1024, "Token manifest too large")
    tokens = json.loads(path.read_text())
    for key, length in (("prompt", 27), ("continuation", 32), ("corpus", 129)):
        ids = tokens[key]
        require(isinstance(ids, list) and len(ids) == length and all(type(v) is int and 0 <= v < 128256 for v in ids),
                "Token contract: " + key)
        if key != "continuation":
            require(ids[0] == 128000, "Token BOS mismatch")
    return tokens


def build_identity(directory):
    manifest_path = directory / "BUILD.json"
    build = json.loads(manifest_path.read_text())
    require(build["build_completed"] is True and build["exit_code"] == 0 and build["unsigned"] is False,
            "A successful signed iPhoneOS build is required")
    require(Path(build["app"]).is_dir(), "Signed app is missing")
    package = build["package_sources_sha256"]
    for relative in ("bitnet/actquant/runner.cpp", "bitnet/actquant/runner.h", "bitnet/ios/src/iphone-actquant.mm",
                     "bitnet/ios/src/CandidateReplay.mm", "bitnet/ios/build.py", "vendor/nlohmann/json.hpp"):
        require(relative in package and sha256(ROOT / relative) == package[relative], "ActQuant package provenance mismatch: " + relative)
    for relative in ("src/models/bitnet.cpp", "ggml/include/ggml.h", "ggml/src/ggml-metal/kernels/act_quant.metal"):
        require(relative in build["source_files_sha256"] and sha256(ROOT / relative) == build["source_files_sha256"][relative],
                "ActQuant framework provenance mismatch: " + relative)
    for relative, digest in build["source_files_sha256"].items():
        require(sha256(ROOT / relative) == digest, "Framework source changed: " + relative)
    for relative, digest in package.items():
        require(sha256(ROOT / relative) == digest, "Package source changed: " + relative)
    for path, digest in build["app_files_sha256"].items():
        require(sha256(path) == digest, "Built app binary changed")
    objects = build["object_sha256"]
    require(any(Path(path).name == "iphone-actquant.o" for path in objects), "ActQuant helper object provenance missing")
    for path, digest in objects.items():
        require(sha256(path) == digest, "Built helper object changed")
    return build, {"build_manifest_sha256": sha256(manifest_path), "app_files_sha256": build["app_files_sha256"],
                   "framework_sha256": build["framework_sha256"], "package_sources_sha256": package,
                   "source_files_sha256": build["source_files_sha256"], "object_sha256": objects}


def operator_gate(run_directory, analysis_path, build):
    require(run_directory is not None and analysis_path is not None,
            "Model launch requires --operator-run and --operator-analysis from all cases in the A16 fixture manifest")
    run_directory, analysis_path = run_directory.expanduser().resolve(), analysis_path.expanduser().resolve()
    summary = json.loads((run_directory / "summary.json").read_text())
    runtime_path = run_directory / "device/native/RUN.json"
    runtime, analysis = json.loads(runtime_path.read_text()), json.loads(analysis_path.read_text())
    fixture_path = run_directory / "FIXTURES.json"
    fixture = json.loads(fixture_path.read_text())
    expected = {v["name"]: v for v in fixture["cases"]}
    require(1 <= len(expected) <= 128 and len(expected) == len(fixture["cases"]) and sha256(fixture_path) == runtime["fixture_sha256"],
            "Operator gate fixture identity missing")
    previous = json.loads((run_directory / "BUILD_IDENTITY.json").read_text())
    require(summary["execution_pass"] is True and summary["GPU_executed"] is True and summary["native_run_sha256"] == sha256(runtime_path)
            and runtime["operation"] == "operator" and runtime["device"] == "Apple A16 GPU" and runtime["backend"] == "MTL0",
            "A16 operator collection identity missing")
    require(analysis["analysis_completed"] is True and analysis["numerical_pass"] is True and analysis["strict_operator_exit_code"] == 0
            and analysis["fixture_manifest_sha256"] == runtime["fixture_sha256"] and len(analysis["cases"]) == len(runtime["cases"]) == len(expected),
            "Independent A16 operator analysis has not passed every fixture case")
    by_name = {v["name"]: v for v in runtime["cases"]}
    require(set(by_name) == set(expected) == {v["name"] for v in analysis["cases"]}, "Operator gate case identities differ")
    for case in analysis["cases"]:
        native = by_name[case["name"]]
        require(case["numerical_pass"] is True and case["runtime_identity_and_completion_pass"] is True and
                case["output_sha256"] == native["output_sha256"] and case["input_sha256"] == native["input_sha256"],
                "Independent analysis does not match native operator case")
        require(sha256(safe_relative(run_directory / "device/native", native["path"])) == native["output_sha256"],
                "Operator output changed after independent analysis")
    require(previous["framework_sha256"] == build["framework_sha256"] and previous["app_files_sha256"] == build["app_files_sha256"],
            "Operator gate must use the same app/framework build")
    return {"operator_run_sha256": sha256(runtime_path), "operator_analysis_sha256": sha256(analysis_path),
            "fixture_manifest_sha256": sha256(fixture_path), "required_cases": len(expected), "operator_cases_gate_pass": True}


def attribution_gate(path, build, tokens_sha256):
    require(path is not None, "Attribution measurement requires a fresh same-build Q4/QB correctness gate")
    gate = json.loads(path.read_text())
    require(gate.get("schema") == 1 and gate.get("scoped_correctness_pass") is True and
            gate.get("official_source_model_output_pass") is None and gate.get("historical_official_strict_exit_preserved") == 1,
            "Incomplete scoped attribution gate")
    require(gate["framework_sha256"] == build["framework_sha256"] and gate["app_files_sha256"] == build["app_files_sha256"] and
            gate["model_sha256"] == PUBLIC_MODEL_SHA256 and
            gate["q_library_sha256"] == build["shader_products_sha256"]["q-stock4x1.metallib"], "Attribution gate build/model/library mismatch")
    require(len(gate["histories"]) == 2 and all(v["same_process_scoped_pass"] is True for v in gate["histories"].values()) and
            tokens_sha256 in {v["tokens_sha256"] for v in gate["histories"].values()}, "Attribution history gate incomplete")
    require(gate["evidence_sha256"], "Attribution gate evidence is missing")
    for name, digest in gate["evidence_sha256"].items():
        require(sha256(name) == digest, "Attribution correctness evidence changed")
    return {"correctness_gate_sha256": sha256(path), "scoped_correctness_gate_pass": True}


def attribution_protocol(kind, direct=False):
    require(kind in ("verify", "formal", "diagnostic", "overhead"), "Invalid attribution kind")
    return {"warmups": 2, "rounds": 1 if kind in ("verify", "diagnostic") else 3,
            "pairs": 1 if kind in ("verify", "diagnostic") else 7 if kind == "overhead" else 21,
            "comparisons": ["AA", "SQB", "SQ4"] + (["QBQ4"] if direct else [])}


def command_plan(args, build, config, remote_fixture, remote_tokens):
    bundle = build.get("bundle_id", "<bundle-id-from-signed-BUILD.json>")
    app = build.get("app", "<signed-app-from-BUILD.json>")
    common = ["xcrun", "devicectl"]
    copy = [*common, "--timeout", "900", "device", "copy", "to", "--device", args.device,
            "--domain-type", "appDataContainer", "--domain-identifier", bundle]
    plan = [[*common, "--timeout", "60", "device", "info", "details", "--device", args.device],
            [*common, "--timeout", "180", "device", "install", "app", app, "--device", args.device]]
    if not args.reuse_model:
        plan.append([*copy, "--source", str(args.model), "--destination", "Documents/" + args.model.name])
    if args.operation == "operator":
        plan.append([*copy, "--source", str(args.fixture), "--destination", "Documents/" + remote_fixture])
    else:
        plan.append([*copy, "--source", str(args.tokens), "--destination", "Documents/" + remote_tokens])
    if args.operation in ("model", "attribution") and args.q_library is not None:
        plan.append([*copy, "--source", str(args.q_library), "--destination", "Documents/" + config["q_library"]])
    plan.append([*common, "--timeout", str(args.timeout), "device", "process", "launch", "--console", "--terminate-existing",
                 "--device", args.device, "--environment-variables", json.dumps({"BITNET_CANDIDATE_CONFIG": json.dumps(config, separators=(",", ":"))}), bundle])
    plan.append([*common, "--timeout", "900", "device", "copy", "from", "--device", args.device,
                 "--domain-type", "appDataContainer", "--domain-identifier", bundle,
                 "--source", "Documents/" + config["output"], "--destination", str(args.output / "device")])
    return plan


def finite_file(path, count):
    require(path.stat().st_size == count * 4, "Output extent mismatch: " + path.name)
    with path.open("rb") as handle:
        while block := handle.read(65536):
            require(all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", block)), "Nonfinite native output")


def validate_collection(args, config, log, device_dir, fixture):
    result_path, runtime_path = device_dir / "result.json", device_dir / "native/RUN.json"
    result, native = json.loads(result_path.read_text()), json.loads(runtime_path.read_text())
    markers = [json.loads(line.split("BITNET_PHONE_COMPLETE ", 1)[1]) for line in log.splitlines() if "BITNET_PHONE_COMPLETE " in line]
    require(len(markers) == 1 and markers[0] == result, "Console/file completion mismatch")
    require(result["operation"] == "actquant" and result["actquant_operation"] == args.operation and
            result["native_process_exit_code"] == result["process_exit_code"] == 0 and result["execution_pass"] is True,
            "Native collection did not complete")
    require(result["model_sha256"] == PUBLIC_MODEL_SHA256 and result["native_run_sha256"] == sha256(runtime_path), "Runtime identity mismatch")
    expected_measurement = args.operation == "attribution" and args.attribution_kind != "verify"
    require(native["operation"] == args.operation and native["backend"] == "MTL0" and native["device"] == "Apple A16 GPU" and
            native["execution_pass"] is True and native["performance_measurement"] is expected_measurement, "Expected actual A16 execution/scope")
    require(native["strict_model_exit_code"] == 2, "Native collection must preserve unverified model acceptance")
    if args.operation == "operator":
        require(result["fixture_sha256"] == native["fixture_sha256"] == config["fixture_sha256"], "Fixture runtime identity mismatch")
        expected = {v["name"]: v for v in fixture["cases"]}
        require(len(native["cases"]) == len(expected) and {v["name"] for v in native["cases"]} == set(expected), "Expected all fixture cases completed")
        for case in native["cases"]:
            original = expected[case["name"]]
            require(case["ne"] == original["ne"] and case["input_sha256"] == original["input_sha256"] and
                    all(case[k] is True for k in ("completed", "finite", "input_unchanged", "canaries_unchanged")), "Operator completion/canary contract")
            path = safe_relative(device_dir / "native", case["path"])
            require(sha256(path) == case["output_sha256"], "Operator output checksum mismatch")
            finite_file(path, original["width"] * original["rows"])
        complete = re.findall(r"ACTQUANT_OPERATOR_COMPLETE case=(\S+) backend=MTL0 values=(\d+) synchronized=1 value_failures=(\d+)", log)
        require(len(complete) == len(expected) and {v[0] for v in complete} == set(expected), "Missing synchronized operator completion markers")
    elif args.operation == "attribution":
        require(native["attribution_kind"] == args.attribution_kind and native["activation_quantization"] is True and
                native["one_model_and_context"] is True and native["offloaded_layers"] == native["total_offload_layers"] == 31,
                "Attribution context/offload mismatch")
        require(native["tokens_sha256"] == config["tokens_sha256"] and native["model_sha256"] == PUBLIC_MODEL_SHA256,
                "Attribution history/model mismatch")
        for key in ("q_library_sha256", "q_entry", "q_kernel_contract_version", "q_kernel_config_sha256", "q_grid", "q_threads"):
            require(native[key] == config[key], "Attribution library/entry mismatch: " + key)
        require("ACTQUANT_ATTRIBUTION_COMPLETE kind=" + args.attribution_kind in log, "Attribution completion marker missing")
        if args.attribution_kind == "verify":
            require(len(native["paths"]) == 6 and native["capture_enabled"] is True and native["trace_enabled"] is True,
                    "Same-process verification paths missing")
            for item in native["paths"]:
                subroot = device_dir / "native" / item["subrun"]
                subfile = safe_relative(device_dir / "native", item["subrun"] + "/RUN.json")
                sub = json.loads(subfile.read_text())
                require(sub["context_id"] == native["context_id"] and sub["execution_pass"] is True and
                        sub["completed_decode_calls"] == 33 and len(sub["logits"]) == 33, "Incomplete same-context verification subrun")
                for vector in sub["logits"]:
                    finite_file(safe_relative(subroot, vector["path"]), 128256)
        else:
            protocol = attribution_protocol(args.attribution_kind, args.direct_qb_q4)
            count = protocol["rounds"] * protocol["pairs"] * 3
            if args.attribution_kind != "overhead" and args.direct_qb_q4:
                count = protocol["rounds"] * protocol["pairs"] * 4
            require(native["sample_pairs"] == count and native["token_rows"] == count * 64 and len(native["paths"]) == count * 2 and
                    native["capture_enabled"] is False and native["trace_enabled"] is False and
                    native["correctness_gate_sha256"] == config["correctness_gate_sha256"], "Attribution protocol/gate mismatch")
            for filename, expected_rows in (("samples.csv", count), ("token-times.csv", count * 64)):
                path = safe_relative(device_dir / "native", filename)
                with path.open(newline="") as file:
                    require(sum(1 for _ in csv.DictReader(file)) == expected_rows, "Attribution raw CSV coverage mismatch")
            for item in native["paths"]:
                if item["diagnostic_file"] is not None:
                    record = json.loads(safe_relative(device_dir / "native", item["diagnostic_file"]).read_text())
                    require(record["complete"] is True and record["dropped_records"] == 0, "Incomplete backend attribution records")
    else:
        tokens = token_identity(args.tokens)
        ids = tokens["corpus"] if args.workload == "corpus" else tokens["prompt"] + tokens["continuation"]
        count = 128 if args.workload == "corpus" else 33
        positions = list(range(128)) if args.workload == "corpus" else list(range(26, 59))
        require(result["tokens_sha256"] == native["tokens_sha256"] == config["tokens_sha256"] and
                native["token_ids"] == ids and native["prompt_ids"] == tokens["prompt"] and native["continuation_ids"] == tokens["continuation"], "Token input identity mismatch")
        require(native["q_route"] == args.q_route and native["q_fallback"] == args.q_fallback and
                native["capture_scope"] == args.capture_scope, "Q route/scope identity mismatch")
        if "q_library" in config:
            for key in ("q_library_sha256", "q_variant", "q_library_source_sha256", "q_library_manifest_sha256", "q_library_sdk", "q_kernel_contract_version", "q_kernel_config_sha256", "q_entry", "q_grid", "q_threads"):
                require(native[key] == config[key], "Q4 source/library runtime identity mismatch: " + key)
        require(native["activation_quantization"] is args.actquant and native["capture_enabled"] is args.capture and
                native["prefill"] == "batched" and native["kv"] == "F16" and native["workload"] == args.workload,
                "Effective model settings mismatch")
        require(native["offloaded_layers"] == native["total_offload_layers"] == 31 and
                re.search(r"offloaded\s+31/31\s+layers to GPU", log), "31/31 offload evidence missing")
        require(native["completed_decode_calls"] == len(native["logits"]) == count and
                [v["input_position"] for v in native["logits"]] == positions and native["finite_logits"] is True,
                "Incomplete teacher/corpus logit collection")
        for item in native["logits"]:
            require(item["vocab"] == 128256, "Wrong logit vocabulary")
            finite_file(safe_relative(device_dir / "native", item["path"]), 128256)
        marker = f"ACTQUANT_MODEL_COMPLETE enabled={int(args.actquant)} backend=MTL0 workload={args.workload} decode_calls={count} logits={count} "
        require(marker in log and "synchronized=1" in log, "Model GPU completion marker missing")
        require(all(native[k] is None for k in ("parity_pass", "fp64_pass", "model_output_pass")), "Collection cannot claim model acceptance")
    require(all(result[k] is None for k in ("parity_pass", "fp64_pass", "model_output_pass", "semantic_acceptance_pass")),
            "Wrapper cannot claim unperformed semantic checks")
    return result, native


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true", help="Explicitly install, transfer, launch and fetch on a physical device")
    action.add_argument("--plan", action="store_true", help="Default: local checks and planned commands only; no device or Xcode commands")
    p.add_argument("--operation", choices=("operator", "model", "attribution"), default="operator")
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--fixture", type=Path)
    p.add_argument("--tokens", type=Path)
    p.add_argument("--operator-run", type=Path, help="Completed A16 operator collection; required before a model launch")
    p.add_argument("--operator-analysis", type=Path, help="Independent all-case SUMMARY.json; required before a model launch")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--actquant", choices=("on", "off"), default="on")
    p.add_argument("--workload", choices=("teacher", "corpus"), default="teacher")
    p.add_argument("--q-route", choices=("stock", "Q4", "QB"), default="stock")
    p.add_argument("--attribution-kind", choices=("verify", "formal", "diagnostic", "overhead"), default="formal")
    p.add_argument("--attribution-gate", type=Path, help="Same-build two-history15-stage/QB/one-process correctness gate")
    p.add_argument("--direct-qb-q4", action="store_true", help="Append the mixed encoding/shader/scheduling QB-Q4 contrast")
    p.add_argument("--q-fallback", choices=("none", "missing-library"), default="none")
    p.add_argument("--capture-scope", choices=("all", "q4-layer0"), default="all")
    p.add_argument("--q-library", type=Path, help="Optional stock4x1 library; defaults to the signed build shader product for Q4-scoped runs")
    p.add_argument("--q-library-manifest", type=Path)
    p.add_argument("--capture", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--reuse-model", action="store_true")
    p.add_argument("--developer-dir", type=Path)
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args(argv)
    args.actquant = args.actquant == "on"
    args.model, args.output, args.build_dir = (v.expanduser().resolve() for v in (args.model, args.output, args.build_dir))
    require(not args.output.exists(), "Refuse existing local output")
    args.output.mkdir(parents=True)
    summary = {"scope": "ActQuant A16 operator/model diagnostic collection", "plan_only": not args.run,
               "execution_pass": None, "parity_pass": None, "fp64_pass": None, "model_output_pass": None,
               "semantic_acceptance_pass": None, "model_quality_pass": None, "strict_model_exit_code": 2,
               "GPU_executed": False, "performance_measurement": False, "errors": []}
    try:
        require(args.timeout > 0 and re.fullmatch(r"[A-Za-z0-9_-]+", args.device), "Invalid timeout or device identifier")
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", args.model.name) and not args.model.name.startswith("."), "Unsafe model basename")
        require(sha256(args.model) == PUBLIC_MODEL_SHA256, "Wrong public model checksum")
        prefix = "candidate-actquant-" + datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%f")
        remote_fixture, remote_tokens = prefix + "-fixtures", prefix + "-tokens.json"
        config = {"operation": "actquant", "actquant_operation": args.operation, "output": prefix, "backend": "MTL0",
                  "model": args.model.name, "model_sha256": PUBLIC_MODEL_SHA256, "actquant": args.actquant,
                  "capture": args.capture, "trace": args.trace, "workload": args.workload, "prefill": "batched", "kv": "F16",
                  "q_route": args.q_route, "q_fallback": args.q_fallback, "capture_scope": args.capture_scope}
        if args.operation == "attribution":
            require(args.actquant and args.workload == "teacher" and args.q_route == "stock" and args.q_fallback == "none" and
                    args.capture is (args.attribution_kind == "verify") and args.trace is (args.attribution_kind == "verify"),
                    "Attribution requires ActQuant ON/teacher, internal route switching and verify-only capture/trace")
            config.update(attribution_kind=args.attribution_kind, **attribution_protocol(args.attribution_kind, args.direct_qb_q4))
        require(args.q_route == "stock" or (args.operation == "model" and args.actquant and args.workload == "teacher"), "Q4 requires an ActQuant-ON teacher model run")
        require(args.q_fallback == "none" or args.q_route == "Q4", "Q fallback requires Q4")
        fixture = None
        if args.operation == "operator":
            require(args.fixture is not None, "Operator collection requires --fixture")
            args.fixture = args.fixture.expanduser().resolve()
            fixture, files = fixture_identity(args.fixture)
            config.update(fixture=remote_fixture, fixture_sha256=files["FIXTURES.json"])
            (args.output / "FIXTURES.json").write_bytes((args.fixture / "FIXTURES.json").read_bytes())
            write_json(args.output / "FIXTURE_IDENTITY.json", files)
        else:
            require(args.tokens is not None, "Model collection requires --tokens")
            args.tokens = args.tokens.expanduser().resolve()
            tokens = token_identity(args.tokens)
            config.update(tokens=remote_tokens, tokens_sha256=sha256(args.tokens))
            write_json(args.output / "TOKENS.json", tokens)
        write_json(args.output / "config.json", config)
        blockers, build = [], {}
        try:
            build, identity = build_identity(args.build_dir)
            if args.operation == "attribution" or (args.operation == "model" and (args.q_route in ("Q4", "QB") or args.capture_scope == "q4-layer0" or args.q_library is not None)):
                shader_dir = Path(build["shaders"])
                args.q_library = (args.q_library or shader_dir / "q-stock4x1.metallib").expanduser().resolve()
                shader_manifest = (args.q_library_manifest or shader_dir / "BUILD.json").expanduser().resolve()
                q_identity = q4_library_identity(args.q_library, shader_manifest, "iphoneos")
                expected = shader_dir / "q-stock4x1.metallib"
                require(sha256(args.q_library) == sha256(expected), "Q4 must match the same signed build stock4x1 product")
                require(build["shader_build_manifest_sha256"] == sha256(shader_manifest) and
                        build["shader_products_sha256"]["q-stock4x1.metallib"] == sha256(args.q_library),
                        "Q4 shader provenance changed after the signed app build")
                config.update(q_identity); config["q_library"] = prefix + "-q-stock4x1.metallib"
                write_json(args.output / "Q4_LIBRARY_IDENTITY.json", q_identity)
                write_json(args.output / "config.json", config)
            write_json(args.output / "BUILD_IDENTITY.json", identity)
            if args.operation in ("model", "attribution"):
                write_json(args.output / "OPERATOR_GATE.json", operator_gate(args.operator_run, args.operator_analysis, build))
            if args.operation == "attribution" and args.attribution_kind != "verify":
                gate = attribution_gate(args.attribution_gate, build, config["tokens_sha256"])
                config.update(gate); write_json(args.output / "ATTRIBUTION_GATE.json", gate); write_json(args.output / "config.json", config)
        except (RuntimeError, OSError, ValueError, KeyError) as error:
            blockers.append(str(error))
        plan = command_plan(args, build, config, remote_fixture, remote_tokens)
        write_json(args.output / "PLAN.json", {"device_commands_executed": False, "local_readiness_blockers": blockers,
                                              "argv": plan, "next_step": "All manifest operator cases must pass independent analysis first; model teacher/PPL remains diagnostic"})
        if not args.run:
            summary.update(planning_exit_code=0, local_build_ready=not blockers, device_readiness_verified=False)
            write_json(args.output / "summary.json", summary)
            print(json.dumps(summary, indent=2))
            return 0
        require(not blockers, "; ".join(blockers))
        commands = Commands(args.output / "logs", developer_environment(args.developer_dir))
        for index, argv_command in enumerate(plan[:-2]):
            commands.run("stage-" + str(index), argv_command, timeout=920)
        launch_code, log = commands.run("launch", plan[-2], timeout=args.timeout + 20, check=False)
        actual_exit = re.findall(r"The app terminated with the exit code (\d+)\.", log)
        summary.update(launch_command_exit_code=launch_code, actual_process_exit_code=int(actual_exit[-1]) if actual_exit else None)
        fetch_code, _ = commands.run("fetch", plan[-1], timeout=920, check=False)
        require(fetch_code == 0, "Fetch failed; native output remains on device")
        require(launch_code == 0 and actual_exit == ["0"], "Actual process exit0 not confirmed")
        result, native = validate_collection(args, config, log, args.output / "device", fixture)
        summary.update(execution_pass=True, GPU_executed=True, collection_exit_code=0,
                       native_process_exit_code=result["native_process_exit_code"], native_device=native["device"],
                       native_run_sha256=sha256(args.output / "device/native/RUN.json"),
                       next_step="Run independent analyzer; native0 alone is not numerical/model-semantic acceptance")
        if args.operation == "attribution":
            summary.update(attribution_kind=args.attribution_kind, performance_measurement=args.attribution_kind != "verify",
                           formal_wall_campaign=args.attribution_kind == "formal")
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
        summary["errors"].append(str(error))
        summary.update(collection_exit_code=2 if args.run else None, planning_exit_code=None if args.run else 2)
        if args.run:
            summary["execution_pass"] = False
        write_json(args.output / "summary.json", summary)
        print(json.dumps(summary, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
