#!/usr/bin/env python3
"""CPU-only, privacy-filtered summary of a complete A16 Q attribution campaign.

Exit0 means complete analysis evidence, never a speedup or quality pass. Exit1
means an observed required correctness gate failed. Missing, malformed, stale or
inconsistent evidence returns2. No device, compiler, model or GPU is launched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bitnet/actquant"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_attribution import audit as audit_measurement
from analyze_q_timing import audit as audit_operator
from summarize_q_attribution_cpu import analyze_path as audit_cpu_path


class RequiredCorrectnessFailure(ValueError):
    """A complete recorded required correctness verdict is explicitly false."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def correct(value, message):
    if value is False:
        raise RequiredCorrectnessFailure(message)
    require(value is True, message + " is unverified")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def map_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stable_power(power):
    """Keep state and power observations, excluding device uptime/identifiers."""
    state = power.get("thermal")
    names = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}
    name = names.get(state) if type(state) is int else state
    if name not in names.values():
        name = "unavailable"
    return {"thermal": name, "low_power": power.get("low_power"),
            "battery_level": power.get("battery_level"), "battery_state": power.get("battery_state")}


def compact_operator_group(row):
    keep = ("round", "comparison", "routes", "pairs", "first_p50_us", "first_p95_us",
            "second_p50_us", "second_p95_us", "paired_second_minus_first_p50_us",
            "second_wins", "equal_pairs", "median_paired_percent_change")
    before, after = stable_power(row["thermal_before"]), stable_power(row["thermal_after"])
    endpoint_match = (before["thermal"] != "unavailable" and before["thermal"] == after["thermal"] and
                      type(before["low_power"]) is bool and before["low_power"] == after["low_power"])
    order_keys = ("pairs", "paired_second_minus_first_p50_us", "second_wins", "median_paired_percent_change")
    return {**{k: row[k] for k in keep},
            "order_groups": {name: {k: value[k] for k in order_keys} for name, value in row["AB_BA"].items()},
            "time_blocks": [{"block": i + 1, "pairs": block["pairs"][1] - block["pairs"][0] + 1,
                             "paired_second_minus_first_p50_us": block["paired_second_minus_first_p50_us"],
                             "second_wins": block["second_wins"]}
                            for i, block in enumerate(row["time_blocks"])],
            "before": before, "after": after,
            "thermal_endpoint_cohort": before["thermal"] if endpoint_match else "transition_or_unavailable",
            "continuous_thermal_observation": False}


def compact_thermal_round(row):
    """Never pool different or unknown temperature/power cohorts."""
    strata = []
    for group in row["stable_strata"]:
        require(group["thermal"] in ("nominal", "fair", "serious", "critical") and
                type(group["low_power"]) is bool, "Unknown state cannot be a stable stratum")
        paired_keys = ("pairs", "paired_B_minus_A_median_ms", "paired_relative_median_percent", "B_wins")
        strata.append({"thermal": group["thermal"], "low_power": group["low_power"],
                       "statistics": group["statistics"],
                       "order_groups": {name: {k: value[k] for k in paired_keys}
                                        for name, value in group["order_groups"].items()},
                       "time_blocks": [{"block": i + 1,
                                        "statistics": {k: b["statistics"][k] for k in paired_keys}}
                                       for i, b in enumerate(group["time_blocks"])]})
    return {"round": row["round"], "contrast": row["contrast"], "stable_strata": strata,
            "transition_or_unavailable_pairs": len(row["transition_or_unavailable_pairs"]),
            "cross_thermal_pooled_statistics": None}


def wall_observation(histories):
    """Describe per-round direction without pooling thermal cohorts."""
    rounds = [row for history in histories for row in history["rounds"] if row["contrast"] == "SQ4"]
    complete = (len(rounds) == 6 and all(row["transition_or_unavailable_pairs"] == 0 and
                len(row["stable_strata"]) == 1 and
                row["stable_strata"][0]["statistics"]["pairs"] == 21 for row in rounds))
    positive = sum(group["statistics"]["paired_B_minus_A_median_ms"] > 0
                   for row in rounds for group in row["stable_strata"])
    regression = True if complete and positive == 6 else None
    return {"measured_SQ4_rounds": len(rounds), "positive_paired_median_strata": positive,
            "all_six_rounds_complete_in_single_known_cohort": complete,
            "observed_regression_in_this_campaign": regression,
            "end_to_end_speedup_established": False if regression is True else None,
            "interpretation": "Six positive per-round Q4-minus-stock wall medians establish no speedup in the collected campaign; no cross-cohort pooling, general thermal claim or pure-shader attribution."}


def run_summary(experiment, progress=None):
    experiment = experiment.resolve()
    progress = {} if progress is None else progress
    evidence = {}

    def document(relative):
        path = experiment / relative
        require(path.resolve().is_relative_to(experiment), "Evidence path escapes experiment")
        result = read(path)
        evidence[str(Path(relative))] = sha(path)
        return result

    completion = document("MEASUREMENT_COMPLETE.json")
    require(completion.get("all_commands_completed") is True and
            completion.get("performance_data_collected") is True and
            completion.get("historical_official_strict_exit_preserved") == 1 and
            completion.get("raw_data_uploaded") is False, "Measurement campaign is not complete")
    commands = document("MEASUREMENT_COMMANDS.json")
    expected = ["prepare-phone-fixture", "phone-operator", "phone-operator-analysis", "measurement-plan"]
    expected += [kind + "-" + history + suffix for kind in ("diagnostic", "overhead", "formal")
                 for history in ("primary", "independent") for suffix in ("", "-analysis")]
    require([r["name"] for r in commands] == expected and completion.get("commands") == len(expected),
            "Recorded command coverage/order differs")
    require(all(r.get("status") == "completed" and type(r.get("exit_code")) is int and
                r["exit_code"] == 0 for r in commands), "A required command did not complete0")

    qual = "qualification-v2"
    gate_file = experiment / qual / "TIMING_GATE.json"
    require(Path(completion["numerical_gate"]).resolve() == gate_file.resolve(), "Completion gate differs")
    qualification = document(qual + "/SUMMARY.json")
    correct(qualification.get("scoped_correctness_pass"), "Scoped qualification")
    require(qualification.get("all_stages_completed") is True and qualification.get("exit_code") == 0,
            "Scoped qualification execution incomplete")
    initial = document(qual + "/correctness/SUMMARY.json")
    correct(initial.get("scoped_qualification_pass"), "15-stage numerical/runtime prerequisite")
    gate = document(qual + "/TIMING_GATE.json")
    correct(gate.get("scoped_correctness_pass"), "Two-history timing gate")
    require(gate.get("historical_official_strict_exit_preserved") == 1 and
            gate.get("official_source_model_output_pass") is None, "Official-model boundary changed")
    # Retained local paths in the gate are verified, never copied to publication.
    require(gate.get("evidence_sha256"), "Gate evidence is missing")
    for filename, digest in gate["evidence_sha256"].items():
        require(sha(Path(filename)) == digest, "Qualified evidence changed")

    contracts = []
    for relative in (qual + "/correctness/contract/summary.json", qual + "/cache-repeat/summary.json"):
        row = document(relative)
        correct(row.get("runtime_contract_pass"), "Actual backend contract/cache")
        require(len(row["cases"]) == 18 and all(r["pass"] is True for r in row["cases"]) and
                row["actual_app_exit_code"] == row["native_exit_code"] == 0 and
                row["selected_build_manifest_sha256"] == gate["build_manifest_sha256"] and
                row["target_Q_dispatches"] == 0, "Contract count/build/completion mismatch")
        contracts.append({"process": len(contracts) + 1, "passed_cases": 18, "target_Q_dispatches": 0})

    op = document(qual + "/correctness/operator-analysis/SUMMARY.json")
    correct(op.get("numerical_pass"), "ActQuant operator prerequisite")
    require(op["strict_operator_exit_code"] == 0 and len(op["cases"]) == 20 and op["values"] == 386567,
            "ActQuant case/value coverage mismatch")
    op_native = document(qual + "/correctness/operator/device/native/RUN.json")
    actual_cases = {r["name"]: r for r in op_native["cases"]}
    require(len(actual_cases) == 20, "Native ActQuant coverage incomplete")
    for row in op["cases"]:
        correct(row.get("numerical_pass"), "ActQuant case")
        native = actual_cases[row["name"]]
        path = experiment / qual / "correctness/operator/device/native" / native["path"]
        require(sha(path) == native["output_sha256"] == row["output_sha256"] == row["reference_sha256"] and
                row["frozen_tolerance_failures"] == row["recovered_integer_code_mismatches"] ==
                row["output_bitwise_differences"] == row["nonfinite"] == 0,
                "ActQuant output/oracle identity mismatch")

    histories = []
    for history in ("primary", "independent"):
        local = document(qual + "/correctness/" + history + "-analysis/SUMMARY.json")
        joint = document(qual + "/verify-" + history + "-analysis/SUMMARY.json")
        for value in (local, joint):
            correct(value.get("scoped_stock_compatibility_pass"), "History stock compatibility")
            require(value["strict_scoped_exit_code"] == 0 and all(v is True for v in value["checks"].values()),
                    "History numerical/runtime checks incomplete")
        require(len(local["FP64"]) == 32 and sorted(r["step"] for r in local["FP64"]) == list(range(1, 33)) and
                local["logits"]["vectors"] == 33 and local["logits"]["bitwise_pass"] is True,
                "Required Q/logit coverage incomplete")
        for row in local["FP64"]:
            for route in ("stock", "Q4"):
                correct(row[route]["pass"], "Local Q FP64")
        native_relative = qual + "/verify-" + history + "/device/native/RUN.json"
        native = document(native_relative)
        host = document(qual + "/verify-" + history + "/summary.json")
        identity = document(qual + "/verify-" + history + "/BUILD_IDENTITY.json")
        require(joint["native_run_sha256"] == host["native_run_sha256"] == sha(experiment / native_relative) and
                host["execution_pass"] is True and host["actual_process_exit_code"] == 0 and
                identity["build_manifest_sha256"] == gate["build_manifest_sha256"] and
                identity["framework_sha256"] == gate["framework_sha256"] and
                identity["app_files_sha256"] == gate["app_files_sha256"], "History source/build/run binding differs")
        require(native["tokens_sha256"] == gate["histories"][history]["tokens_sha256"] and
                native["q_library_sha256"] == gate["q_library_sha256"] and
                native["model_sha256"] == gate["model_sha256"] and native["activation_quantization"] is True and
                native["q_entry"] == "tq2_0_matvec_stock4x1_v1" and native["q_kernel_contract_version"] == 1,
                "History model/input/library/canonical identity differs")
        histories.append({"history": history, "tokens_sha256": native["tokens_sha256"],
                          "all32_Q_FP64_pass": True, "S_Q4_FP64_values": 163840,
                          "FP64_failures": 0, "max_abs_FP64": max(r["Q4"]["max_abs"] for r in local["FP64"]),
                          "all33_logits_bitwise": True, "same_process_QB_Q4_pass": True,
                          "same_PSO_ABI_live_bindings": joint["same_live_bindings_observed"],
                          "residual_checkpoint_vectors": local["residuals"]["vectors"]})
    identity = document(qual + "/verify-primary/BUILD_IDENTITY.json")
    # Preserve an already verified numerical gate if later measurement evidence
    # is unavailable; an environment/collection error is not a numerical reject.
    progress["scoped_correctness_pass"] = True

    # Recalculate isolated CSV statistics and all provenance controls, CPU only.
    operator_qualification = document("phone-operator/q-preflight-qualification.json")
    correct(operator_qualification.get("local_fp64_pass"), "Fresh isolated operator FP64")
    require(len(operator_qualification["cases"]) == 9 and
            all(r["rows"] == 2560 and r["failed_rows"] == r["nonfinite_rows"] == 0 and
                r["GPU_completed"] is True for r in operator_qualification["cases"]),
            "Fresh isolated operator numerical coverage incomplete")
    selected_qualification = [r for r in operator_qualification["cases"] if r["route"] in ("S", "Q4")]
    require(len(selected_qualification) == 6, "Selected isolated route coverage incomplete")
    correct(all(r["stock_bit_different_rows"] == 0 for r in selected_qualification),
            "Fresh isolated stock/Q4 bitwise parity")
    operator_official = document("phone-operator-analysis.json")
    require(operator_official.get("audit_pass") is True and operator_official.get("strict_exit_code") == 0,
            "Isolated timing/provenance audit incomplete")
    operator = audit_operator(SimpleNamespace(performance_run=experiment / "phone-operator",
                            fixture=experiment / "phone-q-fixture",
                            joint_candidate=experiment / qual / "correctness/primary-Q4-capture",
                            contract_profile="canonical-v1", legacy_shader_build=None))
    require(operator["audit_pass"] is True, "Recomputed operator audit differs")
    for relative in ("phone-q-fixture/MANIFEST.json", "phone-operator/summary.json",
                     "phone-operator/q-preflight-qualification.json", "phone-operator/operator/performance/summary.json",
                     "phone-operator/BUILD_IDENTITY.json"):
        document(relative)
    evidence["phone-operator/operator/performance/samples.csv"] = sha(experiment / "phone-operator/operator/performance/samples.csv")
    fixture = read(experiment / "phone-q-fixture/MANIFEST.json")
    require(fixture["selected_build_manifest_sha256"] == gate["build_manifest_sha256"] and
            fixture["libraries"]["Q4"]["sha256"] == gate["q_library_sha256"], "Operator build differs from model gate")

    measurements = {"formal": [], "overhead": []}
    cpu = []
    completed_paths = 0
    for kind in ("diagnostic", "overhead", "formal"):
        for history in ("primary", "independent"):
            relative = "measurement-plan/" + kind + "-" + history
            official = document(relative + "-analysis/SUMMARY.json")
            require(official.get("measurement_integrity_pass") is True and official.get("analysis_completed") is True,
                    "Required measurement analyzer is incomplete")
            raw = document(relative + "/device/native/RUN.json")
            host = document(relative + "/summary.json")
            require(host["execution_pass"] is True and host["actual_process_exit_code"] == 0 and
                    raw["device"] == "Apple A16 GPU" and raw["kv"] == "F16" and
                    raw["model_sha256"] == gate["model_sha256"] and raw["q_library_sha256"] == gate["q_library_sha256"] and
                    raw["tokens_sha256"] == gate["histories"][history]["tokens_sha256"] and
                    raw["correctness_gate_sha256"] == sha(gate_file), "Measurement identity differs")
            for filename, digest in official["raw_hashes"].items():
                actual = experiment / relative / "device/native" / filename
                require(sha(actual) == digest, "Analyzed raw measurement changed")
                evidence[str(actual.relative_to(experiment))] = digest
            checked = audit_measurement(SimpleNamespace(run=experiment / relative,
                                        telemetry_only=kind == "diagnostic", gate=gate_file))
            require(checked["measurement_integrity_pass"] is True, "Recomputed measurement invalid")
            completed_paths += len(raw["paths"])
            if kind != "diagnostic":
                measurements[kind].append({"history": history, "pairs": checked["pairs"],
                    "rounds": [compact_thermal_round(r) for r in checked["performance_statistics"]]})
            else:
                for path in raw["paths"]:
                    actual = experiment / relative / "device/native" / path["diagnostic_file"]
                    record = audit_cpu_path(actual, path)
                    evidence[str(actual.relative_to(experiment))] = sha(actual)
                    total = record["raw_task_sums_ns_or_counts"]
                    if path["route"] not in ("QB", "Q4"):
                        continue
                    cache = record["per_call_cache_elapsed_us"]
                    cpu.append({"history": history, "route": path["route"], "actual_Q0_calls": 32,
                        "guard_attempts": total["guard_attempts"], "guard_passes": total["guard_passes"],
                        "custom_dispatches": total["custom_dispatches"], "stock_target_dispatches": total["stock_target_dispatches"],
                        "natural_CBs": record["command_buffers"], "guard_total_ms": total["guard_ns"] / 1e6,
                        "getter_total_ms": total["getter_ns"] / 1e6,
                        "snapshot_key_total_ms": cache["snapshot_key"]["sum"] / 1e3,
                        "getter_p50_us": record["per_call_elapsed_us"]["getter"]["p50"],
                        "getter_p95_us": record["per_call_elapsed_us"]["getter"]["p95"],
                        "cache_counts": record["cache_counts"]})

    # Optional post-main repeat is a separately qualified thermal cohort. It
    # never changes the required16-command campaign or pools Fair/Serious data.
    additional_operator_cohorts = []
    hot_name = "phone-operator-sustained"
    if (experiment / hot_name).exists() or (experiment / (hot_name + "-analysis.json")).exists():
        hot_official = document(hot_name + "-analysis.json")
        require(hot_official.get("audit_pass") is True and hot_official.get("strict_exit_code") == 0,
                "Optional hot operator audit incomplete")
        hot_qualification = document(hot_name + "/q-preflight-qualification.json")
        correct(hot_qualification.get("local_fp64_pass"), "Hot isolated operator FP64")
        require(len(hot_qualification["cases"]) == 9 and
                all(r["rows"] == 2560 and r["failed_rows"] == r["nonfinite_rows"] == 0 and
                    r["GPU_completed"] is True for r in hot_qualification["cases"]),
                "Hot operator numerical coverage incomplete")
        correct(all(r["stock_bit_different_rows"] == 0 for r in hot_qualification["cases"]
                    if r["route"] in ("S", "Q4")), "Hot isolated stock/Q4 bitwise parity")
        hot = audit_operator(SimpleNamespace(performance_run=experiment / hot_name,
                    fixture=experiment / "phone-q-fixture",
                    joint_candidate=experiment / qual / "correctness/primary-Q4-capture",
                    contract_profile="canonical-v1", legacy_shader_build=None))
        require(hot["audit_pass"] is True and hot["inputs"]["X_step16_sha256"] ==
                operator["inputs"]["X_step16_sha256"] and hot["inputs"]["libraries"] ==
                operator["inputs"]["libraries"], "Hot fixture/library differs")
        hot_identity = document(hot_name + "/BUILD_IDENTITY.json")
        require(hot_identity["build_manifest_sha256"] == gate["build_manifest_sha256"] and
                hot_identity["framework_sha256"] == gate["framework_sha256"], "Hot package differs")
        for relative in (hot_name + "/summary.json", hot_name + "/operator/performance/summary.json"):
            document(relative)
        evidence[hot_name + "/operator/performance/samples.csv"] = sha(
            experiment / hot_name / "operator/performance/samples.csv")
        additional_operator_cohorts.append({"campaign": hot_name,
            "scope": "Post-main isolated repeat; separate thermal/power cohort, no Fair/Serious pooling",
            "same_fixture_library_and_build": True, "total_pairs_AA_SQ_CQ": 1809,
            "timed_dispatches": 3618, "warmups_per_path": 30,
            "fresh_qualification_cases": 9, "fresh_qualification_rows": 23040,
            "FP64_failures": 0, "selected_S_Q4_stock_bitwise_differences": 0,
            "groups": [compact_operator_group(r) for r in hot["groups"]]})

    observed_wall = wall_observation(measurements["formal"])
    return {"schema": 1, "analysis_completed": True, "analysis_integrity_pass": True,
        "exit_code": 0, "exit0_meaning": "Complete CPU evidence analysis only; not a performance/quality acceptance",
        "scope": "Current canonical-contract ActQuant-ON A16 layer0 Qcur-0 S/QB/Q4; no new kernel selection",
        "new_GPU_device_or_model_run_by_summarizer": False,
        "scoped_correctness_pass": True, "contracts": contracts,
        "ActQuant_operator": {"cases": 20, "values": 386567, "output_bitwise_differences": 0,
                              "recovered_grid_code_mismatches": 0,
                              "code_scope": "Recovered from exported F32 output and official scale; internal integer register not observed"},
        "histories": histories,
        "provenance": {"model_sha256": gate["model_sha256"], "framework_sha256": gate["framework_sha256"],
            "GPU": operator["device"], "iOS": operator["iOS"],
            "build_manifest_sha256": gate["build_manifest_sha256"], "q4_library_sha256": gate["q_library_sha256"],
            "Q4_entry": "tq2_0_matvec_stock4x1_v1", "contract_version": 1,
            "source_files_map_sha256": map_sha(identity["source_files_sha256"]),
            "source_files_count": len(identity["source_files_sha256"]),
            "package_sources_map_sha256": map_sha(identity["package_sources_sha256"]),
            "package_sources_count": len(identity["package_sources_sha256"]),
            "app_product_hashes": sorted(identity["app_files_sha256"].values()),
            "weight_sha256": fixture["source_weight_sha256"],
            "step16_postActQuant_X_sha256": fixture["timing_step16_activation_sha256"],
            "canonical_config_sha256": fixture["q_kernel_config_sha256"]},
        "isolated_operator": {"metric": operator["metric"], "scope": operator["scope"],
            "total_pairs_AA_SQ_CQ": 1809, "timed_dispatches": 3618, "warmups_per_path": 30,
            "fresh_qualification": {"cases_S_C8_Q4": 9, "rows_all_routes": 23040,
                "FP64_failures": 0, "nonfinite": 0, "selected_S_Q4_rows": 15360,
                "selected_S_Q4_stock_bitwise_differences": 0,
                "max_abs_FP64_selected_three_fixtures": max(r["max_abs"] for r in selected_qualification)},
            "groups": [compact_operator_group(r) for r in operator["groups"]],
            "SQ_route": "stock/Q4", "CQ_route": "extra C8/Q4 isolated research only"},
        "formal": measurements["formal"], "instrumentation_overhead": measurements["overhead"],
        "additional_isolated_operator_cohorts": additional_operator_cohorts,
        "CPU_diagnostic_regions": cpu, "measurement_paths_verified": completed_paths,
        "normal_graph_Q_critical_path_share": None,
        "end_to_end_speedup_established": observed_wall["end_to_end_speedup_established"],
        "formal_wall_observation": observed_wall,
        "historical_trusted_reference_whole_model_pass": False,
        "historical_official_strict_exit_preserved": 1, "new_official_reference_rerun": None,
        "full_model_FP64_pass": None, "model_quality_pass": None, "energy_measured": False,
        "interpretation_boundaries": [
            "Formal wall and isolated GPU command-buffer elapsed are separate scopes.",
            "Different thermal/power states are not pooled; transition/unknown pairs retain counts only.",
            "CPU regions are nested worker elapsed including waits/preemption; they are not additive CPU cycles or model critical-path time.",
            "Snapshot/key work runs under the shared library lock; only selected-Q lock waits were measured, not global selector contention or pure realpath cost.",
            "Matching thermal-state endpoints are coarse observations, not proof of continuously fixed temperature/frequency.",
            "CB GPU/host clocks are separate; no pure per-shader cost or hardware bottleneck inferred.",
            "All local, stock-relative and isolated checks leave the prior official-model strict failure unchanged."],
        "evidence": [{"artifact": name, "sha256": digest} for name, digest in sorted(evidence.items())]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory for compact SUMMARY.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must be fresh")
    args.output.mkdir(parents=True)
    progress = {}
    try:
        result = run_summary(args.experiment, progress)
        code = 0
    except RequiredCorrectnessFailure as error:
        result = {"analysis_completed": False, "analysis_integrity_pass": None, "scoped_correctness_pass": False,
                  "exit_code": 1, "error": str(error), "historical_official_strict_exit_preserved": 1}
        code = 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        result = {"analysis_completed": False, "analysis_integrity_pass": None,
                  "scoped_correctness_pass": progress.get("scoped_correctness_pass"), "exit_code": 2,
                  "error": "Required evidence is missing, malformed or inconsistent; see local stderr.",
                  "historical_official_strict_exit_preserved": 1}
        code = 2
    (args.output / "SUMMARY.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in
          ("analysis_completed", "analysis_integrity_pass", "scoped_correctness_pass", "exit_code", "error")}))
    return code


if __name__ == "__main__":
    sys.exit(main())
