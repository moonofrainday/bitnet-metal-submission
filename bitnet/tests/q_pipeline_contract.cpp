#include "ggml-metal-device.h"
#include "ggml-metal-bitnet-q-contract.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
namespace fs = std::filesystem;
using json = nlohmann::json;

void require(bool value, const std::string & message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

ggml_metal_pipeline_with_params load(ggml_metal_library_t library,
        const fs::path & path, const ggml_metal_bitnet_q_contract & contract) {
    require(setenv("GGML_METAL_BITNET_POC_METALLIB", path.c_str(), 1) == 0, "set library path");
    return ggml_metal_library_get_bitnet_poc_pipeline(library, &contract);
}

void replace(const fs::path & source, const fs::path & target) {
    const fs::path staged = target.string() + ".replacement";
    fs::copy_file(source, staged);
    fs::rename(staged, target);
}
}

extern "C" int bitnet_q_pipeline_contract_run(const char * q4_path, const char * c8_path,
        const char * output_directory) {
    if (!q4_path || !c8_path || !output_directory) {
        return 2;
    }
    const fs::path output(output_directory);
    std::error_code output_error;
    if (fs::exists(output, output_error) || output_error) {
        std::fprintf(stderr, "Refuse existing or inaccessible contract output\n");
        return 2;
    }
    json result = {{"schema", 1}, {"test", "actual backend Q pipeline contract/cache"},
        {"contract_version", GGML_BITNET_Q_CONFIG_VERSION},
        {"target_Q_dispatches", 0}, {"model_loaded", false},
        {"device_initialization_may_run_capability_probes", true},
        {"runtime_contract_pass", nullptr}, {"fp64_pass", nullptr},
        {"model_output_pass", nullptr}, {"performance_measured", false},
        {"cases", json::array()}};
    int code = 2;
    const char * old_path = std::getenv("GGML_METAL_BITNET_POC_METALLIB");
    const bool had_path = old_path != nullptr;
    const std::string saved_path = old_path ? old_path : "";
    try {
        const auto q4_source = fs::canonical(q4_path), c8_source = fs::canonical(c8_path);
        require(fs::is_regular_file(q4_source) && fs::is_regular_file(c8_source), "missing Q libraries");
        fs::create_directories(output / "files");
        ggml_metal_bitnet_q_contract q4, c8;
        require(ggml_metal_bitnet_q_select("stock4x1", nullptr, nullptr, &q4), "select Q4");
        require(ggml_metal_bitnet_q_select("c8x2", nullptr, nullptr, &c8), "select C8");
        auto check = [&](const char * name, bool pass) {
            result["cases"].push_back({{"name", name}, {"pass", pass}});
            if (!pass) {
                result["runtime_contract_pass"] = false;
            }
            require(pass, name);
        };
        ggml_metal_bitnet_q_contract bad;
        check("Q4_4x2_rejected", !ggml_metal_bitnet_q_select("stock4x1", "4", "2", &bad));
        check("Q4_4x4_rejected", !ggml_metal_bitnet_q_select("stock4x1", "4", "4", &bad));
        check("legacy_4x2_rejected", !ggml_metal_bitnet_q_select(nullptr, "4", "2", &bad));
        check("legacy_4x4_rejected", !ggml_metal_bitnet_q_select(nullptr, "4", "4", &bad));

        std::unique_ptr<ggml_metal_device, decltype(&ggml_metal_device_free)> device(
            ggml_metal_device_init(0, 1), ggml_metal_device_free);
        require(device != nullptr, "Metal device unavailable");
        auto library = ggml_metal_device_get_library(device.get());
        require(library != nullptr, "Metal library unavailable");
        const auto current = output / "files/current.metallib";
        const auto copied = output / "files/copied.metallib";
        const auto alias = output / "files/./current.metallib";
        fs::copy_file(q4_source, current);
        fs::copy_file(q4_source, copied);
        const auto first = load(library, current, q4);
        check("Q4_unique_entry_load", first.pipeline && std::strcmp(ggml_metal_pipeline_name(first), q4.entry) == 0);
        check("Q4_unchanged_file_reuses_pipeline", load(library, current, q4).pipeline == first.pipeline);
        check("canonical_path_alias_reuses_pipeline", load(library, alias, q4).pipeline == first.pipeline);
        const auto duplicate = load(library, copied, q4);
        check("different_file_has_distinct_cache_identity", duplicate.pipeline && duplicate.pipeline != first.pipeline);
        check("C8_library_cannot_satisfy_cached_Q4", !load(library, c8_source, q4).pipeline);
        const auto c8_pipeline = load(library, c8_source, c8);
        check("C8_unique_entry_load", c8_pipeline.pipeline && std::strcmp(ggml_metal_pipeline_name(c8_pipeline), c8.entry) == 0);
        check("Q4_library_cannot_satisfy_cached_C8", !load(library, q4_source, c8).pipeline);
        check("missing_path_cannot_reuse_Q4", !load(library, output / "files/missing.metallib", q4).pipeline);
        fs::remove(current);
        check("deleted_cached_file_rejected", !load(library, current, q4).pipeline);
        fs::copy_file(q4_source, current);
        const auto restored = load(library, current, q4);
        check("restored_new_file_identity_gets_new_pipeline", restored.pipeline && restored.pipeline != first.pipeline);
        replace(c8_source, current);
        check("same_path_replaced_with_C8_rejects_Q4", !load(library, current, q4).pipeline);
        replace(q4_source, current);
        const auto restored_again = load(library, current, q4);
        check("same_path_restored_Q4_loads_new_identity", restored_again.pipeline && restored_again.pipeline != restored.pipeline);
        bad = q4;
        bad.entry = c8.entry;
        check("forged_entry_rejected_by_actual_getter", !load(library, current, bad).pipeline);
        bad = q4;
        bad.groups = 2;
        check("forged_groups_rejected_by_actual_getter", !load(library, current, bad).pipeline);
        result["runtime_contract_pass"] = true;
        code = 0;
    } catch (const std::exception & error) {
        result["error"] = error.what();
        code = result["runtime_contract_pass"].is_boolean() ? 1 : 2;
    }
    if (had_path) {
        setenv("GGML_METAL_BITNET_POC_METALLIB", saved_path.c_str(), 1);
    } else {
        unsetenv("GGML_METAL_BITNET_POC_METALLIB");
    }
    result["exit_code"] = code;
    try {
        fs::create_directories(output);
        std::ofstream stream(output / "RESULT.json");
        require(bool(stream << result.dump(2) << '\n'), "save contract result");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "Q_CONTRACT_RESULT_WRITE_FAILED %s\n", error.what());
        return 2;
    }
    std::printf("BITNET_Q_CONTRACT_COMPLETE exit=%d cases=%zu target_Q_dispatches=0\n", code, result["cases"].size());
    return code;
}

#ifdef BITNET_Q_CONTRACT_TEST_MAIN
int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: q-pipeline-contract Q4.metallib C8.metallib NEW_OUTPUT\n");
        return 2;
    }
    return bitnet_q_pipeline_contract_run(argv[1], argv[2], argv[3]);
}
#endif
