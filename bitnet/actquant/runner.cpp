#include "runner.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../../ggml/src/ggml-metal/ggml-metal-bitnet-q-config.h"
#include "../../vendor/nlohmann/json.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <time.h>
#include <vector>
#if defined(__OBJC__)
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#endif

namespace bitnet_actquant {
namespace fs = std::filesystem;
using json = nlohmann::ordered_json;
static constexpr const char * MODEL_SHA = "52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2";
static void require(bool value, const std::string & message) {
    if (!value) { throw std::runtime_error(message); }
}
static json read_json(const fs::path & path) {
    std::ifstream file(path); require(bool(file), "missing JSON: " + path.string());
    json result; file >> result; return result;
}
static void write_json(const fs::path & path, const json & value) {
    fs::create_directories(path.parent_path()); std::ofstream file(path);
    require(bool(file), "cannot create JSON"); file << value.dump(2) << '\n'; require(bool(file), "JSON write failed");
}
static std::string hash_file(const fs::path & path) {
    std::ifstream file(path, std::ios::binary); require(bool(file), "cannot hash " + path.string());
    CC_SHA256_CTX state; CC_SHA256_Init(&state); std::vector<char> bytes(1 << 20);
    while (file) { file.read(bytes.data(), bytes.size()); if (file.gcount()) { CC_SHA256_Update(&state, bytes.data(), (CC_LONG) file.gcount()); } }
    require(file.eof(), "hash read failed"); unsigned char digest[CC_SHA256_DIGEST_LENGTH]; CC_SHA256_Final(digest, &state);
    static const char hex[] = "0123456789abcdef"; std::string result;
    for (unsigned char value : digest) { result += hex[value >> 4]; result += hex[value & 15]; }
    return result;
}
static fs::path relative_path(const fs::path & root, const std::string & name) {
    fs::path path(name); require(!path.empty() && !path.is_absolute(), "expected relative path");
    for (const auto & part : path) { require(part != ".." && part != ".", "unsafe relative path"); }
    return root / path;
}
static std::vector<float> read_f32(const fs::path & path, size_t count) {
    require(count <= (1u << 26) && fs::file_size(path) == count * sizeof(float), "wrong F32 input size");
    std::vector<float> result(count); std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char *>(result.data()), count * sizeof(float)); require(bool(file), "F32 read failed");
    require(std::all_of(result.begin(), result.end(), [](float x) { return std::isfinite(x); }), "nonfinite input");
    return result;
}
static void write_f32(const fs::path & path, const std::vector<float> & values) {
    fs::create_directories(path.parent_path()); std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float)); require(bool(file), "F32 write failed");
}
static json tensor_metadata(const ggml_tensor * tensor) {
    json result = {{"name", tensor->name}, {"op", ggml_op_name(tensor->op)}, {"type", ggml_type_name(tensor->type)}, {"ne", json::array()}, {"nb", json::array()}, {"bytes", ggml_nbytes(tensor)}};
    for (int i = 0; i < 4; ++i) { result["ne"].push_back(tensor->ne[i]); result["nb"].push_back(tensor->nb[i]); }
    if (tensor->buffer) {
        result["buffer_name"] = ggml_backend_buffer_name(tensor->buffer);
        result["ggml_buffer_offset"] = uintptr_t(tensor->data) - uintptr_t(ggml_backend_buffer_get_base(tensor->buffer));
    }
    return result;
}
static std::vector<float> read_tensor(const ggml_tensor * tensor) {
    require(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor), "capture requires contiguous F32");
    const int64_t count = ggml_nelements(tensor); require(count > 0 && count <= (1 << 26), "capture size out of bounds");
    std::vector<float> result(size_t(count), 0); ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
    require(std::all_of(result.begin(), result.end(), [](float x) { return std::isfinite(x); }), "nonfinite captured tensor"); return result;
}
static void clear_custom_routes() {
    for (const char * key : {"LLAMA_BITNET_ACTQUANT_Q4", "GGML_METAL_BITNET_POC", "GGML_METAL_BITNET_POC_VARIANT", "GGML_METAL_BITNET_Q_GUARD_ONLY", "GGML_METAL_BITNET_Q_ATTRIBUTION", "GGML_METAL_BITNET_POC_ROWS_PER_SIMD", "GGML_METAL_BITNET_POC_SIMD_PER_GROUP", "GGML_METAL_BITNET_POC_METALLIB", "GGML_METAL_BITNET_LAYERS", "GGML_METAL_BITNET_FFN", "GGML_METAL_BITNET_FFN_DECODE", "GGML_METAL_BITNET_FFN_STEP", "GGML_METAL_BITNET_FFN_MECHANISM", "GGML_METAL_BITNET_FFN_ABLATION", "GGML_METAL_BITNET_FFN_SHADOW_DIR", "GGML_METAL_BITNET_FFN_K_SHADOW_DIR", "GGML_METAL_BITNET_QREADBACK_DIR", "GGML_METAL_BITNET_POC_TRACE", "GGML_METAL_BITNET_FFN_TRACE", "GGML_METAL_GRAPH_DEBUG", "GGML_METAL_DEBUG_GRAPH", "GGML_METAL_CAPTURE_COMPUTE", "METAL_CAPTURE_ENABLED"}) { unsetenv(key); }
}
static int operator_test(const json & config, const fs::path & out) {
    const fs::path fixture = config.at("fixture").get<std::string>(); const auto manifest = read_json(fixture / "FIXTURES.json");
    const std::string backend_name = config.value("backend", "CPU");
    require(backend_name == "CPU" || backend_name == "MTL0", "backend must be CPU or MTL0"); ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name(backend_name.c_str()); require(device != nullptr, "backend missing");
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    require(bool(backend), "backend initialization failed"); if (backend_name == "CPU") { ggml_backend_cpu_set_n_threads(backend.get(), 6); }
    json runtime = {{"schema", 1}, {"operation", "operator"}, {"fixture_sha256", hash_file(fixture / "FIXTURES.json")}, {"backend", backend_name}, {"device", ggml_backend_dev_description(device)}, {"cases", json::array()}, {"performance_measurement", false}};
    for (const auto & item : manifest.at("cases")) {
        const std::string name = item.at("name"); require(name.find('/') == std::string::npos && name.find("..") == std::string::npos, "bad case name");
        const int64_t width = item.at("width"), rows = item.at("rows"); require(width > 0 && width <= 16384 && rows > 0 && rows <= 128, "operator dimensions invalid");
        const auto input_path = relative_path(fixture, item.at("input")); require(hash_file(input_path) == item.at("input_sha256"), "input hash mismatch");
        auto input = read_f32(input_path, size_t(width * rows)); auto expected = read_f32(relative_path(fixture, item.at("expected")), input.size());
        ggml_init_params params = {size_t(8 << 20), nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free); require(bool(ctx), "ggml init failed");
        auto * before = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
        auto * y = ggml_act_quant(ctx.get(), x, manifest.at("epsilon").get<float>());
        auto * after = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
        ggml_set_name(x, "actquant-fixture-input"); ggml_set_name(y, name.c_str()); ggml_set_input(x); ggml_set_output(y);
        auto * graph = ggml_new_graph_custom(ctx.get(), 32, false); ggml_build_forward_expand(graph, y);
        require(ggml_backend_dev_supports_op(device, y), "selected backend does not support ActQuant");
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
        require(bool(buffer), "backend allocation failed");
        std::vector<float> guard(32, 0.8125f), guard_before(32), guard_after(32), output(input.size(), std::numeric_limits<float>::quiet_NaN());
        ggml_backend_tensor_set(before, guard.data(), 0, guard.size() * sizeof(float)); ggml_backend_tensor_set(after, guard.data(), 0, guard.size() * sizeof(float));
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float)); ggml_backend_tensor_set(y, output.data(), 0, output.size() * sizeof(float));
        require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "ActQuant graph compute failed"); ggml_backend_synchronize(backend.get());
        ggml_backend_tensor_get(y, output.data(), 0, output.size() * sizeof(float));
        auto input_after = read_tensor(x); ggml_backend_tensor_get(before, guard_before.data(), 0, guard.size() * sizeof(float)); ggml_backend_tensor_get(after, guard_after.data(), 0, guard.size() * sizeof(float));
        require(std::memcmp(input.data(), input_after.data(), input.size() * sizeof(float)) == 0 && guard == guard_before && guard == guard_after, "input/canary changed");
        size_t failures = 0, bit_differences = 0; double max_abs = 0;
        for (size_t i = 0; i < output.size(); ++i) {
            require(std::isfinite(output[i]), "unwritten/nonfinite operator output"); double error = std::abs(double(output[i]) - expected[i]); max_abs = std::max(max_abs, error);
            failures += error > 2e-4 + 2e-5 * std::abs(double(expected[i])); bit_differences += std::memcmp(&output[i], &expected[i], sizeof(float)) != 0;
        }
        const std::string output_file = name + "/y.f32"; write_f32(out / output_file, output);
        auto entry = tensor_metadata(y); entry.update(json{{"name", name}, {"input_sha256", hash_file(input_path)}, {"output_sha256", hash_file(out / output_file)}, {"path", output_file}, {"completed", true}, {"finite", true}, {"input_unchanged", true}, {"canaries_unchanged", true}, {"native_value_tolerance_failures", failures}, {"native_value_bit_differences", bit_differences}, {"max_abs", max_abs}});
        runtime["cases"].push_back(entry); write_json(out / "RUN.json", runtime);
        std::printf("ACTQUANT_OPERATOR_COMPLETE case=%s backend=%s values=%zu synchronized=1 value_failures=%zu\n", name.c_str(), backend_name.c_str(), output.size(), failures); std::fflush(stdout);
    }
    runtime["execution_pass"] = true; runtime["semantic_acceptance_pass"] = nullptr; runtime["strict_model_exit_code"] = 2;
    write_json(out / "RUN.json", runtime); return 0;
}
struct Capture {
    fs::path out; std::vector<int> positions; json checkpoints = json::array();
    std::string q_weight_sha; const ggml_tensor * q_weight = nullptr;
    std::string scope = "all";
    size_t actquant_callbacks = 0, files = 0; bool enabled = false; std::string error;
    bool wanted(ggml_tensor * t) const {
        if (scope == "q4-layer0") {
            return std::string(t->name) == "actq_qkv_out-0" || std::string(t->name) == "l_out-0" ||
                (t->op == GGML_OP_MUL_MAT && std::string(t->name) == "Qcur-0");
        }
        if (t->op == GGML_OP_ACT_QUANT) { return true; }
        const std::string name = t->name;
        return name.rfind("l_out-", 0) == 0 || (t->op == GGML_OP_MUL_MAT && name.rfind("Qcur-", 0) == 0);
    }
    static bool callback(ggml_tensor * tensor, bool ask, void * opaque) {
        auto & self = *static_cast<Capture *>(opaque);
        if (!self.enabled || self.positions.empty() || !self.wanted(tensor)) { return ask ? false : true; }
        if (ask) { return true; }
        try { self.save(tensor); return true; } catch (const std::exception & e) { self.error = e.what(); return false; }
    }
    void save(ggml_tensor * tensor) {
        std::string name = tensor->name; const auto hyphen = name.rfind('-'); require(hyphen != std::string::npos, "captured node has no layer suffix");
        const int layer = std::stoi(name.substr(hyphen + 1)); require(layer >= 0 && layer < 30, "bad capture layer");
        for (char & ch : name) { if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' && ch != '_') { ch = '_'; } }
        auto meta = tensor_metadata(tensor); const int64_t count_rows = ggml_nelements(tensor) / tensor->ne[0]; require(count_rows == int64_t(positions.size()) || count_rows == 1, "unknown captured token row mapping");
        std::vector<int> rows = count_rows == 1 ? std::vector<int>{positions.back()} : positions;
        const std::string base = "captures/" + std::to_string(files++) + "-" + name;
        meta.update(json{{"node", tensor->name}, {"layer", layer}, {"input_positions", rows}, {"input_position", rows.back()}});
        auto output = read_tensor(tensor);
        if (tensor->op == GGML_OP_ACT_QUANT) {
            ++actquant_callbacks; require(tensor->src[0] != nullptr, "ActQuant input missing"); auto input = read_tensor(tensor->src[0]);
            write_f32(out / (base + "-input.f32"), input); write_f32(out / (base + "-output.f32"), output);
            auto source = tensor_metadata(tensor->src[0]); meta.update(json{{"kind", "actquant"}, {"input_path", base + "-input.f32"}, {"output_path", base + "-output.f32"}, {"src_ne", source["ne"]}, {"src_nb", source["nb"]}, {"src_buffer_name", source.value("buffer_name", "unknown")}});
        } else {
            write_f32(out / (base + ".f32"), output); meta["path"] = base + ".f32";
            if (tensor->op == GGML_OP_MUL_MAT) {
                auto input = read_tensor(tensor->src[1]); write_f32(out / (base + "-input.f32"), input); meta.update(json{{"kind", "Q_preRoPE"}, {"input_path", base + "-input.f32"}});
                if (layer == 0) {
                    const auto * weight = tensor->src[0]; require(weight && weight->type == GGML_TYPE_TQ2_0 &&
                        ggml_is_contiguous(weight) && weight->ne[0] == 2560 && weight->ne[1] == 2560 &&
                        weight->ne[2] == 1 && weight->ne[3] == 1 && ggml_nbytes(weight) == 1689600 &&
                        std::string(weight->name) == "blk.0.attn_q.weight", "wrong live Q0 packed-weight contract");
                    const std::string relative = "weights/blk.0.attn_q.weight.tq2";
                    if (!q_weight) {
                        q_weight = weight; std::vector<char> bytes(ggml_nbytes(weight));
                        ggml_backend_tensor_get(weight, bytes.data(), 0, bytes.size()); fs::create_directories((out / relative).parent_path());
                        std::ofstream file(out / relative, std::ios::binary); file.write(bytes.data(), bytes.size());
                        require(bool(file), "Q weight capture failed"); file.close(); q_weight_sha = hash_file(out / relative);
                    }
                    require(q_weight == weight, "Q0 weight tensor identity changed during replay");
                    meta.update(json{{"weight_path", relative}, {"weight_sha256", q_weight_sha}, {"weight", tensor_metadata(weight)},
                        {"activation", tensor_metadata(tensor->src[1])}, {"phase", positions.front() < 27 ? "prefill" : "decode"},
                        {"decode_step", positions.front() < 27 ? 0 : positions.front() - 26}});
                    meta["input_sha256"] = hash_file(out / (base + "-input.f32")); meta["output_sha256"] = hash_file(out / (base + ".f32"));
                }
            } else { meta["kind"] = "l_out"; }
        }
        checkpoints.push_back(meta);
    }
};
struct ModelLog { int offloaded = -1, total = -1; };
static void model_logger(ggml_log_level, const char * text, void * opaque) {
    auto & log = *static_cast<ModelLog *>(opaque); std::fputs(text, stderr); std::fflush(stderr); const char * p = std::strstr(text, "offloaded ");
    if (p) { int a, b; if (std::sscanf(p, "offloaded %d/%d layers to GPU", &a, &b) == 2) { log.offloaded = a; log.total = b; } }
}
static int model_test(const json & config, const fs::path & out) {
    const fs::path model = config.at("model").get<std::string>(), token_file = config.at("tokens").get<std::string>();
    require(hash_file(model) == MODEL_SHA, "wrong public model SHA256"); const auto tokens = read_json(token_file);
    const bool enabled = config.value("actquant", false), capture_enabled = config.value("capture", false);
    const std::string q_route = config.value("q_route", "stock"), q_fallback = config.value("q_fallback", "none");
    const std::string capture_scope = config.value("capture_scope", "all");
    const std::string workload = config.value("workload", "teacher"), backend_name = config.value("backend", "MTL0"), prefill = config.value("prefill", "batched"), kv = config.value("kv", "F16");
    require(workload == "teacher" || workload == "corpus", "bad workload"); require(backend_name == "CPU" || backend_name == "MTL0", "bad backend");
    require(prefill == "batched" || prefill == "sequential", "bad prefill mode"); require(kv == "F16" || kv == "F32", "bad KV dtype");
    require(q_route == "stock" || q_route == "Q4" || q_route == "QB", "bad q_route");
    require(capture_scope == "all" || capture_scope == "q4-layer0", "bad capture_scope");
    require(q_fallback == "none" || q_fallback == "missing-library", "bad q_fallback (invalid geometry is a rejected graph contract, not an accepted route)");
    require(q_fallback == "none" || q_route == "Q4", "Q fallback requires Q4 route");
    require(q_route == "stock" || (enabled && backend_name == "MTL0" && workload == "teacher" && prefill == "batched" && kv == "F16"), "Q4 requires ActQuant ON, MTL0 teacher/batched/F16");
    clear_custom_routes(); setenv("LLAMA_BITNET_ACTQUANT", enabled ? "1" : "0", 1);
    setenv("GGML_METAL_ACTQUANT_TRACE", config.value("trace", false) ? "1" : "0", 1);
    std::string q_library_sha;
    if (config.contains("q_library")) {
        const fs::path library = config.at("q_library").get<std::string>();
        require(library.is_absolute() && fs::is_regular_file(library), "Q library must be an existing absolute file");
        q_library_sha = hash_file(library); require(q_library_sha == config.at("q_library_sha256").get<std::string>(), "Q library SHA256 mismatch");
        if (q_route == "Q4" || q_route == "QB") {
            const auto selected = q_fallback == "missing-library" ? out / "intentionally-missing.metallib" : library;
            require(q_fallback == "none" || !fs::exists(selected), "missing-library control path exists");
            setenv("GGML_METAL_BITNET_POC_METALLIB", selected.c_str(), 1);
        }
    }
    require(q_route == "stock" || !q_library_sha.empty(), "Q4 requires a SHA-verified library (also for intentional missing-library control)");
    if (q_route == "Q4" || q_route == "QB") {
        require(config.value("q_variant", "") == "stock4x1" && config.value("q_library_source_sha256", "").size() == 64 &&
            config.value("q_library_manifest_sha256", "").size() == 64, "Q4 requires stock4x1 source/build provenance from q4_contract.py");
        require(config.value("q_kernel_contract_version", 0) == GGML_BITNET_Q_CONFIG_VERSION &&
            config.value("q_entry", "") == GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_ENTRY) &&
            config.value("q_kernel_config_sha256", "").size() == 64 &&
            config.at("q_grid") == json::array({GGML_BITNET_Q_NROWS/(GGML_BITNET_Q4_ROWS*GGML_BITNET_Q4_NSG),1,1}) &&
            config.at("q_threads") == json::array({GGML_BITNET_Q_SIMD_WIDTH,GGML_BITNET_Q4_NSG,1}),
            "Q4 requires the current canonical entry/version/geometry; old shared-entry products are not runtime aliases");
    }
    if (q_route == "Q4" || q_route == "QB") {
        setenv("LLAMA_BITNET_ACTQUANT_Q4", "1", 1);
        setenv(q_route == "QB" ? "GGML_METAL_BITNET_Q_GUARD_ONLY" : "GGML_METAL_BITNET_POC", "1", 1);
        setenv("GGML_METAL_BITNET_POC_VARIANT", "stock4x1", 1);
        setenv("GGML_METAL_BITNET_POC_ROWS_PER_SIMD", GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_ROWS), 1);
        setenv("GGML_METAL_BITNET_POC_SIMD_PER_GROUP", GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_NSG), 1);
        setenv("GGML_METAL_BITNET_LAYERS", "0", 1);
    }
    if (config.value("trace", false)) { setenv("GGML_METAL_BITNET_POC_TRACE", "1", 1); }
    ModelLog log; llama_log_set(model_logger, &log); llama_backend_init();
    struct BackendGuard { ~BackendGuard() { llama_backend_free(); llama_log_set(nullptr, nullptr); } } backend_guard;
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name(backend_name.c_str()), nullptr}; require(devices[0] != nullptr, "model backend missing");
    auto mp = llama_model_default_params(); mp.devices = devices; mp.n_gpu_layers = backend_name == "MTL0" ? 999 : 0; mp.check_tensors = true;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> m(llama_model_load_from_file(model.c_str(), mp), llama_model_free); require(bool(m), "model load failed");
    require(llama_model_n_layer(m.get()) == 30 && llama_vocab_n_tokens(llama_model_get_vocab(m.get())) == 128256, "wrong public model geometry");
    if (backend_name == "MTL0") { require(log.offloaded == 31 && log.total == 31, "actual31/31 offload not confirmed"); }
    Capture capture; capture.out = out; capture.enabled = capture_enabled; capture.scope = capture_scope;
    auto cp = llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = cp.n_ubatch = 128; cp.n_threads = cp.n_threads_batch = 6;
    cp.type_k = cp.type_v = kv == "F16" ? GGML_TYPE_F16 : GGML_TYPE_F32; cp.offload_kqv = backend_name == "MTL0"; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = capture_enabled ? Capture::callback : nullptr; cp.cb_eval_user_data = capture_enabled ? &capture : nullptr;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(m.get(), cp), llama_free); require(bool(ctx), "context init failed");
    llama_memory_clear(llama_get_memory(ctx.get()), true); llama_synchronize(ctx.get());
    require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == -1, "KV not empty");
    const std::vector<llama_token> prompt = tokens.at("prompt").get<std::vector<llama_token>>(), continuation = tokens.at("continuation").get<std::vector<llama_token>>();
    std::vector<llama_token> ids = workload == "corpus" ? tokens.at("corpus").get<std::vector<llama_token>>() : prompt;
    if (workload == "teacher") { ids.insert(ids.end(), continuation.begin(), continuation.end()); require(prompt.size() == 27 && continuation.size() == 32, "teacher token counts mismatch"); }
    require(ids.size() == (workload == "corpus" ? 129 : 59) && ids.front() == 128000, "input token counts/BOS mismatch");
    for (auto token : ids) { require(token >= 0 && token < 128256, "token outside vocabulary"); }
    json record = {{"schema", 1}, {"operation", "model"}, {"workload", workload}, {"backend", backend_name}, {"device", ggml_backend_dev_description(devices[0])}, {"activation_quantization", enabled}, {"model_sha256", MODEL_SHA}, {"tokens_sha256", hash_file(token_file)}, {"prompt_ids", prompt}, {"continuation_ids", continuation}, {"token_ids", ids}, {"prefill", prefill}, {"kv", kv}, {"capture_enabled", capture_enabled}, {"offloaded_layers", log.offloaded}, {"total_offload_layers", log.total}, {"logits", json::array()}, {"performance_measurement", false}, {"CPU_extra_Q8_K_packing_source_expected", backend_name == "CPU"}, {"settings", {{"n_ctx", 512}, {"batch", 128}, {"ubatch", 128}, {"threads", 6}, {"flash_attention", false}}}};
    record.update(json{{"q_route", q_route}, {"q_fallback", q_fallback}, {"q_library_sha256", q_library_sha}, {"q_target", "Qcur-0"}, {"capture_scope", capture_scope},
        {"trace_enabled", config.value("trace", false)}, {"q_rows_per_simd", q_route != "stock" ? GGML_BITNET_Q4_ROWS : 0}, {"q_simd_per_group", q_route != "stock" ? GGML_BITNET_Q4_NSG : 0},
        {"q_kernel_contract_version", GGML_BITNET_Q_CONFIG_VERSION}, {"q_entry", GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_ENTRY)},
        {"official_source_model_acceptance", nullptr}, {"scoped_stock_compatibility_pass", nullptr}});
    for (const char * key : {"q_variant", "q_library_source_sha256", "q_library_manifest_sha256", "q_library_sdk", "q_kernel_config_sha256", "q_grid", "q_threads"}) {
        if (config.contains(key)) { record[key] = config.at(key); }
    }
    int decode_calls = 0; long double nll_total = 0; size_t scored = 0;
    const size_t limit = workload == "corpus" ? 128 : ids.size();
    for (size_t position = 0; position < limit;) {
        const size_t count = workload == "teacher" && prefill == "batched" && position == 0 ? prompt.size() : 1;
        const int q_step = workload == "teacher" && position >= prompt.size() ? int(position - prompt.size() + 1) : 0;
        const std::string q_step_text = std::to_string(q_step);
        if (q_step > 0) { setenv("GGML_METAL_BITNET_FFN_DECODE", "1", 1); } else { unsetenv("GGML_METAL_BITNET_FFN_DECODE"); }
        setenv("GGML_METAL_BITNET_FFN_STEP", q_step_text.c_str(), 1);
        if (config.value("trace", false)) { std::fprintf(stderr, "ACTQUANT_Q_PHASE_BEGIN q_route=%s phase=%s decode_step=%d input_position=%zu count=%zu\n", q_route.c_str(), q_step ? "decode" : "prefill", q_step, position, count); }
        capture.positions.clear(); for (size_t i = 0; i < count; ++i) { capture.positions.push_back(int(position + i)); }
        require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == int(position) - 1, "KV position before decode differs");
        int code = llama_decode(ctx.get(), llama_batch_get_one(ids.data() + position, int(count))); require(code == 0 && capture.error.empty(), "model decode/capture failed: " + capture.error); llama_synchronize(ctx.get()); ++decode_calls;
        if (config.value("trace", false)) { std::fprintf(stderr, "ACTQUANT_Q_PHASE_COMPLETE q_route=%s decode_step=%d synchronized=1\n", q_route.c_str(), q_step); }
        const int at = int(position + count - 1); require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == at, "KV position after decode differs");
        if (workload == "corpus" || at >= 26) {
            const float * y = llama_get_logits_ith(ctx.get(), -1); require(y != nullptr, "missing full logits"); std::vector<float> values(y, y + 128256);
            require(std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }), "nonfinite logits");
            const std::string path = "logits/position-" + std::to_string(at) + ".f32"; write_f32(out / path, values);
            json row = {{"input_position", at}, {"path", path}, {"vocab", 128256}, {"greedy_token", int(std::max_element(values.begin(), values.end()) - values.begin())}};
            if (at + 1 < int(ids.size())) {
                row["target_token"] = ids[at + 1]; const double maximum = *std::max_element(values.begin(), values.end()); double total = 0;
                for (float value : values) { total += std::exp(double(value) - maximum); }
                const double nll = maximum + std::log(total) - values[ids[at + 1]]; row["NLL"] = nll; nll_total += nll; ++scored;
            } else { row["target_token"] = nullptr; }
            record["logits"].push_back(row);
        }
        position += count;
    }
    record["checkpoints"] = capture.checkpoints; record["completed_decode_calls"] = decode_calls; record["actquant_callbacks_count"] = capture.actquant_callbacks;
    record["execution_pass"] = true; record["finite_logits"] = true; record["parity_pass"] = nullptr; record["fp64_pass"] = nullptr; record["model_output_pass"] = nullptr;
    record["scored_tokens"] = scored; if (scored) { record["mean_NLL"] = double(nll_total / scored); record["PPL"] = std::exp(double(nll_total / scored)); }
    record["strict_model_exit_code"] = 2; write_json(out / "RUN.json", record);
    std::printf("ACTQUANT_Q_ROUTE_COMPLETE q_route=%s q_fallback=%s eligible_decode_calls=%d synchronized=1 numerical_acceptance_unverified=1\n", q_route.c_str(), q_fallback.c_str(), workload == "teacher" ? 32 : 0);
    std::printf("ACTQUANT_MODEL_COMPLETE enabled=%d backend=%s workload=%s decode_calls=%d logits=%zu actquant_callbacks=%zu synchronized=1\n", enabled, backend_name.c_str(), workload.c_str(), decode_calls, record["logits"].size(), capture.actquant_callbacks); std::fflush(stdout);
    return 0;
}
#include "attribution.h"
static int run(const json & config, const fs::path & out) {
    require(!fs::exists(out), "refuse existing output directory"); fs::create_directories(out); write_json(out / "REQUEST.json", config);
    clear_custom_routes(); setenv("GGML_METAL_ACTQUANT_TRACE", config.value("trace", false) ? "1" : "0", 1); const std::string operation = config.at("operation");
    return operation == "operator" ? operator_test(config, out) : operation == "model" ? model_test(config, out) : operation == "attribution" ? attribution_test(config, out) : throw std::runtime_error("unknown operation");
}
}
extern "C" int bitnet_actquant_run_config(const char * config_json, const char * output_directory) {
    try { return bitnet_actquant::run(bitnet_actquant::json::parse(config_json), output_directory); }
    catch (const std::exception & e) { std::fprintf(stderr, "ACTQUANT_ERROR %s\n", e.what()); std::fflush(stderr); return 2; }
}
#ifndef BITNET_ACTQUANT_NO_MAIN
int main(int argc, char ** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: bitnet-actquant-runner CONFIG.json NEW_OUTPUT\n"); return 2; }
    try { const auto config = bitnet_actquant::read_json(argv[1]).dump(); return bitnet_actquant_run_config(config.c_str(), argv[2]); }
    catch (const std::exception & e) { std::fprintf(stderr, "ACTQUANT_ERROR %s\n", e.what()); return 2; }
}
#endif
