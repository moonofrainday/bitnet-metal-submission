// Actual GGML Metal Q dispatch controls. This program performs no timing.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "../validation/reference.h"
#include "../../ggml/src/ggml-metal/ggml-metal-bitnet-q-config.h"
#include "../../vendor/nlohmann/json.hpp"
#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cmath>
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
#include <vector>

extern char ** environ;
namespace fs = std::filesystem;
using json = nlohmann::ordered_json;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using Backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using Buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
constexpr size_t guard_bytes = 512;
constexpr size_t y_bytes = GGML_BITNET_Q_NROWS * sizeof(float);
static_assert(TQ2_ROWS == GGML_BITNET_Q_NROWS && TQ2_COLUMNS == GGML_BITNET_Q_NCOLS, "Reference shape differs");

struct Failure : std::runtime_error {
    int code;
    Failure(int value, const std::string & message) : std::runtime_error(message), code(value) {}
};

static void require(bool value, const std::string & message, int code = 2) {
    if (!value) { throw Failure(code, message); }
}

static std::string hash(const void * data, size_t size) {
    require(size <= std::numeric_limits<CC_LONG>::max(), "Hash extent is too large");
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data, static_cast<CC_LONG>(size), digest);
    const char hex[] = "0123456789abcdef";
    std::string result;
    for (const unsigned char value : digest) { result += hex[value >> 4]; result += hex[value & 15]; }
    return result;
}

static std::vector<uint8_t> read(const fs::path & path, size_t bytes) {
    require(fs::is_regular_file(path) && fs::file_size(path) == bytes, "Wrong file extent: " + path.string());
    std::vector<uint8_t> data(bytes);
    std::ifstream stream(path, std::ios::binary);
    require(bool(stream.read(reinterpret_cast<char *>(data.data()), bytes)), "Read failed: " + path.string());
    return data;
}

static std::string file_hash(const fs::path & path) {
    const size_t size = fs::file_size(path);
    require(size > 0 && size <= (64 << 20), "Metallib extent outside this small test");
    const auto data = read(path, size);
    return hash(data.data(), data.size());
}

static void save(const fs::path & path, const void * data, size_t bytes) {
    std::ofstream stream(path, std::ios::binary);
    require(bool(stream.write(static_cast<const char *>(data), bytes)), "Write failed: " + path.string());
}

static void save_json(const fs::path & path, const json & value) {
    std::ofstream stream(path);
    require(bool(stream << value.dump(2) << '\n'), "JSON write failed");
}

static void clear_routes() {
    std::vector<std::string> names;
    for (char ** p = environ; p && *p; ++p) {
        const std::string item(*p);
        const std::string name = item.substr(0, item.find('='));
        if (name.rfind("GGML_METAL_BITNET_", 0) == 0 || name.rfind("LLAMA_BITNET_", 0) == 0) { names.push_back(name); }
    }
    for (const auto & name : names) { unsetenv(name.c_str()); }
    for (const char * name : {"METAL_CAPTURE_ENABLED", "GGML_METAL_CAPTURE_COMPUTE", "GGML_METAL_GRAPH_DEBUG", "GGML_METAL_DEBUG_GRAPH"}) { unsetenv(name); }
    setenv("LLAMA_BITNET_ACTQUANT", "0", 1);
    setenv("LLAMA_BITNET_ACTQUANT_Q4", "0", 1);
    setenv("GGML_METAL_BITNET_POC_TRACE", "1", 1);
    setenv("GGML_METAL_BITNET_FFN_DECODE", "1", 1);
    setenv("GGML_METAL_BITNET_FFN_STEP", "16", 1);
    setenv("GGML_METAL_BITNET_LAYERS", "0", 1);
}

struct Case {
    const char * name;
    const char * variant;
    const char * rows;
    const char * groups;
    fs::path library;
    const char * expected_entry;
    bool enabled = true;
    bool stock_parity = true;
};

static void select(const Case & item) {
    clear_routes();
    setenv("GGML_METAL_BITNET_POC", item.enabled ? "1" : "0", 1);
    if (item.variant) { setenv("GGML_METAL_BITNET_POC_VARIANT", item.variant, 1); }
    if (item.rows) { setenv("GGML_METAL_BITNET_POC_ROWS_PER_SIMD", item.rows, 1); }
    if (item.groups) { setenv("GGML_METAL_BITNET_POC_SIMD_PER_GROUP", item.groups, 1); }
    if (!item.library.empty()) { setenv("GGML_METAL_BITNET_POC_METALLIB", item.library.c_str(), 1); }
}

static void atomic_copy(const fs::path & source, const fs::path & target) {
    const fs::path temporary = target.string() + ".new";
    require(!fs::exists(temporary), "Refuse existing replacement temporary");
    fs::copy_file(source, temporary);
    require(file_hash(source) == file_hash(temporary), "Replacement copy hash differs");
    fs::rename(temporary, target);
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "Usage: q-graph-contract W.tq2 X.f32 Q4.metallib C8.metallib NEW_OUTPUT\n");
        return 2;
    }
    const fs::path out = fs::absolute(argv[5]);
    bool output_created = false;
    json report = {{"scope", "Actual GGML Metal legacy Q guard; fixed captured post-ActQuant input, ActQuant graph OFF"},
                   {"performance_measured", false}, {"model_or_quality_acceptance", nullptr}, {"dispatch_audit_pass", nullptr},
                   {"cases", json::array()}, {"native_numerical_pass", nullptr}, {"native_exit_code", 2}};
    try {
        require(!fs::exists(out), "Refuse existing output");
        require(fs::create_directories(out), "Output directory was not freshly created");
        output_created = true;
        require(std::freopen((out / "backend-trace.log").c_str(), "w", stderr) != nullptr, "Cannot capture backend trace");
        Input input;
        input.weights = read(argv[1], GGML_BITNET_Q_WEIGHT_BYTES);
        const auto activation = read(argv[2], GGML_BITNET_Q_NCOLS * sizeof(float));
        std::memcpy(input.activation.data(), activation.data(), activation.size());
        require(std::all_of(input.activation.begin(), input.activation.end(), [](float value) { return std::isfinite(value); }), "Nonfinite X");
        for (size_t b = 0; b < input.weights.size(); b += TQ2_BLOCK_BYTES) {
            for (size_t i = 0; i < TQ2_CODES_BYTES; ++i) {
                for (int plane = 0; plane < 4; ++plane) { require(((input.weights[b + i] >> (2 * plane)) & 3) != 3, "Invalid trit code"); }
            }
        }
        const auto reference = reference_matvec(input);
        require(std::all_of(reference.begin(), reference.end(), [](double value) { return std::isfinite(value); }), "Nonfinite independent reference");
        save(out / "reference.f64", reference.data(), sizeof(reference));
        const fs::path q4 = fs::canonical(argv[3]), c8 = fs::canonical(argv[4]);
        const std::string q4_sha = file_hash(q4), c8_sha = file_hash(c8);
        require(q4_sha != c8_sha, "Q libraries must be distinct products");
        report.update({{"W_sha256", hash(input.weights.data(), input.weights.size())},
                       {"X_sha256", hash(activation.data(), activation.size())},
                       {"Q4_library_sha256", q4_sha}, {"C8_library_sha256", c8_sha},
                       {"reference_sha256", hash(reference.data(), sizeof(reference))},
                       {"frozen_tolerance", {{"absolute", 2e-4}, {"relative", 2e-5}}}});
        clear_routes();
        Backend backend(ggml_backend_metal_init(), ggml_backend_free);
        require(bool(backend), "Metal backend initialization failed");
        report["backend"] = ggml_backend_name(backend.get());
        report["device"] = ggml_backend_dev_description(ggml_backend_get_device(backend.get()));
        Context context(ggml_init({8 << 20, nullptr, true}), ggml_free);
        require(bool(context), "GGML context allocation failed");
        auto * w = ggml_new_tensor_2d(context.get(), GGML_TYPE_TQ2_0, GGML_BITNET_Q_NCOLS, GGML_BITNET_Q_NROWS);
        auto * x = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, GGML_BITNET_Q_NCOLS, 1);
        auto * before = ggml_new_tensor_1d(context.get(), GGML_TYPE_I8, guard_bytes);
        auto * y = ggml_mul_mat(context.get(), w, x);
        auto * after = ggml_new_tensor_1d(context.get(), GGML_TYPE_I8, guard_bytes);
        ggml_set_name(w, "blk.0.attn_q.weight"); ggml_set_name(x, "captured-post-ActQuant-step16-X"); ggml_set_name(y, "Qcur-0");
        ggml_set_input(x); ggml_set_output(y);
        auto * graph = ggml_new_graph_custom(context.get(), 16, false);
        ggml_build_forward_expand(graph, y);
        require(ggml_backend_dev_supports_op(ggml_backend_get_device(backend.get()), y), "Actual Metal backend does not support Q node");
        Buffer wb(ggml_backend_alloc_buffer(backend.get(), ggml_nbytes(w)), ggml_backend_buffer_free);
        Buffer xb(ggml_backend_alloc_buffer(backend.get(), ggml_nbytes(x)), ggml_backend_buffer_free);
        Buffer yb(ggml_backend_alloc_buffer(backend.get(), guard_bytes + y_bytes + guard_bytes), ggml_backend_buffer_free);
        require(wb && xb && yb, "Backend buffers unavailable");
        auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(yb.get()));
        require(ggml_backend_tensor_alloc(wb.get(), w, ggml_backend_buffer_get_base(wb.get())) == GGML_STATUS_SUCCESS &&
                ggml_backend_tensor_alloc(xb.get(), x, ggml_backend_buffer_get_base(xb.get())) == GGML_STATUS_SUCCESS &&
                ggml_backend_tensor_alloc(yb.get(), before, base) == GGML_STATUS_SUCCESS &&
                ggml_backend_tensor_alloc(yb.get(), y, base + guard_bytes) == GGML_STATUS_SUCCESS &&
                ggml_backend_tensor_alloc(yb.get(), after, base + guard_bytes + y_bytes) == GGML_STATUS_SUCCESS, "Tensor allocation failed");
        require(!y->view_src && !x->view_src && !w->view_src && ggml_nbytes(y) == y_bytes, "Unexpected graph layout");
        ggml_backend_tensor_set(w, input.weights.data(), 0, input.weights.size());
        ggml_backend_tensor_set(x, input.activation.data(), 0, activation.size());
        std::array<uint8_t, guard_bytes> guards{}; guards.fill(0xa5);
        std::array<float, GGML_BITNET_Q_NROWS> baseline{};
        bool all_pass = true;
        const fs::path mutable_library = out / "mutable.metallib";
        const char * q4_entry = GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_ENTRY);
        const char * c8_entry = GGML_BITNET_Q_STRINGIFY(GGML_BITNET_C8_ENTRY);
        std::vector<Case> cases = {
            {"stock", nullptr, nullptr, nullptr, q4, nullptr, false},
            {"Q4-valid", "stock4x1", "4", "1", q4, q4_entry},
            {"Q4-wrong-4x2", "stock4x1", "4", "2", q4, nullptr},
            {"Q4-wrong-4x4", "stock4x1", "4", "4", q4, nullptr},
            {"legacy-wrong-4x2", nullptr, "4", "2", q4, nullptr},
            {"legacy-wrong-4x4", nullptr, "4", "4", q4, nullptr},
            {"C8-with-Q4-geometry", "c8x2", "4", "1", c8, nullptr},
            {"Q4-descriptor-C8-library", "stock4x1", "4", "1", c8, nullptr},
            {"C8-valid", "c8x2", "8", "2", c8, c8_entry, true, false},
            {"C8-descriptor-Q4-library", "c8x2", "8", "2", q4, nullptr},
            {"same-path-Q4-initial", "stock4x1", "4", "1", mutable_library, q4_entry},
            {"same-path-replace-C8", "stock4x1", "4", "1", mutable_library, nullptr},
            {"same-path-restore-Q4", "stock4x1", "4", "1", mutable_library, q4_entry},
            {"same-path-missing-after-valid", "stock4x1", "4", "1", mutable_library, nullptr},
            {"same-path-restore-after-missing", "stock4x1", "4", "1", mutable_library, q4_entry},
        };
        for (const auto & item : cases) {
            const std::string name(item.name);
            if (name == "same-path-missing-after-valid") { require(fs::remove(mutable_library), "Missing control could not remove local copy"); }
            else if (name == "same-path-replace-C8") { atomic_copy(c8, mutable_library); }
            else if (name.rfind("same-path-", 0) == 0) { atomic_copy(q4, mutable_library); }
            select(item);
            const fs::path folder = out / name; fs::create_directory(folder);
            json entry = {{"name", name}, {"expected_custom_dispatches", item.expected_entry ? 1 : 0},
                          {"expected_entry", item.expected_entry ? json(item.expected_entry) : json(nullptr)},
                          {"stock_bitwise_required", item.stock_parity}, {"variant", item.variant ? item.variant : "legacy"},
                          {"rows", item.rows ? item.rows : "unset"}, {"groups", item.groups ? item.groups : "unset"},
                          {"library_exists", fs::is_regular_file(item.library)},
                          {"library_sha256", fs::is_regular_file(item.library) ? json(file_hash(item.library)) : json(nullptr)}};
            std::fprintf(stderr, "Q_GRAPH_CASE_BEGIN %s\n", entry.dump().c_str()); std::fflush(stderr);
            std::array<float, GGML_BITNET_Q_NROWS> output{}; output.fill(std::numeric_limits<float>::quiet_NaN());
            ggml_backend_tensor_set(before, guards.data(), 0, guards.size()); ggml_backend_tensor_set(after, guards.data(), 0, guards.size());
            ggml_backend_tensor_set(y, output.data(), 0, y_bytes);
            const auto status = ggml_backend_graph_compute(backend.get(), graph);
            ggml_backend_synchronize(backend.get());
            ggml_backend_tensor_get(y, output.data(), 0, y_bytes);
            std::array<uint8_t, guard_bytes> pre{}, post{};
            ggml_backend_tensor_get(before, pre.data(), 0, pre.size()); ggml_backend_tensor_get(after, post.data(), 0, post.size());
            save(folder / "canary-before.bin", pre.data(), pre.size()); save(folder / "canary-after.bin", post.data(), post.size());
            std::vector<uint8_t> w_after(input.weights.size()); std::array<float, GGML_BITNET_Q_NCOLS> x_after{};
            ggml_backend_tensor_get(w, w_after.data(), 0, w_after.size()); ggml_backend_tensor_get(x, x_after.data(), 0, activation.size());
            save(folder / "y.f32", output.data(), y_bytes);
            if (name == "stock") { baseline = output; }
            size_t failures = 0, nonfinite = 0, different = 0; double maximum = 0, ratio = 0, square = 0, reference_square = 0;
            std::ofstream rows(folder / "rows.csv"); rows << "row,reference_f64,output_f32,stock_f32,abs_error,tolerance,fp64_fail,stock_bitwise_equal\n" << std::setprecision(17);
            for (size_t i = 0; i < output.size(); ++i) {
                const double error = std::abs(double(output[i]) - reference[i]), bound = 2e-4 + 2e-5 * std::abs(reference[i]);
                const bool finite = std::isfinite(output[i]), same = std::memcmp(&output[i], &baseline[i], sizeof(float)) == 0;
                nonfinite += !finite; failures += !finite || error > bound; different += !same;
                if (finite) { maximum = std::max(maximum, error); ratio = std::max(ratio, error / bound); square += error * error; }
                reference_square += reference[i] * reference[i];
                rows << i << ',' << reference[i] << ',' << output[i] << ',' << baseline[i] << ',' << error << ',' << bound << ',' << (!finite || error > bound) << ',' << same << '\n';
            }
            const bool immutable = w_after == input.weights && std::memcmp(x_after.data(), input.activation.data(), activation.size()) == 0;
            const bool canaries = pre == guards && post == guards;
            const bool passed = status == GGML_STATUS_SUCCESS && immutable && canaries && failures == 0 && (!item.stock_parity || different == 0);
            entry.update({{"synchronized", true}, {"graph_status", int(status)}, {"values", output.size()}, {"FP64_failures", failures},
                          {"nonfinite", nonfinite}, {"stock_bitwise_differences", different}, {"max_abs_error", maximum},
                          {"max_error_over_tolerance", ratio}, {"relative_L2", reference_square > 0 ? json(std::sqrt(square / reference_square)) : json(nullptr)},
                          {"canaries_unchanged", canaries}, {"inputs_unchanged", immutable}, {"output_sha256", hash(output.data(), y_bytes)},
                          {"W_after_sha256", hash(w_after.data(), w_after.size())}, {"X_after_sha256", hash(x_after.data(), activation.size())},
                          {"native_numerical_pass", passed}, {"dispatch_audit_pass", nullptr}});
            save_json(folder / "RESULT.json", entry); report["cases"].push_back(entry);
            all_pass = all_pass && passed; save_json(out / "RUN.json", report);
            std::fprintf(stderr, "Q_GRAPH_CASE_END %s\n", entry.dump().c_str()); std::fflush(stderr);
            require(canaries && immutable, "Canary or input mutation; stop before further cases", 1);
            require(status == GGML_STATUS_SUCCESS, "Graph execution failed; stop before further cases", 1);
        }
        report["native_numerical_pass"] = all_pass; report["native_exit_code"] = all_pass ? 0 : 1;
        report["graph_and_buffers_reused"] = true; report["same_calling_thread"] = true;
        report["guard_bytes_before_and_after_Y"] = guard_bytes;
        report["runtime_dispatch_verification"] = "Run analyze_q_graph_contract.py; native exit0 alone does not prove the requested path.";
        save_json(out / "RUN.json", report);
        std::printf("Q graph collection complete; numerical_exit=%d; dispatch audit required; output=%s\n", all_pass ? 0 : 1, out.c_str());
        return all_pass ? 0 : 1;
    } catch (const std::exception & error) {
        const auto * failure = dynamic_cast<const Failure *>(&error);
        const int code = failure ? failure->code : 2;
        report["error"] = error.what(); report["native_exit_code"] = code;
        if (code == 1) { report["native_numerical_pass"] = false; }
        if (output_created) { save_json(out / "RUN.json", report); }
        std::fprintf(stderr, "Q_GRAPH_CONTRACT_ERROR %s\n", error.what());
        return code;
    }
}
