#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <mach-o/dyld.h>

#include "reference.h"
#include "stock-runtime.h"
#include "ggml-metal-bitnet-q-config.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace {
constexpr size_t weight_bytes = size_t(TQ2_ROWS) * TQ2_BLOCKS_PER_ROW * TQ2_BLOCK_BYTES;
constexpr size_t vector_bytes = size_t(TQ2_ROWS) * sizeof(float);
static_assert(TQ2_ROWS == GGML_BITNET_Q_NROWS && TQ2_COLUMNS == GGML_BITNET_Q_NCOLS &&
              TQ2_SIMD_WIDTH == GGML_BITNET_Q_SIMD_WIDTH, "Standalone Q shape differs from canonical contract");
constexpr size_t guard_bytes = 128;
constexpr double abs_tolerance = 2e-4;
constexpr double rel_tolerance = 2e-5;

struct Failure : std::runtime_error {
    int code;
    Failure(int value, const std::string &message) : std::runtime_error(message), code(value) {}
};
void require(bool value, const std::string &message, int code = 2) {
    if (!value) throw Failure(code, message);
}
NSString *ns(const fs::path &path) { return [NSString stringWithUTF8String:path.c_str()]; }
std::string str(NSString *value) { return value ? value.UTF8String : ""; }
id number(double value) { return std::isfinite(value) ? (id)@(value) : (id)[NSNull null]; }
std::string digest(const void *bytes, size_t size) {
    require(size <= std::numeric_limits<CC_LONG>::max(), "SHA input exceeds this small-artifact interface");
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(bytes, static_cast<CC_LONG>(size), hash);
    char result[2 * CC_SHA256_DIGEST_LENGTH + 1];
    for (size_t i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) std::snprintf(result + 2*i, 3, "%02x", hash[i]);
    return result;
}
NSData *read_data(const fs::path &path) {
    NSError *error = nil;
    NSData *data = [NSData dataWithContentsOfFile:ns(path) options:0 error:&error];
    require(data != nil, "Read failed: " + path.string() + ": " + str(error.localizedDescription));
    return data;
}
std::string file_sha(const fs::path &path) {
    NSData *data = read_data(path);
    return digest(data.bytes, data.length);
}
void save(const fs::path &path, const void *bytes, size_t size) {
    std::ofstream file(path, std::ios::binary);
    require(bool(file.write(static_cast<const char *>(bytes), size)), "Write failed: " + path.string());
}
void json(const fs::path &path, id value) {
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:&error];
    require(data != nil, "JSON serialization failed: " + str(error.localizedDescription));
    require([data writeToFile:ns(path) options:NSDataWritingAtomic error:&error], "JSON write failed: " + path.string());
}

struct Options {
    fs::path poc, q4, manifest, output, build;
    bool cpu_only = false;
};
Options parse(int argc, const char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--cpu-only") { o.cpu_only = true; continue; }
        require(i + 1 < argc, "Missing value after " + key);
        const fs::path value = argv[++i];
        if (key == "--poc") o.poc = value;
        else if (key == "--stock4x1") o.q4 = value;
        else if (key == "--inputs") o.manifest = value;
        else if (key == "--output") o.output = value;
        else if (key == "--expected-build") o.build = value;
        else throw Failure(2, "Unknown argument " + key);
    }
    require(!o.output.empty() && !o.build.empty(), "Require --output FRESH --expected-build FRESH_BUILD");
    require(o.cpu_only || (!o.poc.empty() && !o.q4.empty()), "GPU mode requires --poc and --stock4x1");
    return o;
}

struct Case {
    std::string id;
    int step = -1;
    Input input;
    std::string w_sha, x_sha;
    bool recorded_hashes = false;
};
Case make_generated(const char *id, Input input) {
    Case c; c.id = id; c.input = std::move(input);
    c.w_sha = digest(c.input.weights.data(), c.input.weights.size());
    c.x_sha = digest(c.input.activation.data(), sizeof(c.input.activation));
    return c;
}
std::vector<Case> inputs(const Options &o) {
    std::vector<Case> cases;
    cases.push_back(make_generated("known-answer", known_answer_input()));
    cases.push_back(make_generated("dense-seed42", dense_input()));
    if (o.manifest.empty()) return cases;
    NSError *error = nil;
    id object = [NSJSONSerialization JSONObjectWithData:read_data(o.manifest) options:0 error:&error];
    require([object isKindOfClass:[NSDictionary class]], "Invalid manifest JSON");
    NSArray *list = object[@"cases"];
    require([list isKindOfClass:[NSArray class]] && list.count > 0 && list.count <= 64, "Manifest cases missing/empty/too large");
    std::set<std::string> ids{"known-answer", "dense-seed42"};
    for (id record in list) {
        require([record isKindOfClass:[NSDictionary class]], "Manifest case is not an object");
        for (NSString *key in @[@"id", @"weights", @"activation", @"weights_sha256", @"activation_sha256"])
            require([record[key] isKindOfClass:[NSString class]], "Manifest missing string " + str(key));
        Case c; c.id = str(record[@"id"]);
        require(!c.id.empty() && c.id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") == std::string::npos && c.id != "." && c.id != "..", "Unsafe case id");
        require(ids.insert(c.id).second, "Duplicate case id " + c.id);
        if (record[@"step"]) { require([record[@"step"] isKindOfClass:[NSNumber class]], "Invalid step"); c.step = [record[@"step"] intValue]; }
        fs::path wp(str(record[@"weights"])), xp(str(record[@"activation"]));
        if (wp.is_relative()) wp = o.manifest.parent_path() / wp;
        if (xp.is_relative()) xp = o.manifest.parent_path() / xp;
        NSData *w = read_data(wp), *x = read_data(xp);
        require(w.length == weight_bytes && x.length == vector_bytes, "Input byte-size mismatch for " + c.id);
        c.w_sha = digest(w.bytes, w.length); c.x_sha = digest(x.bytes, x.length);
        require(c.w_sha == str(record[@"weights_sha256"]) && c.x_sha == str(record[@"activation_sha256"]), "Input SHA256 mismatch for " + c.id);
        c.input.weights.resize(weight_bytes); std::memcpy(c.input.weights.data(), w.bytes, weight_bytes);
        std::memcpy(c.input.activation.data(), x.bytes, vector_bytes);
        c.recorded_hashes = true; cases.push_back(std::move(c));
    }
    return cases;
}

NSDictionary *validate(const Case &c) {
    require(c.input.weights.size() == weight_bytes, "Invalid weight byte count");
    size_t zero_scales = 0, negative_scales = 0;
    for (size_t start = 0; start < weight_bytes; start += TQ2_BLOCK_BYTES) {
        const uint8_t *block = c.input.weights.data() + start;
        for (size_t j = 0; j < TQ2_CODES_BYTES; ++j)
            for (int plane = 0; plane < 4; ++plane)
                require(((block[j] >> (2*plane)) & 3) != 3, "Illegal trit code 3 in " + c.id);
        const uint16_t bits = uint16_t(block[64]) | (uint16_t(block[65]) << 8);
        _Float16 scale; std::memcpy(&scale, &bits, 2);
        require(std::isfinite(static_cast<float>(scale)), "Nonfinite FP16 scale in " + c.id);
        zero_scales += scale == 0; negative_scales += scale < 0;
    }
    for (float x : c.input.activation) require(std::isfinite(x), "Nonfinite activation in " + c.id);
    const auto *traits = ggml_get_type_traits(GGML_TYPE_TQ2_0);
    require(traits && traits->to_float, "Fresh GGML lacks TQ2_0 dequantizer");
    std::array<float,TQ2_COLUMNS> row{};
    size_t different = 0;
    double max_abs = 0;
    for (size_t r = 0; r < TQ2_ROWS; ++r) {
        traits->to_float(c.input.weights.data() + r*TQ2_BLOCKS_PER_ROW*TQ2_BLOCK_BYTES, row.data(), TQ2_COLUMNS);
        for (size_t k = 0; k < TQ2_COLUMNS; ++k) {
            const double expected = decoded_weight(c.input,r,k);
            const double delta = std::abs(expected - double(row[k]));
            different += !std::isfinite(row[k]) || delta != 0;
            if (delta > max_abs) max_abs = delta;
        }
    }
    require(different == 0, "Fresh GGML dequantizer differs from independent decoder in " + c.id, 1);
    return @{@"legal_trits":@YES,@"finite_scales_and_activation":@YES,@"zero_scale_blocks":@(zero_scales),
             @"negative_scale_blocks":@(negative_scales),@"scale_policy":@"finite FP16; signed finite values are not silently rejected",
             @"fresh_ggml_dequantizer_crosscheck_pass":@YES,@"weights_checked":@(size_t(TQ2_ROWS)*TQ2_COLUMNS),
             @"dequantizer_max_abs":@(max_abs),@"ggml_dot_used_as_reference":@NO};
}

NSArray *loaded_libraries(const Options &o) {
    const fs::path expected = fs::canonical(o.build / "bin");
    NSMutableArray *records = [NSMutableArray array];
    bool base = false, metal = false;
    std::set<std::string> seen;
    for (uint32_t i = 0; i < _dyld_image_count(); ++i) {
        const char *name = _dyld_get_image_name(i);
        if (!name) continue;
        fs::path image(name);
        if (image.filename().string().rfind("libggml",0) != 0) continue;
        const fs::path resolved = fs::canonical(image);
        require(resolved.parent_path() == expected, "Loaded non-fresh GGML image: " + resolved.string());
        if (!seen.insert(resolved.string()).second) continue;
        base |= resolved.filename().string().rfind("libggml-base",0) == 0;
        metal |= resolved.filename().string().rfind("libggml-metal",0) == 0;
        [records addObject:@{@"dyld_name":ns(image),@"resolved_path":ns(resolved),@"sha256":@(file_sha(resolved).c_str()),@"under_expected_fresh_build":@YES}];
    }
    require(base && metal, "Fresh GGML base/Metal images missing from actual dyld list");
    return records;
}

struct Metrics {
    double max_abs = 0, square_error = 0, square_reference = 0, max_relative = 0;
    size_t failing = 0, nonfinite = 0, worst = 0, zero_reference = 0;
    void add(size_t row, double reference, double actual) {
        if (!std::isfinite(reference) || !std::isfinite(actual)) { ++nonfinite; ++failing; max_abs = square_error = std::numeric_limits<double>::infinity(); worst = row; return; }
        const double e = std::abs(actual-reference);
        if (e > max_abs) { max_abs = e; worst = row; }
        square_error += e*e; square_reference += reference*reference;
        if (reference == 0) ++zero_reference;
        else max_relative = std::max(max_relative,e/std::abs(reference));
        failing += e > abs_tolerance + rel_tolerance*std::abs(reference);
    }
    NSDictionary *dict() const {
        const double l2 = square_reference == 0 ? (square_error == 0 ? 0 : std::numeric_limits<double>::infinity()) : std::sqrt(square_error/square_reference);
        return @{@"max_abs":number(max_abs),@"relative_l2":number(l2),@"max_relative_nonzero_reference":number(max_relative),
                 @"zero_reference_rows":@(zero_reference),@"failing_rows":@(failing),@"nonfinite_rows":@(nonfinite),@"worst_row":@(worst),@"rows_checked":@TQ2_ROWS,@"pass":@(failing == 0)};
    }
};
struct Route {
    const char *name;
    const char *symbol;
    size_t simd_groups;
    size_t rows_per_simd;
    bool stock;
    id<MTLComputePipelineState> pipeline = nil;
    std::string library, sha;
    Route(const char *n, const char *s, size_t groups, bool native, size_t rows = TQ2_POC_ROWS_PER_SIMD) : name(n), symbol(s), simd_groups(groups), rows_per_simd(rows), stock(native) {}
};
Route custom(id<MTLDevice> device, const fs::path &path, const char *name, const char *symbol, size_t groups, size_t rows = TQ2_POC_ROWS_PER_SIMD) {
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:ns(path)] error:&error];
    require(library != nil, "Load custom library: " + path.string() + ": " + str(error.localizedDescription));
    id<MTLFunction> function = [library newFunctionWithName:@(symbol)];
    require(function != nil, "Missing custom kernel " + std::string(symbol));
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline != nil, "Custom pipeline creation: " + str(error.localizedDescription));
    require(pipeline.threadExecutionWidth == TQ2_SIMD_WIDTH && pipeline.maxTotalThreadsPerThreadgroup >= TQ2_SIMD_WIDTH*groups, "Custom pipeline thread limit/width mismatch");
    Route route{name,symbol,groups,false,rows}; route.pipeline = pipeline; route.library = fs::canonical(path).string(); route.sha = file_sha(path);
    return route;
}
NSDictionary *dispatch(StockRuntime &stock, const Route &route, const Case &c, id<MTLBuffer> w, id<MTLBuffer> x,
                       std::array<float,TQ2_ROWS> &values, const fs::path &folder) {
    id<MTLDevice> device = stock.device();
    id<MTLBuffer> y = [device newBufferWithLength:guard_bytes + vector_bytes + guard_bytes options:MTLResourceStorageModeShared];
    require(y != nil, "Output allocation failed");
    std::memset(y.contents,0xA5,y.length);
    float *body = reinterpret_cast<float *>(static_cast<uint8_t *>(y.contents)+guard_bytes);
    std::fill_n(body,TQ2_ROWS,std::numeric_limits<float>::quiet_NaN());
    id<MTLCommandBuffer> command = [stock.queue() commandBuffer];
    require(command != nil, "Command allocation failed");
    command.label = [NSString stringWithFormat:@"BitNet-clean-correctness %@ %@",@(c.id.c_str()),@(route.name)];
    const size_t rows_per_simd = route.rows_per_simd;
    const size_t grid = TQ2_ROWS/(rows_per_simd*route.simd_groups);
    if (route.stock) stock.encode(command,w,x,y,0,0,guard_bytes);
    else {
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        require(encoder != nil, "Encoder allocation failed"); encoder.label = command.label;
        [encoder setComputePipelineState:route.pipeline];
        [encoder setBuffer:w offset:0 atIndex:0]; [encoder setBuffer:x offset:0 atIndex:1]; [encoder setBuffer:y offset:guard_bytes atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(grid,1,1) threadsPerThreadgroup:MTLSizeMake(TQ2_SIMD_WIDTH,route.simd_groups,1)];
        [encoder endEncoding];
    }
    [command commit]; [command waitUntilCompleted];
    NSDictionary *base = @{@"route":@(route.name),@"kernel_symbol":@(route.symbol),@"command_label":command.label,
        @"command_status":@(command.status),@"completed":@(command.status == MTLCommandBufferStatusCompleted),
        @"actual_stock_selector":@(route.stock),@"stock_library_source":route.stock ? @"fresh ggml-metal embedded-source native selector; not an offline stock copy" : @"fresh offline custom metallib",
        @"grid":@[@(grid),@1,@1],@"threads":@[@TQ2_SIMD_WIDTH,@(route.simd_groups),@1],
        @"weight_handle":[NSString stringWithFormat:@"%p",w],@"activation_handle":[NSString stringWithFormat:@"%p",x],@"output_handle":[NSString stringWithFormat:@"%p",y],
        @"buffer_offsets":@[@0,@0,@(guard_bytes)],@"buffer_lengths":@[@(w.length),@(x.length),@(y.length)],
        @"binding_slots":route.stock ? @{@"args":@0,@"W":@1,@"X":@2,@"Y":@3,@"auxY":@4,@"auxX":@5} : @{@"W":@0,@"X":@1,@"Y":@2},
        @"stock_argument_bytes":route.stock ? @(sizeof(ggml_metal_kargs_mul_mv)) : @0,
        @"native_selector_nr0":route.stock ? @(stock.pipeline().nr0) : (id)[NSNull null],
        @"native_selector_nsg":route.stock ? @(stock.pipeline().nsg) : (id)[NSNull null],
        @"library_path":route.stock ? (id)[NSNull null] : @(route.library.c_str()),@"library_sha256":route.stock ? (id)[NSNull null] : @(route.sha.c_str()),
        @"thread_execution_width":@(route.stock ? ggml_metal_pipeline_thread_execution_width(stock.pipeline()) : route.pipeline.threadExecutionWidth),
        @"max_threads":@(route.stock ? ggml_metal_pipeline_max_theads_per_threadgroup(stock.pipeline()) : route.pipeline.maxTotalThreadsPerThreadgroup),
        @"timing_measured":@NO};
    json(folder/(std::string(route.name)+"-dispatch.json"),base);
    require(command.status == MTLCommandBufferStatusCompleted,"GPU execution failed: " + str(command.error.localizedDescription));
    // Read only after completion; preserve output even if its numerical checks fail.
    std::memcpy(values.data(),body,vector_bytes);
    save(folder/(std::string(route.name)+".f32"),values.data(),vector_bytes);
    size_t corrupt = 0, finite = 0;
    const uint8_t *all = static_cast<const uint8_t *>(y.contents);
    for (size_t i = 0; i < guard_bytes; ++i) corrupt += all[i] != 0xA5;
    for (size_t i = guard_bytes+vector_bytes; i < y.length; ++i) corrupt += all[i] != 0xA5;
    for (float v : values) finite += std::isfinite(v);
    std::array<unsigned,TQ2_ROWS> ownership{};
    for (size_t g = 0; g < grid; ++g) for (size_t s = 0; s < route.simd_groups; ++s)
        for (size_t r = 0; r < rows_per_simd; ++r) ++ownership[(g*route.simd_groups+s)*rows_per_simd+r];
    const bool once = std::all_of(ownership.begin(),ownership.end(),[](unsigned n){return n == 1;});
    const bool unchanged = digest(w.contents,w.length) == c.w_sha && digest(x.contents,x.length) == c.x_sha;
    NSMutableDictionary *report = [base mutableCopy];
    [report addEntriesFromDictionary:@{@"finite_rows":@(finite),@"nonfinite_rows":@(TQ2_ROWS-finite),@"canary_corrupt_bytes":@(corrupt),@"canary_pass":@(corrupt == 0),
        @"coverage_pass":@(once && finite == TQ2_ROWS && corrupt == 0),@"expected_index_ownership_once":@(once),
        @"coverage_evidence":@"enumerated expected row map + NaN-to-finite readback + prefix/suffix canaries; no instrumented write counter",
        @"input_immutable_pass":@(unchanged)}];
    json(folder/(std::string(route.name)+"-dispatch.json"),report);
    require(unchanged,"GPU changed the common W/X input; remaining routes cannot be compared");
    std::printf("CLEAN_POC_GPU_COMPLETE case=%s route=%s pipeline=%s grid=%zu threads=(32,%zu,1) status=%lu rows=%zu canary=%zu\n",c.id.c_str(),route.name,route.symbol,grid,route.simd_groups,(unsigned long)command.status,finite,corrupt);
    return report;
}

int perform(const Options &o, NSMutableDictionary *summary) {
    NSArray *images = loaded_libraries(o);
    summary[@"loaded_ggml_images"] = images;
    summary[@"expected_fresh_build"] = ns(fs::canonical(o.build));
    summary[@"real_inputs_manifest"] = o.manifest.empty() ? (id)[NSNull null] : ns(fs::canonical(o.manifest));
    const std::vector<Case> cases = inputs(o);
    summary[@"requested_cases"] = @(cases.size());
    NSMutableArray *results = [NSMutableArray array]; summary[@"cases"] = results;
    bool known_exact = true;
    const auto expected_known = reference_matvec(cases.front().input);
    for (size_t i = 0; i < TQ2_ROWS; ++i) {
        const double expected = i == 0 ? 1.375 : (i == TQ2_ROWS-1 ? -1.25 : 0);
        known_exact &= expected_known[i] == expected;
    }
    summary[@"known_answer_cpu_exact_pass"] = @(known_exact);
    require(known_exact,"Independent CPU known-answer reference failed",1);
    std::unique_ptr<StockRuntime> stock;
    std::vector<Route> routes;
    if (!o.cpu_only) {
        stock = std::make_unique<StockRuntime>();
        require(stock->device() != nil && stock->queue() != nil,"Stock runtime device/queue missing");
        summary[@"device"] = stock->device().name;
        summary[@"metal_runtime_initialized"] = @YES;
        routes.push_back(Route{"S","kernel_mul_mv_tq2_0_f32",2,true});
        routes.push_back(custom(stock->device(),o.poc,"PoC","tq2_0_matvec_poc",2));
        routes.push_back(custom(stock->device(),o.q4,"Q4",GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_ENTRY),GGML_BITNET_Q4_NSG,GGML_BITNET_Q4_ROWS));
    }
    bool all_fp64 = true, all_parity = true, all_coverage = true, all_known = true;
    for (const Case &c : cases) {
        const fs::path folder = o.output/c.id; fs::create_directory(folder);
        NSDictionary *input_checks = validate(c);
        const auto reference = reference_matvec(c.input);
        require(std::all_of(reference.begin(),reference.end(),[](double x){return std::isfinite(x);}),"Nonfinite CPU reference",1);
        std::array<float,TQ2_ROWS> rounded{};
        std::transform(reference.begin(),reference.end(),rounded.begin(),[](double x){return static_cast<float>(x);});
        save(folder/"W.tq2_0",c.input.weights.data(),c.input.weights.size());
        save(folder/"X.f32",c.input.activation.data(),sizeof(c.input.activation));
        save(folder/"reference.f64",reference.data(),sizeof(reference)); save(folder/"reference.f32",rounded.data(),sizeof(rounded));
        NSMutableDictionary *result = [@{@"id":@(c.id.c_str()),@"step":c.step < 0 ? (id)[NSNull null] : @(c.step),
            @"weights_sha256":@(c.w_sha.c_str()),@"activation_sha256":@(c.x_sha.c_str()),@"recorded_input_hashes_verified":@(c.recorded_hashes),
            @"input_checks":input_checks,@"rows":@TQ2_ROWS,@"reference_finite":@YES,@"raw_directory":ns(folder)} mutableCopy];
        [results addObject:result];
        if (o.cpu_only) { result[@"fp64_pass"] = [NSNull null]; result[@"parity_pass"] = [NSNull null]; result[@"coverage_pass"] = [NSNull null]; continue; }
        id<MTLBuffer> w = [stock->device() newBufferWithBytes:c.input.weights.data() length:weight_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> x = [stock->device() newBufferWithBytes:c.input.activation.data() length:vector_bytes options:MTLResourceStorageModeShared];
        require(w && x,"Common W/X allocation failed");
        require(digest(w.contents,w.length) == c.w_sha && digest(x.contents,x.length) == c.x_sha,"Resident input copy hash mismatch");
        std::array<std::array<float,TQ2_ROWS>,3> output{};
        NSMutableArray *dispatches = [NSMutableArray array]; result[@"dispatches"] = dispatches;
        bool coverage = true;
        for (size_t r = 0; r < routes.size(); ++r) {
            NSDictionary *record = dispatch(*stock,routes[r],c,w,x,output[r],folder);
            [dispatches addObject:record]; coverage &= [record[@"coverage_pass"] boolValue];
        }
        std::array<Metrics,5> metrics{}; size_t poc_bits = 0, q4_bits = 0;
        std::ofstream rows(folder/"rows.csv"); require(bool(rows),"Open rows.csv failed");
        rows << "row,reference_f64,S_f32,PoC_f32,Q4_f32,S_abs_error,PoC_abs_error,Q4_abs_error,tolerance,PoC_stock_abs_error,Q4_stock_abs_error,PoC_stock_bitwise_equal,Q4_stock_bitwise_equal\n" << std::setprecision(17);
        for (size_t i = 0; i < TQ2_ROWS; ++i) {
            for (size_t r = 0; r < 3; ++r) metrics[r].add(i,reference[i],output[r][i]);
            metrics[3].add(i,output[0][i],output[1][i]); metrics[4].add(i,output[0][i],output[2][i]);
            const bool pb = std::memcmp(&output[0][i],&output[1][i],sizeof(float)) == 0;
            const bool qb = std::memcmp(&output[0][i],&output[2][i],sizeof(float)) == 0;
            poc_bits += !pb; q4_bits += !qb;
            rows << i << ',' << reference[i]; for (size_t r = 0; r < 3; ++r) rows << ',' << output[r][i];
            for (size_t r = 0; r < 3; ++r) rows << ',' << std::abs(reference[i]-output[r][i]);
            rows << ',' << abs_tolerance+rel_tolerance*std::abs(reference[i]) << ',' << std::abs(double(output[1][i])-output[0][i]) << ',' << std::abs(double(output[2][i])-output[0][i]) << ',' << pb << ',' << qb << '\n';
        }
        rows.close(); require(bool(rows),"Save rows.csv failed");
        bool fp64 = true, parity = true;
        for (size_t r = 0; r < 3; ++r) fp64 &= metrics[r].failing == 0;
        for (size_t r = 3; r < 5; ++r) parity &= metrics[r].failing == 0;
        result[@"comparisons"] = @{@"S_vs_fp64":metrics[0].dict(),@"PoC_vs_fp64":metrics[1].dict(),@"Q4_vs_fp64":metrics[2].dict(),@"PoC_vs_S":metrics[3].dict(),@"Q4_vs_S":metrics[4].dict()};
        result[@"bitwise_diff_rows"] = @{@"PoC_vs_S":@(poc_bits),@"Q4_vs_S":@(q4_bits)};
        result[@"fp64_pass"] = @(fp64); result[@"parity_pass"] = @(parity); result[@"coverage_pass"] = @(coverage); result[@"input_immutable_pass"] = @YES;
        result[@"overall_pass"] = @(fp64 && parity && coverage);
        all_fp64 &= fp64; all_parity &= parity; all_coverage &= coverage;
        if (c.id == "known-answer") all_known &= fp64 && parity && coverage;
    }
    summary[@"processed_cases"] = @(results.count); summary[@"execution_pass"] = @YES;
    summary[@"knownanswer_pass"] = @(known_exact && all_known);
    summary[@"fp64_pass"] = o.cpu_only ? (id)[NSNull null] : @(all_fp64);
    summary[@"parity_pass"] = o.cpu_only ? (id)[NSNull null] : @(all_parity);
    summary[@"coverage_pass"] = o.cpu_only ? (id)[NSNull null] : @(all_coverage);
    const bool pass = o.cpu_only || (all_fp64 && all_parity && all_coverage && all_known);
    summary[@"overall_pass"] = @(pass); summary[@"exit_code"] = @(pass ? 0 : 1);
    return pass ? 0 : 1;
}
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        fs::path output; bool created = false;
        NSMutableDictionary *summary = [@{@"schema":@1,@"scope":@"standalone operator M2560 K2560 N1; no real-graph dispatch or model quality acceptance",
            @"abs_tolerance":@(abs_tolerance),@"rel_tolerance":@(rel_tolerance),@"required_row_failures":@0,
            @"parity_definition":@"candidate-vs-actual-S pointwise frozen tolerance; bitwise differences separately reported",
            @"reference_definition":@"independent scalar FP64, logical block/half/plane/column K order; fresh GGML only cross-checks unpacked weights",
            @"execution_pass":@NO,@"knownanswer_pass":[NSNull null],@"parity_pass":[NSNull null],@"fp64_pass":[NSNull null],@"coverage_pass":[NSNull null],
            @"model_output_pass":[NSNull null],@"overall_pass":@NO,@"metal_runtime_initialized":@NO,@"formal_performance_run":@NO} mutableCopy];
        try {
            const Options o = parse(argc,argv); output = fs::absolute(o.output);
            require(!fs::exists(output),"Refuse existing output: " + output.string());
            fs::create_directories(output); created = true;
            summary[@"mode"] = o.cpu_only ? @"cpu_only" : @"gpu";
            summary[@"overall_definition"] = o.cpu_only ? @"all mandatory CPU prerequisite checks; GPU verdicts unverified/null" : @"all requested cases, all three GPU paths: execution, known answer, finite coverage/canaries, FP64 and pointwise stock parity";
            const int code = perform(o,summary);
            json(output/"summary.json",summary);
            std::printf("CLEAN_POC_COMPLETE execution_pass=1 overall_pass=%d exit_code=%d summary=%s\n",code == 0,code,(output/"summary.json").c_str());
            return code;
        } catch (const std::exception &error) {
            const auto *typed = dynamic_cast<const Failure *>(&error); const int code = typed ? typed->code : 2;
            summary[@"error"] = @(error.what()); summary[@"exit_code"] = @(code); summary[@"overall_pass"] = @NO;
            if (created) { try { json(output/"summary.json",summary); } catch (...) {} }
            std::fprintf(stderr,"CLEAN_POC_FAILED exit_code=%d error=%s\n",code,error.what());
            return code;
        }
    }
}
