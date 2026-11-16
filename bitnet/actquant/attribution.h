// Included by runner.cpp inside bitnet_actquant.
static uint64_t attribution_ns() {
    timespec value;
    require(clock_gettime(CLOCK_UPTIME_RAW, &value) == 0, "host clock failed");
    return uint64_t(value.tv_sec)*1000000000ULL + uint64_t(value.tv_nsec);
}

static json attribution_state() {
#if defined(__OBJC__)
    static const char * names[] = {"nominal", "fair", "serious", "critical"};
    const auto state = NSProcessInfo.processInfo.thermalState;
    return {{"thermal", state >= 0 && state < 4 ? names[state] : "unknown"},
        {"low_power", bool(NSProcessInfo.processInfo.lowPowerModeEnabled)},
        {"battery_level", double(UIDevice.currentDevice.batteryLevel)},
        {"battery_state", int(UIDevice.currentDevice.batteryState)},
        {"iOS", UIDevice.currentDevice.systemVersion.UTF8String}};
#else
    return {{"thermal", "unavailable"}, {"low_power", nullptr}, {"battery_level", nullptr}, {"battery_state", nullptr}};
#endif
}

static void attribution_route(const std::string & route) {
    require(route == "S" || route == "QB" || route == "Q4", "bad attribution route");
    unsetenv("GGML_METAL_BITNET_POC"); unsetenv("GGML_METAL_BITNET_Q_GUARD_ONLY");
    if (route != "S") { setenv(route == "QB" ? "GGML_METAL_BITNET_Q_GUARD_ONLY" : "GGML_METAL_BITNET_POC", "1", 1); }
}

struct AttributionPath {
    std::string route, final_logits_sha256, diagnostic_file, subrun;
    bool metrics = false;
    uint64_t begin = 0, end = 0, reset_ns = 0, prefill_ns = 0;
    std::vector<uint64_t> token_begin, token_end;
    json before, after;
};

static std::string attribution_digest(const float * values, size_t count) {
    unsigned char bytes[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(values, CC_LONG(count*sizeof(float)), bytes);
    static const char hex[] = "0123456789abcdef";
    std::string result;
    for (unsigned char byte : bytes) { result += hex[byte >> 4]; result += hex[byte & 15]; }
    return result;
}

static int attribution_test(const json & config, const fs::path & out) {
#if defined(__OBJC__)
    UIDevice.currentDevice.batteryMonitoringEnabled = YES;
#endif
    const std::string kind = config.value("attribution_kind", "formal");
    require(kind == "formal" || kind == "diagnostic" || kind == "overhead" || kind == "verify", "bad attribution kind");
    require(config.value("actquant", false) && config.value("backend", "MTL0") == "MTL0" &&
        config.value("kv", "F16") == "F16" && config.value("prefill", "batched") == "batched" &&
        config.value("workload", "teacher") == "teacher" &&
        config.value("capture", false) == (kind == "verify") && config.value("trace", false) == (kind == "verify"),
        "attribution requires ActQuant ON, MTL0/batched/F16/teacher; capture/trace are verify-only");
    const bool verification = kind == "verify";
    const int rounds = config.value("rounds", kind == "diagnostic" || verification ? 1 : 3);
    const int pairs = config.value("pairs", kind == "diagnostic" || verification ? 1 : kind == "overhead" ? 7 : 21);
    const int warmups = config.value("warmups", 2);
    require(warmups == 2 && rounds == (kind == "diagnostic" || verification ? 1 : 3) &&
        pairs == (kind == "diagnostic" || verification ? 1 : kind == "overhead" ? 7 : 21), "attribution protocol must remain frozen");
    require(verification || config.value("correctness_gate_sha256", "").size() == 64, "same-build Q4/QB correctness gate is required");
    const fs::path model = config.at("model").get<std::string>(), token_file = config.at("tokens").get<std::string>();
    const fs::path library = config.at("q_library").get<std::string>();
    require(hash_file(model) == MODEL_SHA && library.is_absolute() && fs::is_regular_file(library), "model/library identity");
    const std::string library_sha = hash_file(library);
    require(library_sha == config.at("q_library_sha256").get<std::string>() && config.value("q_variant", "") == "stock4x1" &&
        config.value("q_library_source_sha256", "").size() == 64 && config.value("q_library_manifest_sha256", "").size() == 64 &&
        config.value("q_kernel_contract_version", 0) == GGML_BITNET_Q_CONFIG_VERSION &&
        config.value("q_kernel_config_sha256", "").size() == 64 &&
        config.value("q_entry", "") == GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_ENTRY) &&
        config.at("q_grid") == json::array({GGML_BITNET_Q_NROWS/(GGML_BITNET_Q4_ROWS*GGML_BITNET_Q4_NSG),1,1}) &&
        config.at("q_threads") == json::array({GGML_BITNET_Q_SIMD_WIDTH,GGML_BITNET_Q4_NSG,1}), "Q4 canonical provenance");
    const auto tokens = read_json(token_file);
    const auto prompt = tokens.at("prompt").get<std::vector<llama_token>>();
    const auto continuation = tokens.at("continuation").get<std::vector<llama_token>>();
    require(prompt.size() == 27 && continuation.size() == 32 && prompt.front() == 128000, "attribution token counts/BOS");
    for (const auto & sequence : {prompt, continuation}) {
        for (auto token : sequence) { require(token >= 0 && token < 128256, "attribution token outside vocabulary"); }
    }
    clear_custom_routes(); setenv("LLAMA_BITNET_ACTQUANT", "1", 1); setenv("LLAMA_BITNET_ACTQUANT_Q4", "1", 1);
    setenv("GGML_METAL_ACTQUANT_TRACE", "0", 1); setenv("GGML_METAL_BITNET_LAYERS", "0", 1);
    setenv("GGML_METAL_BITNET_POC_VARIANT", "stock4x1", 1);
    setenv("GGML_METAL_BITNET_POC_ROWS_PER_SIMD", GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_ROWS), 1);
    setenv("GGML_METAL_BITNET_POC_SIMD_PER_GROUP", GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_NSG), 1);
    setenv("GGML_METAL_BITNET_POC_METALLIB", library.c_str(), 1); attribution_route("S");
    ModelLog log; llama_log_set(model_logger, &log); llama_backend_init();
    struct Cleanup { ~Cleanup() { clear_custom_routes(); llama_backend_free(); llama_log_set(nullptr, nullptr); } } cleanup;
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name("MTL0"), nullptr}; require(devices[0], "MTL0 missing");
    using reset_fn = void (*)(void); using write_fn = int (*)(const char *);
    const auto reg = ggml_backend_dev_backend_reg(devices[0]);
    const auto reset = reinterpret_cast<reset_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_q_attribution_diag_reset"));
    const auto write = reinterpret_cast<write_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_q_attribution_diag_write"));
    require(reset && write, "same-build backend diagnostic API missing");
    auto mp = llama_model_default_params(); mp.devices = devices; mp.n_gpu_layers = 999; mp.check_tensors = true;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> m(llama_model_load_from_file(model.c_str(), mp), llama_model_free);
    require(m && llama_model_n_layer(m.get()) == 30 && llama_vocab_n_tokens(llama_model_get_vocab(m.get())) == 128256 &&
        log.offloaded == 31 && log.total == 31, "wrong model/offload geometry");
    Capture capture; capture.scope = "q4-layer0";
    auto cp = llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = cp.n_ubatch = 128;
    cp.n_threads = cp.n_threads_batch = 6; cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.offload_kqv = true; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = verification ? Capture::callback : nullptr; cp.cb_eval_user_data = verification ? &capture : nullptr;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(m.get(), cp), llama_free);
    require(bool(ctx), "attribution context creation failed");
    json paths = json::array(); uint64_t serial = 0; std::string expected_final_sha;
    auto replay = [&](const std::string & route, bool metrics, bool keep_diagnostic,
            bool record_verification = false, bool captured = false, bool traced = false) {
        AttributionPath result; result.route = route; result.metrics = metrics;
        const uint64_t path_id = ++serial;
        const fs::path folder = out/"verification"/("path-" + std::to_string(path_id) + "-" + route + (captured?"-capture":"-plain"));
        capture = Capture{}; capture.scope = "q4-layer0"; capture.out = folder; capture.enabled = captured;
        json logit_records = json::array();
        auto snapshot = [&](int position) {
            const float * y = llama_get_logits_ith(ctx.get(), -1); require(y, "verify logits missing");
            std::vector<float> values(y,y+128256);
            require(std::all_of(values.begin(),values.end(),[](float x){return std::isfinite(x);}),"verify nonfinite logits");
            const std::string path = "logits/position-" + std::to_string(position) + ".f32"; write_f32(folder/path,values);
            logit_records.push_back({{"input_position",position},{"path",path},{"vocab",128256},
                {"greedy_token",int(std::max_element(values.begin(),values.end())-values.begin())}});
        };
        result.token_begin.reserve(32); result.token_end.reserve(32);
        unsetenv("GGML_METAL_BITNET_Q_ATTRIBUTION"); attribution_route("S"); unsetenv("GGML_METAL_BITNET_FFN_DECODE");
        auto before = attribution_ns(); llama_synchronize(ctx.get()); llama_memory_clear(llama_get_memory(ctx.get()), true); llama_synchronize(ctx.get());
        require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == -1, "attribution KV reset failed");
        result.reset_ns = attribution_ns() - before; before = attribution_ns();
        setenv("GGML_METAL_BITNET_FFN_STEP", "0", 1);
        capture.positions.clear(); for (int i=0;i<27;++i) { capture.positions.push_back(i); }
        const std::string native_route = route == "S" ? "stock" : route;
        if (traced) {
            setenv("GGML_METAL_BITNET_POC_TRACE","1",1);
            std::fprintf(stderr,"ACTQUANT_ATTRIBUTION_PATH_BEGIN id=%llu route=%s capture=%d context=%p\n",(unsigned long long)path_id,native_route.c_str(),captured,(void*)ctx.get());
            std::fprintf(stderr,"ACTQUANT_Q_PHASE_BEGIN q_route=%s decode_step=0\n",native_route.c_str());
        }
        require(llama_decode(ctx.get(), llama_batch_get_one(const_cast<llama_token *>(prompt.data()), 27)) == 0, "attribution prefill failed");
        llama_synchronize(ctx.get()); require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == 26, "attribution prefill KV mismatch");
        require(capture.error.empty(),"attribution prefill capture failed: " + capture.error);
        if (traced) { std::fprintf(stderr,"ACTQUANT_Q_PHASE_COMPLETE q_route=%s decode_step=0 synchronized=1\n",native_route.c_str()); }
        result.prefill_ns = attribution_ns() - before;
        if (record_verification) { snapshot(26); }
        attribution_route(route); setenv("GGML_METAL_BITNET_FFN_DECODE", "1", 1);
        if (metrics) { reset(); setenv("GGML_METAL_BITNET_Q_ATTRIBUTION", "1", 1); }
        result.before = attribution_state(); result.begin = attribution_ns();
        for (int step = 0; step < 32; ++step) {
            const auto begin = attribution_ns(); const std::string text = std::to_string(step+1);
            setenv("GGML_METAL_BITNET_FFN_STEP", text.c_str(), 1);
            capture.positions = {27+step};
            if (traced) { std::fprintf(stderr,"ACTQUANT_Q_PHASE_BEGIN q_route=%s decode_step=%d\n",native_route.c_str(),step+1); }
            require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == 26+step, "attribution token KV prefix mismatch");
            require(llama_decode(ctx.get(), llama_batch_get_one(const_cast<llama_token *>(&continuation[step]), 1)) == 0, "attribution decode failed");
            llama_synchronize(ctx.get());
            require(capture.error.empty(),"attribution decode capture failed: " + capture.error);
            require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == 27+step, "attribution token KV end mismatch");
            result.token_begin.push_back(begin); result.token_end.push_back(attribution_ns());
            if (traced) { std::fprintf(stderr,"ACTQUANT_Q_PHASE_COMPLETE q_route=%s decode_step=%d synchronized=1\n",native_route.c_str(),step+1); }
            if (record_verification) { snapshot(27+step); }
        }
        result.end = attribution_ns(); result.after = attribution_state(); unsetenv("GGML_METAL_BITNET_Q_ATTRIBUTION");
        unsetenv("GGML_METAL_BITNET_POC_TRACE");
        const float * logits = llama_get_logits_ith(ctx.get(), -1); require(logits != nullptr, "attribution final logits missing");
        require(std::all_of(logits, logits+128256, [](float x) { return std::isfinite(x); }), "attribution nonfinite final logits");
        result.final_logits_sha256 = attribution_digest(logits, 128256);
        if (expected_final_sha.empty()) { require(route=="S","first attribution replay must be stock"); expected_final_sha=result.final_logits_sha256; }
        require(result.final_logits_sha256==expected_final_sha,"attribution final logits differ: route="+route+
            " metrics="+std::to_string(metrics)+" expected="+expected_final_sha+" actual="+result.final_logits_sha256);
        if (metrics && keep_diagnostic) {
            result.diagnostic_file = "diagnostics/path-" + std::to_string(path_id) + ".json";
            fs::create_directories((out/result.diagnostic_file).parent_path());
            require(write((out/result.diagnostic_file).c_str()) == 0, "backend diagnostics incomplete or write failed");
        }
        if (record_verification) {
            json record={{"schema",1},{"operation","model"},{"workload","teacher"},{"backend","MTL0"},{"device",ggml_backend_dev_description(devices[0])},
                {"activation_quantization",true},{"model_sha256",MODEL_SHA},{"tokens_sha256",hash_file(token_file)},{"prompt_ids",prompt},{"continuation_ids",continuation},
                {"token_ids",prompt},{"prefill","batched"},{"kv","F16"},{"capture_enabled",captured},{"offloaded_layers",31},{"total_offload_layers",31},
                {"logits",logit_records},{"performance_measurement",false},{"q_route",native_route},{"q_fallback","none"},{"capture_scope","q4-layer0"},
                {"q_library_sha256",library_sha},{"q_rows_per_simd",route=="S"?0:GGML_BITNET_Q4_ROWS},{"q_simd_per_group",route=="S"?0:GGML_BITNET_Q4_NSG},
                {"q_entry",GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_ENTRY)},{"q_kernel_contract_version",GGML_BITNET_Q_CONFIG_VERSION},
                {"settings",{{"n_ctx",512},{"batch",128},{"ubatch",128},{"threads",6},{"flash_attention",false}}},
                {"checkpoints",capture.checkpoints},{"completed_decode_calls",33},{"finite_logits",true},{"execution_pass",true},
                {"context_id",std::to_string(uintptr_t(ctx.get()))},{"callback_pointer_present",true},{"trace_enabled",traced},
                {"parity_pass",nullptr},{"fp64_pass",nullptr},{"model_output_pass",nullptr},{"strict_model_exit_code",2}};
            for (auto token:continuation) { record["token_ids"].push_back(token); }
            for (const char * key : {"q_variant","q_library_source_sha256","q_library_manifest_sha256","q_library_sdk","q_kernel_config_sha256","q_grid","q_threads"}) { record[key]=config.at(key); }
            write_json(folder/"RUN.json",record); result.subrun=fs::relative(folder,out).string();
            if (traced) {
                std::fprintf(stderr,"ACTQUANT_MODEL_COMPLETE enabled=1 backend=MTL0 workload=teacher decode_calls=33 logits=33 actquant_callbacks=%zu synchronized=1\n",capture.actquant_callbacks);
                std::fprintf(stderr,"ACTQUANT_ATTRIBUTION_PATH_END id=%llu route=%s synchronized=1\n",(unsigned long long)path_id,native_route.c_str());
            }
        }
        return result;
    };
    const std::vector<std::string> routes = {"S", "QB", "Q4"};
    for (int i = 0; i < warmups; ++i) {
        for (const auto & route : routes) { replay(route, kind == "diagnostic", false); if (kind == "overhead") { replay(route, true, false); } }
    }
    if (verification) {
        for (bool captured : {true,false}) {
            for (const auto & route : routes) {
                const auto result=replay(route,false,false,true,captured,captured);
                paths.push_back({{"route",route},{"capture_enabled",captured},{"subrun",result.subrun},{"final_logits_sha256",result.final_logits_sha256}});
            }
        }
        json runtime={{"schema",1},{"operation","attribution"},{"attribution_kind",kind},{"backend","MTL0"},
            {"device",ggml_backend_dev_description(devices[0])},{"model_sha256",MODEL_SHA},{"tokens_sha256",hash_file(token_file)},
            {"one_model_and_context",true},{"context_id",std::to_string(uintptr_t(ctx.get()))},{"activation_quantization",true},
            {"offloaded_layers",31},{"total_offload_layers",31},{"paths",paths},{"capture_enabled",true},{"trace_enabled",true},
            {"q_library_sha256",library_sha},{"q_entry",GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_ENTRY)},{"q_kernel_contract_version",GGML_BITNET_Q_CONFIG_VERSION},
            {"execution_pass",true},{"finite_logits",true},{"performance_measurement",false},{"parity_pass",nullptr},{"fp64_pass",nullptr},
            {"model_output_pass",nullptr},{"semantic_acceptance_pass",nullptr},{"strict_model_exit_code",2}};
        for (const char * key : {"q_variant","q_library_source_sha256","q_library_manifest_sha256","q_library_sdk","q_kernel_config_sha256","q_grid","q_threads"}) { runtime[key]=config.at(key); }
        write_json(out/"RUN.json",runtime);
        std::printf("ACTQUANT_ATTRIBUTION_COMPLETE kind=verify paths=6 synchronized=1 numerical_acceptance_unverified=1\n");std::fflush(stdout);return 0;
    }
    std::vector<std::string> contrasts;
    if (kind == "overhead") { contrasts = {"S_METRICS", "QB_METRICS", "Q4_METRICS"}; }
    else { contrasts = config.value("comparisons", std::vector<std::string>{"AA", "SQB", "SQ4"}); }
    require(kind=="overhead" || contrasts==std::vector<std::string>({"AA","SQB","SQ4"}) ||
        contrasts==std::vector<std::string>({"AA","SQB","SQ4","QBQ4"}), "frozen major contrasts are required");
    auto pair_routes = [](const std::string & name) -> std::pair<std::string,std::string> {
        if (name == "AA") { return {"S","S"}; } if (name == "SQB") { return {"S","QB"}; }
        if (name == "SQ4") { return {"S","Q4"}; } if (name == "QBQ4") { return {"QB","Q4"}; }
        if (name == "S_METRICS") { return {"S","S"}; } if (name == "QB_METRICS") { return {"QB","QB"}; }
        if (name == "Q4_METRICS") { return {"Q4","Q4"}; } throw std::runtime_error("unknown attribution contrast");
    };
    std::ofstream samples(out/"samples.csv"), token_times(out/"token-times.csv");
    samples << std::setprecision(17); token_times << std::setprecision(17);
    samples << "round,block,contrast,pair,order,a_route,b_route,a_metrics,b_metrics,a_wall_ms,b_wall_ms,b_minus_a_ms,a_reset_ms,b_reset_ms,a_prefill_ms,b_prefill_ms,a_begin_ns,a_end_ns,b_begin_ns,b_end_ns,a_thermal_before,a_thermal_after,b_thermal_before,b_thermal_after\n";
    token_times << "round,block,contrast,pair,order,path,route,metrics,step,begin_ns,end_ns,wall_ms\n";
    json schedule = json::array(); int block = 0;
    for (int round = 1; round <= rounds; ++round) {
        auto sequence = contrasts; std::rotate(sequence.begin(), sequence.begin()+((round-1)%sequence.size()), sequence.end());
        for (const auto & contrast : sequence) {
            ++block; const auto selected = pair_routes(contrast); schedule.push_back({{"round",round},{"block",block},{"contrast",contrast}});
            for (int pair = 0; pair < pairs; ++pair) {
                const bool ab = ((pair%4 == 0 || pair%4 == 3) != (round%2 == 0));
                const bool am = kind == "diagnostic", bm = kind != "formal";
                AttributionPath a,b;
                if (ab) { a = replay(selected.first,am,am); b = replay(selected.second,bm,bm); }
                else { b = replay(selected.second,bm,bm); a = replay(selected.first,am,am); }
                const double aw = double(a.end-a.begin)/1e6, bw = double(b.end-b.begin)/1e6;
                samples << round << ',' << block << ',' << contrast << ',' << pair << ',' << (ab?"AB":"BA") << ',' << a.route << ',' << b.route << ','
                    << am << ',' << bm << ',' << aw << ',' << bw << ',' << bw-aw << ',' << double(a.reset_ns)/1e6 << ',' << double(b.reset_ns)/1e6 << ','
                    << double(a.prefill_ns)/1e6 << ',' << double(b.prefill_ns)/1e6 << ',' << a.begin << ',' << a.end << ',' << b.begin << ',' << b.end << ','
                    << a.before["thermal"].get<std::string>() << ',' << a.after["thermal"].get<std::string>() << ',' << b.before["thermal"].get<std::string>() << ',' << b.after["thermal"].get<std::string>() << '\n';
                for (const auto & side : {std::make_pair("A",&a),std::make_pair("B",&b)}) {
                    const auto & path = *side.second;
                    paths.push_back({{"round",round},{"block",block},{"contrast",contrast},{"pair",pair},{"side",side.first},{"route",path.route},
                        {"metrics",path.metrics},{"final_logits_sha256",path.final_logits_sha256},{"before",path.before},{"after",path.after},
                        {"diagnostic_file",path.diagnostic_file.empty()?json(nullptr):json(path.diagnostic_file)}});
                    for (int step = 0; step < 32; ++step) {
                        token_times << round << ',' << block << ',' << contrast << ',' << pair << ',' << (ab?"AB":"BA") << ',' << side.first << ',' << path.route << ',' << path.metrics << ',' << step+1 << ','
                            << path.token_begin[step] << ',' << path.token_end[step] << ',' << double(path.token_end[step]-path.token_begin[step])/1e6 << '\n';
                    }
                }
            }
        }
    }
    require(bool(samples) && bool(token_times), "attribution CSV write failed"); samples.close(); token_times.close();
    require(hash_file(library) == library_sha, "Q library changed during attribution");
    timespec resolution; require(clock_getres(CLOCK_UPTIME_RAW,&resolution) == 0, "clock resolution unavailable");
    json runtime = {{"schema",1},{"operation","attribution"},{"attribution_kind",kind},{"backend","MTL0"},{"device",ggml_backend_dev_description(devices[0])},
        {"model_sha256",MODEL_SHA},{"tokens_sha256",hash_file(token_file)},{"prompt_ids",prompt},{"continuation_ids",continuation},
        {"activation_quantization",true},{"offloaded_layers",31},{"total_offload_layers",31},{"prefill","batched"},{"kv","F16"},
        {"settings",{{"n_ctx",512},{"batch",128},{"ubatch",128},{"threads",6},{"flash_attention",false}}},
        {"q_library_sha256",library_sha},{"q_entry",GGML_BITNET_Q_STRINGIFY(GGML_BITNET_Q4_ENTRY)},{"q_kernel_contract_version",GGML_BITNET_Q_CONFIG_VERSION},
        {"correctness_gate_sha256",config.at("correctness_gate_sha256")},{"one_model_and_context",true},{"KV_data_and_metadata_reset_each_path",true},
        {"prefix_restored_outside_wall",true},{"pipeline_warmup_replays_per_route",warmups},{"tokens_per_path",32},{"rounds",rounds},{"pairs",pairs},
        {"comparisons",contrasts},{"schedule",schedule},{"paths",paths},{"sample_pairs",rounds*pairs*int(contrasts.size())},
        {"token_rows",64*rounds*pairs*int(contrasts.size())},{"clock","CLOCK_UPTIME_RAW"},{"clock_resolution_ns",uint64_t(resolution.tv_sec)*1000000000ULL+uint64_t(resolution.tv_nsec)},
        {"wall_interval","32 fixed N1 llama_decode calls including existing synchronize per token; reset/prefill excluded"},
        {"capture_enabled",false},{"trace_enabled",false},{"backend_metrics_enabled_during_formal",false},{"new_synchronization_points",0},
        {"performance_measurement",true},{"formal_wall_campaign",kind == "formal"},{"critical_path_timestamps_diagnostic_only",true},
        {"execution_pass",true},{"finite_logits",true},{"parity_pass",nullptr},{"fp64_pass",nullptr},{"model_output_pass",nullptr},
        {"semantic_acceptance_pass",nullptr},{"official_source_model_acceptance",nullptr},{"strict_model_exit_code",2}};
    for (const char * key : {"q_variant","q_library_source_sha256","q_library_manifest_sha256","q_library_sdk","q_kernel_config_sha256","q_grid","q_threads"}) { runtime[key]=config.at(key); }
    write_json(out/"RUN.json",runtime);
    std::printf("ACTQUANT_ATTRIBUTION_COMPLETE kind=%s pairs=%d token_rows=%d synchronized=1 numerical_acceptance_unverified=1\n",kind.c_str(),rounds*pairs*int(contrasts.size()),64*rounds*pairs*int(contrasts.size()));
    std::fflush(stdout); return 0;
}
