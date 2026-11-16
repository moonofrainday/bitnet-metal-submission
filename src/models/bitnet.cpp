#include "models.h"
#include "llama-impl.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

void llama_model_bitnet::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    hparams.llm_ffn_op = LLM_FFN_SILU;
    std::string hidden_act;
    if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false)) {
        if (hidden_act == "relu2") {
            hparams.llm_ffn_op = LLM_FFN_RELU_SQR;
        } else if (hidden_act != "silu") {
            throw std::runtime_error("unsupported BitNet hidden activation: " + hidden_act);
        }
    }

    switch (hparams.n_layer()) {
        case 26: type = LLM_TYPE_3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_bitnet::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    // bitnet has tied embeddings; duplicate tok_embd so the lm_head weight is
    // classified as an output tensor and can be offloaded
    output      = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sub_norm = create_tensor(tn(LLM_TENSOR_ATTN_SUB_NORM, "weight", i), {n_embd}, 0);

        layer.wq       = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd}, 0);
        layer.wq_s     = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
        layer.wk       = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
        layer.wk_s     = create_tensor(tn(LLM_TENSOR_ATTN_K,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
        layer.wv       = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
        layer.wv_s     = create_tensor(tn(LLM_TENSOR_ATTN_V,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
        layer.wo       = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.wo_s     = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale",  i), {1}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm     = create_tensor(tn(LLM_TENSOR_FFN_NORM,     "weight", i), {n_embd}, 0);
        layer.ffn_sub_norm = create_tensor(tn(LLM_TENSOR_FFN_SUB_NORM, "weight", i), {n_ff}, 0);

        layer.ffn_gate       = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_gate_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "scale",  i), {1}, TENSOR_NOT_REQUIRED);
        layer.ffn_down       = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale",  i), {1}, TENSOR_NOT_REQUIRED);
        layer.ffn_up         = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_s   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "scale",  i), {1}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_bitnet::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_bitnet::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const char * act_quant_env = std::getenv("LLAMA_BITNET_ACTQUANT");
    if (act_quant_env && std::strcmp(act_quant_env, "0") != 0 && std::strcmp(act_quant_env, "1") != 0) {
        throw std::runtime_error("LLAMA_BITNET_ACTQUANT must be 0 or 1");
    }
    const bool act_quant_enabled = act_quant_env && std::strcmp(act_quant_env, "1") == 0;
    const char * act_quant_q4_env = std::getenv("LLAMA_BITNET_ACTQUANT_Q4");
    if (act_quant_q4_env && std::strcmp(act_quant_q4_env, "0") != 0 && std::strcmp(act_quant_q4_env, "1") != 0) {
        throw std::runtime_error("LLAMA_BITNET_ACTQUANT_Q4 must be 0 or 1");
    }
    const bool act_quant_q4 = act_quant_q4_env && std::strcmp(act_quant_q4_env, "1") == 0;
    if (act_quant_q4 && !act_quant_enabled) {
        throw std::runtime_error("LLAMA_BITNET_ACTQUANT_Q4 requires LLAMA_BITNET_ACTQUANT=1");
    }
    if (act_quant_enabled) {
        if (n_layer != 30 || hparams.n_embd != 2560 || model.vocab.n_tokens() != 128256 ||
                hparams.llm_ffn_op != LLM_FFN_RELU_SQR || cparams.training || !loras->empty()) {
            throw std::runtime_error("experimental ActQuant requires the public 2B inference profile without LoRA");
        }
        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];
            if (hparams.n_ff(il) != 6912 || hparams.n_head(il) != 20 || hparams.n_head_kv(il) != 5) {
                throw std::runtime_error("experimental ActQuant has an unsupported public 2B layer configuration");
            }
            const ggml_tensor * weights[] = {
                layer.wq, layer.wk, layer.wv, layer.wo, layer.ffn_gate, layer.ffn_up, layer.ffn_down,
            };
            const int64_t columns[] = { 2560, 2560, 2560, 2560, 2560, 2560, 6912 };
            const int64_t rows[] = { 2560, 640, 640, 2560, 6912, 6912, 2560 };
            for (size_t i = 0; i < sizeof(weights)/sizeof(weights[0]); ++i) {
                const auto * w = weights[i];
                if (!w || w->type != GGML_TYPE_TQ2_0 || w->ne[0] != columns[i] || w->ne[1] != rows[i] ||
                        w->ne[2] != 1 || w->ne[3] != 1) {
                    throw std::runtime_error("experimental ActQuant requires the audited public 2B TQ2_0 weights");
                }
            }
        }
        for (const char * name : { "GGML_METAL_BITNET_Q_GUARD_ONLY",
                "GGML_METAL_BITNET_FFN", "GGML_METAL_BITNET_FFN_MECHANISM" }) {
            const char * value = std::getenv(name);
            if (value && value[0] != '\0' && std::strcmp(value, "0") != 0) {
                throw std::runtime_error("experimental ActQuant cannot be combined with custom Q or FFN routes");
            }
        }
        const char * custom_q = std::getenv("GGML_METAL_BITNET_POC");
        if (custom_q && custom_q[0] != '\0' && std::strcmp(custom_q, "0") != 0) {
            const char * rows = std::getenv("GGML_METAL_BITNET_POC_ROWS_PER_SIMD");
            const char * groups = std::getenv("GGML_METAL_BITNET_POC_SIMD_PER_GROUP");
            const char * layers = std::getenv("GGML_METAL_BITNET_LAYERS");
            const char * ablation = std::getenv("GGML_METAL_BITNET_FFN_ABLATION");
            if (!act_quant_q4 || std::strcmp(custom_q, "1") != 0 || !rows || std::strcmp(rows, "4") != 0 ||
                    !groups || std::strcmp(groups, "1") != 0 || !layers || std::strcmp(layers, "0") != 0 ||
                    (ablation && ablation[0] != '\0' && std::strcmp(ablation, "0") != 0)) {
                throw std::runtime_error("ActQuant custom Q requires explicit Q4 permission, rows=4, groups=1, layer=0 and no FFN ablation");
            }
            static std::once_flag q4_log_once;
            std::call_once(q4_log_once, []() {
                LLAMA_LOG_INFO("BitNet ActQuant Q4: allow only layer-0 single-token Qcur-0, stock-helper rows=4/groups=1; "
                        "runtime quantizer, layout and buffer guards remain required.\n");
            });
        }
        static std::once_flag log_once;
        std::call_once(log_once, []() {
            LLAMA_LOG_INFO("BitNet ActQuant: experimental public 2B F32 profile; per-token absmax INT8, "
                    "eps=1e-5, round-to-even, 120 shared stages for 210 projections; lm_head unchanged. "
                    "Caller must verify checkpoint identity.\n");
        });
    }
    auto act_quant = [&](ggml_tensor * input, const char * output_name, int il) {
        if (!act_quant_enabled) {
            return input;
        }
        if (input->type != GGML_TYPE_F32) {
            throw std::runtime_error("experimental ActQuant requires F32 activations");
        }
        if (!ggml_is_contiguous(input)) {
            input = ggml_cont(ctx0, input);
        }
        ggml_tensor * output = ggml_act_quant(ctx0, input, 1e-5f);
        cb(output, output_name, il);
        return output;
    };

    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);
        cur = act_quant(cur, "actq_qkv_out", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    NULL, NULL, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);

            cur = build_norm(cur,
                    model.layers[il].attn_sub_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "attn_sub_norm", il);
            cur = act_quant(cur, "actq_attn_out_out", il);

            cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
            if (model.layers[il].wo_b) {
                cur = ggml_add(ctx0, cur, model.layers[il].wo_b);
            }
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward forward
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);
        cur = act_quant(cur, "actq_ffn_up_gate_out", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                NULL,                      NULL, NULL,
                NULL,
                hparams.llm_ffn_op, LLM_FFN_PAR, il);
        cb(cur, "ffn_sub_out", il);

        cur = build_norm(cur,
                model.layers[il].ffn_sub_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_sub_norm", il);
        cur = act_quant(cur, "actq_ffn_down_out", il);

        cur = build_lora_mm(model.layers[il].ffn_down, cur, model.layers[il].ffn_down_s);
        cb(cur, "ffn_down", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
