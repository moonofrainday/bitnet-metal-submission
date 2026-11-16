# ActQuant＋Q4 联合验收：2026年11月16日

M2 Pro 和 iPhone 14 Pro/A16 真机上，两组固定 history 的 **ActQuant-ON 第0层 Q-only stock 兼容验收均通过，scoped exit0**。这是对同构建 stock-ON 的检查；可信 source-F32-ON 完整模型对照仍保留既有 **strict exit1**，本轮新增 A16 独立算子配对改善2.61%-2.68%，联合路线整段 decode 与 corpus/PPL 未测。

## 改了什么

- 固定 QVAC 为 `2db8b046d555316019ae75d1be90484adc8b9330`；本轮联合实现来自 `edbd109bc152decbbc7274b703d21bd6099bf98c`，基于 `e84732ced307d7dbb0dabe1f7c60c8c4b3bdc091`。早先 ActQuant-only 真机构建是 `4c249e3`，其派生证据整理于 `e84732c`，不能混用二进制身份。
- 新增默认关闭的 `LLAMA_BITNET_ACTQUANT_Q4=1`，只允许 ActQuant ON、第0层、单 token `Qcur-0`、每 SIMD4行及每 threadgroup1个 SIMD。实际 `actq_qkv_out-0` 必须是正确 epsilon 的 ACT_QUANT producer；其余节点、权重、类型、shape/stride、buffer/offset、边界与不重叠检查继续执行。C8、FFN、其他层及 guard-only 组合不授权。
- Q4 复用完整 stock helper 算术；它与独立原创4x2 shader 分开归属。函数名仍为 `tq2_0_matvec_poc_c8x2`，实际 source/build 哈希及 launch 表明它是 stock4x1。本轮不改变 shader 算法。
- 同一 GGUF SHA-256 为 `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2`。W 为 TQ2_0 `[2560,2560]`，实际 ActQuant 后 X 和 RoPE 前 Y 为 F32 `[2560,1]`。两路使用相同实际 W/X，FP64 独立解包同一 packing 与 block scales。

## 实测结果

两组 history 各使用27-token prefill＋32个固定 continuation IDs；context512、batch/ubatch128、6线程、F16 KV、flash attention 关闭。每路捕获包含 prefill 和全部32个 N1 Q，FP64 检查覆盖全部32次 decode；prefill 只声明 stock parity，不声明 FP64 通过。

| 设备 | History | Q4 dispatch/completion | 每路 FP64 超差数 | 最大绝对误差 | 对 stock ON 的残差/logits 向量 | Scoped exit |
|---|---|---|---|---|---|---|
| M2 Pro | primary | 32/32 | 0 / 81,920 | 1.267946573e-05 | 3 / 33，逐位一致 | 0 |
| M2 Pro | independent | 32/32 | 0 / 81,920 | 1.267150037e-05 | 3 / 33，逐位一致 | 0 |
| A16 | primary | 32/32 | 0 / 81,920 | 1.267946573e-05 | 3 / 33，逐位一致 | 0 |
| A16 | independent | 32/32 | 0 / 81,920 | 1.267150037e-05 | 3 / 33，逐位一致 | 0 |

每台设备/每组 history 的 stock 与 Q4 共33个 Q 捕获向量全部逐位一致。每路81,920个 decode 输出均在原容差 `abs(error) <= 2e-4 + 2e-5*abs(reference)` 内，超差与非有限值均为0。第1/16/32步3份残差，以及全部33份完整128256-logit 向量，与**同构建 stock-ON** 逐位一致。最大“误差/容差”为 primary0.060105211、independent0.062065417。另有独立 CPU-only 归档审计确认两组手机 history 的全部33份第0层输出逐位一致；严格 analyzer 仍保留原要求的3个残差检查点。

每组实际命中32次 guard、执行32次 custom Q，并确认32次 GPU completion；stock 路径 custom 为0。30次 prefill Q 和928次非目标 decode Q 保持 stock，总990次 Q encode 全部完成。stock/Q4 的 probe-on/plain 均保持33份 logits 一致。每组 missing-library 对照均为32次 guard、32次 stock fallback、0 custom，33份 logits 与 stock 一致。6项 CPU-only 启动守卫对照接受有效配置，拒绝缺少权限、不支持布局/层、混入 FFN、未启用 ActQuant；这些不是手机数值验证。

新签名 A16 构建的 ActQuant 前置样例20 cases/386,567 values 独立通过，bitwise、整数码和原容差失败均为0，operator strict0。新增 quantizer library 的实际 runtime getter 为 languageVersion262144、mathMode0/Safe、mathFloatingPointFunctions1/Precise；既有 stock library 的编译数学配置不变。每路/每组本轮 Q 输入 quantizer 捕获33对、151,040 values，在各自实际输入上与官方 F32 逐位一致。此局部捕获不是重新审计完整120节点图；已有完整审计单独保留。

## 新 post-ActQuant A16 算子配对

同签名构建复用实际手机 step16 W/X；W SHA-256 为 `b2dea7f42cacfd08ed780df979f6f28240026795b1f5c87ca8c16fc7da23dc58`，X SHA-256 为 `faca9f62eb02a1da3d87cf39df20816a4a15b79806995b4225df68b008e0cbcb`，Q4 metallib 与通过联合图验收的哈希一致。新 native qualification 后，同一已安装 App 再计时。

每路预热30次，每轮交替201对，共3轮；每个 command buffer 只包含一次 dispatch。计时关闭 capture、诊断日志和 tensor 读回。指标为 GPU 完成后的 `(GPUEndTime-GPUStartTime)*1e6`，即 GPU command-buffer elapsed 微秒，不是纯 shader cycles。固定 post-ActQuant X 的准备位于计时外；不包含 ActQuant、CPU编码、完整 decode 或 energy。

| 轮次 | Stock p50 / p95 us | Q4 p50 / p95 us | 逐对 Q4 减 stock 中位差 us | 逐对相对变化中位数 | Q4 胜出对数 |
|---|---|---|---:|---:|---:|
| 1 | 225.0000 / 228.7917 | 218.7917 / 223.1250 | -5.9167 | -2.6089% | 195/201 |
| 2 | 224.7500 / 228.8750 | 218.2917 / 223.9583 | -5.9583 | -2.6754% | 197/201 |
| 3 | 224.5000 / 228.6250 | 217.9583 / 222.5000 | -5.8750 | -2.6509% | 196/201 |

A/A 配对中位差为 -0.4167/-0.2500/-0.2083us。全部 AB/BA 分组和每轮全部三个时间分块都支持 Q4；对照组顺序轮换。整个九组 campaign 的18个热状态端点均 nominal，充电100%，低功耗关闭。约2.90秒不构成持续负载或独立热状态 cohort 证据。百分比为逐对计算后的中位数，不是两路边际中位数比值；A/A不作为通用修正量扣除。

结论限于**实际 ActQuant-ON 输入上的独立 Q 算子改善约2.61%-2.68%**。联合路线的整段 decode 与 corpus/PPL 未测。保留 harness 另有三组 C8/Q4 独立研究对照；C8 不获新图授权，也没有新增模型验收。完整派生指标、顺序/时间分组及原始文件摘要见 `SUMMARY.json`，raw CSV 留本地。

## 验收与性能边界

本轮 `parity_pass`、`fp64_pass`、`model_output_pass` 均为 true，但最后一项明确表示**同构建 stock-ON** 的已声明残差/logits 兼容，不表示微软源码执行已完全等价。本轮 joint analyzer 不重算可信源码完整模型对照、不声明全 Transformer FP64 或质量通过；此前 A16 的 source-derived eager F32 ON 完整对照已执行且 strict1，继续保留。

旧 Q4、same-stock guard 和 FFN 性能均是 ActQuant OFF。旧 Q4 单算子2.68%–3.11%收益及未观察到整段 decode 收益，不能移用于当前 ON 图。本轮已测新的 ActQuant-ON 输入单算子收益；**联合整段 decode 与 corpus/PPL 未测**，也不新增 Mac 性能数据。

## 代码、复跑和证据

路线修改位于 `src/models/bitnet.cpp`、`ggml/src/ggml-metal/ggml-metal-ops.cpp`；共用 native runner 为 `bitnet/actquant/runner.cpp`，shader 身份校验为 `q4_contract.py`，手机入口为 `run_phone.py`，独立验收为 `analyze_q4.py`。Shader 沿用 `bitnet/shaders/stock4x1.metal`。需用本轮源码新建签名构建，先完成20-case operator 前置验收，再用相同 tokens 和 build 分别运行 stock/Q4 capture、stock/Q4 plain、Q4 missing-library；完整可复制命令见本目录 `README.md` 及 `../../docs/BUILD_AND_RUN.md`。新 `prepare_q_fixture.py` 从实际 A16 post-ActQuant 捕获构造计时输入；`analyze_q_timing.py` 为可复跑 CPU-only 审计，已独立复算 exit0。

完整 joint analyzer 在全部必需检查通过时为 scoped exit0；完整证据下必需检查失败为1，缺失、格式错误或无效证据、`EvidenceError` 契约拒绝及 CLI 参数错误为2。Native/collection0 只代表执行与采集，不能替代独立数值验收；可信 source-F32-ON 对照仍为独立 strict1。

本地实验标识为 `actquant-q4-2026-11-16_084624+0900`，真机子目录为 `phone-2026-11-16_090500+0900`。本目录 `SUMMARY.json` 只保留派生指标、steps1/16/32 输入输出哈希及本地证据名称/摘要。原始 tensor、logits、CSV、capture、模型、签名产物和设备/签名信息均留本地。可信 reference 对照见 `../actquant-iphone-2026-11-16/`。
