# iPhone ActQuant 验证

修复后的 V2 ActQuant 已通过新一轮实体 iPhone 14 Pro / Apple A16 GPU sample-I/O：**20 cases、386,567 个值，全部输出与固定官方 F32 oracle 逐位一致，算子严格退出码 0**。这是 ActQuant 算子结果，不代表整模型数值验收。同一个 signed build 也已完成整模型诊断，但冻结的 reference-output 检查仍失败，严格退出码为 1。本轮不报告性能。

旧 baseline 和冻结容差不变：`abs(actual-reference) <= 2e-4 + 2e-5*abs(reference)`。公开 checkpoint、团队尚未确认的 checkpoint、原始 matvec PoC 与本次 activation quantization 修复属于不同范围。

## 已实测的算子结果

| 项目 | 结果 |
|---|---|
| 硬件 / 系统 | 实体 iPhone 14 Pro，Apple A16 GPU，iOS 27.2.1 |
| 构建 | Source commit `4c249e3e13dfce7b21bc2bab0912a1ff4807bc7e`；Xcode 27.2、iPhoneOS SDK 27.2 |
| Backend / 实际 shader | MTL0 / `kernel_act_quant_f32`；全部输出 buffer 为 MTL0 |
| 执行证据 | 20 条真实 encode、20 条同步完成标记；native/app/collection 退出码均为 0 |
| 独立官方 F32 对照 | 386,567 个有限值；逐位差异 0、ULP 0、原容差超差 0 |
| 恢复出的整数网格 code | 错误 0；不是直接读取内部整数寄存器 |
| 完整性 | 所有输入与 canary 未改变，输出 hash 全部核对 |
| 跨设备控制 | 全部 20 个实际输出文件与先前 Mac V2 MTL0 输出逐字节一致 |
| 算子独立 analyzer | `numerical_pass=true`，严格退出码 0 |
| 完整链路 FP64 / 模型输出 / 模型质量 | 不能由算子结果推出；native collection 的这些标志保持 null |

Fixture 覆盖零与 signed zero、epsilon 及相邻值、正负 half ties、seeded dense、真实 Q steps 1/16/32、两个真实图失败边界行，以及一个七元素除法顺序反例。宽度为 2560、6912，行数为 1/3/27，并包含七元素尾部样例。

iPhone 入口直接 include 同一个 `bitnet/actquant/runner.cpp`，调用真实 `ggml_act_quant` backend，没有用另一份独立算术实现代替。只读核查确认 signed build 记录的 1,067 个 framework source、25 个 package source、3 个 app binary 和 6 个 helper object hash 全部匹配当前本地文件。源码与此前清洁 Mac 算子构建匹配；不同平台的二进制本身不同。

## 算术与 runtime 证据

V2 使用 F32 `(1 / max(absmax(X), 1e-5)) * 127`，ties-to-even 舍入，clamp `[-128,127]`，再除以相同 scale。这对应移除官方 compile 装饰器后观察到的固定 Torch eager F32 profile，不代表所有 compiler 或 BF16 profile。直接使用 `127/denominator` 的 V1 与失败记录仍保存在本地。

A16 log 记录了新增 `act_quant` library 的实际编译选项 getter：`languageVersion=262144`、`mathMode=0` 对应 `MTLMathModeSafe`、`mathFloatingPointFunctions=1` 对应 Precise。实际 pipeline 的 SIMD width 为 32、threadgroup 上限为 1024；每 token 行一个 256-thread threadgroup。实际 buffer offset 与非重叠范围亦有日志。原 stock library 和 ROUND 行为未改。这些是执行/配置证据，不是硬件 counter 或性能数据。

## 整模型诊断结果

同一个 signed app/framework 在 Apple A16 GPU 上完成 teacher OFF、teacher ON capture、teacher ON plain、corpus OFF、corpus ON。每个 native/app 进程均退出 0，logits 为有限值，确认 31/31 offload。日志确认 F16 K/V cache 位于 MTL0，共 37.50 MiB，K/V 各 18.75 MiB。所有结束热状态均为 nominal；这不是持续负载或性能研究。

Teacher 使用相同 27 个 prompt IDs 和 32 个固定 continuation IDs，保存 33 份各含 128,256 个值的完整 logits。Corpus 使用相同 BOS 加 128 个固定 test-split targets，保存 128 份完整 logits。Context 512、batch/ubatch 128、6 个 CPU threads，关闭 flash attention。这是 teacher forcing，不是新增自由生成 token-parity 结论。

| 检查项 | 实测结果 |
|---|---|
| 真实图量化器执行 | 3,960 条实际 Metal encode 与 callback，模型同步完成标记存在 |
| 全阶段同输入官方 F32 对照 | 25,581,568 个值逐位一致，code/原容差错误为 0，覆盖通过 |
| Capture/plain 控制 | 全部 33 份 teacher 完整 logits 逐位一致 |
| 捕获的残差层输出对 F32 ON | 4,462,437 / 4,464,640 个值超过原容差 |
| 整模型输出检查 | 保留 `model_output_pass=false`，模型 analyzer 严格退出码 1 |
| 完整链路 FP64 / 代表性模型质量 | 未验证 `null`；不能用局部量化器检查代替 |

| 对 source-derived F32 ON 的比较 | 超差 / 总 logit 元素数 | RMSE | 最大绝对误差 | Greedy 一致位置数 |
|---|---:|---:|---:|---:|
| Teacher OFF | 4,229,647 / 4,232,448 | 0.251979214 | 1.438055038 | 31/33 |
| Teacher ON，capture/plain | 4,227,009 / 4,232,448 | 0.174339405 | 1.203764677 | 31/33 |
| Corpus OFF | 16,396,195 / 16,416,768 | 0.213242325 | 1.826897383 | 126/128 |
| Corpus ON | 16,384,271 / 16,416,768 | 0.124216825 | 1.136595760 | 126/128 |

五个 scoped model analyzer 都保留严格退出码 1。ON/OFF teacher/corpus 的完整 logits 文件也与此前保存的相同配置 Mac 文件逐位一致；这是跨设备数值控制，不是性能结果。OFF 对自身 F32 OFF reference 的 RMSE 较小（teacher `0.000724058`、corpus `0.000536808`），但仍不满足冻结规则。关闭必需的 quantization 不能作为正确性修复。

第 18 个 token 有一个具体失败证据：相同 27-token prompt 和 17 个共享 continuation tokens 后（input position 43），OFF 的 `computations`（83699）/`graphics`（14515）logit 为 22.3536472321 / 22.3587913513，分差 `-0.0051441193`；ON 为 22.3594665527 / 22.3896751404，分差 `-0.0302085876`。A16 两路都选择 `graphics`；已缓存的 source-derived F32 ON 选择 `computations`，分差 `+0.0949440002`。因此，局部 ActQuant 修复没有恢复这一参考 greedy 决策。这是固定输入的数值证据，不是语言质量结论。

128-target corpus PPL 为 **ON 19.127701918**、**OFF 19.088221109**；source-derived F32 ON/OFF 分别为 **19.129529529 / 19.088566108**。这些小规模 test-split smoke 分数使用统一 FP64 logsumexp 重算。分数接近不能证明全 logits 等价、代表性模型质量通过，或允许省略 ActQuant。固定续写的 teacher PPL 为 ON 1.826027356，source F32 ON 为 1.812893568；该 continuation 不是 held-out 质量评测。

量化器在各自输入上精确，整模型仍可能因为上游输入不同而偏离 reference。此前 Mac 控制已定位 prefill activation 缩窄、KV 精度与量化阈值不连续性，见 `bitnet/analysis/actquant-2026-11-16/README.md`；这些不是新做的 A16 定位实验。本轮手机证据确认了算子实际执行和捕获图内同输入量化器正确性。完整模型忠实性和产品质量验收仍未完成，不覆盖任何历史严格失败。

## 复跑

配置本地变量 `PY`、`SIGNED_BUILD`、`IPHONE_DEVICE_ID`、`GGUF`、`FIXTURE`、`TOKENS`、`REFERENCE_F32`、`REFERENCE_BF16`、`ARTIFACTS`。原始数据放在仓库外，每次使用新输出目录。手机保持连接与解锁。

```sh
"$PY" bitnet/actquant/run_phone.py --run --operation operator \
  --build-dir "$SIGNED_BUILD" --device "$IPHONE_DEVICE_ID" \
  --model "$GGUF" --fixture "$FIXTURE" --trace \
  --output "$ARTIFACTS/operator"
"$PY" bitnet/actquant/analyze_op.py --fixture "$FIXTURE" \
  --actual "$ARTIFACTS/operator/device/native" \
  --runtime "$ARTIFACTS/operator/device/native/RUN.json" --require-runtime \
  --output "$ARTIFACTS/operator-analysis"
"$PY" bitnet/actquant/run_phone.py --run --operation model \
  --build-dir "$SIGNED_BUILD" --device "$IPHONE_DEVICE_ID" \
  --model "$GGUF" --tokens "$TOKENS" --actquant on --workload teacher \
  --capture --trace --operator-run "$ARTIFACTS/operator" \
  --operator-analysis "$ARTIFACTS/operator-analysis/SUMMARY.json" \
  --output "$ARTIFACTS/teacher-on-capture"
"$PY" bitnet/actquant/compare_model.py \
  --run "$ARTIFACTS/teacher-on-capture/device/native" \
  --reference-f32 "$REFERENCE_F32" --reference-bf16 "$REFERENCE_BF16" \
  --output "$ARTIFACTS/teacher-on-analysis"
```

`run_phone.py` 返回 0 仅表示采集与身份检查完成，数值验收要看独立 analyzer。模型进程退出 0 不能覆盖 analyzer 的严格退出码 1 或缺失检查。代码入口见 `bitnet/actquant/README.md` 和 `bitnet/analysis/actquant-2026-11-16/README.md`。本目录仅包含派生分析与证据文件名/hash；原始 log、tensor、完整 logits、CSV、签名二进制以及 device/team 标识均留在本地。
