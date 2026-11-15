# 公开 BitNet 2B ActQuant 修复：Mac 数值验证

本记录针对微软公开 checkpoint，不代表团队尚未确认规则的 checkpoint。旧 QVAC 路径、历史严格失败和原容差全部保留：`abs(actual-reference) <= 2e-4 + 2e-5*abs(reference)`。本轮没有 Mac 性能数据，修复版本也**尚未在实体 iPhone 上验证**。

## 实现范围与参考精度

通过独立开关 `LLAMA_BITNET_ACTQUANT=1`，在公开模型每层七个投影前补入 per-token activation fake quantization。每层四组共享输入分别覆盖 Q/K/V、attention output、gate/up、down；30 层共有 120 个 ActQuant 节点，覆盖 210 个线性投影。共享 embedding 的 `lm_head` 不变，自写 Q/FFN 路径关闭。开关 OFF 保留旧图；切换开关必须创建新进程/context，不能复用不包含该开关状态的图缓存。

V2 对每个连续 F32 token 行执行：

```
d = max(max(abs(X)), float32(1e-5))
r = float32(1 / d)
s = float32(r * 127)
q = clamp(round_to_nearest_ties_even(float32(X * s)), -128, 127)
Y = float32(q / s)
```

输出 `Y` 是反量化后的 F32，并没有新增 INT8 矩阵乘。本次对齐的是固定官方函数移除 `torch.compile` 装饰器后的 **Torch eager F32 执行 profile**；不能外推为其他编译方式或 BF16 profile 的逐位等价。V1 使用 `s=float32(127/d)`，两者实数公式相同，浮点求值路径不同。

## V1 为什么失败，V2 修复了什么

| 检查范围 | V1：直接除法 | V2：先倒数再乘 127 |
|---|---:|---:|
| 初始 fixture，每个 backend | 17 cases / 377,088 values | 包含在扩充套件中 |
| 扩充 fixture，每个 backend | 20 cases / 386,567 values | 20 cases / 386,567 values |
| 扩充套件恢复出的整数 code 错误数 | 4 | 0 |
| 扩充套件原容差失败数 | 4 | 0 |
| 扩充套件算子严格退出码 | 1 | 0 |
| 扩充套件对官方 F32 的逐位差异数 | 65,505 | 0 |
| 扩充套件最大 value ULP | 8,388,607 | 0 |

以上 CPU 和 MTL0 均分别实跑。初始 377,088 个值的 CPU/Metal 原始输出逐位一致。扩充套件增加两个真实图失败行和一个七元素边界输入；V1 在两个真实行各错一个 code，构造输入错两个；V2 全部输出与官方 F32 逐位一致，没有非有限输出。整数 code 是用输出和官方 scale 在 F64 中恢复的网格检查，并非直接读取 GPU 的内部整数寄存器。

初始 17 cases 虽然 code 与原容差均通过，却有 61,964 个值与官方 F32 不同，最多 2 ULP。独立 CPU 对照确认：直接除法能逐位复现全部 native 输出，先倒数再乘能逐位复现全部官方输出。随后真实图在 25,581,568 个捕获值中发现两个错误 code：第 18 层 prefill 的 attention-output 输入，以及第 25 层 decode 的 FFN-down 输入。对应“预期/实际”code 为 `-25/-24` 和 `13/12`。因此，小套件通过不能替代真实输入覆盖。

V2 只改变 scale 求值。最大值归约、舍入、clamp、输出除法、guard 和 launch 均不变。V1 源码、二进制和失败仍保存在本地。离线 AIR 确认倒数除法与乘法保留；实际 runtime 编译选项 getter 确认**仅新增 library**使用 safe math 和 precise floating-point functions。Stock library 和已有 ROUND 行为未改。

## 分开报告的验收结果

| 检查项 | 结果与边界 |
|---|---|
| V2 独立算子执行与同输入官方对照 | PASS：CPU、MTL0 各 20 cases / 386,567 values，严格退出码 0 |
| Backend 回归 | MTL0 ACT_QUANT 16/16，原 TQ2_0 MUL_MAT 59/59，已含 V2 复跑 |
| V2 真实图量化与覆盖 | PASS：batched 3,960 次捕获 / 25,581,568 个值，sequential 7,080 / 25,827,840；每个实际输入的输出均逐位等于官方 F32，code 与原容差错误均为 0 |
| V2 层输出、teacher-forced 完整 logits | FAIL：两种 prefill 均为 `model_output_pass=false`，scoped strict exit 1 |
| V2 128-target test-split PPL smoke | 实测 19.127701918，官方 F32 ON 为 19.129529529；质量验收仍为 null |
| 完整链路 FP64 / 产品质量 | 未验证 `null`；旧完整 FFN 的失败继续保留 |
| 实体 iPhone | 未验证；新 iPhoneOS 无签名构建成功，尚未安装或执行手机 GPU |

两种 prefill 的实际 Metal ActQuant encode 记录分别为 3,960/7,080 条，均确认 31/31 层 offload。V2 teacher 的 capture/plain 运行在全部 33 个位置的完整 logits 逐位一致；原版本与 V2 的开关 OFF teacher 输出也在全部 33 个位置逐位一致。

| V2 对 F32 ON 的比较 | 超差 / 总 logit 元素数 | RMSE | Greedy 一致位置数 | 严格退出码 |
|---|---:|---:|---:|---:|
| Teacher，batched prefill | 4,227,009 / 4,232,448 | 0.174339405 | 31/33 | 1 |
| Teacher，sequential prefill | 4,225,157 / 4,232,448 | 0.175027301 | 30/33 | 1 |
| Teacher，sequential F32-KV 控制 | 4,223,875 / 4,232,448 | 0.146621968 | 31/33 | 1 |
| 128-target corpus | 16,384,271 / 16,416,768 | 0.124216825 | 126/128 | 1 |

残差层输出同样未通过原容差：batched 为 4,462,437/4,464,640 个元素，sequential 为 4,529,005/4,531,200。固定续写的 teacher-forced PPL 分别为 1.826027356（batched）、1.822546944（sequential），官方 F32 ON 为 1.812893568；它不是 held-out 质量评测。另一个 128-target test-split PPL 只是小规模 smoke，接近参考不能证明全 logits 等价或产品质量通过。Sequential prefill 保持局部 ActQuant 精确，但没有消除整模型分歧，因此不能仅归因于 batched prefill。

整模型使用相同的 27 个 prompt token 和 32 个 continuation token，F16 KV、context 512、batch/ubatch 128、6 个 CPU threads。目标参考为 F32 ON，F32 OFF 与 BF16 另列。剩余执行差异可能包括 CPU Q8_K 重打包、TQ2_0 折叠 scale、矩阵乘与归约顺序、F16 KV，以及官方源代码的无缓存执行。RMSE 降低、PPL 更近或 greedy token 一致，均不能覆盖原逐元素失败。极端幅度/subnormal F32 不在本 fixture 验证范围内。

## 剩余分歧已定位到量化器之外

独立的单层官方源码 probe 在全部 59 个位置、151,040 个值上逐位复现已保存的 F32 ON 第 0 层 cache。它是经过检查的参考 probe，不是新增完整模型实现。两组控制区分了剩余传播分歧与 V2 量化器缺陷。

- **Batched prefill Q：**固定捕获的 W/Y，只将 reference X 从 F32 改成 F16 round-trip，同输入 FP64 超差从 11,168/69,120 降到 0。Sequential N=1 Q 在全部 59 个位置的同输入 FP64 检查通过。这定位了初始 prefill Q 的 activation 精度路径差异，尚未修复整模型。
- **Sequential 传播：**F16 KV 时，attention-output 量化前的 X 相对 reference 最大差为 `3.380700946e-5`、原容差失败为 0；量化后跨 reference 的输出有 676 个超差值，position 0 首先出现 7 个。改用 F32 KV 后，量化前 X 最大差降至 `4.202593118e-7`；position 0 的首个失败移到 FFN-down effective input，一个值相差 `0.070197657`。量化阈值附近的不同输入可以产生不同 code，同时量化器仍在各自输入上完全正确。FFN-down 第 1808 列仅 `4.768371582e-7` 的输入差异就产生 code 1 与 2，而 V2 在自身输入上仍逐位等于官方量化器。完整 F32-KV 控制有 7,080 次捕获、25,827,840 个局部量化输出逐位一致，但全 logits 严格退出码仍为 1，teacher-forced PPL 为 1.815986257。这是独立精度控制，不改变原 F16-KV 验收配置。

下一项应针对这些边界的上游算术/精度继续控制，分开保留同输入检查与误差传播检查。新 iPhoneOS arm64 无签名构建成功，但尚未安装，也没有手机 GPU 执行，实体 iPhone 状态仍为未验证。

可复用实现补丁为 `bitnet/patches/public-2b-actquant.patch`。另一次通过 operator 入口的全新构建也通过 CPU、MTL0 各 20 cases / 386,567 values、ACT_QUANT 16/16 和 TQ2_0 MUL_MAT 59/59。

## 代码入口

- `src/models/bitnet.cpp` 的 `llama_model_bitnet::graph`：公开模型 guard、开关与插入位置。
- `ggml/include/ggml.h`、`ggml/src/ggml.c` 的 `ggml_act_quant`：追加 op/API 与独立 F32 输出。
- `ggml/src/ggml-cpu/ops.cpp` 的 `ggml_compute_forward_act_quant`：CPU per-token maximum、scale 与 ties-even 舍入。
- `ggml/src/ggml-metal/kernels/act_quant.metal` 的 `kernel_act_quant_f32`：每 token 一个 threadgroup。
- `ggml/src/ggml-metal/ggml-metal-ops.cpp` 的 `ggml_metal_op_act_quant`：实际 buffer、边界、offset、非重叠输出、绑定与 launch。
- `ggml/src/ggml-metal/ggml-metal-device.m`：新 library 的精度选项及 getter 记录。
- `tests/test-backend-ops.cpp` 的 `test_act_quant`：16 个跨 backend case。
- `bitnet/actquant/{prepare_fixtures,analyze_op,compare_model,run_mac}.py`：官方 oracle、独立检查和清洁构建入口。

## 复跑与证据

在本地配置 `PY`、`ARTIFACTS`、`REAL_Q_FIXTURE`、`GGUF`、`TOKENS`、`REFERENCE_F32`、`REFERENCE_BF16`、`DEVELOPER_DIR`。`ARTIFACTS` 必须在仓库外，运行输出目录须为新目录。

```sh
"$PY" bitnet/actquant/test_analyze_op.py
"$PY" bitnet/actquant/test_model_contract.py
"$PY" bitnet/actquant/prepare_fixtures.py \
  --official-source bitnet/reference/official-bitnet-096f25ae.py \
  --real-fixture "$REAL_Q_FIXTURE" --output "$ARTIFACTS/fixtures"
"$PY" bitnet/actquant/run_mac.py --mode full \
  --output "$ARTIFACTS/new-run" --fixture "$ARTIFACTS/fixtures" \
  --model "$GGUF" --tokens "$TOKENS" \
  --reference-f32 "$REFERENCE_F32" --reference-bf16 "$REFERENCE_BF16" \
  --developer-dir "$DEVELOPER_DIR"
```

要重建 20 cases，fixture 准备还需传入本地保留的 `--boundary-capture`、`--boundary-failures`、`--counterexample`。算子 analyzer 严格退出码：0 为通过、1 为实测失败、2 为证据缺失、3 为证据格式错误。模型另有 scoped strict exit。`--exit-mode processing` 的处理成功不能当作数值验收。

`SUMMARY.json` 只收录派生汇总和证据文件名/hash。证据索引包括 `boundary-v1-{CPU,MTL0}-analysis/SUMMARY.json`、`operator-v2-{CPU,MTL0}-analysis/SUMMARY.json`、`ACTQUANT_FIRST_LOCAL_FAILURES.json`、`backend-actquant-v2.log`、`backend-tq2-v2.log` 及 V1/V2 source manifest。本目录不包含原始 log、tensor、完整 logits、CSV、个人设备标识或本地绝对路径。
