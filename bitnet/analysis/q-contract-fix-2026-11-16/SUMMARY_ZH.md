# Q kernel 线程契约与 pipeline cache 修复

本轮修复了可选 Q 路径中“host 配置与 shader 实际布局不绑定”和“同名 pipeline 忽略库路径变化”两个问题。新的 Mac 正确性验证通过，新的 iPhoneOS unsigned/signed 构建均退出0；**本修复二进制尚未安装或运行到手机**。

基线为 `6fa8468ec1a28bb47aa22484125ef1337dbb898b`，QVAC 固定 `2db8b046d555316019ae75d1be90484adc8b9330`。修复提交待填写，实测由构建清单中的源码、配置与产物哈希绑定。这里只提交分析，原始数据留在本地实验 `q-contract-fix-2026-11-16_123607+0900`。

## 改了什么

旧 ActQuant-OFF 底层路径可接受4x2/4x4，但 stock4x1 shader 内部固定1个 SIMD group；旧 Q4/C8 又共用一个函数名，cache 只按名字查找。

现在使用共享 canonical config、完整 descriptor 与独立入口：

| 候选 | Shader entry | 每 SIMD 行数 / 每 threadgroup 组数 | Grid / threads |
|---|---|---|---|
| Q4 | `tq2_0_matvec_stock4x1_v1` |4 /1|640,1,1 /32,1,1|
| C8 | `tq2_0_matvec_c8x2_v1` |8 /2|160,1,1 /32,2,1|

Host 从 descriptor 生成 launch，getter 再次检查；不支持的几何、variant 混用与伪造 descriptor 在 custom dispatch 前拒绝。Cache 先检查当前文件，key 包含 canonical path、entry、contract 与 inode/size/mtime/ctime 等身份；读取与加载前后检查稳定快照，缺失或替换文件不再简单复用旧同名 PSO。

Shader 点积算法未改。ActQuant ON 仍只允许第0层 Q4，并保留真实 quantizer producer、epsilon、shape/stride、buffer/offset、bounds/disjoint 等守卫。Stock 默认行为保留；FFN 算术和 ABI 不变，共享外部库加载器增加文件身份检查。

## 实测结果

| 项目 | 结果 | 边界 |
|---|---|---|
| 共用生产 helper 的 CPU 测试 |10,347项检查通过|布局、完整覆盖、错误 descriptor、路径别名和真实文件变化|
| 离线 Metal / schema |20个错误编译配置被拒绝；12个 schema 测试通过|不等于 GPU 执行|
| 首次实际 getter 测试 |第5项 valid Q4 load 失败，退出1|原因仍为 **UNEXPLAINED**，保留失败|
| 诊断版及两次独立复跑 |三组均18/18通过|实际 Mac PSO/cache；没有目标 Q dispatch|
| 真实 GGML 算子图 |15个case、38,400个输出；FP64/非有限失败均0；canary、输入未改变|覆盖错误几何、错库、同路径替换、删除与恢复|
| Metal backend |TQ2_0 59/59；ActQuant 16/16|M2 Pro MTL0 正确性|
| 两组真实模型固定 history |各32/32 Q4 dispatch/GPU完成；九项 scoped gate 全通过，退出0|ActQuant ON，第0层 Q-only|
| 新 iPhoneOS 构建 |unsigned0、signed0|未安装、未执行真机|

首次失败只有笼统错误信息，无法确定内部失败条件。之后生产代码仅增加 WARN 诊断字段，诊断版与两次复跑通过，独立加载与 CPU 文件读取探针也通过。**不能据此说“日志修复了根因”**，首次失败继续保留。

两组 history 中，每条路线的全部81,920个 decode 输出均通过原 FP64 容差：

`abs(error) <= 2e-4 + 2e-5*abs(reference)`

最大绝对误差分别为1.267946573e-5和1.267150037e-5。Stock/Q4 的 W/X 相同，全部33次 Q 读回和33份完整 logits 逐位一致；正式 residual gate 抽查steps1/16/32。每组实际32次 custom，30次 prefill Q及928次非目标 decode Q走 stock，990次 Q均完成；probe中立性与32次缺失库回退通过。Prefill只证明 parity，没有记成新的 FP64 通过。

独立 CPU 归档审计额外核对了全部33份 `l_out-0`、33组 Q X/Y、33组 quantizer 输入/输出及33份完整 logits，均与 stock 逐位一致；新 stock ON 与此前保留的 stock ON 也一致。独立 logical-K 解包和 FP64 重算复现全部32步通过；每条路线的局部 ActQuant 为33次、151,040个值，实际权重文件也单独校验哈希。

独立审计首次 CPU oracle 曾先转成 int8 再反量化，丢失 negative-zero 符号，造成逐位比较误报。该失败尝试保留，修正仅恢复官方“浮点舍入码继续反量化”的语义，没有修改 native/shader 或原容差。这是分析工具修正，不是运行时精度修复。

独立算子图中的 C8 对 stock 有2,533行逐位差异，但局部 FP64 通过，最大误差5.761347e-6；stock/Q4 为8.837353e-6。这不等于 C8 已通过 ActQuant-ON 模型图兼容性。

## 尚未证明什么

微软可信参考的完整模型比较仍严格退出1，旧 FFN/full-chain 失败不覆盖。新 iPhone 二进制的 cache、dispatch、数值、性能要晚上重新验证。

旧 A16 2.61%-2.68%的算子收益属于上一版二进制，本轮不能沿用为修复后性能。没有新的 Mac 性能数据，也没有新手机性能或整段 decode 性能。新增 realpath/stat/lock 会产生尚未测量的 CPU 编码工作；孤立 GPU shader 时间不包含它。

文件 metadata 用于 cache 失效判断，不是密码学认证，也不是对并发/恶意修改文件的原子锁。运行中应冻结库文件，保留源码、构建和metallib哈希。旧身份的 PSO保留到 backend 销毁，研究 campaign间可重建backend。

新 iPhoneOS framework SHA-256：`2895f30ad294b47f6eb42dfc419277afcd7738e1151c969af43cc002610495af`。
新 Q4 metallib SHA-256：`abc05c6f28fa186dc9587d70dc7ec137903c385bcbd8203202ba95780f14cab6`。

## 复跑与下一步

先做 CPU 检查：

```bash
"$PY" bitnet/tests/run_q_contract_cpu.py --output "$ARTIFACTS/new-cpu-contract"
```

完整构建、实际 Mac cache、模型回放和晚间 iPhone 入口见 [英文修复说明](../../docs/Q_KERNEL_CONTRACT_FIX.md)。使用全新输出目录。旧记录分析必须显式指定 `--contract-profile legacy-shared-v0 --legacy-shader-build "$OLD_SHADER_BUILD_JSON"`；它只用于分析，不能让新 runtime 调用旧同名 entry。

晚间统一入口为 `bitnet/tests/run_q_fix_phone.py`：`--plan` 只生成15个串行阶段，已完成白天的无设备调用检查；`--run` 才启动A16实际18项cache/entry检查、同构建20-case ActQuant前置验证及两组Q4/stock图正确性与回退。统一入口不含性能阶段；正确性通过后再测性能。原始日志、tensor、CSV、模型与签名产物留本地；`SUMMARY.json`仅保存分析与证据哈希。

另用随提交提供的 `build_q_contract.py` 在全新目录实际重建：1,070 份 native 输入哈希与上述验收源码一致，未复用旧 dylib；随后用新构建跑实际 backend cache 入口，18/18 通过、退出 0。此项是额外的复现入口验证，不是新的模型或性能结论。

源码修复提交：`8ebcdb78680829c5453e0c7c72770c0b8e80225a`。编译发生在提交前，二进制内嵌 Git 标签仍为旧 `6fa8468`；已逐文件确认1,070个 Mac native输入、1,069个iOS framework输入和29个iOS包输入均与修复源码相同。补丁为 `bitnet/patches/q-kernel-contract-fix.patch`，基于旧提交的增量，不应再对本分支重复应用。
