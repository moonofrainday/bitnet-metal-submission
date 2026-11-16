# 中文版审阅入口

这是2026年11月16日的实测更新版。中英文PDF和可编辑Word均保留流程图、关键英文术语与历史手机数字，并补入当前14a9f103硬化包的真实A16结果。报告生成不运行推理；新增实测来自单独保存的真机实验。

中英文均为28页PDF和28页渲染后的Word；全部最终页面均已逐页查看，或与已查看页面进行精确图片一致性核对。

## 文件与重点

- `BitNet_Mobile_GPU_Proposal_ZH.pdf`：中文审阅版。
- `BitNet_Mobile_GPU_Proposal_ZH.docx`：可编辑中文 Word。
- `PROPOSAL_ZH.md`：中文源稿，与英文保持 16 节对应。
- `TRANSLATION_CHECK.json`：数值、表格、代码、URL 与引用一致性检查。
- `PENDING_A16_RESULTS.json`：结果登记；已测项填写，未核实项仍为null，不能当作零或通过。
- [今晚手机测试顺序与命令](../docs/IPHONE_TEST_PLAN_2026-11-16.md)。

英文 PDF、Word、源稿继续保留，并与本版同步更新。

第 9 节新增 M2 Pro 归因补充：已选 Q4 的局部收益成立，但当前单层单投影接入没有整段 decode 收益。完整 guard 后调用同一 stock pipeline 的对照，定位了 warm 文件身份检查这一显著 host 区间；仍不能完整拆分 CPU/GPU/热漂移，也未取得正常关键路径占比。独立 encoder 时间戳探针改变调度，未进入正常或手机构建。

第10节保留旧A16数据并填写当前包：两次18/18契约、20项ActQuant、两组history各32次Q和33份logits、同进程stock/QB/Q4、独立算子、CPU诊断及整段decode。Fair和持续模型负载后的Serious算子结果分开。Q4局部快约2.18%–2.78%，但当前包QB/Q4整段decode均变慢。snapshot/key是已测host热点，尚不能拆清CPU/GPU/热漂移。[本轮分析与逐轮数据](../analysis/iphone-qcur0-attribution-2026-11-16/README.md)另存，原始数据只留本地。

报告明确解释：本作业交付一个 Transformer 层内矩阵 kernel、独立 sample-I/O harness、源码脚本和技术方案 PDF；没有要求自研整个 Transformer block、最低加速比例或整模型加速。系统 profiling 用于量化收益空间和限制结论，不能把“可能存在其他瓶颈”写成“已证明瓶颈在别处”。

原始自写 PoC、复用 stock helper 的 Q4 和 FFN 性能探索保持独立归属。可信模型 reference 的已观测失败继续保留 strict exit 1，不会因为局部 FP64、stock parity 或文档整理而改判。团队 checkpoint、代表性质评测、Android、energy 和持续负载仍按各自范围记录。

## 生成与复核

使用 Python 3.10+ 和 `requirements-docs.txt`；本轮实际使用 Codex bundled Python、python-docx、ReportLab 和 Pillow。保留旧版本于 Git 历史。

```sh
"$DOC_PY" bitnet/proposal/render_proposal.py --preserve-figure --output "$LOCAL_REPORT/english"
"$DOC_PY" bitnet/proposal/render_proposal_zh.py --output "$LOCAL_REPORT/chinese"
"$DOC_PY" bitnet/proposal/check_translation.py --output "$NEW_TRANSLATION_CHECK"
```

生成结果批准、manifest 哈希更新后再运行最后一条。中文 renderer 不覆盖英文或流程图。字体使用本机已安装的 Arial Unicode MS 与 Heiti SC，不随仓库分发。Headless Word 渲染需要本地 Fontconfig 明确加载中文字体；本轮已检查缺字、裁切、表格和分页。原始 QA 图片和配置留本地。

源码、补丁、派生分析和最终报告上传私有仓库；原始 tensor/logit/CSV/trace、模型和签名产物留本地。报告发布不等于向题目团队提交。正式交付前仍需作者确认待测项与已保留限制。
