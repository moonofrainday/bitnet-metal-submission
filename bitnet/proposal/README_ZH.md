# 中文版审阅入口

本中文版与英文 proposal 的 16 节编号一一对应，供逐节检查。正文、表格和结论均按英文原稿翻译，没有新增推理实测，也没有更改原有验收阈值。英文 Markdown、PDF、Word 及原流程图保持原字节不变。

## 文件

- `BitNet_Mobile_GPU_Proposal_ZH.pdf`：中文版 PDF，16 页。
- `BitNet_Mobile_GPU_Proposal_ZH.docx`：可编辑中文版 Word，已渲染检查 16 页。
- `PROPOSAL_ZH.md`：可编辑中文源稿，与 `PROPOSAL.md` 按节对应。
- `TRANSLATION_CHECK.json`：表格、数值、公式、代码、引用与英文保留检查。
- `MANIFEST_ZH.json`：中文产物哈希、英文源稿身份与排版验收范围。

原英文文件仍为 `BitNet_Mobile_GPU_Proposal.pdf`、`BitNet_Mobile_GPU_Proposal.docx` 和 `PROPOSAL.md`。流程图继续保留英文标签，图像与英文原图相同。

## 检查重点

建议先检查第 4 节的 PoC 范围、第 7 节验收语义、第 10 节手机性能和第 12 节成功标准。原始 4x2 PoC、C8 研究版与复用 stock helper 的 Q4 分开归属；局部 FP64、stock parity 和 model_output_pass 是不同检查。Q4 的单算子收益不代表整模型加速，既有 FFN 失败与未验证项继续保留。

关键术语保留原文，包括 stock、PoC、Qcur-0、TQ2_0、F32、FP64 reference、dispatch、SIMD group、threadgroup、prefill、decode、GPU command-buffer elapsed time、p50/p95、paired difference、parity_pass、fp64_pass 和 model_output_pass。

## 复算与生成

```sh
python3 check_translation.py --output /absolute/path/to/new-translation-check.json
python3 render_proposal_zh.py --output /absolute/path/to/local-chinese-build
```

检查工具使用 Python 标准库。生成工具复用英文 parser 和表格宽度，但不重建或覆盖英文文件，也不刷新原流程图。生成依赖见 `requirements-docs.txt`；本机使用 Arial Unicode MS 和 Heiti SC，字体文件不随仓库分发。PDF 可通过 `--cjk-font`、`--cjk-bold-font` 及 collection index 参数选择已安装的兼容 TTF/TTC 字体。

Word 的 headless 排版检查需要渲染器加载可用中文字体；本次使用明确的 Fontconfig 配置、中文字体映射和逐页图像检查，确认中文及流程图正常显示。文字使用固定行距，包含图片的段落保留自动行高。原始 QA 图片、中间失败记录及本地字体缓存均不上传。

中文版仅用于作者审阅。模型实验仍属于公开 checkpoint，团队模型、Android、能耗及原始 PoC 精确版本的手机补测等限制与英文保持一致。
