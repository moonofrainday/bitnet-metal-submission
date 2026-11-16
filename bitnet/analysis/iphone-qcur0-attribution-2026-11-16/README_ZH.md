# 今晚 A16 验证摘要

当前硬化版本的局部正确性通过：两次18/18契约/缓存检查、20项ActQuant；两组history各32次Q读回全部通过FP64，33份完整logits与stock逐位一致。QB实际执行32次完整守卫后使用同一stock pipeline，Q4各32次自定义dispatch。原公开模型可信参考严格失败exit1继续保留。

Q4独立单dispatch收益：初始Fair状态三轮配对改善2.509%/2.596%/2.775%；持续模型负载后Serious状态为2.377%/2.184%/2.441%。两种热状态分开保存，不据此声称长时吞吐或能耗改善。

整段32-token decode：两组history、每组3轮21对，所有正式样本均Serious。QB比stock慢约68–94ms（2.714%–3.738%）；Q4慢约69–88ms（2.858%–3.759%）。A/A配对中位差约-7.101至+4.961ms。当前包没有整模型收益，stock保持默认。

独立CPU诊断中，32次getter均命中缓存；snapshot加key构造占据约68–95ms嵌套区间。它是优先检查的host热点，不能与另一场正式墙钟简单相减，也不能据此声称已拆清CPU/GPU/热漂移。一次受限System Trace因Device disconnected提前结束；没有有效正常图关键路径或硬件counter证据。

英文README含每轮p50/p95、配对差、胜出数、配置、边界和复跑方法；SUMMARY.json为机器可读派生分析。原始CSV、张量、logits、日志、trace和签名产品只留本地。
