# 今晚 iPhone 14 Pro / A16 实验顺序

目标：先证明新包正确，再区分接入检查、CPU编码、GPU执行/调度与热漂移，最后判断下一步值得优化哪里。保持当前shader算法和原数值容差。

| 顺序 | 实验 | 关注点 |
|---|---|---|
| 1 | 新签名包冷启动及18项合约/缓存检查，独立进程再复跑一次 | Q4/C8身份、启动几何、错误库/配置回退；保留任何首次加载失败 |
| 2 | 同构建20例ActQuant前置检查及两组history的Q4回归 | 原27-token prefill+32固定tokens，F16KV、ActQuantON；32次真实dispatch/GPU完成、原FP64容差、33份logits、探针中立性及回退 |
| 3 | 同进程S/QB/Q4完整对照 | QB完整guard命中32次、custom为0；S/QB同stockPSO、ABI、真实buffer/offset和launch；全部输出对照 |
| 4 | CPU/自然CB诊断及插桩开关对照 | 分开记录张量/buffer检查、cache/getter、CPU编码、原commandbuffer提交/完成/GPU区间；校准计时开销，不新增同步/切分 |
| 5 | stock关键路径观察 | 查看output head、FFN、Attention、其他投影及CPU空隙；尝试单独Metal System Trace。只能取得CB级数据时，不声称已测得逐算子占比 |
| 6 | 正式整段decode配对 | 两组history；每路2次完整预热、每对照3轮×21对，平衡AB/BA并轮换对照次序；A/A、S/QB、S/Q4；关闭诊断、capture与读回 |
| 7 | 独立算子补测（可选）与分析 | 新构建stock/Q4单dispatch用同W/X、30次预热、3轮×201对；与整段decode分开。保留p50/p95、逐对差、胜出数、顺序/时间趋势及热状态 |

正式墙钟与诊断数据分开。GPU command-buffer elapsed不能当成单kernel耗时；嵌套/并发CPU区间与GPU区间不能相加减后称为完整墙钟拆解。未知热状态不进入稳定温度组，温度转换样本仍保留。没有硬件counter时，不推断occupancy、带宽或寄存器压力改善。

白天已完成：新Mac及iOS构建；MTL0 TQ2_0 59/59、ActQuant16/16、实际cache18/18；两组同进程S/QB/Q4的全部32步局部FP64、stock parity、同PSO/ABI/buffer及33份callback-null stock logits均通过；6条诊断路径采集完整。没有发布Mac性能比较，也没有运行新iPhone包。

原模型可信参考严格失败退出码1继续保留。上述是局部算术、stock兼容性和实验方法验收，不是完整公开模型语义/质量通过。

手机尚未安装或运行这个新包。默认入口只生成计划，必须显式--run才使用设备。正确性失败或证据缺失时阻止后续性能。可通过--through correctness先验收，再用--qualified-root复用同构建合格证据运行后续阶段，避免重跑正确性。原始数据留本地，GitHub仅代码、补丁和分析。

英文入口与接口：[IPHONE_ATTRIBUTION_PLAN.md](../../docs/IPHONE_ATTRIBUTION_PLAN.md)。本地完整日志和精确命令见同名时间戳实验目录 README.md。

CPU侧记录的是线程上的elapsed区间，包含锁/文件等待和被调度出去的时间，不是纯CPU指令时间。scheduled/completed记录的是host回调送达时间，也可能包含回调调度延迟，不能当作精确GPU排队边界。
