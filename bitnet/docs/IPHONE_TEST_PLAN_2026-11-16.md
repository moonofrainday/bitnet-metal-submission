# iPhone 14 Pro: staged Qcur-0 verification and performance plan

Status at preparation: **the current canonical-contract attribution package has
not yet been qualified on the phone**. This document records an executable plan,
not completed measurements. No phone, signing, build or GPU command was executed
to prepare this plan.

The scope is the existing layer-0 `Qcur-0` candidate on A16: actual stock (S),
full guards followed by actual stock (QB), and stock4×1 (Q4). There is no new
kernel selection, FFN change, multilayer candidate expansion or model-quality
claim. The separate Mac encoder-splitting probe must **not** be imported into
this iPhone build.

## Fixed conditions and version boundaries

| Item | Frozen condition |
| --- | --- |
| QVAC base | `2db8b046d555316019ae75d1be90484adc8b9330` plus the recorded compatibility, ActQuant and canonical-contract patches |
| Prepared native attribution source | `14a9f103c23b5a24bd0da1e0dbc5a0d5a37ade3a`; bind actual source/product hashes, not only embedded Git metadata |
| Prepared phone package | The source-bound `ios-signed-v2` package; build success is recorded, new phone runtime is PENDING |
| Earlier qualified phone version | `edbd109`; retain its results separately, never relabel them as current-package results |
| Model | Public Microsoft BitNet 2B converted TQ2_0 checkpoint; not the team's private checkpoint |
| GGUF SHA-256 | `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2` |
| Histories | Original and independent prompts; each has 27 prefill tokens and the saved 32 continuation token IDs |
| Runtime | ActQuant ON, F16 K/V, context 512, batch/ubatch 128, 6 CPU threads, flash attention OFF |
| S | Actual `kernel_mul_mv_tq2_0_f32`; four rows/SIMD, two SIMD groups/threadgroup, grid 320, threads `[32,2,1]` |
| QB | All candidate checks/getter; same-process S pipeline, stock ABI, buffers, offsets and launch; zero custom dispatches |
| Q4 | `tq2_0_matvec_stock4x1_v1`; four rows/SIMD, one SIMD group/threadgroup, grid 640, threads `[32,1,1]` |
| Numerical tolerance | `abs(output-reference) <= 2e-4 + 2e-5 * abs(reference)`; do not relax it |
| Preserved failure | Earlier official-source whole-model comparison **failed, strict exit 1**; that observed result is not PENDING and is not overwritten |

Q4 reuses stock-helper arithmetic. Its qualification is distinct from the
original self-written PoC and C8×2. Local Q FP64 accuracy and stock-ON parity do
not establish official-model fidelity, complete Transformer-block accuracy or
held-out model quality.

## Ordered gates and stop rules

| Stage | Execute | Required evidence before proceeding |
| --- | --- | --- |
| 1. Identity and cold loader/cache | Check physical device, SDK, signed app/framework, shaders, canonical config and model hashes. Run the actual 18-case backend contract/cache test; repeat in a second fresh app process. | Correct unique entries and 4×1 geometry; wrong-entry/library/path/cache controls pass. Preserve any first-valid-load failure, launch error or termination report. Do not time an unexplained valid-load failure. |
| 2. ActQuant and model regression | Run all 20 operator cases, then both histories' S/Q4 capture/plain/missing-library controls. | All 386,567 operator values pass; all 32 actual N=1 Q outputs pass FP64, W/X identities agree, GPU completion is recorded, residual checkpoints and all 33 logits match stock-ON. Capture/plain neutrality and fallback evidence are mandatory. |
| 3. Same-process QB control | Run same-context S/QB/Q4 verification for both histories. | QB has 32 full guard passes, 32 stock target dispatches and zero custom dispatches. S/QB use the same observed native PSO, ABI, live bindings/offsets and launch. All 32 W/X/Y and 33 logits agree; callback-null stock control is neutral. Mint a build-bound `TIMING_GATE.json` only after complete evidence. |
| 4. CPU/natural-CB diagnostics | Collect one bounded diagnostic per history; separately measure instrumentation OFF/ON (three rounds × seven pairs per route). | All guard attempts/successes, nested CPU getter/cache/encode regions and natural CB status are complete. Preserve clock calibration, record drops and callback drain status. No new synchronization, CB split or tensor readback in these diagnostic decode paths. |
| 5. Isolated S/Q4 operator | Extract fresh A16 post-ActQuant steps 1/16/32 W/X/Y from the newly accepted graph run. Requalify the same installed app, then time A/A and S/Q4 on step 16. | All 2,560 rows, finite values, FP64, stock comparison, reported write coverage and hashes pass first. Per path: 30 warmups; three rounds × 201 alternating pairs; one dispatch per CB; completed GPU timing. |
| 6. Formal 32-token wall | Both histories, same build/model/context: A/A, S/QB and S/Q4; two full replay warmups/route, three rounds × 21 pairs/contrast. | Restore identical KV/prefix/history per path; balanced AB/BA and rotating blocks. Capture, diagnostics and readback OFF. Record final-logit hashes outside the timer, per-token wall, thermal/power state, order and time trend. |
| 7. Optional normal trace | After the core gates, make at most one bounded normal-stock Metal System Trace attempt, separately from formal timing. | Preserve exact success/error and available interval coverage. If unavailable, mark unavailable. No repeated blind attempts, no Mac encoder-probe patch, no invented per-op timing shares. |

Invalid startup geometry is a **rejected configuration**, not a successful
stock-fallback run. The supported missing-library control must actually reach
the guarded route and then use stock safely. A QB run that skips full checks is
not a valid same-shader control. C8 may appear in the fixed 18-case loader safety
tests and the preserved operator harness; it is not a new optimization candidate
or the main performance contrast tonight.

If a required correctness check is false, stop performance for that package.
Missing evidence, device disconnection or environment failure remains
unverified/null and nonzero; it must not be reported as a measured numerical
failure or a pass. Preserve raw results before any routine repair.

## Executable entry points

Run from the repository root. Set local variables to the selected artifacts;
do not place real device identifiers, signing identities or personal paths in
published documentation:

```sh
export BUILD_DIR="/local/prepared/ios-signed-v2"
export DEVICE_ID="YOUR_CONNECTED_DEVICE_IDENTIFIER"
export MODEL="/local/models/public-bitnet-2b-tq2_0.gguf"
export ACTQUANT_FIXTURES="/local/fixtures/actquant-20-cases"
export TOKENS_PRIMARY="/local/inputs/tokens-primary.json"
export TOKENS_INDEPENDENT="/local/inputs/tokens-independent.json"
export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
export QUALIFIED_ROOT="/local/results/NEW-q-correctness"
```

These paths are placeholders, not bundled data. `ACTQUANT_FIXTURES` must be the
existing 20-case manifest, not the model-free 15-case subset. Outputs must be
fresh. The host validates the selected package and shader provenance before
using the phone. Models, fixtures and raw captures remain outside the checkout.

For the prepared package, run these tools from its recorded source checkout.
Its shader compiler provenance binds that source path as well as file hashes;
reusing the package from a different documentation worktree is rejected with
`Q4 compile provenance missing`. A new clone needs its own clean package build.
Do not edit the manifest to bypass this check. The local evening command index
uses the original prepared checkout and passes the plan-only gate.

First generate and inspect the plan. This command performs no device work:

```sh
python3 bitnet/tests/run_iphone_attribution.py --plan \
  --build-dir "$BUILD_DIR" --device "$DEVICE_ID" --model "$MODEL" \
  --operator-fixtures "$ACTQUANT_FIXTURES" \
  --tokens-primary "$TOKENS_PRIMARY" --tokens-independent "$TOKENS_INDEPENDENT" \
  --developer-dir "$DEVELOPER_DIR" --through correctness \
  --output "/local/results/NEW-correctness-plan"
```

When device execution is authorized, stages 1–3 have one guarded entry:

```sh
python3 bitnet/tests/run_iphone_attribution.py --run \
  --build-dir "$BUILD_DIR" --device "$DEVICE_ID" --model "$MODEL" \
  --operator-fixtures "$ACTQUANT_FIXTURES" \
  --tokens-primary "$TOKENS_PRIMARY" --tokens-independent "$TOKENS_INDEPENDENT" \
  --developer-dir "$DEVELOPER_DIR" --through correctness \
  --output "$QUALIFIED_ROOT"
```

This runs the 15-stage contract/operator/two-history S/Q4 gate, an independent
second-process contract repeat, both same-process S/QB/Q4 controls and their
analyses, then creates the timing gate. Its `PLAN.json` exposes each actual argv
if execution must proceed one stage at a time. Do not substitute an old package's
passed gate. Reuse of a model copy still verifies its checksum.

After correctness passes, generate the remaining exact stage commands without
rerunning those gates:

```sh
python3 bitnet/tests/run_iphone_attribution.py --plan \
  --build-dir "$BUILD_DIR" --device "$DEVICE_ID" --model "$MODEL" \
  --operator-fixtures "$ACTQUANT_FIXTURES" \
  --tokens-primary "$TOKENS_PRIMARY" --tokens-independent "$TOKENS_INDEPENDENT" \
  --developer-dir "$DEVELOPER_DIR" --qualified-root "$QUALIFIED_ROOT" \
  --through formal --output "/local/results/NEW-measurement-plan"
```

Execute the serialized `PLAN.json` argv entries in this order, recording command,
exit code and log for every entry:

1. `diagnostic-primary`, its `-analysis`, then independent equivalents.
2. `overhead-primary`, its `-analysis`, then independent equivalents.
3. Pause for the isolated-operator stage below and thermal review.
4. `formal-primary`, its `-analysis`, then independent equivalents.

The existing wrapper's `--run --through formal --qualified-root ...` can execute
diagnostic/overhead/formal as one campaign. To avoid duplicating diagnostics
already completed, use the planned argv for the remaining stages. A formal
command requires the same immutable timing gate. A changed binary, model,
shader, history or required evidence invalidates reuse.

### Fresh phone-input operator check and timing

Use the newly completed host collection and its actual scoped analysis:

```sh
python3 bitnet/actquant/prepare_q_fixture.py \
  --run "$QUALIFIED_ROOT/correctness/primary-Q4-capture" \
  --acceptance "$QUALIFIED_ROOT/correctness/primary-analysis/SUMMARY.json" \
  --build-dir "$BUILD_DIR" --output "/local/fixtures/NEW-phone-post-actquant-Q"

python3 bitnet/ios/run.py --mode q-benchmark --install --reuse-model \
  --build-dir "$BUILD_DIR" --device "$DEVICE_ID" --model "$MODEL" \
  --fixture "/local/fixtures/NEW-phone-post-actquant-Q" \
  --developer-dir "$DEVELOPER_DIR" \
  --output "/local/results/NEW-phone-Q-operator"

python3 bitnet/actquant/analyze_q_timing.py \
  --performance-run "/local/results/NEW-phone-Q-operator" \
  --fixture "/local/fixtures/NEW-phone-post-actquant-Q" \
  --joint-candidate "$QUALIFIED_ROOT/correctness/primary-Q4-capture" \
  --contract-profile canonical-v1 \
  --output "/local/results/NEW-phone-Q-operator-analysis.json"
```

`prepare_q_fixture.py` rejects Mac substitution and binds the physical A16
post-ActQuant producer, W/X/Y hashes, accepted model run and signed package.
The preserved `q-benchmark` entry performs fresh qualification and timing in one
installation, with no reinstall between them. Its fixed AA/SQ/CQ harness also
records the historical C8 comparison: `AA` is S/S, `SQ` is S/Q4 and `CQ` is
C8/Q4. Publish AA and SQ as the selected-candidate result and label CQ separately.
Do not rename CQ to S/Q4. The operator protocol is fixed at
30 warmups, 201 pairs and three rounds; generic decode flags do not change it.
The preserved phone operator entry initializes every output row to NaN and
checks all rows, but does not add the Mac microharness's prefix/suffix canaries.
Do not claim that extra canary coverage for this phone run.

## Measurement and interpretation rules

- **CPU:** guard/getter/encoding intervals are worker-thread elapsed time,
  including waits and descheduling, not CPU cycles. They are nested and may be
  concurrent; never sum them into a wall-time partition.
- **Isolated GPU:** `(GPUEndTime-GPUStartTime)` after completion measures the
  command-buffer elapsed interval. It contains exactly one dispatch here and
  excludes ActQuant preparation, graph guards/cache lookup, CPU encoding and
  normal model scheduling. Preserve the actual fixture hashes.
- **Model wall:** fixed 32-token decode includes existing completion waits;
  reset, prefix restoration and warmup stay outside the interval. No free-token
  divergence or changing KV history is allowed in this comparison.
- **Natural CB evidence:** keep existing encoder/CB boundaries. Repeated log
  lines for one enclosing CB are not independent kernel times. Host callback
  delivery is not an exact queue boundary; uncorrelated clocks cannot be
  subtracted. Unavailable counters or per-dispatch intervals remain unavailable.
- **Environment:** record iPhone model/iOS, charging/power source, battery,
  low-power flag, before/after thermal state, execution order and time trend.
  Do not pool nominal with serious/critical or transition samples. Unknown
  thermal/low-power evidence is unqualified, never a stable-temperature pass.
- **Statistics:** per round retain p50/p95, median of the actual paired B−A
  differences, B wins, AB/BA groups and time blocks. Compare with A/A. A local
  GPU improvement does not establish model-wall, tail-latency or energy gains.
- **Attribution:** QB−S estimates the guarded path effect. Q4−QB also changes
  encoding, launch and scheduling; it is not pure shader cost. Previous Mac
  results motivate this test but are not A16 results.

## Pending result ledger

Fill only from the actual artifacts. `PENDING` is intentionally not a pass.

| Current-package field | Result/evidence to fill |
| --- | --- |
| Physical device model, iOS, GPU | **PENDING** |
| App/framework/source/shader/canonical-config hashes | **PENDING** |
| GGUF and both history hashes reverified | **PENDING** |
| First cold 18-case contract and second-process repeat | **PENDING** |
| 20-case ActQuant counts/codes/scales/finite output | **PENDING** |
| Both-history all-32 Q FP64 and W/X identity | **PENDING** |
| Both-history residual checkpoints and 33 logits | **PENDING** |
| QB 32 guards, 0 custom; actual same PSO/ABI/live bindings | **PENDING** |
| Off/prefill/non-target/missing-library behavior | **PENDING** |
| Capture/plain and metrics-neutrality controls | **PENDING** |
| CPU/cache/natural-CB diagnostic completeness | **PENDING** |
| Fresh phone fixture and AA/SQ4 three-round timing | **PENDING** |
| Two-history formal AA/SQB/SQ4 three-round wall | **PENDING** |
| Optional normal trace/counters | **PENDING**, or exact unavailable/error |
| Normal-schedule selected-Q critical-path share | **UNVERIFIED** unless directly supported; never infer from static counts |
| Earlier official-source model comparison | **OBSERVED FAIL; strict exit 1 preserved** |
| New official-source whole-model or model-quality acceptance | **NOT ESTABLISHED by this protocol** |

Raw commands, logs, captures, vectors and CSV stay in fresh local timestamped
directories. Publish only scripts, patches and derived analysis. Keep each
package/campaign separate from the earlier `edbd109` phone results and from the
Mac campaign. Preserve strict numerical failures and missing evidence.

## Time allocation and stop points

These are planning timeboxes, not a completion promise: allow roughly 15 minutes
for identity/cold-loader inspection, 45 minutes for correctness, 30 minutes for
diagnostic review and 20 minutes for the isolated operator, with cooldown and
data transfer separate. If signing, loading or evidence collection fails, save
the reproduction and diagnose before timing.

Estimate formal duration after observing the phone's actual replay and prefix
restore times. The fixed protocol contains **756 timed 32-token paths**
(two histories × three rounds × 21 pairs × three contrasts × two sides), plus
12 full warmup paths and per-path restoration, install/fetch/analysis and any
cooldown. Do not promise a Mac-based ETA or cut the protocol silently. If the
evening ends early, save completed rounds and mark the remainder PENDING.

## 中文执行清单

1. 先核对新包、模型和输入哈希，跑两次独立进程的 18 项合约/缓存检查；旧
   `edbd109` 的通过不能替代新包验收，首次加载失败必须保留。
2. 同构建跑 20 例 ActQuant，两组 history 的全部 32 次 Q 和 33 份 logits；
   补齐同进程 S/QB/Q4、真实 PSO/ABI/buffer/offset、FP64、读回中立性与回退。
3. 正确性过关后才测。先做不切分 CB 的 CPU/cache/自然 CB 诊断，再单独测插桩
   开销；不把嵌套 CPU 时间和 GPU 时间相加当成完整墙钟拆解。
4. 从新手机结果提取实际 post-ActQuant W/X，做独立算子 A/A 与 stock/Q4：
   预热 30 次，三轮各 201 对。此耗时不包含量化、接入检查和整模型调度。
5. 再测两组 history 的整段 decode：A/A、S/QB、S/Q4，预热两次完整回放，三轮
   各 21 对，平衡顺序。记录热状态、供电、低功耗模式和时间趋势，不混池。
6. 可选正常 trace 只尝试一次；无法取得的 counter/逐算子占比写 unavailable。
   不把 Mac 的 encoder/fence 探针带入手机基线。
7. 原容差和官方参考严格失败退出码 1 均保留。局部正确、stock parity、单算子
   加速和整模型加速分开写；未测填 PENDING。原始数据留本地，只上传代码与分析。

Related source protocols: [IPHONE_ATTRIBUTION_PLAN.md](IPHONE_ATTRIBUTION_PLAN.md),
[prepared-build scope](../analysis/iphone-attribution-preparation-2026-11-16/README.md),
and [Mac scope and limits](../analysis/mac-qcur0-value-2026-11-16/README.md).
