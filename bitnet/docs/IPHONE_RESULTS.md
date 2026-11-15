# Physical iPhone results

These are **historical measurements from the preserved experiments**, not fresh performance tests of the repackaged checkout. Device: iPhone 14 Pro, Apple A16 GPU, iOS 27.2 (24B119); host Xcode 27.2 / iPhoneOS SDK 27.2. Public 2B GGUF SHA-256: `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2`. Raw samples stay local. Digest-indexed source notes are listed in `../provenance/EVIDENCE_INDEX.json`.

## Q stock4x1: operator scope

Same real phone step-16 W/X, one dispatch per command buffer; 30 warmups/path, 201 balanced alternating pairs/round, three rounds. Timing is **GPU command-buffer elapsed time**, `(GPUEndTime-GPUStartTime)*1e6` after completion, in microseconds. Capture, tensor readback and diagnostic trace are disabled. Stock is the actual native `kernel_mul_mv_tq2_0_f32` PSO, not a copied approximation.

| Round | Stock p50/p95 us | Q4 p50/p95 us | Paired Q4-stock median us | Median paired relative change | Q4 wins |
|---:|---:|---:|---:|---:|---:|
| 1 | 227.375 / 230.250 | 222.500 / 224.208 | -5.167 | -2.265% | 198/201 |
| 2 | 229.208 / 232.167 | 221.917 / 223.625 | -7.542 | -3.291% | 201/201 |
| 3 | 228.917 / 231.708 | 221.875 / 223.500 | -6.875 | -3.021% | 200/201 |

A/A paired medians: +0.083, -0.125, 0.000 us. All direct order/time-quartile medians favored Q4. Every before/after thermal snapshot was **serious**; charging at 80%, low-power mode off. GPU clocks, occupancy, bandwidth and power counters were unavailable. These numbers characterize the measured hot device, not nominal-temperature performance.

Q4 passes FP64 and equals stock bit for bit on all 7,680 outputs from steps 1/16/32. Two fixed histories each show 32/32 layer-0 Q dispatches and 33 stock-identical full-logit vectors; stock prefill and non-target nodes remain stock. All-32-call pre-RoPE tensor/FP64 and residual-output readback are unverified. No whole-model Q4 throughput benefit or complete model acceptance is claimed.

W SHA-256: `b2dea7f42cacfd08ed780df979f6f28240026795b1f5c87ca8c16fc7da23dc58`.
Step-16 X SHA-256: `8314cd7fe2f5deaa64ef8a7c9cbb1d9254bc430eed863ea4792c46a3c99c887e`.

## G/F8: complete fixed 32-token decode

The later direct experiment used one process/model/context per history. Each path clears KV, restores the same 27-token prefix and runs the same 32 continuation IDs; loading, KV reset and prefix restoration are outside the measured interval. Normal CPU encoding, GPU execution and completion waiting are included. This is **synchronized host wall time**, not GPU timestamps or isolated fusion cost.

Both histories used three rounds of 21 direct G/F8 pairs and 21 G/G calibration pairs per round, with balanced execution order. Actual scope: 252 pairs, 16,128 token records and 12 round/contrast summaries. No samples were excluded.

| History / round | G p50/p95 ms | F8 p50/p95 ms | Paired F8-G median ms | Median paired relative change | F8 wins |
|---|---:|---:|---:|---:|---:|
| Primary / 1 | 1661.550 / 1688.809 | 1623.549 / 1636.249 | -36.692 | -2.208% | 20/21 |
| Primary / 2 | 1683.039 / 1705.284 | 1644.256 / 1670.426 | -32.744 | -1.961% | 21/21 |
| Primary / 3 | 2202.177 / 2308.150 | 2127.355 / 2227.158 | -65.874 | -3.071% | 20/21 |
| Independent / 1 | 1851.386 / 1969.085 | 1805.543 / 1917.458 | -44.809 | -2.386% | 21/21 |
| Independent / 2 | 2202.770 / 2443.178 | 2150.082 / 2341.518 | -78.711 | -3.568% | 21/21 |
| Independent / 3 | 2959.360 / 3055.194 | 2845.515 / 2937.888 | -105.156 | -3.529% | 20/21 |

F8 won 123/126 direct pairs. Ratios of route p50 values imply 2.287%-3.847% reduction, whereas medians of per-pair relative changes imply 1.961%-3.568%; these are distinct statistics. All G/F8 timing samples were serious-thermal, charging at about 90%, low-power mode off. G/G paired medians were -0.140/-0.188/+3.765 ms and +0.724/+6.310/+2.139 ms. Absolute wall times drift substantially; do not pool them into a universal latency or claim cool-device/tail-latency results.

Per history, G qualification recorded 960 dual-projection plus 960 postprocessing dispatches; F8 recorded 960 fused dispatches. Q and down remained stock. G's 33 logits vectors matched stock. F8 retains strict full-chain FP64 and stock-relative model-output failures (**strict exit 1**), despite recorded fixed-prefix greedy agreement. This campaign has **no formal direct stock/S contrast**; do not subtract historical percentages to manufacture one. It also changes dot arithmetic/layout as well as fusion, so it cannot isolate pure fusion cost.

## Selection

Keep stock as default. Q4 is a limited layer-0 stock-compatible example with a small measured operator gain. G is the stock-output-compatible FFN research candidate. F8 is the stronger performance research candidate in the direct G/F8 phone contrast, with preserved numerical/model compatibility failures. K remains experimental, with 28 eligible layers and two safe stock fallbacks in its recorded graph run. No new A16 correctness or timing claim follows merely from compiling this package.
