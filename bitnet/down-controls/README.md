# Minimal A16 down-projection arithmetic control

## Hypothesis recorded before implementation

The independent prompt's layer-0 / step-16 actual stock weighted-normalization input has twelve down-projection rows outside the frozen FP64 tolerance. Keeping the packed dot sum separate from the `-sumy` correction may change cancellation and FP32 rounding. This is a numerical hypothesis, not a proven cause or a proposed model repair.

This experiment changes only correction placement, relative to a newly compiled stock-helper bridge. It does not change projection coefficients, packed-byte floor expressions, block scale, block/lane mapping, block traversal, SIMD reduction, loading, or the four-row/two-SIMD launch.

## Three paths

| Path | Execution |
|---|---|
| S | Actual stock `kernel_mul_mv_tq2_0_f32` selected from the freshly built framework. |
| S-wrap | A new pipeline containing the unmodified stock TQ2_0 helper, with the stock runtime argument ABI and function constants. |
| Correction-end | Identical to S-wrap except `sum=-sumy; add dot` becomes `sum=0; add dot; sum-=sumy`. |

S-wrap is separately compiled. Its equality to S must be measured, not assumed. AIR must show whether correction placement survives optimization; different source alone does not establish different instructions. Both candidates use the same offline compiler options.

## Fixed workload and selection

- Public 2B TQ2_0 GGUF SHA-256: `52257df879ab4366d97228e7bda94a9a3c6a9065ec44e7879a8b87d54e568cd2`.
- Independent prompt, stock graph stream, layer 0, decode step 16.
- Full W is TQ2_0 `[2560 output rows, 6912 K]`: 27 blocks/row, 66 bytes/block, 1782 bytes/row.
- Full W SHA-256: `c032780f9245f4ecfe37fe117f0b6f16b5a76cfb72734c60b1f7e4beabe99888`.
- X is the actual captured F32 weighted-normalization output `[6912]`, SHA-256 `c9e096fc21f1e6ab4dcc1c4329925942219f848b4ab082dc30b930fe2bf43e7a`.
- Known failing rows: 5, 338, 356, 362, 455, 502, 882, 1148, 1508, 1830, 2155, 2508.
- Test the twelve contiguous eight-row groups beginning at 0, 336, 352, 360, 448, 496, 880, 1144, 1504, 1824, 2152 and 2504: 96 outputs/path.

Each group uses a new contiguous eight-row weight buffer and the same X. Runtime arguments use `ne01=8`, `ne0=8`; one threadgroup contains 32 x 2 threads. The actual stock selector must return the same cached PSO for full and sliced tensor descriptions. Candidate wrappers retain runtime arguments rather than hardcoding K or row extent. These are small real-input operator controls, not full-layer/model acceptance. Slicing changes the workload; it does not demonstrate behavior under full-model memory traffic.

## Correctness and exits

An independent logical-K FP64 unpack reference uses each row's actual trits, FP16 scale and identical F32 X. Preserve `abs(actual-reference) <= 2e-4 + 2e-5*abs(reference)` and all failed rows. Each GPU output starts as NaN, has prefix/suffix canaries, and is read only after completed command-buffer status. Compare S-wrap and correction-end against S as well as FP64.

The native entry writes all selected results before returning: 0 means all three routes satisfy the selected FP64 checks, 1 is a completed numerical failure, 2 is an execution/input error. `analysis_completed`, `execution_pass`, `fp64_pass` and `strict_exit_code` are separate fields. Known stock failures must not be hidden by bridge parity. `model_output_pass` stays null. No timing measurement is implemented.

## Build and integration

```sh
python3 bitnet/down-controls/build-shaders.py --output /absolute/path/to/fresh/down-libraries
python3 bitnet/down-controls/prepare.py \
  --weight /absolute/path/to/ffn_down.tq2 \
  --weight-sha256 c032780f9245f4ecfe37fe117f0b6f16b5a76cfb72734c60b1f7e4beabe99888 \
  --input /absolute/path/to/down-input.f32 \
  --libraries /absolute/path/to/fresh/down-libraries \
  --output /absolute/path/to/fresh/down-fixture
```

The builder records commands, source digests and AIR. No compiled library or fixture is committed. Source-only native C API:

```c
int bitnet_run_iphone_down(const char *fixture_root, const char *library_root,
                          const char *output_root, const char *mode);
```

Compile `bitnet/ios/src/iphone-down-controls.mm` as a separate arm64 Objective-C++ object against the same source/private headers and freshly built framework, with the existing helper flags (`-O2 -fno-fast-math -ffp-contract=off -fobjc-arc`). Link that object into the staged app. Pass mode `verify`, fixture and library directories in Documents, and a fresh output directory. The staged app exposes the separate `DOWN` operator kind, and the main builder links this helper. The `down-verify` runner mode records completed numerical exit 1 separately from an installation, launch or execution error. No default model kernel is changed.

The fixture manifest binds full W, X, model identity, known failing rows and groups. Passing `--libraries` to the preparer verifies the builder output and writes both actual library hashes into the fixture manifest before transfer. Runtime checks reject missing hashes, altered input, unsupported device, changed stock PSO geometry, or output overwrites. Expand to a full 2560-output experiment only if this small control provides useful evidence. No performance, repair or model-quality claim follows from these sources.

## Recorded A16 outcome

The fresh signed-build campaign on iPhone 14 Pro/A16 completed 36 command buffers and checked 288 finite outputs. S-wrap reproduces all 96 stock F32 results exactly. Stock, S-wrap and correction-end each retain the same 12 FP64 failure positions. Correction-end changes 48 rows, makes 22 closer and 26 farther from the independent FP64 reference, and increases the maximum error/bound ratio from 1.566268 to 1.829765. It is retained as a negative result, not expanded to a full matrix or timed. Native and independent strict exits remain 1. See [the full result note](../docs/IPHONE_Q4_COMPLETE_2026-11-15.md) and the derived analysis folder; raw slices and per-row results remain local.
