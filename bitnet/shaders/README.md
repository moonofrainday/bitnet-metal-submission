# Preserved Metal shader sources

These source copies preserve the tested shader arithmetic. The Q-only entry and layout guards have the recorded versioned-contract edits. `SOURCES.json` records current file hashes, historical source paths and, where applicable, historical hashes. Historical paths identify evidence in the local research archive; they are not build dependencies.

Run the portable builder from the repository root:

```sh
python3 bitnet/scripts/build-shaders.py --sdk iphoneos --output /tmp/bitnet-phone-shaders
python3 bitnet/scripts/build-shaders.py --sdk macosx --output /tmp/bitnet-mac-shaders
```

Set `DEVELOPER_DIR` or pass `--developer-dir` when the active Xcode selection is not the intended full installation. The deployment targets default to iOS 16.4 and macOS 13.3. The default output is a timestamped directory outside this repository under `../bitnet-local/`. A specified output directory must be new or empty; nonempty directories are rejected before tool execution. The builder records every command, compiler log, source hash, and library hash in the output directory. Use `--plan` to inspect the command list without invoking tools; `--emit-air` also saves textual compiler output.

| Library | Implementation | Export used by the harness/backend |
| --- | --- | --- |
| `original-poc.metallib` | Original 4-row, 2-SIMD matvec | `tq2_0_matvec_poc` |
| `stock4x1.metallib` | Stock arithmetic, 4 rows, 1 SIMD group | `tq2_0_matvec_stock4x1_v1` |
| `qcur0-c8x2.metallib` | Historical C mask matvec, 8 rows, 2 SIMD groups | `tq2_0_matvec_c8x2_v1` |
| `ffn-mechanisms.metallib` | Stock-arithmetic dual projection, strict postprocessing, and historical up epilogue | `ffn_dual_projection`, `ffn_post_strict`, `ffn_up_epilogue`, `ffn_shadow_copy` |
| `ffn-gate-up-8x2.metallib` | F8 dual projection plus ReLU-squared multiplication | `tq2_0_matvec_ffn_gate_up_8x2` |
| `ffn-expand.metallib` | Split C FFN projection | `tq2_0_matvec_ffn_expand` |
| `k0-postnorm-inplace.metallib` | K postprocessing plus weighted RMSNorm, in place | `ffn_post_weighted_norm_inplace` |
| `k0-postnorm-reference.metallib` | Independent-output K normalization comparison | `ffn_post_weighted_norm` |

The individual `dual`, `post`, and `up-epilogue` libraries are also retained for the operator harness. The builder applies the historical safe-math/no-contraction flags to postprocessing and K normalization, and the recorded function remaps and macros to C/F8.

`original/contract.h` is the 2560-by-2560 Q contract. `include-expand/contract.h` is the 6912-by-2560 FFN contract. They are intentionally separate. Sharing or substituting them silently changes output coverage.

The current Q contract comes from `ggml/src/ggml-metal/ggml-metal-bitnet-q-config.h`. The Python builder reads that same header through `bitnet/q_kernel_contract.py`; Q4 accepts only 4x1 and C8 only 8x2. For 2,560 outputs, their respective grids are 640 and 160 threadgroups, with 32x1 and 32x2 threads per threadgroup. The builder checks each versioned entry in compiler AIR and the metallib name table, and rejects the legacy shared symbol or the other Q variant's entry.

The current Q external-pipeline cache binds the canonical library path, file identity, versioned entry and launch contract before lookup. Actual same-process cache, wrong-library, deletion and replacement controls are part of qualification. The old shared `tq2_0_matvec_poc_c8x2` symbol and function-only cache describe historical binaries, not the current Q4/C8 build. Historical results retain their original binary scope.

The iOS builder currently compiles the optional FFN shaders and links their native harness routes. They remain in this source package to preserve that build dependency chain; they are inactive by default and are not the recommended submission kernels. The original authored PoC and stock-derived Q4 are separately identified in [submission scope](../docs/SUBMISSION_SCOPE.md). Compiling a library is not device correctness evidence. See the report and acceptance documents for each route's passing, failing and unverified scope.
