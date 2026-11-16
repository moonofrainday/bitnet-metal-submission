# English technical report

[Read the submission PDF](BitNet_Mobile_GPU_Proposal.pdf). The report covers the assignment's eleven topics, the authored matrix-kernel PoC, stock-derived Q4 specialization, physical-iPhone validation, performance controls and limitations. Current A16 correctness and stock/QB/Q4 results are included; the complete trusted-model strict failure remains explicit.

## Included files

- `BitNet_Mobile_GPU_Proposal.pdf`: the report deliverable.
- `PROPOSAL.md`: editable English content source.
- `render_proposal.py`, `requirements-docs.txt` and `assets/`: reproducible report builder and reviewed figure source.
- `MANIFEST.json`: report/source hashes, measured version boundaries and visual-review status.

The Chinese companion and editable Word copies are kept in the author's local archive. The current GitHub tree contains the English submission materials. [Current machine-readable derived evidence](../analysis/iphone-qcur0-attribution-2026-11-16/SUMMARY.json) retains unverified fields as null; numerical and model-reference failures are not removed by packaging cleanup.

## Rebuild locally

Use Python 3.10+ with `requirements-docs.txt` and installed fonts supported by the renderer. No font binaries are distributed. A fresh local output directory keeps generated files outside the source tree:

```sh
python3 bitnet/proposal/render_proposal.py \
  --preserve-figure --output "$LOCAL_REPORT_OUTPUT"
```

The builder can also create an editable Word copy locally; only the PDF is included as the report deliverable. It does not execute inference. The manifest binds the final source and PDF, rather than promising byte-identical rendering across arbitrary font/tool versions.

Raw experimental evidence and QA page images remain local. The [submission scope](../docs/SUBMISSION_SCOPE.md), [build guide](../docs/BUILD_AND_RUN.md) and [current result summary](../analysis/iphone-qcur0-attribution-2026-11-16/README.md) provide the code and evidence entry points.
