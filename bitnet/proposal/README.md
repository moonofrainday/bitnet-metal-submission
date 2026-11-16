# Mobile BitNet technical proposal

English proposal for the Tether mobile GPU assignment. `PROPOSAL.md` is the editable content source. `OUTLINE.md` maps the eleven requested topics before the sections were drafted. The implementation evidence cutoff is 16 November 2026. The historical reviewed/import snapshot is `bbf70e9cc70302534ff87b4f34d4ce52a9670baa`; the ActQuant A16 executable is tied to source `4c249e3`, and the documentation/evidence snapshot is `e84732c`. The combined ActQuant/Q4 binary is `edbd109`; both histories now pass its own scoped stock-ON gate. Its post-ActQuant operator timing is separate from earlier results.

- `BitNet_Mobile_GPU_Proposal.pdf`: technical plan report for author review.
- `BitNet_Mobile_GPU_Proposal.docx`: editable Word version of the same content.
- `render_proposal.py`: renders both formats; it runs no inference.
- `assets/execution-flow.svg` and `.png`: source and publication figure for graph/dispatch flow.

The report separates measured A16 results, preserved numerical failures and proposed Android work. Original authored 4x2 PoC, C8 research and stock-helper-derived Q4 are distinct. Q4 has legacy ActQuant-OFF acceptance and new two-history stock-ON numerical acceptance. Fresh post-ActQuant operator pairing measures 2.61%-2.68% reduction; joint whole-decode and corpus/PPL are unmeasured. All historical Q4/guard/FFN performance uses that earlier semantic profile. The exact original 4x2 PoC now passes a separate physical-phone sample-I/O test. The preserved default path omits ActQuant; a separate opt-in now passes the same-input source-derived F32 quantizer checks, while full-model frozen-reference acceptance remains false and new A16 operator/actual-input quantizer tests pass. Team checkpoint, semantic repair, representative quality, Android and energy remain explicit gaps. Use `../analysis/review-2026-11-15/` for historical review controls, `../analysis/actquant-2026-11-16/` for Mac numerical localization, and `../analysis/actquant-iphone-2026-11-16/` for the completed A16 quantizer and measured full-model strict exit 1. Joint qualification and current timing are indexed in `../analysis/actquant-q4-joint-2026-11-16/`. Build/run entry selection is documented in `../docs/BUILD_AND_RUN.md`.

## Rebuild the documents

Use an isolated document environment with the packages in `requirements-docs.txt`. The tested artifact runtime has Python, python-docx, ReportLab and Pillow. Arial on macOS or DejaVu on Linux is required by the renderer's font selection.

```sh
python render_proposal.py --output /absolute/path/to/local-proposal-build
```

This writes the PDF and DOCX into that output folder and refreshes the code-generated figure. Runtime PDF metadata may change across builds; the published checksums bind the reviewed deliverables, not an assertion of byte-identical reconstruction on another font/toolchain.

The Word output was separately rendered with the packaged DOCX renderer and visually reviewed, as was every final PDF page. QA images, intermediate documents, local review memos, raw experiments and logs remain outside this source package. Only final documents, source and derived validation information are published.

The report is a draft for the author's technical review. Private repository publication grants no access to the assignment team and does not submit the assessment.

FFN/fusion branches are consolidated as one page of optional performance exploration in Section 10. It preserves mechanisms, actual coverage, earlier direct stock contrasts, later direct G/F8 pairs and numerical limitations. Detailed historical aggregates and source digests are in `../analysis/ffn-performance-exploration/`. The latest revision adds a separate ActQuant progress page with measured Mac and A16 numerical evidence and preserved full-model failures. No Mac performance result is included. Pagination is regenerated and visually verified for this revision.

Both language editions contain 23 PDF pages and 23 rendered Word pages in this revision.
