# Mobile BitNet technical proposal

English proposal for the Tether mobile GPU assignment. `PROPOSAL.md` is the editable content source. `OUTLINE.md` maps the eleven requested topics before the sections were drafted. The implementation evidence cutoff is 15 November 2026, and the reviewed compute-code snapshot is `bbf70e9cc70302534ff87b4f34d4ce52a9670baa`.

- `BitNet_Mobile_GPU_Proposal.pdf`: technical plan report for author review.
- `BitNet_Mobile_GPU_Proposal.docx`: editable Word version of the same content.
- `render_proposal.py`: renders both formats; it runs no inference.
- `assets/execution-flow.svg` and `.png`: source and publication figure for graph/dispatch flow.

The report separates measured A16 results, preserved numerical failures and proposed Android work. Original authored 4x2 PoC, C8 research and stock-helper-derived Q4 are distinct. Q4 has current phone numerical acceptance and a small isolated operator benefit, with no measured whole-decode gain. The exact original 4x2 PoC's fresh physical-phone test, the team checkpoint, Android and energy remain explicit qualification gaps.

## Rebuild the documents

Use an isolated document environment with the packages in `requirements-docs.txt`. The tested artifact runtime has Python, python-docx, ReportLab and Pillow. Arial on macOS or DejaVu on Linux is required by the renderer's font selection.

```sh
python render_proposal.py --output /absolute/path/to/local-proposal-build
```

This writes the PDF and DOCX into that output folder and refreshes the code-generated figure. Runtime PDF metadata may change across builds; the published checksums bind the reviewed deliverables, not an assertion of byte-identical reconstruction on another font/toolchain.

The Word output was separately rendered with the packaged DOCX renderer and visually reviewed, as was every final PDF page. QA images, intermediate documents, local review memos, raw experiments and logs remain outside this source package. Only final documents, source and derived validation information are published.

The report is a draft for the author's technical review. Private repository publication grants no access to the assignment team and does not submit the assessment.
