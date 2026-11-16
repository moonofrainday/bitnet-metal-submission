# Mobile BitNet technical proposal

Submission-preparation revision, 16 November 2026. The English and Chinese editions each contain 25 PDF pages and 25 rendered Word pages. The reviewed flow figure and historical A16 measurements are preserved. This revision performs no inference, GPU test, deployment or signing.

## Read and review

- `BitNet_Mobile_GPU_Proposal.pdf`: English technical plan for the assignment.
- `BitNet_Mobile_GPU_Proposal.docx`: editable English companion.
- `PROPOSAL.md`: editable English content source; `OUTLINE.md` maps all eleven requested topics.
- `README_ZH.md`: Chinese companion and section-by-section review entry.
- `PENDING_A16_RESULTS.json`: machine-readable pending ledger; null means unmeasured, never zero or a pass.
- [Tonight's test order and commands](../docs/IPHONE_TEST_PLAN_2026-11-16.md).

Section 9 adds the focused M2 Pro attribution supplement. It evaluates the already selected Q projection, rather than claiming it is the largest model bottleneck. Local Q4 GPU benefit is measured, but current layer-0 integration gives no whole-decode improvement. Same-stock full-guard controls identify warm file-identity work as a substantial measured host region; total CPU/GPU/thermal attribution and normal critical-path share remain unresolved. The separate encoder timestamp probe changes scheduling and is not in the normal or phone build.

Section 10 retains the earlier A16 operator and historical guard/FFN tables. A separate results page leaves the current package's result cells blank and marks them PENDING. Earlier phone qualification from `edbd109` is distinct from the hardened attribution source `14a9f103`; the latter has Mac qualification and a prepared signed package, but no new physical-phone result yet. Mac numbers are not relabeled as mobile performance.

The report explicitly relates the scope to the brief: one matrix kernel within a transformer layer, sample-I/O harness, source scripts and a technical-plan PDF. An entire custom Transformer block, minimum speedup and whole-model acceleration are not stated brief requirements. System profiling still informs the optimization budget and limits performance claims. No bottleneck elsewhere is asserted without runtime evidence.

Original authored PoC, stock-helper-derived Q4 and optional FFN research remain separately attributed. The complete trusted-reference model comparison remains an observed FAIL, strict exit 1. It is not blanked out or repaired by stock parity, local FP64 accuracy or document revision. Team-checkpoint, representative quality, Android, energy and sustained operation remain explicitly qualified or unverified.

## Rebuild

Use Python 3.10+ with `requirements-docs.txt`. The recorded edit uses the Codex bundled Python with python-docx, ReportLab and Pillow. Installed Arial, Arial Unicode MS and Heiti SC fonts are used; no font files are distributed.

```sh
"$DOC_PY" bitnet/proposal/render_proposal.py --preserve-figure --output "$LOCAL_REPORT/english"
"$DOC_PY" bitnet/proposal/render_proposal_zh.py --output "$LOCAL_REPORT/chinese"
```

The English renderer's `--preserve-figure` keeps the already reviewed asset when only text changes. Regeneration without it is deliberate figure authoring. Both renderers write fresh PDF/DOCX outputs without inference. After approving output files and refreshing the manifest hashes, run:

```sh
"$DOC_PY" bitnet/proposal/check_translation.py --output "$NEW_TRANSLATION_CHECK"
```

The literal checker compares numbers, tables, code, URLs and citations. PDF and Word renders are additionally reviewed visually. This Mac's headless Word renderer requires a local Fontconfig configuration exposing the installed CJK fonts; raw QA images/configuration remain local. Manifests record final source/artifact hashes and the review scope, not byte-identical reconstruction on arbitrary font/tool versions.

Source, patches, derived analyses and final report files are suitable for the owner-only private repository. Raw model tensors, logits, timings, trace bundles, models and signed products remain local. Publication does not submit the assessment or grant access to the assignment team. The author must review the pending fields and retained limitations before submission.
