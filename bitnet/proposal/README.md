# Mobile BitNet technical proposal

Measured A16 revision,16 November2026. English/Chinese PDF and editable Word editions preserve the reviewed flow figure and historical measurements, and add the physical-phone results for the hardened14a9f103 package. Document rendering performs no inference; the separately recorded current device campaign is included.

Both editions contain28PDF pages and28rendered Word pages; every final page is covered by native-image review or exact image identity to the reviewed preceding render.

## Read and review

- `BitNet_Mobile_GPU_Proposal.pdf`: English technical plan for the assignment.
- `BitNet_Mobile_GPU_Proposal.docx`: editable English companion.
- `PROPOSAL.md`: editable English content source; `OUTLINE.md` maps all eleven requested topics.
- `README_ZH.md`: Chinese companion and section-by-section review entry.
- `PENDING_A16_RESULTS.json`: machine-readable result ledger; measured fields filled, unresolved fields remain null, never zero or a pass.
- [Tonight's test order and commands](../docs/IPHONE_TEST_PLAN_2026-11-16.md).

Section 9 adds the focused M2 Pro attribution supplement. It evaluates the already selected Q projection, rather than claiming it is the largest model bottleneck. Local Q4 GPU benefit is measured, but current layer-0 integration gives no whole-decode improvement. Same-stock full-guard controls identify warm file-identity work as a substantial measured host region; total CPU/GPU/thermal attribution and normal critical-path share remain unresolved. The separate encoder timestamp probe changes scheduling and is not in the normal or phone build.

Section10 retains earlier A16 operator/guard/FFN tables separately and fills the current-package result pages from fresh device evidence:18+18contract cases,20ActQuant cases,two-history complete32Q/33logits checks,same-process stock/QB/Q4,isolated operator timing,CPU diagnostics and formal32-token wall. Initial Fair and post-model Serious operator cohorts are separate. Q4 is locally faster, while full-guard QB and Q4 both regress complete decode under Serious thermal observations. The measured snapshot/key host region is a follow-up target, not a complete CPU/GPU/thermal explanation. [Current derived analysis](../analysis/iphone-qcur0-attribution-2026-11-16/README.md) contains all round aggregates and CPU-only reproduction; raw evidence remains local.

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
