# Proposal structure and requirement coverage

Decision: Metal first on iPhone 14 Pro; retain the stock default; carry a source-level Vulkan portability plan for Adreno/Mali. Use original PoC and Q4 as separately attributed deliverables. Evidence cutoff: 16 November 2026, Asia/Tokyo. Required source/PDF/PoC/sample-I/O delivery is separate from stretch speed, energy and Android-execution targets.

| Brief item | Planned report section |
|---|---|
| 1 Kernel design | 3 Model representation; 4 Kernel and sample I O |
| 2 llama.cpp changes and cross platform support | 5 Metal integration; 6 Vulkan extension |
| 3 Profiling and targets | 9 Profiling; 10 Measurements; 12 Success criteria |
| 4 Dependencies | 9 Toolchain and SDK matrix |
| 5 Optimization strategy | 11 Optimization and mobile constraints |
| 6 Benchmarks | 7 Correctness; 10 Measurements; 12 Success criteria |
| 7 Power and hardware limits | 11 Mobile constraints; 12 Power evaluation |
| 8 Schedule and checkpoints | 13 Ten day schedule |
| 9 Risks and mitigation | 14 Risks |
| 10 Assumptions and dependencies | 3 Checkpoint contract; 14 Assumptions |
| 11 Additional constraints | 2 Scope; 15 Delivery and reproducibility |
| PoC and sample harness | 4, 7 and 15 |
| Source scripts and PDF | 15 plus editable proposal source |
| Interview justification | Source walkthrough, measured negative outcomes, acceptance boundaries |

Evidence basis: historical clean package commit bbf70e9cc70302534ff87b4f34d4ce52a9670baa, tested ActQuant phone source 4c249e3, evidence/docs snapshot e84732c, original brief, legacy ActQuant-OFF Q4 measurements, completed A16 ActQuant same-input checks and preserved full-model strict exit 1, and the primary sources listed in the proposal. The joint route from edbd109 passes its own two-history stock-ON Q-only qualification and measures a 2.61%-2.68% A16 isolated-operator reduction; joint full-decode and corpus/PPL remain unmeasured. No trusted-reference result is inferred from stock parity. Existing 13 November drafts are historical reference material, not the current result authority.

FFN/fusion branches are consolidated as optional performance exploration in Section 10, with mechanism, measured scope, stock/direct contrasts, numerical limits and selection decisions. The primary delivery chain remains semantics -> physical-phone kernel -> independent harness -> real graph -> fair timing -> reproducible package.

Current report revision adds a focused M2 Pro selected-Q attribution page in Section 9 and a blank/PENDING current-package A16 page in Section 10. Qcur-0 is selected for a real, bounded single-layer kernel contract; normal critical-path priority remains unproved. The current canonical-contract package must pass new phone gates before its metrics can be filled. Historical reference strict exit 1 is preserved. No device or GPU tests run in this document edit.

Current hardened14a9f103 A16 update: both histories pass all32Q FP64 and all33logit stock parity plus18+18contract controls. Same-build formal stock/QB/Q4 decode is now measured; no end-to-end gain. InitialFair and post-modelSerious isolated Q4 gains remain separate. CPU snapshot/key region is an observed hotspot, not complete causal attribution. Trusted-model strict1 remains unchanged. Derived evidence: `bitnet/analysis/iphone-qcur0-attribution-2026-11-16`.
