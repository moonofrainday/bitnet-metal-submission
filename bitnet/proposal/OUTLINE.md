# Proposal structure and requirement coverage

Decision: Metal first on iPhone 14 Pro; retain the stock default; carry a source-level Vulkan portability plan for Adreno/Mali. Use original PoC and Q4 as separately attributed deliverables. Evidence cutoff: 15 November 2026, Asia/Tokyo.

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

Evidence basis: verified clean package commit bbf70e9cc70302534ff87b4f34d4ce52a9670baa, original brief, current iPhone Q4 experiment, preserved public2B and phone smoke logs, and primary sources listed in the proposal. Existing 13 November drafts are historical reference material, not the current result authority.
