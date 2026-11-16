# Tonight iPhone14Pro/A16 test order

| Order | Experiment | Required evidence / stop rule |
| --- | --- | --- |
| 1 | Device/build identity and actual18-case contract/cache tests | New source-bound package; correct entries; negative configurations rejected; investigate any valid-load failure before timing |
| 2 | Same-build20-case ActQuant prerequisite plus both-history Q4regression | All32targetdispatches/GPUcompletion, FP64 andstockparity,33logits/probe/fallback; preserve old official-reference strict1 |
| 3 | ActQuantON matched stock/QBcontrol | Guard32,custom0,samePSO/ABI/bindings/launch; identical all32W/X/Y and33logits, neutral diagnostics |
| 4 | CPU/cache/encoder + naturalCBdiagnostic | Calibrated clock, allattempts/success, boundedrecords/status; noextra sync/CBsplit; separateinstrumentationoverheadcontrol |
| 5 | Default-stock critical-path observation | NaturalCB/operatorfamilyinventory andCPUschedulinggaps; peroplatency unavailable ifAPIcannot observe it withoutperturbation |
| 6 | Formal AA/SQB/SQ4 wall pairing | Bothhistories,3rounds×21pairs/contrast,2equalreplaywarmups, balancedrotatingorder, trace/readback/metricsOFF; temperature/time trend retained |
| 7 | Analyze and decide | CPUcost vsGPU/queueobservations vsunresolvedthermaldrift; no causeclaim without evidence; choose nexttarget only aftercostshare evidence |

Preflight/builds and Mac correctness can be done now. Phone execution waits for the evening. This is a priority order, not a promise that unsupported per-dispatch/counters can be collected. Optional deeper GPUfamily slicing runs only after normal-scheduling evidence is examined. Results and thresholds remain scoped perACCEPTANCE.md.

Full acceptance and command/source digests are saved with the local timestamped experiment. No A16 measurements have been taken for this new preparation build. Historical guard wall deltas do not price its new cache checks.

## What the daytime Mac investigation already establishes

- Both fixed token histories and the 1,203,563,808-byte public GGUF retain their recorded SHA-256 identities.
- The supplied old A16 traces contain selected Q/ActQuant events, not a complete per-family GPU timing profile. An unobserved output head or FFN is unavailable, never zero latency.
- Existing completion trace lines repeat their enclosing command-buffer interval; summing them as independent kernel times would double count.
- Collecting existing fusion statistics forces a different command-buffer configuration. It is unsuitable as neutral instrumentation of the original schedule.
- iPhone's existing concurrency configuration differs from Mac's. Source/API inspection does not transfer a Mac scheduling explanation to A16.
- Xcode27.2 provides Metal System Trace and Time Profiler templates. Local template/options discovery succeeded without recording; actual A16 trace/counter visibility is not yet validated.
- New diagnostic timestamps stay in separate CPU/GPU domains, retain overlap and report bounded-record completeness. They do not automatically produce per-dispatch GPU durations or an additive wall-time breakdown.

Phone result fields remain unverified until actual evening execution. No Mac inference performance campaign is included.

## Decision rules after the phone data

- A valid QB control requires32successful full guards,0customdispatches and same-process stock PSO/ABI/live bindings. A fallback that skips checks invalidates this control.
- A reproducible QB-minus-S wall delta is a path effect; CPU guard/getter/encode spans and original CB timelines narrow attribution. Nested/concurrent CPU spans and GPU CB intervals cannot be subtracted as an additive wall partition.
- If CPU evidence identifies one expensive repeated operation, test one safety-preserving host change next. Do not rewrite the shader to explain a same-shader wall effect.
- If the delta overlaps A/A, order or temperature/time drift, retain an inconclusive result. Historical positive wall deltas are not erased by a different campaign.
- Whole-model operator priority remains open until normal-scheduling timing evidence exists. Mixed CB durations, family counts and656.7MB stored head weights do not establish per-family critical-path fractions.

## Daytime validation completed

Fresh Mac native build and source-bound iPhoneOS signed package both succeeded. Two histories passed same-context S/QB/Q4 verification, all32Q FP64 checks and stock parity, actual stock PSO/ABI/live-binding identity and all33 callback-null-stock logits. The Mac collector smoke verified six bounded,complete diagnostic paths (32targetcalls and64existingCBs/path), without deriving comparative performance statistics. MTL0 regressions: TQ2_059/59, ACT_QUANT16/16. Production-predicate CPU controls:21startup and49backend; graph inventory parser10tests. These establish preparation correctness, not new A16 runtime/performance or full-model reference acceptance. Source/build hashes are retained with the local experiment.

Preparation source commit: `14a9f103c23b5a24bd0da1e0dbc5a0d5a37ade3a`. The binaries were built from this exact source snapshot before commit; embedded Git metadata names parent `ad4e0238b987e52ea3bf12bbf9c79612255ad27f`. File hashes bind the tested code. Additive patch `bitnet/patches/a16-q-attribution-preparation.patch`, SHA-256 `112905a114df45905fd4196fb15094e643df56ac984d8ea2eccd76e2ba82f8ad`, is already applied and targets that submission baseline, not pristine QVAC.

Host guard/getter/encoding spans measure elapsed time on worker threads, including waits and descheduling; they are not on-core instruction time or CPU cycles. Scheduled/completed host timestamps mark callback delivery and may include callback scheduling delay; they are not precise GPU queue boundaries. Do not derive queue wait by subtracting uncorrelated CPU/GPU timestamps.
