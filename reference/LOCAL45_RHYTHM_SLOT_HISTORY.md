# LOCAL45_RHYTHM_SLOT_HISTORY

Audit date: 2026-09-09. All indices are zero-based. This concerns the local45 combat input, not actor231.

**Finding:** slots 24–36 are inherited categorical event-count bins. Their original use was counting raw event types 1–13; the current Unity sensor clears them and only populates bucket 0 at index 23. Historical production fire used ID 13 and therefore maps to index 36 under the original direct-ID encoding.

## Evidence and explanation

### original_use — HISTORICAL

The original 45-dimensional telemetry concatenated where10 + view13 + rhythm22. The rhythm block contained 14 per-frame event-count bins, indexed by raw action_type 0..13, followed by eight summary/timing features. Thus observation index j=23+k represented the count of events of type k in that frame. Counts can exceed one; the documentation phrase one-hot action counts should not be read as a strict one-hot constraint.

Evidence (ESTABLISHED): [H1: README.md:354](/home/amit/Projects/AIMSLAB/RL-GRADER/README.md:354); [H2: Impl1.md:75](/home/amit/Projects/AIMSLAB/RL-GRADER/Impl1.md:75); [H4: src/datasets/telemetry_preprocess.py:124](/home/amit/Projects/AIMSLAB/RL-GRADER/src/datasets/telemetry_preprocess.py:124); [H7: unity_project/DataPipeline/Assets/_Scripts/DatasetExporter.cs:230](/home/amit/Projects/AIMSLAB/RL-GRADER/unity_project/DataPipeline/Assets/_Scripts/DatasetExporter.cs:230)

### why_fourteen — HISTORICAL

README documents action_vocab_size=14 as maximum action_type plus one, and preprocessing implements that inference rule. The production report identifies fire as event ID 13; under this direct indexing, including ID 13 requires 14 bins. This explains a dense numeric vocabulary, not evidence that fourteen distinct gameplay actions were available. Whether the first designer chose 14 specifically after inspecting this same production dataset is UNKNOWN.

Evidence (ESTABLISHED_RULE_WITH_INFERENCE): [H1: README.md:354](/home/amit/Projects/AIMSLAB/RL-GRADER/README.md:354); [H3: experiments/phase3/distill_production/REPORT.md:20](/home/amit/Projects/AIMSLAB/RL-GRADER/experiments/phase3/distill_production/REPORT.md:20); [H4: src/datasets/telemetry_preprocess.py:124](/home/amit/Projects/AIMSLAB/RL-GRADER/src/datasets/telemetry_preprocess.py:124)

### why_current_zero — CURRENT-PAPER

The current Unity sensor clears all 14 bins on each observation, writes 1 only into event bucket 0 when lastCommandWasShoot is true, and copies all bins into indices 23..36. There are no writes for buckets 1..13. Indices 24..36 therefore equal zero by construction, before normalization. They are not fourteen policy outputs or fields removed by learned feature selection.

Evidence (ESTABLISHED): [C1: temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:262](/home/amit/Projects/AIMSLAB/RL-GRADER/temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:262); [C2: temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:49](/home/amit/Projects/AIMSLAB/RL-GRADER/temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:49); [C4: unity_project/BotArenaPhase5/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:262](/home/amit/Projects/AIMSLAB/RL-GRADER/unity_project/BotArenaPhase5/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:262)

### why_keep_width — CURRENT-PAPER

The retained width is consistent with inheritance of the encoder input contract: Unity documents the schema as the encoder-trained 10+13+22 layout, and the historical telemetry encoder reuses the pretrained 22-input rhythm branch. Shape compatibility is supported by code. An explicit historical design note explaining why developers left live non-shoot event tracking unimplemented, or deliberately reserved exactly thirteen empty fields, was NOT FOUND; that motivation is UNKNOWN. Matching width does not establish matching event semantics.

Evidence (SUPPORTED_COMPATIBILITY_INFERENCE): [C3: temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:6](/home/amit/Projects/AIMSLAB/RL-GRADER/temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:6); [H6: src/rl/online_rl/telemetry_encoder.py:1](/home/amit/Projects/AIMSLAB/RL-GRADER/src/rl/online_rl/telemetry_encoder.py:1)

### production_fire_mapping — HISTORICAL

The production distillation report records event schema {fire:13,reload:null,jump:null,crouch:null} and says the source telemetry records fire events only. Combining this with unchanged raw event IDs and direct count indexing maps historical fire counts to local45 index 23+13=36. Current live Unity instead encodes shooting in index 23 (event bucket 0). Therefore slot 36 had a meaningful original use under the documented production schema; it is incorrect to call the full 24..36 range always unused historically. This audit does not verify the contents of every historical tensor or prove a continuous checkpoint lineage across those schemas.

Evidence (REPORT_PLUS_INDEX_ARITHMETIC): [H3: experiments/phase3/distill_production/REPORT.md:20](/home/amit/Projects/AIMSLAB/RL-GRADER/experiments/phase3/distill_production/REPORT.md:20); [H4: src/datasets/telemetry_preprocess.py:124](/home/amit/Projects/AIMSLAB/RL-GRADER/src/datasets/telemetry_preprocess.py:124); [H7: unity_project/DataPipeline/Assets/_Scripts/DatasetExporter.cs:230](/home/amit/Projects/AIMSLAB/RL-GRADER/unity_project/DataPipeline/Assets/_Scripts/DatasetExporter.cs:230); [C2: temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:49](/home/amit/Projects/AIMSLAB/RL-GRADER/temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:49)

### default_mapping_is_not_production — HISTORICAL

A generic older action extractor supplies configurable defaults shoot=0, reload=5, jump=6, crouch=7. Under those defaults, local45 positions 23,28,29,30 correspond to those event IDs. These defaults are not an authoritative event ontology for the production source, which reports fire=13 and unavailable reload/jump/crouch labels. Exact original gameplay names for the other IDs are UNKNOWN.

Evidence (ESTABLISHED): [H5: src/phase2/action_extraction.py:23](/home/amit/Projects/AIMSLAB/RL-GRADER/src/phase2/action_extraction.py:23); [H3: experiments/phase3/distill_production/REPORT.md:20](/home/amit/Projects/AIMSLAB/RL-GRADER/experiments/phase3/distill_production/REPORT.md:20)

### separate_action14 — HISTORICAL

The Phase 2 fourteen-dimensional action vector mentioned in summary.md is a separate action-summary representation containing movement/look statistics, action flags/counts and totals. It is not the fourteen event-ID bins in the rhythm observation and must not be used to name indices 23..36.

Evidence (ESTABLISHED): [H8: summary.md:18](/home/amit/Projects/AIMSLAB/RL-GRADER/summary.md:18); [H9: src/phase2/action_extraction.py:114](/home/amit/Projects/AIMSLAB/RL-GRADER/src/phase2/action_extraction.py:114)

## Index mapping

The “generic default” column describes configurable historical extractor defaults, not the production event ontology. UNKNOWN means the reviewed sources do not establish the gameplay name.

| Index | Raw event ID | Original count field | Historical production meaning | Generic default only | Current live value |
|---:|---:|---|---|---|---|
| 23 | 0 | `count(action_type=0)` | UNKNOWN (production report records only fire ID 13) | shoot | 1 if lastCommandWasShoot else 0 |
| 24 | 1 | `count(action_type=1)` | UNKNOWN (production report records only fire ID 13) | UNKNOWN | 0 (constant) |
| 25 | 2 | `count(action_type=2)` | UNKNOWN (production report records only fire ID 13) | UNKNOWN | 0 (constant) |
| 26 | 3 | `count(action_type=3)` | UNKNOWN (production report records only fire ID 13) | UNKNOWN | 0 (constant) |
| 27 | 4 | `count(action_type=4)` | UNKNOWN (production report records only fire ID 13) | UNKNOWN | 0 (constant) |
| 28 | 5 | `count(action_type=5)` | UNKNOWN (production report records only fire ID 13) | reload | 0 (constant) |
| 29 | 6 | `count(action_type=6)` | UNKNOWN (production report records only fire ID 13) | jump | 0 (constant) |
| 30 | 7 | `count(action_type=7)` | UNKNOWN (production report records only fire ID 13) | crouch | 0 (constant) |
| 31 | 8 | `count(action_type=8)` | UNKNOWN (production report records only fire ID 13) | UNKNOWN | 0 (constant) |
| 32 | 9 | `count(action_type=9)` | UNKNOWN (production report records only fire ID 13) | UNKNOWN | 0 (constant) |
| 33 | 10 | `count(action_type=10)` | UNKNOWN (production report records only fire ID 13) | UNKNOWN | 0 (constant) |
| 34 | 11 | `count(action_type=11)` | UNKNOWN (production report records only fire ID 13) | UNKNOWN | 0 (constant) |
| 35 | 12 | `count(action_type=12)` | UNKNOWN (production report records only fire ID 13) | UNKNOWN | 0 (constant) |
| 36 | 13 | `count(action_type=13)` | Fire; placement derived from report + direct indexing | UNKNOWN | 0 (constant) |

Table evidence: [H3: experiments/phase3/distill_production/REPORT.md:20](/home/amit/Projects/AIMSLAB/RL-GRADER/experiments/phase3/distill_production/REPORT.md:20); [H4: src/datasets/telemetry_preprocess.py:124](/home/amit/Projects/AIMSLAB/RL-GRADER/src/datasets/telemetry_preprocess.py:124); [H5: src/phase2/action_extraction.py:23](/home/amit/Projects/AIMSLAB/RL-GRADER/src/phase2/action_extraction.py:23); [C1: temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:262](/home/amit/Projects/AIMSLAB/RL-GRADER/temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:262); [C2: temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:49](/home/amit/Projects/AIMSLAB/RL-GRADER/temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:49)

## Paper wording

> The 45-dimensional combat observation retains the telemetry encoder’s original 10+13+22 layout. In the 22-dimensional rhythm block, the live sensor populates only the shooting-event bin; the remaining 13 event-count bins (indices 24–36, zero-based) are fixed to zero.

Do not describe these as thirteen active behavioral features. Do not claim the historical fire ID and current Unity shoot ID are identical.

## Limits and search coverage

Repository-wide Markdown text searches including hidden/ignored files and restored mirrors; inspected relevant matches and implementation code. This is not a claim that every document was manually read in full. Searched 1,096 Markdown paths; the path inventory is recorded in the companion JSON.

- A complete semantic enum mapping original game event IDs 0..13 to gameplay names was not found.
- An explicit Markdown decision to leave thirteen event bins empty in live Unity was not found.
- Why live tracking was implemented only for shooting is UNKNOWN.
- The audit does not establish whether every historical checkpoint encountered fire in slot 36 or which remapping, if any, was applied outside the inspected pipeline.

## Source ledger

| ID | Status | File and line | SHA-256 |
|---|---|---|---|
| H1 | HISTORICAL | [README.md:354](/home/amit/Projects/AIMSLAB/RL-GRADER/README.md:354) | `37ed02d6c364f6c709bd26ffe528e624b31a69aa2ab9bcd07dbe544feeb2e75f` |
| H2 | HISTORICAL | [Impl1.md:75](/home/amit/Projects/AIMSLAB/RL-GRADER/Impl1.md:75) | `2404cf58b8ded2418f356f2b7701bd0e55e3ddf5aa11142278406f7b8c32e96a` |
| H3 | HISTORICAL | [experiments/phase3/distill_production/REPORT.md:20](/home/amit/Projects/AIMSLAB/RL-GRADER/experiments/phase3/distill_production/REPORT.md:20) | `590fb578c1e9120aa00873a1f186435e35a74baa98a796210d8e58eba4d7cd7b` |
| H4 | HISTORICAL | [src/datasets/telemetry_preprocess.py:124](/home/amit/Projects/AIMSLAB/RL-GRADER/src/datasets/telemetry_preprocess.py:124) | `92ea23b3e2b7eee322e3383452430730bc76c685cce2bbc4704c2afc1d5c158d` |
| H5 | HISTORICAL | [src/phase2/action_extraction.py:23](/home/amit/Projects/AIMSLAB/RL-GRADER/src/phase2/action_extraction.py:23) | `fc8330ef47deff4ee43868ca7032ea7d353a8f49a59311d52f7da0ffd3db8c63` |
| H6 | HISTORICAL | [src/rl/online_rl/telemetry_encoder.py:1](/home/amit/Projects/AIMSLAB/RL-GRADER/src/rl/online_rl/telemetry_encoder.py:1) | `856ad998301eab8e5b5a614035b2789309df9701508aa44384f86f823ac9d47f` |
| H7 | HISTORICAL | [unity_project/DataPipeline/Assets/_Scripts/DatasetExporter.cs:230](/home/amit/Projects/AIMSLAB/RL-GRADER/unity_project/DataPipeline/Assets/_Scripts/DatasetExporter.cs:230) | `d443d507e0ad076ba39b0b438f3e1ea666f57e84e0d023d67b8277d661743b62` |
| H8 | HISTORICAL | [summary.md:18](/home/amit/Projects/AIMSLAB/RL-GRADER/summary.md:18) | `504ae564ee9b9fc64098f4ee76a6127cdc7a23a740339f2e6255dedf3b46a396` |
| H9 | HISTORICAL | [src/phase2/action_extraction.py:114](/home/amit/Projects/AIMSLAB/RL-GRADER/src/phase2/action_extraction.py:114) | `fc8330ef47deff4ee43868ca7032ea7d353a8f49a59311d52f7da0ffd3db8c63` |
| C1 | CURRENT-PAPER | [temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:262](/home/amit/Projects/AIMSLAB/RL-GRADER/temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:262) | `555dc333eb8bcfefbc783449a19b9db94363e370f36240b66e8137ee080f9c1d` |
| C2 | CURRENT-PAPER | [temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:49](/home/amit/Projects/AIMSLAB/RL-GRADER/temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:49) | `555dc333eb8bcfefbc783449a19b9db94363e370f36240b66e8137ee080f9c1d` |
| C3 | CURRENT-PAPER | [temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:6](/home/amit/Projects/AIMSLAB/RL-GRADER/temp_restore/vast_ai/BotArenaRLDemo/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:6) | `555dc333eb8bcfefbc783449a19b9db94363e370f36240b66e8137ee080f9c1d` |
| C4 | CURRENT-PAPER | [unity_project/BotArenaPhase5/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:262](/home/amit/Projects/AIMSLAB/RL-GRADER/unity_project/BotArenaPhase5/Assets/Scripts/BotArena/Controllers/GambitAgentController.cs:262) | `555dc333eb8bcfefbc783449a19b9db94363e370f36240b66e8137ee080f9c1d` |
