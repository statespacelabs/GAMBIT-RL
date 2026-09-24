# Phase 4.4 PPO training and learning-curve evidence

Audited 2026-09-11. Classification: HISTORICAL training, with some resulting weights inherited by CURRENT-PAPER bots. This is not CUDA rebaseline training.

## Scope

The tournament population was built through Phase 4.4g GvG family training, safe repairs, Phase 4.4g2 expansion, and Phase 4.4g3 seeker/long-horizon continuations. The earlier guarded stimulus factory is a separate precursor. No single PPO configuration or run length describes all of these. Recipe values below are source-established; complete frozen execution configs for every historical run are unavailable locally.

## Core GvG PPO implementation

| Item | Implementation |
|---|---|
| Actor | local45 telemetry encoder → one-layer GRU(512 input, 512 hidden) → four continuous means, four log-standard-deviation parameters, four binary logits; scalar value head |
| Initialization | Existing supervised/DAgger/residual expert checkpoints; see family table |
| Opponent | Frozen within each segment; learner side A, stochastic actions |
| Optimizer | Adam |
| PPO clip | 0.1 |
| Discount / GAE | 0.99 / 0.95 |
| Value-loss coefficient | 0.5 (MSE) |
| Effective GvG entropy coefficient | 0.003; see configuration discrepancy below |
| Rollout | Nominal 256 Unity ticks × 2 areas = 512 learner transitions per GvG iteration |
| Recurrent minibatch | Sequence length 16; batch size 4 sequences (up to 64 transitions) |
| Update schedule | Builds recurrent minibatches, uses only batches[0], performs one optimizer step per driver iteration; no full-rollout multi-epoch PPO loop |
| Advantage normalization | Rollout mean/std, epsilon 1e-8 |
| Final-value bootstrap | GvG driver supplies zero at rollout end |
| Normalizer | Frozen local45 normalizer; no update |
| Trainability | Default actor/value path enables all parameters including encoder/GRU; anti-strafe family freezes movement output rows/logstd entries. Encoder branches are in eval mode for deterministic dropout, which does not freeze gradients |
| Base logstd bounds | [-2.5,-1.2], with separate continuation overrides |
| Base gradient clipping | 0.5, with separate continuation overrides |

Sources: [src/rl/online_rl/actor_critic.py:21](/home/amit/Projects/AIMSLAB/RL-GRADER/src/rl/online_rl/actor_critic.py:21); [scripts/phase3v2_b_one_sided_gvg.py:104](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase3v2_b_one_sided_gvg.py:104); [scripts/phase3v2_b_one_sided_gvg.py:144](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase3v2_b_one_sided_gvg.py:144); [scripts/phase3v2_b_one_sided_gvg.py:208](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase3v2_b_one_sided_gvg.py:208); [scripts/phase3v2_ppo_readiness_common.py:130](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase3v2_ppo_readiness_common.py:130); [scripts/phase3v2_ppo_loss_smoke.py:238](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase3v2_ppo_loss_smoke.py:238); [src/rl/online_rl/rollout_buffer.py:82](/home/amit/Projects/AIMSLAB/RL-GRADER/src/rl/online_rl/rollout_buffer.py:82).

## Initial GvG family recipes

| Family | Parent | LR | KL stop threshold | KL anchor coefficient | Requested entropy (not effective GvG) |
|---|---|---:|---:|---:|---:|
| gvg_generalist_roster_mix | dag21_default | 6e-07 | 0.0012 | 0.035 | 0.004 |
| gvg_anti_temporal_selector | dag21_default | 5.5e-07 | 0.001 | 0.045 | 0.004 |
| gvg_anti_v000_visible | visible_residual | 6e-07 | 0.0012 | 0.04 | 0.004 |
| gvg_anti_strafe_pressure | dag21_default | 6.5e-07 | 0.001 | 0.04 | 0.003 |
| gvg_underfire_retaliator | underfire_residual | 6e-07 | 0.001 | 0.045 | 0.004 |
| gvg_evasive_survivor | visible_residual | 5e-07 | 0.001 | 0.055 | 0.006 |
| gvg_search_obstacle_pursuer | dag21_retention_heavy | 4.5e-07 | 0.001 | 0.06 | 0.006 |

Source: [scripts/phase4_4g_common.py:112](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase4_4g_common.py:112).

Initial launcher requests 120 updates/family, one update per segment, 256 ticks, two areas, time scale 5, snapshot stride 10; seeds 44901–44907 on GPUs 1–7. The Python entry point alone defaults to 10 updates/segment, so cite the launcher for this wave. Requested counts are not proof of completed updates. Source: [scripts/phase4_4g_launch_gvg_library_wave_gpu30.sh:8](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase4_4g_launch_gvg_library_wave_gpu30.sh:8).

Opponent categories: frozen roster 60%, snapshot pool 25%, self-mirror 10%, scripted retention 5%. Base spawn mix: close/mid/far/obstacle/search_destroy = 20/30/20/20/10%; search family = 10/15/20/35/20%. These are scheduler weights, not measured realized proportions. Sources: [scripts/phase4_4g_common.py:27](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase4_4g_common.py:27); [scripts/phase4_4g_common.py:74](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase4_4g_common.py:74).

Scripted-retention segments use the separate tiny-PPO driver, with rollout_steps=256 rather than the GvG product 256×2. Do not assume identical transition counts or override behavior. Source: [scripts/phase4_4g_train_gvg_candidate.py:275](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase4_4g_train_gvg_candidate.py:275).

## Repairs and expansions

| Source recipe | LR | Target KL | Grad cap | Anchor | Requested entropy | Logstd bounds |
|---|---:|---:|---:|---:|---:|---|
| Safe repair | 3e-6 | 0.0015 | 0.25 | 2.0 | 0 | [-2.5,-0.8] |
| G2 stable expansion | 1e-5 | 0.003 | 0.25 | 0.06 | 0.002 | [-2.5,-0.6] |
| G2 generalist experimental | 2e-6 | 0.001 | 0.20 | 5.0 | 0 | [-2.5,-1.0] |
| G2 underfire experimental | 1e-6 | 0.0008 | 0.15 | 8.0 | 0 | [-2.5,-1.2] |
| G3 seeker cycles | 3e-6 | 0.0015 | 0.20 | 0.06 | 0.001 | [-2.5,-0.9] |
| G3 very-long continuation | 8e-7 | 0.0007 | 0.10 | 0.15 | 0.0005 | [-2.5,-1.2] |

Requested entropy is subject to the GvG override discrepancy. Sources: [scripts/phase4_4g_launch_safe_ppo_repair_gpu30.sh:48](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase4_4g_launch_safe_ppo_repair_gpu30.sh:48); [scripts/phase4_4g2_launch_wave.py:54](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase4_4g2_launch_wave.py:54); [scripts/phase4_4g3_launch_seeker_cycle.py:67](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase4_4g3_launch_seeker_cycle.py:67); [scripts/phase4_4g3_launch_pool_vlong.py:63](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase4_4g3_launch_pool_vlong.py:63).

Safe repair requests 80 updates, one update/segment. G2 requests 120/80/40 updates depending on lane, across six possible waves. G3 seeker launcher requests 300 updates with 50-update segments; very-long launcher defaults to 100,000 updates with 500-update segments. These are launch requests, not completed training budgets; terminal checkpoints and run logs must establish actual execution.

## Rewards and guards

RewardBuilder adds Unity env reward with coefficient 1 and aim/shoot-bootstrap shaping, minus jerk/spam penalties. Extra Python damage, kill, death and miss coefficients are zero; this does not imply Unity has no combat reward. Do not transplant the later CUDA reward contract into this historical run. Source: [scripts/phase3v2_ppo_readiness_common.py:377](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase3v2_ppo_readiness_common.py:377).

| Family | Aim | Shoot bootstrap | Jerk | Spam |
|---|---:|---:|---:|---:|
| gvg_generalist_roster_mix | 0.9 | 0.035 | 0.001 | 0.001 |
| gvg_anti_temporal_selector | 1.0 | 0.04 | 0.001 | 0.001 |
| gvg_anti_v000_visible | 0.9 | 0.04 | 0.001 | 0.001 |
| gvg_anti_strafe_pressure | 1.0 | 0.055 | 0.0007 | 0.001 |
| gvg_underfire_retaliator | 0.9 | 0.045 | 0.0008 | 0.001 |
| gvg_evasive_survivor | 0.7 | 0.025 | 0.0005 | 0.0005 |
| gvg_search_obstacle_pursuer | 0.8 | 0.03 | 0.0005 | 0.0007 |

The loss also adds a KL anchor and a mean-action saturation penalty (default coefficient 0.03, base threshold 0.65; continuation overrides change threshold). GvG checks absolute approximate KL and look-mean saturation after the optimizer call; it stops subsequent iterations when KL exceeds target or more than 15% of look means exceed absolute 0.95. Finite-value checks, gradient clipping, bounded logstd and frozen-row checks also apply. Guards do not undo the already-applied optimizer step. Sources: [scripts/phase3v2_ppo_loss_smoke.py:295](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase3v2_ppo_loss_smoke.py:295); [scripts/phase3v2_b_one_sided_gvg.py:208](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase3v2_b_one_sided_gvg.py:208).

## Configuration discrepancy / correction to previous audit

The Phase 4.4 GvG wrapper writes entropy_coef into segment_config.yaml, but scripts/phase3v2_b_one_sided_gvg.py:144 does not pass it to RuntimeConfig or apply it later. The effective coefficient in the inspected GvG implementation is RuntimeConfig.entropy_coef=0.003. Family/repair/expansion entropy values are requested values only for this path; scripted-retention PPO does pass the entropy setting. This supersedes the earlier methods-audit table wherever it treated Phase 4 family entropy recipes as effective GvG coefficients. Exact correspondence of every historical run to the inspected source revision remains UNKNOWN.

Evidence: [scripts/phase4_4g_train_gvg_candidate.py:210](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase4_4g_train_gvg_candidate.py:210); [scripts/phase3v2_b_one_sided_gvg.py:144](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase3v2_b_one_sided_gvg.py:144); [scripts/phase3v2_ppo_readiness_common.py:136](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase3v2_ppo_readiness_common.py:136); [scripts/phase4_4g_train_gvg_candidate.py:314](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase4_4g_train_gvg_candidate.py:314).

## Training logs available in archive index

The local filename index lists 2081 Phase 4.4 log members:

- `ppo_train.jsonl`: 93
- `ppo_update_metrics.jsonl`: 93
- `train.jsonl`: 80
- `gvg_train.jsonl`: 1,815

No matching raw Phase 4.4 training logs were found extracted in the working tree. The v002 content-addressed catalog does not include these training logs. The results archive is no longer stored locally; the restore document says it was deleted after earlier selective extraction. The archive remains referenced at:

`s3://aimlabs-ml/projects/amd/gambit/phase6-autonomous-continuation-v004-no-datasets/manual_archives/models_logs_results_backup_archives.tar.zst`

Recorded compressed size: 73,208,253,414 bytes. This audit did not download it or verify every log member’s contents. The inventory is evidence of archived paths, not a fresh content-integrity check. Sources: [temp_resotre/aws_recovery_indexes/PHASE6_MODELS_LOGS_RESULTS_ARCHIVES_PATHS.txt:1](/home/amit/Projects/AIMSLAB/RL-GRADER/temp_resotre/aws_recovery_indexes/PHASE6_MODELS_LOGS_RESULTS_ARCHIVES_PATHS.txt:1); [restore_doc.md:153](/home/amit/Projects/AIMSLAB/RL-GRADER/restore_doc.md:153).

The companion PHASE4_4_PPO_LOG_INVENTORY.csv lists exact members and source-index line numbers. Prioritize `training/gvg_generalist_roster_mix/segments/*/gvg_train.jsonl`, `training/gvg_anti_strafe_pressure/segments/*/{gvg_train,ppo_train}.jsonl`, the safe repair branches, and expansion/seeker branches matching the chosen tournament item.

## What can be plotted

The shared loss logs `return_mean = mean(buffer.returns)`, where returns are GAE advantages plus value predictions, computed from shaped rewards. Label this “Mean GAE return target”, not “mean episode return” or win rate. GvG logs do not record a direct per-episode return series. Scripted-PPO logs additionally contain `reward_mean` and `env_reward_mean` (per-transition metrics). Sources: [scripts/phase3v2_ppo_loss_smoke.py:333](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase3v2_ppo_loss_smoke.py:333); [src/rl/online_rl/rollout_buffer.py:82](/home/amit/Projects/AIMSLAB/RL-GRADER/src/rl/online_rl/rollout_buffer.py:82).

Reconstruct each branch in segment order, preserving opponent/category/spawn changes. Use actual accumulated learner_buffer_transitions for a GvG environment-transition axis, or actual logged optimizer iterations. Local iteration resets at each segment; global_update_start/end are scheduler bookkeeping and can overstate completed iterations after a guard stop. Do not join unrelated branches, parent restarts or scripted/GvG reward metrics as one stationary curve. Source: [scripts/phase4_4g_train_gvg_candidate.py:473](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase4_4g_train_gvg_candidate.py:473).

Four distinct retained checkpoint metric records were recovered locally (duplicates excluded):

| Branch | Mean GAE return target |
|---|---:|
| w01_search_obstacle_pursuer_repaired_expansion | 1599.494141 |
| gvg_generalist_roster_mix | 2191.748779 |
| generalist_safe_from_s050 | 1376.906738 |
| gvg_anti_strafe_pressure | 1337.883545 |

These are isolated points from different branches. They are provided in PHASE4_4_PPO_CHECKPOINT_METRICS.csv with provenance and guard metadata; no continuous training curve can be inferred from them. In particular, three GvG checkpoint records carry guard-stop events. A publication training curve requires recovery of the raw archived log series.
