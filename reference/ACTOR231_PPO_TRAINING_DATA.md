# Actor231 training-curve data availability

Audit: 2026-09-11. HISTORICAL Phase 5 training and league continuations; includes ancestry used by CURRENT-PAPER bots. This audit covers the four canonical directories below, not every later actor231 experiment.

**Finding:** Local stage/window-level optimizer metrics and rollout summaries exist. Exact per-episode training return series were not located.

| Family | Training results | Results with accepted updates | Collection summaries |
|---|---:|---:|---:|
| 06_map_general_hunter | 67 | 25 | 68 |
| 06_reacquire | 16 | 8 | 8 |
| 07_local_geometry_peeker | 16 | 16 | 8 |
| 08_league | 168 | 160 | 160 |

267 training results include 209 records with at least one accepted update. The other 58 are initialization or zero-accepted-update records; their zero means must not be plotted as measured optimization losses. There are 244 collection summaries. These counts refer to files, not independent seeds or completed training runs.

## Available exports

- ACTOR231_PPO_TRAINING_METRICS.csv: all training result records, with source paths, stage, environment-step labels, requested/accepted updates, KL stops, KL, losses, LR, BC weight and checkpoint hashes.
- ACTOR231_PPO_COLLECTION_METRICS.csv: source paths, collection decisions, stage/split/preset, reported reward-component totals and terminal counts. These are not reconstructed episode returns.
- ACTOR231_KITER_CONTINUATION_TRAINING_CURVE.csv: ten successive w10–w19 records from kiter_escape_exploiter_continuation_B, sorted by reported environment_steps. This is a continuation segment, not the whole parent lineage.

Example source: [08_league/snapshots/continuations/kiter_escape_exploiter_continuation_B/w19/training_result.json:1](/home/amit/Projects/AIMSLAB/RL-GRADER/08_league/snapshots/continuations/kiter_escape_exploiter_continuation_B/w19/training_result.json:1). Its final window reports four accepted updates out of four requested, no KL stop, mean KL 0.0042775, total loss 0.00183614, and reported environment_steps=20480.

## PPO implementation

- Actor input: 231 fair features; privileged critic input: 428 features. External combat expert remains frozen and excluded from this optimizer.
- AdamW; branch-dependent initial LR, commonly 1e-4, decayed to a 3e-5 floor. BC auxiliary commonly decreases from 0.2 toward a 0.05 floor.
- Clip=0.1; two optimization epochs per randomly sampled batch; default maximum batch size 256; grad norm cap=0.5; KL threshold=0.01 checked before updating.
- The actor uses stored recurrent hidden state, with a one-step observation in the inspected loss. The PPO policy term is masked to hidden-enemy, non-intervened samples.
- Advantage is immediate reward minus detached critic value. Critic target is immediate reward. This loss does not implement multi-step GAE or use gamma/lambda, and has no entropy term. Do not copy the Phase 4.4 GAE description onto it.
- Additional terms include BC, action anchoring, tactical-mode supervision and smoothness; the exact coefficients are in code/checkpoint metadata.

Sources: [scripts/phase5_hunter_train.py:110](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase5_hunter_train.py:110); [scripts/phase5_hunter_train.py:140](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase5_hunter_train.py:140); [scripts/phase5_hunter_ppo.py:359](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase5_hunter_ppo.py:359).

## Plotting limits

Training results store means over accepted optimizer steps within each stage/window, not every individual optimizer step. Use separate panels/series for named branches, preserve continuation boundaries, and omit or explicitly mark zero-update records. A recorded status=PASS does not mean all requested updates were accepted.

The environment_steps field is supplied by the orchestrator; collection summaries also record candidate_decisions and fixed_environment_steps. They are different units. Keep the axis labelled “reported training-step budget” unless its relationship to cumulative decisions has been verified for that branch.

No rollout.npz files were found locally in these four canonical directories. The live collector would save reward, agent_id, episode_id, fixed_step and done arrays. Recovery of those arrays would permit episode-level aggregation, with incomplete boundary episodes handled explicitly. The local results-archive filename index also contains no rollout.npz member under these four roots; this does not prove they are absent from every backup. Source: [scripts/phase5_hunter_live.py:479](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase5_hunter_live.py:479).

Do not simply add terminal-count rewards to component totals and call it exact return: terminal counts increment even when no pending training record exists, whereas a terminal reward is appended only when one does. Source: [scripts/phase5_hunter_live.py:359](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase5_hunter_live.py:359).

The initial actor231 navigator BC/DAgger sweep is supervised training. Its train/validation imitation losses must be labelled separately from these PPO metrics. Source: [scripts/phase5_navigator_train.py:150](/home/amit/Projects/AIMSLAB/RL-GRADER/scripts/phase5_navigator_train.py:150).
