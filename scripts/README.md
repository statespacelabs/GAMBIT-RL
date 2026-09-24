# Script entry points

For argument-driven entries, run `python -m scripts.NAME --help` from the release
root. The reacquisition, peeking and navigation-population orchestrators are fixed
experiment drivers without a help parser; review their source/configuration before
launching them. The shell
distillation launcher is invoked with `bash scripts/train_combat_distillation.sh`.
All implementations are copied from the original repository; only naming,
imports and location-dependent paths were adapted.

| Task | Script/module | Inputs |
| --- | --- | --- |
| Prepare demonstration transitions | `prepare_demonstration_dataset` | Rendered sessions, split manifest, encoder, telemetry statistics |
| Train human BC | `train_human_bc`, `train_human_bc_residual` | BC YAML and transition dataset |
| Prepare command targets and session splits | `build_command_targets`, `split_demonstration_sessions`, `check_command_dataset` | Transition and window manifests |
| Distill and initialize recurrent combat | `train_combat_distillation.sh` | Production distillation configs and ancestral encoder |
| Collect local45 teacher labels | `collect_combat_teacher` | Combat Unity build |
| Fit the local45 normalizer | `build_combat_normalizer` | Teacher rollout shards |
| Train combat BC | `train_combat_teacher_bc` | Teacher shards, parent, normalizer |
| Collect/relabel combat learner states | `collect_combat_dagger`, `relabel_combat_dagger` | Parent, opponents, Unity build and observations |
| Train combat DAgger/residual | `train_combat_dagger`, `train_combat_residual` | Relabeled data and frozen reference |
| Train combat specialist PPO | `train_combat_specialist` | Family ID, parent/roster manifests, Unity build |
| Run the underlying combat PPO driver | `train_combat_ppo` | Explicit policy/runtime flags |
| Generate procedural layout definitions | `generate_procedural_layouts` | Generator seed and split settings |
| Collect navigation teacher/DAgger data | `collect_navigation_dagger` | `launch` or `run-shard` subcommand, certified build, layout and teacher config |
| Train navigator BC | `train_navigation_bc` | DAgger manifest, branch ID, frozen combat checkpoint and normalizers |
| Initialize/update navigator PPO | `train_navigation_ppo` | Initial navigator and collected actor231/critic428 rollout NPZ |
| Collect navigation PPO rollouts | `collect_navigation_rollouts` | Hunter checkpoint, combat expert, certified Unity runtime |
| Run hunter curriculum | `train_hunter_curriculum` | Original eight-worker curriculum and parent artifacts |
| Run reacquisition curriculum | `train_reacquisition_curriculum` | Hunter parent and its certification |
| Run local-geometry peeking curriculum | `train_peeking_curriculum` | Reacquirer parent and its certification |
| Run navigation population continuation | `train_navigation_population` | Original specialist population and parents |
| Evaluate navigator | `evaluate_navigation` | Checkpoints, normalizers, layout/runtime settings |
| Prepare/run frozen calibration | `calibrate_examiner_bank` | `prepare` or `run`; exact checkpoint/build SHA identities |
| Refit ratings and select examiner bank | `select_examiner_bank` | Recorded calibration output directory |
| Run one examiner match | `examiner_match` | Two policies, checksums, normalizers, Unity and layout settings |
| Assess player skill | `assess_player_skill` | Frozen proxy demo prerequisite files, or import its mathematical functions |

Shared modules such as `navigation_policy`, `navigation_ppo`, `combat_runtime`,
`local_geometry_peeker`, `tactical_handoff` and the runtime adapters are imported
by these entry points. They are not independent training commands. The retained
`combat_ppo_canary` is a historical small-run diagnostic, not the manuscript's
full training recipe.

See [TRAINING.md](../docs/TRAINING.md) for ordered commands, dependencies, outputs
and the distinction between original multi-GPU orchestration and single-branch
training.

