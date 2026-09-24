# GAMBIT source-port inventory

This release ports **199 individual files** from the existing repository: **122 byte-identical** copies and **77 adapted** copies. All Python implementations and the training shell script originate in the repository. No new model, trainer, reward, optimizer, controller or rating algorithm was implemented.

The scope follows `paper/sections/appendix_new.tex`. The package is named `gambit`; Unity is maintained separately and no Unity source/build is included. Source paths below are relative to the original RL-GRADER checkout. Destination links are relative to this release.

Full original and ported SHA-256 hashes, together with change descriptions, are in [port_manifest.json](port_manifest.json). They describe the actual local working-tree files, including restored/untracked sources; they do not claim all files belong to a single Git revision. Original source hashes were checked again after the port.

## Naming and permitted adaptations

| Original label/namespace | Publication name |
| --- | --- |
| `src` | `gambit` |
| `src.phase2` | `gambit.dataset_preparation` |
| Phase 1 | Demonstration representation pretraining |
| Phase 2 | Demonstration preparation and human behavioral cloning |
| Phase 3 / 3v2 | Recurrent combat initialization, teacher BC and DAgger |
| Phase 4.4 | Conservative combat PPO and specialist continuation |
| Phase 5 | Fair navigation, hunter, reacquisition, peeking and population continuation |
| Phase 6 paper evaluation | Examiner calibration, bank selection and player assessment |

Mechanical edits update package imports, filenames, subprocess command paths, dynamically loaded runtime locations and machine-specific default paths. Configurations retain their original numerical settings. Legacy schema IDs, checkpoint keys, class names, candidate IDs, protocol statuses (including historical `TANDEMFPS_...` statuses) and artifact paths remain where frozen files depend on them.

The only new executable control flow is the small `--analysis-only` option in `scripts/select_examiner_bank.py`: it runs the original numerical analysis and record audits, labels the output `GAMBIT_RECORDED_CALIBRATION_REPLAY`, and returns before checking absent external checkpoint/build files. The default certification path and all mathematical functions are retained. This option enables replay of bundled evidence while Unity/weights remain separate.

## Individually ported files

| Original repository file | Ported file | Treatment |
| --- | --- | --- |
| `06_map_general_hunter/CURRICULUM.json` | [06_map_general_hunter/CURRICULUM.json](06_map_general_hunter/CURRICULUM.json) | Verbatim |
| `configs/bc_train.yaml` | [configs/bc_train.yaml](configs/bc_train.yaml) | Names/imports/paths |
| `configs/bc_train_advanced.yaml` | [configs/bc_train_advanced.yaml](configs/bc_train_advanced.yaml) | Names/imports/paths |
| `configs/distill.yaml` | [configs/distill.yaml](configs/distill.yaml) | Verbatim |
| `configs/distill_production.yaml` | [configs/distill_production.yaml](configs/distill_production.yaml) | Names/imports/paths |
| `configs/ppo_production.yaml` | [configs/ppo_production.yaml](configs/ppo_production.yaml) | Names/imports/paths |
| `configs/ppo_train.yaml` | [configs/ppo_train.yaml](configs/ppo_train.yaml) | Verbatim |
| `configs/gambit_phase1.yaml` | [configs/representation_pretraining.yaml](configs/representation_pretraining.yaml) | Verbatim |
| `configs/train_v2_large.yaml` | [configs/representation_pretraining_large.yaml](configs/representation_pretraining_large.yaml) | Names/imports/paths |
| `experiments/phase5_map_general_headless_hunter/02_layouts/HELDOUT_LAYOUTS.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/HELDOUT_LAYOUTS.json](experiments/phase5_map_general_headless_hunter/02_layouts/HELDOUT_LAYOUTS.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/TRAIN_LAYOUTS.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/TRAIN_LAYOUTS.json](experiments/phase5_map_general_headless_hunter/02_layouts/TRAIN_LAYOUTS.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/VALIDATION_LAYOUTS.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/VALIDATION_LAYOUTS.json](experiments/phase5_map_general_headless_hunter/02_layouts/VALIDATION_LAYOUTS.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/LAYOUT_RUNTIME_SMOKE.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/LAYOUT_RUNTIME_SMOKE.json](experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/LAYOUT_RUNTIME_SMOKE.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/heldout_u_traps_dead_ends_560000/layout_runtime_audit.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/heldout_u_traps_dead_ends_560000/layout_runtime_audit.json](experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/heldout_u_traps_dead_ends_560000/layout_runtime_audit.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/heldout_u_traps_dead_ends_560000/runtime_summary.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/heldout_u_traps_dead_ends_560000/runtime_summary.json](experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/heldout_u_traps_dead_ends_560000/runtime_summary.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_narrow_wide_transitions_540004/layout_runtime_audit.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_narrow_wide_transitions_540004/layout_runtime_audit.json](experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_narrow_wide_transitions_540004/layout_runtime_audit.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_narrow_wide_transitions_540004/runtime_summary.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_narrow_wide_transitions_540004/runtime_summary.json](experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_narrow_wide_transitions_540004/runtime_summary.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_open_scattered_cover_540000/layout_runtime_audit.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_open_scattered_cover_540000/layout_runtime_audit.json](experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_open_scattered_cover_540000/layout_runtime_audit.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_open_scattered_cover_540000/runtime_summary.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_open_scattered_cover_540000/runtime_summary.json](experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_open_scattered_cover_540000/runtime_summary.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_parallel_corridors_540001/layout_runtime_audit.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_parallel_corridors_540001/layout_runtime_audit.json](experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_parallel_corridors_540001/layout_runtime_audit.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_parallel_corridors_540001/runtime_summary.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_parallel_corridors_540001/runtime_summary.json](experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_parallel_corridors_540001/runtime_summary.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_pillars_alternating_cover_540003/layout_runtime_audit.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_pillars_alternating_cover_540003/layout_runtime_audit.json](experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_pillars_alternating_cover_540003/layout_runtime_audit.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_pillars_alternating_cover_540003/runtime_summary.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_pillars_alternating_cover_540003/runtime_summary.json](experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_pillars_alternating_cover_540003/runtime_summary.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_rooms_doorways_540002/layout_runtime_audit.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_rooms_doorways_540002/layout_runtime_audit.json](experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_rooms_doorways_540002/layout_runtime_audit.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_rooms_doorways_540002/runtime_summary.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_rooms_doorways_540002/runtime_summary.json](experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/train_rooms_doorways_540002/runtime_summary.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/validation_open_scattered_cover_550000/layout_runtime_audit.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/validation_open_scattered_cover_550000/layout_runtime_audit.json](experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/validation_open_scattered_cover_550000/layout_runtime_audit.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/validation_open_scattered_cover_550000/runtime_summary.json` | [experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/validation_open_scattered_cover_550000/runtime_summary.json](experiments/phase5_map_general_headless_hunter/02_layouts/runtime_smoke/validation_open_scattered_cover_550000/runtime_summary.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_telemetry/normalizers/phase5_actor_obs_v001_normalizer_v001.json` | [experiments/phase5_map_general_headless_hunter/02_telemetry/normalizers/phase5_actor_obs_v001_normalizer_v001.json](experiments/phase5_map_general_headless_hunter/02_telemetry/normalizers/phase5_actor_obs_v001_normalizer_v001.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_telemetry/schemas/phase5_actor_obs_v001.json` | [experiments/phase5_map_general_headless_hunter/02_telemetry/schemas/phase5_actor_obs_v001.json](experiments/phase5_map_general_headless_hunter/02_telemetry/schemas/phase5_actor_obs_v001.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_telemetry/schemas/phase5_privileged_critic_obs_v001.json` | [experiments/phase5_map_general_headless_hunter/02_telemetry/schemas/phase5_privileged_critic_obs_v001.json](experiments/phase5_map_general_headless_hunter/02_telemetry/schemas/phase5_privileged_critic_obs_v001.json) | Verbatim |
| `experiments/phase5_map_general_headless_hunter/02_telemetry/schemas/phase5_tactical_state_v001.json` | [experiments/phase5_map_general_headless_hunter/02_telemetry/schemas/phase5_tactical_state_v001.json](experiments/phase5_map_general_headless_hunter/02_telemetry/schemas/phase5_tactical_state_v001.json) | Verbatim |
| `src/__init__.py` | [gambit/__init__.py](gambit/__init__.py) | Names/imports/paths |
| `src/phase2/__init__.py` | [gambit/dataset_preparation/__init__.py](gambit/dataset_preparation/__init__.py) | Verbatim |
| `src/phase2/action_extraction.py` | [gambit/dataset_preparation/action_extraction.py](gambit/dataset_preparation/action_extraction.py) | Verbatim |
| `src/phase2/configs.py` | [gambit/dataset_preparation/configs.py](gambit/dataset_preparation/configs.py) | Verbatim |
| `src/phase2/latent_cache.py` | [gambit/dataset_preparation/latent_cache.py](gambit/dataset_preparation/latent_cache.py) | Names/imports/paths |
| `src/phase2/reward_extraction.py` | [gambit/dataset_preparation/reward_extraction.py](gambit/dataset_preparation/reward_extraction.py) | Verbatim |
| `src/phase2/schemas.py` | [gambit/dataset_preparation/schemas.py](gambit/dataset_preparation/schemas.py) | Verbatim |
| `src/phase2/transition_manifest.py` | [gambit/dataset_preparation/transition_manifest.py](gambit/dataset_preparation/transition_manifest.py) | Verbatim |
| `src/phase2/validate_dataset.py` | [gambit/dataset_preparation/validate_dataset.py](gambit/dataset_preparation/validate_dataset.py) | Verbatim |
| `src/phase2/window_manifest.py` | [gambit/dataset_preparation/window_manifest.py](gambit/dataset_preparation/window_manifest.py) | Verbatim |
| `src/datasets/__init__.py` | [gambit/datasets/__init__.py](gambit/datasets/__init__.py) | Verbatim |
| `src/datasets/builders.py` | [gambit/datasets/builders.py](gambit/datasets/builders.py) | Verbatim |
| `src/datasets/builders_ddp.py` | [gambit/datasets/builders_ddp.py](gambit/datasets/builders_ddp.py) | Names/imports/paths |
| `src/datasets/clip_dataset.py` | [gambit/datasets/clip_dataset.py](gambit/datasets/clip_dataset.py) | Verbatim |
| `src/datasets/collate.py` | [gambit/datasets/collate.py](gambit/datasets/collate.py) | Names/imports/paths |
| `src/datasets/manifest.py` | [gambit/datasets/manifest.py](gambit/datasets/manifest.py) | Verbatim |
| `src/datasets/pk_sampler.py` | [gambit/datasets/pk_sampler.py](gambit/datasets/pk_sampler.py) | Verbatim |
| `src/datasets/telemetry_preprocess.py` | [gambit/datasets/telemetry_preprocess.py](gambit/datasets/telemetry_preprocess.py) | Verbatim |
| `src/datasets/video_decode.py` | [gambit/datasets/video_decode.py](gambit/datasets/video_decode.py) | Verbatim |
| `src/models/__init__.py` | [gambit/models/__init__.py](gambit/models/__init__.py) | Verbatim |
| `src/models/gated_fusion.py` | [gambit/models/gated_fusion.py](gambit/models/gated_fusion.py) | Verbatim |
| `src/models/inquisitor_encoder.py` | [gambit/models/inquisitor_encoder.py](gambit/models/inquisitor_encoder.py) | Verbatim |
| `src/models/loss.py` | [gambit/models/loss.py](gambit/models/loss.py) | Verbatim |
| `src/models/projector_head.py` | [gambit/models/projector_head.py](gambit/models/projector_head.py) | Verbatim |
| `src/models/rhythm_branch.py` | [gambit/models/rhythm_branch.py](gambit/models/rhythm_branch.py) | Verbatim |
| `src/models/telemetry_fusion.py` | [gambit/models/telemetry_fusion.py](gambit/models/telemetry_fusion.py) | Verbatim |
| `src/models/temporal_tcn.py` | [gambit/models/temporal_tcn.py](gambit/models/temporal_tcn.py) | Verbatim |
| `src/models/temporal_transformer.py` | [gambit/models/temporal_transformer.py](gambit/models/temporal_transformer.py) | Verbatim |
| `src/models/types.py` | [gambit/models/types.py](gambit/models/types.py) | Verbatim |
| `src/models/video_encoder.py` | [gambit/models/video_encoder.py](gambit/models/video_encoder.py) | Verbatim |
| `src/models/view_branch.py` | [gambit/models/view_branch.py](gambit/models/view_branch.py) | Verbatim |
| `src/models/where_branch.py` | [gambit/models/where_branch.py](gambit/models/where_branch.py) | Verbatim |
| `src/rl/__init__.py` | [gambit/rl/__init__.py](gambit/rl/__init__.py) | Verbatim |
| `src/rl/datasets/__init__.py` | [gambit/rl/datasets/__init__.py](gambit/rl/datasets/__init__.py) | Verbatim |
| `src/rl/datasets/transition_collate.py` | [gambit/rl/datasets/transition_collate.py](gambit/rl/datasets/transition_collate.py) | Verbatim |
| `src/rl/datasets/transition_dataset.py` | [gambit/rl/datasets/transition_dataset.py](gambit/rl/datasets/transition_dataset.py) | Verbatim |
| `src/rl/models/__init__.py` | [gambit/rl/models/__init__.py](gambit/rl/models/__init__.py) | Verbatim |
| `src/rl/models/actor_critic.py` | [gambit/rl/models/actor_critic.py](gambit/rl/models/actor_critic.py) | Verbatim |
| `src/rl/models/advanced_policy.py` | [gambit/rl/models/advanced_policy.py](gambit/rl/models/advanced_policy.py) | Verbatim |
| `src/rl/models/policy.py` | [gambit/rl/models/policy.py](gambit/rl/models/policy.py) | Verbatim |
| `src/rl/models/q_network.py` | [gambit/rl/models/q_network.py](gambit/rl/models/q_network.py) | Verbatim |
| `src/rl/models/v_network.py` | [gambit/rl/models/v_network.py](gambit/rl/models/v_network.py) | Verbatim |
| `src/rl/online_rl/__init__.py` | [gambit/rl/online_rl/__init__.py](gambit/rl/online_rl/__init__.py) | Verbatim |
| `src/rl/online_rl/action_schema.py` | [gambit/rl/online_rl/action_schema.py](gambit/rl/online_rl/action_schema.py) | Verbatim |
| `src/rl/online_rl/actor_critic.py` | [gambit/rl/online_rl/actor_critic.py](gambit/rl/online_rl/actor_critic.py) | Verbatim |
| `src/rl/online_rl/build_telemetry_npz.py` | [gambit/rl/online_rl/build_telemetry_npz.py](gambit/rl/online_rl/build_telemetry_npz.py) | Names/imports/paths |
| `src/rl/online_rl/checkpoint_registry.py` | [gambit/rl/online_rl/checkpoint_registry.py](gambit/rl/online_rl/checkpoint_registry.py) | Verbatim |
| `src/rl/online_rl/core_action_projector.py` | [gambit/rl/online_rl/core_action_projector.py](gambit/rl/online_rl/core_action_projector.py) | Verbatim |
| `src/rl/online_rl/distillation_dataset.py` | [gambit/rl/online_rl/distillation_dataset.py](gambit/rl/online_rl/distillation_dataset.py) | Verbatim |
| `src/rl/online_rl/distributed_utils.py` | [gambit/rl/online_rl/distributed_utils.py](gambit/rl/online_rl/distributed_utils.py) | Verbatim |
| `src/rl/online_rl/evaluate_policy.py` | [gambit/rl/online_rl/evaluate_policy.py](gambit/rl/online_rl/evaluate_policy.py) | Verbatim |
| `src/rl/online_rl/hybrid_distribution.py` | [gambit/rl/online_rl/hybrid_distribution.py](gambit/rl/online_rl/hybrid_distribution.py) | Verbatim |
| `src/rl/online_rl/invariant_check.py` | [gambit/rl/online_rl/invariant_check.py](gambit/rl/online_rl/invariant_check.py) | Names/imports/paths |
| `src/rl/online_rl/lr_schedule.py` | [gambit/rl/online_rl/lr_schedule.py](gambit/rl/online_rl/lr_schedule.py) | Names/imports/paths |
| `src/rl/online_rl/multi_area_runner.py` | [gambit/rl/online_rl/multi_area_runner.py](gambit/rl/online_rl/multi_area_runner.py) | Verbatim |
| `src/rl/online_rl/obs_normalizer.py` | [gambit/rl/online_rl/obs_normalizer.py](gambit/rl/online_rl/obs_normalizer.py) | Names/imports/paths |
| `src/rl/online_rl/opponent_pool.py` | [gambit/rl/online_rl/opponent_pool.py](gambit/rl/online_rl/opponent_pool.py) | Verbatim |
| `src/rl/online_rl/ppo_loss.py` | [gambit/rl/online_rl/ppo_loss.py](gambit/rl/online_rl/ppo_loss.py) | Verbatim |
| `src/rl/online_rl/reward_builder.py` | [gambit/rl/online_rl/reward_builder.py](gambit/rl/online_rl/reward_builder.py) | Verbatim |
| `src/rl/online_rl/rollout_buffer.py` | [gambit/rl/online_rl/rollout_buffer.py](gambit/rl/online_rl/rollout_buffer.py) | Verbatim |
| `src/rl/online_rl/selfplay_runner.py` | [gambit/rl/online_rl/selfplay_runner.py](gambit/rl/online_rl/selfplay_runner.py) | Names/imports/paths |
| `src/rl/online_rl/shot_geometry_summary.py` | [gambit/rl/online_rl/shot_geometry_summary.py](gambit/rl/online_rl/shot_geometry_summary.py) | Verbatim |
| `src/rl/online_rl/telemetry_encoder.py` | [gambit/rl/online_rl/telemetry_encoder.py](gambit/rl/online_rl/telemetry_encoder.py) | Names/imports/paths |
| `src/rl/online_rl/train_ppo.py` | [gambit/rl/online_rl/train_ppo.py](gambit/rl/online_rl/train_ppo.py) | Names/imports/paths |
| `src/rl/online_rl/vectorized_buffer.py` | [gambit/rl/online_rl/vectorized_buffer.py](gambit/rl/online_rl/vectorized_buffer.py) | Verbatim |
| `src/rl/online_rl/vectorized_runner.py` | [gambit/rl/online_rl/vectorized_runner.py](gambit/rl/online_rl/vectorized_runner.py) | Verbatim |
| `src/rl/online_rl/visual_actor_critic.py` | [gambit/rl/online_rl/visual_actor_critic.py](gambit/rl/online_rl/visual_actor_critic.py) | Names/imports/paths |
| `src/rl/online_rl/visual_runner.py` | [gambit/rl/online_rl/visual_runner.py](gambit/rl/online_rl/visual_runner.py) | Verbatim |
| `src/rl/online_rl/weapon_state_summary.py` | [gambit/rl/online_rl/weapon_state_summary.py](gambit/rl/online_rl/weapon_state_summary.py) | Verbatim |
| `src/rl/training/__init__.py` | [gambit/rl/training/__init__.py](gambit/rl/training/__init__.py) | Verbatim |
| `src/rl/training/eval_bc.py` | [gambit/rl/training/eval_bc.py](gambit/rl/training/eval_bc.py) | Names/imports/paths |
| `src/rl/training/eval_iql.py` | [gambit/rl/training/eval_iql.py](gambit/rl/training/eval_iql.py) | Names/imports/paths |
| `src/rl/training/train_bc.py` | [gambit/rl/training/train_bc.py](gambit/rl/training/train_bc.py) | Names/imports/paths |
| `src/rl/training/train_bc_advanced.py` | [gambit/rl/training/train_bc_advanced.py](gambit/rl/training/train_bc_advanced.py) | Names/imports/paths |
| `src/rl/training/train_iql.py` | [gambit/rl/training/train_iql.py](gambit/rl/training/train_iql.py) | Names/imports/paths |
| `src/rl/utils/__init__.py` | [gambit/rl/utils/__init__.py](gambit/rl/utils/__init__.py) | Verbatim |
| `src/rl/utils/checkpointing.py` | [gambit/rl/utils/checkpointing.py](gambit/rl/utils/checkpointing.py) | Verbatim |
| `src/rl/utils/metrics.py` | [gambit/rl/utils/metrics.py](gambit/rl/utils/metrics.py) | Verbatim |
| `src/rl/utils/normalization.py` | [gambit/rl/utils/normalization.py](gambit/rl/utils/normalization.py) | Verbatim |
| `src/training/__init__.py` | [gambit/training/__init__.py](gambit/training/__init__.py) | Verbatim |
| `src/training/checkpoint.py` | [gambit/training/checkpoint.py](gambit/training/checkpoint.py) | Verbatim |
| `src/training/config.py` | [gambit/training/config.py](gambit/training/config.py) | Verbatim |
| `src/training/ddp_utils.py` | [gambit/training/ddp_utils.py](gambit/training/ddp_utils.py) | Verbatim |
| `src/training/distributed_pk_sampler.py` | [gambit/training/distributed_pk_sampler.py](gambit/training/distributed_pk_sampler.py) | Verbatim |
| `src/training/ema.py` | [gambit/training/ema.py](gambit/training/ema.py) | Verbatim |
| `src/training/gather.py` | [gambit/training/gather.py](gambit/training/gather.py) | Verbatim |
| `src/training/train_ddp.py` | [gambit/training/train_ddp.py](gambit/training/train_ddp.py) | Names/imports/paths |
| `ACTOR231_PPO_TRAINING_DATA.md` | [reference/ACTOR231_PPO_TRAINING_DATA.md](reference/ACTOR231_PPO_TRAINING_DATA.md) | Verbatim |
| `ACTOR231_SCHEMA.json` | [reference/ACTOR231_SCHEMA.json](reference/ACTOR231_SCHEMA.json) | Verbatim |
| `ACTOR231_SCHEMA.md` | [reference/ACTOR231_SCHEMA.md](reference/ACTOR231_SCHEMA.md) | Verbatim |
| `LOCAL45_RHYTHM_SLOT_HISTORY.md` | [reference/LOCAL45_RHYTHM_SLOT_HISTORY.md](reference/LOCAL45_RHYTHM_SLOT_HISTORY.md) | Verbatim |
| `PHASE4_4_PPO_TRAINING_DETAILS.md` | [reference/PHASE4_4_PPO_TRAINING_DETAILS.md](reference/PHASE4_4_PPO_TRAINING_DETAILS.md) | Verbatim |
| `experiments/tandemfps_paper_draft1_package_v001/evidence/examiner_bank/experiments/paper_goal3_cuda_examiner_bank_v001/00_contract/ANCHOR_CONTROLLER_SPEC.json` | [reference/examiner_bank/00_contract/ANCHOR_CONTROLLER_SPEC.json](reference/examiner_bank/00_contract/ANCHOR_CONTROLLER_SPEC.json) | Verbatim |
| `experiments/tandemfps_paper_draft1_package_v001/evidence/examiner_bank/experiments/paper_goal3_cuda_examiner_bank_v001/00_contract/CALIBRATION_CONTRACT.json` | [reference/examiner_bank/00_contract/CALIBRATION_CONTRACT.json](reference/examiner_bank/00_contract/CALIBRATION_CONTRACT.json) | Verbatim |
| `experiments/tandemfps_paper_draft1_package_v001/evidence/examiner_bank/experiments/paper_goal3_cuda_examiner_bank_v001/00_contract/CANDIDATE_LIBRARY.json` | [reference/examiner_bank/00_contract/CANDIDATE_LIBRARY.json](reference/examiner_bank/00_contract/CANDIDATE_LIBRARY.json) | Verbatim |
| `experiments/tandemfps_paper_draft1_package_v001/evidence/examiner_bank/experiments/paper_goal3_cuda_examiner_bank_v001/00_contract/ROUND_ROBIN_SCHEDULE.json` | [reference/examiner_bank/00_contract/ROUND_ROBIN_SCHEDULE.json](reference/examiner_bank/00_contract/ROUND_ROBIN_SCHEDULE.json) | Verbatim |
| `experiments/tandemfps_paper_draft1_package_v001/evidence/examiner_bank/experiments/paper_goal3_cuda_examiner_bank_v001/02_analysis/ALL_CANDIDATE_METRICS.json` | [reference/examiner_bank/02_analysis/ALL_CANDIDATE_METRICS.json](reference/examiner_bank/02_analysis/ALL_CANDIDATE_METRICS.json) | Verbatim |
| `experiments/tandemfps_paper_draft1_package_v001/evidence/examiner_bank/experiments/paper_goal3_cuda_examiner_bank_v001/02_analysis/PAIR_DIAGNOSTICS.csv` | [reference/examiner_bank/02_analysis/PAIR_DIAGNOSTICS.csv](reference/examiner_bank/02_analysis/PAIR_DIAGNOSTICS.csv) | Verbatim |
| `experiments/tandemfps_paper_draft1_package_v001/evidence/examiner_bank/experiments/paper_goal3_cuda_examiner_bank_v001/02_analysis/SELECTION_AUDIT.json` | [reference/examiner_bank/02_analysis/SELECTION_AUDIT.json](reference/examiner_bank/02_analysis/SELECTION_AUDIT.json) | Verbatim |
| `experiments/tandemfps_paper_draft1_package_v001/evidence/examiner_bank/experiments/paper_goal3_cuda_examiner_bank_v001/02_analysis/TECHNICAL_AUDIT.json` | [reference/examiner_bank/02_analysis/TECHNICAL_AUDIT.json](reference/examiner_bank/02_analysis/TECHNICAL_AUDIT.json) | Verbatim |
| `experiments/tandemfps_paper_draft1_package_v001/evidence/examiner_bank/experiments/paper_goal3_cuda_examiner_bank_v001/BOT_BANK_HASHES.sha256` | [reference/examiner_bank/BOT_BANK_HASHES.sha256](reference/examiner_bank/BOT_BANK_HASHES.sha256) | Verbatim |
| `experiments/tandemfps_paper_draft1_package_v001/evidence/examiner_bank/experiments/paper_goal3_cuda_examiner_bank_v001/BOT_CALIBRATION_REPORT.md` | [reference/examiner_bank/BOT_CALIBRATION_REPORT.md](reference/examiner_bank/BOT_CALIBRATION_REPORT.md) | Verbatim |
| `experiments/tandemfps_paper_draft1_package_v001/evidence/examiner_bank/experiments/paper_goal3_cuda_examiner_bank_v001/BOT_PAYOFF_MATRIX.csv` | [reference/examiner_bank/BOT_PAYOFF_MATRIX.csv](reference/examiner_bank/BOT_PAYOFF_MATRIX.csv) | Verbatim |
| `experiments/tandemfps_paper_draft1_package_v001/evidence/examiner_bank/experiments/paper_goal3_cuda_examiner_bank_v001/BOT_RATINGS_CUDA.json` | [reference/examiner_bank/BOT_RATINGS_CUDA.json](reference/examiner_bank/BOT_RATINGS_CUDA.json) | Verbatim |
| `experiments/tandemfps_paper_draft1_package_v001/evidence/examiner_bank/experiments/paper_goal3_cuda_examiner_bank_v001/FINAL_CUDA_BOT_BANK.json` | [reference/examiner_bank/FINAL_CUDA_BOT_BANK.json](reference/examiner_bank/FINAL_CUDA_BOT_BANK.json) | Verbatim |
| `experiments/phase4_active_diagnostic/phase4_6_final_integration/laptop_demo_package_v001/phase4_6_laptop_demo_v001_20260713_212403/requirements-linux-cpu.lock` | [requirements-unity-historical.txt](requirements-unity-historical.txt) | Verbatim |
| `src/__init__.py` | [scripts/__init__.py](scripts/__init__.py) | Verbatim |
| `scripts/phase6_run_bayesian_scoring_demo.py` | [scripts/assess_player_skill.py](scripts/assess_player_skill.py) | Names/imports/paths |
| `scripts/phase6_examiner_action_audit_match.py` | [scripts/audit_examiner_actions.py](scripts/audit_examiner_actions.py) | Names/imports/paths |
| `scripts/phase5_autonomous_baseline.py` | [scripts/autonomous_combat_baseline.py](scripts/autonomous_combat_baseline.py) | Names/imports/paths |
| `scripts/phase3v2_c_build_local45_normalizer.py` | [scripts/build_combat_normalizer.py](scripts/build_combat_normalizer.py) | Names/imports/paths |
| `scripts/build_phase3_action_targets.py` | [scripts/build_command_targets.py](scripts/build_command_targets.py) | Names/imports/paths |
| `scripts/phase6_calibrate_examiner_bank.py` | [scripts/calibrate_examiner_bank.py](scripts/calibrate_examiner_bank.py) | Names/imports/paths |
| `scripts/check_phase3_dataset.py` | [scripts/check_command_dataset.py](scripts/check_command_dataset.py) | Names/imports/paths |
| `scripts/check_recurrent_distillation_delta.py` | [scripts/check_recurrent_initialization.py](scripts/check_recurrent_initialization.py) | Names/imports/paths |
| `scripts/phase3v2_collect_gvg_dagger_v2_policy_states.py` | [scripts/collect_combat_dagger.py](scripts/collect_combat_dagger.py) | Names/imports/paths |
| `scripts/phase3v2_c_collect_local45_teacher.py` | [scripts/collect_combat_teacher.py](scripts/collect_combat_teacher.py) | Names/imports/paths |
| `scripts/phase5_dagger_collect.py` | [scripts/collect_navigation_dagger.py](scripts/collect_navigation_dagger.py) | Names/imports/paths |
| `scripts/phase5_hunter_live.py` | [scripts/collect_navigation_rollouts.py](scripts/collect_navigation_rollouts.py) | Names/imports/paths |
| `scripts/phase3v2_c_localobs_common.py` | [scripts/combat_observations.py](scripts/combat_observations.py) | Verbatim |
| `scripts/phase3v2_tiny_ppo_canary.py` | [scripts/combat_ppo_canary.py](scripts/combat_ppo_canary.py) | Names/imports/paths |
| `scripts/phase3v2_ppo_loss_smoke.py` | [scripts/combat_ppo_update.py](scripts/combat_ppo_update.py) | Names/imports/paths |
| `scripts/phase3v2_ppo_readiness_common.py` | [scripts/combat_runtime.py](scripts/combat_runtime.py) | Names/imports/paths |
| `scripts/phase4_4g_common.py` | [scripts/combat_specialist_families.py](scripts/combat_specialist_families.py) | Names/imports/paths |
| `experiments/phase6_research_demo_v003/01_package/laptop_demo_package_v003_research/repo_overlay/experiments/phase6_population_league_psro/13_final_selection/phase6_release_match_batch.py` | [scripts/dual_policy_runtime.py](scripts/dual_policy_runtime.py) | Names/imports/paths |
| `experiments/phase6_research_demo_v003/01_package/laptop_demo_package_v003_research/repo_overlay/experiments/phase6_demo_release_v003/01_showcase/d2_recovery/minimum_viable_hunter/tools/minimum_cert_match.py` | [scripts/encounter_runtime.py](scripts/encounter_runtime.py) | Names/imports/paths |
| `scripts/phase5_navigator_eval.py` | [scripts/evaluate_navigation.py](scripts/evaluate_navigation.py) | Names/imports/paths |
| `scripts/phase6_examiner_bank_match.py` | [scripts/examiner_match.py](scripts/examiner_match.py) | Names/imports/paths |
| `experiments/phase6_research_demo_v003/01_package/laptop_demo_package_v003_research/runtime/research_repaired_match.py` | [scripts/frozen_hybrid_runtime.py](scripts/frozen_hybrid_runtime.py) | Names/imports/paths |
| `scripts/phase5_layout_generator.py` | [scripts/generate_procedural_layouts.py](scripts/generate_procedural_layouts.py) | Verbatim |
| `experiments/phase6_research_demo_v003/01_package/laptop_demo_package_v003_research/repo_overlay/experiments/phase6_demo_release_v003/01_showcase/d2_recovery/minimum_viable_hunter/tools/minimum_hunter_match.py` | [scripts/hunter_runtime.py](scripts/hunter_runtime.py) | Names/imports/paths |
| `scripts/phase5_peeker.py` | [scripts/local_geometry_peeker.py](scripts/local_geometry_peeker.py) | Names/imports/paths |
| `scripts/phase5_navigator.py` | [scripts/navigation_policy.py](scripts/navigation_policy.py) | Verbatim |
| `scripts/phase5_league.py` | [scripts/navigation_population.py](scripts/navigation_population.py) | Names/imports/paths |
| `scripts/phase5_hunter_ppo.py` | [scripts/navigation_ppo.py](scripts/navigation_ppo.py) | Names/imports/paths |
| `scripts/run_phase2_full.py` | [scripts/prepare_demonstration_dataset.py](scripts/prepare_demonstration_dataset.py) | Names/imports/paths |
| `scripts/phase3v2_c_relabel_gvg_dagger_v21.py` | [scripts/relabel_combat_dagger.py](scripts/relabel_combat_dagger.py) | Verbatim |
| `experiments/phase6_research_demo_v003/01_package/laptop_demo_package_v003_research/repo_overlay/experiments/phase6_demo_release_v003/01_showcase/d2_recovery/minimum_viable_hunter/tools/hard_negative_repaired_match.py` | [scripts/repaired_navigation_runtime.py](scripts/repaired_navigation_runtime.py) | Names/imports/paths |
| `experiments/phase6_research_demo_v003/01_package/laptop_demo_package_v003_research/runtime/research_hud.py` | [scripts/runtime_observer.py](scripts/runtime_observer.py) | Names/imports/paths |
| `scripts/phase6_bot_screen_match_v002.py` | [scripts/screen_anchor_controllers.py](scripts/screen_anchor_controllers.py) | Names/imports/paths |
| `scripts/phase6_bot_screen_match.py` | [scripts/screen_learned_controllers.py](scripts/screen_learned_controllers.py) | Names/imports/paths |
| `scripts/phase6_analyze_examiner_bank.py` | [scripts/select_examiner_bank.py](scripts/select_examiner_bank.py) | Replay CLI adaptation |
| `scripts/make_phase3_splits.py` | [scripts/split_demonstration_sessions.py](scripts/split_demonstration_sessions.py) | Verbatim |
| `experiments/phase6_population_league_psro/03_tournament/phase6_symmetric_tournament.py` | [scripts/symmetric_tournament.py](scripts/symmetric_tournament.py) | Names/imports/paths |
| `scripts/phase5_tactical_handoff.py` | [scripts/tactical_handoff.py](scripts/tactical_handoff.py) | Names/imports/paths |
| `scripts/phase3v2_train_gvg_dagger_v2_supervised.py` | [scripts/train_combat_dagger.py](scripts/train_combat_dagger.py) | Names/imports/paths |
| `scripts/run_phase3_distill_production.sh` | [scripts/train_combat_distillation.sh](scripts/train_combat_distillation.sh) | Names/imports/paths |
| `scripts/phase3v2_b_one_sided_gvg.py` | [scripts/train_combat_ppo.py](scripts/train_combat_ppo.py) | Names/imports/paths |
| `scripts/phase3v2_c_train_v22_bounded_residual.py` | [scripts/train_combat_residual.py](scripts/train_combat_residual.py) | Names/imports/paths |
| `scripts/phase4_4g_train_gvg_candidate.py` | [scripts/train_combat_specialist.py](scripts/train_combat_specialist.py) | Names/imports/paths |
| `scripts/phase3v2_bc_actor_head_train_v21.py` | [scripts/train_combat_teacher_bc.py](scripts/train_combat_teacher_bc.py) | Names/imports/paths |
| `src/rl/training/train_bc.py` | [scripts/train_human_bc.py](scripts/train_human_bc.py) | Names/imports/paths |
| `src/rl/training/train_bc_advanced.py` | [scripts/train_human_bc_residual.py](scripts/train_human_bc_residual.py) | Names/imports/paths |
| `scripts/phase5_hunter_orchestrate.py` | [scripts/train_hunter_curriculum.py](scripts/train_hunter_curriculum.py) | Names/imports/paths |
| `scripts/phase5_navigator_train.py` | [scripts/train_navigation_bc.py](scripts/train_navigation_bc.py) | Names/imports/paths |
| `scripts/phase5_league_orchestrate.py` | [scripts/train_navigation_population.py](scripts/train_navigation_population.py) | Names/imports/paths |
| `scripts/phase5_hunter_train.py` | [scripts/train_navigation_ppo.py](scripts/train_navigation_ppo.py) | Names/imports/paths |
| `scripts/phase5_peek_orchestrate.py` | [scripts/train_peeking_curriculum.py](scripts/train_peeking_curriculum.py) | Names/imports/paths |
| `scripts/phase5_reacquire_orchestrate.py` | [scripts/train_reacquisition_curriculum.py](scripts/train_reacquisition_curriculum.py) | Names/imports/paths |
| `scripts/phase4_4g_safe_port_allocator.py` | [scripts/unity_port_allocator.py](scripts/unity_port_allocator.py) | Verbatim |
| `experiments/phase6_research_demo_v003/01_package/laptop_demo_package_v003_research/repo_overlay/experiments/phase6_demo_release_v003/01_showcase/tools/phase6_showcase_match.py` | [scripts/visible_combat_runtime.py](scripts/visible_combat_runtime.py) | Names/imports/paths |
| `experiments/phase6_research_demo_v003/01_package/laptop_demo_package_v003_research/repo_overlay/experiments/phase6_population_league_psro/07_exploiters/phase6_exploiter_eval_core.py` | [scripts/visible_tactics.py](scripts/visible_tactics.py) | Names/imports/paths |
| `scripts/write_phase3_report.py` | [scripts/write_distillation_report.py](scripts/write_distillation_report.py) | Verbatim |
| `tests/test_phase3_production.py` | [tests/test_command_distillation.py](tests/test_command_distillation.py) | Names/imports/paths |
| `tests/test_phase5_peeker.py` | [tests/test_local_geometry_peeker.py](tests/test_local_geometry_peeker.py) | Names/imports/paths |
| `tests/test_phase5_reacquire.py` | [tests/test_reacquisition.py](tests/test_reacquisition.py) | Names/imports/paths |
| `tests/test_phase5_tactical_handoff.py` | [tests/test_tactical_handoff.py](tests/test_tactical_handoff.py) | Names/imports/paths |
| `tests/test_vectorized_ppo.py` | [tests/test_vectorized_ppo.py](tests/test_vectorized_ppo.py) | Names/imports/paths |

## Archived encounter files

[`reference/calibration_inputs.tar.gz`](reference/calibration_inputs.tar.gz) groups **5,285 existing files**: 2,640 `matches.jsonl` records, 2,640 `run_summary.json` files, one execution manifest, and four contract JSON files. The record bodies are unchanged. Only their container/archive was created for submission.

The complete per-file list, original source paths, archive member paths and SHA-256 hashes are in [calibration_archive_members.json](reference/calibration_archive_members.json). Records come from `temp_resotre/vast_evidence/paper_goal3_cuda_examiner_bank_v001/01_round_robin/`; contract files come from the original frozen paper evidence package. No encounter or training data was fabricated.

## New packaging and documentation

These files describe, install or audit reused code; they are not ported research implementations:

| New file | Purpose |
| --- | --- |
| [README.md](README.md) | Installation, overview and recorded replay |
| [pyproject.toml](pyproject.toml) | Package metadata, dependencies and console entry points |
| [MANIFEST.in](MANIFEST.in) | Source-distribution contents |
| [requirements.txt](requirements.txt) | Minimal numerical dependencies |
| [.gitignore](.gitignore) | Generated-output exclusions |
| [scripts/README.md](scripts/README.md) | Method-named entry-point index |
| [docs/TRAINING.md](docs/TRAINING.md) | Ordered commands and original orchestration constraints |
| [docs/METHODS.md](docs/METHODS.md) | Manuscript-to-code mapping and method distinctions |
| [docs/ARTIFACTS.md](docs/ARTIFACTS.md) | Input contracts and separate Unity integration |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Checks performed and limits |
| [docs/validation_results.json](docs/validation_results.json) | Measured validation results |
| [port_manifest.json](port_manifest.json) | Per-file provenance and hashes |
| [ported.md](ported.md) | This inventory |
| [reference/calibration_archive_members.json](reference/calibration_archive_members.json) | Per-member provenance of compressed original records |

## Material not ported

Unity source/scenes/builds (a separate package), human demonstration payloads, model checkpoints, development logs, unrelated experiment backups, experimental Gen 3 training and the older Davidson rating package are not bundled. A few historical IQL helpers remain only because the reused package initializers/evaluation code import them; they are not a claimed lineage of the final paper bank.

Validation: 29 reused CPU tests pass; all 2,640 archived calibration records pass their original audit; the selected roles match the frozen bank; rating differences are below 5e-13. See [VALIDATION.md](docs/VALIDATION.md) for environments and external work not performed.
