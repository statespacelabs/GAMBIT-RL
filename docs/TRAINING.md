# Training and evaluation

Run commands from `code_for_paper/` after editable installation. Variables such
as `$COMBAT_PARENT`, `$UNITY_ENV` and `$DAGGER_MANIFEST` refer to your actual
artifacts. The Unity project is supplied separately. All training implementations
are reused from the repository; no replacement trainer was written.

These commands select existing branches and expose the workflow. They do not
reconstruct every historical continuation producing a published checkpoint.
See [ARTIFACTS.md](ARTIFACTS.md) for inputs and [METHODS.md](METHODS.md) for the
important differences between combat PPO and navigation PPO.

## Demonstration representation and human BC

This historical initialization precedes local45 combat training. The original
multimodal encoder is not an extra network in the final deployed actor. The
human-BC window target has 14 components, distinct from eight deployed controls.

```bash
# Edit dataset, checkpoint and log paths in the YAML first.
torchrun --standalone --nproc_per_node=1 -m gambit.training.train_ddp \
  --config configs/representation_pretraining_large.yaml

python -m scripts.prepare_demonstration_dataset \
  --rendered-dir "$RENDERED_SESSIONS" \
  --output-dir artifacts/demonstrations/transitions \
  --checkpoint "$ENCODER_CHECKPOINT" \
  --telemetry-stats "$TELEMETRY_STATS" \
  --joint-manifest "$SESSION_SPLITS" --device cuda

python -m scripts.train_human_bc --config configs/bc_train.yaml
# Alternative inherited residual-MLP BC implementation:
python -m scripts.train_human_bc_residual --config configs/bc_train_advanced.yaml
```

Preserve session-level splits. The manuscript reports 111,680 / 12,712 / 14,709
training/validation/test transitions from 1,644 sessions. Human data is not
included, so the release cannot independently regenerate these counts.
The preparation driver builds manifests, latent caches, transitions, scalers
and validation reports. `--skip-steps` can resume already completed steps.
IQL helpers survive only as compatibility imports; they are not claimed as a
promoted paper-policy training stage.

## Recurrent combat initialization

The reused shell script builds command labels, creates session splits, distills
the feedforward mapping, warms up recurrence and runs invariant checks.
It does not start PPO training.

```bash
PYTHON=python UNITY_BUILD="$UNITY_ENV" \
  bash scripts/train_combat_distillation.sh
```

Provide the encoder, demonstration manifests, telemetry statistics and external
combat build before running. Review `configs/distill_production.yaml` and
`configs/ppo_production.yaml`. The latter intentionally disables PPO for this
initialization workflow. Individual offline stages are also available:

```bash
python -m gambit.rl.online_rl.train_ppo \
  --ppo_config configs/ppo_production.yaml \
  --distill_config configs/distill_production.yaml --stage feedforward_distill_only
python -m gambit.rl.online_rl.train_ppo \
  --ppo_config configs/ppo_production.yaml \
  --distill_config configs/distill_production.yaml --stage recurrent_distill_only
```

## Local45 teacher BC and DAgger

```bash
python -m scripts.collect_combat_teacher \
  --env-path "$UNITY_ENV" --output-root runs/combat_teacher
python -m scripts.build_combat_normalizer --help

python -m scripts.train_combat_teacher_bc \
  --dataset-root "$COMBAT_TEACHER_DATA" \
  --resume-from "$COMBAT_PARENT" --normalizer "$COMBAT_NORMALIZER" \
  --output-dir runs/combat_teacher_bc --device cuda

python -m scripts.collect_combat_dagger --help
python -m scripts.relabel_combat_dagger --help
python -m scripts.train_combat_dagger \
  --original-root "$COMBAT_TEACHER_DATA" --gvg-relabel "$COMBAT_RELABEL_NPZ" \
  --resume-from "$COMBAT_PARENT" --normalizer "$COMBAT_NORMALIZER" \
  --output-dir runs/combat_dagger --device cuda
```

Collection/relabeling require experiment-specific opponent and runtime settings;
use their inherited CLI help. Supply the matching **local45** parent/normalizer
explicitly: the DAgger trainer's historical defaults predate that interface.
NPZ files must follow the original collector schema. The retained
`train_combat_residual` implements the bounded-residual continuation.

## Conservative combat PPO specialists

The wrapper reads learning rates, KL guards, opponent/spawn mixtures and parents
from `scripts/combat_specialist_families.py`. It also requires the historical
frozen opponent/selector manifests. `--resume-from` does not eliminate that
opponent-roster dependency.

```bash
python -m scripts.train_combat_specialist \
  --family-id gvg_generalist_roster_mix \
  --resume-from "$COMBAT_PARENT" --env-path "$UNITY_ENV" \
  --output-dir runs/combat_generalist --device cuda --python python \
  --updates 120 --segment-updates 10 --steps-per-iter 256 --num-areas 2
```

Other family IDs, including anti-strafe and search/obstacle, appear in `--help`
and `FAMILIES`. Nominal shared settings are clip 0.1, gamma 0.99, GAE 0.95,
four length-16 sequences and one optimizer step per rollout. Later repair and
expansion runs used their own overrides. `train_combat_ppo` is the underlying
driver, and `combat_ppo_update` contains the update routine.

## Actor231 navigation BC/DAgger

The original collector has `launch` (full shard schedule) and `run-shard`
(one configured collection job). It requires the certified harness prerequisite,
frozen combat checkpoint, local45 normalizer and procedural layouts. Privileged
teacher labels are supervision; the actor receives the deployable actor231 input.

```bash
python -m scripts.collect_navigation_dagger run-shard --help
python -m scripts.collect_navigation_dagger launch --help
python -m scripts.train_navigation_bc \
  --manifest "$DAGGER_MANIFEST" --branch-id gpu0_base_gru128_seed_a \
  --output-dir runs/navigation_bc --target-updates 1000 \
  --checkpoint "$FROZEN_COMBAT" --normalizer "$COMBAT_NORMALIZER" \
  --actor-normalizer "$ACTOR231_NORMALIZER" --device cuda:0
```

`1000` is an example budget, not the historical reported training duration.
`--target-updates` is an absolute target when resuming. Use `--supplemental-dir`
for the retained supplemental learner-state mix and validation source. Checkpoint
and normalizer identity checks remain active. Existing architectural/augmentation
branches are listed in `scripts/navigation_policy.py:BRANCHES`.

## Navigation PPO: initialize, collect, update

The legacy flag `--goal7-navigator` means the selected BC/DAgger navigator.
It is retained for compatibility with the copied orchestrators.

```bash
python -m scripts.train_navigation_ppo \
  --goal7-navigator "$BC_NAVIGATOR" --branch-id gpu0_base_seed_a \
  --output-dir runs/hunter/init --environment-steps 0 \
  --initialize-only --device cuda:0

python -m scripts.collect_navigation_rollouts --help

python -m scripts.train_navigation_ppo \
  --goal7-navigator "$BC_NAVIGATOR" --branch-id gpu0_base_seed_a \
  --resume runs/hunter/init/hunter_s000000.pt \
  --rollout "$HUNTER_ROLLOUT_NPZ" --stage H0 --environment-steps 2048 \
  --output-dir runs/hunter/update --updates 8 --batch-size 256 --device cuda:0
```

Use real collector output. Required NPZ arrays: `actor_obs`, `critic_obs`,
`policy_action`, `old_mean`, `old_log_std`, `old_mode`, `nav_hidden`, `bc_target`,
`reward`, `visible`, `intervened`. The critic prefix must equal actor231, and
arrays must be finite. `--environment-steps` is the supplied stage-budget label;
its equality to cumulative decisions is not guaranteed across historical runs.

Hunter, reacquisition, peeker and population continuations use this same reused
update driver with branch-specific settings. See `BRANCHES`, `REACQUIRE_VARIANTS`,
`PEEKER_VARIANTS` and the league definitions in `scripts/navigation_ppo.py`.
Its immediate-reward loss is distinct from combat GAE.

## Original staged orchestration

```bash
python -m scripts.train_hunter_curriculum --help
# These three archived orchestrators have no --help parser.
# Run only after reviewing their module constants and providing prerequisites.
python -m scripts.train_reacquisition_curriculum
python -m scripts.train_peeking_curriculum
python -m scripts.train_navigation_population
```

These preserve the original eight-GPU programs and renamed child-script paths.
The last three are fixed experiment drivers, not argument-driven CLIs; even
passing `--help` enters their training workflow after prerequisite checks.
They encode Linux `taskset` affinity and historical artifact/certification
locations. Some later launchers set only `HIP_VISIBLE_DEVICES`, reflecting their
ROCm origin. They are not portable one-GPU launchers. Use the explicit branch
initialization/collection/update workflow on other hardware, or adapt deployment
configuration when integrating Unity. The original hunter curriculum JSON is
included; remaining prerequisite certificates come from prior training stages.

## Frozen calibration and examiner selection

With exact external assets at their documented locations:

```bash
python -m scripts.calibrate_examiner_bank prepare --output runs/calibration
python -m scripts.calibrate_examiner_bank run --output runs/calibration \
  --device cuda:0 --max-parallel 3 --time-scale 20
python -m scripts.select_examiner_bank --output runs/calibration
```

`prepare` checks identities; it is not an asset-free dry run. Rebuilt Unity
binaries/new policies have new hashes and constitute a new experiment. For
included recorded encounters, use the README's `--analysis-only` replay.

## Player posterior

`scripts.assess_player_skill` exposes `initial_posterior`, `stats`, `win_curve`,
`update`, `eig`, and `select_eig`. These mathematical functions require only
NumPy. The CLI runs the original frozen proxy demonstration and requires its
calibration and demonstration prerequisite files. It is not a new interactive UI.
