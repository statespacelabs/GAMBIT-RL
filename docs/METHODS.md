# Manuscript methods and implementation

Scope: `paper/sections/appendix_new.tex`, including the agent interface and
training-lineage supplement. Module names below are relative to this release.
The original-to-ported mapping, including hashes, is in [ported.md](../ported.md).

| Manuscript component | Reused implementation | Interpretation |
| --- | --- | --- |
| Human demonstration preparation | `gambit/dataset_preparation/`, `scripts/prepare_demonstration_dataset.py` | Session/window manifests, latent caches, transition/action/reward extraction |
| Historical representation pretraining | `gambit/models/`, `gambit/training/` | Ancestral multimodal encoder; not the final deployed actor |
| Human BC and recurrent combat initialization | `gambit/rl/training/train_bc.py`, `gambit/rl/online_rl/core_action_projector.py`, `scripts/train_combat_distillation.sh` | Historical 14-component window targets are projected to the eight deployed controls |
| Local45 combat architecture | `gambit/rl/online_rl/telemetry_encoder.py`, `actor_critic.py` | Where/view/rhythm branches, fused representation, GRU, continuous/binary/value heads |
| Combat teacher BC and DAgger | `scripts/collect_combat_teacher.py`, `collect_combat_dagger.py`, `relabel_combat_dagger.py`, `train_combat_teacher_bc.py`, `train_combat_dagger.py` | Existing teacher labels, learner-state collection, retention-weighted supervised updates |
| Combat specialist PPO | `scripts/train_combat_specialist.py`, `combat_specialist_families.py`, `train_combat_ppo.py`, `combat_ppo_update.py` | The family launcher calls the inherited one-sided recurrent PPO driver |
| Actor231 navigation architecture and imitation | `scripts/navigation_policy.py`, `train_navigation_bc.py`, `collect_navigation_dagger.py` | Separate local-geometry/context branches, GRU, navigation/mode/blend/auxiliary heads |
| Privileged critic and navigation PPO | `scripts/navigation_ppo.py`, `train_navigation_ppo.py`, `collect_navigation_rollouts.py` | Actor231/critic428 separation, frozen combat expert, conservative continuation |
| Hunter, reacquisition, peeking, tactical continuations | `scripts/train_hunter_curriculum.py`, `train_reacquisition_curriculum.py`, `train_peeking_curriculum.py`, `train_navigation_population.py` | Original staged orchestration and branch definitions |
| Runtime handoff and local safety | `scripts/navigation_policy.py`, `tactical_handoff.py`, `local_geometry_peeker.py`, `screen_learned_controllers.py` | Search, recovery, LOS-gated combat and persistent within-encounter recurrent state |
| Six parametric anchors | `scripts/calibrate_examiner_bank.py:ANCHORS`, `scripts/examiner_match.py:ExaminerAnchorRuntime` | Interpretable movement, shooting-angle and alternation parameters |
| Symmetric calibration | `scripts/calibrate_examiner_bank.py`, `examiner_match.py`, `symmetric_tournament.py` | Twelve candidates, 66 unordered pairs, 20 seeds and two role assignments per pair |
| Rating and bank selection | `scripts/select_examiner_bank.py` | Gaussian-regularized fractional Bradley–Terry fit, curvature uncertainty, reliability and payoff diversity |
| Online posterior and next examiner | `scripts/assess_player_skill.py` | Grid posterior, Gauss–Hermite marginalization, expected information gain and style/repeat constraints |
| Unity observations and action transport | Separate Unity package; Python interfaces in the collectors/runtimes | C# actor231, local45, privileged critic, procedural environment and action mapping |

## Distinct optimization procedures

The Gen 1 combat continuation uses a frozen local45 normalizer, stochastic
training actions, Adam, clip 0.1, gamma 0.99 and GAE lambda 0.95. The nominal
rollout is 256 steps in two arenas. The inspected specialist driver performs one
optimizer update on the first minibatch of four length-16 recurrent sequences.
Its applied entropy coefficient is 0.003 even when a family definition contains
a different value. Family learning rates, KL limits, parents and opponent mixes
remain in `combat_specialist_families.py`.

The navigator continuation must not be described using those combat settings.
Its original `conservative_ppo_loss` uses the stored recurrent state and a
one-step observation. Advantage is immediate reward minus detached critic value;
the critic target is immediate reward. There is no multi-step GAE or entropy
term in that loss. The actor policy term is masked to hidden-enemy,
non-intervened samples. BC, action anchoring, mode supervision and smoothness
terms remain unchanged. The optimizer uses AdamW, clip 0.1, two epochs per
sampled batch, KL guard 0.01 and gradient cap 0.5; branch/stage metadata supplies
the learning-rate and auxiliary-loss schedules.

Some compatibility imports still expose historical IQL model/training helpers.
They are transitive dependencies of existing package initializers and the
historical evaluation helpers, not evidence that IQL produced the reported examiner bank.
The teacher/residual/temporal-selector ancestry similarly does not mean the
final examiner runtime actively deploys a temporal ensemble.

## Interface and fairness

The control interval is 0.02 seconds (50 Hz). Actions are four continuous
movement/look channels and binary shoot, reload, jump, crouch channels in that
order. The final evaluator uses deterministic actions. The actor has 231 fair,
map-independent features, while the training critic has 428 features. The
local45 combat transport is not intrinsically safe under occlusion: the runtime
must withhold combat inference and freeze its recurrent state when LOS is absent.
The copied Python wrappers implement that condition. Hidden search can use stale
last-seen state and the explicitly permitted noisy, quantized hunt cue.

The 152 geometry features and 79 contextual features are encoded separately.
The actor231 normalizer is analytic/identity; local45 uses its frozen fitted
normalizer. Checkpoint and normalizer SHA checks are retained. Archived JSON
occasionally contains stale action labels; the applied C# ordering above is the
contract. The detailed original observation audit is in `reference/`.

## Calibration, selection and assessment

Calibration timeouts count as draws. Outcomes map to scores 0, 0.5, 1.
`fit_bt` uses rating prior N(1000, 350²), Elo scale 400, Newton updates and
inverse-information diagonal uncertainty. Ratings are centered over all twelve
candidates, then the selected ten retain that scale.

Eligibility excludes sigma > 120, timeout fraction > 0.90, or uniformly extreme
payoff profiles. Within each class, selection maximizes mean payoff-profile
diversity, rating span, and reliability utility with coefficients 8, 1/400,
0.25 for timeouts, and 1/300 for sigma, as specified in the appendix. There are
four primaries and one reserve per class, and a navigation-enabled learned
primary is required.

The player model retains a grid distribution on 0..2000 in unit increments and
integrates bot uncertainty using 11 Gauss–Hermite nodes. The nominal N(1000,350²)
prior has discrete/truncated SD about 343.195. EIG averages the two win/loss
posterior entropies; an observed draw uses sqrt(p*(1-p)). The frozen driver runs
six diagnostic and two validation encounters. Its timeout score uses damage
differential, unlike calibration's draw-on-timeout rule. This protocol distinction
is inherited and has not been silently changed during packaging.

## Reused source rather than reconstructed experiments

The selected learned policies share ancestry. Retraining a family does not by
itself regenerate its exact historical final checkpoint. The original repository
does not freeze every intermediate historical launch override. The release
preserves the available implementations, configurations and checkpoint guards;
the frozen tournament replay provides a directly reproducible numerical result.

