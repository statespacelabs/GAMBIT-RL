# CUDA Examiner Bank Calibration

Status: **TANDEMFPS_CUDA_EXAMINER_BANK_CALIBRATED**

## Outcome

Frozen bank: 8 primary and 2 reserve bots, selected from 12 candidates after 2,640 terminal matches.
Composition is four parametric and four learned/hybrid primaries, plus one reserve of each class.

## Frozen protocol

- CUDA device cuda:0; no training or normalizer updates.
- 66 unordered pairs; 20 paired seeds; both side assignments.
- Frozen rooms_doorways validation family; no final heldout.
- Unity terminal events are the only rating payoff source.
- Terminal, damage, weapon/reset, schema, and fairness audits passed.

## Selected library

| Bot | Role | Class | mu | sigma | Score | Contact | Timeout | Max side delta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bank_original_combat_expert | primary | learned_or_hybrid | 1137.6 | 102.4 | 0.693 | 0.864 | 0.214 | 0.625 |
| bank_anchor_face_shooter | primary | parametric_anchor | 1058.0 | 102.3 | 0.583 | 0.977 | 0.057 | 0.750 |
| bank_phase5_peeker | primary | learned_or_hybrid | 1022.2 | 102.3 | 0.531 | 0.864 | 0.352 | 0.500 |
| bank_phase4_generalist | primary | learned_or_hybrid | 1014.5 | 102.3 | 0.519 | 0.977 | 0.034 | 0.750 |
| bank_anchor_kiter | primary | parametric_anchor | 1009.9 | 102.3 | 0.512 | 0.977 | 0.116 | 0.500 |
| bank_phase4_anti_strafe | primary | learned_or_hybrid | 989.8 | 102.3 | 0.483 | 0.977 | 0.057 | 0.500 |
| bank_anchor_strafe_shooter | primary | parametric_anchor | 852.6 | 102.5 | 0.292 | 0.977 | 0.084 | 0.275 |
| bank_anchor_cover_sweeper | primary | parametric_anchor | 802.2 | 102.7 | 0.232 | 0.977 | 0.109 | 0.400 |
| bank_phase4_obstacle_search | reserve | learned_or_hybrid | 1100.5 | 102.3 | 0.643 | 0.977 | 0.045 | 0.750 |
| bank_anchor_rusher | reserve | parametric_anchor | 964.2 | 102.3 | 0.445 | 0.977 | 0.045 | 0.400 |

## Screened but not selected

| Bot | Reason | Nearest profile | RMSE | Correlation |
|---|---|---|---:|---:|
| bank_anchor_search_pursuer | capacity-pruned for lower incremental profile utility | bank_anchor_rusher | 0.086 | 0.782 |
| bank_phase5_hunter | capacity-pruned for lower incremental profile utility | bank_phase5_peeker | 0.148 | 0.679 |

## Bradley-Terry sensitivity

| Fit | Pearson | Spearman | Max mu delta |
|---|---:|---:|---:|
| classic_bt_weak_prior | 1.0000 | 1.0000 | 0.6 |
| bayes_prior_sigma_200 | 1.0000 | 1.0000 | 1.3 |
| bayes_prior_sigma_700 | 1.0000 | 1.0000 | 0.5 |
| schedule_base_only | 0.8967 | 0.8182 | 89.8 |
| schedule_swapped_only | 0.8954 | 0.8671 | 91.6 |
| paired_seeds_00_09 | 0.9938 | 0.9930 | 22.4 |
| paired_seeds_10_19 | 0.9939 | 0.9790 | 22.5 |

## Integrity

- Technical audit: PASS (2640/2640 matches).
- Missing terminal outcomes and incomplete damage rows: 0.
- Hidden-state leaks, privileged actor inputs, human input, and final-heldout use: 0.
- Weapon ownership/fire and side-configuration failures: 0.

Ratings measure diagnostic difficulty on this frozen CUDA protocol, not universal skill. Selection uses payoff-profile separation and reliability rather than win rate alone.

TANDEMFPS_CUDA_EXAMINER_BANK_CALIBRATED
