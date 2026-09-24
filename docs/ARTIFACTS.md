# Artifacts and external integration

The release stands alone for recorded calibration, posterior calculations and
included CPU tests. **Unity is a separate package**, to be integrated later.
No Unity C# source, scene, executable or build tooling is included here.

## Included inputs and evidence

- `reference/calibration_inputs.tar.gz`: 2,640 original match records, 2,640
  run summaries, an execution manifest and four contract JSON files.
- `reference/calibration_archive_members.json`: original path, archive-member
  path and SHA-256 for each of these 5,285 files.
- `reference/examiner_bank/`: frozen ratings, selection, payoffs, schedule,
  protocol and audit outputs. Historical paths/status identifiers are preserved.
- `experiments/phase5_map_general_headless_hunter/02_layouts/`: original
  procedural split/layout JSON definitions consumed by the Python collectors.
- Adjacent telemetry schema/normalizer JSON: actor231/critic428 interface data.
- `reference/`: original observation/training audits. Links inside verbatim
  historical audits refer to the original checkout, rather than this package.

`port_manifest.json` records source and destination hashes for every copied file.
Archive hashes separately cover the compressed encounter records. These describe
actual local working-tree sources, including existing untracked files, rather
than claiming a Git commit contains all recovered experimental artifacts.

## External training inputs

| Input | Used by | Where to configure it |
| --- | --- | --- |
| Human sessions and split manifest | Representation/data preparation | `--rendered-dir`, `--joint-manifest` |
| Encoder and telemetry statistics | Latent cache and historical initialization | `--checkpoint`, `--telemetry-stats`, YAML |
| Transition/window manifests and scalers | Human BC and command distillation | `configs/bc_train*.yaml`, `configs/distill_production.yaml` |
| Teacher shards and relabeled learner states | Combat BC/DAgger | `--dataset-root`, `--original-root`, `--gvg-relabel` |
| Combat parent, normalizer and frozen opponent roster | Combat PPO | `--resume-from`, `PRODUCTION_SELECTOR_MANIFEST` and family paths |
| DAgger manifest and NPZ shards | Navigator BC | `--manifest`, optional `--supplemental-dir` |
| BC navigator and rollout NPZ | Navigator PPO | `--goal7-navigator`, `--resume`, repeatable `--rollout` |
| Frozen combat expert and local45 normalizer | Navigation and learned evaluation | Existing CLI flags and SHA guards |
| Unity executable with all build data | Online collection/evaluation | `--env-path`; frozen calibration constants |

Some inherited drivers/metadata retain historical relative artifact paths.
Run from the checkout, use explicit CLI paths where available, and restore the
expected artifact tree for orchestration without path overrides. `RL_GRADER_ROOT`
can set the combat-family artifact root; it is a compatibility environment name,
not a dependency on the parent repository's Python modules.

## Frozen identities and replay

`scripts/calibrate_examiner_bank.py` retains exact checkpoint/build/normalizer
and layout hashes in `EXPECTED` and `LEARNED`. The corresponding contract and
candidate library are in `reference/examiner_bank/00_contract/`. Policy weights
and executables are not copied into this code release.

`--analysis-only` stops before external-file certification and reports a distinct
recorded-replay status. It runs the original record audits, rating fit,
sensitivity analysis and bank selection. The normal path retains all hash checks.
Do not label a new game build or newly trained policy as the frozen experiment.

## Boundary for later Unity integration

The Python side uses `mlagents_envs`. The separate game package must provide
actor231, local45 and (for privileged training) critic428 observations; four
continuous and four binary action branches; a 0.02-second control step; compatible
terminal/reset/side/layout telemetry; and the existing prerequisite manifests.
Schema IDs and checkpoint/candidate identities are deliberately unchanged.

`requirements-unity-historical.txt` is copied from an existing laptop release.
It is a historical interface dependency lock, not a complete CUDA training lock.
Use a separate Python 3.10 environment for that old stack: ML-Agents 1.1.0 pins
NumPy 1.23.5 and the historical gRPC upper bound has limited wheels on newer
Python. Install a compatible PyTorch/torchvision pair for your hardware separately,
then install this source checkout editable. The copied lock also contains ONNX
Runtime for the older demo; these training implementations use PyTorch.
The `unity` optional dependency extra installs the interface, not the game.

## Distribution metadata

No project license was available to assign to this port. No license, dataset
release permission, author list or download URL has been invented. The authors
retain control of the final submission metadata and external assets.
