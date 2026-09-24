# GAMBIT: paper code

This is a source release assembled from the existing RL-GRADER repository for
the methods in `paper/sections/appendix_new.tex`. Training algorithms, model
architectures, controller behavior, losses, and rating rules are reused. New
material is limited to packaging, documentation, provenance metadata,
mechanical name/path changes, and an analysis-only CLI option for recorded
calibration replay. The original repository files are untouched.

Start with [the training guide](docs/TRAINING.md), [the manuscript-to-code
map](docs/METHODS.md), and [the complete port inventory](ported.md).

## Install

Use a source checkout and run commands from this directory. An editable install
keeps the scripts' existing relative configuration and artifact paths valid.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

That is enough for the recorded tournament replay and the Bayesian mathematical
functions. For neural models, offline training, and the existing tests:

```bash
python -m pip install -e '.[training,test]'
python -m pytest -q
```

Choose a PyTorch/torchvision pair appropriate to your CPU or GPU environment.
These dependency ranges describe the source APIs; they are not a claim that all
versions reproduce the original CUDA training numerically. The actual validation
environment and results are in [VALIDATION.md](docs/VALIDATION.md).

Unity interaction additionally needs ML-Agents and a compatible compiled game.
Use a separate Python 3.10 environment for the historical Unity dependency lock
described in [ARTIFACTS.md](docs/ARTIFACTS.md). CPU numerical replay does not need
Unity, model checkpoints, demonstration data, or a GPU.

## Reproduce the recorded tournament analysis

The included archive contains the original match records and summaries for all
2,640 calibration encounters, plus their schedule and candidate contracts. It
contains no synthetic replacements. Extract it into a fresh output directory
**inside this checkout**; the inherited hash-ledger writer requires that location.

```bash
mkdir -p runs/calibration_replay
tar -xzf reference/calibration_inputs.tar.gz -C runs/calibration_replay
python -m scripts.select_examiner_bank --output runs/calibration_replay --analysis-only
```

The command audits every encounter, fits ratings, evaluates sensitivity, and
selects eight primary examiners plus two reserves. Outputs include
`BOT_RATINGS_CUDA.json`, `BOT_PAYOFF_MATRIX.csv`, `FINAL_CUDA_BOT_BANK.json`,
and `BOT_CALIBRATION_REPORT.md`. Frozen reference outputs are in
[`reference/examiner_bank/`](reference/examiner_bank/).

The replay reports `GAMBIT_RECORDED_CALIBRATION_REPLAY`. It audits the recorded
observations and repeats the numerical analysis, without certifying external
checkpoint/build hashes or rerunning Unity matches. Omitting `--analysis-only`
retains the original artifact verification requirements.

## Method names

| Historical development label | Name used in this release | Main entry point |
| --- | --- | --- |
| Phase 1 | Demonstration representation pretraining | `gambit.training.train_ddp` |
| Phase 2 | Demonstration preparation and human behavioral cloning | `scripts.prepare_demonstration_dataset`, `scripts.train_human_bc` |
| Phase 3 / 3v2 | Recurrent combat initialization and teacher/DAgger refinement | `scripts.train_combat_distillation.sh`, `scripts.train_combat_dagger` |
| Phase 4.4 | Conservative combat PPO and specialist continuations | `scripts.train_combat_specialist` |
| Phase 5 | Fair navigation BC/DAgger, hunter, reacquisition, and peeking | `scripts.train_navigation_bc`, `scripts.train_navigation_ppo` |
| Phase 6 paper evaluation | Examiner calibration, diversity selection, and skill assessment | `scripts.calibrate_examiner_bank`, `scripts.select_examiner_bank`, `scripts.assess_player_skill` |

`gambit/` contains the reused representation, dataset, recurrent policy, and
optimization package. `scripts/` contains the method-named entry points and their
shared controller modules. See its [entry-point index](scripts/README.md).

Historical schema IDs, checkpoint keys, candidate IDs, Unity class names, and
some artifact directory names remain intact because frozen artifacts refer to
them. For example, `phase5_actor_obs_v001` still means the 231-dimensional actor
schema. These are compatibility identifiers, not new method stages.

## Included and external material

Included: Python sources, existing CPU tests, training configurations,
procedural layout/schema JSON, calibration inputs,
reference outputs, and source/ported SHA-256 inventories.

External training inputs: human demonstrations, encoder and policy checkpoints,
BC/DAgger shards, and the separately maintained Unity package. Their locations
and contracts are documented in [ARTIFACTS.md](docs/ARTIFACTS.md). Unity source,
scenes and builds are outside this release and will be integrated separately.
Full historical checkpoint
lineages require those external artifacts; no training was started during this port.

The archived player-assessment driver is the paper's frozen neural-proxy systems
demonstration. Its numerical posterior/selection functions can be imported
independently. It is not a human participant study or a generic interactive client.
The older Davidson rating implementation and experimental Gen 3 workflows are
not substituted for the paper's reported methods.

No license has been invented for the original code or third-party data/assets.
The authors should attach the intended distribution license to the final submission.

