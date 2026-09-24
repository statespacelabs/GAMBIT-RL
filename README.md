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

