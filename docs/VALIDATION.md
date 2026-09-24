# Validation of the source port

Validated on 2026-09-23. Numerical results and dependency versions are recorded in
[validation_results.json](validation_results.json).

## Completed checks

- **29 existing repository tests passed** on CPU after renaming the package to
  `gambit`. Tests cover recurrent/vectorized PPO, command distillation and session
  splits, peeking, reacquisition and tactical handoff. No new algorithm tests were
  written for the port.
- Editable installation succeeded under the distribution name `gambit-paper`.
- Source distribution and wheel builds succeeded. Every port-manifest file is
  present in the source archive with its expected hash; Unity and run outputs
  are excluded.
- **24 argument-driven entry points** returned help successfully when invoked
  from `/tmp`, outside both source roots. The three original fixed multi-GPU
  orchestrators have no help parser and are documented accordingly.
- All local Python imports/direct script paths resolve within the release.
- The original **2,640 calibration records** passed the original terminal,
  damage, side, fairness and runtime-summary audits.
- Recomputed ratings and selection agree with the frozen reference: maximum
  mean-rating difference **1.14e-13**, maximum uncertainty difference **4.84e-13**;
  all primary/reserve/excluded assignments match. The payoff CSV is byte-identical.
- Bayesian checks confirm normalized posteriors, the expected direction of
  win/loss updates, positive information gain and prevention of immediate repeats.
  The truncated-grid prior has mean 1000 and SD 343.19516830387903.

The CPU tests used Python 3.11.14, PyTorch 2.10.0, torchvision 0.25.0, NumPy 2.4.1,
pandas 3.0.0, PyYAML 6.0.3 and pytest 9.1.1. Recorded tournament replay used the
system Python/NumPy versions listed separately in the JSON report. These are
port-validation environments, not claims about the historical GPU training lock.

## Reproduce locally

```bash
python -m pip install -e '.[training,test]'
python -m pytest -q
mkdir -p runs/calibration_replay
tar -xzf reference/calibration_inputs.tar.gz -C runs/calibration_replay
python -m scripts.select_examiner_bank \
  --output runs/calibration_replay --analysis-only
```

Use a fresh replay output directory. The analysis-only option is a small CLI
adaptation: it changes the output status to `GAMBIT_RECORDED_CALIBRATION_REPLAY`
and stops before certifying external checkpoint/build hashes. All mathematics,
record audits and selection rules remain the original implementation. The
ordinary mode retains the original full artifact checks.

## Not run

No policy training, Unity process, GPU tournament, human session or original
fixed eight-GPU orchestration was launched. Unity is a separate package.
Online reproduction still requires that game package, the matching input data,
checkpoint/normalizer artifacts and prerequisite certificates. The included
recordings reproduce the analysis of existing experiments, not a new experiment.
