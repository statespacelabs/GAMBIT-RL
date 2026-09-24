#!/usr/bin/env python3
"""Execute Work Order 7 as an honest Bayesian backend demonstration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments/paper_work_orders_v001/human_demo"
GOAL3 = ROOT / "experiments/paper_goal3_cuda_examiner_bank_v001"
RATINGS_PATH = GOAL3 / "BOT_RATINGS_CUDA.json"
CANDIDATES_PATH = GOAL3 / "00_contract/CANDIDATE_LIBRARY.json"
SCHEDULE_PATH = GOAL3 / "00_contract/ROUND_ROBIN_SCHEDULE.json"
DEMO_REPORT = ROOT / "experiments/paper_work_orders_v001/demo/TWO_MAP_DEMO_REPORT.md"
RUNNER = ROOT / "scripts/audit_examiner_actions.py"
sys.path.insert(0, str(ROOT / "scripts"))
import calibrate_examiner_bank as calibrate  # noqa: E402


PREREQ = "TANDEMFPS_CUDA_EXAMINER_BANK_CALIBRATED"
WORK6_STATUS = "TANDEMFPS_TWO_MAP_AUTONOMOUS_DEMO_READY"
STATUS = "TANDEMFPS_TWO_MAP_AND_BAYESIAN_SCORING_DEMO_READY"
PROXY = "bank_phase4_generalist"
PARTICIPANT = "systems_proxy_phase4_generalist_v001"
SESSION_ID = "work_order7_backend_demo_v001"
SESSION_SEED = 760307
PRIOR_MU = 1000.0
PRIOR_SIGMA = 350.0
ELO_SCALE = 400.0
DIAGNOSTIC_COUNT = 6
VALIDATION_IDS = ("bank_phase5_hunter", "bank_anchor_search_pursuer")
SEED_INDEXES = (2, 5, 8, 11, 14, 17, 18, 19)
SIDES = ("base", "swapped", "base", "swapped", "base", "swapped", "base", "swapped")
REPLAY_SEEDS = (0, 3, 6, 9, 12, 15)
GRID = np.linspace(0.0, 2000.0, 2001, dtype=np.float64)
GH_X, GH_W = np.polynomial.hermite.hermgauss(11)
HIT_PATTERN = re.compile(r"\[MatchManager\] HIT: Area\d+_Player_([AB])\s")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_text(path, "".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def normalized(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("invalid posterior mass")
    return values / total


def initial_posterior() -> np.ndarray:
    return normalized(np.exp(-0.5 * ((GRID - PRIOR_MU) / PRIOR_SIGMA) ** 2))


def stats(posterior: np.ndarray) -> tuple[float, float]:
    mu = float(np.sum(posterior * GRID))
    variance = float(np.sum(posterior * (GRID - mu) ** 2))
    return mu, math.sqrt(max(0.0, variance))


def entropy(posterior: np.ndarray) -> float:
    values = posterior[posterior > 0.0]
    return float(-np.sum(values * np.log(values)))


def win_curve(mu: float, sigma: float) -> np.ndarray:
    samples = mu + math.sqrt(2.0) * sigma * GH_X
    exponent = np.clip((samples[:, None] - GRID[None, :]) / ELO_SCALE, -12, 12)
    p = 1.0 / (1.0 + np.power(10.0, exponent))
    return np.sum(GH_W[:, None] * p, axis=0) / math.sqrt(math.pi)


def eig(posterior: np.ndarray, mu: float, sigma: float) -> float:
    p = win_curve(mu, sigma)
    q = float(np.sum(posterior * p))
    return entropy(posterior) - (
        q * entropy(normalized(posterior * p))
        + (1.0 - q) * entropy(normalized(posterior * (1.0 - p)))
    )


def update(posterior: np.ndarray, mu: float, sigma: float, outcome: str) -> np.ndarray:
    p = win_curve(mu, sigma)
    if outcome == "win":
        likelihood = p
    elif outcome == "loss":
        likelihood = 1.0 - p
    elif outcome == "draw":
        likelihood = np.sqrt(p * (1.0 - p))
    else:
        raise RuntimeError(f"bad outcome: {outcome}")
    return normalized(posterior * likelihood)


def style_group(bot_id: str) -> str:
    if any(
        x in bot_id
        for x in ("cover_sweeper", "search_pursuer", "obstacle_search", "peeker")
    ):
        return "search_cover"
    if any(x in bot_id for x in ("face_shooter", "combat_expert")):
        return "precision_duel"
    if any(x in bot_id for x in ("strafe_shooter", "rusher", "kiter", "anti_strafe")):
        return "movement_pressure"
    return "general_hunt"


def diagnostic_pool(ratings: dict[str, Any]) -> list[str]:
    return sorted(
        bot
        for bot, row in ratings.items()
        if row["role"] in {"primary", "reserve"}
        and bot != PROXY
        and bot not in VALIDATION_IDS
    )


def select_eig(
    posterior: np.ndarray,
    pool: list[str],
    ratings: dict[str, Any],
    previous: str | None,
) -> tuple[str, float]:
    choices = [bot for bot in pool if bot != previous]
    if previous is not None:
        diverse = [bot for bot in choices if style_group(bot) != style_group(previous)]
        if diverse:
            choices = diverse
    scored = [
        (bot, eig(posterior, float(ratings[bot]["mu"]), float(ratings[bot]["sigma"])))
        for bot in choices
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    if not scored:
        raise RuntimeError("no eligible EIG opponent")
    return scored[0]


def schedule_row(
    schedule: list[dict[str, Any]],
    opponent: str,
    seed_index: int,
    side: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in schedule
        if {row["policy_a_id"], row["policy_b_id"]} == {PROXY, opponent}
        and int(row["paired_seed_index"]) == seed_index
        and row["side_assignment"] == side
    ]
    if len(rows) != 1:
        raise RuntimeError(f"schedule resolution failed for {opponent}: {len(rows)}")
    return dict(rows[0])


def prefix_for(match: dict[str, Any], policy_id: str) -> str:
    for prefix in ("policy_a", "policy_b"):
        if match[f"{prefix}_id"] == policy_id:
            return prefix
    raise RuntimeError(f"policy absent: {policy_id}")


def classify(match: dict[str, Any]) -> tuple[str, str, float, float]:
    prefix = prefix_for(match, PROXY)
    other = "policy_b" if prefix == "policy_a" else "policy_a"
    dealt = float(match[f"{prefix}_damage"])
    taken = float(match[f"{other}_damage"])
    if bool(match["timeout"]):
        if dealt > taken:
            return "win", "timeout_damage_win", dealt, taken
        if dealt < taken:
            return "loss", "timeout_damage_loss", dealt, taken
        return "draw", "timeout_equal_damage_draw", dealt, taken
    payoff = float(match[f"terminal_payoff_{prefix}"])
    outcome = "win" if payoff > 0 else "loss" if payoff < 0 else "draw"
    return outcome, f"terminal_{match['terminal_type']}", dealt, taken


def parse_run(run_dir: Path, opponent: str) -> dict[str, Any]:
    summary = read_json(run_dir / "run_summary.json")
    match = read_jsonl(run_dir / "matches.jsonl")[0]
    audit = read_json(run_dir / "action_audit.json")
    weapon = read_jsonl(run_dir / "weapon_state.jsonl")
    unity = (run_dir / "unity.log").read_text(errors="replace")
    prefix = prefix_for(match, PROXY)
    slot = str(match[f"{prefix}_physical_slot"])
    requested = int(audit["policies"][PROXY]["requested_fire_steps"])
    blocked = sum(
        row.get("event") == "blocked_fire"
        and str(row.get("agent_id", "")).endswith(slot)
        for row in weapon
    )
    confirmed = requested - blocked
    hits = HIT_PATTERN.findall(unity).count(slot)
    if confirmed < 0:
        raise RuntimeError("negative confirmed shots")
    outcome, timeout_class, dealt, taken = classify(match)
    zero_keys = (
        "hidden_state_leakage_events",
        "side_config_mismatches",
        "weapon_ownership_mismatch_count",
        "weapon_fire_mismatch_count",
        "observation_nonfinite_values",
        "action_nonfinite_values",
        "missing_terminal_outcomes",
        "human_controller_count",
        "manual_input_events",
    )
    failures = [key for key in zero_keys if int(summary.get(key, 0)) != 0]
    if summary.get("status") != "PASS":
        failures.append("runner_status")
    for policy_id in (PROXY, opponent):
        if int(audit["policies"][policy_id].get("reset_residual_count", 0)):
            failures.append(f"reset_residual:{policy_id}")
    if failures:
        raise RuntimeError(f"encounter integrity failures: {failures}")
    return {
        "run_status": "PASS",
        "duration_seconds": float(match["duration_seconds"]),
        "terminal_type": match["terminal_type"],
        "outcome": outcome,
        "timeout": bool(match["timeout"]),
        "timeout_classification": timeout_class,
        "damage_for": dealt,
        "damage_against": taken,
        "damage_differential": dealt - taken,
        "contact_time_seconds": (
            None
            if match.get("contact_time_seconds") is None
            else float(match["contact_time_seconds"])
        ),
        "requested_fire_steps": requested,
        "blocked_fire_steps": blocked,
        "confirmed_shots": confirmed,
        "hits": hits,
        "accuracy": float(hits / confirmed) if confirmed else 0.0,
        "proxy_action_sha256": audit["policies"][PROXY]["applied_action_sha256"],
        "opponent_action_sha256": audit["policies"][opponent]["applied_action_sha256"],
        "matches_sha256": sha256(run_dir / "matches.jsonl"),
        "integrity_failures": [],
    }


def run_encounter(
    number: int,
    opponent: str,
    schedule: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = schedule_row(
        schedule, opponent, SEED_INDEXES[number - 1], SIDES[number - 1]
    )
    item = {
        **frozen,
        "execution_index": number - 1,
        "run_id": f"encounter_{number:02d}_{PROXY}_vs_{opponent}",
    }
    command, run_dir, stdout_path, stderr_path = calibrate.command_for(
        item, candidates, OUT, 29400, 20.0, "cuda:0"
    )
    command[1] = str(RUNNER)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HIP_VISIBLE_DEVICES": "",
            "MVH_RENDERED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PHASE5_NAVMESH_UPPER_BOUND": "0",
            "PHASE5_ENABLE_NAVMESH_ORACLE": "0",
            "PHASE6_HUMAN_DEMO_ACTION_AUDIT_PATH": str(run_dir / "action_audit.json"),
            "RESEARCH_HUD_STATE_PATH": str(run_dir / "inert_hud.json"),
        }
    )
    done = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(done.stdout, encoding="utf-8")
    stderr_path.write_text(done.stderr, encoding="utf-8")
    if done.returncode:
        raise RuntimeError(f"encounter {number} failed: {done.stderr[-800:]}")
    return parse_run(run_dir, opponent), {
        "run_id": item["run_id"],
        "run_dir": str(run_dir.relative_to(ROOT)),
        "layout_family": frozen["layout_family"],
        "layout_seed": int(frozen["layout_seed"]),
        "paired_seed_index": int(frozen["paired_seed_index"]),
        "runtime_seed": int(frozen["runtime_seed"]),
        "side_assignment": frozen["side_assignment"],
    }


def replay_outcome(
    opponent: str,
    seed_index: int,
    side: str,
    schedule: list[dict[str, Any]],
) -> str:
    row = schedule_row(schedule, opponent, seed_index, side)
    match_path = GOAL3 / "01_round_robin/raw" / row["run_id"] / "matches.jsonl"
    return classify(read_jsonl(match_path)[0])[0]


def strategy_choice(
    name: str,
    step: int,
    posterior: np.ndarray,
    pool: list[str],
    ratings: dict[str, Any],
    previous: str | None,
    rng: random.Random,
) -> str:
    choices = [bot for bot in pool if bot != previous]
    if name == "active_eig":
        return select_eig(posterior, pool, ratings, previous)[0]
    if name == "random":
        return rng.choice(choices)
    if name == "fixed_easy_to_hard":
        return sorted(pool, key=lambda bot: (ratings[bot]["mu"], bot))[step]
    if name == "nearest_rating":
        mu, _ = stats(posterior)
        return min(choices, key=lambda bot: (abs(float(ratings[bot]["mu"]) - mu), bot))
    raise RuntimeError(name)


def compare_backends(
    ratings: dict[str, Any],
    schedule: list[dict[str, Any]],
    pool: list[str],
) -> list[dict[str, Any]]:
    rows = []
    names = ("active_eig", "random", "fixed_easy_to_hard", "nearest_rating")
    for offset, name in enumerate(names):
        posterior = initial_posterior()
        rng = random.Random(SESSION_SEED + offset)
        previous = None
        selected = []
        total_eig = 0.0
        for step in range(DIAGNOSTIC_COUNT):
            bot = strategy_choice(name, step, posterior, pool, ratings, previous, rng)
            information = eig(
                posterior, float(ratings[bot]["mu"]), float(ratings[bot]["sigma"])
            )
            outcome = replay_outcome(bot, REPLAY_SEEDS[step], SIDES[step], schedule)
            posterior = update(
                posterior,
                float(ratings[bot]["mu"]),
                float(ratings[bot]["sigma"]),
                outcome,
            )
            selected.append(bot)
            total_eig += information
            previous = bot
        mu, sigma = stats(posterior)
        brier = []
        for index, bot in enumerate(VALIDATION_IDS):
            probability = float(
                np.sum(
                    posterior
                    * win_curve(float(ratings[bot]["mu"]), float(ratings[bot]["sigma"]))
                )
            )
            outcome = replay_outcome(bot, 16 + index, SIDES[6 + index], schedule)
            score = {"win": 1.0, "draw": 0.5, "loss": 0.0}[outcome]
            brier.append((probability - score) ** 2)
        rows.append(
            {
                "strategy": name,
                "evidence_mode": "frozen_round_robin_replay",
                "encounters": DIAGNOSTIC_COUNT,
                "selected_bots": "|".join(selected),
                "unique_bots": len(set(selected)),
                "unique_style_categories": len({style_group(bot) for bot in selected}),
                "final_posterior_mu": mu,
                "final_posterior_sigma": sigma,
                "sigma_contraction": PRIOR_SIGMA - sigma,
                "absolute_error_vs_proxy_frozen_mu": abs(
                    mu - float(ratings[PROXY]["mu"])
                ),
                "cumulative_expected_information_gain_nats": total_eig,
                "validation_brier": float(np.mean(brier)),
                "session_seed": SESSION_SEED,
            }
        )
    return rows


def protocol(pool: list[str], candidates: dict[str, dict[str, Any]]) -> str:
    entries = "\n".join(f"- {bot}: {candidates[bot]['style']}" for bot in pool)
    return f"""# Human Bayesian-scoring demo protocol

This freezes the Work Order 7 systems demonstration. The participant is {PARTICIPANT}, a frozen neural policy proxy. It is not a human participant, not a human pilot, and not evidence about human performance. The only restored real operator record was a one-encounter development smoke explicitly invalid for behavioral scoring; it is not reused or relabeled.

A future human pilot must use this frozen configuration with consent, anonymized IDs, and applicable approval.

## Sequence

- Six adaptive diagnostic encounters with a 30-second maximum each.
- Prior: Rating 1000 \u00b1 350.
- Maximum expected-information-gain selection with bot rating uncertainty.
- No immediate bot repeat; change public style group when possible.
- Kill uses terminal winner. Timeout uses damage differential; equal damage is a draw.
- Update after diagnostics, then run two withheld validations without updating.

## Public HUD

Show rating and uncertainty, opponent style, encounter number, posterior after the outcome, and confidence improvement. Internal Bayesian variable names remain research-only.

## Diagnostic pool

{entries}

## Withheld validation

- {VALIDATION_IDS[0]}
- {VALIDATION_IDS[1]}

The backend comparison covers active EIG, seeded random, easy-to-hard ladder, and nearest-rating selection using frozen round-robin outcomes.
"""


def report(
    trace: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    ratings: dict[str, Any],
) -> str:
    diagnostics = [row for row in trace if row["phase"] == "diagnostic"]
    final = trajectory[DIAGNOSTIC_COUNT]
    sequence = "\n".join(
        f"| {r['encounter_number']} | {r['selected_bot_id']} | {r['selected_style_category']} | "
        f"{r['outcome']} | {r['posterior_mu']:.1f} | {r['posterior_sigma']:.1f} | "
        f"{r['expected_information_gain_nats']:.5f} | {r['contact_time_seconds']} | {r['accuracy']:.3f} |"
        for r in diagnostics
    )
    valid = "\n".join(
        f"| {r['encounter_number']} | {r['selected_bot_id']} | {r['outcome']} | "
        f"{r['predicted_win_probability']:.3f} | {r['brier_score']:.4f} |"
        for r in validation
    )
    comp = "\n".join(
        f"| {r['strategy']} | {r['final_posterior_mu']:.1f} | "
        f"{r['final_posterior_sigma']:.1f} | "
        f"{r['absolute_error_vs_proxy_frozen_mu']:.1f} | {r['validation_brier']:.4f} |"
        for r in comparison
    )
    return f"""# Human Bayesian-scoring systems demonstration

Combined status: **{STATUS}**

This is an end-to-end backend demonstration with a frozen neural participant proxy and real Unity encounter outcomes. It is not a human study and does not relabel the restored one-encounter operator development smoke.

Six adaptive encounters used frozen CUDA examiner means and uncertainties, a 1000 \u00b1 350 prior, EIG selection, no immediate repeats, style diversity, and the frozen terminal/timeout-damage rule. Two predeclared validations followed without score updates.

## Adaptive trace

| Encounter | Examiner | Style group | Outcome | Posterior mu | Posterior sigma | EIG nats | Contact s | Accuracy |
|---:|---|---|---|---:|---:|---:|---:|---:|
{sequence}

Uncertainty contracted from {PRIOR_SIGMA:.1f} to {float(final["posterior_sigma"]):.1f}. Final mean was {float(final["posterior_mu"]):.1f}; the proxy's independently frozen bank rating is {float(ratings[PROXY]["mu"]):.1f}. This is an integration check, not a human-validity estimate.

## Withheld validation

| Encounter | Examiner | Outcome | Predicted win | Brier |
|---:|---|---|---:|---:|
{valid}

## Selector backend comparison

| Strategy | Final mu | Final sigma | Error vs proxy bank mu | Validation Brier |
|---|---:|---:|---:|---:|
{comp}

The comparison is a deterministic replay over already-executed CUDA round-robin outcomes. It is systems evidence, not a participant sample.

## Claim boundary

- Human participant: **no**
- Human pilot complete: **no**
- Training/checkpoint/rating update: **no**
- Hidden or privileged scoring state: **no**
- Systems sequence ready for a future operator-facing pilot: **yes**

The combined status means the two-map demo and Bayesian scoring backend demonstration are technically ready. A genuine human pilot still requires an operator, consent, an anonymized identity, and applicable approval.

{STATUS}
"""


def main() -> int:
    argparse.ArgumentParser().parse_args()
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite: {OUT}")
    ratings_payload = read_json(RATINGS_PATH)
    if ratings_payload.get("status") != PREREQ:
        raise RuntimeError("examiner prerequisite missing")
    if WORK6_STATUS not in DEMO_REPORT.read_text():
        raise RuntimeError("Work Order 6 prerequisite missing")
    ratings = ratings_payload["ratings"]
    candidate_payload = read_json(CANDIDATES_PATH)
    candidates = {row["canonical_id"]: row for row in candidate_payload["candidates"]}
    schedule = read_json(SCHEDULE_PATH)["schedule"]
    pool = diagnostic_pool(ratings)
    if len(pool) < DIAGNOSTIC_COUNT:
        raise RuntimeError("diagnostic pool too small")
    OUT.mkdir(parents=True, exist_ok=False)

    config = {
        "schema_version": "phase6_bayesian_scoring_demo_config_v001",
        "status": "FROZEN_BEFORE_ENCOUNTERS",
        "study_type": "systems_backend_demonstration",
        "participant": {
            "participant_id": PARTICIPANT,
            "mode": "frozen_neural_proxy",
            "policy_id": PROXY,
            "human_participant": False,
            "human_study_evidence": False,
            "proxy_frozen_rating": ratings[PROXY],
        },
        "session_id": SESSION_ID,
        "session_seed": SESSION_SEED,
        "diagnostic_encounters": DIAGNOSTIC_COUNT,
        "validation_encounters": len(VALIDATION_IDS),
        "encounter_time_limit_seconds": 30,
        "player_prior": {"mu": PRIOR_MU, "sigma": PRIOR_SIGMA},
        "rating_model": {
            "name": "Bayesian Bradley-Terry grid posterior",
            "elo_logistic_scale": ELO_SCALE,
            "grid": [float(GRID[0]), float(GRID[-1]), float(GRID[1] - GRID[0])],
            "bot_uncertainty": "11-point Gauss-Hermite integration",
            "draw_update": "fractional sqrt(p_win*(1-p_win)) likelihood",
        },
        "selector": {
            "name": "expected_information_gain",
            "no_immediate_repeat": True,
            "style_constraint": "change public style group when possible",
            "diagnostic_pool": pool,
        },
        "withheld_validation_ids": list(VALIDATION_IDS),
        "timeout_damage_rule": {
            "kill": "terminal winner",
            "timeout_positive_damage": "win",
            "timeout_negative_damage": "loss",
            "timeout_equal_damage": "draw",
        },
        "backend_comparators": [
            "active_eig",
            "random",
            "fixed_easy_to_hard",
            "nearest_rating",
        ],
        "hud": {
            "public_fields": [
                "Rating mean +/- uncertainty",
                "opponent style",
                "encounter number",
                "posterior after outcome",
                "confidence improvement",
            ],
            "internal_names_hidden": True,
        },
        "hashes": {
            "ratings": sha256(RATINGS_PATH),
            "candidate_library": sha256(CANDIDATES_PATH),
            "schedule": sha256(SCHEDULE_PATH),
            "audit_runner": sha256(RUNNER),
        },
        "training": False,
        "checkpoint_update": False,
        "rating_update": False,
    }
    write_json(OUT / "SCORING_CONFIG.json", config)
    write_json(
        OUT / "00_contract/PROXY_DISCLOSURE.json",
        {
            "schema_version": "phase6_human_demo_proxy_disclosure_v001",
            "human_participant": False,
            "human_study_evidence": False,
            "real_operator_smoke_reused": False,
            "allowed_claim": "systems backend demonstration",
            "forbidden_claims": [
                "human pilot complete",
                "human score validated",
                "human-study result",
            ],
            "reason": (
                "No restored real session has 5-8 adaptive encounters; the only "
                "identified operator session is a one-encounter development smoke "
                "marked invalid for behavioral scoring."
            ),
        },
    )
    write_text(OUT / "HUMAN_DEMO_PROTOCOL.md", protocol(pool, candidates))

    posterior = initial_posterior()
    initial_mu, initial_sigma = stats(posterior)
    trace: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = [
        {
            "encounter_number": 0,
            "phase": "initial",
            "selected_bot_id": "",
            "outcome": "",
            "posterior_mu": initial_mu,
            "posterior_sigma": initial_sigma,
            "uncertainty_contraction": 0.0,
            "confidence_improvement_pct": 0.0,
            "posterior_updated": False,
            "session_seed": SESSION_SEED,
        }
    ]
    previous = None
    for number in range(1, DIAGNOSTIC_COUNT + 1):
        bot, information = select_eig(posterior, pool, ratings, previous)
        result, provenance = run_encounter(number, bot, schedule, candidates)
        prior_mu, prior_sigma = stats(posterior)
        posterior = update(
            posterior,
            float(ratings[bot]["mu"]),
            float(ratings[bot]["sigma"]),
            result["outcome"],
        )
        post_mu, post_sigma = stats(posterior)
        confidence = 100.0 * (PRIOR_SIGMA - post_sigma) / PRIOR_SIGMA
        row = {
            "schema_version": "phase6_human_demo_session_trace_v001",
            "session_id": SESSION_ID,
            "participant_id": PARTICIPANT,
            "participant_mode": "frozen_neural_proxy",
            "human_participant": False,
            "encounter_number": number,
            "phase": "diagnostic",
            "selected_bot_id": bot,
            "selected_bot_mu": float(ratings[bot]["mu"]),
            "selected_bot_sigma": float(ratings[bot]["sigma"]),
            "selected_opponent_style": candidates[bot]["style"],
            "selected_style_category": style_group(bot),
            "expected_information_gain_nats": information,
            "prior_mu": prior_mu,
            "prior_sigma": prior_sigma,
            "posterior_mu": post_mu,
            "posterior_sigma": post_sigma,
            "uncertainty_contraction": PRIOR_SIGMA - post_sigma,
            "confidence_improvement_pct": confidence,
            "hud_rating": f"Rating {round(post_mu)} \u00b1 {round(post_sigma)}",
            "hud_encounter": f"Encounter {number}/{DIAGNOSTIC_COUNT}",
            "session_seed": SESSION_SEED,
            "encounter_time_limit_seconds": 30,
            **result,
            **provenance,
        }
        trace.append(row)
        trajectory.append(
            {
                "encounter_number": number,
                "phase": "diagnostic",
                "selected_bot_id": bot,
                "outcome": result["outcome"],
                "posterior_mu": post_mu,
                "posterior_sigma": post_sigma,
                "uncertainty_contraction": PRIOR_SIGMA - post_sigma,
                "confidence_improvement_pct": confidence,
                "posterior_updated": True,
                "session_seed": SESSION_SEED,
            }
        )
        write_jsonl(OUT / "SESSION_TRACE.jsonl", trace)
        write_csv(OUT / "POSTERIOR_TRAJECTORY.csv", trajectory)
        print(
            json.dumps(
                {
                    "encounter": number,
                    "opponent": bot,
                    "outcome": result["outcome"],
                    "posterior_mu": post_mu,
                    "posterior_sigma": post_sigma,
                    "eig": information,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        previous = bot

    validation = []
    for offset, bot in enumerate(VALIDATION_IDS, 1):
        number = DIAGNOSTIC_COUNT + offset
        predicted = float(
            np.sum(
                posterior
                * win_curve(float(ratings[bot]["mu"]), float(ratings[bot]["sigma"]))
            )
        )
        result, provenance = run_encounter(number, bot, schedule, candidates)
        score = {"win": 1.0, "draw": 0.5, "loss": 0.0}[result["outcome"]]
        post_mu, post_sigma = stats(posterior)
        brier = (predicted - score) ** 2
        confidence = 100.0 * (PRIOR_SIGMA - post_sigma) / PRIOR_SIGMA
        row = {
            "schema_version": "phase6_human_demo_session_trace_v001",
            "session_id": SESSION_ID,
            "participant_id": PARTICIPANT,
            "participant_mode": "frozen_neural_proxy",
            "human_participant": False,
            "encounter_number": number,
            "phase": "withheld_validation",
            "selected_bot_id": bot,
            "selected_bot_mu": float(ratings[bot]["mu"]),
            "selected_bot_sigma": float(ratings[bot]["sigma"]),
            "selected_opponent_style": candidates[bot]["style"],
            "selected_style_category": style_group(bot),
            "expected_information_gain_nats": 0.0,
            "prior_mu": post_mu,
            "prior_sigma": post_sigma,
            "posterior_mu": post_mu,
            "posterior_sigma": post_sigma,
            "posterior_updated": False,
            "uncertainty_contraction": PRIOR_SIGMA - post_sigma,
            "confidence_improvement_pct": confidence,
            "predicted_win_probability": predicted,
            "brier_score": brier,
            "hud_rating": f"Final Rating {round(post_mu)} \u00b1 {round(post_sigma)}",
            "hud_encounter": f"Validation {offset}/{len(VALIDATION_IDS)}",
            "session_seed": SESSION_SEED,
            "encounter_time_limit_seconds": 30,
            **result,
            **provenance,
        }
        trace.append(row)
        trajectory.append(
            {
                "encounter_number": number,
                "phase": "withheld_validation",
                "selected_bot_id": bot,
                "outcome": result["outcome"],
                "posterior_mu": post_mu,
                "posterior_sigma": post_sigma,
                "uncertainty_contraction": PRIOR_SIGMA - post_sigma,
                "confidence_improvement_pct": confidence,
                "posterior_updated": False,
                "session_seed": SESSION_SEED,
            }
        )
        validation.append(
            {
                "session_id": SESSION_ID,
                "encounter_number": number,
                "selected_bot_id": bot,
                "selected_opponent_style": candidates[bot]["style"],
                "bot_mu": float(ratings[bot]["mu"]),
                "bot_sigma": float(ratings[bot]["sigma"]),
                "outcome": result["outcome"],
                "timeout_classification": result["timeout_classification"],
                "damage_differential": result["damage_differential"],
                "contact_time_seconds": result["contact_time_seconds"],
                "accuracy": result["accuracy"],
                "predicted_win_probability": predicted,
                "observed_score": score,
                "brier_score": brier,
                "posterior_updated": False,
                "session_seed": SESSION_SEED,
                "run_id": provenance["run_id"],
            }
        )
        write_jsonl(OUT / "SESSION_TRACE.jsonl", trace)
        write_csv(OUT / "POSTERIOR_TRAJECTORY.csv", trajectory)
        write_csv(OUT / "VALIDATION_RESULTS.csv", validation)
        print(
            json.dumps(
                {
                    "encounter": number,
                    "validation_opponent": bot,
                    "outcome": result["outcome"],
                    "predicted_win": predicted,
                    "posterior_updated": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    comparison = compare_backends(ratings, schedule, pool)
    write_csv(OUT / "BACKEND_SELECTOR_COMPARISON.csv", comparison)
    diagnostics = [row for row in trace if row["phase"] == "diagnostic"]
    final_mu, final_sigma = stats(posterior)
    gates = {
        "six_diagnostic_encounters": len(diagnostics) == DIAGNOSTIC_COUNT,
        "two_withheld_validation_encounters": len(validation) == len(VALIDATION_IDS),
        "initial_broad_uncertainty": initial_sigma >= 300,
        "uncertainty_contracted": final_sigma < initial_sigma,
        "no_immediate_repeat": all(
            diagnostics[i]["selected_bot_id"] != diagnostics[i - 1]["selected_bot_id"]
            for i in range(1, len(diagnostics))
        ),
        "different_style_after_first": all(
            diagnostics[i]["selected_style_category"]
            != diagnostics[i - 1]["selected_style_category"]
            for i in range(1, len(diagnostics))
        ),
        "at_least_three_styles": len(
            {row["selected_style_category"] for row in diagnostics}
        )
        >= 3,
        "all_integrity_pass": all(
            row["run_status"] == "PASS" and not row["integrity_failures"]
            for row in trace
        ),
        "validation_did_not_update": all(
            row["posterior_updated"] is False for row in validation
        ),
        "four_selectors_compared": {row["strategy"] for row in comparison}
        == {"active_eig", "random", "fixed_easy_to_hard", "nearest_rating"},
        "human_claim_not_made": all(row["human_participant"] is False for row in trace),
    }
    write_json(
        OUT / "00_contract/FINAL_GATES.json",
        {
            "status": "PASS" if all(gates.values()) else "FAIL",
            "gates": gates,
            "initial_mu": initial_mu,
            "initial_sigma": initial_sigma,
            "final_mu": final_mu,
            "final_sigma": final_sigma,
            "combined_status": STATUS if all(gates.values()) else "",
        },
    )
    if not all(gates.values()):
        raise RuntimeError(f"Work Order 7 gates failed: {gates}")
    write_text(
        OUT / "HUMAN_SCORING_DEMO_REPORT.md",
        report(trace, trajectory, validation, comparison, ratings),
    )
    hash_files = [
        OUT / "HUMAN_DEMO_PROTOCOL.md",
        OUT / "SCORING_CONFIG.json",
        OUT / "SESSION_TRACE.jsonl",
        OUT / "POSTERIOR_TRAJECTORY.csv",
        OUT / "VALIDATION_RESULTS.csv",
        OUT / "BACKEND_SELECTOR_COMPARISON.csv",
        OUT / "HUMAN_SCORING_DEMO_REPORT.md",
        OUT / "00_contract/PROXY_DISCLOSURE.json",
        OUT / "00_contract/FINAL_GATES.json",
    ]
    write_text(
        OUT / "HUMAN_DEMO_HASHES.sha256",
        "".join(f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in hash_files),
    )
    print(
        json.dumps(
            {
                "status": STATUS,
                "participant_mode": "frozen_neural_proxy",
                "human_study_evidence": False,
                "diagnostic_encounters": DIAGNOSTIC_COUNT,
                "validation_encounters": len(VALIDATION_IDS),
                "initial_sigma": initial_sigma,
                "final_mu": final_mu,
                "final_sigma": final_sigma,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
