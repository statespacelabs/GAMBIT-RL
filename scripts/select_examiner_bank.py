#!/usr/bin/env python3
"""Audit, rate, select, and seal the Goal 3 CUDA examiner bank."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "experiments/paper_goal3_cuda_examiner_bank_v001"
FINAL_STATUS = "TANDEMFPS_CUDA_EXAMINER_BANK_CALIBRATED"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def load_matches(
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    candidates = read_json(
        output / "00_contract/CANDIDATE_LIBRARY.json"
    )["candidates"]
    schedule_payload = read_json(
        output / "00_contract/ROUND_ROBIN_SCHEDULE.json"
    )
    schedule = schedule_payload["schedule"]
    execution = read_json(
        output / "01_round_robin/EXECUTION_MANIFEST.json"
    )
    failures: list[str] = []
    if execution.get("status") != "COMPLETE":
        failures.append("execution manifest is not COMPLETE")
    if execution.get("completed") != len(schedule):
        failures.append("execution count differs from frozen schedule")
    rows: list[dict[str, Any]] = []
    for item in schedule:
        run_dir = output / "01_round_robin/raw" / item["run_id"]
        summary_path = run_dir / "run_summary.json"
        match_path = run_dir / "matches.jsonl"
        if not summary_path.is_file() or not match_path.is_file():
            failures.append(f"{item['run_id']}: output missing")
            continue
        summary = read_json(summary_path)
        matches = [
            json.loads(line)
            for line in match_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        checks = {
            "runner_pass": summary.get("status") == "PASS",
            "one_terminal": (
                summary.get("matches_completed") == 1
                and summary.get("target_matches") == 1
                and len(matches) == 1
            ),
            "policy_ids": (
                summary.get("policy_a", {}).get("id") == item["policy_a_id"]
                and summary.get("policy_b", {}).get("id")
                == item["policy_b_id"]
            ),
            "side": summary.get("side_assignment")
            == item["side_assignment"],
            "validation_layout": (
                summary.get("layout_split") == "validation"
                and summary.get("layout_family") == "rooms_doorways"
                and summary.get("layout_seed") == item["layout_seed"]
            ),
            "terminal_complete": (
                summary.get("missing_terminal_outcomes") == 0
                and summary.get("unity_audit_terminal_count_mismatch") == 0
            ),
            "fair_actor": (
                summary.get("actor_received_opponent_id") is False
                and summary.get("actor_received_map_identity") is False
                and summary.get("actor_received_privileged_suffix") is False
                and summary.get("human_controller_count") == 0
                and summary.get("final_heldout_used") is False
            ),
            "runtime_integrity": (
                summary.get("hidden_state_leakage_events") == 0
                and summary.get("observation_nonfinite_values") == 0
                and summary.get("action_nonfinite_values") == 0
                and float(summary.get("critic_prefix_max_abs_error", 1.0))
                <= 1e-5
                and summary.get("side_config_mismatches") == 0
            ),
            "weapon_integrity": (
                summary.get("weapon_ownership_mismatch_count") == 0
                and summary.get("weapon_fire_mismatch_count") == 0
            ),
            "terminal_only_payoff": (
                summary.get("payoff_source")
                == "unity_terminal_event_only"
                and summary.get("dense_reward_used_for_payoff") is False
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            failures.append(f"{item['run_id']}: {failed}")
            continue
        row = matches[0]
        row_valid = (
            row.get("policy_a_id") == item["policy_a_id"]
            and row.get("policy_b_id") == item["policy_b_id"]
            and row.get("side_assignment") == item["side_assignment"]
            and row.get("terminal_payoff_policy_a")
            in (-1.0, 0.0, 1.0)
            and isinstance(row.get("policy_a_damage"), (int, float))
            and isinstance(row.get("policy_b_damage"), (int, float))
        )
        if not row_valid:
            failures.append(f"{item['run_id']}: malformed match row")
            continue
        row["_run_id"] = item["run_id"]
        row["_paired_seed_index"] = item["paired_seed_index"]
        rows.append(row)
    return candidates, rows, failures


def perspective(
    row: dict[str, Any], bot: str
) -> tuple[float, str, float, float, str]:
    if row["policy_a_id"] == bot:
        return (
            (float(row["terminal_payoff_policy_a"]) + 1.0) / 2.0,
            "policy_a",
            float(row["policy_a_damage"]),
            float(row["policy_b_damage"]),
            row["policy_a_physical_slot"],
        )
    return (
        (float(row["terminal_payoff_policy_b"]) + 1.0) / 2.0,
        "policy_b",
        float(row["policy_b_damage"]),
        float(row["policy_a_damage"]),
        row["policy_b_physical_slot"],
    )


def aggregate(
    ids: list[str], rows: list[dict[str, Any]]
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, float]],
]:
    metrics: dict[str, dict[str, Any]] = {}
    for bot in ids:
        relevant = [
            row
            for row in rows
            if bot in (row["policy_a_id"], row["policy_b_id"])
        ]
        values = [perspective(row, bot) for row in relevant]
        scores = [value[0] for value in values]
        prefixes = [value[1] for value in values]
        physical_a = [
            value[0] for value in values if value[4] == "A"
        ]
        physical_b = [
            value[0] for value in values if value[4] == "B"
        ]
        saturations: dict[str, list[float]] = {
            "mean": [],
            "sampled": [],
            "applied": [],
        }
        for row, prefix in zip(relevant, prefixes):
            for source, target in (
                ("mean_saturation", "mean"),
                ("sampled_saturation", "sampled"),
                ("applied_saturation", "applied"),
            ):
                value = row.get(f"{prefix}_{source}")
                if value is not None:
                    saturations[target].append(float(value))
        metrics[bot] = {
            "matches": len(scores),
            "wins": sum(value == 1.0 for value in scores),
            "draws": sum(value == 0.5 for value in scores),
            "losses": sum(value == 0.0 for value in scores),
            "empirical_score": float(np.mean(scores)),
            "damage_for_mean": float(np.mean([value[2] for value in values])),
            "damage_against_mean": float(
                np.mean([value[3] for value in values])
            ),
            "timeout_rate": float(
                np.mean([float(row["timeout"]) for row in relevant])
            ),
            "contact_rate": float(
                np.mean(
                    [
                        float(
                            row.get(f"{prefix}_contact_time_seconds")
                            is not None
                        )
                        for row, prefix in zip(relevant, prefixes)
                    ]
                )
            ),
            "duration_seconds_mean": float(
                np.mean([float(row["duration_seconds"]) for row in relevant])
            ),
            "mu_action_saturation_mean": float(
                np.mean(saturations["mean"])
            ),
            "sampled_action_saturation_mean": float(
                np.mean(saturations["sampled"])
            ),
            "env_applied_action_saturation_mean": float(
                np.mean(saturations["applied"])
            ),
            "physical_a_score": float(np.mean(physical_a)),
            "physical_b_score": float(np.mean(physical_b)),
            "global_physical_side_delta": abs(
                float(np.mean(physical_a)) - float(np.mean(physical_b))
            ),
        }

    pair_rows: list[dict[str, Any]] = []
    profile = {left: {right: 0.5 for right in ids} for left in ids}
    for left, right in itertools.combinations(ids, 2):
        bucket = [
            row
            for row in rows
            if {row["policy_a_id"], row["policy_b_id"]}
            == {left, right}
        ]
        values = [(row, perspective(row, left)) for row in bucket]
        scores = [value[1][0] for value in values]
        side_a = [value[1][0] for value in values if value[1][4] == "A"]
        side_b = [value[1][0] for value in values if value[1][4] == "B"]
        seed_scores: dict[int, list[float]] = {}
        for row, value in values:
            seed_scores.setdefault(row["_paired_seed_index"], []).append(
                value[0]
            )
        score = float(np.mean(scores))
        profile[left][right] = score
        profile[right][left] = 1.0 - score
        pair_rows.append(
            {
                "bot_i": left,
                "bot_j": right,
                "matches": len(bucket),
                "wins_i": sum(value == 1.0 for value in scores),
                "draws": sum(value == 0.5 for value in scores),
                "losses_i": sum(value == 0.0 for value in scores),
                "empirical_score_i": score,
                "empirical_score_j": 1.0 - score,
                "score_i_physical_a": float(np.mean(side_a)),
                "score_i_physical_b": float(np.mean(side_b)),
                "side_score_delta": abs(
                    float(np.mean(side_a)) - float(np.mean(side_b))
                ),
                "paired_side_outcome_delta_mean": float(
                    np.mean(
                        [
                            abs(pair[0] - pair[1])
                            for pair in seed_scores.values()
                            if len(pair) == 2
                        ]
                    )
                ),
                "timeout_rate": float(
                    np.mean([float(row["timeout"]) for row in bucket])
                ),
                "damage_i_mean": float(
                    np.mean([value[1][2] for value in values])
                ),
                "damage_j_mean": float(
                    np.mean([value[1][3] for value in values])
                ),
            }
        )
    for bot in ids:
        relevant_pairs = [
            row
            for row in pair_rows
            if bot in (row["bot_i"], row["bot_j"])
        ]
        opponent_scores = [
            profile[bot][other] for other in ids if other != bot
        ]
        metrics[bot]["max_pair_side_score_delta"] = max(
            row["side_score_delta"] for row in relevant_pairs
        )
        metrics[bot]["min_pair_score"] = min(opponent_scores)
        metrics[bot]["max_pair_score"] = max(opponent_scores)
    return metrics, pair_rows, profile


def fit_bt(
    ids: list[str],
    rows: list[dict[str, Any]],
    prior_sigma: float,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    index = {bot: position for position, bot in enumerate(ids)}
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        if predicate is not None and not predicate(row):
            continue
        left, right = sorted((row["policy_a_id"], row["policy_b_id"]))
        grouped.setdefault((index[left], index[right]), []).append(
            perspective(row, left)[0]
        )
    expected = len(ids) * (len(ids) - 1) // 2
    if len(grouped) != expected:
        raise RuntimeError(f"rating subset has {len(grouped)}/{expected} pairs")
    scale = 400.0 / math.log(10.0)
    rating = np.zeros(len(ids), dtype=np.float64)
    information = np.eye(len(ids), dtype=np.float64)
    for _ in range(100):
        gradient = -rating / prior_sigma**2
        information = np.eye(len(ids), dtype=np.float64) / prior_sigma**2
        for (left, right), outcomes in grouped.items():
            delta = np.clip(
                (rating[left] - rating[right]) / scale, -30.0, 30.0
            )
            probability = 1.0 / (1.0 + math.exp(-float(delta)))
            gradient_value = (
                sum(outcomes) - len(outcomes) * probability
            ) / scale
            gradient[left] += gradient_value
            gradient[right] -= gradient_value
            weight = (
                len(outcomes)
                * probability
                * (1.0 - probability)
                / scale**2
            )
            information[left, left] += weight
            information[right, right] += weight
            information[left, right] -= weight
            information[right, left] -= weight
        update = np.linalg.solve(information, gradient)
        rating += update
        rating -= float(np.mean(rating))
        if float(np.max(np.abs(update))) < 1e-9:
            break
    covariance = np.linalg.inv(information)
    mu = {
        bot: float(1000.0 + rating[index[bot]]) for bot in ids
    }
    sigma = {
        bot: float(
            math.sqrt(max(0.0, covariance[index[bot], index[bot]]))
        )
        for bot in ids
    }
    return mu, sigma


def correlations(
    baseline: dict[str, float],
    comparison: dict[str, float],
    ids: list[str],
) -> tuple[float, float]:
    left = np.asarray([baseline[bot] for bot in ids])
    right = np.asarray([comparison[bot] for bot in ids])
    pearson = float(np.corrcoef(left, right)[0, 1])
    rank_left = np.argsort(np.argsort(left)).astype(float)
    rank_right = np.argsort(np.argsort(right)).astype(float)
    spearman = float(np.corrcoef(rank_left, rank_right)[0, 1])
    return pearson, spearman


def combo_utility(
    combo: tuple[str, ...],
    profile: dict[str, dict[str, float]],
    metrics: dict[str, dict[str, Any]],
) -> float:
    distances = []
    for left, right in itertools.combinations(combo, 2):
        opponents = [
            bot for bot in profile if bot not in (left, right)
        ]
        distances.append(
            math.sqrt(
                float(
                    np.mean(
                        [
                            (profile[left][bot] - profile[right][bot]) ** 2
                            for bot in opponents
                        ]
                    )
                )
            )
        )
    rating_span = (
        max(metrics[bot]["bayesian_mu"] for bot in combo)
        - min(metrics[bot]["bayesian_mu"] for bot in combo)
    ) / 400.0
    penalty = float(
        np.mean(
            [
                metrics[bot]["max_pair_side_score_delta"]
                + 0.25 * metrics[bot]["timeout_rate"]
                + metrics[bot]["bayesian_sigma"] / 300.0
                for bot in combo
            ]
        )
    )
    return 8.0 * float(np.mean(distances)) + rating_span - penalty


def choose_class(
    class_ids: list[str],
    profile: dict[str, dict[str, float]],
    metrics: dict[str, dict[str, Any]],
    required_primary_prefix: str | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    hard_flags: dict[str, list[str]] = {}
    for bot in class_ids:
        row = metrics[bot]
        flags = []
        if row["bayesian_sigma"] > 120.0:
            flags.append("posterior_uncertainty")
        if row["timeout_rate"] > 0.90:
            flags.append("trivial_timeout_controller")
        if row["max_pair_score"] < 0.05:
            flags.append("trivial_to_library")
        if row["min_pair_score"] > 0.95:
            flags.append("impossible_for_library")
        hard_flags[bot] = flags
    eligible = [bot for bot in class_ids if not hard_flags[bot]]
    if len(eligible) < 5:
        raise RuntimeError(f"fewer than five eligible in class: {hard_flags}")
    five_scores = [
        (combo_utility(combo, profile, metrics), combo)
        for combo in itertools.combinations(eligible, 5)
    ]
    selected = list(max(five_scores, key=lambda item: item[0])[1])
    four_scores = [
        (combo_utility(combo, profile, metrics), combo)
        for combo in itertools.combinations(selected, 4)
        if required_primary_prefix is None
        or any(bot.startswith(required_primary_prefix) for bot in combo)
    ]
    if not four_scores:
        raise RuntimeError(
            "no primary combination satisfies the family-diversity constraint"
        )
    primary = list(max(four_scores, key=lambda item: item[0])[1])
    reserve = [bot for bot in selected if bot not in primary]
    nearest: dict[str, Any] = {}
    for bot in class_ids:
        options = []
        for other in class_ids:
            if bot == other:
                continue
            opponents = [
                value for value in profile if value not in (bot, other)
            ]
            left = np.asarray([profile[bot][value] for value in opponents])
            right = np.asarray([profile[other][value] for value in opponents])
            rmse = float(np.sqrt(np.mean((left - right) ** 2)))
            correlation = (
                float(np.corrcoef(left, right)[0, 1])
                if float(np.std(left)) > 0 and float(np.std(right)) > 0
                else 0.0
            )
            options.append((rmse, -correlation, other, correlation))
        best = min(options)
        nearest[bot] = {
            "bot_id": best[2],
            "profile_rmse": best[0],
            "profile_correlation": best[3],
            "redundant": (
                best[0] <= 0.08
                and best[3] >= 0.97
                and abs(
                    metrics[bot]["bayesian_mu"]
                    - metrics[best[2]]["bayesian_mu"]
                )
                <= 60.0
            ),
        }
    return primary, reserve, {
        "hard_flags": hard_flags,
        "eligible": eligible,
        "selected": selected,
        "primary": primary,
        "reserve": reserve,
        "excluded": [bot for bot in class_ids if bot not in selected],
        "nearest_profile": nearest,
        "five_candidate_combinations": [
            {"bots": list(combo), "utility": utility}
            for utility, combo in sorted(five_scores, reverse=True)
        ],
    }


def main() -> int:
    global FINAL_STATUS
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--analysis-only", action="store_true", help="Replay recorded outcomes without certifying external checkpoint/build files.")
    args = parser.parse_args()
    FINAL_STATUS = "GAMBIT_RECORDED_CALIBRATION_REPLAY" if args.analysis_only else "TANDEMFPS_CUDA_EXAMINER_BANK_CALIBRATED"
    output = args.output.resolve()
    candidates, rows, failures = load_matches(output)
    ids = sorted(row["canonical_id"] for row in candidates)
    expected = len(ids) * (len(ids) - 1) // 2 * 20 * 2
    technical = {
        "schema_version": "phase6_cuda_examiner_technical_audit_v001",
        "status": (
            "PASS" if not failures and len(rows) == expected else "FAIL"
        ),
        "candidate_count": len(ids),
        "expected_matches": expected,
        "audited_matches": len(rows),
        "failures": failures,
        "terminal_payoff_source": "unity_terminal_event_only",
        "dense_reward_used_for_payoff": False,
        "damage_complete": all(
            isinstance(row.get("policy_a_damage"), (int, float))
            and isinstance(row.get("policy_b_damage"), (int, float))
            for row in rows
        ),
        "final_heldout_used": False,
        "layout_family": "rooms_doorways",
    }
    write_json(output / "02_analysis/TECHNICAL_AUDIT.json", technical)
    if technical["status"] != "PASS":
        raise RuntimeError(f"technical audit failed: {failures[:3]}")

    metrics, pair_rows, profile = aggregate(ids, rows)
    bayes_mu, bayes_sigma = fit_bt(ids, rows, 350.0)
    for bot in ids:
        metrics[bot]["bayesian_mu"] = bayes_mu[bot]
        metrics[bot]["bayesian_sigma"] = bayes_sigma[bot]

    sensitivity_specs: list[
        tuple[
            str,
            float,
            Callable[[dict[str, Any]], bool] | None,
        ]
    ] = [
        ("classic_bt_weak_prior", 5000.0, None),
        ("bayes_prior_sigma_200", 200.0, None),
        ("bayes_prior_sigma_700", 700.0, None),
        (
            "schedule_base_only",
            350.0,
            lambda row: row["side_assignment"] == "base",
        ),
        (
            "schedule_swapped_only",
            350.0,
            lambda row: row["side_assignment"] == "swapped",
        ),
        (
            "paired_seeds_00_09",
            350.0,
            lambda row: row["_paired_seed_index"] < 10,
        ),
        (
            "paired_seeds_10_19",
            350.0,
            lambda row: row["_paired_seed_index"] >= 10,
        ),
    ]
    sensitivity: dict[str, Any] = {}
    for name, prior, predicate in sensitivity_specs:
        mu, sigma = fit_bt(ids, rows, prior, predicate)
        pearson, spearman = correlations(bayes_mu, mu, ids)
        sensitivity[name] = {
            "prior_sigma": prior,
            "ratings": mu,
            "posterior_sigma": sigma,
            "pearson_with_baseline": pearson,
            "spearman_with_baseline": spearman,
            "maximum_absolute_mu_delta": max(
                abs(mu[bot] - bayes_mu[bot]) for bot in ids
            ),
        }

    predicted = {
        left: {
            right: (
                0.5
                if left == right
                else 1.0
                / (
                    1.0
                    + 10.0
                    ** (-(bayes_mu[left] - bayes_mu[right]) / 400.0)
                )
            )
            for right in ids
        }
        for left in ids
    }
    class_by_id = {
        row["canonical_id"]: row["candidate_class"] for row in candidates
    }
    anchors = [
        bot for bot in ids if class_by_id[bot] == "parametric_anchor"
    ]
    learned = [
        bot for bot in ids if class_by_id[bot] == "learned_or_hybrid"
    ]
    anchor_primary, anchor_reserve, anchor_audit = choose_class(
        anchors, profile, metrics
    )
    learned_primary, learned_reserve, learned_audit = choose_class(
        learned,
        profile,
        metrics,
        required_primary_prefix="bank_phase5_",
    )
    primary = sorted(
        anchor_primary + learned_primary,
        key=lambda bot: bayes_mu[bot],
        reverse=True,
    )
    reserve = sorted(
        anchor_reserve + learned_reserve,
        key=lambda bot: bayes_mu[bot],
        reverse=True,
    )
    selected = primary + reserve
    composition = {
        "primary": len(primary),
        "reserve": len(reserve),
        "primary_anchor": sum(
            class_by_id[bot] == "parametric_anchor" for bot in primary
        ),
        "primary_learned": sum(
            class_by_id[bot] == "learned_or_hybrid" for bot in primary
        ),
        "reserve_anchor": sum(
            class_by_id[bot] == "parametric_anchor" for bot in reserve
        ),
        "reserve_learned": sum(
            class_by_id[bot] == "learned_or_hybrid" for bot in reserve
        ),
    }
    if composition != {
        "primary": 8,
        "reserve": 2,
        "primary_anchor": 4,
        "primary_learned": 4,
        "reserve_anchor": 1,
        "reserve_learned": 1,
    }:
        raise RuntimeError(f"composition gate failed: {composition}")

    candidate_by_id = {
        row["canonical_id"]: row for row in candidates
    }
    selection = {
        "schema_version": "phase6_cuda_examiner_selection_audit_v001",
        "status": "PASS",
        "method": (
            "hard reliability/triviality filters followed by exhaustive "
            "within-class payoff-profile diversity selection"
        ),
        "parametric_anchor": anchor_audit,
        "learned_or_hybrid": learned_audit,
        "primary": primary,
        "reserve": reserve,
        "not_selected": [bot for bot in ids if bot not in selected],
    }
    write_json(output / "02_analysis/SELECTION_AUDIT.json", selection)
    write_json(
        output / "02_analysis/ALL_CANDIDATE_METRICS.json",
        {
            "schema_version": "phase6_cuda_examiner_metrics_v001",
            "metrics": metrics,
        },
    )
    pair_path = output / "02_analysis/PAIR_DIAGNOSTICS.csv"
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    with pair_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    ratings_path = output / "BOT_RATINGS_CUDA.json"
    write_json(
        ratings_path,
        {
            "schema_version": "phase6_cuda_examiner_ratings_v001",
            "status": FINAL_STATUS,
            "model": {
                "name": "Bayesian Bradley-Terry",
                "elo_logistic_scale": 400.0,
                "gaussian_prior_mean": 1000.0,
                "gaussian_prior_sigma": 350.0,
                "draw_score": 0.5,
                "mean_rating": float(np.mean(list(bayes_mu.values()))),
                "payoff_source": "unity_terminal_event_only",
            },
            "ratings": {
                bot: {
                    "mu": bayes_mu[bot],
                    "sigma": bayes_sigma[bot],
                    "role": (
                        "primary"
                        if bot in primary
                        else "reserve"
                        if bot in reserve
                        else "screened_not_selected"
                    ),
                    "candidate_class": class_by_id[bot],
                    "metrics": metrics[bot],
                }
                for bot in ids
            },
            "bradley_terry_sensitivity": sensitivity,
            "predicted_win_score_matrix_all_screened": predicted,
        },
    )

    payoff_path = output / "BOT_PAYOFF_MATRIX.csv"
    with payoff_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bot_id"] + selected)
        for left in selected:
            writer.writerow(
                [left]
                + [
                    f"{profile[left][right]:.6f}"
                    if left != right
                    else "0.500000"
                    for right in selected
                ]
            )

    bots = []
    for bot in primary + reserve:
        source = candidate_by_id[bot]
        bots.append(
            {
                "canonical_id": bot,
                "role": "primary" if bot in primary else "reserve",
                "candidate_class": source["candidate_class"],
                "family": source["family"],
                "style": source["style"],
                "diagnostic_purpose": source["diagnostic_purpose"],
                "difficulty_parameters": source["difficulty_parameters"],
                "checkpoint": source["checkpoint"],
                "checkpoint_sha256": source["checkpoint_sha256"],
                "model_or_controller_sha256": source[
                    "model_or_controller_sha256"
                ],
                "combat_checkpoint": source["combat_checkpoint"],
                "combat_sha256": source["combat_sha256"],
                "normalizers": source["normalizers"],
                "observation_schema": source["observation_schema"],
                "action_schema": source["action_schema"],
                "fairness": source["fairness"],
                "rating": {
                    "mu": bayes_mu[bot],
                    "sigma": bayes_sigma[bot],
                },
                "calibration_metrics": metrics[bot],
            }
        )
    final_path = output / "FINAL_CUDA_BOT_BANK.json"
    write_json(
        final_path,
        {
            "schema_version": "final_cuda_bot_bank_v001",
            "status": FINAL_STATUS,
            "training_performed": False,
            "calibration": {
                "device": "cuda:0",
                "candidate_count": len(ids),
                "primary_count": len(primary),
                "reserve_count": len(reserve),
                "unordered_pair_count": len(ids) * (len(ids) - 1) // 2,
                "paired_seeds_per_unordered_pair": 20,
                "side_swapped": True,
                "terminal_matches": len(rows),
                "layout_split": "validation",
                "layout_family": "rooms_doorways",
                "layout_seeds": [550002, 550007, 550012, 550017],
                "final_heldout_used": False,
                "terminal_and_damage_complete": True,
                "mean_bayesian_mu": float(
                    np.mean(list(bayes_mu.values()))
                ),
            },
            "primary_ids": primary,
            "reserve_ids": reserve,
            "bots": bots,
            "artifacts": {
                "ratings": "BOT_RATINGS_CUDA.json",
                "payoff_matrix": "BOT_PAYOFF_MATRIX.csv",
                "hash_ledger": "BOT_BANK_HASHES.sha256",
                "report": "BOT_CALIBRATION_REPORT.md",
                "technical_audit": "02_analysis/TECHNICAL_AUDIT.json",
                "selection_audit": "02_analysis/SELECTION_AUDIT.json",
                "pair_diagnostics": "02_analysis/PAIR_DIAGNOSTICS.csv",
            },
        },
    )

    selected_lines = []
    for bot in primary + reserve:
        row = metrics[bot]
        selected_lines.append(
            f"| {bot} | {'primary' if bot in primary else 'reserve'} | "
            f"{class_by_id[bot]} | {bayes_mu[bot]:.1f} | "
            f"{bayes_sigma[bot]:.1f} | {row['empirical_score']:.3f} | "
            f"{row['contact_rate']:.3f} | {row['timeout_rate']:.3f} | "
            f"{row['max_pair_side_score_delta']:.3f} |"
        )
    excluded_lines = []
    for bot in ids:
        if bot in selected:
            continue
        audit = (
            anchor_audit
            if class_by_id[bot] == "parametric_anchor"
            else learned_audit
        )
        nearest = audit["nearest_profile"][bot]
        flags = audit["hard_flags"][bot]
        reason = (
            ", ".join(flags)
            if flags
            else "capacity-pruned for lower incremental profile utility"
        )
        excluded_lines.append(
            f"| {bot} | {reason} | {nearest['bot_id']} | "
            f"{nearest['profile_rmse']:.3f} | "
            f"{nearest['profile_correlation']:.3f} |"
        )
    sensitivity_lines = [
        f"| {name} | {row['pearson_with_baseline']:.4f} | "
        f"{row['spearman_with_baseline']:.4f} | "
        f"{row['maximum_absolute_mu_delta']:.1f} |"
        for name, row in sensitivity.items()
    ]
    report_path = output / "BOT_CALIBRATION_REPORT.md"
    write_text(
        report_path,
        "\n".join(
            [
                "# CUDA Examiner Bank Calibration",
                "",
                f"Status: **{FINAL_STATUS}**",
                "",
                "## Outcome",
                "",
                (
                    f"Frozen bank: {len(primary)} primary and {len(reserve)} "
                    f"reserve bots, selected from {len(ids)} candidates after "
                    f"{len(rows):,} terminal matches."
                ),
                (
                    "Composition is four parametric and four learned/hybrid "
                    "primaries, plus one reserve of each class."
                ),
                "",
                "## Frozen protocol",
                "",
                "- CUDA device cuda:0; no training or normalizer updates.",
                "- 66 unordered pairs; 20 paired seeds; both side assignments.",
                "- Frozen rooms_doorways validation family; no final heldout.",
                "- Unity terminal events are the only rating payoff source.",
                "- Terminal, damage, weapon/reset, schema, and fairness audits passed.",
                "",
                "## Selected library",
                "",
                "| Bot | Role | Class | mu | sigma | Score | Contact | Timeout | Max side delta |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
                *selected_lines,
                "",
                "## Screened but not selected",
                "",
                "| Bot | Reason | Nearest profile | RMSE | Correlation |",
                "|---|---|---|---:|---:|",
                *excluded_lines,
                "",
                "## Bradley-Terry sensitivity",
                "",
                "| Fit | Pearson | Spearman | Max mu delta |",
                "|---|---:|---:|---:|",
                *sensitivity_lines,
                "",
                "## Integrity",
                "",
                f"- Technical audit: PASS ({len(rows)}/{expected} matches).",
                "- Missing terminal outcomes and incomplete damage rows: 0.",
                "- Hidden-state leaks, privileged actor inputs, human input, and final-heldout use: 0.",
                "- Weapon ownership/fire and side-configuration failures: 0.",
                "",
                "Ratings measure diagnostic difficulty on this frozen CUDA "
                "protocol, not universal skill. Selection uses payoff-profile "
                "separation and reliability rather than win rate alone.",
                "",
                FINAL_STATUS,
                "",
            ]
        ),
    )

    if args.analysis_only:
        print(json.dumps({"status": FINAL_STATUS, "matches": len(rows), "primary": primary, "reserve": reserve, "output": str(output)}, sort_keys=True))
        return 0

    ledger_entries: dict[str, str] = {}
    for bot in selected:
        source = candidate_by_id[bot]
        ledger_entries[source["checkpoint"]] = source["checkpoint_sha256"]
        ledger_entries[source["combat_checkpoint"]] = source["combat_sha256"]
    frozen = read_json(
        output / "00_contract/CALIBRATION_CONTRACT.json"
    )["frozen_hashes"]
    for value in frozen.values():
        ledger_entries[value["path"]] = value["sha256"]
    for path in (
        final_path,
        ratings_path,
        payoff_path,
        report_path,
        output / "02_analysis/TECHNICAL_AUDIT.json",
        output / "02_analysis/SELECTION_AUDIT.json",
        pair_path,
    ):
        ledger_entries[str(path.relative_to(ROOT))] = sha256_file(path)
    ledger_path = output / "BOT_BANK_HASHES.sha256"
    write_text(
        ledger_path,
        "".join(
            f"{digest}  {path}\n"
            for path, digest in sorted(ledger_entries.items())
        ),
    )
    for relative, expected_hash in ledger_entries.items():
        actual = sha256_file(ROOT / relative)
        if actual != expected_hash:
            raise RuntimeError(
                f"hash verification failed for {relative}: "
                f"{actual} != {expected_hash}"
            )
    mean_rating = float(np.mean(list(bayes_mu.values())))
    if abs(mean_rating - 1000.0) > 1e-8:
        raise RuntimeError(f"ratings not centered: {mean_rating}")
    print(
        json.dumps(
            {
                "status": FINAL_STATUS,
                "primary": primary,
                "reserve": reserve,
                "matches": len(rows),
                "mean_rating": mean_rating,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
