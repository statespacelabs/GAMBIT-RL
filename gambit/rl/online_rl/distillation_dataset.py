"""Real-data dataset for Phase 3 command distillation.

The dataset joins Phase 2 transitions to their state windows, resolves the
normalized telemetry frame at the transition decision time, and converts raw
Phase 2 action summaries into bounded Unity command targets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .action_schema import ActionCommandConverter

logger = logging.getLogger(__name__)


class DistillationDataset(Dataset):
    """Dataset of aligned ``(obs_45, command_action_8)`` samples."""

    def __init__(
        self,
        transition_manifest_path: str | Path,
        window_manifest_path: str | Path | None = None,
        mode: str = "single",
        seq_len: int = 32,
        split: str = "train",
        action_scaler_path: str | Path | None = None,
        action_converter: ActionCommandConverter | None = None,
        telemetry_fps: float = 30.0,
        step_seconds: float = 1.0,
        require_normalized_telemetry: bool = True,
    ):
        if mode not in ("single", "sequence"):
            raise ValueError(f"mode must be 'single' or 'sequence', got {mode!r}")
        if telemetry_fps <= 0 or step_seconds <= 0:
            raise ValueError("telemetry_fps and step_seconds must be positive")

        self.mode = mode
        self.seq_len = seq_len
        self.telemetry_fps = float(telemetry_fps)
        self.step_seconds = float(step_seconds)
        self.require_normalized_telemetry = require_normalized_telemetry
        self.action_converter = action_converter or ActionCommandConverter()
        self.transition_manifest_path = Path(transition_manifest_path)
        if window_manifest_path is None:
            window_manifest_path = (
                self.transition_manifest_path.parent / "phase2_window_manifest.csv"
            )
        self.window_manifest_path = Path(window_manifest_path)

        if not self.transition_manifest_path.exists():
            raise FileNotFoundError(
                f"Transition manifest not found: {self.transition_manifest_path}"
            )
        if not self.window_manifest_path.exists():
            raise FileNotFoundError(
                f"Window manifest not found: {self.window_manifest_path}"
            )

        transitions = pd.read_csv(self.transition_manifest_path)
        windows = pd.read_csv(self.window_manifest_path)
        self._validate_manifest_columns(transitions, windows)
        self.df = self._join_manifests(transitions, windows, split)
        self._resolve_action_paths()
        self._build_telemetry_source_index(windows)

        if action_scaler_path:
            logger.warning(
                "Ignoring action_scaler_path=%s. Phase 2 action files are raw "
                "physical summaries and are converted by ActionCommandConverter.",
                action_scaler_path,
            )

        if mode == "sequence":
            self._build_sequence_indices()

        logger.info(
            "DistillationDataset [%s/%s]: %d transitions",
            split,
            mode,
            len(self.df),
        )

    @staticmethod
    def _validate_manifest_columns(
        transitions: pd.DataFrame, windows: pd.DataFrame
    ) -> None:
        required_transition = {
            "transition_id",
            "session_id",
            "t",
            "state_window_id",
            "action_path",
            "done",
            "split",
        }
        required_window = {
            "window_id",
            "session_id",
            "t_start",
            "t_end",
            "tel_npz_path",
            "split",
        }
        missing_transition = required_transition - set(transitions.columns)
        missing_window = required_window - set(windows.columns)
        if missing_transition:
            raise ValueError(
                f"Transition manifest missing columns: {sorted(missing_transition)}"
            )
        if missing_window:
            raise ValueError(
                f"Window manifest missing columns: {sorted(missing_window)}"
            )
        if windows["window_id"].duplicated().any():
            raise ValueError("Window manifest contains duplicate window_id values")

    @staticmethod
    def _join_manifests(
        transitions: pd.DataFrame, windows: pd.DataFrame, split: str
    ) -> pd.DataFrame:
        transitions = transitions[transitions["split"] == split].copy()
        state_windows = windows[
            ["window_id", "session_id", "t_start", "t_end", "tel_npz_path", "split"]
        ].rename(
            columns={
                "session_id": "window_session_id",
                "split": "window_split",
            }
        )
        merged = transitions.merge(
            state_windows,
            left_on="state_window_id",
            right_on="window_id",
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        unmatched = merged[merged["_merge"] != "both"]
        if len(unmatched):
            first = unmatched.iloc[0]
            raise ValueError(
                "Transition state_window_id has no window-manifest row: "
                f"{first['state_window_id']}"
            )
        if (
            merged["session_id"].astype(str) != merged["window_session_id"].astype(str)
        ).any():
            raise ValueError("Transition/window session_id mismatch")
        if (merged["split"] != merged["window_split"]).any():
            raise ValueError("Transition/window split mismatch")
        return merged.drop(columns=["_merge"]).reset_index(drop=True)

    def _resolve_action_paths(self) -> None:
        base = self.transition_manifest_path.parent
        for idx, value in self.df["action_path"].items():
            path = Path(str(value))
            if not path.is_absolute():
                path = (base / path).resolve()
            self.df.at[idx, "action_path"] = str(path)

    def _build_telemetry_source_index(self, windows: pd.DataFrame) -> None:
        base = self.window_manifest_path.parent
        self._sources_by_session: dict[str, list[Path]] = {}
        self._source_meta_cache: dict[Path, tuple[float, float, int]] = {}
        for _, row in windows.iterrows():
            raw_path = row.get("tel_npz_path", "")
            if pd.isna(raw_path) or not str(raw_path).strip():
                continue
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = (base / path).resolve()
            sources = self._sources_by_session.setdefault(str(row["session_id"]), [])
            if path not in sources:
                sources.append(path)

    def _source_meta(self, path: Path) -> tuple[float, float, int]:
        if path in self._source_meta_cache:
            return self._source_meta_cache[path]
        if not path.exists():
            raise FileNotFoundError(f"Telemetry NPZ not found: {path}")
        with np.load(path) as data:
            required = {"where_tel", "view_tel", "rhythm_tel"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(
                    f"Telemetry NPZ {path} missing arrays: {sorted(missing)}"
                )
            n_frames = int(data["where_tel"].shape[0])
            start = float(data["start_time"]) if "start_time" in data else 0.0
            end = (
                float(data["end_time"])
                if "end_time" in data
                else start + n_frames / self.telemetry_fps
            )
        meta = (start, end, n_frames)
        self._source_meta_cache[path] = meta
        return meta

    def _select_telemetry_source(self, row: pd.Series) -> tuple[Path, int, float]:
        session_id = str(row["session_id"])
        decision_t = float(row["t"])
        # Phase 2 state windows are right-open [t_start, t_end). The action at
        # decision_t must therefore observe the final frame before decision_t.
        sample_t = decision_t - 1.0 / self.telemetry_fps
        sources = self._sources_by_session.get(session_id, [])
        if not sources:
            raise FileNotFoundError(
                f"No telemetry NPZ sources for session {session_id}"
            )

        candidates: list[tuple[float, Path, int, float]] = []
        for path in sources:
            start, end, n_frames = self._source_meta(path)
            if start - 1e-6 <= sample_t < end - 1e-6:
                frame_idx = int(
                    np.floor((sample_t - start) * self.telemetry_fps + 1e-6)
                )
                frame_idx = min(max(frame_idx, 0), n_frames - 1)
                frame_t = start + frame_idx / self.telemetry_fps
                candidates.append((start, path, frame_idx, frame_t))

        if not candidates:
            for path in sources:
                start, end, n_frames = self._source_meta(path)
                if abs(sample_t - end) <= 1.0 / self.telemetry_fps + 1e-6:
                    frame_idx = n_frames - 1
                    frame_t = start + frame_idx / self.telemetry_fps
                    candidates.append((start, path, frame_idx, frame_t))

        if not candidates:
            raise ValueError(
                f"Pre-action sample time {sample_t:.6f} for transition "
                f"{row.get('transition_id', '?')} is not covered by session telemetry"
            )

        _, path, frame_idx, frame_t = max(candidates, key=lambda item: item[0])
        return path, frame_idx, frame_t

    def _load_telemetry_with_meta(
        self, row: pd.Series
    ) -> tuple[np.ndarray, dict[str, float | str | int]]:
        path, frame_idx, frame_t = self._select_telemetry_source(row)
        with np.load(path) as data:
            if self.require_normalized_telemetry and (
                "normalized" not in data or not bool(data["normalized"])
            ):
                raise ValueError(f"Telemetry NPZ is not marked normalized: {path}")
            where = data["where_tel"]
            view = data["view_tel"]
            rhythm = data["rhythm_tel"]
            n_frames = where.shape[0]
            if where.shape != (n_frames, 10):
                raise ValueError(f"Bad where telemetry shape in {path}: {where.shape}")
            if view.shape != (n_frames, 13):
                raise ValueError(f"Bad view telemetry shape in {path}: {view.shape}")
            if rhythm.shape != (n_frames, 22):
                raise ValueError(
                    f"Bad rhythm telemetry shape in {path}: {rhythm.shape}"
                )
            tel_45 = np.concatenate(
                [where[frame_idx], view[frame_idx], rhythm[frame_idx]]
            ).astype(np.float32)

        if not np.isfinite(tel_45).all():
            raise ValueError(f"Nonfinite telemetry in {path} at frame {frame_idx}")
        if np.allclose(tel_45, 0.0):
            raise ValueError(f"All-zero telemetry in {path} at frame {frame_idx}")
        return tel_45, {
            "path": str(path),
            "frame_idx": frame_idx,
            "frame_time": frame_t,
            "decision_time": float(row["t"]),
        }

    def _load_telemetry(self, row: pd.Series) -> np.ndarray:
        telemetry, _ = self._load_telemetry_with_meta(row)
        return telemetry

    def _load_core_action(self, row: pd.Series) -> np.ndarray:
        action_path = Path(str(row.get("action_path", "")))
        if not action_path.exists():
            raise FileNotFoundError(
                "Missing action for transition "
                f"{row.get('transition_id', '?')}: {action_path}"
            )
        with np.load(action_path) as data:
            action_14 = data["action_vector"].astype(np.float32)
        return self.action_converter.phase2_action14_to_command8(action_14)

    def _build_sequence_indices(self) -> None:
        self.sequences: list[list[int]] = []
        for _, group in self.df.groupby("session_id"):
            runs: list[list[int]] = []
            current: list[int] = []
            previous_t: float | None = None
            previous_done = False
            for row_idx, row in group.sort_values("t").iterrows():
                t = float(row["t"])
                contiguous = previous_t is None or (
                    not previous_done
                    and abs((t - previous_t) - self.step_seconds) <= 1e-4
                )
                if not contiguous and current:
                    runs.append(current)
                    current = []
                current.append(int(row_idx))
                previous_t = t
                previous_done = bool(row["done"])
                if previous_done:
                    runs.append(current)
                    current = []
                    previous_t = None
                    previous_done = False
            if current:
                runs.append(current)

            for indices in runs:
                for start in range(len(indices) - self.seq_len + 1):
                    self.sequences.append(indices[start : start + self.seq_len])

        logger.info(
            "DistillationDataset: %d sequences of length %d",
            len(self.sequences),
            self.seq_len,
        )

    def __len__(self) -> int:
        return len(self.sequences) if self.mode == "sequence" else len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "sequence":
            rows = [self.df.iloc[i] for i in self.sequences[idx]]
            telemetry = np.stack([self._load_telemetry(row) for row in rows])
            actions = np.stack([self._load_core_action(row) for row in rows])
        else:
            row = self.df.iloc[idx]
            telemetry = self._load_telemetry(row)
            actions = self._load_core_action(row)
        return torch.from_numpy(telemetry).float(), torch.from_numpy(actions).float()

    def verify_alignment(self, n_sessions: int = 20) -> dict:
        """Verify state/action IDs and decision-time frame selection."""
        rng = np.random.default_rng(42)
        session_ids = self.df["session_id"].unique()
        if len(session_ids) > n_sessions:
            session_ids = rng.choice(session_ids, size=n_sessions, replace=False)

        errors: list[str] = []
        timestamp_checks = 0
        action_timestamp_checks = 0
        paired = {offset: ([], []) for offset in (-1, 0, 1)}

        for session_id in session_ids:
            rows = [
                group_row
                for _, group_row in self.df[self.df["session_id"] == session_id]
                .sort_values("t")
                .iterrows()
            ]
            telemetry_values: list[float] = []
            action_values: list[float] = []
            for row in rows:
                try:
                    telemetry, meta = self._load_telemetry_with_meta(row)
                    command = self._load_core_action(row)
                    frame_error = abs(
                        float(meta["decision_time"]) - float(meta["frame_time"])
                    )
                    if frame_error > 1.0 / self.telemetry_fps + 1e-5:
                        errors.append(
                            f"{row['transition_id']}: decision/frame error={frame_error}"
                        )
                    timestamp_checks += 1

                    action_path = Path(str(row["action_path"]))
                    with np.load(action_path) as action_data:
                        for key, expected in (
                            ("transition_id", row["transition_id"]),
                            ("state_window_id", row["state_window_id"]),
                        ):
                            if key in action_data and str(action_data[key]) != str(expected):
                                errors.append(
                                    f"{row['transition_id']}: action {key} mismatch"
                                )
                        if "t_start" in action_data and "t_end" in action_data:
                            action_timestamp_checks += 1
                            if (
                                abs(float(action_data["t_start"]) - float(row["t"]))
                                > 1e-5
                            ):
                                errors.append(
                                    f"{row['transition_id']}: action t_start mismatch"
                                )
                            expected_end = float(row["t"]) + self.step_seconds
                            if abs(float(action_data["t_end"]) - expected_end) > 1e-5:
                                errors.append(
                                    f"{row['transition_id']}: action t_end mismatch"
                                )
                    telemetry_values.append(float(telemetry[14]))
                    action_values.append(float(command[2]))
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    errors.append(f"{row.get('transition_id', '?')}: {exc}")

            tel_series = np.asarray(telemetry_values)
            action_series = np.asarray(action_values)
            for offset in (-1, 0, 1):
                lo = max(0, -offset)
                hi = min(len(tel_series), len(tel_series) - offset)
                if hi - lo < 2:
                    continue
                indices = np.arange(lo, hi)
                paired[offset][0].extend(tel_series[indices].tolist())
                paired[offset][1].extend(action_series[indices + offset].tolist())

        def correlation(xs: list[float], ys: list[float]) -> float:
            x = np.asarray(xs)
            y = np.asarray(ys)
            if len(x) < 2 or x.std() < 1e-8 or y.std() < 1e-8:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1])

        correlations = {offset: correlation(*paired[offset]) for offset in (-1, 0, 1)}
        result = {
            "correlations": correlations,
            "correlation_look_dx": correlations[0],
            "best_offset": max(correlations, key=correlations.get),
            "n_sessions": int(len(session_ids)),
            "timestamp_checks": timestamp_checks,
            "action_timestamp_checks": action_timestamp_checks,
            "errors": errors[:20],
            "passed": len(errors) == 0 and timestamp_checks > 0,
        }
        if result["passed"]:
            logger.info(
                "Alignment check passed: %d decision frames, %d action timestamps",
                timestamp_checks,
                action_timestamp_checks,
            )
        else:
            logger.error(
                "Alignment check failed with %d errors. First errors: %s",
                len(errors),
                errors[:3],
            )
        return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli_main() -> None:
    """CLI for dataset path verification and alignment checks."""
    import argparse
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Verify DistillationDataset paths and alignment."
    )
    parser.add_argument(
        "--transition_manifest",
        required=True,
        help="Path to Phase 2 transition manifest CSV.",
    )
    parser.add_argument(
        "--window_manifest",
        default=None,
        help="Path to window manifest CSV. Defaults to sibling of transition manifest.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to check.",
    )
    parser.add_argument(
        "--check_paths",
        action="store_true",
        help="Verify all paths in the manifest resolve to existing files.",
    )
    parser.add_argument(
        "--verify_alignment",
        action="store_true",
        help="Run the alignment verification.",
    )
    parser.add_argument(
        "--print_sample_stats",
        action="store_true",
        help="Print statistics of a few sampled observations and actions.",
    )
    args = parser.parse_args()

    if args.check_paths:
        logger.info("=" * 60)
        logger.info("PATH VERIFICATION")
        logger.info("=" * 60)

        import csv as csv_mod

        # Check transition manifest
        trans_path = Path(args.transition_manifest)
        if not trans_path.exists():
            logger.error("Transition manifest NOT FOUND: %s", trans_path)
            sys.exit(1)
        logger.info("✓ Transition manifest exists: %s", trans_path)

        with open(trans_path) as f:
            reader = csv_mod.DictReader(f)
            trans_rows = [r for r in reader if r.get("split", "") == args.split]
        logger.info("  %d transitions in split '%s'", len(trans_rows), args.split)

        # Check action paths
        missing_actions = 0
        for row in trans_rows:
            ap = Path(row.get("action_path", ""))
            if not ap.exists():
                missing_actions += 1
        if missing_actions:
            logger.error(
                "  ✗ %d/%d action paths missing", missing_actions, len(trans_rows)
            )
        else:
            logger.info("  ✓ All %d action paths exist", len(trans_rows))

        # Check window manifest
        win_path = (
            Path(args.window_manifest)
            if args.window_manifest
            else (trans_path.parent / "phase2_window_manifest.csv")
        )
        if not win_path.exists():
            logger.error("Window manifest NOT FOUND: %s", win_path)
            sys.exit(1)
        logger.info("✓ Window manifest exists: %s", win_path)

        with open(win_path) as f:
            reader = csv_mod.DictReader(f)
            win_rows = list(reader)
        logger.info("  %d windows total", len(win_rows))

        # Check tel_npz_path
        has_tel = sum(1 for r in win_rows if r.get("tel_npz_path", "").strip())
        missing_tel = len(win_rows) - has_tel
        if missing_tel > 0:
            logger.warning(
                "  ✗ %d/%d windows have EMPTY tel_npz_path. "
                "Run build_telemetry_npz first!",
                missing_tel,
                len(win_rows),
            )
        else:
            # Check that files actually exist
            missing_files = sum(
                1
                for r in win_rows
                if r.get("tel_npz_path", "").strip()
                and not Path(r["tel_npz_path"]).exists()
            )
            if missing_files:
                logger.error(
                    "  ✗ %d tel_npz files referenced but not found", missing_files
                )
            else:
                logger.info("  ✓ All %d tel_npz_path entries exist", has_tel)

        logger.info("=" * 60)
        logger.info("PATH CHECK COMPLETE")
        logger.info("=" * 60)

    if args.verify_alignment or args.print_sample_stats:
        # Build dataset
        logger.info("Loading DistillationDataset...")
        ds = DistillationDataset(
            transition_manifest_path=args.transition_manifest,
            window_manifest_path=args.window_manifest,
            mode="single",
            split=args.split,
            require_normalized_telemetry=True,
        )

        if args.verify_alignment:
            logger.info("=" * 60)
            logger.info("ALIGNMENT VERIFICATION")
            logger.info("=" * 60)
            result = ds.verify_alignment()
            for k, v in result.items():
                if k != "errors":
                    logger.info("  %s: %s", k, v)
            if result.get("errors"):
                for err in result["errors"][:5]:
                    logger.error("  error: %s", err)

        if args.print_sample_stats:
            logger.info("=" * 60)
            logger.info("SAMPLE STATISTICS")
            logger.info("=" * 60)

            rng = np.random.default_rng(42)
            n = min(20, len(ds))
            indices = rng.choice(len(ds), size=n, replace=False)

            obs_all = []
            act_all = []
            for idx in indices:
                obs, act = ds[int(idx)]
                obs_all.append(obs.numpy())
                act_all.append(act.numpy())

            obs_arr = np.stack(obs_all)  # [N, 45]
            act_arr = np.stack(act_all)  # [N, 8]

            logger.info("obs shape: %s", obs_arr.shape)
            logger.info(
                "obs all-zero check: %s",
                "FAIL — all zeros!" if np.allclose(obs_arr, 0) else "PASS — nonzero",
            )
            logger.info("obs mean: %s", np.mean(obs_arr, axis=0).round(3))
            logger.info("obs std:  %s", np.std(obs_arr, axis=0).round(3))
            logger.info("obs range: [%.3f, %.3f]", obs_arr.min(), obs_arr.max())
            logger.info(
                "obs finite: %s", "PASS" if np.isfinite(obs_arr).all() else "FAIL"
            )

            logger.info("")
            logger.info("action shape: %s", act_arr.shape)
            logger.info(
                "action continuous [:4] range: [%.3f, %.3f]",
                act_arr[:, :4].min(),
                act_arr[:, :4].max(),
            )
            logger.info(
                "action continuous [:4] in [-1,1]: %s",
                "PASS"
                if act_arr[:, :4].min() >= -1.01 and act_arr[:, :4].max() <= 1.01
                else "FAIL",
            )
            logger.info(
                "action binary [4:8] unique values: %s",
                np.unique(act_arr[:, 4:8]).tolist(),
            )
            logger.info(
                "action binary [4:8] are 0/1: %s",
                "PASS"
                if set(np.unique(act_arr[:, 4:8]).tolist()).issubset({0.0, 1.0})
                else "FAIL",
            )

            logger.info("=" * 60)
            logger.info("SAMPLE STATS COMPLETE")
            logger.info("=" * 60)


if __name__ == "__main__":
    _cli_main()
