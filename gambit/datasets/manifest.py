import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "clip_id",
    "player_id",
    "session_id",
    "video_path",
    "tel_json_path",
    "tel_npz_path",
    "split",
}


@dataclass(frozen=True)
class ManifestInfo:
    df: pd.DataFrame
    player_to_int: dict[str, int]
    int_to_player: dict[int, str]


def _resolve_manifest_paths(df: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    base = manifest_path.parent
    df = df.copy()
    for col in ("video_path", "tel_json_path", "tel_npz_path"):
        if col in df.columns:
            df[col] = [
                str((base / value).resolve()) if not Path(str(value)).is_absolute() else str(value)
                for value in df[col].astype(str).tolist()
            ]
    return df


def validate_session_split(df: pd.DataFrame) -> None:
    """Reject manifests where one continuous session appears in multiple splits."""
    if "session_id" not in df.columns or "split" not in df.columns:
        return

    split_counts = df.groupby(["player_id", "session_id"])["split"].nunique()
    leaking_sessions = split_counts[split_counts > 1].index.tolist()
    if leaking_sessions:
        preview = leaking_sessions[:10]
        raise ValueError(
            "Session leakage across splits detected. "
            f"Example (player_id, session_id): {preview}"
        )


def load_manifest(
    manifest_path: str | Path,
    split: str | None = None,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
    check_session_leakage: bool = True,
    resolve_paths: bool = True,
) -> ManifestInfo:
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    if check_session_leakage:
        validate_session_split(df)

    if resolve_paths:
        df = _resolve_manifest_paths(df, manifest_path)

    players = sorted(df["player_id"].astype(str).unique().tolist())
    player_to_int = {pid: i for i, pid in enumerate(players)}
    int_to_player = {i: pid for pid, i in player_to_int.items()}

    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"No rows found in manifest for split={split}")

    return ManifestInfo(
        df=df,
        player_to_int=player_to_int,
        int_to_player=int_to_player,
    )


def validate_paths(df: pd.DataFrame, check_video: bool = True, check_npz: bool = True) -> None:
    for _, row in df.iterrows():
        if check_video and not Path(row["video_path"]).exists():
            raise FileNotFoundError(f"Missing video file: {row['video_path']}")

        if check_npz and not Path(row["tel_npz_path"]).exists():
            raise FileNotFoundError(f"Missing telemetry npz file: {row['tel_npz_path']}")


def build_manifest_from_analytics_dir(
    analytics_dir: str | Path,
    output_manifest_path: str | Path,
    split: str = "train",
    player_id: str | None = None,
    session_id: str | None = None,
    npz_dir: str | Path | None = None,
    relative_paths: bool = True,
) -> pd.DataFrame:
    """Create a manifest from a directory of matching ID_XXXX.json/ID_XXXX.webm files."""
    analytics_dir = Path(analytics_dir).resolve()
    output_manifest_path = Path(output_manifest_path).resolve()
    npz_dir = Path(npz_dir).resolve() if npz_dir is not None else analytics_dir / "telemetry_npz"

    if not analytics_dir.exists():
        raise FileNotFoundError(f"Analytics directory not found: {analytics_dir}")

    rows: list[dict[str, str]] = []
    for json_path in sorted(analytics_dir.glob("*.json")):
        video_path = json_path.with_suffix(".webm")
        if not video_path.exists():
            raise FileNotFoundError(f"Missing matching video for {json_path}: {video_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            chunk = json.load(f)

        clip_id = str(chunk.get("chunk_id", json_path.stem))
        row_player_id = player_id or str(chunk.get("player_id", analytics_dir.name))
        row_session_id = session_id or str(chunk.get("session_id", analytics_dir.name))
        tel_npz_path = npz_dir / f"{json_path.stem}.npz"

        rows.append(
            {
                "clip_id": clip_id,
                "player_id": row_player_id,
                "session_id": row_session_id,
                "video_path": str(video_path),
                "tel_json_path": str(json_path),
                "tel_npz_path": str(tel_npz_path),
                "split": split,
            }
        )

    if not rows:
        raise ValueError(f"No JSON chunks found in {analytics_dir}")

    df = pd.DataFrame(rows)

    if relative_paths:
        output_dir = output_manifest_path.parent
        for col in ("video_path", "tel_json_path", "tel_npz_path"):
            df[col] = [str(Path(value).resolve().relative_to(output_dir)) if _is_relative_to(Path(value).resolve(), output_dir) else str(value) for value in df[col]]

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_manifest_path, index=False)
    return df


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
