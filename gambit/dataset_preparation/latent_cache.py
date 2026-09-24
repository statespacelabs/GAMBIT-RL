"""Phase 2 encoder latent cache: run frozen encoder over windows, save z_raw/z_norm.

The Phase2WindowDataset loads windowed sub-clips from continuous session
videos/telemetry. The EncoderLatentCacheWriter runs batched inference and
writes one .npz latent file per window.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from gambit.datasets.video_decode import load_video_clip
from gambit.datasets.telemetry_preprocess import (
    _load_fixed_frames,
    _extract_arrays_from_frames,
    _apply_stats,
    TelemetryStats,
    WHERE_DIM,
    VIEW_DIM,
    RHYTHM_EXTRA_DIM,
)
from gambit.datasets.collate import clip_collate_fn
from gambit.models.inquisitor_encoder import InquisitorEncoder
from gambit.models.types import EncoderConfig, ClipBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_latent_npz(
    output_path: str | Path,
    window_id: str,
    player_id: str,
    session_id: str,
    t_start: float,
    t_end: float,
    z_raw: np.ndarray,
    z_norm: np.ndarray,
) -> None:
    """Save one latent cache file.

    The file must be loadable by TransitionDataset and should contain only
    compact arrays and metadata. Do not store video or full telemetry here.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        window_id=np.asarray(window_id),
        player_id=np.asarray(player_id),
        session_id=np.asarray(session_id),
        t_start=np.asarray(t_start, dtype=np.float32),
        t_end=np.asarray(t_end, dtype=np.float32),
        z_raw=z_raw.astype(np.float32),
        z_norm=z_norm.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Phase 2 windowed dataset
# ---------------------------------------------------------------------------

def _slice_frames_by_time(
    tel_json_path: str | Path,
    t_start: float,
    t_end: float,
    seq_len: int,
    fps: float,
) -> tuple[dict, list[dict]]:
    """Load JSON and slice frames to [t_start, t_end], pad/truncate to seq_len."""
    with open(tel_json_path, "r", encoding="utf-8") as f:
        chunk = json.load(f)

    all_frames = chunk.get("frames", [])
    if not all_frames:
        raise ValueError(f"No frames in {tel_json_path}")

    # Filter frames in the time window
    sliced = [
        fr for fr in all_frames
        if t_start <= float(fr.get("time", -1.0)) < t_end
    ]

    # Pad or truncate to exactly seq_len
    if len(sliced) > seq_len:
        sliced = sliced[:seq_len]
    elif len(sliced) < seq_len:
        if len(sliced) == 0:
            # No frames in window — use the last frame of the chunk as padding
            sliced = [all_frames[-1]]
        last = sliced[-1]
        last_time = float(last.get("time", t_start))
        for j in range(seq_len - len(sliced)):
            new_frame = dict(last)
            new_frame["actions"] = []
            new_frame["time"] = last_time + (j + 1) / fps
            sliced.append(new_frame)
        sliced = sliced[:seq_len]

    # Override chunk metadata for this window
    window_chunk = dict(chunk)
    window_chunk["start_time"] = t_start
    window_chunk["end_time"] = t_end

    return window_chunk, sliced


class Phase2WindowDataset(Dataset):
    """Map-style dataset for Phase 2 encoder inference windows.

    For each window, loads video via start_frame seeking and telemetry
    via time-based frame slicing from the source JSON.
    """

    def __init__(
        self,
        window_manifest_df: pd.DataFrame,
        image_size: tuple[int, int] = (160, 240),
        seq_len: int = 150,
        fps: float = 30.0,
        action_vocab_size: int = 14,
        imagenet_normalize: bool = True,
        telemetry_stats: TelemetryStats | str | Path | None = None,
        shoot_action_type: int = 0,
        reload_action_type: int = 5,
    ):
        self.df = window_manifest_df.reset_index(drop=True)
        self.image_size = image_size
        self.seq_len = seq_len
        self.fps = fps
        self.action_vocab_size = action_vocab_size
        self.imagenet_normalize = imagenet_normalize
        self.shoot_action_type = shoot_action_type
        self.reload_action_type = reload_action_type
        self.clip_seconds = seq_len / fps

        if isinstance(telemetry_stats, (str, Path)):
            self.telemetry_stats = TelemetryStats.load(telemetry_stats)
        else:
            self.telemetry_stats = telemetry_stats

        # Build a simple player→int mapping
        players = sorted(self.df["player_id"].astype(str).unique().tolist())
        self.player_to_int = {pid: i for i, pid in enumerate(players)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        t_start = float(row["t_start"])
        t_end = float(row["t_end"])
        player_id_int = self.player_to_int.get(str(row["player_id"]), 0)

        try:
            # ── Video ──
            video = load_video_clip(
                row["video_path"],
                image_size=self.image_size,
                seq_len=self.seq_len,
                start_frame=int(row["start_frame"]),
                imagenet_normalize=self.imagenet_normalize,
            )

            # ── Telemetry ──
            chunk, frames = _slice_frames_by_time(
                row["tel_json_path"], t_start, t_end,
                self.seq_len, self.fps,
            )

            where_tel, view_tel, rhythm_tel = _extract_arrays_from_frames(
                chunk=chunk,
                frames=frames,
                action_vocab_size=self.action_vocab_size,
                seq_len=self.seq_len,
                clip_seconds=self.clip_seconds,
                fps=self.fps,
                shoot_action_type=self.shoot_action_type,
                reload_action_type=self.reload_action_type,
                normalize_time_since=True,
                strict_action_vocab=False,
            )

            where_tel, view_tel, rhythm_tel = _apply_stats(
                where_tel, view_tel, rhythm_tel, self.telemetry_stats,
            )

            where_t = torch.from_numpy(where_tel).float()
            view_t = torch.from_numpy(view_tel).float()
            rhythm_t = torch.from_numpy(rhythm_tel).float()
            is_valid = True
        except Exception as e:
            logger.warning("Failed to load sample idx %d (window %s): %s", idx, row["window_id"], str(e))
            video = torch.zeros(self.seq_len, 3, self.image_size[0], self.image_size[1])
            where_t = torch.zeros(self.seq_len, 10)
            view_t = torch.zeros(self.seq_len, 13)
            rhythm_t = torch.zeros(self.seq_len, self.action_vocab_size + 8)
            is_valid = False

        return {
            "video": video,
            "where_tel": where_t,
            "view_tel": view_t,
            "rhythm_tel": rhythm_t,
            "player_id": torch.tensor(player_id_int, dtype=torch.long),
            "clip_id": str(row.get("source_clip_id", "")),
            "session_id": str(row["session_id"]),
            # Extra metadata for cache writing
            "window_id": str(row["window_id"]),
            "t_start": t_start,
            "t_end": t_end,
            "player_id_str": str(row["player_id"]),
            "split": str(row.get("split", "train")),
            "is_valid": is_valid,
        }


# ---------------------------------------------------------------------------
# Encoder latent cache writer
# ---------------------------------------------------------------------------

class EncoderLatentCacheWriter:
    """Runs a frozen InquisitorEncoder over Phase 2 windows and saves z_raw/z_norm.

    This class:
        1. Loads the trained encoder checkpoint.
        2. Builds a DataLoader over the Phase 2 window manifest.
        3. Runs the encoder in eval/no_grad mode.
        4. Saves one .npz latent file per window.
        5. Returns a DataFrame mapping window_id → latent file path.

    The encoder is not trained here.
    """

    def __init__(
        self,
        encoder_checkpoint_path: str | Path,
        window_manifest_path: str | Path,
        output_latent_dir: str | Path,
        image_size: tuple[int, int] = (160, 240),
        seq_len: int = 150,
        fps: float = 30.0,
        action_vocab_size: int = 14,
        batch_size: int = 128,
        num_workers: int = 8,
        device: str = "cuda",
        amp: bool = True,
        telemetry_stats_path: str | Path | None = None,
        encoder_config: EncoderConfig | None = None,
    ):
        """Initialize the latent cache writer.

        Loads paths, runtime parameters, and encoder settings. Actual checkpoint
        loading happens inside load_encoder().
        """
        self.encoder_checkpoint_path = Path(encoder_checkpoint_path)
        self.window_manifest_path = Path(window_manifest_path)
        self.output_latent_dir = Path(output_latent_dir)
        self.image_size = image_size
        self.seq_len = seq_len
        self.fps = fps
        self.action_vocab_size = action_vocab_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(device)
        self.amp = amp
        self.telemetry_stats_path = telemetry_stats_path
        self.encoder_config = encoder_config or EncoderConfig()

    def load_encoder(self) -> InquisitorEncoder:
        """Load the trained InquisitorEncoder checkpoint.

        The returned model is in eval mode. All parameters have
        requires_grad=False.

        Returns:
            Frozen InquisitorEncoder ready for inference.
        """
        ckpt = torch.load(
            self.encoder_checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        # Prefer EMA weights if available (smoothed model, better quality)
        if "ema" in ckpt:
            state_dict = ckpt["ema"]
            logger.info("Using EMA weights from checkpoint")
        else:
            state_dict = ckpt.get("model", ckpt)

        # Handle DDP (module.) and torch.compile (_orig_mod.) prefixes
        cleaned = {}
        for k, v in state_dict.items():
            clean_k = k.removeprefix("module.").removeprefix("_orig_mod.")
            cleaned[clean_k] = v

        # Auto-detect transformer model dimensions from state_dict keys/shapes
        try:
            linear1_key = None
            for k in cleaned.keys():
                if "temporal_transformer.transformer.layers.0.linear1.weight" in k:
                    linear1_key = k
                    break
            if linear1_key is not None:
                dim_ff, d_mod = cleaned[linear1_key].shape
                self.encoder_config.transformer.dim_feedforward = dim_ff
                self.encoder_config.transformer.d_model = d_mod
                logger.info(f"Auto-detected model dims from checkpoint: d_model={d_mod}, dim_feedforward={dim_ff}")
            
            layer_ids = set()
            for k in cleaned.keys():
                if "temporal_transformer.transformer.layers." in k:
                    parts = k.split("temporal_transformer.transformer.layers.")[1].split(".")
                    layer_ids.add(int(parts[0]))
            if layer_ids:
                num_layers = max(layer_ids) + 1
                self.encoder_config.transformer.num_layers = num_layers
                logger.info(f"Auto-detected model layers: num_layers={num_layers}")
        except Exception as e:
            logger.warning(f"Failed to auto-detect model architecture from checkpoint: {e}. Using default config.")

        model = InquisitorEncoder(self.encoder_config)
        model.load_state_dict(cleaned)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(self.device)

        logger.info("Loaded encoder from %s", self.encoder_checkpoint_path)
        return model

    def run(self) -> pd.DataFrame:
        """Cache latents for all windows in the manifest.

        For every window, saves z_raw, z_norm, and metadata to a .npz file.

        Returns:
            DataFrame mapping window_id to latent file path.
        """
        manifest_df = pd.read_csv(self.window_manifest_path)
        logger.info("Window manifest: %d windows", len(manifest_df))

        dataset = Phase2WindowDataset(
            window_manifest_df=manifest_df,
            image_size=self.image_size,
            seq_len=self.seq_len,
            fps=self.fps,
            action_vocab_size=self.action_vocab_size,
            telemetry_stats=self.telemetry_stats_path,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=self._collate,
            drop_last=False,
        )

        encoder = self.load_encoder()
        index_rows: list[dict] = []

        for batch_idx, batch_data in enumerate(loader):
            clip_batch = batch_data["clip_batch"].to(self.device, non_blocking=True)
            meta = batch_data["meta"]

            with torch.no_grad():
                if self.amp and self.device.type == "cuda":
                    with torch.autocast(device_type="cuda"):
                        out = encoder(clip_batch)
                else:
                    out = encoder(clip_batch)

            z_raw_np = out.z_raw.cpu().numpy()
            z_norm_np = out.z_norm.cpu().numpy()

            for i in range(z_raw_np.shape[0]):
                if not meta["is_valid"][i]:
                    logger.warning("Skipping invalid window %s from cache saving", meta["window_id"][i])
                    continue
                wid = meta["window_id"][i]
                split = meta["split"][i]
                latent_path = self.output_latent_dir / split / f"{wid}.npz"

                save_latent_npz(
                    output_path=latent_path,
                    window_id=wid,
                    player_id=meta["player_id_str"][i],
                    session_id=meta["session_id"][i],
                    t_start=meta["t_start"][i],
                    t_end=meta["t_end"][i],
                    z_raw=z_raw_np[i],
                    z_norm=z_norm_np[i],
                )

                index_rows.append({
                    "window_id": wid,
                    "latent_path": str(latent_path),
                    "split": split,
                })

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    "Cached %d / %d batches",
                    batch_idx + 1,
                    len(loader),
                )

        index_df = pd.DataFrame(index_rows)
        logger.info("Cached %d latents", len(index_df))
        return index_df

    @staticmethod
    def _collate(samples: list[dict]) -> dict:
        """Custom collate that separates ClipBatch tensors from metadata."""
        clip_batch = clip_collate_fn(samples)

        # Collect window-level metadata that isn't part of ClipBatch
        meta = {
            "window_id": [s["window_id"] for s in samples],
            "t_start": [s["t_start"] for s in samples],
            "t_end": [s["t_end"] for s in samples],
            "player_id_str": [s["player_id_str"] for s in samples],
            "session_id": [s["session_id"] for s in samples],
            "split": [
                # Look up split from the original manifest row
                # (passed through __getitem__ would be cleaner, adding it)
                s.get("split", "train") for s in samples
            ],
            "is_valid": [s.get("is_valid", True) for s in samples],
        }
        return {"clip_batch": clip_batch, "meta": meta}
