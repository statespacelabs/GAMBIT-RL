"""Online observation normalizer for PPO rollouts.

Replicates EXACTLY the partial normalization applied to the offline Phase 3
telemetry NPZs (`gambit.datasets.telemetry_preprocess._apply_stats`), so the live
Unity observation distribution matches what the encoder/policy were distilled on:

  where  (obs[0:10])  : (x - mean) / std on all 10 dims
  view   (obs[10:23]) : dims 0:3 (sin/cos pitch/yaw -> obs[10:14]) stay RAW;
                        dims 4:13 (-> obs[14:23]) normalized
  rhythm (obs[23:45]) : RAW (never normalized offline)

No epsilon is added to std (plain division), matching `_apply_stats` exactly.
This is the single normalization point — the encoder does NOT normalize — so
applying this once in the runner avoids double normalization.

Phase 3F addition: configurable post-normalization clipping to bound observations
into [-clip_value, clip_value]. Multi-area arenas have spatial offsets (500m per
area) that produce huge raw position values for where_tel[0:3]. Without clipping
these dominate the input and destabilize PPO.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from gambit.datasets.telemetry_preprocess import TelemetryStats

logger = logging.getLogger(__name__)

OBS_DIM = 45
WHERE = slice(0, 10)
VIEW = slice(10, 23)
RHYTHM = slice(23, 45)
VIEW_CONT_OBS = slice(14, 23)
VIEW_CONT_SRC = slice(4, 13)


class ObservationNormalizer:
    """Applies the offline `_apply_stats` transform to a raw 45-dim Unity obs."""

    def __init__(self, stats_path: str | Path, clip_value: float = 0.0):
        if not stats_path or not Path(stats_path).exists():
            raise FileNotFoundError(
                f"telemetry_stats not found for online obs normalization: {stats_path!r}. "
                "PPO requires online normalization — refusing to run with raw observations."
            )
        stats = TelemetryStats.load(stats_path)

        mean = np.zeros(OBS_DIM, dtype=np.float32)
        scale = np.ones(OBS_DIM, dtype=np.float32)
        mean[WHERE] = stats.where_mean.astype(np.float32)
        scale[WHERE] = stats.where_std.astype(np.float32)
        mean[VIEW_CONT_OBS] = stats.view_mean[VIEW_CONT_SRC].astype(np.float32)
        scale[VIEW_CONT_OBS] = stats.view_std[VIEW_CONT_SRC].astype(np.float32)

        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError(f"telemetry_stats contains NaN/Inf: {stats_path}")
        if np.any(scale == 0.0):
            bad = np.where(scale == 0.0)[0].tolist()
            raise ValueError(f"telemetry_stats has zero std at normalized obs dims {bad}")

        self.stats_path = str(stats_path)
        self.mean = mean
        self.scale = scale
        self.clip_value = float(clip_value) if clip_value and clip_value > 0 else 0.0

        # Running stats for diagnostic logging
        self._call_count = 0
        self._clip_count = 0
        self._total_elements = 0

        if self.clip_value > 0:
            logger.info(
                "ObservationNormalizer: clip_value=%.1f (post-normalization)",
                self.clip_value,
            )

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize a raw obs array of shape [..., 45]. Fails hard on bad input."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape[-1] != OBS_DIM:
            raise ValueError(f"obs last dim {obs.shape[-1]} != {OBS_DIM}")
        if not np.isfinite(obs).all():
            raise ValueError("raw observation contains NaN/Inf")

        out = (obs - self.mean) / self.scale

        if self.clip_value > 0:
            clipped_mask = np.abs(out) > self.clip_value
            n_clipped = int(clipped_mask.sum())
            self._clip_count += n_clipped
            self._total_elements += out.size
            out = np.clip(out, -self.clip_value, self.clip_value)

        if not np.isfinite(out).all():
            raise ValueError("normalized observation contains NaN/Inf")

        self._call_count += 1
        return out.astype(np.float32)

    def get_stats_and_reset(self) -> dict[str, float]:
        """Return clip statistics since last reset, then reset counters."""
        clip_rate = (
            self._clip_count / max(self._total_elements, 1)
            if self._total_elements > 0 else 0.0
        )
        stats = {
            "obs_norm/calls": self._call_count,
            "obs_norm/clip_rate": clip_rate,
            "obs_norm/total_clipped": self._clip_count,
        }
        self._call_count = 0
        self._clip_count = 0
        self._total_elements = 0
        return stats
