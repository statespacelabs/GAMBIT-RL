from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .manifest import load_manifest
from .video_decode import load_video_clip


class ClipDataset(Dataset):
    """
    Map-style dataset.

    __getitem__ returns:
      video:      [T, 3, H, W]
      where_tel:  [T, 10]
      view_tel:   [T, 13]
      rhythm_tel: [T, A+8]
      player_id:  scalar long
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_size: tuple[int, int] = (160, 240),
        seq_len: int = 150,
        imagenet_normalize: bool = True,
    ):
        info = load_manifest(manifest_path, split=split)

        self.df = info.df
        self.player_to_int = info.player_to_int
        self.int_to_player = info.int_to_player

        self.image_size = image_size
        self.seq_len = seq_len
        self.imagenet_normalize = imagenet_normalize

        self.player_ids_int = [
            self.player_to_int[str(pid)]
            for pid in self.df["player_id"].tolist()
        ]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        video = load_video_clip(
            row["video_path"],
            image_size=self.image_size,
            seq_len=self.seq_len,
            imagenet_normalize=self.imagenet_normalize,
        )

        tel_path = Path(row["tel_npz_path"])
        if not tel_path.exists():
            raise FileNotFoundError(f"Telemetry npz not found: {tel_path}")

        # Adding context manager for safety against file handle leaks
        with np.load(tel_path) as tel:
            if "normalized" in tel and not bool(tel["normalized"]):
                raise ValueError(f"Telemetry file is not normalized: {tel_path}")
            where_tel_np = tel["where_tel"]
            view_tel_np = tel["view_tel"]
            rhythm_tel_np = tel["rhythm_tel"]

        where_tel = torch.from_numpy(where_tel_np).float()
        view_tel = torch.from_numpy(view_tel_np).float()
        rhythm_tel = torch.from_numpy(rhythm_tel_np).float()

        if where_tel.shape != (self.seq_len, 10):
            raise ValueError(f"Bad where_tel shape for {tel_path}: {where_tel.shape}")

        if view_tel.shape != (self.seq_len, 13):
            raise ValueError(f"Bad view_tel shape for {tel_path}: {view_tel.shape}")

        if rhythm_tel.shape[0] != self.seq_len:
            raise ValueError(f"Bad rhythm_tel shape for {tel_path}: {rhythm_tel.shape}")

        return {
            "video": video,
            "where_tel": where_tel,
            "view_tel": view_tel,
            "rhythm_tel": rhythm_tel,
            "player_id": torch.tensor(self.player_ids_int[idx], dtype=torch.long),
            "clip_id": str(row["clip_id"]),
            "session_id": str(row["session_id"]),
        }
