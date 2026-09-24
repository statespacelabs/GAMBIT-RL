import random
from collections import defaultdict
from typing import Iterator

from torch.utils.data import Sampler


class PKBatchSampler(Sampler[list[int]]):
    """
    Produces batches with:
      P distinct players
      K clips per player

    Batch size = P * K
    """

    def __init__(
        self,
        player_ids: list[int],
        players_per_batch: int,
        clips_per_player: int,
        batches_per_epoch: int,
        seed: int = 0,
    ):
        if players_per_batch <= 0:
            raise ValueError("players_per_batch must be positive")

        if clips_per_player < 2:
            raise ValueError("clips_per_player must be >= 2 for SupCon positives")

        self.player_ids = [int(x) for x in player_ids]
        self.P = players_per_batch
        self.K = clips_per_player
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0

        self.player_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, pid in enumerate(self.player_ids):
            self.player_to_indices[pid].append(idx)

        self.players = [
            pid for pid, idxs in self.player_to_indices.items()
            if len(idxs) >= self.K
        ]

        if len(self.players) < self.P:
            raise ValueError(
                f"Need at least P={self.P} players with K={self.K} clips each. "
                f"Found only {len(self.players)} valid players."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)

        for _ in range(self.batches_per_epoch):
            selected_players = rng.sample(self.players, self.P)

            indices: list[int] = []
            for pid in selected_players:
                candidates = self.player_to_indices[pid]
                chosen = rng.sample(candidates, self.K)
                indices.extend(chosen)

            rng.shuffle(indices)
            yield indices

    def __len__(self) -> int:
        return self.batches_per_epoch
