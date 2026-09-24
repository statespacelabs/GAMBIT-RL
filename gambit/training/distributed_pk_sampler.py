import random
from collections import defaultdict
from typing import Iterator

from torch.utils.data import Sampler


class DistributedPKBatchSampler(Sampler[list[int]]):
    """
    DDP-aware P×K sampler.

    Each rank receives:
      players_per_rank players × clips_per_player clips

    Across all ranks:
      players_per_rank * world_size players × clips_per_player clips
    """

    def __init__(
        self,
        player_ids: list[int],
        players_per_rank: int,
        clips_per_player: int,
        batches_per_epoch: int,
        rank: int,
        world_size: int,
        seed: int = 0,
    ):
        if players_per_rank <= 0:
            raise ValueError("players_per_rank must be positive")

        if clips_per_player < 2:
            raise ValueError("clips_per_player must be >= 2 for SupCon positives")

        self.player_ids = [int(x) for x in player_ids]
        self.players_per_rank = int(players_per_rank)
        self.clips_per_player = int(clips_per_player)
        self.batches_per_epoch = int(batches_per_epoch)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0

        self.global_players_per_batch = self.players_per_rank * self.world_size

        self.player_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, pid in enumerate(self.player_ids):
            self.player_to_indices[pid].append(idx)

        self.valid_players = [
            pid
            for pid, idxs in self.player_to_indices.items()
            if len(idxs) >= self.clips_per_player
        ]

        if len(self.valid_players) < self.global_players_per_batch:
            raise ValueError(
                f"Need at least {self.global_players_per_batch} valid players "
                f"with K={self.clips_per_player} clips each, "
                f"found {len(self.valid_players)}."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        # This RNG must remain perfectly synced across all DDP ranks
        global_rng = random.Random(self.seed + self.epoch)
        # This RNG can advance independently per rank without breaking global selection
        local_rng = random.Random(self.seed + self.epoch + self.rank + 1000)

        for step in range(self.batches_per_epoch):
            selected_global_players = global_rng.sample(
                self.valid_players,
                self.global_players_per_batch,
            )

            start = self.rank * self.players_per_rank
            end = start + self.players_per_rank
            selected_local_players = selected_global_players[start:end]

            local_indices: list[int] = []
            for pid in selected_local_players:
                candidates = self.player_to_indices[pid]
                chosen = local_rng.sample(candidates, self.clips_per_player)
                local_indices.extend(chosen)

            local_rng.shuffle(local_indices)
            yield local_indices

    def __len__(self) -> int:
        return self.batches_per_epoch
