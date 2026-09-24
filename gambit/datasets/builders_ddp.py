from pathlib import Path

from torch.utils.data import DataLoader

from .clip_dataset import ClipDataset
from .collate import clip_collate_fn
from gambit.training.distributed_pk_sampler import DistributedPKBatchSampler


def build_ddp_train_loader(
    manifest_path: str | Path,
    rank: int,
    world_size: int,
    image_size: tuple[int, int] = (160, 240),
    seq_len: int = 150,
    players_per_rank: int = 8,
    clips_per_player: int = 4,
    batches_per_epoch: int = 1000,
    num_workers: int = 8,
    seed: int = 42,
) -> tuple[DataLoader, DistributedPKBatchSampler]:
    dataset = ClipDataset(
        manifest_path=manifest_path,
        split="train",
        image_size=image_size,
        seq_len=seq_len,
    )

    sampler = DistributedPKBatchSampler(
        player_ids=dataset.player_ids_int,
        players_per_rank=players_per_rank,
        clips_per_player=clips_per_player,
        batches_per_epoch=batches_per_epoch,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=clip_collate_fn,
    )

    return loader, sampler

def build_ddp_val_loader(
    manifest_path: str | Path,
    rank: int,
    world_size: int,
    image_size: tuple[int, int] = (160, 240),
    seq_len: int = 150,
    players_per_rank: int = 8,
    clips_per_player: int = 4,
    batches_per_epoch: int = 20,
    num_workers: int = 8,
    seed: int = 42,
) -> tuple[DataLoader, DistributedPKBatchSampler]:
    dataset = ClipDataset(
        manifest_path=manifest_path,
        split="val",
        image_size=image_size,
        seq_len=seq_len,
    )

    sampler = DistributedPKBatchSampler(
        player_ids=dataset.player_ids_int,
        players_per_rank=players_per_rank,
        clips_per_player=clips_per_player,
        batches_per_epoch=batches_per_epoch,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=clip_collate_fn,
    )

    return loader, sampler
