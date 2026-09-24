from pathlib import Path

from torch.utils.data import DataLoader

from .clip_dataset import ClipDataset
from .collate import clip_collate_fn
from .pk_sampler import PKBatchSampler


def build_train_loader(
    manifest_path: str | Path,
    image_size: tuple[int, int] = (160, 240),
    seq_len: int = 150,
    players_per_batch: int = 64,
    clips_per_player: int = 4,
    batches_per_epoch: int = 1000,
    num_workers: int = 8,
    seed: int = 42,
) -> DataLoader:
    dataset = ClipDataset(
        manifest_path=manifest_path,
        split="train",
        image_size=image_size,
        seq_len=seq_len,
    )

    batch_sampler = PKBatchSampler(
        player_ids=dataset.player_ids_int,
        players_per_batch=players_per_batch,
        clips_per_player=clips_per_player,
        batches_per_epoch=batches_per_epoch,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=clip_collate_fn,
    )


def build_pk_val_loader(
    manifest_path: str | Path,
    image_size: tuple[int, int] = (160, 240),
    seq_len: int = 150,
    players_per_batch: int = 64,
    clips_per_player: int = 4,
    batches_per_epoch: int = 100,
    num_workers: int = 8,
    seed: int = 123,
) -> DataLoader:
    dataset = ClipDataset(
        manifest_path=manifest_path,
        split="val",
        image_size=image_size,
        seq_len=seq_len,
    )

    batch_sampler = PKBatchSampler(
        player_ids=dataset.player_ids_int,
        players_per_batch=players_per_batch,
        clips_per_player=clips_per_player,
        batches_per_epoch=batches_per_epoch,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=clip_collate_fn,
    )


def build_embedding_loader(
    manifest_path: str | Path,
    split: str = "val",
    image_size: tuple[int, int] = (160, 240),
    seq_len: int = 150,
    batch_size: int = 128,
    num_workers: int = 8,
) -> DataLoader:
    dataset = ClipDataset(
        manifest_path=manifest_path,
        split=split,
        image_size=image_size,
        seq_len=seq_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=clip_collate_fn,
    )
