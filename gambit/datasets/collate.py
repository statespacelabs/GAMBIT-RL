import torch

from gambit.models.types import ClipBatch


def clip_collate_fn(samples: list[dict]) -> ClipBatch:
    video = torch.stack([s["video"] for s in samples], dim=0)
    where_tel = torch.stack([s["where_tel"] for s in samples], dim=0)
    view_tel = torch.stack([s["view_tel"] for s in samples], dim=0)
    rhythm_tel = torch.stack([s["rhythm_tel"] for s in samples], dim=0)
    player_ids = torch.stack([s["player_id"] for s in samples], dim=0)

    clip_ids = [s["clip_id"] for s in samples] if "clip_id" in samples[0] else None
    session_ids = [s["session_id"] for s in samples] if "session_id" in samples[0] else None

    return ClipBatch(
        video=video,
        where_tel=where_tel,
        view_tel=view_tel,
        rhythm_tel=rhythm_tel,
        player_ids=player_ids,
        clip_ids=clip_ids,
        session_ids=session_ids,
    )
