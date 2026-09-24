from pathlib import Path

import cv2
import torch
import torch.nn.functional as F


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_video_clip(
    path: str | Path,
    image_size: tuple[int, int] = (160, 240),
    seq_len: int = 150,
    start_frame: int = 0,
    normalize_01: bool = True,
    imagenet_normalize: bool = True,
) -> torch.Tensor:
    """
    Load a video clip as a tensor of frames.

    Args:
        path: Path to video file.
        image_size: Target (H, W) for spatial resizing.
        seq_len: Number of frames to return.
        start_frame: Frame index to start reading from. Uses
            cap.set(CAP_PROP_POS_FRAMES) for seeking. For compressed
            codecs this may be slightly inaccurate on non-keyframes;
            acceptable for first implementation.
        normalize_01: Scale pixel values to [0, 1].
        imagenet_normalize: Apply ImageNet mean/std normalization.

    Returns:
      video: [T, 3, H, W], float32
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[torch.Tensor] = []

    while len(frames) < seq_len:
        ok, frame = cap.read()
        if not ok:
            break

        # OpenCV: BGR HWC uint8 → RGB CHW uint8
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
        frames.append(tensor)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Could not decode any frames from: {path}")

    video = torch.stack(frames, dim=0)  # [T, 3, H, W]

    # Fixed length.
    if video.shape[0] > seq_len:
        video = video[:seq_len]
    elif video.shape[0] < seq_len:
        pad_n = seq_len - video.shape[0]
        pad = video[-1:].repeat(pad_n, 1, 1, 1)
        video = torch.cat([video, pad], dim=0)

    video = video.float()

    if normalize_01:
        video = video / 255.0

    target_h, target_w = image_size
    if video.shape[-2:] != (target_h, target_w):
        video = F.interpolate(
            video,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

    if imagenet_normalize:
        mean = IMAGENET_MEAN.to(video.device, dtype=video.dtype)
        std = IMAGENET_STD.to(video.device, dtype=video.dtype)
        video = (video - mean) / std

    return video
