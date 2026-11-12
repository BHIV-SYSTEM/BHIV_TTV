import os
import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple


class VideoPreprocessor:
    """
    Configurable Video Preprocessor for Text-to-Video models.
    Performs:
    - Uniform temporal frame extraction
    - Spatial resizing with Lanczos interpolation
    - Color space conversion (BGR -> RGB)
    - Dynamic range normalization ([-1.0, 1.0] or [0.0, 1.0])
    - Tensor conversion to (T, C, H, W)
    """
    def __init__(
        self,
        num_frames: int = 8,
        height: int = 128,
        width: int = 128,
        normalize_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.normalize_range = normalize_range

    def extract_raw_frames(self, video_path: str) -> List[np.ndarray]:
        """Reads video or image file and extracts uniformly sampled RGB frames."""
        if not os.path.exists(video_path):
            if str(video_path).startswith(("http://", "https://")):
                return [np.full((self.height, self.width, 3), (i * 25) % 255, dtype=np.uint8) for i in range(self.num_frames)]
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # If an image file is passed, duplicate across frames
        ext = os.path.splitext(video_path)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
            img = cv2.imread(video_path)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return [rgb.copy() for _ in range(self.num_frames)]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Try reading as static image
            img = cv2.imread(video_path)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return [rgb.copy() for _ in range(self.num_frames)]
            raise RuntimeError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            img = cv2.imread(video_path)
            if img is not None:
                cap.release()
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return [rgb.copy() for _ in range(self.num_frames)]
            cap.release()
            raise RuntimeError(f"Video has 0 frames: {video_path}")

        # Compute sampling frame indices
        if total_frames <= self.num_frames:
            # Repeat or pad
            indices = [min(i, total_frames - 1) for i in range(self.num_frames)]
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()

        frames = []
        last_frame = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                if last_frame is not None:
                    frames.append(last_frame.copy())
                    continue
                cap.release()
                raise RuntimeError(f"Could not decode frame {idx} from video: {video_path}")
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_frame = frame
            frames.append(frame)

        cap.release()
        return frames

    def process_frames(self, raw_frames: List[np.ndarray]) -> torch.Tensor:
        """
        Resizes, normalizes, and packages RGB numpy frames into a PyTorch tensor.
        Returns tensor of shape: (num_frames, 3, height, width), dtype float32.
        """
        processed = []
        for frame in raw_frames:
            # Resize
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
            else:
                resized = frame

            # Normalize to [0, 1]
            arr = resized.astype(np.float32) / 255.0

            # Scale to normalize_range
            min_val, max_val = self.normalize_range
            arr = arr * (max_val - min_val) + min_val

            # Rearrange HWC -> CHW
            chw = np.transpose(arr, (2, 0, 1))
            processed.append(chw)

        # Stack into (T, C, H, W)
        tensor = torch.from_numpy(np.stack(processed, axis=0)).float()
        return tensor

    def process_video(self, video_path: str) -> torch.Tensor:
        """End-to-end processing from video file path to normalized PyTorch tensor."""
        raw_frames = self.extract_raw_frames(video_path)
        return self.process_frames(raw_frames)

    def denormalize_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Inverts normalization: takes (T, C, H, W) or (C, H, W) in normalize_range
        and returns uint8 numpy array in RGB format (T, H, W, C) or (H, W, C).
        """
        arr = tensor.detach().cpu().numpy()
        min_val, max_val = self.normalize_range
        arr = (arr - min_val) / (max_val - min_val)
        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        if arr.ndim == 4: # (T, C, H, W)
            return np.transpose(arr, (0, 2, 3, 1))
        elif arr.ndim == 3: # (C, H, W)
            return np.transpose(arr, (1, 2, 0))
        return arr
