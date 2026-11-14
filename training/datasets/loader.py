import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
import torch
from torch.utils.data import Dataset, DataLoader

from training.preprocessing.video_preprocessor import VideoPreprocessor
from training.preprocessing.text_preprocessor import TextPreprocessor

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


class TextToVideoDataset(Dataset):
    """
    PyTorch Dataset for Text-to-Video training.
    Loads video files from Phase 4 JSONL manifests, extracts normalized frame sequences,
    and tokenizes prompt text into tensor representations.
    Excludes invalid, duplicate, or restricted license records.
    """
    def __init__(
        self,
        manifest_path: str,
        split: Optional[str] = None,
        num_frames: int = 8,
        height: int = 128,
        width: int = 128,
        max_prompt_length: int = 32,
        vocab_size: int = 10000,
        transform: Optional[Callable] = None
    ):
        p = Path(manifest_path)
        if not p.is_absolute() and not p.exists():
            p = WORKSPACE_ROOT / manifest_path

        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        self.manifest_path = p
        self.samples: List[Dict[str, Any]] = []

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # License & eligibility gate
                if record.get("training_eligible") is False:
                    continue
                if record.get("permission_state") == "RESTRICTED":
                    continue
                if record.get("validation_status") in ("INVALID", "DUPLICATE"):
                    continue

                # Split filter if specified
                if split:
                    rec_split = str(record.get("split") or record.get("dataset_split") or "train").lower()
                    if rec_split != split.lower():
                        continue

                self.samples.append(record)

        self.video_preprocessor = VideoPreprocessor(
            num_frames=num_frames,
            height=height,
            width=width
        )
        self.text_preprocessor = TextPreprocessor(max_length=max_prompt_length, vocab_size=vocab_size)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        raw_video_path = sample.get("video_path") or sample.get("video_url") or sample.get("source_url") or sample.get("storage_path") or ""
        prompt = sample.get("caption") or sample.get("prompt") or sample.get("original_caption") or sample.get("original_prompt") or ""

        # Resolve path relative to workspace if necessary
        video_path = raw_video_path
        if not os.path.isabs(video_path) and not str(video_path).startswith(("http://", "https://")):
            cand = WORKSPACE_ROOT / video_path
            if cand.exists():
                video_path = str(cand)

        # 1. Process video frames -> Tensor shape: (T, C, H, W) normalized to [-1, 1]
        frames_tensor = self.video_preprocessor.process_video(video_path)
        if self.transform is not None:
            frames_tensor = self.transform(frames_tensor)

        # 2. Process text prompt -> Tensor shape: (L,) int64 tokens
        tokens_tensor = self.text_preprocessor.tokenize_to_tensor(prompt)

        return {
            "id": sample.get("id") or sample.get("media_id", f"sample_{idx}"),
            "prompt": prompt,
            "prompt_tokens": tokens_tensor,
            "frames": frames_tensor,  # Shape: (T, C, H, W)
            "duration": float(sample.get("duration", 0.0)),
            "quality_score": torch.tensor(float(sample.get("quality_score", 1.0)), dtype=torch.float32)
        }


def collate_ttv_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collator for Text-to-Video batching."""
    frames = torch.stack([item["frames"] for item in batch], dim=0) # (B, T, C, H, W)
    tokens = torch.stack([item["prompt_tokens"] for item in batch], dim=0) # (B, L)
    quality_scores = torch.stack([item["quality_score"] for item in batch], dim=0)
    prompts = [item["prompt"] for item in batch]
    ids = [item["id"] for item in batch]

    return {
        "ids": ids,
        "prompts": prompts,
        "prompt_tokens": tokens,
        "frames": frames,
        "quality_scores": quality_scores
    }


def create_dataloader(
    manifest_path: str,
    split: Optional[str] = None,
    batch_size: int = 1,
    num_frames: int = 8,
    height: int = 128,
    width: int = 128,
    max_prompt_length: int = 32,
    vocab_size: int = 10000,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Factory creating an optimized DataLoader from a manifest file."""
    dataset = TextToVideoDataset(
        manifest_path=manifest_path,
        split=split,
        num_frames=num_frames,
        height=height,
        width=width,
        max_prompt_length=max_prompt_length,
        vocab_size=vocab_size
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_ttv_batch,
        pin_memory=torch.cuda.is_available()
    )
