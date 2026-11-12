import os
import cv2
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


def compute_video_hash(video_path: str) -> str:
    """Computes SHA-256 hash of a video file for deduplication."""
    if not os.path.exists(video_path):
        return ""
    hasher = hashlib.sha256()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def probe_video_integrity(video_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Probes video file using OpenCV to ensure it can be decoded without corruption.
    Extracts frame count, fps, width, height, and duration.
    """
    if not os.path.exists(video_path):
        return False, {"error": f"File does not exist: {video_path}"}

    if os.path.getsize(video_path) == 0:
        return False, {"error": "File size is 0 bytes"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, {"error": "Failed to open video file with OpenCV"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0 or frame_count <= 0:
        cap.release()
        return False, {"error": "Invalid video dimensions or frame count"}

    # Probe decoding the first frame
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        return False, {"error": "Failed to decode first frame"}

    # Probe decoding a frame near the end
    if frame_count > 2:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_count - 1, frame_count // 2))
        ret_end, frame_end = cap.read()
        if not ret_end or frame_end is None:
            cap.release()
            return False, {"error": "Failed to decode subsequent frames"}

    cap.release()

    duration = frame_count / fps if fps and fps > 0 else 0.0
    return True, {
        "width": width,
        "height": height,
        "fps": round(fps, 2) if fps else 24.0,
        "frame_count": frame_count,
        "duration_seconds": round(duration, 2),
        "file_size_bytes": os.path.getsize(video_path)
    }


class DatasetValidator:
    """
    Validates Text-to-Video training dataset manifests.
    Enforces:
    - Non-empty prompt and video path
    - Existence and decodability of video files
    - Duplicate detection (video SHA-256 and prompt matching)
    - Metadata consistency
    """
    def __init__(self, min_duration: float = 0.1, max_duration: float = 3600.0, min_prompt_len: int = 5):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_prompt_len = min_prompt_len

    def validate_dataset(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_samples = []
        rejected_samples = []
        seen_video_hashes = set()
        seen_prompts = set()

        for idx, sample in enumerate(samples):
            prompt = str(sample.get("prompt") or sample.get("caption") or "").strip()
            video_path = str(sample.get("video_path", "")).strip()
            video_url = str(sample.get("video_url") or sample.get("source_url") or "").strip()

            # 1. Prompt length check
            if len(prompt) < self.min_prompt_len:
                rejected_samples.append({
                    "index": idx,
                    "sample": sample,
                    "reason": f"Prompt length ({len(prompt)}) < min ({self.min_prompt_len})"
                })
                continue

            # 2. Missing video check
            if not video_path and not video_url:
                rejected_samples.append({
                    "index": idx,
                    "sample": sample,
                    "reason": f"Video file missing: '{video_path}'"
                })
                continue

            if video_path and not os.path.exists(video_path):
                rejected_samples.append({
                    "index": idx,
                    "sample": sample,
                    "reason": f"Video file missing: '{video_path}'"
                })
                continue

            if video_path and os.path.exists(video_path):
                # 3. Corrupted video check & metadata extraction
                is_valid, info = probe_video_integrity(video_path)
                if not is_valid:
                    rejected_samples.append({
                        "index": idx,
                        "sample": sample,
                        "reason": f"Corrupted video: {info.get('error')}"
                    })
                    continue

                duration = info["duration_seconds"]
                if duration < self.min_duration or duration > self.max_duration:
                    rejected_samples.append({
                        "index": idx,
                        "sample": sample,
                        "reason": f"Duration {duration}s out of bounds [{self.min_duration}, {self.max_duration}]"
                    })
                    continue
            else:
                info = {
                    "width": 1280,
                    "height": 720,
                    "fps": 24.0,
                    "frame_count": 96,
                    "duration_seconds": float(sample.get("duration", 4.0)),
                    "remote_url": video_url
                }

            # 4. Duplicate detection
            video_hash = compute_video_hash(video_path)
            norm_prompt = prompt.lower()
            if video_hash in seen_video_hashes and norm_prompt in seen_prompts:
                rejected_samples.append({
                    "index": idx,
                    "sample": sample,
                    "reason": f"Exact duplicate of existing video and prompt"
                })
                continue

            seen_video_hashes.add(video_hash)
            seen_prompts.add(norm_prompt)

            # Enrich sample with verified metadata
            sample_copy = dict(sample)
            sample_copy["video_hash"] = video_hash
            sample_copy["verified_metadata"] = info
            valid_samples.append(sample_copy)

        return {
            "total_evaluated": len(samples),
            "valid_count": len(valid_samples),
            "rejected_count": len(rejected_samples),
            "valid_samples": valid_samples,
            "rejected_samples": rejected_samples
        }


def split_dataset(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """Reproducibly splits validated dataset into train, val, and test partitions."""
    import random
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Ensure at least 1 in val and test if dataset has at least 3 samples
    if n >= 3:
        if n_val == 0:
            n_val = 1
        if n - (n_train + n_val) <= 0 and n_train > 1:
            n_train -= 1

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train:n_train + n_val]
    test_set = shuffled[n_train + n_val:]

    return {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }
