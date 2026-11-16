import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional


def validate_video_file(
    file_path: str,
    expected_duration: Optional[float] = None,
    expected_fps: Optional[float] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Rigorously validate that a video file exists, is readable, is a valid MP4 container,
    and contains meaningful visual content without corruption, pure noise, or black/blank frames.
    """
    if not os.path.exists(file_path):
        return False, {"error": f"File does not exist: {file_path}"}

    file_size = os.path.getsize(file_path)
    if file_size < 10000:
        return False, {"error": f"File too small ({file_size} bytes), likely corrupt or empty"}

    # Check MP4 container magic header bytes
    with open(file_path, "rb") as f:
        header = f.read(32)
        if b"ftyp" not in header and b"moov" not in header:
            return False, {"error": "Invalid MP4 header (missing 'ftyp' or 'moov' atom)"}

    # Open with OpenCV to verify actual stream decoding
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return False, {"error": "Failed to decode video stream via OpenCV"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps and fps > 0 else 0.0

    if frame_count <= 0 or duration <= 0:
        cap.release()
        return False, {"error": f"Video stream has no valid frames (count={frame_count})"}

    # Duration check if requested
    if expected_duration is not None and expected_duration > 0:
        if abs(duration - expected_duration) > 1.5:
            cap.release()
            return False, {
                "error": f"Video duration mismatch: actual {duration:.2f}s vs expected {expected_duration:.2f}s",
                "actual_duration": duration,
                "expected_duration": expected_duration
            }

    # Extract sample frames across timeline to ensure valid visual content
    sample_indices = [
        0,
        max(0, int(frame_count * 0.25)),
        max(0, int(frame_count * 0.50)),
        max(0, int(frame_count * 0.75)),
        max(0, frame_count - 1)
    ]
    # Remove duplicates while preserving order
    sample_indices = sorted(list(set(sample_indices)))

    sample_stats = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return False, {"error": f"Failed to decode frame at index {idx} in {file_path}"}

        # Check for NaN / Inf
        if not np.isfinite(frame).all():
            cap.release()
            return False, {"error": f"Frame at index {idx} contains NaN or Inf pixel values"}

        # Calculate luminance and variance metrics
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Check for completely black, white, or monochrome flat frames
        if std_val < 2.0:
            cap.release()
            return False, {
                "error": f"Frame at index {idx} is completely monochrome/flat (std={std_val:.2f})",
                "mean": mean_val,
                "std": std_val
            }
        if mean_val < 2.0 or mean_val > 253.0:
            cap.release()
            return False, {
                "error": f"Frame at index {idx} is clipped black or white (mean={mean_val:.2f})",
                "mean": mean_val,
                "std": std_val
            }

        sample_stats.append({
            "frame_idx": idx,
            "mean_luminance": round(mean_val, 2),
            "std_luminance": round(std_val, 2),
            "laplacian_variance": round(lap_var, 2)
        })

    cap.release()

    return True, {
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration": round(duration, 2),
        "file_size_bytes": file_size,
        "sample_frames_checked": len(sample_stats),
        "sample_stats": sample_stats
    }
