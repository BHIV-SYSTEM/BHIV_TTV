"""
Media Preprocessing Engine for Video, Image, and Text Datasets.
Extracts technical metadata, generates thumbnails, samples representative keyframes,
and validates media integrity without modifying or moving original files.
"""
from __future__ import annotations

import os
import re
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

from app.core.logging import logger


class MediaPreprocessor:
    """Handles integrity validation, metadata extraction, frame sampling, and thumbnails."""

    def __init__(self, storage_root: str = "data/datasets"):
        self.storage_root = Path(storage_root)
        self.thumbnails_dir = self.storage_root / "thumbnails"
        self.keyframes_dir = self.storage_root / "keyframes"
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitizes filename against path traversal and special characters."""
        base = os.path.basename(filename)
        # Remove any path traversal tokens or dangerous symbols
        clean = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', base)
        return clean.strip('._') or "unnamed_media"

    def inspect_and_preprocess_video(
        self,
        video_path: str,
        media_id: str,
        extract_frames: bool = True,
        max_keyframes: int = 3,
    ) -> Dict[str, Any]:
        """
        Validates video file integrity, extracts metadata, produces keyframes and a thumbnail.
        """
        path = Path(video_path)
        if not path.exists() or path.stat().st_size < 1024:
            return {
                "valid": False,
                "error": "File does not exist or is empty (< 1KB)",
                "width": 0, "height": 0, "fps": 0.0, "duration": 0.0, "frame_count": 0,
                "keyframe_paths": [], "thumbnail_path": None, "codec": "unknown",
            }

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {
                "valid": False,
                "error": "Corrupted video file: OpenCV could not open container",
                "width": 0, "height": 0, "fps": 0.0, "duration": 0.0, "frame_count": 0,
                "keyframe_paths": [], "thumbnail_path": None, "codec": "unknown",
            }

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = round(frame_count / fps, 2) if fps > 0 and frame_count > 0 else 0.0

        # Try reading fourcc codec code
        try:
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).strip()
        except Exception:
            codec = "h264"

        # Keyframe extraction & thumbnail generation
        keyframe_paths: List[str] = []
        thumbnail_path: Optional[str] = None

        if frame_count > 0:
            media_kf_dir = self.keyframes_dir / media_id
            media_kf_dir.mkdir(parents=True, exist_ok=True)
            thumb_dest = self.thumbnails_dir / f"{media_id}_thumb.jpg"

            # Determine keyframe intervals
            step = max(1, frame_count // (max_keyframes + 1))
            sampled_frames = []

            for i in range(1, max_keyframes + 1):
                target_idx = min(frame_count - 1, i * step)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    sampled_frames.append(frame)
                    if extract_frames:
                        kf_file = media_kf_dir / f"keyframe_{i}.jpg"
                        cv2.imwrite(str(kf_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        keyframe_paths.append(str(kf_file))

            # Use middle frame for thumbnail
            if sampled_frames:
                mid_frame = sampled_frames[len(sampled_frames) // 2]
                # Resize thumbnail to max 320x180 maintaining aspect ratio
                h, w = mid_frame.shape[:2]
                scale = min(320 / max(1, w), 180 / max(1, h), 1.0)
                thumb_img = cv2.resize(mid_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(thumb_dest), thumb_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                thumbnail_path = str(thumb_dest)

        cap.release()

        is_valid = width > 0 and height > 0 and frame_count > 0 and duration > 0.0

        return {
            "valid": is_valid,
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "codec": codec or "h264",
            "keyframe_paths": keyframe_paths,
            "thumbnail_path": thumbnail_path,
            "error": None if is_valid else "Invalid or empty video stream",
        }

    def inspect_and_preprocess_image(
        self,
        image_path: str,
        media_id: str,
    ) -> Dict[str, Any]:
        """
        Validates image file integrity, extracts dimensions, format, blur score, and generates a thumbnail.
        """
        path = Path(image_path)
        if not path.exists() or path.stat().st_size < 100:
            return {
                "valid": False,
                "error": "Image file does not exist or is empty (< 100B)",
                "width": 0, "height": 0, "format": "unknown", "blur_score": 0.0,
                "thumbnail_path": None,
            }

        try:
            with Image.open(path) as img:
                img.verify()
            # Reopen for metadata after verify()
            with Image.open(path) as img:
                width, height = img.size
                fmt = (img.format or "JPEG").lower()
                img_rgb = img.convert("RGB")
                
                # Thumbnail generation
                thumb_dest = self.thumbnails_dir / f"{media_id}_thumb.jpg"
                thumb_copy = img_rgb.copy()
                thumb_copy.thumbnail((320, 320), Image.Resampling.LANCZOS)
                thumb_copy.save(thumb_dest, "JPEG", quality=85)
                thumbnail_path = str(thumb_dest)

            # Blur assessment using OpenCV Laplacian variance
            cv_img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            blur_score = float(cv2.Laplacian(cv_img, cv2.CV_64F).var()) if cv_img is not None else 0.0

            return {
                "valid": True,
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "format": fmt,
                "blur_score": round(blur_score, 2),
                "thumbnail_path": thumbnail_path,
                "error": None,
            }

        except Exception as exc:
            return {
                "valid": False,
                "error": f"Image corruption or unsupported format: {exc}",
                "width": 0, "height": 0, "format": "unknown", "blur_score": 0.0,
                "thumbnail_path": None,
            }

    def inspect_text(self, text: str) -> Dict[str, Any]:
        """Inspects text content integrity and length."""
        cleaned = (text or "").strip()
        word_count = len(cleaned.split())
        char_count = len(cleaned)
        is_valid = word_count >= 3 and char_count >= 15
        return {
            "valid": is_valid,
            "word_count": word_count,
            "char_count": char_count,
            "cleaned_text": cleaned,
            "error": None if is_valid else "Text too short (< 3 words or < 15 chars)",
        }
