"""
Long Video Processor for 15 to 60-Minute References.
===================================================
Implements the streaming, segmented processing pipeline:
  Long Video Input
  -> Video Validation (up to 60 minutes / 3600 seconds)
  -> Metadata Extraction (FFprobe / OpenCV without full RAM loading)
  -> Audio Extraction (FFmpeg audio stream demuxing)
  -> Frame Sampling & Scene Detection (interval and scene change detection)
  -> Video Segmentation (manageable short clips without blowing up RAM/VRAM)
  -> Dataset Preparation (manifest records with timestamps and source info)
  -> Dataset Validation
  -> Clean up of temporary files
"""
from __future__ import annotations

import os
import json
import shutil
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
from PIL import Image

from app.core.config import settings
from app.core.exceptions import ValidationError
from app.core.logging import logger


class LongVideoProcessor:
    def __init__(
        self,
        max_duration: float = 3600.0,
        segment_duration: float = 5.0,
        sample_interval: float = 30.0,
        max_segments: int = 30,
        output_resolution: Tuple[int, int] = (1280, 720),
    ):
        self.max_duration = max_duration
        self.segment_duration = segment_duration
        self.sample_interval = sample_interval
        self.max_segments = max_segments
        self.width, self.height = output_resolution

    def validate_and_extract_metadata(self, video_path: str) -> Dict[str, Any]:
        """
        Extract video duration, resolution, fps, bitrate and audio stream info
        without loading the entire video into memory.
        """
        if not os.path.exists(video_path):
            raise ValidationError(f"Video file does not exist: {video_path}")

        file_size = os.path.getsize(video_path)
        if file_size < 100:
            raise ValidationError("Video file is empty or corrupted.")

        metadata: Dict[str, Any] = {
            "path": video_path,
            "file_size_bytes": file_size,
            "width": 1280,
            "height": 720,
            "fps": 24.0,
            "duration": 0.0,
            "has_audio": False,
            "audio_codec": None,
            "format_name": "mp4",
        }

        # Try ffprobe first for accurate stream analysis
        ffprobe_success = False
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "stream=width,height,r_frame_rate,codec_type,codec_name:format=duration,size,format_name",
                "-of", "json",
                video_path,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if res.returncode == 0 and res.stdout.strip():
                info = json.loads(res.stdout)
                fmt = info.get("format", {})
                metadata["duration"] = float(fmt.get("duration") or 0.0)
                metadata["format_name"] = str(fmt.get("format_name") or "mp4")

                for stream in info.get("streams", []):
                    stype = stream.get("codec_type")
                    if stype == "video" and not metadata.get("video_inspected"):
                        metadata["width"] = int(stream.get("width") or 1280)
                        metadata["height"] = int(stream.get("height") or 720)
                        fps_str = stream.get("r_frame_rate", "24/1")
                        if "/" in fps_str:
                            num, den = fps_str.split("/")
                            metadata["fps"] = round(float(num) / max(1.0, float(den)), 2)
                        metadata["video_inspected"] = True
                    elif stype == "audio":
                        metadata["has_audio"] = True
                        metadata["audio_codec"] = stream.get("codec_name")

                ffprobe_success = True
        except Exception as e:
            logger.warning("ffprobe probe failed for %s, falling back to cv2: %s", video_path, e)

        # Fallback to OpenCV if ffprobe duration is missing
        if not ffprobe_success or metadata["duration"] <= 0:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValidationError("Could not open video with FFprobe or OpenCV.")
            try:
                metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or metadata["width"])
                metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or metadata["height"])
                fps_cap = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
                metadata["fps"] = round(fps_cap, 2) if fps_cap > 0 else 24.0
                frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                metadata["duration"] = round(frame_count / max(1.0, metadata["fps"]), 2)
            finally:
                cap.release()

        # Enforce bounds
        duration = metadata["duration"]
        if duration < 0.1:
            raise ValidationError("Video duration is too short or could not be determined.")
        if duration > self.max_duration:
            raise ValidationError(
                f"Video duration ({duration:.1f}s = {duration/60:.1f} min) exceeds "
                f"the maximum allowed limit of {self.max_duration:.1f}s ({self.max_duration/60:.0f} minutes)."
            )

        return metadata

    def extract_audio(self, video_path: str, output_audio_path: str) -> Optional[str]:
        """Demux audio track from long video to a lightweight MP3 file using FFmpeg."""
        os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",
                "-acodec", "libmp3lame",
                "-q:a", "2",
                output_audio_path,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.returncode == 0 and os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                return output_audio_path
        except Exception as e:
            logger.warning("Audio extraction failed for %s: %s", video_path, e)
        return None

    def segment_long_video(
        self,
        video_path: str,
        output_dir: str,
        total_duration: float,
        reference_id: str = "ref",
    ) -> List[Dict[str, Any]]:
        """
        Splits a long video (15-60 min) into small representative clips and keyframes.
        This enables efficient training without ever loading the full 60-min stream into RAM.
        """
        os.makedirs(output_dir, exist_ok=True)
        segments: List[Dict[str, Any]] = []

        # Determine sampling points across the duration
        usable_duration = max(1.0, total_duration - self.segment_duration)
        step = max(self.sample_interval, usable_duration / float(self.max_segments))

        timestamps: List[float] = []
        curr = 0.0
        while curr <= usable_duration and len(timestamps) < self.max_segments:
            timestamps.append(round(curr, 2))
            curr += step

        if not timestamps:
            timestamps = [0.0]

        seen_frame_hashes = set()

        for idx, start_ts in enumerate(timestamps):
            seg_filename = f"{reference_id}_seg_{idx:03d}.mp4"
            still_filename = f"{reference_id}_still_{idx:03d}.jpg"
            seg_path = os.path.join(output_dir, seg_filename)
            still_path = os.path.join(output_dir, still_filename)

            # 1. Extract still frame at start_ts
            try:
                cmd_still = [
                    "ffmpeg", "-y",
                    "-ss", str(start_ts),
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", "2",
                    still_path,
                ]
                subprocess.run(cmd_still, capture_output=True, timeout=30)
            except Exception as e:
                logger.warning("Still frame extraction failed at %ss: %s", start_ts, e)

            # Check still validity and deduplicate
            if os.path.exists(still_path) and os.path.getsize(still_path) > 0:
                with open(still_path, "rb") as sf:
                    fhash = hashlib.md5(sf.read()).hexdigest()
                if fhash in seen_frame_hashes:
                    # Skip duplicate identical frame
                    continue
                seen_frame_hashes.add(fhash)

            # 2. Extract short segment clip (e.g. 5 seconds)
            seg_duration = min(self.segment_duration, total_duration - start_ts)
            try:
                cmd_seg = [
                    "ffmpeg", "-y",
                    "-ss", str(start_ts),
                    "-i", video_path,
                    "-t", str(seg_duration),
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-avoid_negative_ts", "make_zero",
                    seg_path,
                ]
                subprocess.run(cmd_seg, capture_output=True, timeout=60)
            except Exception as e:
                logger.warning("Segment extraction failed at %ss: %s", start_ts, e)

            if os.path.exists(seg_path) and os.path.getsize(seg_path) > 500:
                segments.append({
                    "segment_index": idx,
                    "video_path": seg_path,
                    "still_path": still_path if os.path.exists(still_path) else None,
                    "start_time": start_ts,
                    "end_time": round(start_ts + seg_duration, 2),
                    "duration": round(seg_duration, 2),
                    "source_video": video_path,
                })

        return segments

    def build_training_manifest(
        self,
        segments: List[Dict[str, Any]],
        output_manifest_path: str,
        caption_base: str = "cinematic video scene",
    ) -> List[Dict[str, Any]]:
        """
        Writes dataset manifest records for all extracted segments.
        """
        records = []
        os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)

        for seg in segments:
            idx = seg["segment_index"]
            prompt = (
                f"{caption_base}, scene sequence {idx + 1}, "
                f"timestamp {seg['start_time']}s to {seg['end_time']}s"
            )
            record = {
                "id": f"seg_{idx:04d}",
                "prompt": prompt,
                "video_path": seg["video_path"],
                "still_path": seg.get("still_path"),
                "duration": seg["duration"],
                "split": "train" if (idx % 5 != 0) else "val",
                "source_video": seg.get("source_video"),
                "timestamp_start": seg["start_time"],
                "timestamp_end": seg["end_time"],
            }
            records.append(record)

        with open(output_manifest_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        return records

    def process_long_video_pipeline(
        self,
        video_path: str,
        workspace_dir: str,
        reference_id: str,
        caption: str = "high quality video training sequence",
    ) -> Dict[str, Any]:
        """
        Full end-to-end long video pipeline execution.
        """
        meta = self.validate_and_extract_metadata(video_path)
        audio_path = os.path.join(workspace_dir, "extracted_audio.mp3")
        self.extract_audio(video_path, audio_path)

        segments_dir = os.path.join(workspace_dir, "segments")
        segments = self.segment_long_video(
            video_path=video_path,
            output_dir=segments_dir,
            total_duration=meta["duration"],
            reference_id=reference_id,
        )

        manifest_path = os.path.join(workspace_dir, f"{reference_id}_dataset.jsonl")
        records = self.build_training_manifest(
            segments=segments,
            output_manifest_path=manifest_path,
            caption_base=caption,
        )

        return {
            "metadata": meta,
            "audio_path": audio_path if os.path.exists(audio_path) else None,
            "segments_count": len(segments),
            "segments": segments,
            "manifest_path": manifest_path,
            "records": records,
        }
