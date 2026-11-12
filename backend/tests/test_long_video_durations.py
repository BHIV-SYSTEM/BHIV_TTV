"""
Comprehensive tests for 15, 30, 45, and 60-Minute Reference Videos,
Wan 2.2 LoRA Compatibility, and Duration Validation.
"""
import os
import sys
import json
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure backend and root are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_BACKEND_DIR = _PROJECT_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.core.exceptions import ValidationError, VisualGenerationError
from app.adapters.reference.validator import validate_media_file
from training.datasets.validator import DatasetValidator
from training.preprocessing.long_video_processor import LongVideoProcessor
from app.adapters.video.wan_video_adapter import WanVideoAdapter
from app.pipelines.text_to_video import pipeline


def _generate_synthetic_video(output_path: str, duration_sec: int) -> str:
    """Generate a lightweight valid MP4 container with exact duration using FFmpeg."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=1",
        "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono",
        "-t", str(duration_sec),
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage", "-crf", "45",
        "-c:a", "aac", "-b:a", "16k",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=60)
    return output_path


# ===========================================================================
# 1. Tests for 15, 30, 45, and 60-Minute Videos
# ===========================================================================

@pytest.mark.parametrize("duration_sec,duration_min", [
    (900, 15),
    (1800, 30),
    (2700, 45),
    (3600, 60),
])
def test_long_video_durations_pipeline(tmp_path, duration_sec, duration_min):
    """
    Verifies that videos of 15, 30, 45, and 60 minutes are:
    1. Accepted by media validator (up to 3600s / 60 min)
    2. Correctly probed for duration and streams
    3. Segmented into manageable clips without loading into RAM
    4. Converted into valid manifest records with timestamps
    5. Validated by DatasetValidator
    """
    video_file = str(tmp_path / f"test_{duration_min}min.mp4")
    _generate_synthetic_video(video_file, duration_sec)

    # 1. Media validation
    val_result = validate_media_file(video_file, max_duration=3600.0)
    assert val_result["media_type"] == "video"
    assert abs(val_result["duration"] - duration_sec) <= 2.0

    # 2. LongVideoProcessor metadata & segmentation
    processor = LongVideoProcessor(
        max_duration=3600.0,
        segment_duration=5.0,
        sample_interval=float(duration_sec / 3),
        max_segments=3
    )
    meta = processor.validate_and_extract_metadata(video_file)
    assert abs(meta["duration"] - duration_sec) <= 2.0
    assert meta["has_audio"] is True

    # 3. Audio extraction
    audio_path = str(tmp_path / f"audio_{duration_min}.mp3")
    extracted_audio = processor.extract_audio(video_file, audio_path)
    assert extracted_audio is not None
    assert os.path.exists(extracted_audio)

    # 4. Video segmentation
    segments_dir = str(tmp_path / f"segments_{duration_min}")
    segments = processor.segment_long_video(video_file, segments_dir, total_duration=meta["duration"], reference_id=f"ref_{duration_min}m")
    assert len(segments) >= 1
    for seg in segments:
        assert os.path.exists(seg["video_path"])
        assert "start_time" in seg
        assert "end_time" in seg
        assert seg["end_time"] <= duration_sec + 2.0

    # 5. Build Manifest & Validate
    manifest_path = str(tmp_path / f"manifest_{duration_min}.jsonl")
    records = processor.build_training_manifest(segments, manifest_path)
    assert len(records) == len(segments)
    assert os.path.exists(manifest_path)

    dataset_validator = DatasetValidator(min_duration=0.1, max_duration=3600.0)
    dataset_res = dataset_validator.validate_dataset(records)
    assert dataset_res["valid_count"] == len(records)
    assert dataset_res["rejected_count"] == 0


def test_video_exceeding_60_minutes_rejected(tmp_path):
    """Videos exceeding 60 minutes (e.g. 3605s) must be rejected with clear ValidationError."""
    video_file = str(tmp_path / "test_61min.mp4")
    _generate_synthetic_video(video_file, 3605)

    processor = LongVideoProcessor(max_duration=3600.0)
    with pytest.raises(ValidationError) as exc:
        processor.validate_and_extract_metadata(video_file)
    assert "exceeds" in str(exc.value).lower()
    assert "60" in str(exc.value)


# ===========================================================================
# 2. Wan 2.2 LoRA Compatibility Tests
# ===========================================================================

def test_wan_lora_rejects_incompatible_spatial_temporal_weights(tmp_path):
    """
    Loading legacy SpatialTemporalTTVModel weights into Wan 2.2 must be rejected
    with a clear architecture incompatibility error rather than silent corruption.
    """
    import torch
    incompatible_dir = tmp_path / "legacy_lora"
    incompatible_dir.mkdir()

    # Create dummy legacy state dict
    legacy_weights = {
        "spatial_conv.weight": torch.randn(64, 3, 3, 3),
        "temporal_conv.weight": torch.randn(64, 64, 3, 1),
    }
    torch.save(legacy_weights, incompatible_dir / "adapter_model.pt")

    # Metadata file specifying legacy architecture
    (incompatible_dir / "version_info.json").write_text(
        json.dumps({"architecture": "SpatialTemporalTTVModel", "version_name": "legacy_v1"}),
        encoding="utf-8"
    )

    adapter = WanVideoAdapter()
    with pytest.raises((VisualGenerationError, ValueError)) as exc:
        adapter.load_lora_weights(str(incompatible_dir))

    err_text = str(exc.value)
    assert "incompatible" in err_text.lower() or "spatialtemporalttvmodel" in err_text.lower()


def test_wan_lora_accepts_valid_wan_checkpoint(tmp_path):
    """
    Valid Wan 2.2 LoRA checkpoint with Wan2.2-TI2V-5B architecture metadata is accepted.
    """
    import torch
    wan_dir = tmp_path / "wan_lora_valid"
    wan_dir.mkdir()

    wan_weights = {
        "blocks.0.self_attn.q.lora_A": torch.randn(16, 4096),
        "blocks.0.self_attn.q.lora_B": torch.randn(4096, 16),
    }
    torch.save(wan_weights, wan_dir / "adapter_model.pt")
    (wan_dir / "version_info.json").write_text(
        json.dumps({
            "architecture": "Wan2.2-TI2V-5B",
            "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "lora_rank": 16
        }),
        encoding="utf-8"
    )

    adapter = WanVideoAdapter()
    loaded_path = adapter.load_lora_weights(str(wan_dir))
    assert loaded_path == str(wan_dir)
    assert adapter._lora_weights_path == str(wan_dir)


# ===========================================================================
# 3. Output Video Duration Validation Tests
# ===========================================================================

def test_video_duration_validation_matching(tmp_path):
    """Validates that requested duration matching actual MP4 duration passes."""
    video_path = str(tmp_path / "correct_duration.mp4")
    _generate_synthetic_video(video_path, 4)

    # Expected 4.0s vs Actual ~4.0s
    res = pipeline.validate_video(video_path, execution_id="test_run_1", expected_duration=4.0)
    assert abs(res["duration"] - 4.0) <= 1.5


def test_video_duration_validation_mismatch(tmp_path):
    """Validates that requested duration mismatching actual MP4 duration raises VisualGenerationError."""
    video_path = str(tmp_path / "mismatch_duration.mp4")
    _generate_synthetic_video(video_path, 2)

    # Requested 8.0s but actual is 2.0s -> outside tolerance (+/-1.5s)
    with pytest.raises(VisualGenerationError) as exc:
        pipeline.validate_video(video_path, execution_id="test_run_2", expected_duration=8.0)

    assert "duration mismatch" in str(exc.value).lower()
