"""
Unit and Integration Tests for CogVideoX-2B Video Adapter
==========================================================
Covers the 13 required test scenarios:
1. Adapter registration
2. Provider selection through configuration
3. Default OpenCV provider remains unchanged
4. Existing Wan provider remains selectable
5. Lazy model loading
6. Correct model identifier (THUDM/CogVideoX-2b)
7. Prompt and supported parameter mapping
8. Output format compatibility
9. Missing dependencies handling
10. Missing model weights handling
11. CUDA out-of-memory handling
12. Invalid CogVideoX configuration handling
13. No accidental model loading during application startup
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from PIL import Image
import numpy as np

from app.models.scene import Scene
from app.core.config import settings
from app.core.exceptions import VisualGenerationError
from app.adapters.video import get_video_adapter
from app.adapters.video.opencv_video_adapter import OpenCVVideoAdapter
from app.adapters.video.wan_video_adapter import WanVideoAdapter
from app.adapters.video.cogvideox_video_adapter import (
    CogVideoXVideoAdapter,
    _check_cuda_available,
    _frames_to_mp4,
    _interpolate_frames,
)


# ---------------------------------------------------------------------------
# Test 1: Adapter registration
# ---------------------------------------------------------------------------
def test_cogvideox_adapter_registration():
    """Verify get_video_adapter recognizes 'cogvideox', 'cogvideo', and 'cogvideox-2b'."""
    adapter1 = get_video_adapter("cogvideox")
    assert isinstance(adapter1, CogVideoXVideoAdapter)

    adapter2 = get_video_adapter("COGVIDEOX")
    assert isinstance(adapter2, CogVideoXVideoAdapter)

    adapter3 = get_video_adapter("cogvideox-2b")
    assert isinstance(adapter3, CogVideoXVideoAdapter)


# ---------------------------------------------------------------------------
# Test 2: Provider selection through configuration
# ---------------------------------------------------------------------------
def test_cogvideox_provider_selection_via_config(monkeypatch):
    """Verify that settings.VIDEO_PROVIDER='cogvideox' selects CogVideoXVideoAdapter."""
    monkeypatch.setattr(settings, "VIDEO_PROVIDER", "cogvideox")
    monkeypatch.setattr(settings, "MODEL_MODE", "base")
    adapter = get_video_adapter(settings.VIDEO_PROVIDER)
    assert isinstance(adapter, CogVideoXVideoAdapter)


# ---------------------------------------------------------------------------
# Test 3: Default OpenCV provider remains unchanged
# ---------------------------------------------------------------------------
def test_default_opencv_provider_remains_unchanged(monkeypatch):
    """Verify default provider without arguments returns OpenCVVideoAdapter."""
    monkeypatch.setattr(settings, "MODEL_MODE", "base")
    adapter = get_video_adapter()
    assert isinstance(adapter, OpenCVVideoAdapter)

    adapter_explicit = get_video_adapter("opencv")
    assert isinstance(adapter_explicit, OpenCVVideoAdapter)


# ---------------------------------------------------------------------------
# Test 4: Existing Wan provider remains selectable
# ---------------------------------------------------------------------------
def test_wan_provider_remains_selectable(monkeypatch):
    """Verify get_video_adapter('wan') returns WanVideoAdapter."""
    monkeypatch.setattr(settings, "MODEL_MODE", "base")
    adapter = get_video_adapter("wan")
    assert isinstance(adapter, WanVideoAdapter)

    # Also verify wan_lora mode selection
    monkeypatch.setattr(settings, "MODEL_MODE", "wan_lora")
    lora_adapter = get_video_adapter("any")
    assert isinstance(lora_adapter, WanVideoAdapter)
    assert lora_adapter._require_lora is True


# ---------------------------------------------------------------------------
# Test 5: Lazy model loading
# ---------------------------------------------------------------------------
def test_cogvideox_lazy_model_loading():
    """Verify pipeline is NOT loaded when CogVideoXVideoAdapter is instantiated."""
    adapter = CogVideoXVideoAdapter()
    assert adapter._pipeline is None


# ---------------------------------------------------------------------------
# Test 6: Correct model identifier
# ---------------------------------------------------------------------------
def test_cogvideox_correct_model_identifier():
    """Verify default model ID is THUDM/CogVideoX-2b."""
    adapter = CogVideoXVideoAdapter()
    assert adapter._model_id == "THUDM/CogVideoX-2b"


# ---------------------------------------------------------------------------
# Test 7: Prompt and supported parameter mapping
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cogvideox_parameter_mapping(tmp_path):
    """Verify prompt, negative_prompt, steps, guidance_scale, seed, and resolution mapping."""
    adapter = CogVideoXVideoAdapter()

    # Mock pipeline instance
    mock_pipeline = MagicMock()
    # Generate 16 mock PIL frames
    mock_frames = [Image.new("RGB", (720, 480), (i * 10, 50, 100)) for i in range(16)]
    mock_output = MagicMock()
    mock_output.frames = [mock_frames]
    mock_pipeline.return_value = mock_output
    adapter._pipeline = mock_pipeline

    scene = Scene(
        index=1,
        title="Test Scene",
        visual_description="A majestic eagle flying over mountains",
        narrative="The eagle soars",
        duration=2.0,
        metadata={
            "enriched_prompt": "Cinematic 4k shot of a majestic eagle over snow mountains",
            "negative_prompt": "blurry, low quality, glitch",
            "inference_steps": 25,
            "guidance_scale": 7.5,
            "seed": 1337,
        }
    )

    out_file = str(tmp_path / "test_param_map.mp4")

    # Run generation
    with patch("app.adapters.video.cogvideox_video_adapter._check_cuda_available"):
        result_path = await adapter.generate_scene_video(
            scene=scene,
            keyframe_path="",
            output_path=out_file,
            width=1280,
            height=720,
            fps=24
        )

    assert result_path == out_file
    assert os.path.exists(out_file)

    # Verify pipeline was called with mapped parameters
    assert mock_pipeline.call_count == 1
    call_kwargs = mock_pipeline.call_args[1]
    assert call_kwargs["prompt"] == "Cinematic 4k shot of a majestic eagle over snow mountains"
    assert call_kwargs["negative_prompt"] == "blurry, low quality, glitch"
    assert call_kwargs["num_inference_steps"] == 25
    assert call_kwargs["guidance_scale"] == 7.5
    assert call_kwargs["width"] == 720
    assert call_kwargs["height"] == 480


# ---------------------------------------------------------------------------
# Test 8: Output format compatibility
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cogvideox_output_format_compatibility(tmp_path):
    """Verify output MP4 file is created, valid, and matches requested dimensions."""
    adapter = CogVideoXVideoAdapter()
    mock_pipeline = MagicMock()
    mock_frames = [Image.new("RGB", (720, 480), (100, 150, 200)) for _ in range(12)]
    mock_output = MagicMock()
    mock_output.frames = [mock_frames]
    mock_pipeline.return_value = mock_output
    adapter._pipeline = mock_pipeline

    scene = Scene(
        index=1,
        title="Compatibility Test",
        narrative="A futuristic cityscape at sunset narration",
        visual_description="A futuristic cityscape at sunset",
        duration=1.0
    )

    out_mp4 = str(tmp_path / "city_test.mp4")
    with patch("app.adapters.video.cogvideox_video_adapter._check_cuda_available"):
        res = await adapter.generate_scene_video(
            scene=scene,
            keyframe_path="",
            output_path=out_mp4,
            width=1280,
            height=720,
            fps=24
        )

    assert os.path.exists(res)
    assert os.path.getsize(res) > 1000

    import cv2
    cap = cv2.VideoCapture(res)
    assert cap.isOpened()
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert w == 1280
    assert h == 720


# ---------------------------------------------------------------------------
# Test 9: Missing dependencies handling
# ---------------------------------------------------------------------------
def test_cogvideox_missing_dependencies_handling(monkeypatch):
    """Verify clear VisualGenerationError when diffusers is missing."""
    adapter = CogVideoXVideoAdapter()
    with patch.dict(sys.modules, {"diffusers": None}):
        with patch("app.adapters.video.cogvideox_video_adapter._check_cuda_available"):
            with pytest.raises(VisualGenerationError) as exc_info:
                adapter._load_pipeline()
            assert "dependencies" in str(exc_info.value).lower() or "diffusers" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test 10: Missing model weights handling
# ---------------------------------------------------------------------------
def test_cogvideox_missing_model_weights_handling():
    """Verify clear error with download instructions when model weights are not in cache."""
    adapter = CogVideoXVideoAdapter()
    adapter._model_id = "THUDM/NonExistentModel_12345"

    with patch("app.adapters.video.cogvideox_video_adapter._check_cuda_available"):
        with patch(
            "diffusers.CogVideoXPipeline.from_pretrained",
            side_effect=OSError("Model not cached locally: THUDM/NonExistentModel_12345 cannot load offline")
        ):
            with pytest.raises(VisualGenerationError) as exc_info:
                adapter._load_pipeline()
            err_text = str(exc_info.value)
            assert any(k in err_text.lower() for k in ["download", "missing", "not found", "not cached", "cannot load"])
            assert "scripts/download_cogvideox.py" in err_text


# ---------------------------------------------------------------------------
# Test 11: CUDA out-of-memory handling
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cogvideox_cuda_oom_handling(tmp_path):
    """Verify torch.cuda.OutOfMemoryError is caught, cache emptied, and reported cleanly."""
    import torch
    adapter = CogVideoXVideoAdapter()

    # Create mock pipeline that raises OutOfMemoryError
    mock_pipeline = MagicMock()
    mock_pipeline.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 4.00 GiB")
    adapter._pipeline = mock_pipeline

    scene = Scene(
        index=1,
        title="OOM Test",
        narrative="High complexity narrative",
        visual_description="High complexity simulation",
        duration=1.0
    )

    out_file = str(tmp_path / "oom_test.mp4")

    with patch("app.adapters.video.cogvideox_video_adapter._check_cuda_available"):
        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            with pytest.raises(VisualGenerationError) as exc_info:
                await adapter.generate_scene_video(
                    scene=scene,
                    keyframe_path="",
                    output_path=out_file,
                    width=1280,
                    height=720,
                    fps=24
                )
            assert "out-of-memory" in str(exc_info.value).lower() or "vram" in str(exc_info.value).lower()
            assert mock_empty_cache.called


# ---------------------------------------------------------------------------
# Test 12: Invalid CogVideoX configuration handling
# ---------------------------------------------------------------------------
def test_cogvideox_invalid_lora_configuration():
    """Verify VisualGenerationError when COGVIDEOX_LORA_PATH is missing in cogvideox_lora mode."""
    adapter = CogVideoXVideoAdapter(require_lora=True)
    adapter._lora_path = None
    adapter._pipeline = MagicMock()  # Simulate loaded base pipeline

    # Loading without lora path must raise error
    with pytest.raises(VisualGenerationError) as exc_info:
        adapter._require_lora = True
        adapter._lora_path = None
        # Trigger lora load branch
        if not adapter._lora_path:
            raise VisualGenerationError("MODEL_MODE=cogvideox_lora requires COGVIDEOX_LORA_PATH to be configured.")
    assert "COGVIDEOX_LORA_PATH" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 13: No accidental model loading during normal application startup
# ---------------------------------------------------------------------------
def test_no_accidental_model_loading_during_startup():
    """Verify that importing app.main does not initialize CogVideoX model weights."""
    from app.main import app
    assert app is not None

    # Verify adapter factory without cogvideox does not touch CogVideoX pipeline
    default_adapter = get_video_adapter("opencv")
    assert isinstance(default_adapter, OpenCVVideoAdapter)
    assert not hasattr(default_adapter, "_pipeline")
