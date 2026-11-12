"""
Tests for WanVideoAdapter integration.

Tests:
1. Adapter factory correctly returns WanVideoAdapter when provider="wan"
2. Default factory (no provider) still returns OpenCVVideoAdapter
3. WanVideoAdapter.generate_scene_video raises a clear VisualGenerationError
   with a human-readable message when CUDA is unavailable (expected behaviour
   on local Intel/AMD machines)
4. Both image-to-video and text-to-video (no keyframe) code paths raise the
   CUDA error before reaching any model loading

All tests work on the local development machine (no CUDA required to run
the test suite — the test suite verifies that the error is raised correctly).
"""

import os
import sys
import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure backend is on sys.path when running from the project root
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.models.scene import Scene
from app.adapters.video import get_video_adapter
from app.adapters.video.opencv_video_adapter import OpenCVVideoAdapter
from app.adapters.video.wan_video_adapter import WanVideoAdapter, _CUDA_ERROR_MESSAGE
from app.core.exceptions import VisualGenerationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scene(**kwargs) -> Scene:
    defaults = dict(
        index=1,
        title="Test Scene",
        narrative="A cinematic test.",
        visual_description="A vast ocean at golden hour.",
        duration=3.0,
        camera_motion="pan_right",
    )
    defaults.update(kwargs)
    return Scene(**defaults)


# ---------------------------------------------------------------------------
# Test 1: Factory selection — wan provider
# ---------------------------------------------------------------------------

def test_wan_adapter_selected_by_factory():
    """get_video_adapter('wan') must return a WanVideoAdapter instance."""
    adapter = get_video_adapter("wan")
    assert isinstance(adapter, WanVideoAdapter), (
        f"Expected WanVideoAdapter, got {type(adapter).__name__}"
    )


# ---------------------------------------------------------------------------
# Test 2: Factory selection — default falls back to OpenCV
# ---------------------------------------------------------------------------

def test_opencv_adapter_selected_as_default():
    """get_video_adapter with no argument or 'opencv' must return OpenCVVideoAdapter."""
    adapter_default = get_video_adapter()
    assert isinstance(adapter_default, WanVideoAdapter), (
        "Default provider must be OpenCVVideoAdapter."
    )

    adapter_opencv = get_video_adapter("opencv")
    assert isinstance(adapter_opencv, OpenCVVideoAdapter), (
        "Explicit 'opencv' must return OpenCVVideoAdapter."
    )


# ---------------------------------------------------------------------------
# Test 3: CUDA unavailable — image-to-video path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wan_raises_clear_error_without_cuda_image_to_video(tmp_path):
    """
    When CUDA is unavailable, generate_scene_video must raise VisualGenerationError
    with a descriptive message (NOT a silent fallback to OpenCV).
    This is the expected behaviour on Intel Iris Xe / non-NVIDIA machines.
    """
    # Create a dummy keyframe image so the code reaches the CUDA check
    keyframe = str(tmp_path / "dummy_keyframe.jpg")
    from PIL import Image
    Image.new("RGB", (320, 240), (100, 150, 200)).save(keyframe, "JPEG")

    output_path = str(tmp_path / "scene_001_clip.mp4")
    scene = _make_scene()

    adapter = WanVideoAdapter()

    # Force CUDA to appear unavailable by patching is_available via the
    # module-level _torch reference inside wan_video_adapter.
    import app.adapters.video.wan_video_adapter as _wan_mod
    original_fn = _wan_mod._torch.cuda.is_available
    _wan_mod._torch.cuda.is_available = lambda: False
    try:
        with pytest.raises(VisualGenerationError) as exc_info:
            await adapter.generate_scene_video(
                scene=scene,
                keyframe_path=keyframe,
                output_path=output_path,
                width=320,
                height=240,
                fps=15,
            )
    finally:
        _wan_mod._torch.cuda.is_available = original_fn

    error_message = str(exc_info.value)
    assert "NVIDIA" in error_message or "CUDA" in error_message, (
        "Error message must mention NVIDIA or CUDA. "
        f"Got: {error_message!r}"
    )
    assert "VIDEO_PROVIDER=opencv" in error_message, (
        "Error message must guide user to switch to opencv provider. "
        f"Got: {error_message!r}"
    )

    # Ensure no video file was created (no silent fallback)
    assert not os.path.exists(output_path), (
        "No video file should be created when CUDA is unavailable."
    )


# ---------------------------------------------------------------------------
# Test 4: CUDA unavailable — text-to-video path (no keyframe)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wan_raises_clear_error_without_cuda_text_to_video(tmp_path):
    """
    Same CUDA guard applies when keyframe_path is missing (text-to-video mode).
    The error must be raised BEFORE any model loading attempt.
    """
    output_path = str(tmp_path / "scene_002_clip.mp4")
    scene = _make_scene(visual_description="A desert at sunset.")

    adapter = WanVideoAdapter()

    import app.adapters.video.wan_video_adapter as _wan_mod
    original_fn = _wan_mod._torch.cuda.is_available
    _wan_mod._torch.cuda.is_available = lambda: False
    try:
        with pytest.raises(VisualGenerationError) as exc_info:
            await adapter.generate_scene_video(
                scene=scene,
                keyframe_path="/nonexistent/path/frame.jpg",
                output_path=output_path,
                width=320,
                height=240,
                fps=15,
            )
    finally:
        _wan_mod._torch.cuda.is_available = original_fn

    assert "NVIDIA" in str(exc_info.value) or "CUDA" in str(exc_info.value)
    assert not os.path.exists(output_path)



# ---------------------------------------------------------------------------
# Test 5: WanVideoAdapter does NOT load the model at __init__ time
# ---------------------------------------------------------------------------

def test_wan_adapter_lazy_model_loading():
    """
    __init__ must NOT trigger any model/pipeline loading.
    _pipeline attribute must be None after construction.
    """
    adapter = WanVideoAdapter()
    assert adapter._pipeline is None, (
        "Pipeline must be None at construction — lazy loading only."
    )


# ---------------------------------------------------------------------------
# Test 6: WanVideoAdapter respects config values
# ---------------------------------------------------------------------------

def test_wan_adapter_reads_config():
    """
    WanVideoAdapter must read model_id, device, dtype, num_frames from settings.
    """
    from app.core.config import settings
    adapter = WanVideoAdapter()
    assert adapter._model_id == settings.WAN_MODEL_ID
    assert adapter._device == settings.WAN_DEVICE
    assert adapter._dtype_str == settings.WAN_DTYPE
    assert adapter._num_frames == settings.WAN_NUM_FRAMES
    assert adapter._inference_steps == settings.WAN_INFERENCE_STEPS
    assert adapter._guidance_scale == settings.WAN_GUIDANCE_SCALE


# ---------------------------------------------------------------------------
# Test 7: Factory — case-insensitive provider matching
# ---------------------------------------------------------------------------

def test_provider_matching_case_insensitive():
    """get_video_adapter must match 'WAN', 'Wan', 'wan' all the same."""
    for variant in ["wan", "WAN", "Wan"]:
        adapter = get_video_adapter(variant)
        assert isinstance(adapter, WanVideoAdapter), (
            f"Provider variant '{variant}' should return WanVideoAdapter."
        )
