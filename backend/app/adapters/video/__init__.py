"""
Video generation adapters.
Primary AI provider: Wan 2.2 TI2V-5B (Wan-AI/Wan2.2-TI2V-5B-Diffusers).
OpenCV is preserved only for deterministic video/image processing, frame extraction, and testing.
"""
from typing import Optional
from app.adapters.video.opencv_video_adapter import OpenCVVideoAdapter
from app.adapters.video.external_video_adapter import ExternalVideoAdapter
from app.core.config import settings


def get_video_adapter(provider: Optional[str] = None):
    """
    Returns the configured video adapter.
    Primary AI model: Wan 2.2 TI2V-5B.
    """
    mode = getattr(settings, "MODEL_MODE", "base").lower()
    if mode == "wan_lora":
        from app.adapters.video.wan_video_adapter import WanVideoAdapter
        return WanVideoAdapter(require_lora=True)
    if mode == "finetuned":
        from app.adapters.video.neural_video_adapter import NeuralVideoAdapter
        return NeuralVideoAdapter()

    # Determine provider (default to configured VIDEO_PROVIDER, which defaults to 'wan')
    chosen_provider = provider or getattr(settings, "VIDEO_PROVIDER", "wan") or "wan"
    p = str(chosen_provider).strip().lower()

    if p == "wan":
        # Imported lazily: diffusers/torch are NOT required unless VIDEO_PROVIDER=wan
        from app.adapters.video.wan_video_adapter import WanVideoAdapter
        return WanVideoAdapter()
    if p == "opencv":
        return OpenCVVideoAdapter()
    if p == "external":
        return ExternalVideoAdapter()

    raise ValueError(f"Unknown video provider: '{provider}'. Configured primary provider is 'wan'.")
