"""
Video generation adapters.
"""
from app.adapters.video.opencv_video_adapter import OpenCVVideoAdapter
from app.adapters.video.external_video_adapter import ExternalVideoAdapter
from app.core.config import settings

def get_video_adapter(provider: str = "opencv"):
    mode = getattr(settings, "MODEL_MODE", "base").lower()
    if mode == "wan_lora":
        from app.adapters.video.wan_video_adapter import WanVideoAdapter
        return WanVideoAdapter(require_lora=True)
    if mode == "cogvideox_lora":
        from app.adapters.video.cogvideox_video_adapter import CogVideoXVideoAdapter
        return CogVideoXVideoAdapter(require_lora=True)
    if mode == "finetuned":
        from app.adapters.video.neural_video_adapter import NeuralVideoAdapter
        return NeuralVideoAdapter()
    p = (provider or "opencv").lower()
    if p == "external":
        return ExternalVideoAdapter()
    if p == "wan":
        # Imported lazily: diffusers/torch are NOT required unless VIDEO_PROVIDER=wan
        from app.adapters.video.wan_video_adapter import WanVideoAdapter
        return WanVideoAdapter()
    if p in ("cogvideox", "cogvideo", "cogvideox-2b"):
        # Imported lazily: diffusers/torch are NOT required unless VIDEO_PROVIDER=cogvideox
        from app.adapters.video.cogvideox_video_adapter import CogVideoXVideoAdapter
        return CogVideoXVideoAdapter()
    return OpenCVVideoAdapter()

