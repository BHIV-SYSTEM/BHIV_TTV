"""
Wan 2.2 TI2V-5B Video Generation Adapter
==========================================
Uses HuggingFace diffusers WanImageToVideoPipeline (Wan-AI/Wan2.2-TI2V-5B-Diffusers).

Hardware requirements
---------------------
* NVIDIA GPU with CUDA is REQUIRED for inference.
* Wan 2.2 TI2V-5B has ~5 Billion parameters (Diffusion Transformer).
* Minimum 14 GB VRAM required (24 GB recommended for 720p).
* On a local machine without NVIDIA GPU, a clear VisualGenerationError is raised.
* Inference is intended to run on the company NVIDIA GPU server.

Cloud deployment
----------------
Set the following in your cloud .env:
    VIDEO_PROVIDER=wan
    WAN_MODEL_ID=Wan-AI/Wan2.2-TI2V-5B-Diffusers
    WAN_DEVICE=cuda
    WAN_DTYPE=bfloat16
    HF_TOKEN=<your_huggingface_token>       # optional if model is public

The model is loaded lazily on the first request — NOT at application startup.

Supported modes
---------------
* Image-to-Video (primary): uses scene keyframe image + text prompt
* Text-to-Video (supported): synthesises an evocative cinematic background image
  derived from scene visual description and lighting prompts when no keyframe image is available.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING, List, Dict, Any

import numpy as np

from app.adapters.base import BaseVideoAdapter
from app.models.scene import Scene
from app.core.config import settings
from app.core.logging import telemetry, logger
from app.core.exceptions import VisualGenerationError

# torch is imported at module level so tests can mock torch.cuda.is_available.
# If torch is not installed, the ImportError is surfaced at inference time.
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from diffusers import WanImageToVideoPipeline  # noqa: F401 — type hint only


# ---------------------------------------------------------------------------
# Helpers & Compatibility Verification
# ---------------------------------------------------------------------------

_CUDA_ERROR_MESSAGE = (
    "\n"
    "╔══════════════════════════════════════════════════════════════════════╗\n"
    "║          Wan 2.2 TI2V-5B — NVIDIA GPU Required                      ║\n"
    "╠══════════════════════════════════════════════════════════════════════╣\n"
    "║  Wan inference requires an NVIDIA CUDA GPU.                         ║\n"
    "║  Your current environment does NOT have a CUDA-capable GPU.         ║\n"
    "║                                                                      ║\n"
    "║  Recommended options:                                                ║\n"
    "║  1. Deploy on a cloud NVIDIA GPU (RunPod, Lambda Labs, Colab Pro).  ║\n"
    "║  2. Switch to the local OpenCV provider for development:            ║\n"
    "║       Set VIDEO_PROVIDER=opencv in your .env file.                  ║\n"
    "╚══════════════════════════════════════════════════════════════════════╝\n"
)


def _check_cuda_available() -> None:
    """Raise VisualGenerationError with a clear message if CUDA is not available."""
    if not _TORCH_AVAILABLE or _torch is None:
        raise VisualGenerationError(
            "torch is not installed. "
            "Install it for your CUDA version before using the Wan adapter.\n"
            "See: https://pytorch.org/get-started/locally/"
        )
    if not _torch.cuda.is_available():
        raise VisualGenerationError(_CUDA_ERROR_MESSAGE)


def _verify_hardware_and_vram(device_str: str) -> None:
    """
    Verifies CUDA availability and checks available VRAM against Wan 2.2 requirements.
    Wan 2.2 TI2V-5B is a 5-billion parameter Diffusion Transformer requiring at least
    14 GB VRAM for inference (24 GB recommended for 720p).
    """
    _check_cuda_available()

    try:
        device_idx = 0
        if ":" in device_str:
            try:
                device_idx = int(device_str.split(":")[1])
            except Exception:
                device_idx = 0

        gpu_count = _torch.cuda.device_count()
        if device_idx >= gpu_count:
            device_idx = 0

        props = _torch.cuda.get_device_properties(device_idx)
        vram_total_gb = props.total_memory / (1024 ** 3)
        if vram_total_gb < 14.0:
            raise VisualGenerationError(
                f"Insufficient GPU VRAM for Wan 2.2 TI2V-5B: detected {vram_total_gb:.1f} GB "
                f"on GPU {device_idx} ('{props.name}'). Minimum 14.0 GB VRAM is required "
                "(24.0 GB recommended for 720p).\n"
                "Please allocate a company NVIDIA GPU server with sufficient VRAM."
            )
    except Exception as exc:
        if isinstance(exc, VisualGenerationError):
            raise
        logger.warning("[WanVideoAdapter] Could not inspect GPU VRAM properties: %s", exc)


def _validate_wan_lora_weights(lora_path_str: str) -> None:
    """
    Ensures that configured LoRA weights are actually compatible with Wan 2.2 DiT,
    and prevents loading legacy SpatialTemporalTTVModel weights.
    """
    p = Path(lora_path_str)
    if not p.exists():
        raise VisualGenerationError(f"Wan LoRA adapter does not exist: {lora_path_str}")

    # Inspect version_info.json or manifest in checkpoint dir if present
    target_dir = p if p.is_dir() else p.parent
    version_info = target_dir / "version_info.json"
    if version_info.exists():
        try:
            data = json.loads(version_info.read_text(encoding="utf-8"))
            base_model = str(data.get("base_model") or data.get("architecture") or "")
            if "SpatialTemporal" in base_model:
                raise VisualGenerationError(
                    f"Incompatible LoRA weights: '{lora_path_str}' was trained for "
                    f"'{base_model}', which is incompatible with Wan 2.2 TI2V-5B (DiT). "
                    "Do not load SpatialTemporal weights into Wan 2.2."
                )
        except Exception as exc:
            if isinstance(exc, VisualGenerationError):
                raise

    # If single weight file is given, inspect keys if PyTorch state_dict
    if p.is_file() and p.suffix.lower() == ".pt":
        try:
            state = _torch.load(str(p), map_location="cpu")
            if isinstance(state, dict):
                weights = state.get("lora_state_dict") or state.get("model_state_dict") or state
                sample_keys = list(weights.keys())[:10]
                if any("temp_block" in k or "spatial_encoder" in k or "text_encoder" in k for k in sample_keys):
                    raise VisualGenerationError(
                        f"Incompatible LoRA weights: '{lora_path_str}' contains SpatialTemporalTTVModel "
                        "parameters. Wan 2.2 TI2V-5B requires a Diffusion Transformer (DiT) LoRA adapter."
                    )
        except Exception as exc:
            if isinstance(exc, VisualGenerationError):
                raise


def _enhance_cinematic_prompt(scene: Scene, base_prompt: str) -> str:
    """
    Enriches the generation prompt with explicit camera motion, lighting,
    and photorealistic human/character motion descriptors.
    """
    camera_motion = scene.camera_motion or "cinematic camera tracking"
    enhancements = [
        base_prompt.strip(),
        f"camera movement: {camera_motion}",
        "cinematic lighting",
        "photorealistic human and character movement",
        "natural physical dynamics",
        "high scene consistency",
        "8k resolution",
        "finely detailed textures",
    ]
    return ", ".join(e for e in enhancements if e)


def _build_synthetic_image(scene: Scene, width: int, height: int):
    """
    Create a mood-aligned PIL Image representing the scene when no keyframe is available.
    Uses Pillow to create a smooth gradient/color composition reflecting scene visual tone.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise VisualGenerationError(
            "Pillow is required for Wan adapter. Install with: pip install Pillow"
        ) from exc

    desc = (scene.visual_description or "").lower()
    if any(k in desc for k in ["night", "dark", "space", "galaxy", "cosmos"]):
        top_color = (10, 10, 30)
        bot_color = (25, 20, 50)
    elif any(k in desc for k in ["desert", "sand", "mars", "arid"]):
        top_color = (210, 160, 100)
        bot_color = (160, 110, 60)
    elif any(k in desc for k in ["ocean", "sea", "water", "beach"]):
        top_color = (40, 100, 180)
        bot_color = (15, 45, 90)
    elif any(k in desc for k in ["forest", "jungle", "tree", "green"]):
        top_color = (35, 105, 45)
        bot_color = (15, 55, 20)
    elif any(k in desc for k in ["snow", "ice", "arctic", "winter", "blizzard"]):
        top_color = (230, 240, 255)
        bot_color = (180, 205, 230)
    elif any(k in desc for k in ["sunset", "golden", "dusk", "twilight", "sunrise"]):
        top_color = (220, 120, 50)
        bot_color = (90, 40, 60)
    else:
        top_color = (70, 75, 90)
        bot_color = (40, 45, 55)

    base = Image.new("RGB", (width, height), top_color)
    draw = ImageDraw.Draw(base)
    for y in range(height):
        factor = y / max(1, height - 1)
        r = int(top_color[0] + factor * (bot_color[0] - top_color[0]))
        g = int(top_color[1] + factor * (bot_color[1] - top_color[1]))
        b = int(top_color[2] + factor * (bot_color[2] - top_color[2]))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    return base


def _frames_to_mp4(frames, output_path: str, fps: int, width: int, height: int) -> None:
    """
    Write a list of PIL Images or numpy arrays to an MP4 file using OpenCV.
    OpenCV is used strictly for deterministic video writing/encoding.
    """
    try:
        from PIL import Image as _PILImage
        _pil_available = True
    except ImportError:
        _pil_available = False

    bgr_frames = []
    for f in frames:
        if _pil_available and isinstance(f, _PILImage.Image):
            arr = np.array(f.convert("RGB"))
        elif isinstance(f, np.ndarray):
            arr = f
        else:
            raise VisualGenerationError(f"Unknown frame type from Wan pipeline: {type(f)}")

        import cv2
        arr_resized = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LANCZOS4)
        bgr_frames.append(cv2.cvtColor(arr_resized, cv2.COLOR_RGB2BGR))

    if not bgr_frames:
        raise VisualGenerationError("Wan pipeline returned zero frames.")

    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        if not out.isOpened():
            raise VisualGenerationError(
                f"OpenCV VideoWriter failed to open output path: {output_path}"
            )

    for frame in bgr_frames:
        out.write(frame)
    out.release()


def _interpolate_frames(frames: list, target_count: int) -> list:
    """Temporally interpolate a frame list to match exact target frame count."""
    if len(frames) == 0:
        raise VisualGenerationError("Cannot interpolate — frame list is empty.")
    if target_count <= 0:
        raise VisualGenerationError("target_count must be positive.")
    if len(frames) == 1:
        return frames * target_count

    from PIL import Image as _PILImage
    indices = np.linspace(0, len(frames) - 1, target_count)
    result = []
    for idx_f in indices:
        lo = int(np.floor(idx_f))
        hi = min(len(frames) - 1, int(np.ceil(idx_f)))
        alpha = idx_f - lo
        if lo == hi or alpha == 0.0:
            result.append(frames[lo])
        else:
            a = np.array(frames[lo].convert("RGB")).astype(np.float32)
            b = np.array(frames[hi].convert("RGB")).astype(np.float32)
            blended = np.clip((1 - alpha) * a + alpha * b, 0, 255).astype(np.uint8)
            result.append(_PILImage.fromarray(blended))
    return result


# ---------------------------------------------------------------------------
# Main Adapter
# ---------------------------------------------------------------------------

class WanVideoAdapter(BaseVideoAdapter):
    """
    Wan 2.2 TI2V-5B Image-to-Video and Text-to-Video adapter.

    * Primary video generation model: Wan-AI/Wan2.2-TI2V-5B-Diffusers.
    * Loaded lazily on first inference request (NOT at startup).
    * Requires an NVIDIA CUDA GPU with sufficient VRAM (>=14GB).
    * Never silently falls back to OpenCV for AI video generation.
    * Surfaces actual errors with detailed diagnostics.
    """

    def __init__(self, require_lora: bool = False) -> None:
        self._pipeline = None  # Loaded lazily
        self._model_id: str = settings.WAN_MODEL_ID
        self._device: str = settings.WAN_DEVICE
        self._dtype_str: str = settings.WAN_DTYPE
        self._num_frames: int = settings.WAN_NUM_FRAMES
        self._inference_steps: int = settings.WAN_INFERENCE_STEPS
        self._guidance_scale: float = settings.WAN_GUIDANCE_SCALE
        self._require_lora = require_lora
        self._lora_path: Optional[str] = settings.WAN_LORA_PATH
        self._lora_scale: float = settings.WAN_LORA_SCALE
        logger.info(
            f"[WanVideoAdapter] Initialised (model={self._model_id}, "
            f"device={self._device}, dtype={self._dtype_str}). "
            "Model will be loaded on first inference call."
        )

    # ------------------------------------------------------------------
    # Pipeline loading (called lazily)
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """
        Load the Wan 2.2 TI2V-5B pipeline from HuggingFace Hub or local path.
        Called once on the first inference request.
        """
        if self._pipeline is not None:
            return

        logger.info(
            f"[WanVideoAdapter] Loading pipeline: {self._model_id} "
            f"on {self._device} / {self._dtype_str} ..."
        )

        try:
            import torch
            from diffusers import WanImageToVideoPipeline
        except ImportError as exc:
            raise VisualGenerationError(
                "Required packages for Wan video generation are not installed.\n"
                "Run: pip install diffusers>=0.33.0 transformers>=4.51.0 "
                "accelerate>=1.6.0 safetensors>=0.5.3"
            ) from exc

        # Check CUDA and VRAM before attempting model load
        _verify_hardware_and_vram(self._device)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._dtype_str.lower(), torch.bfloat16)
        hf_token: Optional[str] = getattr(settings, "HF_TOKEN", None) or None

        try:
            self._pipeline = WanImageToVideoPipeline.from_pretrained(
                self._model_id,
                torch_dtype=torch_dtype,
                token=hf_token,
            )

            # Check and apply LoRA adapter if requested
            if self._require_lora:
                if not self._lora_path:
                    raise VisualGenerationError(
                        "MODEL_MODE=wan_lora requires WAN_LORA_PATH. "
                        "Specify the path to a trained Wan 2.2 LoRA adapter."
                    )
                lora_path = os.path.abspath(self._lora_path)
                _validate_wan_lora_weights(lora_path)

                if not hasattr(self._pipeline, "load_lora_weights"):
                    raise VisualGenerationError(
                        "The installed diffusers Wan pipeline does not expose LoRA loading. "
                        "Upgrade diffusers before enabling wan_lora."
                    )
                self._pipeline.load_lora_weights(lora_path)
                if hasattr(self._pipeline, "fuse_lora"):
                    self._pipeline.fuse_lora(lora_scale=self._lora_scale)

            # Enable memory efficient features
            if hasattr(self._pipeline, "enable_model_cpu_offload"):
                # If VRAM is between 14GB and 20GB, CPU offloading ensures smooth generation
                try:
                    props = torch.cuda.get_device_properties(0)
                    if (props.total_memory / (1024**3)) < 22.0:
                        self._pipeline.enable_model_cpu_offload()
                        logger.info("[WanVideoAdapter] Enabled model CPU offload for optimal VRAM usage.")
                    else:
                        self._pipeline.to(self._device)
                except Exception:
                    self._pipeline.to(self._device)
            else:
                self._pipeline.to(self._device)

            logger.info("[WanVideoAdapter] Pipeline loaded and moved to device successfully.")

        except Exception as exc:
            self._pipeline = None
            if isinstance(exc, VisualGenerationError):
                raise
            raise VisualGenerationError(
                f"Failed to load Wan pipeline '{self._model_id}': {exc}\n"
                "Check that:\n"
                "  1. diffusers>=0.33.0 is installed.\n"
                "  2. Model ID is accessible on HuggingFace.\n"
                "  3. HF_TOKEN is valid if model requires authorization.\n"
                "  4. GPU has sufficient VRAM (>=14GB for 480p, >=24GB for 720p)."
            ) from exc

    # ------------------------------------------------------------------
    # Public interface (matches BaseVideoAdapter)
    # ------------------------------------------------------------------

    async def generate_scene_video(
        self,
        scene: Scene,
        keyframe_path: str,
        output_path: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
    ) -> str:
        """
        Generate a scene video using Wan 2.2 TI2V-5B.

        Parameters
        ----------
        scene : Scene
            Scene containing prompt details, duration, camera motion, and metadata.
        keyframe_path : str
            Reference image / keyframe path.
        output_path : str
            Destination MP4 path.
        width, height : int
            Target output resolution.
        fps : int
            Frames per second.
        """
        # ── 1. CUDA & hardware guard ──────────────────────────────────
        _check_cuda_available()

        # ── 2. Prepare output directory ────────────────────────────────
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ── 3. Prompt enhancement for cinematic & character movement ───
        raw_prompt = (
            scene.metadata.get("enriched_prompt")
            or scene.visual_description
            or scene.narrative
            or "cinematic video scene"
        )
        prompt = _enhance_cinematic_prompt(scene, raw_prompt)

        # ── 4. Input image loading (Image-to-Video vs Text-to-Video) ───
        image_mode: str
        if keyframe_path and os.path.exists(keyframe_path):
            try:
                from PIL import Image
                input_image = Image.open(keyframe_path).convert("RGB").resize((width, height))
                image_mode = "image_to_video"
            except Exception as exc:
                raise VisualGenerationError(
                    f"[WanVideoAdapter] Failed to open keyframe image '{keyframe_path}': {exc}"
                ) from exc
        else:
            logger.warning(
                "[WanVideoAdapter] No valid keyframe found at '%s'. "
                "Synthesizing visual reference (text-to-video mode).",
                keyframe_path,
            )
            input_image = _build_synthetic_image(scene, width, height)
            image_mode = "text_to_video_synthetic"

        # ── 5. Lazy-load the pipeline ──────────────────────────────────
        self._load_pipeline()

        # ── 6. Run inference with memory management ────────────────────
        telemetry.emit(
            "wan_video_generation",
            scene.title,
            {
                "engine": "wan2.2-ti2v-5b",
                "mode": image_mode,
                "device": self._device,
                "dtype": self._dtype_str,
                "num_frames": self._num_frames,
                "lora_enabled": self._require_lora,
                "resolution": f"{width}x{height}",
            },
        )

        logger.info(
            "[WanVideoAdapter] Running inference for scene '%s' "
            "(prompt=%r, mode=%s, frames=%d, steps=%d).",
            scene.title,
            prompt[:80],
            image_mode,
            self._num_frames,
            self._inference_steps,
        )

        # Clear GPU cache before running inference
        if _torch is not None and _torch.cuda.is_available():
            _torch.cuda.empty_cache()

        try:
            import torch
            with torch.inference_mode():
                output = self._pipeline(
                    image=input_image,
                    prompt=prompt,
                    num_frames=self._num_frames,
                    num_inference_steps=self._inference_steps,
                    guidance_scale=self._guidance_scale,
                )
            wan_frames = output.frames[0]  # list of PIL Images
        except Exception as exc:
            # Check specifically for GPU Out Of Memory
            if _torch is not None and isinstance(exc, getattr(_torch.cuda, "OutOfMemoryError", tuple())):
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                raise VisualGenerationError(
                    f"[WanVideoAdapter] GPU Out Of Memory during inference for scene '{scene.title}'.\n"
                    "Suggestions:\n"
                    "  1. Reduce video resolution (e.g. standard 720p or 480p).\n"
                    "  2. Ensure no other processes are consuming VRAM on the NVIDIA GPU server."
                ) from exc
            raise VisualGenerationError(
                f"[WanVideoAdapter] Inference failed for scene '{scene.title}': {exc}"
            ) from exc
        finally:
            if _torch is not None and _torch.cuda.is_available():
                _torch.cuda.empty_cache()

        # ── 7. Temporal interpolation to match exact scene duration ────
        target_frame_count = max(12, int(round(scene.duration * fps)))
        if len(wan_frames) != target_frame_count:
            logger.info(
                "[WanVideoAdapter] Interpolating %d Wan frames → %d frames "
                "(scene duration=%.2fs @ %d fps).",
                len(wan_frames),
                target_frame_count,
                scene.duration,
                fps,
            )
            wan_frames = _interpolate_frames(wan_frames, target_frame_count)

        # ── 8. Write frames to MP4 using OpenCV ────────────────────────
        _frames_to_mp4(wan_frames, output_path, fps=fps, width=width, height=height)

        logger.info(
            "[WanVideoAdapter] Scene '%s' written to: %s",
            scene.title,
            output_path,
        )
        return output_path
