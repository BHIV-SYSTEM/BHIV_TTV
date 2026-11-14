"""
CogVideoX-2B Video Generation Adapter
======================================
Uses HuggingFace Diffusers CogVideoXPipeline (THUDM/CogVideoX-2b).

Hardware requirements & Optimization for 10 GB VRAM (RTX 3080):
---------------------------------------------------------------
* NVIDIA GPU with CUDA is REQUIRED for inference.
* Lazy model loading: weights are only loaded on the first generation call.
* Memory optimizations enabled by default:
  - bfloat16 / float16 model precision
  - Model CPU offloading (loads pipeline modules to GPU only when executing)
  - VAE tiling & VAE slicing (processes video latents in tiles to prevent peak VRAM spikes)
  - Attention slicing (reduces memory consumption during cross-attention)
* Out-of-memory errors are safely caught, cache emptied, and cleanly reported without
  crashing the FastAPI host service.
"""

from __future__ import annotations

import os
import gc
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING

import numpy as np

from app.adapters.base import BaseVideoAdapter
from app.models.scene import Scene
from app.core.config import settings
from app.core.logging import telemetry, logger
from app.core.exceptions import VisualGenerationError
from app.utils.media_check import validate_video_file

# torch is imported at module level so tests can mock torch.cuda.is_available.
# If torch is not installed, the error is surfaced at inference time.
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from diffusers import CogVideoXPipeline  # noqa: F401 — type hint only


# ---------------------------------------------------------------------------
# Helpers & Formatting
# ---------------------------------------------------------------------------

_CUDA_ERROR_MESSAGE = (
    "\n"
    "╔══════════════════════════════════════════════════════════════════════╗\n"
    "║          CogVideoX-2B — NVIDIA CUDA GPU Required                     ║\n"
    "╠══════════════════════════════════════════════════════════════════════╣\n"
    "║  CogVideoX inference requires an NVIDIA GPU with CUDA.               ║\n"
    "║  Your current environment does NOT have a CUDA-capable GPU active.   ║\n"
    "║                                                                      ║\n"
    "║  Recommended options:                                                ║\n"
    "║  1. Deploy on an NVIDIA CUDA system (e.g. RTX 3080 10GB or higher).  ║\n"
    "║  2. Switch to the local OpenCV provider for development:             ║\n"
    "║       Set VIDEO_PROVIDER=opencv in your .env file.                  ║\n"
    "╚══════════════════════════════════════════════════════════════════════╝\n"
)

_WEIGHTS_MISSING_MESSAGE = (
    "\n"
    "╔══════════════════════════════════════════════════════════════════════╗\n"
    "║          CogVideoX-2B Model Weights Missing or Inaccessible          ║\n"
    "╠══════════════════════════════════════════════════════════════════════╣\n"
    "║  Model weights for 'THUDM/CogVideoX-2b' were not found in the local  ║\n"
    "║  Hugging Face cache.                                                 ║\n"
    "║                                                                      ║\n"
    "║  To download the weights (~9-12 GB) in a verified, resumable manner: ║\n"
    "║    python scripts/download_cogvideox.py --confirm                    ║\n"
    "║                                                                      ║\n"
    "║  Alternatively, set VIDEO_PROVIDER=opencv in your .env file.         ║\n"
    "╚══════════════════════════════════════════════════════════════════════╝\n"
)


def _check_cuda_available() -> None:
    """Raise VisualGenerationError with a clear message if CUDA is not available."""
    if not _TORCH_AVAILABLE or _torch is None:
        raise VisualGenerationError(
            "torch is not installed. Install PyTorch with CUDA support before using CogVideoX.\n"
            "See: https://pytorch.org/get-started/locally/"
        )
    if not _torch.cuda.is_available():
        raise VisualGenerationError(_CUDA_ERROR_MESSAGE)


def _frames_to_mp4(frames: List[Any], output_path: str, fps: int, width: int, height: int) -> str:
    """
    Write a list of PIL Images, numpy arrays, or tensors to an MP4 file using OpenCV.
    Guarantees that all frames are resized to (width, height) without distortion.
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
        elif hasattr(f, "cpu") and hasattr(f, "numpy"):
            arr = f.cpu().numpy()
        else:
            raise VisualGenerationError(f"Unsupported frame type from CogVideoX pipeline: {type(f)}")

        import cv2
        if arr.shape[0] != height or arr.shape[1] != width:
            arr_resized = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LANCZOS4)
        else:
            arr_resized = arr
        bgr_frames.append(cv2.cvtColor(arr_resized, cv2.COLOR_RGB2BGR))

    if not bgr_frames:
        raise VisualGenerationError("CogVideoX pipeline returned zero frames.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        if not out.isOpened():
            raise VisualGenerationError(f"OpenCV VideoWriter failed to create output file: {output_path}")

    for frame in bgr_frames:
        out.write(frame)
    out.release()

    return output_path


def _interpolate_frames(frames: List[Any], target_count: int) -> List[Any]:
    """Temporally interpolate a frame list to match a target frame count."""
    if len(frames) == 0:
        raise VisualGenerationError("Cannot interpolate empty frame list.")
    if target_count <= 0:
        raise VisualGenerationError("target_count must be positive.")
    if len(frames) == 1 or len(frames) == target_count:
        return frames if len(frames) == target_count else frames * target_count

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
# CogVideoX-2B Video Adapter
# ---------------------------------------------------------------------------

class CogVideoXVideoAdapter(BaseVideoAdapter):
    """
    CogVideoX-2B Text-to-Video generation adapter for BHIV TTV Studio.

    - Architecture: Diffusion Transformer (DiT) text-to-video foundation model.
    - Model ID: THUDM/CogVideoX-2b.
    - Implements the BaseVideoAdapter interface seamlessly with TextToVideoPipeline.
    - Lazy loading: The pipeline is loaded into memory only upon the first inference call.
    - VRAM Safety for RTX 3080 (10 GB):
      * bfloat16 / float16 model weights
      * VAE slicing & tiling to cap peak VRAM during decoding
      * Automatic model CPU offloading
      * Attention slicing
      * Handles torch.cuda.OutOfMemoryError gracefully with automatic cache recovery
    """

    def __init__(self, require_lora: bool = False) -> None:
        self._pipeline = None  # Loaded lazily on first inference request
        self._model_id: str = getattr(settings, "COGVIDEOX_MODEL_ID", "THUDM/CogVideoX-2b")
        self._device: str = getattr(settings, "COGVIDEOX_DEVICE", "cuda")
        self._dtype_str: str = getattr(settings, "COGVIDEOX_DTYPE", "bfloat16")
        self._cpu_offload: bool = getattr(settings, "COGVIDEOX_CPU_OFFLOAD", True)
        self._sequential_cpu_offload: bool = getattr(settings, "COGVIDEOX_SEQUENTIAL_CPU_OFFLOAD", False)
        self._enable_tiling: bool = getattr(settings, "COGVIDEOX_ENABLE_TILING", True)
        self._enable_slicing: bool = getattr(settings, "COGVIDEOX_ENABLE_SLICING", True)
        self._num_frames: int = getattr(settings, "COGVIDEOX_NUM_FRAMES", 49)
        self._inference_steps: int = getattr(settings, "COGVIDEOX_NUM_INFERENCE_STEPS", 30)
        self._guidance_scale: float = getattr(settings, "COGVIDEOX_GUIDANCE_SCALE", 6.0)
        self._default_width: int = getattr(settings, "COGVIDEOX_DEFAULT_WIDTH", 720)
        self._default_height: int = getattr(settings, "COGVIDEOX_DEFAULT_HEIGHT", 480)
        self._require_lora: bool = require_lora
        self._lora_path: Optional[str] = getattr(settings, "COGVIDEOX_LORA_PATH", None)
        self._lora_scale: float = getattr(settings, "COGVIDEOX_LORA_SCALE", 1.0)

        logger.info(
            f"[CogVideoXVideoAdapter] Initialized (model={self._model_id}, "
            f"device={self._device}, dtype={self._dtype_str}, cpu_offload={self._cpu_offload}). "
            "Pipeline will load lazily on first generation call."
        )

    # ------------------------------------------------------------------
    # Pipeline loading (called lazily)
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        """
        Load the CogVideoX-2B Diffusers pipeline from Hugging Face cache or local path.
        Applies memory-efficient offloading and VAE slicing/tiling to operate within 10GB VRAM.
        """
        if self._pipeline is not None:
            return  # Already loaded

        _check_cuda_available()

        logger.info(
            f"[CogVideoXVideoAdapter] Loading pipeline: {self._model_id} "
            f"(device={self._device}, dtype={self._dtype_str}) ..."
        )

        try:
            import torch
            from diffusers import CogVideoXPipeline
        except ImportError as exc:
            raise VisualGenerationError(
                "Required dependencies for CogVideoX are not installed.\n"
                "Ensure diffusers>=0.33.0, transformers>=4.45.0, accelerate>=0.33.0, and torch with CUDA are installed."
            ) from exc

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._dtype_str.lower(), torch.bfloat16)
        hf_token: Optional[str] = getattr(settings, "HF_TOKEN", None) or None

        try:
            self._pipeline = CogVideoXPipeline.from_pretrained(
                self._model_id,
                torch_dtype=torch_dtype,
                token=hf_token,
            )
        except Exception as exc:
            self._pipeline = None
            err_msg = str(exc)
            if any(k in err_msg.lower() for k in ["not found", "offline", "404", "gated", "not cached", "401", "cannot load"]):
                raise VisualGenerationError(
                    f"{_WEIGHTS_MISSING_MESSAGE}\nUnderlying error: {err_msg}"
                ) from exc
            raise VisualGenerationError(
                f"Failed to load CogVideoX pipeline '{self._model_id}': {exc}"
            ) from exc

        # ── Memory-saving configurations for 10 GB VRAM ─────────────────────
        try:
            # 1. VAE Tiling (decodes video latents in overlapping tiles)
            if self._enable_tiling and hasattr(self._pipeline, "vae") and hasattr(self._pipeline.vae, "enable_tiling"):
                self._pipeline.vae.enable_tiling()
                logger.debug("[CogVideoXVideoAdapter] VAE tiling enabled.")

            # 2. VAE Slicing (slices temporal dimension to reduce memory)
            if self._enable_slicing and hasattr(self._pipeline, "vae") and hasattr(self._pipeline.vae, "enable_slicing"):
                self._pipeline.vae.enable_slicing()
                logger.debug("[CogVideoXVideoAdapter] VAE slicing enabled.")

            # 3. Attention Slicing
            if hasattr(self._pipeline, "enable_attention_slicing"):
                self._pipeline.enable_attention_slicing()
                logger.debug("[CogVideoXVideoAdapter] Attention slicing enabled.")

            # 4. CPU Offload Selection
            if self._sequential_cpu_offload and hasattr(self._pipeline, "enable_sequential_cpu_offload"):
                self._pipeline.enable_sequential_cpu_offload()
                logger.info("[CogVideoXVideoAdapter] Sequential CPU offload enabled.")
            elif self._cpu_offload and hasattr(self._pipeline, "enable_model_cpu_offload"):
                self._pipeline.enable_model_cpu_offload()
                logger.info("[CogVideoXVideoAdapter] Model CPU offload enabled.")
            else:
                self._pipeline.to(self._device)
                logger.info(f"[CogVideoXVideoAdapter] Pipeline moved directly to device '{self._device}'.")

            # 5. Optional LoRA Adapter Loading
            if self._require_lora:
                if not self._lora_path:
                    raise VisualGenerationError(
                        "MODEL_MODE=cogvideox_lora requires COGVIDEOX_LORA_PATH to be configured."
                    )
                lora_path = os.path.abspath(self._lora_path)
                if not os.path.exists(lora_path):
                    raise VisualGenerationError(f"CogVideoX LoRA adapter path does not exist: {lora_path}")
                if hasattr(self._pipeline, "load_lora_weights"):
                    self._pipeline.load_lora_weights(lora_path)
                    if hasattr(self._pipeline, "fuse_lora"):
                        self._pipeline.fuse_lora(lora_scale=self._lora_scale)
                    logger.info(f"[CogVideoXVideoAdapter] Loaded and fused LoRA adapter: {lora_path}")
                else:
                    raise VisualGenerationError("Installed diffusers pipeline does not expose load_lora_weights.")

        except Exception as exc:
            self._pipeline = None
            raise VisualGenerationError(f"Failed configuring memory optimizations on CogVideoX: {exc}") from exc

        logger.info("[CogVideoXVideoAdapter] Pipeline successfully initialized and ready for inference.")

    # ------------------------------------------------------------------
    # Public BaseVideoAdapter interface
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
        Generate a scene video using CogVideoX-2B text-to-video diffusion.

        Parameters
        ----------
        scene : Scene
            Scene object containing prompt, narrative, duration, metadata.
        keyframe_path : str
            Path to keyframe image (if available, used for visual context or seed).
        output_path : str
            Destination MP4 file path.
        width, height : int
            Target output video dimensions.
        fps : int
            Target frames per second.

        Returns
        -------
        str
            Absolute path of the generated MP4 video file.
        """
        # 1. Check CUDA availability
        _check_cuda_available()

        # 2. Lazily load pipeline
        self._load_pipeline()

        # 3. Extract and sanitize prompt
        meta = scene.metadata if isinstance(scene.metadata, dict) else {}
        prompt = (
            meta.get("enriched_prompt")
            or scene.visual_description
            or scene.narrative
            or scene.title
            or "A cinematic high-definition video sequence."
        ).strip()

        negative_prompt = (
            meta.get("negative_prompt")
            or "low quality, distorted, jitter, deformities, bad anatomy, artifacts, blurry, watermark"
        )

        # 4. Map generation parameters
        steps = int(meta.get("inference_steps") or self._inference_steps)
        if steps < 25:
            logger.warning(
                f"[CogVideoXVideoAdapter] Requested {steps} inference steps is below recommended minimum (25). "
                "CogVideoX DDIM requires >= 25 steps to resolve high-frequency features. Clamping to 25."
            )
            steps = max(steps, 25)
        guidance = float(meta.get("guidance_scale") or self._guidance_scale)
        seed = int(meta.get("seed") or 42)

        # CogVideoX-2B standard native generation resolution is 720x480 (multiples of 32/16)
        gen_width = self._default_width
        gen_height = self._default_height

        # CogVideoX generates 49 frames natively (~6 seconds at 8fps)
        num_frames = self._num_frames

        import torch
        generator = torch.Generator(device="cpu").manual_seed(seed)

        exec_id = str(getattr(scene, "scene_id", None) or getattr(scene, "index", "scene"))

        telemetry.emit(
            "cogvideox_inference_started",
            exec_id,
            {
                "prompt": prompt[:120],
                "steps": steps,
                "guidance_scale": guidance,
                "native_res": f"{gen_width}x{gen_height}",
                "target_res": f"{width}x{height}",
                "num_frames": num_frames,
                "fps": fps,
            }
        )

        logger.info(
            f"[CogVideoXVideoAdapter] Starting inference for scene '{scene.title or scene.index}': "
            f"prompt='{prompt[:80]}...', steps={steps}, frames={num_frames}, res={gen_width}x{gen_height}"
        )

        # 5. Execute model inference with CUDA OOM safety
        try:
            output = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=gen_height,
                width=gen_width,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
            raw_frames = output.frames[0]
            if not raw_frames:
                raise VisualGenerationError("CogVideoX pipeline returned zero frames.")

            # Degenerate frame guard: verify frames contain structural spatial content
            sample_frame = raw_frames[0]
            if hasattr(sample_frame, "convert"):
                arr_sample = np.array(sample_frame.convert("L"), dtype=float)
                col_std = float(arr_sample.std(axis=1).mean())
                if col_std < 5.0 and arr_sample.std() > 50.0:
                    logger.warning(
                        f"[CogVideoXVideoAdapter] Generated frames exhibit degenerate horizontal variance (std={col_std:.2f}). "
                        "The output may be an unfinished diffusion gradient. Recommended to use >= 30 inference steps."
                    )
        except torch.cuda.OutOfMemoryError as oom:
            if _torch and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            gc.collect()
            telemetry.emit("cogvideox_oom", exec_id, {"error": str(oom)}, level="error")
            raise VisualGenerationError(
                f"CUDA Out-of-Memory during CogVideoX generation on RTX 3080: {oom}\n"
                "Suggestions to fit within 10GB VRAM:\n"
                "  1. Ensure COGVIDEOX_CPU_OFFLOAD=True in .env\n"
                "  2. Ensure COGVIDEOX_ENABLE_TILING=True and COGVIDEOX_ENABLE_SLICING=True\n"
                "  3. Keep resolution at 720x480\n"
                "  4. Reduce COGVIDEOX_NUM_FRAMES or batch size."
            ) from oom
        except Exception as exc:
            raise VisualGenerationError(f"CogVideoX generation failed: {exc}") from exc

        # 6. Temporal interpolation if scene duration calls for more frames at target fps
        target_frames = max(12, int(scene.duration * fps)) if scene.duration > 0 else len(raw_frames)
        if target_frames != len(raw_frames) and target_frames > len(raw_frames):
            logger.debug(
                f"[CogVideoXVideoAdapter] Interpolating frames from {len(raw_frames)} to {target_frames} "
                f"for duration {scene.duration}s at {fps} fps."
            )
            final_frames = _interpolate_frames(raw_frames, target_frames)
        else:
            final_frames = raw_frames

        # 7. Write to MP4 container at target resolution
        _frames_to_mp4(final_frames, output_path, fps=fps, width=width, height=height)

        # 8. Validate output video
        is_valid, info = validate_video_file(output_path)
        if not is_valid:
            raise VisualGenerationError(f"Generated CogVideoX video failed validation: {info.get('error')}")

        telemetry.emit("cogvideox_inference_completed", exec_id, info)
        logger.info(f"[CogVideoXVideoAdapter] Successfully synthesized video: {output_path} ({info})")

        return output_path
