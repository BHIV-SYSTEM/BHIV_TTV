import os
import sys
import json
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[4]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from app.adapters.base import BaseVideoAdapter
from app.models.scene import Scene
from app.core.config import settings
from app.core.logging import telemetry, logger
from app.core.exceptions import VisualGenerationError

from training.fine_tuning.model import SpatialTemporalTTVModel
from training.fine_tuning.lora import apply_lora_to_model, load_lora_state_dict
from training.preprocessing.video_preprocessor import VideoPreprocessor
from training.preprocessing.text_preprocessor import TextPreprocessor


class NeuralVideoAdapter(BaseVideoAdapter):
    """
    Fine-Tuned Neural Video Generation Adapter.
    Uses trained SpatialTemporalTTVModel with LoRA weights to synthesize
    neural motion trajectories from scene keyframes and prompts.
    Does NOT use fake fallback generation — reports explicit diagnostic status.
    """
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or settings.FINE_TUNED_CHECKPOINT_PATH
        self.model: Optional[SpatialTemporalTTVModel] = None
        self.preprocessor = VideoPreprocessor(num_frames=8, height=128, width=128)
        self.text_preprocessor = TextPreprocessor(max_length=128)
        self.checkpoint_info: Dict[str, Any] = {}

        self._init_model()

    def is_available(self) -> bool:
        return self.model is not None

    def _find_default_checkpoint(self) -> Optional[str]:
        """Locates the latest trained checkpoint recursively across models and training directories."""
        # 1. Check models/lora
        lora_dir = settings.get_absolute_path("models/lora")
        if lora_dir.exists():
            for info_file in sorted(lora_dir.glob("*/version_info.json"), reverse=True):
                try:
                    with open(info_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    ckpt_p = data.get("checkpoint_path")
                    if ckpt_p and os.path.exists(ckpt_p):
                        return ckpt_p
                except Exception:
                    pass
            for p in sorted(lora_dir.glob("**/adapter_weights.pt"), reverse=True):
                if p.is_file():
                    return str(p)
            for p in sorted(lora_dir.glob("**/*.pt"), reverse=True):
                if p.is_file():
                    return str(p)

        # 2. Check training/checkpoints
        ckpt_dir = settings.get_absolute_path("training/checkpoints")
        if ckpt_dir.exists():
            # Prefer real trained jobs over smoke tests
            for p in sorted(ckpt_dir.glob("**/final_model.pt"), reverse=True):
                if "smoke" not in str(p) and p.is_file():
                    return str(p)
            for p in sorted(ckpt_dir.glob("**/final_model.pt"), reverse=True):
                if p.is_file():
                    return str(p)
            for p in sorted(ckpt_dir.glob("**/*.pt"), reverse=True):
                if p.is_file():
                    return str(p)
        return None

    def _init_model(self):
        target_ckpt = self.checkpoint_path or self._find_default_checkpoint()
        if not target_ckpt or not os.path.exists(target_ckpt):
            logger.warning(f"[NeuralVideoAdapter] No checkpoint found at '{target_ckpt}'.")
            self.model = None
            return

        self.checkpoint_path = target_ckpt
        try:
            ckpt = torch.load(target_ckpt, map_location=self.device)
            sd = ckpt.get("model_state_dict", ckpt)

            # Auto-detect architecture parameters from checkpoint state_dict
            latent_channels = 64
            if "spatial_encoder.3.weight" in sd:
                latent_channels = sd["spatial_encoder.3.weight"].shape[0]

            vocab_size = 10000
            if "text_encoder.token_embeddings.weight" in sd:
                vocab_size = sd["text_encoder.token_embeddings.weight"].shape[0]

            max_prompt_length = 128
            if "text_encoder.pos_embeddings" in sd:
                max_prompt_length = sd["text_encoder.pos_embeddings"].shape[1]

            num_frames = 8
            if "temporal_pos_embed" in sd:
                num_frames = sd["temporal_pos_embed"].shape[1]

            to_k_key = "temp_block1.to_k.base_layer.weight" if "temp_block1.to_k.base_layer.weight" in sd else "temp_block1.to_k.weight"
            text_embed_dim = 128
            if to_k_key in sd:
                text_embed_dim = sd[to_k_key].shape[1]

            lora_rank = ckpt.get("config", {}).get("lora_rank", 4)
            for k, v in sd.items():
                if "lora_A" in k:
                    lora_rank = v.shape[0]
                    break
            lora_alpha = ckpt.get("config", {}).get("lora_alpha", float(lora_rank * 2))

            # Instantiate model architecture matching checkpoint
            self.model = SpatialTemporalTTVModel(
                in_channels=3,
                latent_channels=latent_channels,
                num_frames=num_frames,
                vocab_size=vocab_size,
                max_prompt_length=max_prompt_length,
                text_embed_dim=text_embed_dim
            ).to(self.device)

            use_lora = ckpt.get("use_lora", True) or any("lora_" in k for k in sd.keys())
            if use_lora:
                apply_lora_to_model(self.model, rank=lora_rank, alpha=lora_alpha)

            # Load weights
            res = self.model.load_state_dict(sd, strict=False)
            missing_keys = res.missing_keys
            unexpected_keys = res.unexpected_keys

            if "lora_state_dict" in ckpt:
                load_lora_state_dict(self.model, ckpt["lora_state_dict"])

            self.model = self.model.to(self.device)
            self.model.eval()

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            self.checkpoint_info = {
                "base_model": "SpatialTemporalTTVModel",
                "checkpoint": str(target_ckpt),
                "use_lora": use_lora,
                "lora_rank": lora_rank,
                "loaded_parameters": total_params,
                "trainable_parameters": trainable_params,
                "missing_keys": len(missing_keys),
                "unexpected_keys": len(unexpected_keys),
                "device": str(self.device),
                "latent_channels": latent_channels,
                "text_embed_dim": text_embed_dim,
                "num_frames": num_frames
            }

            logger.info(
                f"[NeuralVideoAdapter] Successfully loaded checkpoint: {target_ckpt} | "
                f"Params: {total_params:,} | Trainable: {trainable_params:,} | Missing keys: {len(missing_keys)}"
            )
        except Exception as e:
            logger.error(f"[NeuralVideoAdapter] Error loading checkpoint {target_ckpt}: {e}")
            self.model = None
            raise VisualGenerationError(f"Neural checkpoint loading failed: {e}")

    async def generate_scene_video(
        self,
        scene: Scene,
        keyframe_path: str,
        output_path: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 24
    ) -> str:
        if self.model is None:
            raise VisualGenerationError(
                "NeuralVideoAdapter: No trained model checkpoint is loaded. "
                "Fake fallback generation is disabled. Train or provide a valid checkpoint."
            )

        telemetry.emit("neural_video_generation", scene.title, {
            "engine": "spatial_temporal_lora",
            "device": str(self.device),
            "checkpoint": self.checkpoint_info.get("checkpoint")
        })
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if not os.path.exists(keyframe_path):
            raise VisualGenerationError(f"Keyframe image missing: {keyframe_path}")

        img_bgr = cv2.imread(keyframe_path)
        if img_bgr is None:
            raise VisualGenerationError(f"Could not load keyframe image: {keyframe_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1. Preprocess keyframe to model input resolution
        resized_kf = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        kf_tensor = torch.from_numpy(
            np.transpose((resized_kf.astype(np.float32) / 127.5) - 1.0, (2, 0, 1))
        ).unsqueeze(0).float().to(self.device)

        # 2. Tokenize prompt
        prompt_text = scene.metadata.get("enriched_prompt", scene.visual_description)
        tokens = self.text_preprocessor.tokenize_to_tensor(prompt_text).unsqueeze(0).to(self.device)

        # 3. Model inference
        with torch.no_grad():
            pred_frames, _, _ = self.model(kf_tensor, tokens)
            pred_frames = pred_frames.squeeze(0) # (T, C, H, W)

        # 4. Denormalize frames
        model_frames = [
            self.preprocessor.denormalize_tensor(pred_frames[t])
            for t in range(pred_frames.shape[0])
        ]

        # 5. Temporal interpolation to match requested scene duration
        total_output_frames = max(12, int(scene.duration * fps))
        indices = np.linspace(0, len(model_frames) - 1, total_output_frames)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
            if not out.isOpened():
                raise VisualGenerationError(f"Failed to open VideoWriter for {output_path}")

        for idx_float in indices:
            low_idx = int(np.floor(idx_float))
            high_idx = min(len(model_frames) - 1, int(np.ceil(idx_float)))
            alpha = idx_float - low_idx

            frame_a = model_frames[low_idx]
            frame_b = model_frames[high_idx]
            interp = cv2.addWeighted(frame_a, 1.0 - alpha, frame_b, alpha, 0.0)

            # Upscale to requested target resolution
            upscaled = cv2.resize(interp, (width, height), interpolation=cv2.INTER_LANCZOS4)
            out_bgr = cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR)
            out.write(out_bgr)

        out.release()
        return output_path
