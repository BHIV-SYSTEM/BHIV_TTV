import os
import gc
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import DataLoader

from training.fine_tuning.model import SpatialTemporalTTVModel
from training.fine_tuning.lora import apply_lora_to_model, get_lora_state_dict, load_lora_state_dict, get_trainable_parameters
from training.utils.versioning import ModelRegistry


def set_reproducible_seed(seed: int = 42):
    """Sets seeds across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TTVLoss(nn.Module):
    """
    Multi-objective Loss for Text-to-Video Fine-Tuning:
    1. Spatial Reconstruction Loss (L1/MSE)
    2. Temporal Continuity & Motion Loss (First-order difference matching)
    3. Semantic Text-Video Alignment Loss (Cosine similarity)
    """
    def __init__(self, lambda_temp: float = 0.5, lambda_align: float = 0.2):
        super().__init__()
        self.lambda_temp = lambda_temp
        self.lambda_align = lambda_align

    def forward(
        self,
        pred_frames: torch.Tensor,
        target_frames: torch.Tensor,
        video_emb: torch.Tensor,
        text_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Spatial reconstruction loss (L1 is sharper than MSE)
        loss_recon = F.l1_loss(pred_frames, target_frames)

        # 2. Temporal motion continuity loss: match frame-to-frame velocity vectors
        if pred_frames.shape[1] > 1:
            pred_diff = pred_frames[:, 1:] - pred_frames[:, :-1]
            target_diff = target_frames[:, 1:] - target_frames[:, :-1]
            loss_temp = F.mse_loss(pred_diff, target_diff)
        else:
            loss_temp = torch.tensor(0.0, device=pred_frames.device)

        # 3. Semantic prompt alignment loss (1 - cosine_similarity)
        cos_sim = F.cosine_similarity(video_emb, text_emb, dim=-1)
        loss_align = (1.0 - cos_sim).mean()

        total_loss = loss_recon + self.lambda_temp * loss_temp + self.lambda_align * loss_align

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_recon": float(loss_recon.item()),
            "loss_temporal": float(loss_temp.item()),
            "loss_alignment": float(loss_align.item())
        }
        return total_loss, metrics


class TTVTrainer:
    """
    Memory-Optimized Fine-Tuning Trainer for 10 GB VRAM Video Models.
    Features:
    - Low-Rank Adaptation (LoRA) on attention and convolution layers
    - Gradient checkpointing
    - Mixed precision (native bfloat16 or float16 with GradScaler)
    - Micro-batching with gradient accumulation
    - Peak VRAM telemetry and CUDA cache clearing
    - Graceful Out-Of-Memory (OOM) handling
    - Periodic and final checkpoint saving with resume capability
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: str = "training/checkpoints"
    ):
        self.config = config or {}
        self.seed = self.config.get("seed", 42)
        set_reproducible_seed(self.seed)

        self.device = torch.device(
            "cuda" if (torch.cuda.is_available() and self.config.get("device") != "cpu") else "cpu"
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training Hyperparameters
        self.learning_rate = float(self.config.get("learning_rate", 1e-4))
        self.weight_decay = float(self.config.get("weight_decay", 1e-2))
        self.epochs = int(self.config.get("epochs", 5))
        self.grad_accum_steps = max(1, int(self.config.get("gradient_accumulation_steps", 4)))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        self.use_lora = bool(self.config.get("use_lora", True))
        self.lora_rank = int(self.config.get("lora_rank", 4))
        self.lora_alpha = float(self.config.get("lora_alpha", 8.0))
        self.lora_dropout = float(self.config.get("lora_dropout", 0.0))
        self.target_modules = self.config.get("target_modules", ["to_q", "to_k", "to_v", "proj", "temporal_conv", "temporal_proj"])

        # Gradient Checkpointing
        self.use_grad_checkpointing = bool(self.config.get("gradient_checkpointing", True))
        if self.use_grad_checkpointing and hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing()

        # Mixed Precision Setup
        self.precision = str(self.config.get("precision", "bfloat16")).lower()
        self.scaler = None
        self.autocast_dtype = None

        if self.device.type == "cuda":
            if self.precision in ("bfloat16", "bf16") and torch.cuda.is_bf16_supported():
                self.autocast_dtype = torch.bfloat16
            elif self.precision in ("float16", "fp16"):
                self.autocast_dtype = torch.float16
                self.scaler = torch.amp.GradScaler('cuda')

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry()

        # Apply LoRA if requested
        if self.use_lora:
            apply_lora_to_model(
                self.model,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                target_names=self.target_modules
            )
            self.model = self.model.to(self.device)
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            trainable_params = list(self.model.parameters())

        self.trainable_params_count, self.total_params_count, self.trainable_ratio = get_trainable_parameters(self.model)

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.epochs)
        )
        self.criterion = TTVLoss(
            lambda_temp=float(self.config.get("lambda_temp", 0.5)),
            lambda_align=float(self.config.get("lambda_align", 0.2))
        )

        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.history: List[Dict[str, Any]] = []

    def _get_vram_telemetry(self) -> Dict[str, float]:
        """Captures real-time CUDA memory utilization."""
        if self.device.type != "cuda":
            return {"vram_allocated_mb": 0.0, "vram_reserved_mb": 0.0, "vram_peak_mb": 0.0}
        return {
            "vram_allocated_mb": round(torch.cuda.memory_allocated(self.device) / (1024**2), 2),
            "vram_reserved_mb": round(torch.cuda.memory_reserved(self.device) / (1024**2), 2),
            "vram_peak_mb": round(torch.cuda.max_memory_allocated(self.device) / (1024**2), 2),
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses = []
        self.optimizer.zero_grad(set_to_none=True)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                frames = batch["frames"].to(self.device, non_blocking=True)  # (B, T, C, H, W)
                tokens = batch["prompt_tokens"].to(self.device, non_blocking=True)  # (B, L)
                keyframe = frames[:, 0, :, :, :]

                # Forward pass under autocast when enabled
                if self.autocast_dtype is not None:
                    with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                        pred_frames, video_emb, text_emb = self.model(keyframe, tokens)
                        loss, metrics = self.criterion(pred_frames, frames, video_emb, text_emb)
                        scaled_loss = loss / self.grad_accum_steps
                else:
                    pred_frames, video_emb, text_emb = self.model(keyframe, tokens)
                    loss, metrics = self.criterion(pred_frames, frames, video_emb, text_emb)
                    scaled_loss = loss / self.grad_accum_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # Step optimizer on gradient accumulation boundary
                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    if self.scaler is not None:
                        if self.max_grad_norm > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                epoch_losses.append(metrics["loss_total"])

            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" in str(exc).lower():
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                        gc.collect()
                    telemetry = self._get_vram_telemetry()
                    raise RuntimeError(
                        f"CUDA Out-of-Memory during training epoch {epoch+1}, step {batch_idx+1}. "
                        f"Allocated: {telemetry['vram_allocated_mb']}MB, Peak: {telemetry['vram_peak_mb']}MB. "
                        f"Recommendation: Enable gradient_checkpointing=True, reduce batch_size to 1, or lower resolution."
                    ) from exc
                raise

        self.scheduler.step()
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return {"train_loss": avg_loss}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {"val_loss": 0.0}

        self.model.eval()
        val_losses = []
        for batch in self.val_loader:
            frames = batch["frames"].to(self.device, non_blocking=True)
            tokens = batch["prompt_tokens"].to(self.device, non_blocking=True)
            keyframe = frames[:, 0, :, :, :]

            if self.autocast_dtype is not None:
                with torch.amp.autocast('cuda', dtype=self.autocast_dtype):
                    pred_frames, video_emb, text_emb = self.model(keyframe, tokens)
                    loss, metrics = self.criterion(pred_frames, frames, video_emb, text_emb)
            else:
                pred_frames, video_emb, text_emb = self.model(keyframe, tokens)
                loss, metrics = self.criterion(pred_frames, frames, video_emb, text_emb)

            val_losses.append(metrics["loss_total"])

        avg_val = float(np.mean(val_losses)) if val_losses else 0.0
        return {"val_loss": avg_val}

    def train(self) -> Dict[str, Any]:
        """Runs complete training loop over specified epochs."""
        telemetry = self._get_vram_telemetry()
        print(f"[Training] Starting training on device: {self.device} (Precision: {self.precision})")
        print(f"[Training] Trainable parameters: {self.trainable_params_count:,} / {self.total_params_count:,} ({self.trainable_ratio:.2f}%)")
        print(f"[Training] Gradient Checkpointing: {self.use_grad_checkpointing} | Accumulation Steps: {self.grad_accum_steps}")
        if self.device.type == "cuda":
            print(f"[Training] Initial VRAM: {telemetry['vram_allocated_mb']}MB allocated | {telemetry['vram_reserved_mb']}MB reserved")

        start_time = time.time()
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            vram_stats = self._get_vram_telemetry()

            combined = {
                "epoch": epoch + 1,
                "global_step": self.global_step,
                **train_metrics,
                **val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
                **vram_stats,
            }
            self.history.append(combined)

            print(
                f"  Epoch [{epoch+1}/{self.epochs}] Step {self.global_step} - "
                f"Train Loss: {train_metrics['train_loss']:.4f} - Val Loss: {val_metrics['val_loss']:.4f} - "
                f"Peak VRAM: {vram_stats['vram_peak_mb']}MB"
            )

            # Save checkpoint if best or at end
            if val_metrics["val_loss"] < self.best_val_loss or (epoch + 1) == self.epochs:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", metrics=combined)

        elapsed = time.time() - start_time
        final_ckpt = self.save_checkpoint("final_model.pt", metrics={"history": self.history, "elapsed_s": elapsed})
        return {
            "elapsed_seconds": round(elapsed, 2),
            "final_checkpoint": final_ckpt,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "telemetry": self._get_vram_telemetry(),
        }

    def save_checkpoint(self, filename: str, metrics: Optional[Dict[str, Any]] = None) -> str:
        """Saves weights, optimizer state, LoRA adapter weights, and training metadata."""
        ckpt_path = self.checkpoint_dir / filename
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "precision": self.precision,
            "config": self.config,
            "metrics": metrics or {},
            "telemetry": self._get_vram_telemetry(),
        }
        if self.use_lora:
            state["lora_state_dict"] = get_lora_state_dict(self.model)

        torch.save(state, str(ckpt_path))

        # Also save standalone LoRA weights for portable inference
        if self.use_lora and filename == "final_model.pt":
            lora_only_path = self.checkpoint_dir / "final_lora_adapter.pt"
            torch.save({
                "lora_state_dict": get_lora_state_dict(self.model),
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "target_modules": self.target_modules,
            }, str(lora_only_path))

        return str(ckpt_path.resolve())

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resumes training state from existing checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("global_step", 0)
        print(f"[Training] Successfully resumed from {checkpoint_path} (resuming at epoch {self.current_epoch + 1})")
