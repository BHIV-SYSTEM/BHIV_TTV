import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import DataLoader

from training.fine_tuning.model import SpatialTemporalTTVModel
from training.fine_tuning.lora import apply_lora_to_model, get_lora_state_dict, load_lora_state_dict
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
    Real Fine-Tuning Trainer for Text-to-Video models.
    Supports:
    - Full fine-tuning or LoRA parameter-efficient fine-tuning
    - Gradient accumulation
    - Checkpoint saving & resuming
    - Validation loop
    - Model registry integration
    """
    def __init__(
        self,
        model: SpatialTemporalTTVModel,
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
        self.grad_accum_steps = int(self.config.get("gradient_accumulation_steps", 1))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        self.use_lora = bool(self.config.get("use_lora", True))
        self.lora_rank = int(self.config.get("lora_rank", 4))
        self.lora_alpha = float(self.config.get("lora_alpha", 8.0))

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry()

        # Apply LoRA if requested
        if self.use_lora:
            apply_lora_to_model(self.model, rank=self.lora_rank, alpha=self.lora_alpha)
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        else:
            trainable_params = list(self.model.parameters())

        self.trainable_params_count = sum(p.numel() for p in trainable_params)
        self.total_params_count = sum(p.numel() for p in self.model.parameters())

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

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses = []
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            frames = batch["frames"].to(self.device)  # (B, T, C, H, W)
            tokens = batch["prompt_tokens"].to(self.device)  # (B, L)

            # Keyframe is the initial frame (index 0)
            keyframe = frames[:, 0, :, :, :]

            # Forward pass
            pred_frames, video_emb, text_emb = self.model(keyframe, tokens)

            # Compute loss
            loss, metrics = self.criterion(pred_frames, frames, video_emb, text_emb)
            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            epoch_losses.append(metrics["loss_total"])

        self.scheduler.step()
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        if self.val_loader is None or len(self.val_loader) == 0:
            return {"val_loss": 0.0}

        self.model.eval()
        val_losses = []
        for batch in self.val_loader:
            frames = batch["frames"].to(self.device)
            tokens = batch["prompt_tokens"].to(self.device)
            keyframe = frames[:, 0, :, :, :]

            pred_frames, video_emb, text_emb = self.model(keyframe, tokens)
            loss, metrics = self.criterion(pred_frames, frames, video_emb, text_emb)
            val_losses.append(metrics["loss_total"])

        avg_val = float(np.mean(val_losses)) if val_losses else 0.0
        return {"val_loss": avg_val}

    def train(self) -> Dict[str, Any]:
        """Runs complete training loop over specified epochs."""
        print(f"[Training] Starting training on device: {self.device}")
        print(f"[Training] Trainable parameters: {self.trainable_params_count:,} / {self.total_params_count:,} ({100 * self.trainable_params_count / self.total_params_count:.2f}%)")

        start_time = time.time()
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()

            combined = {
                "epoch": epoch + 1,
                "global_step": self.global_step,
                **train_metrics,
                **val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"]
            }
            self.history.append(combined)

            print(f"  Epoch [{epoch+1}/{self.epochs}] Step {self.global_step} - Train Loss: {train_metrics['train_loss']:.4f} - Val Loss: {val_metrics['val_loss']:.4f}")

            # Save checkpoint if best or at end
            if val_metrics["val_loss"] < self.best_val_loss or (epoch + 1) == self.epochs:
                self.best_val_loss = val_metrics["val_loss"]
                checkpoint = self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", metrics=combined)
                print(f"Checkpoint saved: {checkpoint}")

        elapsed = time.time() - start_time
        final_ckpt = self.save_checkpoint("final_model.pt", metrics={"history": self.history, "elapsed_s": elapsed})
        print(f"Checkpoint saved: {final_ckpt}")
        return {
            "elapsed_seconds": round(elapsed, 2),
            "final_checkpoint": final_ckpt,
            "best_val_loss": self.best_val_loss,
            "history": self.history
        }

    def save_checkpoint(self, filename: str, metrics: Optional[Dict[str, Any]] = None) -> str:
        """Saves weights, optimizer state, and training metadata."""
        ckpt_path = self.checkpoint_dir / filename
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "use_lora": self.use_lora,
            "config": self.config,
            "metrics": metrics or {}
        }
        if self.use_lora:
            state["lora_state_dict"] = get_lora_state_dict(self.model)

        torch.save(state, str(ckpt_path))
        return str(ckpt_path.resolve())

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resumes training state from existing checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("global_step", 0)
        print(f"[Training] Successfully resumed from {checkpoint_path} (epoch {self.current_epoch})")
