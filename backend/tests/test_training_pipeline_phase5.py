"""
Test suite for Phase 5: 10 GB VRAM-Optimized Video Model LoRA Training Pipeline.
Verifies hardware detection, VRAM budgeting, LoRA module application,
gradient checkpointing, dataset loader eligibility gating, checkpoint saving & resuming,
preflight verification system, and training API endpoints.
"""
from __future__ import annotations

import os
import json
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import WORKSPACE_ROOT
from training.utils.hardware import detect_hardware, estimate_training_vram
from training.fine_tuning.model import SpatialTemporalTTVModel
from training.fine_tuning.lora import (
    apply_lora_to_model,
    get_lora_state_dict,
    load_lora_state_dict,
    get_trainable_parameters,
)
from training.fine_tuning.trainer import TTVTrainer
from training.datasets.loader import TextToVideoDataset, collate_ttv_batch
from training.preflight import run_training_preflight, TrainingPreflightChecker

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Hardware Detection & VRAM Inspection Tests
# ---------------------------------------------------------------------------
def test_hardware_detection_and_vram_inspection():
    """Verify runtime hardware detection reports valid system, CUDA, and VRAM metrics."""
    hw = detect_hardware()
    assert "os" in hw
    assert "cpu_cores_logical" in hw
    assert "ram_total_gb" in hw
    assert "disk_free_gb" in hw
    assert "cuda_available" in hw
    assert "vram_total_gb" in hw
    assert "vram_free_gb" in hw

    if hw["cuda_available"]:
        assert hw["vram_total_gb"] > 0.0
        assert hw["gpu_name"] is not None
        assert "supports_fp16" in hw
        assert "supports_bf16" in hw


def test_vram_budget_estimation():
    """Verify theoretical VRAM budget calculation for 10 GB hardware."""
    config = {
        "model": {"name": "SpatialTemporalTTVModel", "num_frames": 8},
        "training": {
            "use_lora": True,
            "lora_rank": 4,
            "precision": "bfloat16",
            "batch_size": 1,
            "gradient_checkpointing": True,
        },
        "preprocessing": {"height": 128, "width": 128, "num_frames": 8},
    }
    budget = estimate_training_vram("SpatialTemporalTTVModel", config)
    assert budget["model"] == "SpatialTemporalTTVModel"
    assert budget["fits_in_10gb"] is True
    assert budget["total_estimated_gb"] < 3.0
    assert budget["headroom_gb"] > 7.0

    # Test CogVideoX-2B estimate
    cog_budget = estimate_training_vram("CogVideoX-2b", config)
    assert cog_budget["model"] == "CogVideoX-2b"
    assert cog_budget["base_weights_gb"] >= 4.4


# ---------------------------------------------------------------------------
# 2. LoRA Application & Parameter Freezing Tests
# ---------------------------------------------------------------------------
def test_lora_application_and_parameter_freezing():
    """Verify LoRA replaces target layers, freezes base model, and drastically cuts trainable params."""
    model = SpatialTemporalTTVModel(
        in_channels=3,
        latent_channels=64,
        num_frames=8,
        vocab_size=1000,
        max_prompt_length=16,
        text_embed_dim=64,
    )

    # Initial state: inspect trainable parameters
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_before > 0

    # Apply LoRA rank 4
    apply_lora_to_model(model, rank=4, alpha=16.0, dropout=0.05)

    trainable_after, total_after, ratio = get_trainable_parameters(model)
    assert trainable_after < trainable_before
    assert ratio < 20.0, f"Expected trainable ratio < 20%, got {ratio:.2f}%"

    # Base parameters must be frozen
    assert model.spatial_encoder[0].weight.requires_grad is False
    assert model.decoder[0].weight.requires_grad is False

    # LoRA parameters must be trainable
    lora_dict = get_lora_state_dict(model)
    assert len(lora_dict) > 0
    for name, tensor in lora_dict.items():
        assert "lora_" in name
        assert tensor.numel() > 0


# ---------------------------------------------------------------------------
# 3. Gradient Checkpointing Tests
# ---------------------------------------------------------------------------
def test_gradient_checkpointing_in_model():
    """Verify gradient checkpointing can be enabled and executes in forward pass."""
    model = SpatialTemporalTTVModel(
        in_channels=3,
        latent_channels=32,
        num_frames=4,
        vocab_size=500,
        max_prompt_length=8,
        text_embed_dim=32,
    )
    assert model.gradient_checkpointing is False
    model.enable_gradient_checkpointing()
    assert model.gradient_checkpointing is True

    model.train()
    dummy_keyframe = torch.randn(1, 3, 64, 64, requires_grad=True)
    dummy_tokens = torch.randint(0, 500, (1, 8))

    out, v_emb, t_emb = model(dummy_keyframe, dummy_tokens)
    assert out.shape == (1, 4, 3, 64, 64)
    assert v_emb.shape == (1, 32)
    assert t_emb.shape == (1, 32)

    model.disable_gradient_checkpointing()
    assert model.gradient_checkpointing is False


# ---------------------------------------------------------------------------
# 4. Dataset Loader Eligibility Gating Tests
# ---------------------------------------------------------------------------
def test_dataset_loader_filters_ineligible_samples(tmp_path: Path):
    """Verify Dataset loader excludes restricted, duplicate, and invalid manifest items."""
    import cv2
    import numpy as np

    manifest_file = tmp_path / "test_manifest.jsonl"
    dummy_video = tmp_path / "sample.mp4"
    
    # Write a small valid 4-frame mp4 video file
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dummy_video), fourcc, 10.0, (64, 64))
    for _ in range(4):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    records = [
        # Valid item (should be loaded)
        {
            "id": "item_valid_1",
            "video_path": str(dummy_video),
            "caption": "A valid clip description",
            "training_eligible": True,
            "permission_state": "LICENSE_VERIFIED",
            "validation_status": "VALID",
            "split": "train",
        },
        # Restricted license item (must be rejected)
        {
            "id": "item_restricted_2",
            "video_path": str(dummy_video),
            "caption": "A commercial film trailer",
            "training_eligible": False,
            "permission_state": "RESTRICTED",
            "validation_status": "NEEDS_REVIEW",
            "split": "train",
        },
        # Duplicate item (must be rejected)
        {
            "id": "item_dup_3",
            "video_path": str(dummy_video),
            "caption": "Duplicate caption",
            "training_eligible": False,
            "permission_state": "LICENSE_VERIFIED",
            "validation_status": "DUPLICATE",
            "split": "train",
        },
    ]

    with open(manifest_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    dataset = TextToVideoDataset(str(manifest_file), num_frames=4, height=64, width=64)
    # Only the 1 valid record should be loaded
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["id"] == "item_valid_1"


# ---------------------------------------------------------------------------
# 5. Checkpoint Saving & Resuming Tests
# ---------------------------------------------------------------------------
def test_checkpoint_saving_and_resuming(tmp_path: Path):
    """Verify that LoRA checkpoints can be saved and accurately resumed without corruption."""
    model = SpatialTemporalTTVModel(
        in_channels=3,
        latent_channels=32,
        num_frames=4,
        vocab_size=500,
        max_prompt_length=8,
        text_embed_dim=32,
    )

    dummy_loader = [
        {
            "frames": torch.randn(1, 4, 3, 32, 32),
            "prompt_tokens": torch.randint(0, 500, (1, 8)),
        }
    ]

    trainer = TTVTrainer(
        model=model,
        train_loader=dummy_loader,
        config={"device": "cpu", "epochs": 2, "learning_rate": 0.001, "use_lora": True},
        checkpoint_dir=str(tmp_path),
    )

    # Save checkpoint
    saved_path = trainer.save_checkpoint("test_checkpoint.pt", metrics={"train_loss": 0.45})
    assert Path(saved_path).exists()

    # Verify saved state contains expected keys
    data = torch.load(saved_path, map_location="cpu")
    assert "model_state_dict" in data
    assert "lora_state_dict" in data
    assert "optimizer_state_dict" in data
    assert "config" in data

    # Resume into fresh trainer
    fresh_model = SpatialTemporalTTVModel(
        in_channels=3,
        latent_channels=32,
        num_frames=4,
        vocab_size=500,
        max_prompt_length=8,
        text_embed_dim=32,
    )

    resumer = TTVTrainer(
        model=fresh_model,
        train_loader=dummy_loader,
        config={"device": "cpu", "use_lora": True},
        checkpoint_dir=str(tmp_path),
    )
    resumer.resume_from_checkpoint(saved_path)
    assert resumer.current_epoch == 1  # 0-indexed was 0, so resume at 1


# ---------------------------------------------------------------------------
# 6. Preflight & Memory Feasibility Check System Tests
# ---------------------------------------------------------------------------
def test_preflight_check_ready_status():
    """Verify preflight check runs against 10 GB config and returns READY or READY_WITH_LIMITATIONS."""
    config_path = "training/configs/lora_10gb_rtx3080.yaml"
    result = run_training_preflight(config_path=config_path, dry_run=False)

    assert result["status"] in ("READY", "READY_WITH_LIMITATIONS")
    assert len(result["checklist"]) >= 15
    assert len(result["errors"]) == 0
    assert result["feasibility"]["budget"]["fits_in_10gb"] is True


def test_preflight_detects_invalid_manifest():
    """Verify preflight check returns NOT_READY when an invalid manifest path is supplied."""
    config = {
        "model": {"name": "SpatialTemporalTTVModel"},
        "training": {"batch_size": 1, "use_lora": True, "gradient_checkpointing": True},
        "paths": {"dataset_manifest": "non_existent_manifest_file_xyz.jsonl"},
    }
    result = run_training_preflight(config=config, manifest_path="non_existent_manifest_file_xyz.jsonl")
    assert result["status"] == "NOT_READY"
    assert len(result["errors"]) > 0
    assert any("manifest" in e.lower() for e in result["errors"])


# ---------------------------------------------------------------------------
# 7. Training API Endpoints Verification Tests
# ---------------------------------------------------------------------------
def test_training_hardware_endpoint():
    """Verify GET /api/v1/training/hardware returns hardware profile."""
    response = client.get("/api/v1/training/hardware")
    assert response.status_code == 200
    data = response.json()
    assert "cpu_processor" in data
    assert "ram_total_gb" in data
    assert "disk_free_gb" in data
    assert "cuda_available" in data


def test_training_preflight_endpoint():
    """Verify POST /api/v1/training/preflight returns structured readiness evaluation."""
    payload = {
        "manifest_path": "dataset/manifests/openvid_1m_prepared_manifest_train.jsonl",
        "dry_run": False,
        "config": {
            "model": {"name": "SpatialTemporalTTVModel", "num_frames": 8},
            "training": {
                "method": "lora",
                "use_lora": True,
                "lora_rank": 4,
                "lora_alpha": 16.0,
                "batch_size": 1,
                "gradient_accumulation_steps": 4,
                "gradient_checkpointing": True,
                "epochs": 1,
            },
            "preprocessing": {"height": 128, "width": 128, "num_frames": 8},
        },
    }
    response = client.post("/api/v1/training/preflight", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("READY", "READY_WITH_LIMITATIONS")
    assert "checklist" in data
    assert "feasibility" in data
    assert data["feasibility"]["budget"]["fits_in_10gb"] is True


def test_training_checkpoints_list_endpoint():
    """Verify GET /api/v1/training/checkpoints lists saved checkpoints."""
    response = client.get("/api/v1/training/checkpoints")
    assert response.status_code == 200
    data = response.json()
    assert "checkpoints" in data
    assert "total" in data
    assert isinstance(data["checkpoints"], list)
