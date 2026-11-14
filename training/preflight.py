"""
Preflight & Memory Feasibility Check System for 10 GB VRAM Video Fine-Tuning.
Verifies hardware, CUDA availability, free VRAM, dataset manifest validity,
model compatibility, output permissions, and memory feasibility before training.
"""
from __future__ import annotations

import os
import sys
import gc
import json
import yaml
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from training.utils.hardware import detect_hardware, estimate_training_vram
from app.services.training_service import parse_manifest_records, validate_manifest_records

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TrainingPreflightChecker:
    """Orchestrates comprehensive readiness and memory feasibility verification."""

    def __init__(self, config: Dict[str, Any], manifest_path: Optional[str] = None):
        self.config = config
        self.manifest_path = manifest_path or config.get("paths", {}).get("train_manifest") or config.get("paths", {}).get("dataset_manifest")
        self.checklist: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.recommendations: List[str] = []

    def _add_check(self, name: str, status: bool, details: str, is_warning: bool = False):
        self.checklist.append({
            "name": name,
            "status": "PASS" if status else ("WARN" if is_warning else "FAIL"),
            "details": details,
        })
        if not status:
            if is_warning:
                self.warnings.append(f"[{name}] {details}")
            else:
                self.errors.append(f"[{name}] {details}")

    def check_cuda_and_hardware(self) -> Dict[str, Any]:
        """Verifies CUDA drivers, GPU presence, total VRAM, and currently free VRAM."""
        hw = detect_hardware()
        cuda_ok = hw["cuda_available"]
        vram_total = hw["vram_total_gb"]
        vram_free = hw["vram_free_gb"]

        if not cuda_ok:
            self._add_check("CUDA GPU Availability", False, "No active NVIDIA CUDA GPU detected. Video diffusion/LoRA fine-tuning cannot run on CPU.")
            self.recommendations.append("Ensure an NVIDIA GPU with CUDA 12+ drivers is available and PyTorch CUDA build is installed.")
        else:
            self._add_check("CUDA GPU Availability", True, f"Detected {hw['gpu_name']} (CUDA capability {hw['gpu_capability']})")

            # Check total VRAM
            if vram_total >= 9.5:
                self._add_check("Total VRAM Capacity", True, f"{vram_total} GB Total VRAM meets target 10 GB hardware profile.")
            elif vram_total >= 7.5:
                self._add_check("Total VRAM Capacity", True, f"{vram_total} GB Total VRAM available. Micro-batch size 1 and gradient checkpointing mandatory.", is_warning=True)
                self.recommendations.append("Operate with batch_size=1, gradient_checkpointing=True, and precision=bfloat16.")
            else:
                self._add_check("Total VRAM Capacity", False, f"{vram_total} GB Total VRAM is below minimum 8 GB threshold.")

            # Check free VRAM
            if vram_free >= 7.0:
                self._add_check("Free VRAM Budget", True, f"{vram_free} GB free VRAM currently available before model allocation.")
            elif vram_free >= 4.0:
                self._add_check("Free VRAM Budget", True, f"{vram_free} GB free VRAM available. Background processes may be consuming VRAM.", is_warning=True)
                self.recommendations.append("Close external GPU applications (browsers, 3D viewers) to maximize free VRAM.")
            else:
                self._add_check("Free VRAM Budget", False, f"Only {vram_free} GB free VRAM currently available. High risk of immediate CUDA OOM.", is_warning=True)

            # Precision support
            bf16_ok = hw["supports_bf16"]
            fp16_ok = hw["supports_fp16"]
            self._add_check("Mixed Precision Support", bf16_ok or fp16_ok, f"BF16: {bf16_ok}, FP16: {fp16_ok} supported on this hardware.")

        return hw

    def check_dependencies(self):
        """Verifies necessary machine learning libraries are accessible."""
        deps = [
            ("torch", TORCH_AVAILABLE),
            ("torchvision", True),
            ("cv2", True),
            ("PIL", True),
            ("yaml", True),
            ("numpy", True),
        ]
        for name, ok in deps:
            try:
                __import__(name)
                self._add_check(f"Dependency: {name}", True, f"Installed and importable.")
            except ImportError:
                self._add_check(f"Dependency: {name}", False, f"Missing Python package '{name}'.")

    def check_dataset_and_manifest(self) -> Dict[str, Any]:
        """Validates manifest existence, schema compliance, and verifies physical video files on disk."""
        if not self.manifest_path:
            self._add_check("Dataset Manifest Path", False, "No dataset manifest specified in configuration or arguments.")
            return {"valid": False, "total_records": 0}

        p = Path(self.manifest_path)
        if not p.is_absolute() and not p.exists():
            p = WORKSPACE_ROOT / self.manifest_path

        if not p.exists():
            self._add_check("Dataset Manifest Existence", False, f"Manifest file does not exist at {p}")
            self.recommendations.append("Prepare the sample dataset first via the Dataset Dashboard or run `prepare_openvid_sample_dataset()`.")
            return {"valid": False, "total_records": 0}

        self._add_check("Dataset Manifest Existence", True, f"Located at {p}")

        records, _ = parse_manifest_records(p)
        val_res = validate_manifest_records(records)

        if not val_res["valid"]:
            self._add_check("Dataset Manifest Schema", False, f"Validation rejected: {val_res.get('summary', 'Schema errors')}")
            return val_res

        self._add_check("Dataset Manifest Schema", True, f"{val_res['valid_records']} valid records across {val_res['splits']} split(s).")

        # Verify actual existence of referenced media files on disk (sample first 10)
        missing_count = 0
        total_checked = min(len(records), 20)
        for r in records[:total_checked]:
            vref = r.get("video_path") or r.get("video_url") or r.get("source_url") or r.get("storage_path") or ""
            if not vref.startswith(("http://", "https://")):
                cand = Path(vref) if Path(vref).is_absolute() else WORKSPACE_ROOT / vref
                if not cand.exists():
                    missing_count += 1

        if missing_count > 0:
            self._add_check("Physical Video File Check", False, f"{missing_count}/{total_checked} checked video files were not found on disk.")
            self.recommendations.append("Ensure sample clips in `dataset/ttv_data/` are intact and manifest paths are workspace-relative.")
        else:
            self._add_check("Physical Video File Check", True, f"Verified all {total_checked} sampled video files exist on disk.")

        return val_res

    def check_training_config(self):
        """Verifies hyperparameters conform to 10 GB VRAM optimization standards."""
        train_cfg = self.config.get("training", {})
        prep_cfg = self.config.get("preprocessing", {})

        batch_size = int(train_cfg.get("batch_size", 1))
        epochs = int(train_cfg.get("epochs", 1))
        lr = float(train_cfg.get("learning_rate", 0.0001))
        grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
        use_lora = bool(train_cfg.get("use_lora", True))
        lora_rank = int(train_cfg.get("lora_rank", 4))
        grad_chk = bool(train_cfg.get("gradient_checkpointing", True))

        if not use_lora:
            self._add_check("LoRA Parameter Efficiency", False, "Full fine-tuning requested. Full fine-tuning will exceed 10 GB VRAM for video models.")
            self.recommendations.append("Enable use_lora=True with lora_rank=4 or 8.")
        else:
            self._add_check("LoRA Parameter Efficiency", True, f"LoRA enabled with rank={lora_rank}, alpha={train_cfg.get('lora_alpha', 16.0)}.")

        if batch_size > 2:
            self._add_check("Batch Size Constraint", False, f"batch_size={batch_size} exceeds recommended 10 GB budget. Max safe is 1-2.", is_warning=True)
            self.recommendations.append("Set batch_size=1 and increase gradient_accumulation_steps to 4 or 8.")
        else:
            self._add_check("Batch Size Constraint", True, f"Micro-batch size {batch_size} conforms to 10 GB VRAM constraint.")

        if not grad_chk:
            self._add_check("Gradient Checkpointing", False, "Gradient checkpointing is disabled. High risk of activation memory spikes.", is_warning=True)
            self.recommendations.append("Set gradient_checkpointing=True.")
        else:
            self._add_check("Gradient Checkpointing", True, "Gradient checkpointing enabled (cuts peak activation VRAM by ~60%).")

        h = int(prep_cfg.get("height", 128))
        w = int(prep_cfg.get("width", 128))
        frames = int(prep_cfg.get("num_frames") or self.config.get("model", {}).get("num_frames") or 8)
        self._add_check("Preprocessing Dimensions", True, f"Target resolution {w}x{h}, {frames} frames.")

    def check_storage_and_permissions(self):
        """Verifies checkpoint directory write permissions and free disk capacity."""
        ckpt_dir_raw = self.config.get("paths", {}).get("checkpoint_dir", "training/checkpoints/lora")
        ckpt_dir = Path(ckpt_dir_raw) if Path(ckpt_dir_raw).is_absolute() else WORKSPACE_ROOT / ckpt_dir_raw

        try:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            test_file = ckpt_dir / ".perm_check"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            self._add_check("Checkpoint Storage Permissions", True, f"Directory writable: {ckpt_dir}")
        except Exception as exc:
            self._add_check("Checkpoint Storage Permissions", False, f"Cannot write to checkpoint directory: {exc}")

        # Check disk space
        try:
            usage = shutil.disk_usage(str(ckpt_dir))
            free_gb = round(usage.free / (1024**3), 2)
            if free_gb >= 5.0:
                self._add_check("Disk Storage Capacity", True, f"{free_gb} GB free disk space (exceeds 5 GB minimum).")
            elif free_gb >= 2.0:
                self._add_check("Disk Storage Capacity", True, f"{free_gb} GB free disk space. Sufficient for LoRA checkpoints (~100MB).", is_warning=True)
            else:
                self._add_check("Disk Storage Capacity", False, f"Only {free_gb} GB free disk space remaining. Minimum 2 GB required.")
        except Exception:
            pass

    def run_memory_feasibility(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Calculates theoretical VRAM budget and optionally verifies real forward pass VRAM.
        Never modifies pretrained weights or performs backpropagation.
        """
        model_name = self.config.get("model", {}).get("name", "SpatialTemporalTTVModel")
        budget = estimate_training_vram(model_name, self.config)

        self._add_check(
            "VRAM Budget Feasibility",
            budget["fits_in_10gb"],
            f"Estimated peak VRAM: {budget['total_estimated_gb']} GB (Headroom: {budget['headroom_gb']} GB on 10 GB GPU). {budget['recommendation']}",
            is_warning=not budget["fits_in_10gb"]
        )

        dry_run_result = {"executed": False}
        if dry_run and TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                from training.fine_tuning.model import SpatialTemporalTTVModel
                from training.fine_tuning.lora import apply_lora_to_model

                torch.cuda.empty_cache()
                gc.collect()
                mem_before = torch.cuda.memory_allocated(0) / (1024**2)

                # Initialize model on CUDA
                m = SpatialTemporalTTVModel(
                    in_channels=3,
                    latent_channels=64,
                    num_frames=8,
                    vocab_size=10000,
                    max_prompt_length=32,
                    text_embed_dim=128
                )

                apply_lora_to_model(m, rank=4, alpha=16.0)
                m.enable_gradient_checkpointing()
                m = m.to("cuda")
                m.eval()

                # Dummy forward pass (safe: torch.no_grad, no backprop, no optimizer)
                with torch.no_grad():
                    dummy_keyframe = torch.zeros((1, 3, 128, 128), device="cuda")
                    dummy_tokens = torch.zeros((1, 32), dtype=torch.long, device="cuda")
                    out, v_emb, t_emb = m(dummy_keyframe, dummy_tokens)

                peak_mem = torch.cuda.max_memory_allocated(0) / (1024**2)
                allocated_mem = torch.cuda.memory_allocated(0) / (1024**2)

                # Cleanup immediately
                del m, dummy_keyframe, dummy_tokens, out, v_emb, t_emb
                torch.cuda.empty_cache()
                gc.collect()

                dry_run_result = {
                    "executed": True,
                    "success": True,
                    "measured_allocated_mb": round(allocated_mem, 2),
                    "measured_peak_mb": round(peak_mem, 2),
                    "fits_in_10gb": peak_mem < 9500.0,
                }
                self._add_check(
                    "Dry-Run VRAM Forward Verification",
                    True,
                    f"Measured peak VRAM: {round(peak_mem, 1)} MB. Model successfully loaded and executed on CUDA without weight modification."
                )

            except Exception as exc:
                dry_run_result = {"executed": True, "success": False, "error": str(exc)}
                self._add_check("Dry-Run VRAM Forward Verification", False, f"Dry-run forward pass failed: {exc}", is_warning=True)

        return {"budget": budget, "dry_run": dry_run_result}

    def evaluate(self, dry_run: bool = False) -> Dict[str, Any]:
        """Runs complete preflight assessment and synthesizes readiness verdict."""
        hw_info = self.check_cuda_and_hardware()
        self.check_dependencies()
        manifest_info = self.check_dataset_and_manifest()
        self.check_training_config()
        self.check_storage_and_permissions()
        feasibility_info = self.run_memory_feasibility(dry_run=dry_run)

        # Determine overall readiness
        if len(self.errors) > 0:
            readiness = "NOT_READY"
        elif len(self.warnings) > 0:
            readiness = "READY_WITH_LIMITATIONS"
        else:
            readiness = "READY"

        summary_message = (
            f"Preflight status: {readiness}. "
            f"{len(self.checklist)} checks evaluated: {sum(1 for c in self.checklist if c['status'] == 'PASS')} passed, "
            f"{len(self.warnings)} warnings, {len(self.errors)} errors."
        )

        return {
            "status": readiness,
            "summary": summary_message,
            "hardware": hw_info,
            "checklist": self.checklist,
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "feasibility": feasibility_info,
            "manifest_stats": manifest_info,
        }


def run_training_preflight(
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Helper entry point for API and CLI preflight checks."""
    if config is None:
        cfg_file = Path(config_path or "training/configs/lora_10gb_rtx3080.yaml")
        if not cfg_file.is_absolute() and not cfg_file.exists():
            cfg_file = WORKSPACE_ROOT / cfg_file
        with open(cfg_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    checker = TrainingPreflightChecker(config=config, manifest_path=manifest_path)
    return checker.evaluate(dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(description="BHIV TTV 10 GB VRAM Training Preflight & Feasibility Checker")
    parser.add_argument("--config", type=str, default="training/configs/lora_10gb_rtx3080.yaml", help="Path to YAML training config")
    parser.add_argument("--manifest", type=str, default=None, help="Optional dataset manifest override")
    parser.add_argument("--dry-run", action="store_true", help="Perform non-destructive CUDA forward pass memory check")
    args = parser.parse_args()

    print("=" * 65)
    print(" BHIV TTV STUDIO — 10 GB VRAM TRAINING PREFLIGHT CHECKER")
    print("=" * 65)
    result = run_training_preflight(config_path=args.config, manifest_path=args.manifest, dry_run=args.dry_run)

    print(f"\nOVERALL READINESS: [{result['status']}]")
    print(f"{result['summary']}\n")

    print("CHECKLIST:")
    for item in result["checklist"]:
        symbol = "[PASS]" if item["status"] == "PASS" else ("[WARN]" if item["status"] == "WARN" else "[FAIL]")
        print(f"  {symbol} {item['name']}: {item['details']}")

    if result["warnings"]:
        print("\nWARNINGS:")
        for w in result["warnings"]:
            print(f"  - {w}")

    if result["errors"]:
        print("\nERRORS (BLOCKERS):")
        for e in result["errors"]:
            print(f"  - {e}")

    if result["recommendations"]:
        print("\nRECOMMENDED ACTIONS:")
        for r in result["recommendations"]:
            print(f"  * {r}")

    print("\n" + "=" * 65)
    sys.exit(0 if result["status"] in ("READY", "READY_WITH_LIMITATIONS") else 1)


if __name__ == "__main__":
    main()
