from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import uuid
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import yaml

from app.core.config import settings, WORKSPACE_ROOT
from app.core.logging import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def check_cuda_availability() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Evaluates whether the runtime has access to an active NVIDIA CUDA GPU.
    Supports environment flag TTV_FORCE_CUDA_AVAILABLE for simulated test environments.
    """
    force_cuda = os.environ.get("TTV_FORCE_CUDA_AVAILABLE", "").strip().lower() in ("1", "true", "yes")

    hardware_info: Dict[str, Any] = {
        "torch_available": TORCH_AVAILABLE,
        "torch_version": torch.__version__ if TORCH_AVAILABLE else None,
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": None,
        "vram_total_gb": 0.0,
        "device_type": "cpu",
    }

    if force_cuda:
        hardware_info["cuda_available"] = True
        hardware_info["gpu_count"] = 1
        hardware_info["gpu_name"] = "Simulated NVIDIA GPU (CUDA Enabled)"
        hardware_info["vram_total_gb"] = 24.0
        hardware_info["device_type"] = "cuda"
        return True, "CUDA is available (Environment Forced)", hardware_info

    if TORCH_AVAILABLE and torch.cuda.is_available():
        hardware_info["cuda_available"] = True
        hardware_info["gpu_count"] = torch.cuda.device_count()
        hardware_info["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            hardware_info["vram_total_gb"] = round(vram, 2)
        except Exception:
            hardware_info["vram_total_gb"] = 0.0
        hardware_info["device_type"] = "cuda"
        return True, f"NVIDIA GPU detected: {hardware_info['gpu_name']} ({hardware_info['vram_total_gb']} GB VRAM)", hardware_info

    hardware_info["cuda_available"] = False
    hardware_info["device_type"] = "cpu"
    return False, "CUDA is not available on this system. Training video diffusion/LoRA models requires an NVIDIA GPU with CUDA drivers.", hardware_info


def parse_manifest_records(content_or_path: Union[str, Path, bytes, List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Parses manifest records from a JSON array, JSON object with list, JSONL text, file path, or raw bytes.
    Returns (records, resolved_manifest_path).
    """
    if isinstance(content_or_path, list):
        return content_or_path, None

    # Check if content_or_path is an existing file path
    if isinstance(content_or_path, (str, Path)):
        p = Path(content_or_path)
        if not p.is_absolute():
            p = settings.get_absolute_path(str(p))
        if p.exists() and p.is_file():
            text = p.read_text(encoding="utf-8")
            records, _ = parse_manifest_text(text)
            return records, str(p.resolve())

    # Otherwise treat as raw text/bytes
    text = content_or_path.decode("utf-8") if isinstance(content_or_path, bytes) else str(content_or_path)
    records, _ = parse_manifest_text(text)
    return records, None


def parse_manifest_text(text: str) -> Tuple[List[Dict[str, Any]], None]:
    records: List[Dict[str, Any]] = []
    trimmed = text.strip()
    if not trimmed:
        return records, None

    # Try standard JSON first
    if trimmed.startswith("[") or trimmed.startswith("{"):
        try:
            data = json.loads(trimmed)
            if isinstance(data, list):
                return data, None
            if isinstance(data, dict):
                # Search for known list keys
                for key in ("records", "samples", "data", "videos", "items", "dataset"):
                    if key in data and isinstance(data[key], list):
                        return data[key], None
                # If dict itself represents a single record
                if "prompt" in data or "id" in data:
                    return [data], None
        except Exception:
            pass

    # Fallback to JSONL line-by-line parsing
    for line in trimmed.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass

    return records, None


def validate_manifest_records(records: List[Dict[str, Any]], min_prompt_len: int = 3) -> Dict[str, Any]:
    """
    Validates manifest records without modifying original dataset files.
    Accurately supports remote URLs (video_url, source_url) and local video paths (video_path).
    """
    valid_samples: List[Dict[str, Any]] = []
    rejected_samples: List[Dict[str, Any]] = []
    splits_count: Dict[str, int] = {}
    categories_count: Dict[str, int] = {}
    formats_count: Dict[str, int] = {}
    has_remote_urls = False

    seen_ids = set()
    seen_prompts = set()

    for idx, item in enumerate(records):
        item_id = str(item.get("id") or f"record_{idx:04d}").strip()
        prompt = str(item.get("prompt") or item.get("caption") or "").strip()
        video_ref = str(item.get("video_url") or item.get("video_path") or item.get("source_url") or "").strip()
        split = str(item.get("split") or "train").strip().lower()
        category = str(item.get("category") or "general").strip().lower()
        fmt = str(item.get("format") or "mp4").strip().lower()

        issues = []
        if len(prompt) < min_prompt_len:
            issues.append(f"Prompt too short (< {min_prompt_len} characters)")

        if not video_ref:
            issues.append("Missing video reference (neither video_url, video_path, nor source_url found)")

        if item_id in seen_ids:
            issues.append(f"Duplicate id: {item_id}")
        seen_ids.add(item_id)

        if issues:
            rejected_samples.append({
                "index": idx,
                "id": item_id,
                "reasons": issues,
                "sample": item
            })
            continue

        if video_ref.startswith(("http://", "https://")):
            has_remote_urls = True

        splits_count[split] = splits_count.get(split, 0) + 1
        categories_count[category] = categories_count.get(category, 0) + 1
        formats_count[fmt] = formats_count.get(fmt, 0) + 1

        valid_samples.append({
            "id": item_id,
            "prompt": prompt,
            "video_ref": video_ref,
            "split": split,
            "category": category,
            "format": fmt,
            "license_note": item.get("license_note") or item.get("license") or "N/A"
        })

    is_valid = len(valid_samples) > 0 and len(rejected_samples) == 0
    total = len(records)
    summary = f"Validated {len(valid_samples)}/{total} records across {len(splits_count)} split(s)."
    if rejected_samples:
        summary += f" ({len(rejected_samples)} records rejected)"

    preview = valid_samples[:5]

    return {
        "valid": is_valid,
        "total_records": total,
        "valid_records": len(valid_samples),
        "rejected_records": len(rejected_samples),
        "splits": splits_count,
        "categories": categories_count,
        "formats": formats_count,
        "has_remote_urls": has_remote_urls,
        "sample_preview": preview,
        "issues": rejected_samples,
        "summary": summary
    }


class TrainingJobManager:
    """
    Production-grade training execution manager:
    - Never executes Python/CUDA in the browser
    - Validates manifest before queueing
    - Checks hardware/CUDA and flags CUDA_UNAVAILABLE cleanly
    - Retains reviewed CLI command as manual fallback
    - Spawns and supervises training worker subprocess when CUDA is enabled
    - Streams and persists stdout/stderr logs and checkpoints
    """
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._gpu_metrics_cache: Dict[str, Any] = {"sampled_at": 0.0, "devices": None}
        self._gpu_metrics_warning_logged = False
        self.jobs_dir = settings.get_absolute_path("generated/training_jobs")
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._load_persisted_jobs()

    def _load_persisted_jobs(self):
        """Re-loads recent training job states from disk on startup."""
        try:
            for job_folder in self.jobs_dir.iterdir():
                if job_folder.is_dir():
                    state_file = job_folder / "job_state.json"
                    if state_file.exists():
                        try:
                            state = json.loads(state_file.read_text(encoding="utf-8"))
                            job_id = state.get("job_id")
                            if job_id:
                                # Load logs if available
                                log_file = job_folder / "training.log"
                                if log_file.exists():
                                    state["logs"] = log_file.read_text(encoding="utf-8").splitlines()
                                self._jobs[job_id] = state
                        except Exception as e:
                            logger.warning(f"Failed to load persisted training job {job_folder.name}: {e}")
        except Exception as e:
            logger.warning(f"Error scanning persisted training jobs: {e}")

    def _persist_job(self, job_id: str):
        job = self._jobs.get(job_id)
        if not job:
            return
        try:
            job_folder = self.jobs_dir / job_id
            job_folder.mkdir(parents=True, exist_ok=True)
            state_file = job_folder / "job_state.json"
            
            # Serialize state excluding huge log arrays to keep state_file compact
            state_to_save = dict(job)
            logs = state_to_save.pop("logs", [])
            state_file.write_text(json.dumps(state_to_save, indent=2), encoding="utf-8")

            # Persist logs to training.log
            log_file = job_folder / "training.log"
            log_file.write_text("\n".join(logs) + "\n", encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to persist training job state {job_id}: {e}")

    def _read_gpu_metrics(self) -> Optional[List[Dict[str, Any]]]:
        now = time.monotonic()
        if now - self._gpu_metrics_cache["sampled_at"] < 2:
            return self._gpu_metrics_cache["devices"]

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            devices = []
            for line in result.stdout.splitlines():
                values = [value.strip() for value in line.split(",")]
                if len(values) != 4:
                    continue
                devices.append({
                    "index": int(values[0]),
                    "utilization_percent": int(values[1]),
                    "memory_used_mib": int(values[2]),
                    "memory_total_mib": int(values[3]),
                })
            self._gpu_metrics_cache = {"sampled_at": now, "devices": devices or None}
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                devices = []
                for i in range(torch.cuda.device_count()):
                    free_b, total_b = torch.cuda.mem_get_info(i)
                    used_b = total_b - free_b
                    devices.append({
                        "index": i,
                        "utilization_percent": 0,
                        "memory_used_mib": int(used_b / (1024 * 1024)),
                        "memory_total_mib": int(total_b / (1024 * 1024)),
                    })
                self._gpu_metrics_cache = {"sampled_at": now, "devices": devices or None}
                return self._gpu_metrics_cache["devices"]
            elif os.environ.get("TTV_FORCE_CUDA_AVAILABLE", "").strip().lower() in ("1", "true", "yes"):
                devices = [{
                    "index": 0,
                    "utilization_percent": 42,
                    "memory_used_mib": 6144,
                    "memory_total_mib": 24576,
                }]
                self._gpu_metrics_cache = {"sampled_at": now, "devices": devices}
                return devices

            if not self._gpu_metrics_warning_logged:
                logger.warning("GPU utilization metrics are unavailable: %s", exc)
                self._gpu_metrics_warning_logged = True
            self._gpu_metrics_cache = {"sampled_at": now, "devices": None}
        return self._gpu_metrics_cache["devices"]

    async def create_job(
        self,
        manifest_data: Union[str, Path, bytes, List[Dict[str, Any]]],
        config: Dict[str, Any],
        manifest_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        async with self._lock:
            job_id = f"train_job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            # 1. Parse & validate manifest records
            records, source_path = parse_manifest_records(manifest_data)
            if not records and source_path:
                # If path was passed, re-parse directly
                p = Path(source_path)
                if p.exists():
                    records, _ = parse_manifest_records(p.read_text(encoding="utf-8"))

            validation_stats = validate_manifest_records(records)
            if not validation_stats["valid"]:
                reasons = [
                    reason
                    for issue in validation_stats["issues"]
                    for reason in issue.get("reasons", [])
                ]
                raise ValueError(
                    "Dataset validation failed: "
                    + ("; ".join(reasons) if reasons else "the dataset is empty.")
                )

            # Write working copy of manifest for this training job
            job_dir = self.jobs_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            manifest_target = job_dir / "manifest.jsonl"
            with open(manifest_target, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            manifest_path_str = str(manifest_target.resolve())

            # 2. Check CUDA environment
            cuda_available, cuda_msg, hw_profile = check_cuda_availability()

            base_model = config.get("base_model", "Wan2.2-TI2V-5B")
            method = config.get("method", "lora")
            epochs = int(config.get("epochs", 10))
            learning_rate = float(config.get("learning_rate", 0.0001))
            batch_size = int(config.get("batch_size", 2))
            is_wan = "wan" in base_model.lower()
            default_version = f"wan_lora_{job_id}" if is_wan else f"ttv_{method}_{job_id}"
            version_name = config.get("version_name") or default_version

            # 3. Formulate training configuration YAML
            if is_wan:
                yaml_config = {
                    "model": {
                        "name": "Wan2.2-TI2V-5B",
                        "architecture": "Wan2.2-TI2V-5B",
                        "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                        "in_channels": 16,
                        "num_frames": 16,
                        "text_embed_dim": 4096
                    },
                    "training": {
                        "device": "cuda" if cuda_available else "cpu",
                        "learning_rate": learning_rate,
                        "weight_decay": 0.01,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "gradient_accumulation_steps": 1,
                        "max_grad_norm": 1.0,
                        "seed": 42,
                        "use_lora": True,
                        "lora_rank": 16,
                        "lora_alpha": 32.0
                    },
                    "preprocessing": {
                        "height": 720,
                        "width": 1280,
                        "normalize_range": [-1.0, 1.0]
                    },
                    "paths": {
                        "dataset_manifest": str(manifest_target.relative_to(WORKSPACE_ROOT) if manifest_target.is_relative_to(WORKSPACE_ROOT) else manifest_target),
                        "train_manifest": str(manifest_target.relative_to(WORKSPACE_ROOT) if manifest_target.is_relative_to(WORKSPACE_ROOT) else manifest_target),
                        "val_manifest": str(manifest_target.relative_to(WORKSPACE_ROOT) if manifest_target.is_relative_to(WORKSPACE_ROOT) else manifest_target),
                        "checkpoint_dir": f"models/lora/{version_name}",
                        "models_dir": "models/lora"
                    }
                }
            else:
                yaml_config = {
                    "model": {
                        "name": base_model,
                        "in_channels": 3,
                        "latent_channels": 64,
                        "num_frames": 8,
                        "vocab_size": 10000,
                        "max_prompt_length": 32,
                        "text_embed_dim": 128
                    },
                    "training": {
                        "device": "cuda" if cuda_available else "cpu",
                        "learning_rate": learning_rate,
                        "weight_decay": 0.01,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "gradient_accumulation_steps": 1,
                        "max_grad_norm": 1.0,
                        "seed": 42,
                        "use_lora": method in ("lora", "temporal_only"),
                        "lora_rank": 8,
                        "lora_alpha": 16.0
                    },
                    "preprocessing": {
                        "height": 128,
                        "width": 128,
                        "normalize_range": [-1.0, 1.0]
                    },
                    "paths": {
                        "dataset_manifest": str(manifest_target.relative_to(WORKSPACE_ROOT) if manifest_target.is_relative_to(WORKSPACE_ROOT) else manifest_target),
                        "train_manifest": str(manifest_target.relative_to(WORKSPACE_ROOT) if manifest_target.is_relative_to(WORKSPACE_ROOT) else manifest_target),
                        "val_manifest": str(manifest_target.relative_to(WORKSPACE_ROOT) if manifest_target.is_relative_to(WORKSPACE_ROOT) else manifest_target),
                        "checkpoint_dir": f"training/checkpoints/{job_id}",
                        "models_dir": "models"
                    }
                }

            config_file = job_dir / "training_config.yaml"
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(yaml_config, f, sort_keys=False)

            # Reviewed CLI fallback command
            cli_fallback_command = f"python training/run_training.py --config generated/training_jobs/{job_id}/training_config.yaml --version-name {version_name}"

            created_time = datetime.utcnow().isoformat() + "Z"

            initial_logs = [
                f"[System] Training Job Initialized: {job_id}",
                f"[Manifest] Loaded {validation_stats['total_records']} records from {manifest_filename or source_path or 'upload'}.",
                f"[Manifest] Validation Result: {validation_stats['valid_records']} valid, {validation_stats['rejected_records']} rejected.",
                f"[Manifest] Split breakdown: {json.dumps(validation_stats['splits'])}.",
                f"[Config] Method: {method}, Base: {base_model}, Epochs: {epochs}, LR: {learning_rate}, Batch: {batch_size}.",
                f"[Hardware] Python: {hw_profile.get('torch_version', 'N/A')}, CUDA Available: {cuda_available} ({hw_profile.get('gpu_name') or 'No NVIDIA GPU'})."
            ]

            if not cuda_available:
                status = "CUDA_UNAVAILABLE"
                stage = "cuda_unavailable"
                initial_logs.extend([
                    "[CUDA Check] FAILED: torch.cuda.is_available() is False.",
                    f"[Status] {status}: Training worker cannot launch without NVIDIA CUDA acceleration.",
                    "[Manual Fallback] Run the reviewed CLI command in a CUDA training environment:",
                    f"  {cli_fallback_command}"
                ])
                message = "CUDA is unavailable on this system. Job registered with manual CLI fallback command."
            else:
                status = "queued"
                stage = "queued"
                initial_logs.append("[Worker] CUDA environment confirmed. Job queued for worker subprocess launch.")
                message = "Training job successfully queued in CUDA environment."

            job_state = {
                "job_id": job_id,
                "status": status,
                "progress": 0,
                "stage": stage,
                "created_at": created_time,
                "updated_at": created_time,
                "config": {
                    "base_model": base_model,
                    "method": method,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "version_name": version_name
                },
                "manifest_stats": validation_stats,
                "manifest_path": manifest_path_str,
                "cuda_available": cuda_available,
                "device": hw_profile.get("device_type", "cpu"),
                "hardware": hw_profile,
                "cli_fallback_command": cli_fallback_command,
                "checkpoints": [],
                "final_checkpoint": None,
                "metrics": {},
                "logs": initial_logs,
                "error": cuda_msg if not cuda_available else None,
                "message": message
            }

            self._jobs[job_id] = job_state
            self._persist_job(job_id)

            # If CUDA is available, launch worker background thread
            if cuda_available:
                self._run_training_subprocess(job_id, config_file, version_name)

            return job_state

    def _run_training_subprocess(self, job_id: str, config_file: Path, version_name: str):
        """Executes the training worker in a background thread and captures stdout/stderr."""
        def worker_target():
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "running"
                self._jobs[job_id]["stage"] = "training_running"
                self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"
                self._jobs[job_id]["logs"].append("[Worker Subprocess] Launching training process...")
                self._persist_job(job_id)

            try:
                cmd = [
                    sys.executable,
                    "-m",
                    "training.run_training",
                    "--config",
                    str(config_file),
                    "--version-name",
                    version_name
                ]

                process = subprocess.Popen(
                    cmd,
                    cwd=str(WORKSPACE_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace"
                )

                epoch_regex = re.compile(
                    r"Epoch\s*\[(\d+)/(\d+)\].*?Train Loss:\s*([0-9.eE+-]+).*?Val Loss:\s*([0-9.eE+-]+)"
                )
                checkpoint_regex = re.compile(r"Checkpoint\s*(?:saved|Path):\s*([^\r\n]+)", re.IGNORECASE)

                if process.stdout:
                    for line in process.stdout:
                        line = line.rstrip()
                        if not line:
                            continue
                        if job_id in self._jobs:
                            self._jobs[job_id]["logs"].append(line)
                            match = epoch_regex.search(line)
                            if match:
                                cur_ep = int(match.group(1))
                                tot_ep = int(match.group(2))
                                pct = int((cur_ep / tot_ep) * 100)
                                self._jobs[job_id]["progress"] = min(99, max(5, pct))
                                self._jobs[job_id]["stage"] = f"epoch_{cur_ep}_of_{tot_ep}"
                                self._jobs[job_id]["epoch"] = cur_ep
                                self._jobs[job_id]["total_epochs"] = tot_ep
                                self._jobs[job_id]["loss"] = float(match.group(3))
                                self._jobs[job_id]["metrics"]["train_loss"] = float(match.group(3))
                                self._jobs[job_id]["metrics"]["val_loss"] = float(match.group(4))

                            ckpt_match = checkpoint_regex.search(line)
                            if ckpt_match:
                                ckpt_path = ckpt_match.group(1).strip()
                                if ckpt_path not in self._jobs[job_id]["checkpoints"]:
                                    self._jobs[job_id]["checkpoints"].append(ckpt_path)
                                if Path(ckpt_path).name == "final_model.pt":
                                    self._jobs[job_id]["final_checkpoint"] = ckpt_path

                            self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"

                returncode = process.wait()

                if job_id in self._jobs:
                    self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"
                    if returncode == 0:
                        self._jobs[job_id]["status"] = "completed"
                        self._jobs[job_id]["stage"] = "completed"
                        self._jobs[job_id]["progress"] = 100
                        self._jobs[job_id]["logs"].append("[Worker] Training process completed with exit code 0.")
                    else:
                        self._jobs[job_id]["status"] = "failed"
                        self._jobs[job_id]["stage"] = "failed"
                        self._jobs[job_id]["error"] = f"Worker process failed with exit code {returncode}"
                        self._jobs[job_id]["logs"].append(f"[Worker Error] Process terminated with returncode {returncode}.")
                    self._persist_job(job_id)

            except Exception as e:
                if job_id in self._jobs:
                    self._jobs[job_id]["status"] = "failed"
                    self._jobs[job_id]["stage"] = "failed"
                    self._jobs[job_id]["error"] = str(e)
                    self._jobs[job_id]["logs"].append(f"[Worker Exception] {str(e)}")
                    self._persist_job(job_id)

        t = threading.Thread(target=worker_target, daemon=True)
        t.start()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(job_id)
        if job and job.get("cuda_available"):
            job["hardware"] = dict(job.get("hardware") or {})
            job["hardware"]["gpu_utilization"] = self._read_gpu_metrics()
        return job

    def get_job_logs(self, job_id: str, offset: int = 0, limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(job_id)
        if not job:
            return None
        logs = job.get("logs", [])
        sliced = logs[offset:]
        if limit is not None and limit > 0:
            sliced = sliced[:limit]
        return {
            "job_id": job_id,
            "status": job.get("status"),
            "total_lines": len(logs),
            "logs": sliced,
            "cli_fallback_command": job.get("cli_fallback_command")
        }

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        sorted_jobs = sorted(
            self._jobs.values(),
            key=lambda j: j.get("created_at", ""),
            reverse=True
        )
        return sorted_jobs[:limit]


# Global singleton instance
training_job_manager = TrainingJobManager()
