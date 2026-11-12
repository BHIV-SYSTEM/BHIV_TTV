"""Training and Fine-Tuning API routes with Reference and Job Management workflows."""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request, Query
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import telemetry, logger
from app.core.exceptions import ValidationError
from app.services.reference_service import reference_service

from app.schemas.request import TrainingSessionRequest
from app.schemas.response import TrainingSessionResponse
from app.schemas.training import (
    AutomaticTrainingJobResponse,
    AutomaticTrainingRequest,
    TrainingJobConfig,
    TrainingJobResponse,
    TrainingLogsResponse,
    ManifestValidationResponse,
)
from app.services.training_service import (
    training_job_manager,
    parse_manifest_records,
    validate_manifest_records,
)

from training.datasets.manifest_builder import build_manifest_from_directory, prepare_reference_dataset_entry
from training.datasets.validator import DatasetValidator
from training.preprocessing.video_preprocessor import VideoPreprocessor
from training.fine_tuning.model import SpatialTemporalTTVModel
from training.fine_tuning.trainer import TTVTrainer
from training.datasets.loader import create_dataloader

router = APIRouter()

# In-memory training sessions registry for direct reference fine-tuning
_TRAINING_SESSIONS: Dict[str, Dict[str, Any]] = {}
_AUTOMATIC_TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}
_AUTOMATIC_TRAINING_BY_REFERENCE: Dict[str, str] = {}
_AUTOMATIC_TRAINING_LOCK = asyncio.Lock()


def _validate_reference_sample(reference_payload: Dict[str, Any]) -> Dict[str, Any]:
    sample = prepare_reference_dataset_entry(reference_payload)
    media_path = sample["video_path"]

    if reference_payload["media_type"] == "video":
        validation = DatasetValidator(min_duration=0.1).validate_dataset([sample])
        if validation["valid_count"] != 1:
            reasons = [
                issue.get("reason", "Dataset sample validation failed")
                for issue in validation["rejected_samples"]
            ]
            raise ValidationError("; ".join(reasons))

    frames = VideoPreprocessor(num_frames=8, height=128, width=128).process_video(media_path)
    if tuple(frames.shape) != (8, 3, 128, 128) or not bool(frames.isfinite().all()):
        raise ValidationError("Reference frame extraction produced missing or corrupted frames.")

    validation = validate_manifest_records([sample])
    if not validation["valid"]:
        issues = validation["issues"]
        details = "; ".join(
            reason
            for issue in issues
            for reason in issue.get("reasons", [])
        )
        raise ValidationError(details or "The prepared reference dataset is invalid.")
    return sample


async def _run_automatic_training_job(job_id: str, request: AutomaticTrainingRequest) -> None:
    job = _AUTOMATIC_TRAINING_JOBS[job_id]
    try:
        job["status"] = "processing"
        job["stage"] = "reference_validation"
        job["message"] = "Validating the uploaded reference."
        job["logs"].append("[Reference] Validating uploaded media...")
        reference = await reference_service.resolve(reference_id=request.reference_id)
        if not reference:
            raise ValidationError("The uploaded reference could not be found.")

        job["stage"] = "frame_extraction"
        job["logs"].append("[Dataset] Extracting and validating reference frames...")
        sample = await asyncio.to_thread(_validate_reference_sample, reference)

        job["stage"] = "dataset_validation"
        job["logs"].append("[Dataset] Dataset preparation and validation completed.")
        config = {
            "base_model": request.base_model,
            "method": request.method,
            "epochs": request.epochs,
            "learning_rate": request.learning_rate,
            "batch_size": request.batch_size,
            "version_name": None,
        }
        job["logs"].append(f"[Model] Preparing {request.base_model} with {request.method} fine-tuning.")
        queued_job = await training_job_manager.create_job(
            manifest_data=[sample],
            config=config,
            manifest_filename=f"{request.reference_id}.jsonl",
        )
        job["training_job_id"] = queued_job["job_id"]
        job["status"] = queued_job["status"]
        job["stage"] = queued_job["stage"]
        job["message"] = queued_job.get("message")
        job["cuda_available"] = queued_job.get("cuda_available", False)
        job["hardware"] = queued_job.get("hardware")
        job["logs"].append(
            f"[Training] Job {queued_job['job_id']} submitted to the training queue."
        )
    except Exception as exc:
        logger.exception("Automatic training job %s failed: %s", job_id, exc)
        job["status"] = "failed"
        job["stage"] = "failed"
        job["error"] = str(exc)
        job["message"] = "Automatic training failed before or during job submission."
        job["logs"].append(f"[Error] {exc}")


def _automatic_training_snapshot(job_id: str) -> Optional[Dict[str, Any]]:
    job = _AUTOMATIC_TRAINING_JOBS.get(job_id)
    if not job:
        return None

    snapshot = {
        "job_id": job_id,
        "status": job["status"],
        "stage": job["stage"],
        "progress": 0,
        "epoch": 0,
        "total_epochs": 0,
        "loss": None,
        "device": "cpu",
        "reference_id": job["reference_id"],
        "training_job_id": job.get("training_job_id"),
        "cuda_available": job.get("cuda_available", False),
        "hardware": job.get("hardware"),
        "checkpoints": [],
        "final_checkpoint": None,
        "metrics": {},
        "error": job.get("error"),
        "message": job.get("message"),
    }
    training_job_id = job.get("training_job_id")
    queued_job = training_job_manager.get_job(training_job_id) if training_job_id else None
    if queued_job:
        for key in (
            "status", "stage", "progress", "epoch", "total_epochs", "loss",
            "device", "cuda_available", "hardware", "checkpoints", "final_checkpoint",
            "metrics", "error", "message",
        ):
            if key in queued_job:
                snapshot[key] = queued_job[key]
    return snapshot


@router.post("/auto", response_model=AutomaticTrainingJobResponse)
async def start_automatic_training(request: AutomaticTrainingRequest):
    """Validate a reference and queue its fine-tuning job without blocking the request."""
    async with _AUTOMATIC_TRAINING_LOCK:
        existing_id = _AUTOMATIC_TRAINING_BY_REFERENCE.get(request.reference_id)
        existing = _AUTOMATIC_TRAINING_JOBS.get(existing_id) if existing_id else None
        retryable_statuses = {"failed", "CUDA_UNAVAILABLE"}
        if existing and not (
            request.retry and existing["status"] in retryable_statuses
        ):
            return AutomaticTrainingJobResponse(**_automatic_training_snapshot(existing_id))

        job_id = f"train_auto_{uuid.uuid4().hex[:12]}"
        _AUTOMATIC_TRAINING_BY_REFERENCE[request.reference_id] = job_id
        _AUTOMATIC_TRAINING_JOBS[job_id] = {
            "job_id": job_id,
            "reference_id": request.reference_id,
            "status": "preparing",
            "stage": "queued_for_preparation",
            "logs": ["[Session] Automatic training workflow started."],
            "error": None,
            "training_job_id": None,
            "message": "Preparing the reference dataset.",
        }

    asyncio.create_task(_run_automatic_training_job(job_id, request))
    return AutomaticTrainingJobResponse(**_automatic_training_snapshot(job_id))


@router.get("/auto/{job_id}", response_model=AutomaticTrainingJobResponse)
async def get_automatic_training_status(job_id: str):
    snapshot = _automatic_training_snapshot(job_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Automatic training job not found.")
    return AutomaticTrainingJobResponse(**snapshot)


@router.get("/auto/{job_id}/logs", response_model=TrainingLogsResponse)
async def get_automatic_training_logs(
    job_id: str,
    offset: int = Query(default=0, ge=0),
    limit: Optional[int] = Query(default=None, ge=1),
):
    job = _AUTOMATIC_TRAINING_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Automatic training job not found.")

    logs = list(job["logs"])
    training_job_id = job.get("training_job_id")
    queued_logs = training_job_manager.get_job_logs(training_job_id) if training_job_id else None
    if queued_logs:
        logs.extend(queued_logs["logs"])
    selected_logs = logs[offset:offset + limit] if limit is not None else logs[offset:]
    return TrainingLogsResponse(
        job_id=job_id,
        status=_automatic_training_snapshot(job_id)["status"],
        total_lines=len(logs),
        logs=selected_logs,
        cli_fallback_command=(
            training_job_manager.get_job(training_job_id) or {}
        ).get("cli_fallback_command") if training_job_id else None,
    )


async def _execute_training_job(
    job_id: str,
    base_model_name: str,
    method: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    reference_id: Optional[str],
    manifest_path: str,
    version_name: str,
):
    session = _TRAINING_SESSIONS.get(job_id)
    if not session:
        return

    try:
        session["status"] = "running"
        session["logs"].append(f"[Worker] Initializing {method.upper()} fine-tuning for {base_model_name}...")
        
        # 1. Check & prepare DataLoader
        manifest_p = Path(manifest_path)
        if not manifest_p.exists():
            session["logs"].append(f"[Dataset] Manifest {manifest_path} not found. Automatically building from generated/videos...")
            build_manifest_from_directory(
                videos_dir=str(settings.get_absolute_path("generated/videos")),
                output_manifest_path=str(manifest_p)
            )

        session["logs"].append(f"[Dataset] Loaded dataset manifest from {manifest_path}.")
        
        train_loader = create_dataloader(
            manifest_path=str(manifest_p),
            batch_size=max(1, batch_size),
            num_frames=8,
            height=128,
            width=128,
            shuffle=True
        )
        val_loader = create_dataloader(
            manifest_path=str(manifest_p),
            batch_size=max(1, batch_size),
            num_frames=8,
            height=128,
            width=128,
            shuffle=False
        )

        # 2. Instantiate SpatialTemporalTTVModel
        model = SpatialTemporalTTVModel(
            in_channels=3,
            latent_channels=64,
            num_frames=8,
            vocab_size=10000,
            max_prompt_length=32,
            text_embed_dim=128
        )

        # 3. Configure Trainer
        train_cfg = {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "use_lora": (method == "lora"),
            "lora_rank": 8 if method == "lora" else 4,
            "lora_alpha": 16.0 if method == "lora" else 8.0,
            "device": "cpu"
        }
        
        ckpt_dir = str(settings.get_absolute_path("training/checkpoints"))
        trainer = TTVTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_cfg,
            checkpoint_dir=ckpt_dir
        )

        session["logs"].append(f"[Trainer] Model initialized. Trainable parameters: {trainer.trainable_params_count:,} / {trainer.total_params_count:,}")
        session["logs"].append(f"[Training] Running {epochs} epoch(s)...")

        # Run training loop asynchronously
        results = await asyncio.to_thread(trainer.train)

        session["status"] = "completed"
        session["result"] = results
        session["logs"].append(f"[Complete] Training finished successfully! Final loss: {results.get('best_val_loss', 0.0):.4f}")
        session["logs"].append(f"[Checkpoint] Checkpoint saved at {results.get('final_checkpoint')}")

    except Exception as exc:
        logger.exception(f"Training session {job_id} failed: {exc}")
        session["status"] = "failed"
        session["error"] = str(exc)
        session["logs"].append(f"[Error] Training failed: {exc}")


@router.post("/start", response_model=TrainingSessionResponse)
async def start_training_session(request: TrainingSessionRequest):
    """
    Launch a model training or fine-tuning session, incorporating reference media
    if provided.
    """
    job_id = f"train_{uuid.uuid4().hex[:10]}"
    logs: List[str] = [
        f"[Session] Created training job {job_id}",
        f"[Config] Method={request.method}, Epochs={request.epochs}, LR={request.learning_rate}, BatchSize={request.batch_size}"
    ]

    ref_payload = None
    manifest_path = request.dataset_manifest or "training/datasets/dataset.jsonl"
    abs_manifest = settings.get_absolute_path(manifest_path)

    # 1. Resolve and integrate reference if supplied
    if request.reference_id or request.reference_url:
        try:
            ref_payload = await reference_service.resolve(
                reference_url=request.reference_url,
                reference_id=request.reference_id,
                reference_type=request.reference_type
            )
            if ref_payload:
                logs.append(
                    f"[Reference] Attached reference {ref_payload['reference_id']} "
                    f"({ref_payload['media_type']}, {ref_payload['width']}x{ref_payload['height']}, source={ref_payload['source']})"
                )
                # Prepare and insert reference into training dataset
                ref_manifest_path = str(settings.get_absolute_path("training/datasets/reference_train_manifest.jsonl"))
                sample_entry = prepare_reference_dataset_entry(
                    ref_payload,
                    output_manifest_path=ref_manifest_path
                )
                manifest_path = ref_manifest_path
                logs.append(f"[Dataset] Augmented training dataset with reference sample: {sample_entry.get('id')}")
        except Exception as e:
            raise ValidationError(f"Could not process reference for training: {e}")

    # 2. Parse hyperparameters
    try:
        lr_float = float(request.learning_rate)
    except (ValueError, TypeError):
        lr_float = 0.0001

    epochs_val = int(request.epochs or 10)
    batch_size_val = int(request.batch_size or 2)

    # 3. Register session
    _TRAINING_SESSIONS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "reference_id": ref_payload.get("reference_id") if ref_payload else None,
        "reference_info": ref_payload,
        "config": request.model_dump(),
        "logs": logs,
        "result": None,
        "error": None
    }

    # 4. Launch background training runner
    asyncio.create_task(
        _execute_training_job(
            job_id=job_id,
            base_model_name=request.base_model or "SpatialTemporalTTVModel",
            method=request.method or "lora",
            epochs=epochs_val,
            learning_rate=lr_float,
            batch_size=batch_size_val,
            reference_id=ref_payload.get("reference_id") if ref_payload else None,
            manifest_path=manifest_path,
            version_name=request.version_name or "ttv_lora_v001"
        )
    )

    return TrainingSessionResponse(
        job_id=job_id,
        status="running",
        message="Training session launched successfully",
        reference_id=ref_payload.get("reference_id") if ref_payload else None,
        reference_info=ref_payload,
        config=request.model_dump(),
        logs=logs
    )


@router.get("/status/{job_id}", response_model=TrainingSessionResponse)
async def get_training_status(job_id: str):
    """
    Get live progress, metrics, and terminal log outputs for a training session.
    """
    session = _TRAINING_SESSIONS.get(job_id)
    if not session:
        raise HTTPException(status_code=404, detail="Training job not found")

    return TrainingSessionResponse(
        job_id=job_id,
        status=session.get("status", "running"),
        message=f"Training session is {session.get('status')}",
        reference_id=session.get("reference_id"),
        reference_info=session.get("reference_info"),
        config=session.get("config"),
        logs=session.get("logs", [])
    )


# --------------------------------------------------------------------------
# Training Job Management Endpoints (Manifest validation, queueing, logs)
# --------------------------------------------------------------------------

@router.post("/jobs", response_model=TrainingJobResponse)
async def launch_or_queue_training_job(
    request: Request,
    manifest_file: Optional[UploadFile] = File(None),
    manifest_path: Optional[str] = Form(None),
    manifest_content: Optional[str] = Form(None),
    base_model: str = Form("SpatialTemporalTTVModel"),
    method: str = Form("lora"),
    epochs: int = Form(10),
    learning_rate: float = Form(0.0001),
    batch_size: int = Form(2),
    version_name: Optional[str] = Form(None),
):
    """
    Submits a training/fine-tuning job request.
    - Validates uploaded JSON/JSONL manifest without modifying dataset files.
    - If CUDA is unavailable, returns a clear CUDA_UNAVAILABLE status with reviewed CLI command.
    - If CUDA is available, queues job and launches background worker subprocess.
    - Supports both multipart/form-data and application/json.
    """
    content_type = request.headers.get("content-type", "")
    manifest_filename = None

    # Handle application/json request
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        cfg = {
            "base_model": body.get("base_model", "SpatialTemporalTTVModel"),
            "method": body.get("method", "lora"),
            "epochs": int(body.get("epochs", 10)),
            "learning_rate": float(body.get("learning_rate", 0.0001)),
            "batch_size": int(body.get("batch_size", 2)),
            "version_name": body.get("version_name"),
        }
        manifest_raw = body.get("manifest") or body.get("manifest_content") or body.get("manifest_path")
        if not manifest_raw:
            manifest_raw = "data/manifests/ttv_training_manifest_120.jsonl"
            if not settings.get_absolute_path(manifest_raw).exists():
                manifest_raw = "training/datasets/dataset.jsonl"
        manifest_filename = body.get("manifest_filename")

    # Handle multipart/form-data
    else:
        cfg = {
            "base_model": base_model,
            "method": method,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "version_name": version_name,
        }
        if manifest_file and manifest_file.filename:
            manifest_raw = await manifest_file.read()
            manifest_filename = manifest_file.filename
        elif manifest_content:
            manifest_raw = manifest_content
            manifest_filename = "manifest_inline.json"
        elif manifest_path:
            manifest_raw = manifest_path
            manifest_filename = Path(manifest_path).name
        else:
            # Default to bundled 120-record manifest or local dataset
            default_p = settings.get_absolute_path("data/manifests/ttv_training_manifest_120.jsonl")
            if default_p.exists():
                manifest_raw = str(default_p)
                manifest_filename = "ttv_training_manifest_120.jsonl"
            else:
                manifest_raw = "training/datasets/dataset.jsonl"
                manifest_filename = "dataset.jsonl"

    try:
        job = await training_job_manager.create_job(
            manifest_data=manifest_raw,
            config=cfg,
            manifest_filename=manifest_filename
        )
        return TrainingJobResponse(**job)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create training job: {str(e)}")


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job_status(job_id: str):
    """Retrieves current execution status, progress, checkpoints, and hardware diagnostics for a training job."""
    job = training_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found.")
    return TrainingJobResponse(**job)


@router.get("/jobs/{job_id}/logs", response_model=TrainingLogsResponse)
async def get_training_job_logs(
    job_id: str,
    offset: int = Query(default=0, ge=0),
    limit: Optional[int] = Query(default=None, ge=1)
):
    """Streams execution terminal stdout/stderr logs for a given training job."""
    logs_data = training_job_manager.get_job_logs(job_id, offset=offset, limit=limit)
    if not logs_data:
        raise HTTPException(status_code=404, detail=f"Training job '{job_id}' not found.")
    return TrainingLogsResponse(**logs_data)


@router.get("/jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(limit: int = Query(default=20, ge=1, le=100)):
    """Lists recent training and fine-tuning jobs."""
    jobs = training_job_manager.list_jobs(limit=limit)
    return [TrainingJobResponse(**j) for j in jobs]


@router.post("/manifest/validate", response_model=ManifestValidationResponse)
async def validate_manifest_endpoint(
    request: Request,
    manifest_file: Optional[UploadFile] = File(None),
    manifest_path: Optional[str] = Form(None),
    manifest_content: Optional[str] = Form(None)
):
    """
    Validates a training manifest (JSON or JSONL) before submitting training.
    Reports record counts, split partitions, category statistics, and any format issues.
    """
    content_type = request.headers.get("content-type", "")
    manifest_filename = None

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        raw = body.get("manifest") or body.get("manifest_content") or body.get("manifest_path")
        manifest_filename = body.get("manifest_filename")
    else:
        if manifest_file and manifest_file.filename:
            raw = await manifest_file.read()
            manifest_filename = manifest_file.filename
        elif manifest_content:
            raw = manifest_content
        elif manifest_path:
            raw = manifest_path
        else:
            default_p = settings.get_absolute_path("data/manifests/ttv_training_manifest_120.jsonl")
            raw = str(default_p) if default_p.exists() else "training/datasets/dataset.jsonl"

    records, resolved_path = parse_manifest_records(raw)
    result = validate_manifest_records(records)
    result["manifest_path"] = resolved_path or manifest_filename
    return ManifestValidationResponse(**result)
