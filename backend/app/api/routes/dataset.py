"""
Dataset API Routes.
Provides unified dataset collection, ingestion, validation, duplicate detection,
versioning, and training manifest preparation across diverse visual subjects.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel, Field

from app.core.config import settings, WORKSPACE_ROOT
from dataset.pipeline import DatasetPipeline
from dataset.manifest import DatasetManager
from dataset.models import DatasetItem, PermissionState, ValidationStatus

router = APIRouter()
_pipeline = DatasetPipeline(base_dir=str(WORKSPACE_ROOT / "data"))
_manager = DatasetManager(storage_root=str(WORKSPACE_ROOT / "data" / "datasets"))


class CollectRequest(BaseModel):
    source_type: str = Field("url", description="Source type: url, youtube, text_story")
    url: Optional[str] = Field(None, description="Direct media URL or YouTube URL")
    story_text: Optional[str] = Field(None, description="Text narrative or story description")
    dataset_id: str = Field("default", description="Target dataset identifier")
    subject: str = Field("general", description="Subject: e.g. mythology, history, nature, sci-fi, cartoon")
    topic: str = Field("general", description="Specific topic or sub-theme")
    category: str = Field("general", description="Visual category: human_actions, landscape, character")
    subcategory: str = Field("general", description="Fine-grained category")
    content_title: Optional[str] = Field(None, description="Title of the media or narrative")
    description: Optional[str] = Field(None, description="Detailed scene description")
    characters: List[str] = Field(default_factory=list, description="Characters depicted")
    environment: str = Field("general", description="Environment: forest, mountain, palace, city")
    actions: List[str] = Field(default_factory=list, description="Visible actions or movements")
    animation_style: str = Field("live_action", description="Style: 2d_animation, 3d_cartoon, cinematic_3d, realistic")
    visual_style: str = Field("cinematic", description="Visual aesthetic: photorealistic, anime, watercolor")
    scene_type: str = Field("landscape", description="Framing: close_up, wide_shot, action_scene, aerial")
    camera_movement: str = Field("static", description="Camera movement: drone_aerial, panning, tracking, static")
    lighting: str = Field("natural", description="Lighting: golden_hour, sunset, dramatic, studio")
    license: str = Field("user-authorized", description="License identifier or declaration")
    license_url: Optional[str] = Field("", description="URL referencing license terms")
    creator: Optional[str] = Field("user", description="Author or rights-holder")
    attribution: Optional[str] = Field("", description="Attribution text")
    prompt: Optional[str] = Field("", description="Original prompt text")
    caption: Optional[str] = Field("", description="Original caption text")
    split: str = Field("train", description="Dataset split: train, validation, test")


class PrepareRequest(BaseModel):
    dataset_id: str = Field("default", description="Dataset ID to prepare")
    name: str = Field("Custom Dataset", description="Dataset display name")
    version: str = Field("v1.0", description="Version string")
    description: str = Field("", description="Version description")
    seed: int = Field(42, description="Partitioning seed for reproducibility")


@router.get("/api/v1/dataset/dashboard")
async def dataset_dashboard():
    """Returns dataset summary, sample OpenVid dataset status, and hardware training readiness."""
    root = settings.get_absolute_path(settings.DATASET_STORAGE)
    report_path = root / "dataset_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    except (OSError, ValueError):
        report = {}

    try:
        from training.utils.hardware import detect_hardware
        hardware = detect_hardware()
    except Exception:
        hardware = {"cuda_available": False, "recommended_device": "unknown"}

    # Existing OpenVid-1M sample dataset audit
    openvid_dir = WORKSPACE_ROOT / "dataset" / "ttv_data" / "training_data" / "openvid_1m"
    sample_dataset_info = {
        "present": openvid_dir.exists(),
        "total_clips": 0,
        "splits": {"train": 0, "validation": 0, "test": 0},
        "resolution": "720x480",
        "fps": 8.0,
        "frames_per_clip": 49,
        "license": "CC-BY-4.0 (OpenVid-1M / Research)",
        "prepared_manifest": str(WORKSPACE_ROOT / "dataset" / "manifests" / "openvid_1m_prepared_manifest.jsonl"),
    }
    if openvid_dir.exists():
        for s in ("train", "validation", "test"):
            vids = list((openvid_dir / s).glob("*.mp4"))
            sample_dataset_info["splits"][s] = len(vids)
        sample_dataset_info["total_clips"] = sum(sample_dataset_info["splits"].values())

    # Registered datasets count
    registry_dir = root / "registry"
    registered_datasets = []
    if registry_dir.exists():
        for f in registry_dir.glob("*.jsonl"):
            ds_id = f.stem
            items = _manager.load_items(ds_id)
            registered_datasets.append({
                "dataset_id": ds_id,
                "item_count": len(items),
                "training_eligible": sum(1 for it in items if it.training_eligible),
            })

    return {
        "dataset": report,
        "sample_dataset": sample_dataset_info,
        "registered_datasets": registered_datasets,
        "hardware": hardware,
        "training_enabled": settings.ENABLE_DATASET_TRAINING,
        "note": "BHIV TTV Studio Unified Dataset Management System",
    }


@router.post("/api/v1/dataset/collect")
async def collect_media(req: CollectRequest):
    """Collects media from a direct URL, YouTube, or text story description."""
    meta = req.model_dump()
    try:
        if req.source_type == "youtube":
            if not req.url:
                raise HTTPException(status_code=400, detail="Missing YouTube URL.")
            item = _pipeline.collect_from_youtube(req.url, meta, dataset_id=req.dataset_id)
        elif req.source_type == "url":
            if not req.url:
                raise HTTPException(status_code=400, detail="Missing media URL.")
            item = _pipeline.collect_from_url(req.url, meta, dataset_id=req.dataset_id)
        elif req.source_type == "text_story":
            if not req.story_text:
                raise HTTPException(status_code=400, detail="Missing story_text.")
            item = _pipeline.collect_text_story(req.story_text, meta, dataset_id=req.dataset_id)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported source_type '{req.source_type}'. Use url, youtube, or text_story.")

        return {
            "success": True,
            "item": item.to_dict(),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Collection failed: {exc}")


@router.post("/api/v1/dataset/collect/upload")
async def collect_upload(
    file: UploadFile = File(...),
    dataset_id: str = Form("default"),
    subject: str = Form("general"),
    topic: str = Form("general"),
    category: str = Form("general"),
    animation_style: str = Form("live_action"),
    visual_style: str = Form("cinematic"),
    scene_type: str = Form("landscape"),
    camera_movement: str = Form("static"),
    lighting: str = Form("natural"),
    license: str = Form("user-authorized"),
    prompt: str = Form(""),
    caption: str = Form(""),
    split: str = Form("train"),
):
    """Handles direct file upload of videos, images, or text files."""
    try:
        content = await file.read()
        metadata = {
            "subject": subject,
            "topic": topic,
            "category": category,
            "animation_style": animation_style,
            "visual_style": visual_style,
            "scene_type": scene_type,
            "camera_movement": camera_movement,
            "lighting": lighting,
            "license": license,
            "prompt": prompt,
            "caption": caption,
            "split": split,
        }
        item = _pipeline.collect_uploaded_file(content, file.filename, metadata, dataset_id=dataset_id)
        return {"success": True, "item": item.to_dict()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Upload ingestion failed: {exc}")


@router.get("/api/v1/dataset/items")
async def list_items(
    dataset_id: str = Query("default"),
    subject: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    visual_style: Optional[str] = Query(None),
    training_eligible_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Queries dataset items with optional domain and style filtering."""
    items = _manager.load_items(dataset_id)
    if subject:
        items = [it for it in items if it.subject.lower() == subject.lower()]
    if category:
        items = [it for it in items if it.category.lower() == category.lower()]
    if visual_style:
        items = [it for it in items if it.visual_style.lower() == visual_style.lower()]
    if training_eligible_only:
        items = [it for it in items if it.training_eligible and it.validation_status == ValidationStatus.VALID.value]

    total = len(items)
    page = items[offset : offset + limit]
    return {
        "dataset_id": dataset_id,
        "total": total,
        "items": [it.to_dict() for it in page],
    }


@router.post("/api/v1/dataset/validate")
async def validate_dataset(dataset_id: str = Form("default")):
    """Performs full integrity validation and deduplication across a dataset."""
    counts = _pipeline.validate_and_deduplicate(dataset_id)
    return {"success": True, "dataset_id": dataset_id, "summary": counts}


@router.post("/api/v1/dataset/prepare")
async def prepare_dataset(req: PrepareRequest):
    """Partitions dataset into train/val/test splits and exports a training manifest."""
    meta = _pipeline.prepare_training_dataset(
        dataset_id=req.dataset_id,
        name=req.name,
        version=req.version,
        description=req.description,
    )
    return {"success": True, "version_metadata": meta.to_dict()}


@router.post("/api/v1/dataset/sample/prepare")
async def prepare_sample_dataset():
    """Prepares and validates the existing OpenVid-1M sample dataset."""
    res = _pipeline.prepare_sample_openvid()
    return res


@router.get("/api/v1/dataset/manifests")
async def list_manifests():
    """Lists all prepared training manifests on disk."""
    manifests_dir = WORKSPACE_ROOT / "dataset" / "manifests"
    results = []
    if manifests_dir.exists():
        for mf in sorted(list(manifests_dir.glob("*.jsonl"))):
            count = sum(1 for line in mf.read_text(encoding="utf-8").splitlines() if line.strip())
            results.append({
                "filename": mf.name,
                "path": str(mf),
                "sample_count": count,
                "size_bytes": mf.stat().st_size,
            })
    return {"manifests": results}
