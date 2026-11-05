import asyncio
from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional

from app.schemas.request import (
    GenerationRequest,
    GenerationDebugRequest,
    ReferenceProcessRequest,
    SUPPORTED_STYLES,
)
from app.schemas.response import (
    GenerationResponse,
    GenerationSettings,
    ReferenceUploadResponse,
    ReferenceDetailResponse,
)
from app.core.queue import job_manager
from app.pipelines.text_to_video import pipeline
from app.utils.hashing import generate_execution_id
from app.services.pipeline_diagnostic import diagnose_pipeline_execution
from app.core.config import settings
from app.core.logging import telemetry
from app.utils.video_settings import resolve_video_settings, catalog as video_catalog
from app.utils.languages import catalog as language_catalog
from app.services.reference_service import reference_service

router = APIRouter()


def _resolved_settings(request: GenerationRequest) -> GenerationSettings:
    video_cfg = resolve_video_settings(request.aspect_ratio, request.quality, request.resolution)
    effective_style = request.visual_style or request.style or "cinematic"
    return GenerationSettings(
        aspect_ratio=video_cfg["aspect_ratio"],
        quality=video_cfg["quality"],
        resolution=video_cfg["resolution"],
        style=effective_style,
        visual_style=effective_style,
        language=request.language or "en",
        dialogue=request.dialogue,
        fps=request.fps or 24,
        lipsync=True if request.lipsync is None else request.lipsync,
        character_id=request.character_id,
        subtitles=True if request.subtitles is None else request.subtitles,
        realtime_data=False if request.realtime_data is None else request.realtime_data,
    )


async def _run_generation_task(job_id: str, request: GenerationRequest):
    """Background execution runner with progress updates."""
    try:
        async def _progress_callback(stage: str, pct: int):
            await job_manager.update_progress(job_id, stage, pct)

        result = await pipeline.execute(
            prompt=request.prompt,
            dialogue=request.dialogue,
            duration=request.duration or 15,
            style=request.visual_style or request.style or "cinematic",
            voice=request.voice if request.voice is not None else True,
            token=request.token,
            job_id=job_id,
            progress_callback=_progress_callback,
            aspect_ratio=request.aspect_ratio,
            quality=request.quality,
            resolution=request.resolution,
            fps=request.fps,
            language=request.language or "en",
            voice_id=request.voice_id,
            reference_url=request.reference_url,
            reference_type=request.reference_type,
            reference_id=request.reference_id or request.reference_file,
            lipsync=True if request.lipsync is None else request.lipsync,
            model_mode=request.model_mode,
            character_id=request.character_id,
            subtitles=True if request.subtitles is None else request.subtitles,
            realtime_data=False if request.realtime_data is None else request.realtime_data,
            music=True if request.music is None else request.music,
        )
        await job_manager.complete_job(job_id, result.get("metadata", {}))
    except Exception as e:
        telemetry.emit("job_execution_failed", job_id, {"error": str(e)}, level="error")
        await job_manager.fail_job(job_id, str(e))


@router.get("/generation-options")
async def generation_options():
    """Catalog of formats, qualities, styles, and languages for the studio UI."""
    return {
        **video_catalog(),
        **language_catalog(),
        "styles": list(SUPPORTED_STYLES),
        "defaults": {
            "aspect_ratio": settings.DEFAULT_ASPECT_RATIO,
            "quality": settings.DEFAULT_QUALITY,
            "language": settings.DEFAULT_LANGUAGE,
            "style": settings.DEFAULT_STYLE,
            "fps": settings.DEFAULT_FPS,
        },
    }


@router.get("/characters")
async def characters():
    """List built-in persistent character profiles safe to select in the studio."""
    from app.services.character_service import character_service
    return {"characters": character_service.list_profiles()}


@router.post("/references/upload", response_model=ReferenceUploadResponse)
async def upload_reference(
    file: UploadFile = File(...),
    reference_type: Optional[str] = Form(default=None),
):
    content = await file.read()
    payload = reference_service.save_upload(file.filename or "reference.bin", content, file.content_type)
    return ReferenceUploadResponse(
        reference_id=payload["reference_id"],
        media_type=payload["media_type"],
        source="upload",
        width=payload.get("width"),
        height=payload.get("height"),
        preview_url=payload.get("preview_url"),
        still_url=payload.get("still_path") or payload.get("preview_url"),
        status="stored",
        message="Reference uploaded successfully",
    )


@router.post("/references/url", response_model=ReferenceDetailResponse)
@router.post("/references/process", response_model=ReferenceDetailResponse)
async def process_reference_url(request: ReferenceProcessRequest):
    """
    Ingest and process a public reference media URL (YouTube, Instagram, Facebook, X, or Direct URL).
    """
    payload = await reference_service.ingest_url(request.url)
    return ReferenceDetailResponse(
        reference_id=payload["reference_id"],
        media_type=payload["media_type"],
        source=payload.get("source", "url"),
        width=payload.get("width"),
        height=payload.get("height"),
        path=payload.get("path"),
        preview_url=payload.get("preview_url"),
        message="Reference URL processed successfully",
    )


@router.get("/references/{reference_id}", response_model=ReferenceDetailResponse)
async def get_reference_details(reference_id: str):
    """
    Retrieve stored reference media metadata.
    """
    payload = reference_service.load_existing(reference_id)
    return ReferenceDetailResponse(
        reference_id=payload["reference_id"],
        media_type=payload["media_type"],
        source=payload.get("source", "upload"),
        width=payload.get("width"),
        height=payload.get("height"),
        path=payload.get("path"),
        preview_url=payload.get("preview_url"),
        message="Reference retrieved successfully",
    )



@router.post("/generate", response_model=GenerationResponse)
async def generate_video(request: GenerationRequest):
    """
    Initiate video generation workflow.
    Validates payload, initializes queue job, and dispatches processing asynchronously.
    """
    job_id = generate_execution_id()
    resolved = _resolved_settings(request)
    telemetry.emit("request_received", job_id, {
        "received_prompt": request.prompt,
        "dialogue": request.dialogue,
        "generation_request_id": job_id,
        "models_used": {
            "llm": settings.LLM_PROVIDER,
            "image": settings.IMAGE_PROVIDER,
            "video": settings.VIDEO_PROVIDER,
            "tts": settings.TTS_PROVIDER
        },
        "generation_parameters": resolved.model_dump()
    })
    await job_manager.create_job(job_id, request.model_dump(), settings=resolved.model_dump())

    asyncio.create_task(_run_generation_task(job_id, request))

    return GenerationResponse(
        job_id=job_id,
        status="queued",
        message="Video generation job queued successfully",
        settings=resolved,
    )


@router.post("/generate/debug")
async def generate_debug(request: GenerationDebugRequest):
    """
    Diagnostic endpoint that traces prompt tokenization, text embeddings,
    model architecture, checkpoint metadata, device, latent shapes,
    and runs a lightweight pipeline trace.
    """
    return diagnose_pipeline_execution(
        prompt=request.prompt,
        seed=request.seed or 42,
        fps=request.fps or 24,
        num_frames=request.num_frames or 16,
        resolution=request.resolution or "1280x720",
    )

