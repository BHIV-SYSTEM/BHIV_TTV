from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import time
from typing import Optional
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from app.core.config import settings
from app.core.logging import logger, telemetry
from app.core.exceptions import TTVException, ValidationError, GovernanceViolationError
from app.api.routes import generate, jobs, videos, health, dataset, training


def _materialize_bundled_manifest_120(
    jsonl_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
    lock_timeout: float = 5.0
):
    """
    Safely and atomically materializes the bundled 120-sample training manifest JSONL
    from its source JSON if the JSONL file does not already exist.
    Uses cross-process file locking (via the established filelock dependency) to
    guarantee an existing destination is never overwritten, even under concurrent
    multi-process startup.
    """
    jsonl_p = jsonl_path or settings.get_absolute_path("data/manifests/ttv_training_manifest_120.jsonl")
    if jsonl_p.exists():
        return

    json_p = json_path or settings.get_absolute_path("data/manifests/ttv_training_manifest_120.json")
    if not json_p.exists():
        return

    tmp_p = None
    lock_path = jsonl_p.with_name(f".{jsonl_p.name}.lock")

    try:
        from app.services.training_service import parse_manifest_records
        records, _ = parse_manifest_records(json_p)
        if not records:
            return

        jsonl_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = jsonl_p.with_name(f".ttv_training_manifest_120_{uuid.uuid4().hex}.tmp")
        with open(tmp_p, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        # Cross-process safe create-if-absent publication:
        # Use existing cross-platform filelock to serialize the critical publication check
        from filelock import FileLock, Timeout
        lock = FileLock(str(lock_path), timeout=lock_timeout)

        try:
            with lock:
                # Inside critical section: verify destination is still absent before atomic publish
                if not jsonl_p.exists():
                    os.replace(tmp_p, jsonl_p)
                    tmp_p = None  # Consumed by os.replace
                    logger.info("Materialized bundled 120-record manifest at %s", jsonl_p)
        except Timeout:
            if not jsonl_p.exists():
                logger.warning(
                    "Timed out waiting for bundled manifest materialization lock; "
                    "destination %s is still absent.",
                    jsonl_p
                )
    except Exception as e:
        logger.warning(f"Failed to auto-materialize 120 jsonl manifest: {e}")
    finally:
        if tmp_p and tmp_p.exists():
            try:
                tmp_p.unlink()
            except OSError:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
    _materialize_bundled_manifest_120()
    logger.info("Unified Text-to-Video Engine initialized successfully.")
    telemetry.emit("system_startup", "system", {
        "llm_provider": settings.LLM_PROVIDER,
        "video_provider": settings.VIDEO_PROVIDER,
        "tts_provider": settings.TTS_PROVIDER
    })
    yield

app = FastAPI(
    title="Unified Text-to-Video Engine",
    version="1.0.0",
    description="Consolidated production-ready Text-to-Video AI microservice.",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={"error": "VALIDATION_FAILED", "message": exc.message, "details": exc.details}
    )

@app.exception_handler(GovernanceViolationError)
async def governance_exception_handler(request: Request, exc: GovernanceViolationError):
    return JSONResponse(
        status_code=403,
        content={"error": "GOVERNANCE_VIOLATION", "message": exc.message}
    )

@app.exception_handler(TTVException)
async def ttv_exception_handler(request: Request, exc: TTVException):
    return JSONResponse(
        status_code=500,
        content={"error": "PIPELINE_ERROR", "message": exc.message, "details": exc.details}
    )

# Include API Routers
app.include_router(generate.router, prefix="/api/v1", tags=["Generation"])
app.include_router(jobs.router, prefix="/api/v1", tags=["Jobs"])
app.include_router(videos.router, prefix="/api/v1", tags=["Videos"])
app.include_router(training.router, prefix="/api/v1/training", tags=["Training"])
app.include_router(health.router, tags=["Health"])
app.include_router(dataset.router, tags=["Dataset"])


# Mount Generated Media Directory for Direct Streaming
generated_dir = settings.get_absolute_path("generated")
if generated_dir.exists():
    app.mount("/generated", StaticFiles(directory=str(generated_dir)), name="generated")

# Serve Frontend at Root
frontend_dir = settings.get_absolute_path(settings.FRONTEND_DIR)
if frontend_dir.exists():
    @app.get("/")
    async def serve_index():
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Unified Text-to-Video API is running. Frontend index.html not found."}

    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
