from contextlib import asynccontextmanager
import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from app.core.config import settings
from app.core.logging import logger, telemetry
from app.core.exceptions import TTVException, ValidationError, GovernanceViolationError
from app.api.routes import generate, jobs, videos, health, training

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
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
app.include_router(training.router, prefix="/api/v1/training", tags=["Training"])


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
