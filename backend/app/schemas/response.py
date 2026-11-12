from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SceneInfo(BaseModel):
    index: int
    prompt: str
    narrative: str
    duration: float
    image_url: Optional[str] = None
    video_url: Optional[str] = None


class VideoMetadata(BaseModel):
    video_id: str
    prompt: str
    dialogue: Optional[str] = None
    duration: float
    resolution: str
    aspect_ratio: Optional[str] = None
    fps: int
    style: str
    visual_style: Optional[str] = None
    language: Optional[str] = None
    voice_enabled: bool
    file_size_bytes: int
    video_url: str
    scenes: List[SceneInfo] = []
    created_at: str


class GenerationSettings(BaseModel):
    aspect_ratio: str
    quality: str
    resolution: str
    style: str
    visual_style: Optional[str] = None
    language: str
    dialogue: Optional[str] = None
    fps: int
    lipsync: bool = True
    character_id: Optional[str] = None
    subtitles: bool = True
    realtime_data: bool = False


class GenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str = "Generation job submitted successfully"
    settings: Optional[GenerationSettings] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    stage: str
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    aspect_ratio: Optional[str] = None
    visual_style: Optional[str] = None
    language: Optional[str] = None
    reference_type: Optional[str] = None
    reference_url: Optional[str] = None
    tts_status: Optional[str] = None
    video_status: Optional[str] = None
    lip_sync_status: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str = "unified-text-to-video"
    version: str = "1.0.0"
    providers: Dict[str, str]
    ffmpeg_available: bool


class ReferenceUploadResponse(BaseModel):
    reference_id: str
    media_type: str
    source: str = "upload"
    width: Optional[int] = None
    height: Optional[int] = None
    preview_url: Optional[str] = None
    still_url: Optional[str] = None
    status: str = "stored"
    auto_training_job_id: Optional[str] = None
    message: str = "Reference stored successfully"


class ReferenceDetailResponse(BaseModel):
    reference_id: str
    media_type: str
    source: str
    width: Optional[int] = None
    height: Optional[int] = None
    path: Optional[str] = None
    preview_url: Optional[str] = None
    auto_training_job_id: Optional[str] = None
    message: str = "Reference retrieved successfully"


class TrainingSessionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    reference_id: Optional[str] = None
    reference_info: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    logs: List[str] = []

