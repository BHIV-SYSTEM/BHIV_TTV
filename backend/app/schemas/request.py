from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from app.utils.video_settings import (
    ASPECT_RATIOS,
    QUALITIES,
    normalize_aspect_ratio,
    normalize_quality,
    resolve_video_settings,
)
from app.utils.languages import SUPPORTED_LANGUAGES, normalize_language
from app.adapters.reference.validator import validate_public_url, ReferenceValidationError
from app.core.exceptions import ValidationError as TTVValidationError

SUPPORTED_STYLES = (
    "cinematic",
    "realistic",
    "cartoon",
    "3d",
    "anime",
    "fantasy",
    "cyberpunk",
    "documentary",
    "custom",
)


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Text prompt describing the video to generate")
    dialogue: Optional[str] = Field(default=None, description="Spoken dialogue text for character narration and lip-sync")
    duration: Optional[int] = Field(default=15, ge=5, le=60, description="Desired video duration in seconds (5-60s)")
    style: Optional[str] = Field(default="cinematic", description="Visual style preset")
    visual_style: Optional[str] = Field(default=None, description="Visual style preset (alias for style)")
    voice: Optional[bool] = Field(default=True, description="Whether to generate speech narration")
    resolution: Optional[str] = Field(default=None, description="Legacy target resolution (e.g. 1280x720). Prefer aspect_ratio + quality.")
    fps: Optional[int] = Field(default=24, ge=15, le=60, description="Frames per second")
    token: Optional[str] = Field(default=None, description="Optional governance authorization token")
    model_mode: Optional[str] = Field(default="base", description="Generation engine mode: base or finetuned")
    aspect_ratio: Optional[str] = Field(default=None, description="Video format: 16:9 or 9:16")
    quality: Optional[str] = Field(default=None, description="Video quality: standard (720p), high (1080p), ultra (4K)")
    language: Optional[str] = Field(default="en", description="Narration language code (en, hi, mr, gu, fr, es, de, etc.)")
    voice_id: Optional[str] = Field(default=None, description="Optional TTS voice id compatible with the selected language")
    reference_url: Optional[str] = Field(default=None, description="Public image/video URL used as visual reference")
    reference_type: Optional[str] = Field(default=None, description="Optional hint: image or video")
    reference_id: Optional[str] = Field(default=None, description="ID returned by POST /api/v1/references/upload")
    reference_file: Optional[str] = Field(default=None, description="Uploaded reference file or reference ID")
    lipsync: Optional[bool] = Field(default=True, description="Apply automatic character lip-sync where a face is visible")
    character_id: Optional[str] = Field(default=None, max_length=80, description="Saved character profile ID or name")
    subtitles: Optional[bool] = Field(default=True, description="Create SRT and WebVTT from the dialogue sent to TTS")
    music: Optional[bool] = Field(default=True, description="Include the configured ambient music bed")
    realtime_data: Optional[bool] = Field(default=False, description="Request configured live-data enrichment; it fails closed if unavailable")

    @field_validator("aspect_ratio")
    @classmethod
    def _validate_aspect(cls, value):
        if value is None:
            return value
        normalized = normalize_aspect_ratio(value)
        if normalized not in ASPECT_RATIOS:
            raise ValueError(f"Unsupported aspect ratio. Use one of: {', '.join(ASPECT_RATIOS)}")
        return normalized

    @field_validator("quality")
    @classmethod
    def _validate_quality(cls, value):
        if value is None:
            return value
        normalized = normalize_quality(value)
        if normalized not in QUALITIES:
            raise ValueError(f"Unsupported quality. Use one of: {', '.join(QUALITIES)}")
        return normalized

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value):
        if value is None:
            return "en"
        try:
            return normalize_language(value)
        except TTVValidationError as exc:
            raise ValueError(exc.message) from exc

    @field_validator("style", "visual_style")
    @classmethod
    def _validate_style(cls, value):
        if value is None:
            return None
        key = str(value).strip().lower()
        aliases = {
            "photorealistic": "realistic",
            "realistic / photorealistic": "realistic",
            "cartoonish": "cartoon",
            "cartoon / cartoonish": "cartoon",
            "film": "cinematic",
            "film look": "cinematic",
            "cinematic / film look": "cinematic",
            "stylized": "3d",
            "3d / stylized": "3d",
            "documentary": "documentary",
            "custom": "custom",
        }
        key = aliases.get(key, key)
        if key not in SUPPORTED_STYLES:
            raise ValueError(f"Unsupported style. Use one of: {', '.join(SUPPORTED_STYLES)}")
        return key

    @field_validator("reference_type")
    @classmethod
    def _validate_reference_type(cls, value):
        if value in (None, "", "none"):
            return None
        key = str(value).strip().lower()
        if key not in {"image", "video"}:
            raise ValueError("reference_type must be 'image' or 'video'.")
        return key

    @field_validator("reference_url")
    @classmethod
    def _validate_reference_url(cls, value):
        if value in (None, ""):
            return None
        try:
            safe, _source = validate_public_url(value, resolve_dns=False)
            return safe
        except (ReferenceValidationError, TTVValidationError) as exc:
            raise ValueError(getattr(exc, "message", str(exc))) from exc

    @field_validator("reference_id")
    @classmethod
    def _validate_reference_id(cls, value):
        if value in (None, ""):
            return None
        key = str(value).strip()
        if not key.startswith("ref_") or len(key) > 64:
            raise ValueError("Invalid reference_id.")
        return key

    @model_validator(mode="after")
    def _defaults(self):
        # Sync visual_style and style
        if self.visual_style and (not self.style or self.style == "cinematic"):
            self.style = self.visual_style
        elif self.style and not self.visual_style:
            self.visual_style = self.style
        if not self.style:
            self.style = "cinematic"
            self.visual_style = "cinematic"

        # Sync reference_file and reference_id
        if self.reference_file and not self.reference_id:
            cleaned_ref = str(self.reference_file).strip()
            if cleaned_ref.startswith("ref_"):
                self.reference_id = cleaned_ref

        if not self.language:
            self.language = "en"
        if self.fps is None:
            self.fps = 24
        if self.duration is None:
            self.duration = 15
        if self.lipsync is None:
            self.lipsync = True
        if self.voice is None:
            self.voice = True
        if self.subtitles is None:
            self.subtitles = True
        if self.music is None:
            self.music = True
        if self.realtime_data is None:
            self.realtime_data = False
        resolved = resolve_video_settings(self.aspect_ratio, self.quality, self.resolution)
        self.aspect_ratio = resolved["aspect_ratio"]
        self.quality = resolved["quality"]
        self.resolution = resolved["resolution"]
        return self

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "prompt": "A young farmer standing in a green field during sunrise",
                "dialogue": "आज आपण आपल्या शेतातील नवीन पिकाबद्दल माहिती घेणार आहोत.",
                "duration": 15,
                "aspect_ratio": "16:9",
                "quality": "high",
                "visual_style": "cinematic",
                "style": "cinematic",
                "language": "mr",
                "voice": True,
                "reference_url": None,
                "reference_type": None,
                "fps": 24
            }
        }
    }


class ReferenceProcessRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048, description="Public reference media URL")
    reference_type: Optional[str] = Field(default=None, description="Optional hint: image or video")

    @field_validator("reference_type")
    @classmethod
    def _validate_ref_type(cls, value):
        if value in (None, "", "none"):
            return None
        key = str(value).strip().lower()
        if key not in {"image", "video"}:
            raise ValueError("reference_type must be 'image' or 'video'.")
        return key

    @field_validator("url")
    @classmethod
    def _validate_url(cls, value):
        if not value or not str(value).strip():
            raise ValueError("Reference URL cannot be empty.")
        safe, _ = validate_public_url(value, resolve_dns=False)
        return safe


class TrainingSessionRequest(BaseModel):
    base_model: Optional[str] = Field(default="SpatialTemporalTTVModel", description="Base model name or path")
    method: Optional[str] = Field(default="lora", description="Fine-tuning method (lora, full, temporal_only)")
    epochs: Optional[int] = Field(default=10, ge=1, le=100, description="Number of training epochs")
    learning_rate: Optional[str] = Field(default="0.0001", description="Training learning rate")
    batch_size: Optional[int] = Field(default=2, ge=1, le=16, description="Batch size")
    reference_id: Optional[str] = Field(default=None, description="Reference identifier")
    reference_url: Optional[str] = Field(default=None, description="Public reference media URL")
    reference_type: Optional[str] = Field(default=None, description="Optional hint: image or video")
    dataset_manifest: Optional[str] = Field(default=None, description="Optional path to custom dataset manifest")
    version_name: Optional[str] = Field(default="ttv_lora_v001", description="Model version tag")

    model_config = {
        "protected_namespaces": ()
    }
class GenerationDebugRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Text prompt to trace through the pipeline")
    seed: Optional[int] = Field(default=42, description="Random seed for generation")
    fps: Optional[int] = Field(default=24, ge=1, le=60, description="Frames per second")
    num_frames: Optional[int] = Field(default=16, ge=1, le=120, description="Number of frames to generate")
    resolution: Optional[str] = Field(default="1280x720", description="Resolution string (e.g. 1280x720)")

