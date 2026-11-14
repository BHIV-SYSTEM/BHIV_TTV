"""
Data models for dataset ingestion, provenance, licensing, and quality assurance.
Supports flexible, extensible metadata for any subject domain, animation style, or character type.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PermissionState(str, Enum):
    LICENSE_VERIFIED = "LICENSE_VERIFIED"
    PERMISSION_RECORDED = "PERMISSION_RECORDED"
    RESTRICTED = "RESTRICTED"
    UNKNOWN = "UNKNOWN"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ValidationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    DUPLICATE = "DUPLICATE"
    PROCESSING_FAILED = "PROCESSING_FAILED"


@dataclass
class LicenseInfo:
    license: str
    license_url: Optional[str] = None
    creator: Optional[str] = None
    attribution_required: bool = True
    commercial_allowed: bool = True
    training_eligible: bool = True


@dataclass
class QualityMetrics:
    width: int = 0
    height: int = 0
    aspect_ratio: str = "16:9"
    blur_score: float = 0.0
    is_corrupt: bool = False
    is_black_frame: bool = False
    has_audio: bool = False
    speech_detected: bool = False
    snr_db: Optional[float] = None
    phash: Optional[str] = None
    passed_qc: bool = True
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class CaptionMetadata:
    caption: str
    subjects: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    environment: str = "general"
    camera: str = "medium shot"
    lighting: str = "natural daylight"
    weather: str = "clear"
    time: str = "day"
    emotion: str = "neutral"
    motion: str = "static"


@dataclass
class AssetMetadata:
    asset_id: str
    source_url: str
    source_name: str
    license: str
    license_url: Optional[str] = None
    creator: Optional[str] = None
    download_time: str = field(default_factory=_utcnow_iso)
    media_type: str = "image"  # image | video | audio
    resolution: str = "1280x720"
    duration: float = 0.0
    sha256: str = ""
    phash: Optional[str] = None
    dataset_version: str = "v1.0"
    allowed_for_training: bool = True
    category: str = "general"
    local_path: Optional[str] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    caption_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    total_downloaded: int = 0
    valid: int = 0
    duplicates: int = 0
    low_quality: int = 0
    license_unknown: int = 0
    corrupted: int = 0
    training_ready: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    generated_at: str = field(default_factory=_utcnow_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetItem:
    """
    Unified, domain-agnostic metadata record for video, image, or text training items.
    Supports flexible, extensible classification (mythology, history, 3D cartoon, realistic, etc.)
    without hardcoding any single subject.
    """
    media_id: str
    dataset_id: str = "default"
    subject: str = "general"                   # e.g. mythology, history, nature, sci-fi, cartoon, fantasy
    topic: str = "general"                     # e.g. ramayana, space_exploration, wildlife, aerial
    category: str = "general"                  # e.g. human_actions, landscape, character_dialogue, animals
    subcategory: str = "general"
    content_title: str = ""
    description: str = ""
    characters: List[str] = field(default_factory=list)
    environment: str = "general"
    actions: List[str] = field(default_factory=list)
    animation_style: str = "live_action"       # e.g. 2d_animation, 3d_cartoon, cinematic_3d, realistic
    visual_style: str = "cinematic"            # e.g. photorealistic, anime, watercolor, oil_painting
    scene_type: str = "landscape"              # e.g. close_up, wide_shot, action_scene, aerial
    camera_movement: str = "static"            # e.g. drone_aerial, panning, tracking, zoom, static
    lighting: str = "natural"                  # e.g. golden_hour, sunset, dramatic_shadows, studio
    source_type: str = "upload"                # e.g. upload, web_url, youtube, text_story, openvid
    source_url: str = ""
    license: str = "unknown"
    license_url: str = ""
    creator: str = ""
    permission_state: str = PermissionState.NEEDS_REVIEW.value
    commercial_allowed: bool = False
    training_eligible: bool = False
    attribution: str = ""
    collection_timestamp: str = field(default_factory=_utcnow_iso)
    media_type: str = "video"                  # video | image | text
    original_prompt: str = ""
    original_caption: str = ""
    generated_caption: str = ""
    caption_source: str = "user_supplied"      # user_supplied | vlm_generated | structured_synthesis | metadata
    duration: float = 0.0
    resolution: str = "720x480"
    fps: float = 24.0
    frame_count: int = 0
    keyframe_paths: List[str] = field(default_factory=list)
    thumbnail_path: Optional[str] = None
    storage_path: str = ""
    sha256: str = ""
    phash: Optional[str] = None
    duplicate_status: str = "unique"           # unique | duplicate_exact | duplicate_perceptual
    duplicate_of: Optional[str] = None
    validation_status: str = ValidationStatus.NEEDS_REVIEW.value
    validation_reasons: List[str] = field(default_factory=list)
    dataset_split: str = "train"               # train | validation | test | unassigned
    review_notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def final_caption(self) -> str:
        """Returns the primary training caption: original user caption if present, else generated caption."""
        return self.original_caption or self.original_prompt or self.generated_caption or self.description

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetItem":
        # Filter unknown keys to prevent TypeError
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class DatasetVersionMetadata:
    """Version metadata tracking dataset size, quality metrics, and training readiness."""
    dataset_id: str
    name: str
    version: str = "v1.0"
    created_at: str = field(default_factory=_utcnow_iso)
    description: str = ""
    total_media: int = 0
    valid_media: int = 0
    invalid_media: int = 0
    duplicate_count: int = 0
    needs_review_count: int = 0
    training_eligible_count: int = 0
    subjects_summary: Dict[str, int] = field(default_factory=dict)
    categories_summary: Dict[str, int] = field(default_factory=dict)
    visual_styles_summary: Dict[str, int] = field(default_factory=dict)
    sources_summary: Dict[str, int] = field(default_factory=dict)
    splits_summary: Dict[str, int] = field(default_factory=dict)
    storage_directory: str = ""
    manifest_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
