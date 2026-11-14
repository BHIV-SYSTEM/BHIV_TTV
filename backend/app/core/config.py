import os
from typing import Optional, List, Dict
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

# Determine project base directory (workspace root)
# If backend/app/core/config.py, parents[2] is backend, parents[3] is text_to_video_final
CURRENT_FILE = Path(__file__).resolve()
WORKSPACE_ROOT = CURRENT_FILE.parents[3]

class Settings(BaseSettings):
    WORKSPACE_ROOT: Path = WORKSPACE_ROOT

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"

    # Security & Governance
    GOVERNANCE_LOCK: str = "ON"
    WRAPPER_TOKEN: str = "TTV_SECURE_GOVERNED_TOKEN_2026"

    # Providers
    LLM_PROVIDER: str = "local"
    IMAGE_PROVIDER: str = "local"
    VIDEO_PROVIDER: str = "opencv"
    TTS_PROVIDER: str = "local"
    VISION_PROVIDER: str = "standard"

    # Model Mode (base | wan_lora | finetuned).  ``finetuned`` is retained for
    # the small legacy SpatialTemporal research model; production Wan adapters
    # must use ``wan_lora`` so incompatible checkpoints are never loaded.
    MODEL_MODE: str = "base"
    FINE_TUNED_CHECKPOINT_PATH: Optional[str] = None

    # ── Wan 2.2 TI2V-5B Video Generation ─────────────────────────────────
    # Requires NVIDIA GPU at inference time (not needed for local/opencv mode).
    # Set VIDEO_PROVIDER=wan in your cloud environment .env to enable.
    WAN_MODEL_ID: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    WAN_DEVICE: str = "cuda"
    WAN_DTYPE: str = "bfloat16"
    WAN_NUM_FRAMES: int = 81
    WAN_INFERENCE_STEPS: int = 30
    WAN_GUIDANCE_SCALE: float = 5.0
    WAN_LORA_PATH: Optional[str] = None
    WAN_LORA_SCALE: float = 1.0
    # Optional HuggingFace token for gated/private model access
    HF_TOKEN: Optional[str] = None

    # ── CogVideoX-2B Video Generation ────────────────────────────────────
    # High-fidelity Diffusers text-to-video foundation model.
    # Requires NVIDIA GPU with CUDA. Set VIDEO_PROVIDER=cogvideox to enable.
    COGVIDEOX_MODEL_ID: str = "THUDM/CogVideoX-2b"
    COGVIDEOX_DEVICE: str = "cuda"
    COGVIDEOX_DTYPE: str = "bfloat16"
    COGVIDEOX_CPU_OFFLOAD: bool = True
    COGVIDEOX_SEQUENTIAL_CPU_OFFLOAD: bool = False
    COGVIDEOX_ENABLE_TILING: bool = True
    COGVIDEOX_ENABLE_SLICING: bool = True
    COGVIDEOX_NUM_FRAMES: int = 49
    COGVIDEOX_NUM_INFERENCE_STEPS: int = 30
    COGVIDEOX_GUIDANCE_SCALE: float = 6.0
    COGVIDEOX_DEFAULT_WIDTH: int = 720
    COGVIDEOX_DEFAULT_HEIGHT: int = 480
    COGVIDEOX_LORA_PATH: Optional[str] = None
    COGVIDEOX_LORA_SCALE: float = 1.0

    # Semantic adherence needs an explicitly configured VLM evaluator. The
    # deterministic constraints layer is always available, but does not claim a score.
    PROMPT_ADHERENCE_PROVIDER: str = "disabled"
    MAX_VIDEO_RETRIES: int = 2


    # API Keys
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # Paths (relative to workspace root or absolute)
    OUTPUT_DIR: str = "generated/videos"
    TEMP_DIR: str = "generated/temp"
    IMAGES_DIR: str = "generated/images"
    SCENES_DIR: str = "generated/scenes"
    AUDIO_DIR: str = "generated/audio"
    REFERENCES_DIR: str = "generated/references"
    FRONTEND_DIR: str = "frontend"

    # Generation Defaults
    DEFAULT_DURATION: int = 15
    MAX_DURATION: int = 60
    DEFAULT_FPS: int = 24
    DEFAULT_RESOLUTION: str = "1280x720"
    DEFAULT_STYLE: str = "cinematic"
    DEFAULT_VOICE: str = "en-US-GuyNeural"
    DEFAULT_ASPECT_RATIO: str = "16:9"
    DEFAULT_QUALITY: str = "standard"
    DEFAULT_LANGUAGE: str = "en"

    # Dataset / training policy.  These flags deliberately keep data acquisition
    # and model fine-tuning separate from normal generation.
    DATASET_STORAGE: str = "data"
    DATASET_MAX_DOWNLOAD_GB: int = 100
    ENABLE_DATASET_TRAINING: bool = False

    # Real-time facts are opt-in and only considered verified when the configured
    # provider responds successfully.  No embedded/sample market data is used.
    ENABLE_REALTIME_DATA: bool = False
    REALTIME_DATA_URL: Optional[str] = None
    REALTIME_DATA_TIMEOUT: int = 10

    # Reference retrieval
    REFERENCE_MAX_BYTES: int = 104857600
    REFERENCE_DOWNLOAD_TIMEOUT: int = 30

    # Hardware & FFmpeg Configuration
    FFMPEG_PATH: Optional[str] = None
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    DEFAULT_GPU_INDEX: int = 0

    # Lip-sync
    LIPSYNC_ENABLED: bool = True
    LIPSYNC_PROVIDER: str = "viseme"
    LIPSYNC_MODEL_PATH: Optional[str] = None
    MUSETALK_CHECKPOINT_PATH: Optional[str] = None

    # Audio Defaults
    BACKGROUND_MUSIC_ENABLED: bool = True
    AUDIO_DUCKING_RATIO: float = 0.3
    AUDIO_SAMPLE_RATE: int = 44100

    model_config = {
        "env_file": [str(WORKSPACE_ROOT / ".env"), ".env"],
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

    def get_absolute_path(self, relative_path: str) -> Path:
        """Resolve a path relative to the workspace root if not already absolute."""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return (WORKSPACE_ROOT / p).resolve()

    def ensure_directories(self):
        """Ensure all required asset directories exist."""
        for path_attr in ["OUTPUT_DIR", "TEMP_DIR", "IMAGES_DIR", "SCENES_DIR", "AUDIO_DIR", "REFERENCES_DIR"]:
            p = self.get_absolute_path(getattr(self, path_attr))
            p.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
