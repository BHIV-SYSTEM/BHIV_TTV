from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class AutomaticTrainingRequest(BaseModel):
    reference_id: str = Field(..., pattern=r"^ref_[a-f0-9]{12}$")
    base_model: str = Field(default="SpatialTemporalTTVModel", min_length=1)
    method: Literal["lora", "full", "temporal_only"] = "lora"
    epochs: int = Field(default=10, ge=1, le=100)
    learning_rate: float = Field(default=0.0001, gt=0.0)
    batch_size: int = Field(default=2, ge=1, le=64)
    retry: bool = False


class AutomaticTrainingJobResponse(BaseModel):
    job_id: str
    status: str
    stage: str
    progress: int = 0
    epoch: int = 0
    total_epochs: int = 0
    loss: Optional[float] = None
    device: str = "cpu"
    reference_id: Optional[str] = None
    training_job_id: Optional[str] = None
    cuda_available: bool = False
    hardware: Optional[Dict[str, Any]] = None
    checkpoints: List[str] = []
    final_checkpoint: Optional[str] = None
    metrics: Dict[str, Any] = {}
    error: Optional[str] = None
    message: Optional[str] = None


class TrainingJobConfig(BaseModel):
    base_model: str = Field(default="SpatialTemporalTTVModel", description="Base generative model architecture")
    method: str = Field(default="lora", description="Fine-tuning method: lora, full, or temporal_only")
    epochs: int = Field(default=10, ge=1, le=100, description="Number of training epochs")
    learning_rate: float = Field(default=0.0001, gt=0.0, description="Learning rate")
    batch_size: int = Field(default=2, ge=1, le=64, description="Training batch size")
    version_name: Optional[str] = Field(default=None, description="Optional version tag for registration")
    manifest_path: Optional[str] = Field(default=None, description="Optional path to existing manifest file on server")


class ManifestValidationResponse(BaseModel):
    valid: bool
    total_records: int
    valid_records: int
    rejected_records: int
    splits: Dict[str, int] = {}
    categories: Dict[str, int] = {}
    formats: Dict[str, int] = {}
    has_remote_urls: bool = False
    sample_preview: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    manifest_path: Optional[str] = None
    summary: str
    dataset_name: Optional[str] = None


class TrainingJobResponse(BaseModel):
    job_id: str
    status: str  # "queued", "running", "completed", "failed", "CUDA_UNAVAILABLE"
    dataset_samples: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    device: str = "cpu"
    progress: int = 0
    epoch: int = 0
    total_epochs: int = 0
    loss: Optional[float] = None
    stage: str = "queued"
    created_at: str = ""
    updated_at: str = ""
    config: Dict[str, Any] = {}
    manifest_stats: Optional[Dict[str, Any]] = None
    manifest_path: Optional[str] = None
    cuda_available: bool = False
    hardware: Optional[Dict[str, Any]] = None
    cli_fallback_command: str = ""
    checkpoints: List[str] = []
    final_checkpoint: Optional[str] = None
    metrics: Dict[str, Any] = {}
    error: Optional[str] = None
    message: Optional[str] = None


class TrainingLogsResponse(BaseModel):
    job_id: str
    status: str
    total_lines: int
    logs: List[str]
    cli_fallback_command: Optional[str] = None
