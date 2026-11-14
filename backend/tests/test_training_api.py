import os
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.services.training_service import (
    parse_manifest_records,
    validate_manifest_records,
    check_cuda_availability
)

client = TestClient(app)

_cand_paths = [
    Path(__file__).resolve().parents[2] / "data" / "manifests" / "ttv_training_manifest_120.json",
    Path(__file__).resolve().parents[2] / "data" / "manifests" / "ttv_training_manifest_120.jsonl",
    Path(__file__).resolve().parents[3] / "ttv_training_manifest_120.json",
    Path(__file__).resolve().parents[3] / "ttv_training_manifest_120.jsonl"
]
MANIFEST_120_PATH = next((p for p in _cand_paths if p.exists()), _cand_paths[0])


def test_120_record_manifest_validation():
    """Verify that the 120-record manifest validates without dataset modifications."""
    assert MANIFEST_120_PATH.exists(), f"120 manifest not found at {MANIFEST_120_PATH}"
    records, resolved_path = parse_manifest_records(MANIFEST_120_PATH)
    assert len(records) == 120, f"Expected exactly 120 records, got {len(records)}"

    res = validate_manifest_records(records)
    assert res["valid"] is True
    assert res["total_records"] == 120
    assert res["valid_records"] == 120
    assert res["rejected_records"] == 0
    assert res["splits"]["train"] == 90
    assert res["splits"]["val"] == 18
    assert res["splits"]["test"] == 12
    assert len(res["categories"]) == 12
    assert res["has_remote_urls"] is True


def test_manifest_validation_api_endpoint():
    """Verify POST /api/v1/training/manifest/validate with 120-record manifest."""
    with open(MANIFEST_120_PATH, "rb") as f:
        response = client.post(
            "/api/v1/training/manifest/validate",
            files={"manifest_file": ("manifest.jsonl", f, "application/json")}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["total_records"] == 120
    assert data["valid_records"] == 120
    assert data["splits"]["train"] == 90


def test_training_job_submission_cuda_unavailable(monkeypatch):
    """
    On CPU-only hardware, POST /api/v1/training/jobs must:
    1. Validate the 120-record manifest
    2. Create a unique job
    3. Return status CUDA_UNAVAILABLE
    4. Provide the reviewed CLI fallback command
    5. NOT fake successful training
    """
    monkeypatch.setenv("TTV_FORCE_CUDA_UNAVAILABLE", "1")
    monkeypatch.delenv("TTV_FORCE_CUDA_AVAILABLE", raising=False)

    with open(MANIFEST_120_PATH, "rb") as f:
        response = client.post(
            "/api/v1/training/jobs",
            files={"manifest_file": ("ttv_training_manifest_120.jsonl", f, "application/json")},
            data={
                "base_model": "SpatialTemporalTTVModel",
                "method": "lora",
                "epochs": "5",
                "learning_rate": "0.0001",
                "batch_size": "2",
                "version_name": "test_lora_120_v001"
            }
        )

    assert response.status_code == 200
    job_data = response.json()
    job_id = job_data["job_id"]
    assert job_id.startswith("train_job_")
    assert job_data["status"] == "CUDA_UNAVAILABLE"
    assert job_data["cuda_available"] is False
    assert "cli_fallback_command" in job_data
    assert "python training/run_training.py" in job_data["cli_fallback_command"]
    assert job_data["manifest_stats"]["total_records"] == 120

    # Verify GET /api/v1/training/jobs/{job_id}
    status_resp = client.get(f"/api/v1/training/jobs/{job_id}")
    assert status_resp.status_code == 200
    fetched_job = status_resp.json()
    assert fetched_job["job_id"] == job_id
    assert fetched_job["status"] == "CUDA_UNAVAILABLE"
    assert fetched_job["progress"] == 0

    # Verify GET /api/v1/training/jobs/{job_id}/logs
    logs_resp = client.get(f"/api/v1/training/jobs/{job_id}/logs")
    assert logs_resp.status_code == 200
    logs_data = logs_resp.json()
    assert logs_data["job_id"] == job_id
    assert logs_data["status"] == "CUDA_UNAVAILABLE"
    assert any("CUDA_UNAVAILABLE" in line for line in logs_data["logs"])
    assert any("Manual Fallback" in line for line in logs_data["logs"])


def test_training_job_submission_json_payload(monkeypatch):
    """Verify JSON submission with manifest_path pointing to 120-record manifest."""
    monkeypatch.setenv("TTV_FORCE_CUDA_UNAVAILABLE", "1")
    monkeypatch.delenv("TTV_FORCE_CUDA_AVAILABLE", raising=False)

    payload = {
        "manifest_path": str(MANIFEST_120_PATH),
        "base_model": "SpatialTemporalTTVModel",
        "method": "lora",
        "epochs": 3,
        "learning_rate": 0.0002,
        "batch_size": 2,
        "version_name": "test_json_120_v001"
    }

    response = client.post("/api/v1/training/jobs", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"].startswith("train_job_")
    assert data["status"] == "CUDA_UNAVAILABLE"
    assert data["manifest_stats"]["total_records"] == 120


def test_training_jobs_list_endpoint():
    """Verify GET /api/v1/training/jobs returns list of registered jobs."""
    response = client.get("/api/v1/training/jobs")
    assert response.status_code == 200
    jobs = response.json()
    assert isinstance(jobs, list)
    assert len(jobs) >= 1
    assert any(j["status"] in ("CUDA_UNAVAILABLE", "running", "completed") for j in jobs)


def test_job_not_found():
    response = client.get("/api/v1/training/jobs/non_existent_train_job")
    assert response.status_code == 404


def test_cuda_training_worker_launch_simulation(monkeypatch):
    """
    When CUDA is available (simulated), verify that:
    1. Job status is initially 'queued' or 'running'
    2. Subprocess worker is triggered
    3. Logs and status can be polled
    """
    monkeypatch.setenv("TTV_FORCE_CUDA_AVAILABLE", "1")

    payload = {
        "manifest_path": "training/datasets/dataset.jsonl",
        "base_model": "spatial_temporal_ttv_smoke",
        "method": "lora",
        "epochs": 1,
        "learning_rate": 0.001,
        "batch_size": 1,
        "version_name": "sim_cuda_v001"
    }
    response = client.post("/api/v1/training/jobs", json=payload)
    assert response.status_code == 200
    data = response.json()
    job_id = data["job_id"]
    assert data["status"] in ["queued", "running", "completed"]
    assert data["cuda_available"] is True

    # Check logs endpoint
    logs_resp = client.get(f"/api/v1/training/jobs/{job_id}/logs")
    assert logs_resp.status_code == 200
    logs_data = logs_resp.json()
    assert logs_data["job_id"] == job_id
