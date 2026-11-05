import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app, _materialize_bundled_manifest_120
from app.services.reference_service import reference_service
from app.services.training_service import parse_manifest_records, training_job_manager


def _create_dummy_image_bytes() -> bytes:
    img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_import_training_router_does_not_create_manifest(tmp_path):
    """
    Test 1: Verify that importing or accessing training.py router does NOT create
    data/manifests/ttv_training_manifest_120.jsonl at import time.
    """
    jsonl_path = settings.get_absolute_path("data/manifests/ttv_training_manifest_120.jsonl")

    # Temporarily remove or stash jsonl if present to test import side-effect
    backup = None
    if jsonl_path.exists():
        backup = jsonl_path.read_text(encoding="utf-8")
        jsonl_path.unlink()

    try:
        assert not jsonl_path.exists()
        # Re-import training module
        import importlib
        import app.api.routes.training as training_mod
        importlib.reload(training_mod)

        # File should STILL NOT exist
        assert not jsonl_path.exists(), "training.py router must not create files at import time"
    finally:
        # Restore backup if needed
        if backup is not None and not jsonl_path.exists():
            jsonl_path.write_text(backup, encoding="utf-8")


def test_lifespan_startup_materializes_manifest(tmp_path):
    """
    Test 2: Verify that lifespan startup safely materializes
    ttv_training_manifest_120.jsonl if absent.
    """
    jsonl_path = settings.get_absolute_path("data/manifests/ttv_training_manifest_120.jsonl")
    json_path = settings.get_absolute_path("data/manifests/ttv_training_manifest_120.json")

    backup = None
    if jsonl_path.exists():
        backup = jsonl_path.read_text(encoding="utf-8")
        jsonl_path.unlink()

    try:
        assert not jsonl_path.exists()
        assert json_path.exists(), "Source JSON manifest must exist"

        # Call materialization function as called during lifespan
        _materialize_bundled_manifest_120()

        assert jsonl_path.exists(), "Manifest JSONL must be materialized by startup function"
        records, _ = parse_manifest_records(jsonl_path)
        assert len(records) == 120
    finally:
        if backup is not None:
            jsonl_path.write_text(backup, encoding="utf-8")


def test_lifespan_startup_does_not_overwrite_existing_manifest(tmp_path):
    """
    Test 3: Verify that lifespan startup does NOT overwrite an existing JSONL manifest.
    """
    jsonl_path = settings.get_absolute_path("data/manifests/ttv_training_manifest_120.jsonl")
    backup = None
    if jsonl_path.exists():
        backup = jsonl_path.read_text(encoding="utf-8")

    try:
        # Write a sentinel custom content
        sentinel_content = json.dumps({"id": "sentinel_custom_record", "prompt": "custom"}) + "\n"
        jsonl_path.write_text(sentinel_content, encoding="utf-8")

        # Call materialization function
        _materialize_bundled_manifest_120()

        # Should NOT be overwritten
        current_content = jsonl_path.read_text(encoding="utf-8")
        assert current_content == sentinel_content, "Existing JSONL manifest must not be overwritten"
    finally:
        if backup is not None:
            jsonl_path.write_text(backup, encoding="utf-8")


def test_atomic_file_creation_and_error_cleanup(tmp_path, monkeypatch):
    """
    Test 4: Verify atomic file creation writes to temp file first and cleans up on failure.
    """
    fake_json = tmp_path / "manifest.json"
    fake_json.write_text(json.dumps([{"id": "test_1", "prompt": "test"}]), encoding="utf-8")
    fake_jsonl = tmp_path / "manifest.jsonl"

    created_temp_files = []

    def mock_replace(src, dst):
        created_temp_files.append(src)
        raise OSError("Simulated atomic rename failure")

    monkeypatch.setattr(os, "replace", mock_replace)

    # Calling materialization should catch the error and remove the temp file
    _materialize_bundled_manifest_120(jsonl_path=fake_jsonl, json_path=fake_json)

    assert not fake_jsonl.exists()
    assert len(created_temp_files) == 1
    for tmp in created_temp_files:
        assert not Path(tmp).exists(), "Temporary file must be unlinked on failure"


def test_custom_manifest_plus_reference_combined_in_jobs(tmp_path):
    """
    Test 5: Verify POST /api/v1/training/jobs with BOTH custom manifest and reference media:
    - Combines custom records + reference entry
    - Preserves original custom manifest without mutation
    - Avoids duplicate entries
    """
    # 1. Create a reference via upload
    client = TestClient(app)
    img_bytes = _create_dummy_image_bytes()
    upload_res = client.post(
        "/api/v1/references/upload",
        files={"file": ("comb_ref.jpg", img_bytes, "image/jpeg")},
        data={"reference_type": "image"}
    )
    assert upload_res.status_code == 200
    ref_id = upload_res.json()["reference_id"]

    # 2. Create a custom manifest file
    custom_records = [
        {
            "id": "custom_sample_001",
            "prompt": "A futuristic city skyline at dusk",
            "video_path": str(tmp_path / "custom_vid1.mp4"),
            "duration": 5.0,
            "quality_score": 0.9,
            "split": "train"
        },
        {
            "id": "custom_sample_002",
            "prompt": "An astronaut walking on the surface of Mars",
            "video_path": str(tmp_path / "custom_vid2.mp4"),
            "duration": 6.0,
            "quality_score": 0.92,
            "split": "val"
        }
    ]
    custom_manifest_file = tmp_path / "custom_manifest.jsonl"
    with open(custom_manifest_file, "w", encoding="utf-8") as f:
        for r in custom_records:
            f.write(json.dumps(r) + "\n")

    initial_custom_text = custom_manifest_file.read_text(encoding="utf-8")

    # 3. Submit job with both custom manifest file and reference_id
    with open(custom_manifest_file, "rb") as mf:
        response = client.post(
            "/api/v1/training/jobs",
            files={"manifest_file": ("custom_manifest.jsonl", mf, "application/json")},
            data={
                "base_model": "SpatialTemporalTTVModel",
                "method": "lora",
                "epochs": "2",
                "learning_rate": "0.0001",
                "batch_size": "2",
                "reference_id": ref_id,
                "version_name": "test_comb_v1"
            }
        )

    assert response.status_code == 200, response.text
    job_data = response.json()
    job_id = job_data["job_id"]

    # Original file on disk must NOT be mutated
    after_custom_text = custom_manifest_file.read_text(encoding="utf-8")
    assert initial_custom_text == after_custom_text, "Original custom manifest file must not be modified!"

    # Job's manifest must contain 3 records (2 custom + 1 reference)
    job_dir = settings.get_absolute_path("generated/training_jobs") / job_id
    job_manifest = job_dir / "manifest.jsonl"
    assert job_manifest.exists()
    job_records, _ = parse_manifest_records(job_manifest)
    assert len(job_records) == 3, f"Expected 3 combined records, got {len(job_records)}"

    job_record_ids = {r["id"] for r in job_records}
    assert "custom_sample_001" in job_record_ids
    assert "custom_sample_002" in job_record_ids
    assert f"sample_{ref_id}" in job_record_ids

    # Reference should be at index 0 as priority
    assert job_records[0]["id"] == f"sample_{ref_id}"

    # Verify duplicate prevention: submit again with custom manifest that already has reference
    combined_with_ref = [job_records[0]] + custom_records
    dup_manifest_file = tmp_path / "dup_manifest.jsonl"
    with open(dup_manifest_file, "w", encoding="utf-8") as f:
        for r in combined_with_ref:
            f.write(json.dumps(r) + "\n")

    with open(dup_manifest_file, "rb") as dmf:
        dup_resp = client.post(
            "/api/v1/training/jobs",
            files={"manifest_file": ("dup_manifest.jsonl", dmf, "application/json")},
            data={
                "base_model": "SpatialTemporalTTVModel",
                "method": "lora",
                "epochs": "1",
                "learning_rate": "0.0001",
                "batch_size": "1",
                "reference_id": ref_id
            }
        )
    assert dup_resp.status_code == 200
    dup_job_id = dup_resp.json()["job_id"]
    dup_job_dir = settings.get_absolute_path("generated/training_jobs") / dup_job_id
    dup_records, _ = parse_manifest_records(dup_job_dir / "manifest.jsonl")
    # Must NOT have 4 records; still exactly 3 records!
    assert len(dup_records) == 3, f"Duplicate reference should not be added, got {len(dup_records)}"


def test_custom_manifest_without_reference(tmp_path):
    """
    Test 6: Verify POST /api/v1/training/jobs with ONLY custom manifest (no reference).
    """
    client = TestClient(app)
    custom_records = [
        {
            "id": "single_custom_001",
            "prompt": "A serene lake surrounded by pine trees at sunrise",
            "video_path": str(tmp_path / "lake.mp4"),
            "duration": 4.0,
            "quality_score": 0.88,
            "split": "train"
        }
    ]
    manifest_file = tmp_path / "lake_manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        for r in custom_records:
            f.write(json.dumps(r) + "\n")

    with open(manifest_file, "rb") as mf:
        response = client.post(
            "/api/v1/training/jobs",
            files={"manifest_file": ("lake_manifest.jsonl", mf, "application/json")},
            data={
                "base_model": "SpatialTemporalTTVModel",
                "method": "lora",
                "epochs": "1",
                "learning_rate": "0.0001",
                "batch_size": "1"
            }
        )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job_manifest = settings.get_absolute_path("generated/training_jobs") / job_id / "manifest.jsonl"
    records, _ = parse_manifest_records(job_manifest)
    assert len(records) == 1
    assert records[0]["id"] == "single_custom_001"


def test_reference_without_custom_manifest():
    """
    Test 7: Verify POST /api/v1/training/jobs with ONLY reference media (no custom manifest).
    Uses the standalone reference manifest path.
    """
    client = TestClient(app)
    img_bytes = _create_dummy_image_bytes()
    upload_res = client.post(
        "/api/v1/references/upload",
        files={"file": ("standalone_ref.jpg", img_bytes, "image/jpeg")},
        data={"reference_type": "image"}
    )
    assert upload_res.status_code == 200
    ref_id = upload_res.json()["reference_id"]

    response = client.post(
        "/api/v1/training/jobs",
        data={
            "base_model": "SpatialTemporalTTVModel",
            "method": "lora",
            "epochs": "1",
            "learning_rate": "0.0001",
            "batch_size": "1",
            "reference_id": ref_id
        }
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    job_manifest = settings.get_absolute_path("generated/training_jobs") / job_id / "manifest.jsonl"
    records, _ = parse_manifest_records(job_manifest)
    assert len(records) >= 1
    assert records[0]["id"] == f"sample_{ref_id}"


def test_concurrent_materialization_does_not_overwrite_winning_destination(tmp_path):
    """
    Test 8: Verify that when multiple concurrent workers execute _materialize_bundled_manifest_120
    simultaneously, exactly one worker materializes the file and subsequent/concurrent
    workers do NOT overwrite the winning destination.
    """
    import concurrent.futures

    target_jsonl = tmp_path / "concurrent_manifest.jsonl"
    source_json = tmp_path / "concurrent_manifest.json"

    # Write source JSON
    source_json.write_text(
        json.dumps([{"id": f"sample_{i}", "prompt": f"prompt_{i}"} for i in range(5)]),
        encoding="utf-8"
    )

    assert not target_jsonl.exists()

    def run_worker(worker_id):
        _materialize_bundled_manifest_120(jsonl_path=target_jsonl, json_path=source_json)
        return worker_id

    # Run 8 concurrent worker threads racing to materialize
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run_worker, range(8)))

    assert len(results) == 8
    assert target_jsonl.exists()
    records, _ = parse_manifest_records(target_jsonl)
    assert len(records) == 5

    # Verify no dangling lock or temporary files remain in tmp_path
    remaining_files = set(tmp_path.iterdir())
    assert remaining_files == {source_json, target_jsonl}


def test_concurrent_materialization_preserves_winning_content(tmp_path):
    """
    Test 9: Verify that if an initial worker publishes a winning destination,
    subsequent/concurrent materialization calls will NEVER overwrite the winning content.
    """
    target_jsonl = tmp_path / "winning_manifest.jsonl"
    source_json = tmp_path / "winning_manifest.json"
    source_json.write_text(json.dumps([{"id": "loser_record", "prompt": "loser"}]), encoding="utf-8")

    # Seed the winning destination
    winning_text = json.dumps({"id": "winner_record", "prompt": "winning_prompt"}) + "\n"
    target_jsonl.write_text(winning_text, encoding="utf-8")
    initial_stat = target_jsonl.stat()

    # Attempt materialization from source_json
    _materialize_bundled_manifest_120(jsonl_path=target_jsonl, json_path=source_json)

    # Destination MUST remain the winning content, never overwritten by source_json
    assert target_jsonl.read_text(encoding="utf-8") == winning_text
    assert target_jsonl.stat().st_mtime_ns == initial_stat.st_mtime_ns


def test_cross_process_concurrent_materialization(tmp_path):
    """
    Test 10: Multiple independent OS processes competing to materialize the same destination.
    - Exactly one valid destination published.
    - Subprocesses use sys.executable (current Python interpreter).
    - No leftover temporary (.tmp) files after completion.
    """
    import subprocess
    import sys

    target_jsonl = tmp_path / "proc_manifest.jsonl"
    source_json = tmp_path / "proc_manifest.json"

    # Write source JSON
    source_json.write_text(
        json.dumps([{"id": f"proc_sample_{i}", "prompt": f"prompt_{i}"} for i in range(10)]),
        encoding="utf-8"
    )

    assert not target_jsonl.exists()

    # Formulate safe python code using forward slashes for cross-platform subprocess compatibility
    target_str = str(target_jsonl).replace("\\", "/")
    source_str = str(source_json).replace("\\", "/")
    backend_dir_str = str(Path(__file__).resolve().parents[1]).replace("\\", "/")
    child_code = (
        f"import sys; sys.path.insert(0, '{backend_dir_str}'); "
        "from pathlib import Path; "
        "from app.main import _materialize_bundled_manifest_120; "
        f"_materialize_bundled_manifest_120(jsonl_path=Path('{target_str}'), json_path=Path('{source_str}'))"
    )

    # Spawn 4 separate OS processes concurrently
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", child_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        for _ in range(4)
    ]

    for p in procs:
        stdout, stderr = p.communicate(timeout=15)
        assert p.returncode == 0, f"Subprocess failed with code {p.returncode}: {stderr.decode()}"

    # Destination was created and contains all 10 records
    assert target_jsonl.exists()
    records, _ = parse_manifest_records(target_jsonl)
    assert len(records) == 10

    # Verify no dangling temporary files (.tmp) remain
    tmp_files = list(tmp_path.glob(".*.tmp"))
    assert len(tmp_files) == 0, f"Found leftover temporary files: {tmp_files}"


def test_cross_process_preserves_existing_destination(tmp_path):
    """
    Test 11: Existing destination content is never overwritten even when multiple
    separate OS processes execute materialization.
    """
    import subprocess
    import sys

    target_jsonl = tmp_path / "existing_manifest.jsonl"
    source_json = tmp_path / "existing_manifest.json"

    sentinel_text = json.dumps({"id": "sentinel_record", "prompt": "do_not_overwrite"}) + "\n"
    target_jsonl.write_text(sentinel_text, encoding="utf-8")
    initial_stat = target_jsonl.stat()

    source_json.write_text(
        json.dumps([{"id": "loser_record", "prompt": "overwritten"}]),
        encoding="utf-8"
    )

    target_str = str(target_jsonl).replace("\\", "/")
    source_str = str(source_json).replace("\\", "/")
    backend_dir_str = str(Path(__file__).resolve().parents[1]).replace("\\", "/")
    child_code = (
        f"import sys; sys.path.insert(0, '{backend_dir_str}'); "
        "from pathlib import Path; "
        "from app.main import _materialize_bundled_manifest_120; "
        f"_materialize_bundled_manifest_120(jsonl_path=Path('{target_str}'), json_path=Path('{source_str}'))"
    )

    procs = [
        subprocess.Popen(
            [sys.executable, "-c", child_code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        for _ in range(3)
    ]

    for p in procs:
        p.communicate(timeout=15)
        assert p.returncode == 0

    assert target_jsonl.read_text(encoding="utf-8") == sentinel_text
    assert target_jsonl.stat().st_mtime_ns == initial_stat.st_mtime_ns


def test_lock_timeout_logs_warning_when_destination_absent(tmp_path, caplog):
    """
    Test 12: Lock acquisition timeout produces an explicit warning when destination remains absent.
    """
    import logging
    from filelock import FileLock

    target_jsonl = tmp_path / "timeout_manifest.jsonl"
    source_json = tmp_path / "timeout_manifest.json"
    source_json.write_text(json.dumps([{"id": "t1", "prompt": "p1"}]), encoding="utf-8")

    lock_path = target_jsonl.with_name(f".{target_jsonl.name}.lock")

    # Pre-acquire the lock externally so materialization will time out
    external_lock = FileLock(str(lock_path))
    external_lock.acquire()

    try:
        with caplog.at_level(logging.WARNING):
            _materialize_bundled_manifest_120(
                jsonl_path=target_jsonl,
                json_path=source_json,
                lock_timeout=0.1
            )

        # Destination must remain absent
        assert not target_jsonl.exists()
        # Must produce an explicit warning about timeout while destination is absent
        assert any(
            "Timed out waiting for bundled manifest materialization lock" in record.message
            for record in caplog.records
        )
    finally:
        external_lock.release()
