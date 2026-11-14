#!/usr/bin/env python3
"""
CogVideoX-2B Model Weight Download and Cache Utility
====================================================
Controlled, resumable downloader for THUDM/CogVideoX-2b.

Usage:
    python scripts/download_cogvideox.py --check-only
    python scripts/download_cogvideox.py --confirm
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure backend modules can be imported
sys.path.insert(0, os.path.abspath("backend"))

MODEL_ID = "THUDM/CogVideoX-2b"
APPROX_SIZE_GB = 9.2


def check_cache_status() -> dict:
    """Checks if CogVideoX-2b is already populated in the Hugging Face cache."""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id.lower() == MODEL_ID.lower():
                return {
                    "cached": True,
                    "repo_id": repo.repo_id,
                    "size_on_disk": repo.size_on_disk_str,
                    "nb_files": repo.nb_files,
                }
    except Exception:
        pass

    # Direct filesystem fallback check
    hub_cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    repo_folder_name = "models--" + MODEL_ID.replace("/", "--")
    model_dir = hub_cache / repo_folder_name
    if model_dir.exists() and (model_dir / "snapshots").exists():
        snapshots = list((model_dir / "snapshots").iterdir())
        if snapshots:
            size_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            return {
                "cached": True,
                "repo_id": MODEL_ID,
                "size_on_disk": f"{round(size_bytes / (1024**3), 2)} GB",
                "nb_files": len(list(model_dir.rglob("*"))),
            }

    return {"cached": False, "repo_id": MODEL_ID, "size_on_disk": "0 GB", "nb_files": 0}


def get_remote_repo_info() -> dict:
    """Inspects HuggingFace repository to determine total download size."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(MODEL_ID, files_metadata=True)
        total_bytes = sum(f.size for f in info.siblings if getattr(f, "size", None))
        total_gb = round(total_bytes / (1024**3), 2)
        return {
            "total_bytes": total_bytes,
            "total_gb": total_gb,
            "file_count": len(info.siblings),
        }
    except Exception as exc:
        return {
            "total_bytes": int(APPROX_SIZE_GB * (1024**3)),
            "total_gb": APPROX_SIZE_GB,
            "file_count": "N/A",
            "warning": str(exc),
        }


def download_model(token: str = None) -> bool:
    """Downloads model weights into the standard Hugging Face cache."""
    try:
        from huggingface_hub import snapshot_download
        print(f"\n[Download] Commencing download for '{MODEL_ID}'...")
        print("[Download] Target: Standard Hugging Face cache (~/.cache/huggingface/hub)")
        print("[Download] Downloads are resumable if interrupted.\n")

        download_path = snapshot_download(
            repo_id=MODEL_ID,
            repo_type="model",
            token=token,
            resume_download=True,
        )
        print(f"\n[Success] CogVideoX-2b weights downloaded successfully to:\n  {download_path}\n")
        return True
    except Exception as exc:
        print(f"\n[Error] Download failed or interrupted: {exc}\n", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="CogVideoX-2B Weight Download Manager")
    parser.add_argument("--check-only", action="store_true", help="Inspect cache status without downloading")
    parser.add_argument("--confirm", action="store_true", help="Confirm initiation of ~9.2 GB model download")
    parser.add_argument("--token", type=str, default=None, help="Optional Hugging Face access token")
    args = parser.parse_args()

    print("=" * 70)
    print(" CogVideoX-2B Model Weight Download Manager")
    print("=" * 70)

    status = check_cache_status()
    if status["cached"]:
        print(f"\n[Status] Model '{MODEL_ID}' is ALREADY POPULATED in local cache:")
        print(f"  Size on disk: {status['size_on_disk']}")
        print(f"  Files count:  {status['nb_files']}")
        print("No additional download required.\n")
        return 0

    print(f"\n[Status] Model '{MODEL_ID}' is NOT present in local cache.")
    repo_info = get_remote_repo_info()
    print(f"  Estimated download size: ~{repo_info['total_gb']} GB")
    print("  Destination: Standard Hugging Face cache (~/.cache/huggingface/hub)")

    if args.check_only:
        print("\n[Notice] --check-only flag supplied. Halting without downloading.\n")
        return 0

    if not args.confirm:
        print("\n" + "!" * 70)
        print(f" [CONFIRMATION REQUIRED] Downloading '{MODEL_ID}' requires ~{repo_info['total_gb']} GB of disk space.")
        print(" To download, re-run with explicit confirmation:")
        print("   python scripts/download_cogvideox.py --confirm")
        print("!" * 70 + "\n")
        return 1

    success = download_model(token=args.token)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

