#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add backend directory to sys.path and PYTHONPATH for reloader subprocesses
BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BASE_DIR))

current_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{BACKEND_DIR}{os.pathsep}{BASE_DIR}{os.pathsep}{current_pythonpath}" if current_pythonpath else f"{BACKEND_DIR}{os.pathsep}{BASE_DIR}"

if __name__ == "__main__":
    import uvicorn
    from app.core.config import settings

    print(f"Starting Unified Text-to-Video Engine on http://localhost:{settings.PORT} ...")
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
