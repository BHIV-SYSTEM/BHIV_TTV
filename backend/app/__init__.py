"""
Unified Text-to-Video (TTV) Application Package
"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

__version__ = "1.0.0"
