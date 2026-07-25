"""Ensure the repo root is importable as `src...` regardless of how pytest
is invoked (bare `pytest` vs `python -m pytest`, or from another cwd)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
