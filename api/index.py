"""Vercel entry point: exposes the FastAPI app that backend/api.py builds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.api import app  # noqa: E402,F401
