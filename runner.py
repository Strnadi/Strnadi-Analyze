"""Entry point kept for the existing deployment (`uvicorn runner:app`). The service lives in serving/app.py."""
from serving.app import app  # noqa: F401
