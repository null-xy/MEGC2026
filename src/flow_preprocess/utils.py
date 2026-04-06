"""Shared filesystem and serialization helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg", ".m4v"}


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_image_file(path: str | Path) -> bool:
    """Return True when the path points to a supported image file."""
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def is_video_file(path: str | Path) -> bool:
    """Return True when the path points to a supported video file."""
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def natural_sort_key(value: str | Path) -> list[Any]:
    """Build a natural sort key for numbered file names."""
    text = str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def write_json(data: Any, path: str | Path) -> Path:
    """Write JSON data to disk."""
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
    return output_path


def utc_timestamp() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()
