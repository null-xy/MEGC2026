"""Unified loading for frame folders and video files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from flow_preprocess.utils import ensure_directory, is_image_file, is_video_file, natural_sort_key


@dataclass
class LoadedFrame:
    """One loaded frame and its origin metadata."""

    index: int
    name: str
    path: Path | None
    image_bgr: np.ndarray


@dataclass
class LoadedSequence:
    """Ordered sequence returned by the unified loader."""

    source_path: Path
    source_type: str
    video_id: str
    frames: list[LoadedFrame]

    @property
    def frame_names(self) -> list[str]:
        return [frame.name for frame in self.frames]

    @property
    def num_frames(self) -> int:
        return len(self.frames)


def list_frame_files(directory: str | Path) -> list[Path]:
    """Return naturally sorted image files in a directory."""
    directory = Path(directory)
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and is_image_file(path)],
        key=natural_sort_key,
    )


def _extract_video_frames_to_cache(video_path: Path, cache_dir: Path) -> list[Path]:
    extract_dir = ensure_directory(cache_dir / f"{video_path.stem}_frames")
    existing_frames = list_frame_files(extract_dir)
    if existing_frames:
        return existing_frames

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video input: {video_path}")

    frame_paths: list[Path] = []
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_path = extract_dir / f"{frame_index:06d}.jpg"
        if not cv2.imwrite(str(frame_path), frame):
            capture.release()
            raise RuntimeError(f"Failed to write extracted frame: {frame_path}")
        frame_paths.append(frame_path)
        frame_index += 1
    capture.release()

    if not frame_paths:
        raise RuntimeError(f"No frames could be decoded from video: {video_path}")
    return frame_paths


def load_sequence(input_path: str | Path, *, cache_dir: str | Path | None = None) -> LoadedSequence:
    """Load one frame directory or video file as ordered BGR images."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_dir():
        frame_paths = list_frame_files(input_path)
        if not frame_paths:
            raise ValueError(f"No image frames found in directory: {input_path}")
        frames: list[LoadedFrame] = []
        for index, frame_path in enumerate(frame_paths):
            image = cv2.imread(str(frame_path))
            if image is None:
                raise RuntimeError(f"Failed to read frame image: {frame_path}")
            frames.append(
                LoadedFrame(index=index, name=frame_path.name, path=frame_path, image_bgr=image)
            )
        return LoadedSequence(
            source_path=input_path,
            source_type="dir",
            video_id=input_path.name,
            frames=frames,
        )

    if not is_video_file(input_path):
        raise ValueError(f"Unsupported input file type: {input_path}")

    cache_path = ensure_directory(cache_dir or Path("outputs") / "cache")
    frame_paths = _extract_video_frames_to_cache(input_path, cache_path)
    frames = []
    for index, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path))
        if image is None:
            raise RuntimeError(f"Failed to read cached extracted frame: {frame_path}")
        frames.append(LoadedFrame(index=index, name=frame_path.name, path=frame_path, image_bgr=image))
    return LoadedSequence(
        source_path=input_path,
        source_type="video",
        video_id=input_path.stem,
        frames=frames,
    )


def discover_input_units(root_path: str | Path) -> list[Path]:
    """Discover concrete test units from a root path."""
    root_path = Path(root_path)
    if not root_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {root_path}")
    if root_path.is_file():
        return [root_path]

    direct_frame_files = list_frame_files(root_path)
    if direct_frame_files:
        return [root_path]

    units: list[Path] = []
    for child in sorted(root_path.iterdir(), key=natural_sort_key):
        if child.is_file() and is_video_file(child):
            units.append(child)
            continue
        if child.is_dir():
            if list_frame_files(child):
                units.append(child)
                continue
            nested_frame_dirs = [
                nested
                for nested in sorted(child.iterdir(), key=natural_sort_key)
                if nested.is_dir() and list_frame_files(nested)
            ]
            units.extend(nested_frame_dirs)

    if not units:
        raise ValueError(f"No valid video files or frame directories found under: {root_path}")
    return units
