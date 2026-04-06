"""Face crop and alignment helpers for deterministic preprocessing."""

from __future__ import annotations

import math
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from flow_preprocess.utils import ensure_directory


PREDICTOR_MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"


@dataclass
class CropRecord:
    """Metadata for one cropped frame."""

    frame_index: int
    frame_name: str
    crop_image_bgr: np.ndarray | None
    crop_bbox: dict[str, int] | None
    detection_source: str
    crop_path: Path | None = None


def load_face_detector():
    """Create the OpenCV frontal face detector reused from the reference project."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Failed to load face detector: {cascade_path}")
    return detector


def load_eye_detector():
    """Create the OpenCV eye detector used for lightweight alignment."""
    candidate_files = [
        "haarcascade_eye_tree_eyeglasses.xml",
        "haarcascade_eye.xml",
    ]
    for filename in candidate_files:
        cascade_path = Path(cv2.data.haarcascades) / filename
        detector = cv2.CascadeClassifier(str(cascade_path))
        if not detector.empty():
            return detector
    raise RuntimeError("Failed to load an OpenCV eye detector cascade.")


@lru_cache(maxsize=1)
def _load_dlib_alignment_runtime() -> tuple[Any, Any, Path] | None:
    try:
        import dlib
    except Exception:
        return None

    project_root = Path(__file__).resolve().parents[2]
    search_root = project_root.parent
    predictor_path = next(
        (
            path
            for path in sorted(search_root.rglob(PREDICTOR_MODEL_FILENAME))
            if path.is_file()
        ),
        None,
    )
    if predictor_path is None:
        return None

    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(str(predictor_path))
    return face_detector, face_pose_predictor, predictor_path


def _detect_face_bbox(image_bgr: np.ndarray, face_detector, padding_ratio: float = 0.2) -> dict[str, int] | None:
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(image_bgr.shape[1], x + width + pad_x)
    y1 = min(image_bgr.shape[0], y + height + pad_y)
    return {
        "x": int(x0),
        "y": int(y0),
        "width": int(x1 - x0),
        "height": int(y1 - y0),
    }


def _crop_with_bbox(image_bgr: np.ndarray, bbox: dict[str, int]) -> np.ndarray:
    x0 = max(0, int(bbox["x"]))
    y0 = max(0, int(bbox["y"]))
    x1 = min(image_bgr.shape[1], x0 + int(bbox["width"]))
    y1 = min(image_bgr.shape[0], y0 + int(bbox["height"]))
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        raise RuntimeError(f"Crop box produced an empty image: {bbox}")
    return crop


def _find_previous_valid_box(boxes: list[dict[str, int] | None], index: int) -> dict[str, int] | None:
    for candidate_index in range(index - 1, -1, -1):
        if boxes[candidate_index] is not None:
            return boxes[candidate_index]
    return None


def _find_next_valid_box(boxes: list[dict[str, int] | None], index: int) -> dict[str, int] | None:
    for candidate_index in range(index + 1, len(boxes)):
        if boxes[candidate_index] is not None:
            return boxes[candidate_index]
    return None


def _median_box(valid_boxes: list[dict[str, int]]) -> dict[str, int]:
    return {
        key: int(np.median([box[key] for box in valid_boxes]))
        for key in ("x", "y", "width", "height")
    }


def _build_tight_asymmetric_crop_box(
    stable_box: dict[str, int],
    *,
    horizontal_trim_ratio: float = 0.06,
    top_trim_ratio: float = 0.05,
    bottom_trim_ratio: float = 0.09,
) -> dict[str, int]:
    """Shrink the stable sequence box into a tighter, asymmetric face-centered crop."""
    width = max(1, int(stable_box["width"]))
    height = max(1, int(stable_box["height"]))

    trim_left = int(round(width * horizontal_trim_ratio))
    trim_right = int(round(width * horizontal_trim_ratio))
    trim_top = int(round(height * top_trim_ratio))
    trim_bottom = int(round(height * bottom_trim_ratio))

    tight_width = width - trim_left - trim_right
    tight_height = height - trim_top - trim_bottom
    if tight_width <= 1 or tight_height <= 1:
        return {
            "x": int(stable_box["x"]),
            "y": int(stable_box["y"]),
            "width": width,
            "height": height,
        }

    return {
        "x": int(stable_box["x"] + trim_left),
        "y": int(stable_box["y"] + trim_top),
        "width": int(tight_width),
        "height": int(tight_height),
    }


def crop_faces_from_sequence(
    frames: list[np.ndarray],
    frame_names: list[str],
    *,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    padding_ratio: float = 0.2,
) -> dict[str, Any]:
    """Detect, stabilize, crop, and optionally save face crops for one sequence."""
    if len(frames) != len(frame_names):
        raise ValueError("frames and frame_names must have the same length.")
    if not frames:
        raise ValueError("Cannot crop an empty sequence.")

    face_detector = load_face_detector()
    detected_boxes = [_detect_face_bbox(frame, face_detector, padding_ratio=padding_ratio) for frame in frames]
    valid_boxes = [box for box in detected_boxes if box is not None]
    if not valid_boxes:
        raise RuntimeError("Face detection failed on all frames in the sequence.")

    filled_boxes: list[dict[str, int] | None] = []
    fill_sources: list[str] = []
    for index, box in enumerate(detected_boxes):
        if box is not None:
            filled_boxes.append(box)
            fill_sources.append("detected")
            continue
        previous_box = _find_previous_valid_box(detected_boxes, index)
        if previous_box is not None:
            filled_boxes.append(previous_box)
            fill_sources.append("previous_valid_box")
            continue
        next_box = _find_next_valid_box(detected_boxes, index)
        if next_box is not None:
            filled_boxes.append(next_box)
            fill_sources.append("next_valid_box")
            continue
        filled_boxes.append(None)
        fill_sources.append("missing")

    stable_box = _median_box([box for box in filled_boxes if box is not None])
    final_crop_box = _build_tight_asymmetric_crop_box(stable_box)
    records: list[CropRecord] = []
    crop_success_count = 0
    crop_fail_count = 0
    output_path = ensure_directory(output_dir) if output_dir is not None else None

    for index, (frame, frame_name, fallback_source) in enumerate(zip(frames, frame_names, fill_sources)):
        selected_box = filled_boxes[index]
        if selected_box is None:
            records.append(
                CropRecord(
                    frame_index=index,
                    frame_name=frame_name,
                    crop_image_bgr=None,
                    crop_bbox=None,
                    detection_source="missing",
                )
            )
            crop_fail_count += 1
            continue

        # Use one stable median box across the sequence to reduce jitter after fallback filling.
        crop_bbox = final_crop_box
        try:
            crop_image = _crop_with_bbox(frame, crop_bbox)
        except Exception:
            records.append(
                CropRecord(
                    frame_index=index,
                    frame_name=frame_name,
                    crop_image_bgr=None,
                    crop_bbox=None,
                    detection_source="crop_error",
                )
            )
            crop_fail_count += 1
            continue

        crop_path = None
        if output_path is not None:
            crop_path = output_path / f"{Path(frame_name).stem}_crop.png"
            if overwrite or not crop_path.exists():
                if not cv2.imwrite(str(crop_path), crop_image):
                    raise RuntimeError(f"Failed to write cropped face image: {crop_path}")
        records.append(
            CropRecord(
                frame_index=index,
                frame_name=frame_name,
                crop_image_bgr=crop_image,
                crop_bbox=crop_bbox,
                detection_source=f"stable_median_from_{fallback_source}",
                crop_path=crop_path,
            )
        )
        crop_success_count += 1

    return {
        "records": records,
        "stable_box": stable_box,
        "final_crop_box": final_crop_box,
        "crop_box_source": "global_median_bbox_from_detected_prev_next_filled_boxes_then_tight_asymmetric_trim(h=0.06,top=0.05,bottom=0.09)",
        "crop_success_count": crop_success_count,
        "crop_fail_count": crop_fail_count,
    }


def _detect_eye_centers(gray_image: np.ndarray, eye_detector) -> list[tuple[float, float]]:
    top_half = gray_image[: max(1, gray_image.shape[0] // 2), :]
    detections = eye_detector.detectMultiScale(
        top_half,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(12, 12),
    )
    candidates: list[tuple[int, int, int, int]] = []
    for x, y, width, height in detections:
        candidates.append((int(x), int(y), int(width), int(height)))
    candidates.sort(key=lambda item: item[2] * item[3], reverse=True)

    centers: list[tuple[float, float]] = []
    for x, y, width, height in candidates:
        center = (x + width / 2.0, y + height / 2.0)
        if any(abs(center[0] - existing[0]) < width * 0.5 for existing in centers):
            continue
        centers.append(center)
        if len(centers) == 2:
            break
    return sorted(centers, key=lambda item: item[0])


def _detect_dlib_eye_centers(gray_image: np.ndarray) -> tuple[list[tuple[float, float]] | None, dict[str, Any]]:
    runtime = _load_dlib_alignment_runtime()
    if runtime is None:
        return None, {
            "available": False,
            "reason": "dlib_or_predictor_unavailable",
        }

    face_detector, face_pose_predictor, predictor_path = runtime
    detections = face_detector(gray_image, 1)
    if len(detections) == 0:
        return None, {
            "available": True,
            "predictor_model_path": str(predictor_path),
            "reason": "no_face_detected",
        }

    shape = face_pose_predictor(gray_image, detections[0])
    left_eye = [
        (float(shape.part(index).x), float(shape.part(index).y))
        for index in range(36, 42)
    ]
    right_eye = [
        (float(shape.part(index).x), float(shape.part(index).y))
        for index in range(42, 48)
    ]
    left_center = (
        float(sum(point[0] for point in left_eye) / len(left_eye)),
        float(sum(point[1] for point in left_eye) / len(left_eye)),
    )
    right_center = (
        float(sum(point[0] for point in right_eye) / len(right_eye)),
        float(sum(point[1] for point in right_eye) / len(right_eye)),
    )
    return [left_center, right_center], {
        "available": True,
        "predictor_model_path": str(predictor_path),
        "reason": "landmarks_detected",
    }


def _align_from_eye_centers(
    image_bgr: np.ndarray,
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
    *,
    output_size: int,
    method: str,
    extra_status: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    if delta_x == 0:
        resized = cv2.resize(image_bgr, (int(output_size), int(output_size)))
        status = {"status": "fallback_resize", "method": method}
        if extra_status:
            status.update(extra_status)
        return resized, status

    current_distance = math.hypot(delta_x, delta_y)
    desired_left_eye = (0.35, 0.38)
    desired_right_eye_x = 1.0 - desired_left_eye[0]
    desired_distance = (desired_right_eye_x - desired_left_eye[0]) * float(output_size)
    scale = desired_distance / current_distance
    eyes_center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    angle = math.degrees(math.atan2(delta_y, delta_x))

    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    tx = output_size * 0.5 - eyes_center[0]
    ty = output_size * desired_left_eye[1] - eyes_center[1]
    rotation_matrix[0, 2] += tx
    rotation_matrix[1, 2] += ty

    aligned = cv2.warpAffine(
        image_bgr,
        rotation_matrix,
        (int(output_size), int(output_size)),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    status = {
        "status": "aligned",
        "method": method,
        "angle_degrees": float(angle),
        "scale": float(scale),
    }
    if extra_status:
        status.update(extra_status)
    return aligned, status


def align_face_crop(
    crop_image_bgr: np.ndarray,
    *,
    output_size: int = 128,
    eye_detector=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align a cropped face image using dlib when available, then OpenCV, then resize fallback."""
    if crop_image_bgr is None or crop_image_bgr.size == 0:
        raise ValueError("crop_image_bgr must be a non-empty image.")

    gray_image = cv2.cvtColor(crop_image_bgr, cv2.COLOR_BGR2GRAY)
    dlib_eye_centers, dlib_status = _detect_dlib_eye_centers(gray_image)
    if dlib_eye_centers is not None and len(dlib_eye_centers) == 2:
        aligned, status = _align_from_eye_centers(
            crop_image_bgr,
            dlib_eye_centers[0],
            dlib_eye_centers[1],
            output_size=output_size,
            method="dlib_landmark_alignment",
            extra_status={"eye_count": 2, **dlib_status},
        )
        return aligned, status

    eye_detector = eye_detector or load_eye_detector()
    eye_centers = _detect_eye_centers(gray_image, eye_detector)
    if len(eye_centers) < 2:
        resized = cv2.resize(crop_image_bgr, (int(output_size), int(output_size)))
        return resized, {
            "status": "fallback_resize",
            "method": "opencv_eye_alignment",
            "eye_count": len(eye_centers),
            "dlib_status": dlib_status,
        }

    aligned, status = _align_from_eye_centers(
        crop_image_bgr,
        eye_centers[0],
        eye_centers[1],
        output_size=output_size,
        method="opencv_eye_alignment",
        extra_status={"eye_count": len(eye_centers), "dlib_status": dlib_status},
    )
    return aligned, status
