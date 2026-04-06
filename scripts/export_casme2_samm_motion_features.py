#!/usr/bin/env python3
"""Export clip-level motion and AU features for CASME2 and SAMM.

This script scans trainset optical-flow exports for CASME2 and SAMM, then
computes:
  - head motion amplitude/angle
  - per-AU amplitude/angle from the raw flow
  - per-AU amplitude/angle after removing estimated head motion with multiple
    geometric models (translation, similarity, affine)
  - eye height ratio (apex / onset)
  - dataset-specific eye and mouth landmark geometry metrics
  - GT coarse emotion, fine emotion, and AU labels from the merged JSONL

Outputs:
  - JSONL with one nested record per clip
  - CSV with one flat row per clip
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as bo
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


ROW_LABELS = ["forehead", "brow", "eye/nose-bridge", "nose", "mouth", "chin"]

LM_IDX = {
    "left_brow_inner": 107,
    "right_brow_inner": 336,
    "left_brow_outer": 46,
    "right_brow_outer": 276,
    "left_brow_peak": 55,
    "right_brow_peak": 285,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_lid_top": 159,
    "right_lid_top": 386,
    "left_lid_bot": 145,
    "right_lid_bot": 374,
    "nose_top": 6,
    "nose_tip": 1,
    "left_nostril": 49,
    "right_nostril": 279,
    "left_lip_corner": 61,
    "right_lip_corner": 291,
    "upper_lip_top": 13,
    "lower_lip_bot": 14,
    "chin_bottom": 152,
}

AU_REGION_NAMES = [
    "AU1_inner_brow_subR",
    "AU1_inner_brow_subL",
    "AU4_brow_lowerer",
    "AU2_brow_outer_subR",
    "AU2_brow_outer_subL",
    "AU6_cheek_subR",
    "AU6_cheek_subL",
    "AU9_nose_bridge",
    "AU10_upper_lip",
    "AU12_corner_subR",
    "AU12_corner_subL",
    "AU23_lip_center",
    "AU17_chin",
    "AU20_corner_subR",
    "AU20_corner_subL",
    "AU26_jaw",
]

_LM_DETECTOR = None
RIGID_LANDMARKS = [
    "left_eye_outer",
    "right_eye_outer",
    "nose_tip",
    "nose_top",
    "left_nostril",
    "right_nostril",
]
LANDMARK_RESIDUAL_METHOD = "similarity"
GEOMETRY_DATASETS = {"samm", "casme2"}


def angle_to_direction(deg: float) -> str:
    h = deg % 360
    if h < 22.5 or h >= 337.5:
        return "rightward"
    if h < 67.5:
        return "upward-right"
    if h < 112.5:
        return "upward"
    if h < 157.5:
        return "upward-left"
    if h < 202.5:
        return "leftward"
    if h < 247.5:
        return "downward-left"
    if h < 292.5:
        return "downward"
    return "downward-right"


def _rect_mask(height: int, width: int, cx: float, cy: float, hw: float, hh: float) -> np.ndarray:
    r0 = max(0, int(cy - hh))
    r1 = min(height, int(cy + hh))
    c0 = max(0, int(cx - hw))
    c1 = min(width, int(cx + hw))
    mask = np.zeros((height, width), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def _flow_stats(
    u: np.ndarray,
    v: np.ndarray,
    mag: np.ndarray,
    mask_2d: np.ndarray,
    mag_threshold: float = 0.05,
) -> dict[str, Any] | None:
    masked_mag = mag[mask_2d]
    masked_u = u[mask_2d]
    masked_v = v[mask_2d]
    active = masked_mag > mag_threshold
    if active.sum() < 4:
        return None
    weighted_u = (masked_u * masked_mag)[active]
    weighted_v = (masked_v * masked_mag)[active]
    angle = float(np.degrees(np.arctan2(-weighted_v.mean(), weighted_u.mean())) % 360)
    return {
        "angle": round(angle, 1),
        "dir": angle_to_direction(angle),
        "mag": round(float(masked_mag[active].mean()), 3),
        "active": round(float(active.sum() / mask_2d.sum()), 3),
    }


def _get_detector(model_path: Path):
    global _LM_DETECTOR
    if _LM_DETECTOR is None:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=bo.BaseOptions(model_asset_path=str(model_path)),
            num_faces=1,
            min_face_detection_confidence=0.2,
        )
        _LM_DETECTOR = mp_vision.FaceLandmarker.create_from_options(opts)
    return _LM_DETECTOR


def _resolve_existing_image(path: Path) -> Path:
    if path.exists():
        return path
    alt = path.with_suffix(".png" if path.suffix.lower() == ".jpg" else ".jpg")
    return alt if alt.exists() else path


def detect_face_landmarks(image_path: Path, model_path: Path):
    resolved = _resolve_existing_image(image_path)
    image = np.array(Image.open(resolved).convert("RGB"))
    height, width = image.shape[:2]
    detector = _get_detector(model_path)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
    if not result.face_landmarks:
        return None, height, width
    return result.face_landmarks[0], height, width


def landmark_xy(landmarks, width: int, height: int, idx: int) -> np.ndarray:
    return np.array([landmarks[idx].x * width, landmarks[idx].y * height], dtype=np.float32)


def landmark_xyz(landmarks, width: int, height: int, idx: int) -> np.ndarray:
    point = landmarks[idx]
    return np.array([point.x * width, point.y * height, point.z * width], dtype=np.float32)


def landmarks_to_points(landmarks, width: int, height: int, names: list[str]) -> np.ndarray:
    return np.array([landmark_xy(landmarks, width, height, LM_IDX[name]) for name in names], dtype=np.float32)


def landmarks_to_points_xyz(landmarks, width: int, height: int, names: list[str]) -> np.ndarray:
    return np.array([landmark_xyz(landmarks, width, height, LM_IDX[name]) for name in names], dtype=np.float32)


def derived_au_points(landmarks, width: int, height: int) -> dict[str, np.ndarray]:
    def pt(name: str) -> np.ndarray:
        return landmark_xy(landmarks, width, height, LM_IDX[name])

    def mid(*names: str) -> np.ndarray:
        return np.mean([pt(name) for name in names], axis=0)

    return {
        "AU1_inner_brow_subR": np.array([pt("left_brow_inner")], dtype=np.float32),
        "AU1_inner_brow_subL": np.array([pt("right_brow_inner")], dtype=np.float32),
        "AU4_brow_lowerer": np.array([mid("left_brow_inner", "right_brow_inner")], dtype=np.float32),
        "AU2_brow_outer_subR": np.array([mid("left_brow_outer", "left_brow_peak")], dtype=np.float32),
        "AU2_brow_outer_subL": np.array([mid("right_brow_outer", "right_brow_peak")], dtype=np.float32),
        "AU6_cheek_subR": np.array([mid("left_eye_outer", "left_lip_corner")], dtype=np.float32),
        "AU6_cheek_subL": np.array([mid("right_eye_outer", "right_lip_corner")], dtype=np.float32),
        "AU9_nose_bridge": np.array([mid("nose_top", "left_nostril", "right_nostril")], dtype=np.float32),
        "AU10_upper_lip": np.array([mid("left_nostril", "right_nostril", "upper_lip_top")], dtype=np.float32),
        "AU12_corner_subR": np.array([pt("left_lip_corner")], dtype=np.float32),
        "AU12_corner_subL": np.array([pt("right_lip_corner")], dtype=np.float32),
        "AU23_lip_center": np.array([mid("upper_lip_top", "lower_lip_bot")], dtype=np.float32),
        "AU17_chin": np.array([mid("lower_lip_bot", "chin_bottom")], dtype=np.float32),
        "AU20_corner_subR": np.array([pt("left_lip_corner")], dtype=np.float32),
        "AU20_corner_subL": np.array([pt("right_lip_corner")], dtype=np.float32),
        "AU26_jaw": np.array([pt("lower_lip_bot")], dtype=np.float32),
    }


def derived_au_points_xyz(landmarks, width: int, height: int) -> dict[str, np.ndarray]:
    def pt(name: str) -> np.ndarray:
        return landmark_xyz(landmarks, width, height, LM_IDX[name])

    def mid(*names: str) -> np.ndarray:
        return np.mean([pt(name) for name in names], axis=0)

    return {
        "AU1_inner_brow_subR": np.array([pt("left_brow_inner")], dtype=np.float32),
        "AU1_inner_brow_subL": np.array([pt("right_brow_inner")], dtype=np.float32),
        "AU4_brow_lowerer": np.array([mid("left_brow_inner", "right_brow_inner")], dtype=np.float32),
        "AU2_brow_outer_subR": np.array([mid("left_brow_outer", "left_brow_peak")], dtype=np.float32),
        "AU2_brow_outer_subL": np.array([mid("right_brow_outer", "right_brow_peak")], dtype=np.float32),
        "AU6_cheek_subR": np.array([mid("left_eye_outer", "left_lip_corner")], dtype=np.float32),
        "AU6_cheek_subL": np.array([mid("right_eye_outer", "right_lip_corner")], dtype=np.float32),
        "AU9_nose_bridge": np.array([mid("nose_top", "left_nostril", "right_nostril")], dtype=np.float32),
        "AU10_upper_lip": np.array([mid("left_nostril", "right_nostril", "upper_lip_top")], dtype=np.float32),
        "AU12_corner_subR": np.array([pt("left_lip_corner")], dtype=np.float32),
        "AU12_corner_subL": np.array([pt("right_lip_corner")], dtype=np.float32),
        "AU23_lip_center": np.array([mid("upper_lip_top", "lower_lip_bot")], dtype=np.float32),
        "AU17_chin": np.array([mid("lower_lip_bot", "chin_bottom")], dtype=np.float32),
        "AU20_corner_subR": np.array([pt("left_lip_corner")], dtype=np.float32),
        "AU20_corner_subL": np.array([pt("right_lip_corner")], dtype=np.float32),
        "AU26_jaw": np.array([pt("lower_lip_bot")], dtype=np.float32),
    }


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homog = np.concatenate([points.astype(np.float32), ones], axis=1)
    return (homog @ matrix.T).astype(np.float32)


def transform_points_xyz(points: np.ndarray, matrix3d: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homog = np.concatenate([points.astype(np.float32), ones], axis=1)
    return (homog @ matrix3d.T).astype(np.float32)


def _ratio(current: float, reference: float, *, neutral: float = 1.0, min_reference: float = 1.0) -> float:
    if reference <= min_reference:
        return neutral
    return float(current) / float(reference)


def _balance_score(values: list[float]) -> float:
    nonzero = [float(value) for value in values if float(value) > 0.0]
    if len(nonzero) <= 1:
        return 1.0 if nonzero else 0.0
    high = max(nonzero)
    low = min(nonzero)
    if high <= 1e-6:
        return 0.0
    return round(max(0.0, low / high), 4)


def _named_point_map(landmarks, width: int, height: int, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {
        name: landmark_xy(landmarks, width, height, LM_IDX[name])
        for name in names
    }


def _named_point_map_xyz(landmarks, width: int, height: int, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {
        name: landmark_xyz(landmarks, width, height, LM_IDX[name])
        for name in names
    }


def _estimate_similarity_transform_3d(
    src: np.ndarray,
    dst: np.ndarray,
) -> dict[str, Any] | None:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or src.shape[0] < 3:
        return None
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    src_var = float(np.mean(np.sum(src_centered**2, axis=1)))
    if src_var <= 1e-8:
        return None
    covariance = (dst_centered.T @ src_centered) / float(src.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.ones(3, dtype=np.float64)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        correction[-1] = -1.0
    rotation = u @ np.diag(correction) @ vt
    scale = float((singular_values * correction).sum() / src_var)
    translation = dst_mean - scale * (rotation @ src_mean)
    matrix3d = np.concatenate([scale * rotation, translation.reshape(3, 1)], axis=1).astype(np.float32)
    predicted = (scale * (rotation @ src.T)).T + translation
    residuals = np.linalg.norm(dst - predicted, axis=1)
    trace = float(np.trace(rotation))
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    rotation_deg = float(np.degrees(np.arccos(cosine)))
    return {
        "matrix3d": matrix3d,
        "scale3d": round(scale, 6),
        "rotation3d_deg": round(rotation_deg, 4),
        "translation3d": [round(float(value), 4) for value in translation],
        "landmark_fit_rmse_3d": round(float(np.sqrt(np.mean(np.square(residuals)))), 4),
    }


def _transform_point_map(
    point_map: dict[str, np.ndarray],
    transform_info: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    if not transform_info:
        return {name: point.astype(np.float32) for name, point in point_map.items()}
    matrix = transform_info.get("matrix")
    if matrix is None:
        return {name: point.astype(np.float32) for name, point in point_map.items()}
    transformed: dict[str, np.ndarray] = {}
    for name, point in point_map.items():
        transformed[name] = transform_points(np.array([point], dtype=np.float32), matrix)[0]
    return transformed


def _transform_point_map_xyz(
    point_map: dict[str, np.ndarray],
    transform_info: dict[str, Any] | None,
) -> dict[str, np.ndarray]:
    if transform_info:
        matrix3d = transform_info.get("matrix3d")
        if matrix3d is not None:
            transformed: dict[str, np.ndarray] = {}
            for name, point in point_map.items():
                xyz = np.concatenate([point.astype(np.float32), np.ones((1,), dtype=np.float32)])
                transformed[name] = (matrix3d @ xyz).astype(np.float32)
            return transformed
    xy_points = {name: point[:2] for name, point in point_map.items()}
    transformed_xy = _transform_point_map(xy_points, transform_info)
    return {
        name: np.array(
            [
                transformed_xy[name][0],
                transformed_xy[name][1],
                point[2],
            ],
            dtype=np.float32,
        )
        for name, point in point_map.items()
    }


def compute_eye_geometry_metrics(
    onset_landmarks,
    apex_landmarks,
    *,
    dataset: str = "",
    width: int,
    height: int,
    iod: float,
    transform_info: dict[str, Any] | None,
) -> dict[str, float]:
    if onset_landmarks is None or apex_landmarks is None:
        return {}

    names = (
        "left_brow_inner",
        "right_brow_inner",
        "left_brow_outer",
        "right_brow_outer",
        "left_brow_peak",
        "right_brow_peak",
        "left_lid_top",
        "left_lid_bot",
        "right_lid_top",
        "right_lid_bot",
        "nose_top",
    )
    ref_points = _transform_point_map(
        _named_point_map(onset_landmarks, width, height, names),
        transform_info,
    )
    ref_points_xyz = _transform_point_map_xyz(
        _named_point_map_xyz(onset_landmarks, width, height, names),
        transform_info,
    )
    brow_ref_points = (
        {name: point[:2].astype(np.float32) for name, point in ref_points_xyz.items()}
        if dataset == "samm"
        else ref_points
    )
    apex_points = _named_point_map(apex_landmarks, width, height, names)
    apex_points_xyz = _named_point_map_xyz(apex_landmarks, width, height, names)

    left_ref = abs(ref_points["left_lid_bot"][1] - ref_points["left_lid_top"][1])
    right_ref = abs(ref_points["right_lid_bot"][1] - ref_points["right_lid_top"][1])
    left_apex = abs(apex_points["left_lid_bot"][1] - apex_points["left_lid_top"][1])
    right_apex = abs(apex_points["right_lid_bot"][1] - apex_points["right_lid_top"][1])

    left_ratio = _ratio(left_apex, left_ref)
    right_ratio = _ratio(right_apex, right_ref)
    mean_ratio = (left_ratio + right_ratio) / 2.0
    min_ratio = min(left_ratio, right_ratio)
    max_ratio = max(left_ratio, right_ratio)
    asymmetry = abs(left_ratio - right_ratio)
    min_apex_height = min(left_apex, right_apex)
    mean_apex_height = (left_apex + right_apex) / 2.0
    brow_gap_left_ref = abs(brow_ref_points["left_lid_top"][1] - brow_ref_points["left_brow_inner"][1])
    brow_gap_right_ref = abs(brow_ref_points["right_lid_top"][1] - brow_ref_points["right_brow_inner"][1])
    brow_gap_left_apex = abs(apex_points["left_lid_top"][1] - apex_points["left_brow_inner"][1])
    brow_gap_right_apex = abs(apex_points["right_lid_top"][1] - apex_points["right_brow_inner"][1])
    left_gap_ratio = _ratio(brow_gap_left_apex, brow_gap_left_ref)
    right_gap_ratio = _ratio(brow_gap_right_apex, brow_gap_right_ref)
    left_gap_shrink = max(0.0, 1.0 - left_gap_ratio)
    right_gap_shrink = max(0.0, 1.0 - right_gap_ratio)
    left_brow_raise = max(0.0, float(brow_ref_points["left_brow_inner"][1] - apex_points["left_brow_inner"][1])) / max(iod, 1e-6)
    right_brow_raise = max(0.0, float(brow_ref_points["right_brow_inner"][1] - apex_points["right_brow_inner"][1])) / max(iod, 1e-6)
    left_brow_drop = max(0.0, float(apex_points["left_brow_inner"][1] - brow_ref_points["left_brow_inner"][1])) / max(iod, 1e-6)
    right_brow_drop = max(0.0, float(apex_points["right_brow_inner"][1] - brow_ref_points["right_brow_inner"][1])) / max(iod, 1e-6)
    left_lid_raise = max(0.0, float(brow_ref_points["left_lid_top"][1] - apex_points["left_lid_top"][1])) / max(iod, 1e-6)
    right_lid_raise = max(0.0, float(brow_ref_points["right_lid_top"][1] - apex_points["right_lid_top"][1])) / max(iod, 1e-6)
    left_lid_drop = max(0.0, float(apex_points["left_lid_top"][1] - brow_ref_points["left_lid_top"][1])) / max(iod, 1e-6)
    right_lid_drop = max(0.0, float(apex_points["right_lid_top"][1] - brow_ref_points["right_lid_top"][1])) / max(iod, 1e-6)
    left_relative_raise = max(0.0, left_brow_raise - 0.7 * left_lid_raise)
    right_relative_raise = max(0.0, right_brow_raise - 0.7 * right_lid_raise)
    left_relative_drop = max(0.0, left_brow_drop - 0.55 * left_lid_drop)
    right_relative_drop = max(0.0, right_brow_drop - 0.55 * right_lid_drop)
    left_brow_forward = max(
        0.0,
        float(ref_points_xyz["left_brow_inner"][2] - apex_points_xyz["left_brow_inner"][2]),
    ) / max(iod, 1e-6)
    right_brow_forward = max(
        0.0,
        float(ref_points_xyz["right_brow_inner"][2] - apex_points_xyz["right_brow_inner"][2]),
    ) / max(iod, 1e-6)
    left_lid_forward = max(
        0.0,
        float(ref_points_xyz["left_lid_top"][2] - apex_points_xyz["left_lid_top"][2]),
    ) / max(iod, 1e-6)
    right_lid_forward = max(
        0.0,
        float(ref_points_xyz["right_lid_top"][2] - apex_points_xyz["right_lid_top"][2]),
    ) / max(iod, 1e-6)
    left_forward_relative = max(0.0, left_brow_forward - 0.7 * left_lid_forward)
    right_forward_relative = max(0.0, right_brow_forward - 0.7 * right_lid_forward)
    left_outer_ref = (brow_ref_points["left_brow_outer"] + brow_ref_points["left_brow_peak"]) / 2.0
    right_outer_ref = (brow_ref_points["right_brow_outer"] + brow_ref_points["right_brow_peak"]) / 2.0
    left_outer_apex = (apex_points["left_brow_outer"] + apex_points["left_brow_peak"]) / 2.0
    right_outer_apex = (apex_points["right_brow_outer"] + apex_points["right_brow_peak"]) / 2.0
    left_outer_ref_xyz = (ref_points_xyz["left_brow_outer"] + ref_points_xyz["left_brow_peak"]) / 2.0
    right_outer_ref_xyz = (ref_points_xyz["right_brow_outer"] + ref_points_xyz["right_brow_peak"]) / 2.0
    left_outer_apex_xyz = (apex_points_xyz["left_brow_outer"] + apex_points_xyz["left_brow_peak"]) / 2.0
    right_outer_apex_xyz = (apex_points_xyz["right_brow_outer"] + apex_points_xyz["right_brow_peak"]) / 2.0
    left_outer_raise = max(0.0, float(left_outer_ref[1] - left_outer_apex[1])) / max(iod, 1e-6)
    right_outer_raise = max(0.0, float(right_outer_ref[1] - right_outer_apex[1])) / max(iod, 1e-6)
    left_outer_drop = max(0.0, float(left_outer_apex[1] - left_outer_ref[1])) / max(iod, 1e-6)
    right_outer_drop = max(0.0, float(right_outer_apex[1] - right_outer_ref[1])) / max(iod, 1e-6)
    left_outer_relative_raise = max(0.0, left_outer_raise - 0.7 * left_lid_raise)
    right_outer_relative_raise = max(0.0, right_outer_raise - 0.7 * right_lid_raise)
    left_outer_relative_drop = max(0.0, left_outer_drop - 0.55 * left_lid_drop)
    right_outer_relative_drop = max(0.0, right_outer_drop - 0.55 * right_lid_drop)
    left_outer_forward = max(0.0, float(left_outer_ref_xyz[2] - left_outer_apex_xyz[2])) / max(iod, 1e-6)
    right_outer_forward = max(0.0, float(right_outer_ref_xyz[2] - right_outer_apex_xyz[2])) / max(iod, 1e-6)
    left_outer_forward_relative = max(0.0, left_outer_forward - 0.7 * left_lid_forward)
    right_outer_forward_relative = max(0.0, right_outer_forward - 0.7 * right_lid_forward)
    brow_center_ref = (brow_ref_points["left_brow_inner"] + brow_ref_points["right_brow_inner"]) / 2.0
    brow_center_apex = (apex_points["left_brow_inner"] + apex_points["right_brow_inner"]) / 2.0
    brow_center_raise_norm = max(0.0, float(brow_center_ref[1] - brow_center_apex[1])) / max(iod, 1e-6)
    brow_center_drop_norm = max(0.0, float(brow_center_apex[1] - brow_center_ref[1])) / max(iod, 1e-6)

    return {
        "left_ratio": round(left_ratio, 4),
        "right_ratio": round(right_ratio, 4),
        "mean_ratio": round(mean_ratio, 4),
        "min_ratio": round(min_ratio, 4),
        "max_ratio": round(max_ratio, 4),
        "asymmetry": round(asymmetry, 4),
        "apex_min_height_norm": round(min_apex_height / max(iod, 1e-6), 4),
        "apex_mean_height_norm": round(mean_apex_height / max(iod, 1e-6), 4),
        "left_gap_ratio": round(left_gap_ratio, 4),
        "right_gap_ratio": round(right_gap_ratio, 4),
        "mean_gap_ratio": round((left_gap_ratio + right_gap_ratio) / 2.0, 4),
        "left_gap_shrink": round(left_gap_shrink, 4),
        "right_gap_shrink": round(right_gap_shrink, 4),
        "mean_gap_shrink": round((left_gap_shrink + right_gap_shrink) / 2.0, 4),
        "min_gap_ratio": round(min(left_gap_ratio, right_gap_ratio), 4),
        "gap_balance": _balance_score([left_gap_ratio, right_gap_ratio]),
        "left_inner_brow_raise_norm": round(left_brow_raise, 4),
        "right_inner_brow_raise_norm": round(right_brow_raise, 4),
        "inner_brow_raise_mean": round((left_brow_raise + right_brow_raise) / 2.0, 4),
        "left_inner_brow_drop_norm": round(left_brow_drop, 4),
        "right_inner_brow_drop_norm": round(right_brow_drop, 4),
        "inner_brow_drop_mean": round((left_brow_drop + right_brow_drop) / 2.0, 4),
        "left_inner_brow_relative_raise_norm": round(left_relative_raise, 4),
        "right_inner_brow_relative_raise_norm": round(right_relative_raise, 4),
        "inner_brow_relative_raise_mean": round((left_relative_raise + right_relative_raise) / 2.0, 4),
        "left_inner_brow_relative_drop_norm": round(left_relative_drop, 4),
        "right_inner_brow_relative_drop_norm": round(right_relative_drop, 4),
        "inner_brow_relative_drop_mean": round((left_relative_drop + right_relative_drop) / 2.0, 4),
        "inner_brow_raise_balance": _balance_score([left_brow_raise, right_brow_raise]),
        "left_inner_brow_forward_norm": round(left_brow_forward, 4),
        "right_inner_brow_forward_norm": round(right_brow_forward, 4),
        "inner_brow_forward_mean": round((left_brow_forward + right_brow_forward) / 2.0, 4),
        "left_inner_brow_forward_relative_norm": round(left_forward_relative, 4),
        "right_inner_brow_forward_relative_norm": round(right_forward_relative, 4),
        "inner_brow_forward_relative_mean": round((left_forward_relative + right_forward_relative) / 2.0, 4),
        "left_outer_brow_raise_norm": round(left_outer_raise, 4),
        "right_outer_brow_raise_norm": round(right_outer_raise, 4),
        "outer_brow_raise_mean": round((left_outer_raise + right_outer_raise) / 2.0, 4),
        "left_outer_brow_drop_norm": round(left_outer_drop, 4),
        "right_outer_brow_drop_norm": round(right_outer_drop, 4),
        "outer_brow_drop_mean": round((left_outer_drop + right_outer_drop) / 2.0, 4),
        "left_outer_brow_relative_raise_norm": round(left_outer_relative_raise, 4),
        "right_outer_brow_relative_raise_norm": round(right_outer_relative_raise, 4),
        "outer_brow_relative_raise_mean": round((left_outer_relative_raise + right_outer_relative_raise) / 2.0, 4),
        "left_outer_brow_relative_drop_norm": round(left_outer_relative_drop, 4),
        "right_outer_brow_relative_drop_norm": round(right_outer_relative_drop, 4),
        "outer_brow_relative_drop_mean": round((left_outer_relative_drop + right_outer_relative_drop) / 2.0, 4),
        "left_outer_brow_forward_norm": round(left_outer_forward, 4),
        "right_outer_brow_forward_norm": round(right_outer_forward, 4),
        "outer_brow_forward_mean": round((left_outer_forward + right_outer_forward) / 2.0, 4),
        "left_outer_brow_forward_relative_norm": round(left_outer_forward_relative, 4),
        "right_outer_brow_forward_relative_norm": round(right_outer_forward_relative, 4),
        "outer_brow_forward_relative_mean": round((left_outer_forward_relative + right_outer_forward_relative) / 2.0, 4),
        "outer_brow_raise_balance": _balance_score([left_outer_raise, right_outer_raise]),
        "brow_center_raise_norm": round(brow_center_raise_norm, 4),
        "brow_center_drop_norm": round(brow_center_drop_norm, 4),
        "pose_landmark_fit_rmse_3d": round(float((transform_info or {}).get("landmark_fit_rmse_3d", 0.0) or 0.0), 4),
    }


def compute_mouth_geometry_metrics(
    onset_landmarks,
    apex_landmarks,
    *,
    width: int,
    height: int,
    transform_info: dict[str, Any] | None,
) -> dict[str, float]:
    if onset_landmarks is None or apex_landmarks is None:
        return {}

    names = (
        "left_lip_corner",
        "right_lip_corner",
        "upper_lip_top",
        "lower_lip_bot",
        "chin_bottom",
        "nose_top",
        "left_nostril",
        "right_nostril",
    )
    ref_points = _transform_point_map(
        _named_point_map(onset_landmarks, width, height, names),
        transform_info,
    )
    ref_points_xyz = _transform_point_map_xyz(
        _named_point_map_xyz(onset_landmarks, width, height, names),
        transform_info,
    )
    apex_points = _named_point_map(apex_landmarks, width, height, names)
    apex_points_xyz = _named_point_map_xyz(apex_landmarks, width, height, names)

    def mouth_center(points: dict[str, np.ndarray]) -> np.ndarray:
        return (points["upper_lip_top"] + points["lower_lip_bot"]) / 2.0

    def nose_mid(points: dict[str, np.ndarray]) -> np.ndarray:
        return (points["left_nostril"] + points["right_nostril"]) / 2.0

    def mouth_center_xyz(points: dict[str, np.ndarray]) -> np.ndarray:
        return (points["upper_lip_top"] + points["lower_lip_bot"]) / 2.0

    def nose_mid_xyz(points: dict[str, np.ndarray]) -> np.ndarray:
        return (points["left_nostril"] + points["right_nostril"]) / 2.0

    ref_center = mouth_center(ref_points)
    apex_center = mouth_center(apex_points)
    ref_nose = nose_mid(ref_points)
    apex_nose = nose_mid(apex_points)
    ref_center_xyz = mouth_center_xyz(ref_points_xyz)
    apex_center_xyz = mouth_center_xyz(apex_points_xyz)
    ref_nose_xyz = nose_mid_xyz(ref_points_xyz)
    apex_nose_xyz = nose_mid_xyz(apex_points_xyz)

    ref_width = float(np.linalg.norm(ref_points["right_lip_corner"] - ref_points["left_lip_corner"]))
    apex_width = float(np.linalg.norm(apex_points["right_lip_corner"] - apex_points["left_lip_corner"]))
    ref_open = abs(ref_points["lower_lip_bot"][1] - ref_points["upper_lip_top"][1])
    apex_open = abs(apex_points["lower_lip_bot"][1] - apex_points["upper_lip_top"][1])
    ref_upper_nose = float(np.linalg.norm(ref_points["upper_lip_top"] - ref_nose))
    apex_upper_nose = float(np.linalg.norm(apex_points["upper_lip_top"] - apex_nose))
    ref_lower_nose = float(np.linalg.norm(ref_points["lower_lip_bot"] - ref_nose))
    apex_lower_nose = float(np.linalg.norm(apex_points["lower_lip_bot"] - apex_nose))
    ref_bridge = abs(ref_points["nose_top"][1] - ref_nose[1])
    apex_bridge = abs(apex_points["nose_top"][1] - apex_nose[1])
    ref_nostril_width = float(np.linalg.norm(ref_points["right_nostril"] - ref_points["left_nostril"]))
    apex_nostril_width = float(np.linalg.norm(apex_points["right_nostril"] - apex_points["left_nostril"]))

    ref_left_rel = float(ref_points["left_lip_corner"][1] - ref_center[1])
    ref_right_rel = float(ref_points["right_lip_corner"][1] - ref_center[1])
    apex_left_rel = float(apex_points["left_lip_corner"][1] - apex_center[1])
    apex_right_rel = float(apex_points["right_lip_corner"][1] - apex_center[1])
    width_norm = max(ref_width, 1.0)

    left_corner_lift = max(0.0, ref_left_rel - apex_left_rel) / width_norm
    right_corner_lift = max(0.0, ref_right_rel - apex_right_rel) / width_norm
    left_corner_depress = max(0.0, apex_left_rel - ref_left_rel) / width_norm
    right_corner_depress = max(0.0, apex_right_rel - ref_right_rel) / width_norm
    left_corner_raise = max(0.0, float(ref_points["left_lip_corner"][1] - apex_points["left_lip_corner"][1])) / width_norm
    right_corner_raise = max(0.0, float(ref_points["right_lip_corner"][1] - apex_points["right_lip_corner"][1])) / width_norm
    left_corner_drop = max(0.0, float(apex_points["left_lip_corner"][1] - ref_points["left_lip_corner"][1])) / width_norm
    right_corner_drop = max(0.0, float(apex_points["right_lip_corner"][1] - ref_points["right_lip_corner"][1])) / width_norm
    left_corner_outward = max(0.0, float(ref_points["left_lip_corner"][0] - apex_points["left_lip_corner"][0])) / width_norm
    right_corner_outward = max(0.0, float(apex_points["right_lip_corner"][0] - ref_points["right_lip_corner"][0])) / width_norm
    left_corner_inward = max(0.0, float(apex_points["left_lip_corner"][0] - ref_points["left_lip_corner"][0])) / width_norm
    right_corner_inward = max(0.0, float(ref_points["right_lip_corner"][0] - apex_points["right_lip_corner"][0])) / width_norm
    lower_lip_drop_norm = max(
        0.0,
        float(apex_points["lower_lip_bot"][1] - apex_nose[1])
        - float(ref_points["lower_lip_bot"][1] - ref_nose[1]),
    ) / width_norm
    mouth_center_raise_norm = max(
        0.0,
        float(ref_center[1] - ref_nose[1]) - float(apex_center[1] - apex_nose[1]),
    ) / width_norm
    mouth_center_drop_norm = max(
        0.0,
        float(apex_center[1] - apex_nose[1]) - float(ref_center[1] - ref_nose[1]),
    ) / width_norm
    lower_lip_raise_norm = max(
        0.0,
        float(ref_points["lower_lip_bot"][1] - ref_nose[1]) - float(apex_points["lower_lip_bot"][1] - apex_nose[1]),
    ) / width_norm
    upper_lip_raise_norm = max(
        0.0,
        float(ref_points["upper_lip_top"][1] - ref_nose[1]) - float(apex_points["upper_lip_top"][1] - apex_nose[1]),
    ) / width_norm
    chin_raise_norm = max(
        0.0,
        float(ref_points["chin_bottom"][1] - ref_nose[1]) - float(apex_points["chin_bottom"][1] - apex_nose[1]),
    ) / width_norm
    left_corner_raise_relative = max(0.0, left_corner_raise - mouth_center_raise_norm)
    right_corner_raise_relative = max(0.0, right_corner_raise - mouth_center_raise_norm)
    left_nostril_raise = max(
        0.0,
        float(ref_points["left_nostril"][1] - ref_points["nose_top"][1])
        - float(apex_points["left_nostril"][1] - apex_points["nose_top"][1]),
    ) / width_norm
    right_nostril_raise = max(
        0.0,
        float(ref_points["right_nostril"][1] - ref_points["nose_top"][1])
        - float(apex_points["right_nostril"][1] - apex_points["nose_top"][1]),
    ) / width_norm
    mouth_center_backward_norm = max(
        0.0,
        float(apex_center_xyz[2] - apex_nose_xyz[2]) - float(ref_center_xyz[2] - ref_nose_xyz[2]),
    ) / width_norm
    lower_lip_forward_norm = max(
        0.0,
        float(ref_points_xyz["lower_lip_bot"][2] - ref_nose_xyz[2]) - float(apex_points_xyz["lower_lip_bot"][2] - apex_nose_xyz[2]),
    ) / width_norm
    chin_forward_norm = max(
        0.0,
        float(ref_points_xyz["chin_bottom"][2] - ref_nose_xyz[2]) - float(apex_points_xyz["chin_bottom"][2] - apex_nose_xyz[2]),
    ) / width_norm
    left_corner_forward = max(
        0.0,
        float(ref_points_xyz["left_lip_corner"][2] - ref_nose_xyz[2]) - float(apex_points_xyz["left_lip_corner"][2] - apex_nose_xyz[2]),
    ) / width_norm
    right_corner_forward = max(
        0.0,
        float(ref_points_xyz["right_lip_corner"][2] - ref_nose_xyz[2]) - float(apex_points_xyz["right_lip_corner"][2] - apex_nose_xyz[2]),
    ) / width_norm
    mouth_center_forward_norm = max(
        0.0,
        float(ref_center_xyz[2] - ref_nose_xyz[2]) - float(apex_center_xyz[2] - apex_nose_xyz[2]),
    ) / width_norm
    upper_lip_forward_norm = max(
        0.0,
        float(ref_points_xyz["upper_lip_top"][2] - ref_nose_xyz[2]) - float(apex_points_xyz["upper_lip_top"][2] - apex_nose_xyz[2]),
    ) / width_norm
    left_corner_backward = max(
        0.0,
        (
            float(apex_points_xyz["left_lip_corner"][2] - apex_nose_xyz[2])
            - float(ref_points_xyz["left_lip_corner"][2] - ref_nose_xyz[2])
        )
        / width_norm
        - 0.55 * mouth_center_backward_norm,
    )
    right_corner_backward = max(
        0.0,
        (
            float(apex_points_xyz["right_lip_corner"][2] - apex_nose_xyz[2])
            - float(ref_points_xyz["right_lip_corner"][2] - ref_nose_xyz[2])
        )
        / width_norm
        - 0.55 * mouth_center_backward_norm,
    )
    nose_bridge_shrink_norm = max(0.0, ref_bridge - apex_bridge) / width_norm
    upper_lip_raise_relative_norm = max(0.0, upper_lip_raise_norm - 0.6 * mouth_center_raise_norm)

    return {
        "mouth_open_ratio": round(_ratio(apex_open, ref_open), 4),
        "mouth_width_ratio": round(_ratio(apex_width, ref_width), 4),
        "upper_lip_nose_ratio": round(_ratio(apex_upper_nose, ref_upper_nose), 4),
        "lower_lip_nose_ratio": round(_ratio(apex_lower_nose, ref_lower_nose), 4),
        "nostril_width_ratio": round(_ratio(apex_nostril_width, ref_nostril_width), 4),
        "left_corner_lift": round(left_corner_lift, 4),
        "right_corner_lift": round(right_corner_lift, 4),
        "left_corner_depress": round(left_corner_depress, 4),
        "right_corner_depress": round(right_corner_depress, 4),
        "left_corner_raise": round(left_corner_raise, 4),
        "right_corner_raise": round(right_corner_raise, 4),
        "left_corner_drop": round(left_corner_drop, 4),
        "right_corner_drop": round(right_corner_drop, 4),
        "left_corner_outward": round(left_corner_outward, 4),
        "right_corner_outward": round(right_corner_outward, 4),
        "left_corner_inward": round(left_corner_inward, 4),
        "right_corner_inward": round(right_corner_inward, 4),
        "left_corner_raise_relative": round(left_corner_raise_relative, 4),
        "right_corner_raise_relative": round(right_corner_raise_relative, 4),
        "left_nostril_raise_norm": round(left_nostril_raise, 4),
        "right_nostril_raise_norm": round(right_nostril_raise, 4),
        "corner_lift_mean": round((left_corner_lift + right_corner_lift) / 2.0, 4),
        "corner_depress_mean": round((left_corner_depress + right_corner_depress) / 2.0, 4),
        "corner_raise_mean": round((left_corner_raise + right_corner_raise) / 2.0, 4),
        "corner_drop_mean": round((left_corner_drop + right_corner_drop) / 2.0, 4),
        "corner_outward_mean": round((left_corner_outward + right_corner_outward) / 2.0, 4),
        "corner_inward_mean": round((left_corner_inward + right_corner_inward) / 2.0, 4),
        "corner_raise_relative_mean": round((left_corner_raise_relative + right_corner_raise_relative) / 2.0, 4),
        "corner_lift_balance": _balance_score([left_corner_lift, right_corner_lift]),
        "corner_depress_balance": _balance_score([left_corner_depress, right_corner_depress]),
        "corner_raise_balance": _balance_score([left_corner_raise, right_corner_raise]),
        "corner_raise_relative_balance": _balance_score([left_corner_raise_relative, right_corner_raise_relative]),
        "corner_drop_balance": _balance_score([left_corner_drop, right_corner_drop]),
        "corner_outward_balance": _balance_score([left_corner_outward, right_corner_outward]),
        "corner_inward_balance": _balance_score([left_corner_inward, right_corner_inward]),
        "corner_asymmetry": round(abs(left_corner_lift - right_corner_lift), 4),
        "lower_lip_drop_norm": round(lower_lip_drop_norm, 4),
        "mouth_center_raise_norm": round(mouth_center_raise_norm, 4),
        "mouth_center_drop_norm": round(mouth_center_drop_norm, 4),
        "lower_lip_raise_norm": round(lower_lip_raise_norm, 4),
        "upper_lip_raise_norm": round(upper_lip_raise_norm, 4),
        "upper_lip_raise_relative_norm": round(upper_lip_raise_relative_norm, 4),
        "chin_raise_norm": round(chin_raise_norm, 4),
        "nostril_raise_mean": round((left_nostril_raise + right_nostril_raise) / 2.0, 4),
        "nostril_raise_balance": _balance_score([left_nostril_raise, right_nostril_raise]),
        "lower_lip_forward_norm": round(lower_lip_forward_norm, 4),
        "upper_lip_forward_norm": round(upper_lip_forward_norm, 4),
        "chin_forward_norm": round(chin_forward_norm, 4),
        "nose_bridge_shrink_norm": round(nose_bridge_shrink_norm, 4),
        "left_corner_forward_norm": round(left_corner_forward, 4),
        "right_corner_forward_norm": round(right_corner_forward, 4),
        "corner_forward_mean": round((left_corner_forward + right_corner_forward) / 2.0, 4),
        "left_corner_backward_norm": round(left_corner_backward, 4),
        "right_corner_backward_norm": round(right_corner_backward, 4),
        "corner_backward_mean": round((left_corner_backward + right_corner_backward) / 2.0, 4),
        "corner_backward_balance": _balance_score([left_corner_backward, right_corner_backward]),
        "mouth_center_forward_norm": round(mouth_center_forward_norm, 4),
        "mouth_center_backward_norm": round(mouth_center_backward_norm, 4),
    }


def residual_motion_stat(delta: np.ndarray, iod: float) -> dict[str, Any]:
    dx = float(delta[0])
    dy = float(delta[1])
    mag_px = float(np.linalg.norm(delta))
    mag_norm = mag_px / iod if iod > 1e-6 else 0.0
    angle = float(np.degrees(np.arctan2(-dy, dx)) % 360)
    return {
        "dx": round(dx, 4),
        "dy": round(dy, 4),
        "mag_px": round(mag_px, 4),
        "mag": round(mag_norm, 4),
        "angle": round(angle, 1),
        "dir": angle_to_direction(angle),
        "active": round(1.0 if mag_norm >= 0.01 else 0.0, 3),
    }


def compute_landmark_residual_au_features(
    onset_landmarks,
    apex_landmarks,
    dataset: str,
    width: int,
    height: int,
    iod: float,
    transform_info: dict[str, Any] | None,
) -> dict[str, dict[str, Any] | None]:
    onset_points = derived_au_points(onset_landmarks, width, height)
    onset_points_xyz = derived_au_points_xyz(onset_landmarks, width, height)
    apex_points = derived_au_points(apex_landmarks, width, height)
    if transform_info is None:
        predicted_points = onset_points
    else:
        predicted_points = {
            name: transform_points(points, transform_info["matrix"])
            for name, points in onset_points.items()
        }
        matrix3d = transform_info.get("matrix3d")
        if dataset == "samm" and matrix3d is not None:
            brow_regions = {
                "AU1_inner_brow_subR",
                "AU1_inner_brow_subL",
                "AU4_brow_lowerer",
                "AU2_brow_outer_subR",
                "AU2_brow_outer_subL",
            }
            predicted_xyz = {
                name: transform_points_xyz(points, matrix3d)
                for name, points in onset_points_xyz.items()
            }
            for name in brow_regions:
                if name in predicted_xyz:
                    predicted_points[name] = predicted_xyz[name][:, :2]

    result: dict[str, dict[str, Any] | None] = {}
    for name in AU_REGION_NAMES:
        src_pred = predicted_points.get(name)
        dst = apex_points.get(name)
        if src_pred is None or dst is None or len(src_pred) == 0 or len(dst) == 0:
            result[name] = None
            continue
        delta = np.mean(dst - src_pred, axis=0)
        result[name] = residual_motion_stat(delta, iod)
    return result


def estimate_head_motion(
    height: int,
    width: int,
    u: np.ndarray,
    v: np.ndarray,
    mag: np.ndarray,
    landmarks,
) -> dict[str, Any]:
    def px(name: str) -> np.ndarray:
        return np.array([landmarks[LM_IDX[name]].x * width, landmarks[LM_IDX[name]].y * height])

    iod = float(np.linalg.norm(px("right_eye_outer") - px("left_eye_outer")))
    nose_tip = px("nose_tip")
    mask = _rect_mask(height, width, nose_tip[0], nose_tip[1], iod * 0.10, iod * 0.10)
    masked_mag = mag[mask]
    active = masked_mag > 0.03
    if active.sum() < 4:
        return {
            "tx": 0.0,
            "ty": 0.0,
            "head_mag": 0.0,
            "angle": 0.0,
            "dir": "none",
            "angle_spread": 0.0,
            "quality": "unknown",
            "n_ref": 0,
            "per_region": {},
        }

    weights = masked_mag[active]
    masked_u = u[mask][active]
    masked_v = v[mask][active]
    tx = float((masked_u * weights).sum() / weights.sum())
    ty = float((masked_v * weights).sum() / weights.sum())
    head_mag = float(np.sqrt(tx**2 + ty**2))
    head_angle = float(np.degrees(np.arctan2(-ty, tx)) % 360)
    if head_mag < 0.06:
        quality = "minimal"
    elif head_mag < 0.8:
        quality = "clean"
    elif head_mag < 2.5:
        quality = "moderate"
    else:
        quality = "dominated"

    return {
        "tx": round(tx, 4),
        "ty": round(ty, 4),
        "head_mag": round(head_mag, 4),
        "angle": round(head_angle, 1),
        "dir": angle_to_direction(head_angle),
        "angle_spread": 0.0,
        "quality": quality,
        "n_ref": 1,
        "per_region": {"nose_tip": [round(tx, 3), round(ty, 3)]},
    }


def build_head_motion_transform(
    onset_landmarks,
    apex_landmarks,
    width: int,
    height: int,
    method: str,
) -> dict[str, Any] | None:
    src = landmarks_to_points(onset_landmarks, width, height, RIGID_LANDMARKS)
    dst = landmarks_to_points(apex_landmarks, width, height, RIGID_LANDMARKS)
    src_xyz = landmarks_to_points_xyz(onset_landmarks, width, height, RIGID_LANDMARKS)
    dst_xyz = landmarks_to_points_xyz(apex_landmarks, width, height, RIGID_LANDMARKS)

    if method == "translation":
        delta = np.median(dst - src, axis=0)
        matrix = np.array([[1.0, 0.0, float(delta[0])], [0.0, 1.0, float(delta[1])]], dtype=np.float32)
    elif method == "similarity":
        matrix, _ = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.LMEDS,
            ransacReprojThreshold=3.0,
        )
    elif method == "affine":
        matrix, _ = cv2.estimateAffine2D(
            src,
            dst,
            method=cv2.LMEDS,
            ransacReprojThreshold=3.0,
        )
    else:
        raise ValueError(f"Unsupported head-motion method: {method}")

    if matrix is None:
        return None

    linear = matrix[:, :2]
    translation = matrix[:, 2]
    det = float(np.linalg.det(linear))
    scale_x = float(np.linalg.norm(linear[:, 0]))
    scale_y = float(np.linalg.norm(linear[:, 1]))
    if method in {"translation", "similarity"}:
        rotation_deg = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    else:
        rotation_deg = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))

    residuals = []
    for src_pt, dst_pt in zip(src, dst):
        pred = linear @ src_pt + translation
        residuals.append(float(np.linalg.norm(dst_pt - pred)))

    transform_info: dict[str, Any] = {
        "method": method,
        "matrix": matrix.astype(np.float32),
        "rotation_deg": round(rotation_deg, 4),
        "scale_x": round(scale_x, 4),
        "scale_y": round(scale_y, 4),
        "determinant": round(det, 6),
        "landmark_fit_rmse": round(float(np.sqrt(np.mean(np.square(residuals)))), 4),
        "landmark_fit_max_err": round(float(max(residuals) if residuals else 0.0), 4),
    }
    info3d = _estimate_similarity_transform_3d(src_xyz, dst_xyz)
    if info3d is not None:
        transform_info.update(info3d)
    return transform_info


def apply_head_motion_compensation(
    u: np.ndarray,
    v: np.ndarray,
    transform_info: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if transform_info is None:
        mag = np.sqrt(u**2 + v**2).astype(np.float32)
        return u.copy(), v.copy(), mag

    matrix = transform_info["matrix"]
    height, width = u.shape
    xx, yy = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    pred_x = matrix[0, 0] * xx + matrix[0, 1] * yy + matrix[0, 2]
    pred_y = matrix[1, 0] * xx + matrix[1, 1] * yy + matrix[1, 2]
    du = pred_x - xx
    dv = pred_y - yy
    corrected_u = u - du
    corrected_v = v - dv
    corrected_mag = np.sqrt(corrected_u**2 + corrected_v**2).astype(np.float32)
    return corrected_u, corrected_v, corrected_mag


def compute_eye_height_ratio(onset_path: Path, apex_path: Path, model_path: Path) -> float | None:
    def _eye_heights(path: Path) -> tuple[float, float] | None:
        landmarks, height, width = detect_face_landmarks(path, model_path)
        if landmarks is None:
            return None
        left_h = abs(landmarks[LM_IDX["left_lid_bot"]].y - landmarks[LM_IDX["left_lid_top"]].y) * height
        right_h = abs(landmarks[LM_IDX["right_lid_bot"]].y - landmarks[LM_IDX["right_lid_top"]].y) * height
        return left_h, right_h

    onset = _eye_heights(onset_path)
    apex = _eye_heights(apex_path)
    if onset is None or apex is None:
        return None

    left_ratio = apex[0] / onset[0] if onset[0] > 1.0 else 1.0
    right_ratio = apex[1] / onset[1] if onset[1] > 1.0 else 1.0
    return round((left_ratio + right_ratio) / 2, 3)


def extract_landmark_au_regions(
    onset_path: Path,
    flow_npy_path: Path,
    model_path: Path,
    *,
    mag_threshold: float = 0.05,
    onset_landmarks=None,
    apex_landmarks=None,
    head_motion_method: str | None = None,
) -> tuple[dict[str, dict[str, Any] | None] | None, dict[str, Any] | None, float, dict[str, Any] | None]:
    landmarks = onset_landmarks
    if landmarks is None:
        landmarks, image_h, image_w = detect_face_landmarks(onset_path, model_path)
    else:
        resolved = _resolve_existing_image(onset_path)
        image = np.array(Image.open(resolved).convert("RGB"))
        image_h, image_w = image.shape[:2]
    if landmarks is None:
        return None, None, 0.0, None

    def px(name: str) -> np.ndarray:
        return np.array([landmarks[LM_IDX[name]].x * image_w, landmarks[LM_IDX[name]].y * image_h])

    def mid(*names: str) -> np.ndarray:
        return np.mean([px(name) for name in names], axis=0)

    iod = float(np.linalg.norm(px("right_eye_outer") - px("left_eye_outer")))
    brow_width = float(np.linalg.norm(px("right_brow_outer") - px("left_brow_outer")))

    arr = np.load(flow_npy_path)
    u = arr[:, :, 0].copy()
    v = arr[:, :, 1].copy()
    mag = arr[:, :, 2].copy()
    flow_h, flow_w = arr.shape[:2]

    if flow_h != image_h or flow_w != image_w:
        sx = flow_w / image_w
        sy = flow_h / image_h
        orig_px = px

        def px(name: str) -> np.ndarray:
            return orig_px(name) * np.array([sx, sy])

        def mid(*names: str) -> np.ndarray:
            return np.mean([px(name) for name in names], axis=0)

        iod = float(np.linalg.norm(px("right_eye_outer") - px("left_eye_outer")))
        brow_width = float(np.linalg.norm(px("right_brow_outer") - px("left_brow_outer")))
        image_h, image_w = flow_h, flow_w

    head_motion = estimate_head_motion(image_h, image_w, u, v, mag, landmarks)
    transform_info = None
    if head_motion_method is not None:
        if apex_landmarks is None:
            apex_landmarks, _, _ = detect_face_landmarks(_resolve_existing_image(onset_path).parent / "apex.jpg", model_path)
        transform_info = build_head_motion_transform(
            landmarks,
            apex_landmarks,
            image_w,
            image_h,
            head_motion_method,
        ) if apex_landmarks is not None else None
        u, v, mag = apply_head_motion_compensation(u, v, transform_info)

    brow_mid = mid("left_brow_inner", "right_brow_inner")
    left_out = mid("left_brow_outer", "left_brow_peak")
    right_out = mid("right_brow_outer", "right_brow_peak")
    nose_mid = mid("nose_top", "left_nostril", "right_nostril")
    upper_lip = mid("left_nostril", "right_nostril", "upper_lip_top")
    lip_ctr = mid("upper_lip_top", "lower_lip_bot")
    lip_hw = float(np.linalg.norm(px("right_lip_corner") - px("left_lip_corner"))) * 0.35
    chin_ctr = mid("lower_lip_bot", "chin_bottom")
    inner_r = px("left_brow_inner")
    inner_l = px("right_brow_inner")

    regions = {
        "AU1_inner_brow_subR": _rect_mask(image_h, image_w, inner_r[0], inner_r[1], iod * 0.16, iod * 0.16),
        "AU1_inner_brow_subL": _rect_mask(image_h, image_w, inner_l[0], inner_l[1], iod * 0.16, iod * 0.16),
        "AU4_brow_lowerer": _rect_mask(image_h, image_w, brow_mid[0], brow_mid[1], brow_width * 0.52, iod * 0.18),
        "AU2_brow_outer_subR": _rect_mask(image_h, image_w, left_out[0], left_out[1], iod * 0.22, iod * 0.16),
        "AU2_brow_outer_subL": _rect_mask(image_h, image_w, right_out[0], right_out[1], iod * 0.22, iod * 0.16),
        "AU6_cheek_subR": _rect_mask(
            image_h,
            image_w,
            *((px("left_eye_outer") + px("left_lip_corner")) * 0.5),
            iod * 0.30,
            iod * 0.22,
        ),
        "AU6_cheek_subL": _rect_mask(
            image_h,
            image_w,
            *((px("right_eye_outer") + px("right_lip_corner")) * 0.5),
            iod * 0.30,
            iod * 0.22,
        ),
        "AU9_nose_bridge": _rect_mask(image_h, image_w, nose_mid[0], nose_mid[1], iod * 0.28, iod * 0.20),
        "AU10_upper_lip": _rect_mask(image_h, image_w, upper_lip[0], upper_lip[1], iod * 0.44, iod * 0.20),
        "AU12_corner_subR": _rect_mask(
            image_h,
            image_w,
            px("left_lip_corner")[0],
            px("left_lip_corner")[1],
            iod * 0.21,
            iod * 0.17,
        ),
        "AU12_corner_subL": _rect_mask(
            image_h,
            image_w,
            px("right_lip_corner")[0],
            px("right_lip_corner")[1],
            iod * 0.21,
            iod * 0.17,
        ),
        "AU23_lip_center": _rect_mask(image_h, image_w, lip_ctr[0], lip_ctr[1], lip_hw, iod * 0.14),
        "AU17_chin": _rect_mask(image_h, image_w, chin_ctr[0], chin_ctr[1], iod * 0.32, iod * 0.16),
        "AU20_corner_subR": _rect_mask(
            image_h,
            image_w,
            px("left_lip_corner")[0],
            px("left_lip_corner")[1],
            iod * 0.21,
            iod * 0.17,
        ),
        "AU20_corner_subL": _rect_mask(
            image_h,
            image_w,
            px("right_lip_corner")[0],
            px("right_lip_corner")[1],
            iod * 0.21,
            iod * 0.17,
        ),
        "AU26_jaw": _rect_mask(
            image_h,
            image_w,
            px("lower_lip_bot")[0],
            px("lower_lip_bot")[1] + iod * 0.18,
            iod * 0.44,
            iod * 0.16,
        ),
    }

    return {
        name: _flow_stats(u, v, mag, mask, mag_threshold) for name, mask in regions.items()
    }, head_motion, iod, transform_info


def parse_gt_jsonl(jsonl_path: Path, datasets: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    clip_info: dict[tuple[str, str], dict[str, Any]] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            dataset = row.get("dataset")
            if dataset not in datasets:
                continue
            image_id = row["image_id"]
            key = (dataset, image_id)
            info = clip_info.setdefault(
                key,
                {
                    "dataset": dataset,
                    "clip_id": image_id,
                    "subject": row.get("subject"),
                    "filename": row.get("filename"),
                    "gt_coarse_emotion": None,
                    "gt_fine_emotion": None,
                    "gt_au_names": [],
                    "gt_detail": None,
                },
            )

            question = (row.get("question") or "").strip().lower()
            answer = (row.get("answer") or "").strip()

            if question == "what is the coarse expression class?":
                info["gt_coarse_emotion"] = answer
            elif question == "what is the fine-grained expression class?":
                info["gt_fine_emotion"] = answer
            elif question in {"what is the action unit?", "what are the action units?"}:
                parts = [part.strip() for part in answer.split(",") if part.strip()]
                for part in parts:
                    if part not in info["gt_au_names"]:
                        info["gt_au_names"].append(part)
            elif "analyse the micro-expression" in question or "comprehensive analysis" in question:
                info["gt_detail"] = answer

    return clip_info


def build_clip_paths(mame_dir: Path, dataset: str, clip_id: str) -> dict[str, Path]:
    prefix = f"{dataset}_{clip_id}"
    base_dir = mame_dir / "trainset_flow" / dataset / prefix / "tv_l1_direct"
    frames_dir = base_dir / "frames_used"
    onset = _resolve_existing_image(frames_dir / "onset.jpg")
    apex = _resolve_existing_image(frames_dir / "apex.jpg")
    flow = base_dir / "flow_npy" / "onset_to_apex.npy"
    return {"base_dir": base_dir, "onset_path": onset, "apex_path": apex, "flow_npy_path": flow}


def compact_motion_stat(stat: dict[str, Any] | None) -> dict[str, Any] | None:
    if stat is None:
        return None
    return {
        "mag": stat["mag"],
        "angle": stat["angle"],
        "dir": stat["dir"],
        "active": stat["active"],
    }


def compact_residual_stat(stat: dict[str, Any] | None) -> dict[str, Any] | None:
    if stat is None:
        return None
    return {
        "dx": stat["dx"],
        "dy": stat["dy"],
        "mag_px": stat["mag_px"],
        "mag": stat["mag"],
        "angle": stat["angle"],
        "dir": stat["dir"],
        "active": stat["active"],
    }


def serializable_transform_info(transform_info: dict[str, Any] | None) -> dict[str, Any] | None:
    if transform_info is None:
        return None
    return {
        "method": transform_info["method"],
        "rotation_deg": transform_info["rotation_deg"],
        "scale_x": transform_info["scale_x"],
        "scale_y": transform_info["scale_y"],
        "determinant": transform_info["determinant"],
        "landmark_fit_rmse": transform_info["landmark_fit_rmse"],
        "landmark_fit_max_err": transform_info["landmark_fit_max_err"],
        "matrix": np.asarray(transform_info["matrix"], dtype=float).round(6).tolist(),
    }


def _to_builtin_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_builtin_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        "dataset": record["dataset"],
        "clip_id": record["clip_id"],
        "subject": record.get("subject"),
        "filename": record.get("filename"),
        "gt_coarse_emotion": record.get("gt_coarse_emotion"),
        "gt_fine_emotion": record.get("gt_fine_emotion"),
        "gt_au_names": ",".join(record.get("gt_au_names") or []),
        "eye_height_ratio": record.get("eye_height_ratio"),
        "iod": record.get("iod"),
        "head_motion_mag": record.get("head_motion", {}).get("head_mag"),
        "head_motion_angle": record.get("head_motion", {}).get("angle"),
        "head_motion_dir": record.get("head_motion", {}).get("dir"),
        "head_motion_quality": record.get("head_motion", {}).get("quality"),
        "onset_path": record.get("onset_path"),
        "apex_path": record.get("apex_path"),
        "flow_npy_path": record.get("flow_npy_path"),
        "status": record.get("status"),
        "error": record.get("error"),
    }
    eye_metrics = record.get("eye_metrics") or {}
    mouth_metrics = record.get("mouth_metrics") or {}
    for key in (
        "left_ratio",
        "right_ratio",
        "mean_ratio",
        "min_ratio",
        "max_ratio",
        "asymmetry",
        "apex_min_height_norm",
        "apex_mean_height_norm",
        "left_gap_ratio",
        "right_gap_ratio",
        "mean_gap_ratio",
        "left_gap_shrink",
        "right_gap_shrink",
        "mean_gap_shrink",
        "min_gap_ratio",
        "gap_balance",
        "left_inner_brow_raise_norm",
        "right_inner_brow_raise_norm",
        "inner_brow_raise_mean",
        "left_inner_brow_drop_norm",
        "right_inner_brow_drop_norm",
        "inner_brow_drop_mean",
        "left_inner_brow_relative_raise_norm",
        "right_inner_brow_relative_raise_norm",
        "inner_brow_relative_raise_mean",
        "left_inner_brow_relative_drop_norm",
        "right_inner_brow_relative_drop_norm",
        "inner_brow_relative_drop_mean",
        "inner_brow_raise_balance",
        "left_inner_brow_forward_norm",
        "right_inner_brow_forward_norm",
        "inner_brow_forward_mean",
        "left_inner_brow_forward_relative_norm",
        "right_inner_brow_forward_relative_norm",
        "inner_brow_forward_relative_mean",
        "left_outer_brow_raise_norm",
        "right_outer_brow_raise_norm",
        "outer_brow_raise_mean",
        "left_outer_brow_drop_norm",
        "right_outer_brow_drop_norm",
        "outer_brow_drop_mean",
        "left_outer_brow_relative_raise_norm",
        "right_outer_brow_relative_raise_norm",
        "outer_brow_relative_raise_mean",
        "left_outer_brow_relative_drop_norm",
        "right_outer_brow_relative_drop_norm",
        "outer_brow_relative_drop_mean",
        "left_outer_brow_forward_norm",
        "right_outer_brow_forward_norm",
        "outer_brow_forward_mean",
        "left_outer_brow_forward_relative_norm",
        "right_outer_brow_forward_relative_norm",
        "outer_brow_forward_relative_mean",
        "outer_brow_raise_balance",
        "brow_center_raise_norm",
        "brow_center_drop_norm",
    ):
        row[f"eye_metrics_{key}"] = eye_metrics.get(key)
    for key in (
        "mouth_open_ratio",
        "mouth_width_ratio",
        "upper_lip_nose_ratio",
        "lower_lip_nose_ratio",
        "nostril_width_ratio",
        "left_corner_lift",
        "right_corner_lift",
        "left_corner_depress",
        "right_corner_depress",
        "left_corner_raise",
        "right_corner_raise",
        "left_corner_drop",
        "right_corner_drop",
        "left_corner_outward",
        "right_corner_outward",
        "left_corner_inward",
        "right_corner_inward",
        "left_corner_raise_relative",
        "right_corner_raise_relative",
        "left_nostril_raise_norm",
        "right_nostril_raise_norm",
        "corner_lift_mean",
        "corner_depress_mean",
        "corner_raise_mean",
        "corner_drop_mean",
        "corner_outward_mean",
        "corner_inward_mean",
        "corner_raise_relative_mean",
        "corner_lift_balance",
        "corner_depress_balance",
        "corner_raise_balance",
        "corner_raise_relative_balance",
        "corner_drop_balance",
        "corner_outward_balance",
        "corner_inward_balance",
        "corner_asymmetry",
        "lower_lip_drop_norm",
        "mouth_center_raise_norm",
        "mouth_center_drop_norm",
        "lower_lip_raise_norm",
        "upper_lip_raise_norm",
        "upper_lip_raise_relative_norm",
        "chin_raise_norm",
        "nostril_raise_mean",
        "nostril_raise_balance",
        "lower_lip_forward_norm",
        "upper_lip_forward_norm",
        "chin_forward_norm",
        "nose_bridge_shrink_norm",
        "left_corner_forward_norm",
        "right_corner_forward_norm",
        "corner_forward_mean",
        "left_corner_backward_norm",
        "right_corner_backward_norm",
        "corner_backward_mean",
        "corner_backward_balance",
        "mouth_center_forward_norm",
        "mouth_center_backward_norm",
    ):
        row[f"mouth_metrics_{key}"] = mouth_metrics.get(key)

    raw_regions = record.get("au_regions_raw") or {}
    compensated_regions = record.get("au_regions_head_motion_removed_translation") or {}
    residual_regions = record.get(f"au_landmark_residual_{LANDMARK_RESIDUAL_METHOD}") or {}
    for au_name in AU_REGION_NAMES:
        raw = raw_regions.get(au_name)
        comp = compensated_regions.get(au_name)
        residual = residual_regions.get(au_name)
        row[f"{au_name}_raw_mag"] = None if raw is None else raw["mag"]
        row[f"{au_name}_raw_angle"] = None if raw is None else raw["angle"]
        row[f"{au_name}_raw_dir"] = None if raw is None else raw["dir"]
        row[f"{au_name}_raw_active"] = None if raw is None else raw["active"]
        row[f"{au_name}_hm_removed_translation_mag"] = None if comp is None else comp["mag"]
        row[f"{au_name}_hm_removed_translation_angle"] = None if comp is None else comp["angle"]
        row[f"{au_name}_hm_removed_translation_dir"] = None if comp is None else comp["dir"]
        row[f"{au_name}_hm_removed_translation_active"] = None if comp is None else comp["active"]

        for method in ("similarity", "affine"):
            comp_method = (record.get(f"au_regions_head_motion_removed_{method}") or {}).get(au_name)
            row[f"{au_name}_hm_removed_{method}_mag"] = None if comp_method is None else comp_method["mag"]
            row[f"{au_name}_hm_removed_{method}_angle"] = None if comp_method is None else comp_method["angle"]
            row[f"{au_name}_hm_removed_{method}_dir"] = None if comp_method is None else comp_method["dir"]
            row[f"{au_name}_hm_removed_{method}_active"] = None if comp_method is None else comp_method["active"]

        row[f"{au_name}_landmark_residual_{LANDMARK_RESIDUAL_METHOD}_dx"] = None if residual is None else residual["dx"]
        row[f"{au_name}_landmark_residual_{LANDMARK_RESIDUAL_METHOD}_dy"] = None if residual is None else residual["dy"]
        row[f"{au_name}_landmark_residual_{LANDMARK_RESIDUAL_METHOD}_mag_px"] = None if residual is None else residual["mag_px"]
        row[f"{au_name}_landmark_residual_{LANDMARK_RESIDUAL_METHOD}_mag"] = None if residual is None else residual["mag"]
        row[f"{au_name}_landmark_residual_{LANDMARK_RESIDUAL_METHOD}_angle"] = None if residual is None else residual["angle"]
        row[f"{au_name}_landmark_residual_{LANDMARK_RESIDUAL_METHOD}_dir"] = None if residual is None else residual["dir"]
        row[f"{au_name}_landmark_residual_{LANDMARK_RESIDUAL_METHOD}_active"] = None if residual is None else residual["active"]

    for method in ("translation", "similarity", "affine"):
        info = record.get(f"head_motion_transform_{method}") or {}
        row[f"hm_{method}_rotation_deg"] = info.get("rotation_deg")
        row[f"hm_{method}_rotation3d_deg"] = info.get("rotation3d_deg")
        row[f"hm_{method}_scale_x"] = info.get("scale_x")
        row[f"hm_{method}_scale_y"] = info.get("scale_y")
        row[f"hm_{method}_scale3d"] = info.get("scale3d")
        row[f"hm_{method}_determinant"] = info.get("determinant")
        row[f"hm_{method}_landmark_fit_rmse"] = info.get("landmark_fit_rmse")
        row[f"hm_{method}_landmark_fit_rmse_3d"] = info.get("landmark_fit_rmse_3d")
        row[f"hm_{method}_landmark_fit_max_err"] = info.get("landmark_fit_max_err")
    return row


def process_clip(
    mame_dir: Path,
    model_path: Path,
    clip_meta: dict[str, Any],
) -> dict[str, Any]:
    dataset = clip_meta["dataset"]
    clip_id = clip_meta["clip_id"]
    paths = build_clip_paths(mame_dir, dataset, clip_id)

    record = {
        **clip_meta,
        "onset_path": str(paths["onset_path"]),
        "apex_path": str(paths["apex_path"]),
        "flow_npy_path": str(paths["flow_npy_path"]),
        "status": "ok",
        "error": None,
    }

    if not paths["flow_npy_path"].exists():
        record["status"] = "missing_flow"
        record["error"] = f"Missing flow file: {paths['flow_npy_path']}"
        return record
    if not paths["onset_path"].exists() or not paths["apex_path"].exists():
        record["status"] = "missing_frames"
        record["error"] = "Missing onset or apex frame."
        return record

    onset_landmarks, _, _ = detect_face_landmarks(paths["onset_path"], model_path)
    apex_landmarks, _, _ = detect_face_landmarks(paths["apex_path"], model_path)
    if onset_landmarks is None or apex_landmarks is None:
        record["status"] = "landmark_fail"
        record["error"] = "Landmark detection failed on onset or apex frame."
        return record

    raw_regions, head_motion, iod, _ = extract_landmark_au_regions(
        paths["onset_path"],
        paths["flow_npy_path"],
        model_path,
        onset_landmarks=onset_landmarks,
        apex_landmarks=apex_landmarks,
        head_motion_method=None,
    )
    eye_ratio = compute_eye_height_ratio(paths["onset_path"], paths["apex_path"], model_path)

    if raw_regions is None:
        record["status"] = "landmark_fail"
        record["error"] = "Landmark detection failed on onset frame."
        return record

    hm_removed_by_method: dict[str, dict[str, Any] | None] = {}
    transform_info_by_method: dict[str, dict[str, Any] | None] = {}
    transform_info_runtime_by_method: dict[str, dict[str, Any] | None] = {}
    for method in ("translation", "similarity", "affine"):
        regions, _, _, transform_info = extract_landmark_au_regions(
            paths["onset_path"],
            paths["flow_npy_path"],
            model_path,
            onset_landmarks=onset_landmarks,
            apex_landmarks=apex_landmarks,
            head_motion_method=method,
        )
        hm_removed_by_method[method] = {
            name: compact_motion_stat((regions or {}).get(name)) for name in AU_REGION_NAMES
        }
        transform_info_runtime_by_method[method] = transform_info
        transform_info_by_method[method] = serializable_transform_info(transform_info)

    image = np.array(Image.open(_resolve_existing_image(paths["onset_path"])).convert("RGB"))
    image_h, image_w = image.shape[:2]
    residual_method = LANDMARK_RESIDUAL_METHOD
    use_geometry = dataset in GEOMETRY_DATASETS
    geometry_transform = transform_info_runtime_by_method[residual_method]
    eye_metrics = (
        compute_eye_geometry_metrics(
            onset_landmarks,
            apex_landmarks,
            dataset=dataset,
            width=image_w,
            height=image_h,
            iod=iod,
            transform_info=geometry_transform,
        )
        if use_geometry
        else {}
    )
    if eye_metrics.get("mean_ratio") is not None:
        eye_ratio = eye_metrics["mean_ratio"]
    mouth_metrics = (
        compute_mouth_geometry_metrics(
            onset_landmarks,
            apex_landmarks,
            width=image_w,
            height=image_h,
            transform_info=geometry_transform,
        )
        if use_geometry
        else {}
    )
    landmark_residual_features = compute_landmark_residual_au_features(
        onset_landmarks,
        apex_landmarks,
        dataset,
        image_w,
        image_h,
        iod,
        transform_info_runtime_by_method[residual_method],
    )

    record["iod"] = round(iod, 3)
    record["head_motion"] = head_motion
    record["eye_height_ratio"] = eye_ratio
    record["eye_metrics"] = eye_metrics
    record["mouth_metrics"] = mouth_metrics
    record["au_regions_raw"] = {name: compact_motion_stat(raw_regions.get(name)) for name in AU_REGION_NAMES}
    for method in ("translation", "similarity", "affine"):
        record[f"au_regions_head_motion_removed_{method}"] = hm_removed_by_method[method]
        record[f"head_motion_transform_{method}"] = transform_info_by_method[method]
    record[f"au_landmark_residual_{residual_method}"] = {
        name: compact_residual_stat(landmark_residual_features.get(name)) for name in AU_REGION_NAMES
    }
    return record


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_to_builtin_jsonable(record), ensure_ascii=False) + "\n")


def write_csv(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_to_builtin_jsonable(flatten_record(record)) for record in records]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mame-dir",
        type=Path,
        default=REPO_ROOT,
        help="Data root that contains trainset_flow/ and related assets.",
    )
    parser.add_argument(
        "--jsonl-path",
        type=Path,
        default=REPO_ROOT / "me_vqa_samm_casme2_smic_v2.jsonl",
        help="Merged GT JSONL path.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["casme2", "samm"],
        choices=["casme2", "samm"],
        help="Datasets to export.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "face_landmarker.task",
        help="MediaPipe face landmarker model path.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=REPO_ROOT / "outputs" / "casme2_samm_motion_features.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "outputs" / "casme2_samm_motion_features.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of clips to process after filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = set(args.datasets)
    clip_info = parse_gt_jsonl(args.jsonl_path, datasets)

    clips = sorted(clip_info.values(), key=lambda item: (item["dataset"], item["clip_id"]))
    if args.limit is not None:
        clips = clips[: args.limit]

    print(f"Loaded {len(clips)} clips from {sorted(datasets)}")
    records: list[dict[str, Any]] = []
    total = len(clips)
    for idx, clip_meta in enumerate(clips, start=1):
        record = process_clip(args.mame_dir, args.model_path, clip_meta)
        records.append(record)
        print(
            f"[{idx:03d}/{total:03d}] {record['dataset']}:{record['clip_id']} "
            f"status={record['status']}"
        )

    write_jsonl(records, args.output_jsonl)
    write_csv(records, args.output_csv)

    ok_count = sum(1 for record in records if record.get("status") == "ok")
    print(f"Wrote {len(records)} records ({ok_count} ok) to:")
    print(f"  JSONL: {args.output_jsonl}")
    print(f"  CSV:   {args.output_csv}")


if __name__ == "__main__":
    main()
