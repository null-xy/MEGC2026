from __future__ import annotations

import math
from pathlib import Path
from typing import Any

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False
import numpy as np
from PIL import Image

try:
    from .calibration import compute_geometry_profile_support, load_motion_feature_calibration
except ImportError:
    from calibration import compute_geometry_profile_support, load_motion_feature_calibration

try:
    from .config import get_active_ablation
except ImportError:
    from config import get_active_ablation

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core import base_options as bo

    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    mp_vision = None
    bo = None
    _MEDIAPIPE_AVAILABLE = False


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

_lm_detector = None
RIGID_LANDMARKS = [
    "left_eye_outer",
    "right_eye_outer",
    "nose_tip",
    "nose_top",
    "left_nostril",
    "right_nostril",
]
LANDMARK_RESIDUAL_METHOD = "similarity"
UPWARD_RANGES = [(40.0, 130.0)]
DOWNWARD_RANGES = [(215.0, 325.0)]
HORIZONTAL_RANGES = [(0.0, 30.0), (150.0, 210.0), (330.0, 360.0)]

AU_REGION_MAP = {
    "inner brow raiser": ("AU1_inner_brow_subR", "AU1_inner_brow_subL"),
    "outer brow raiser": ("AU2_brow_outer_subR", "AU2_brow_outer_subL"),
    "brow lowerer": ("AU4_brow_lowerer",),
    "nose wrinkler": ("AU9_nose_bridge",),
    "upper lip raiser": ("AU10_upper_lip",),
    "lip corner puller": ("AU12_corner_subR", "AU12_corner_subL"),
    "lip corner depressor": ("AU20_corner_subR", "AU20_corner_subL"),
    "lip stretcher": ("AU20_corner_subR", "AU20_corner_subL"),
    "chin raiser": ("AU17_chin",),
    "lip tightener": ("AU23_lip_center",),
    "jaw drop": ("AU26_jaw",),
}

AU_DIRECTION_RANGES = {
    "inner brow raiser": UPWARD_RANGES,
    "outer brow raiser": UPWARD_RANGES,
    "brow lowerer": DOWNWARD_RANGES,
    "nose wrinkler": UPWARD_RANGES,
    "upper lip raiser": UPWARD_RANGES,
    "lip corner puller": UPWARD_RANGES,
    "lip corner depressor": DOWNWARD_RANGES,
    "lip stretcher": HORIZONTAL_RANGES,
    "chin raiser": UPWARD_RANGES,
    "lip tightener": HORIZONTAL_RANGES,
}

PAIR_AUS = {
    "inner brow raiser",
    "outer brow raiser",
    "lip corner puller",
    "lip corner depressor",
    "lip stretcher",
}
DATA_DRIVEN_GEOMETRY_AUS = {
    "chin raiser",
    "dimpler",
    "lip corner puller",
    "lip stretcher",
}
DATA_DRIVEN_GEOMETRY_DATASETS = {"samm"}
SAMM_BROW_AGENT_AUS = {
    "inner brow raiser",
    "outer brow raiser",
    "brow lowerer",
}
SAMM_GEOMETRY_ONLY_AUS = {
    "dimpler",
    "eye closure",
    "lip corner depressor",
    "lip stretcher",
    "lips part",
    "jaw drop",
}
CASME2_GEOMETRY_AUS = {
    "upper lid raiser",
    "lid tightener",
    "upper lip raiser",
}
EVIDENCE_AUS = [
    "inner brow raiser",
    "outer brow raiser",
    "brow lowerer",
    "upper lid raiser",
    "lid tightener",
    "eye closure",
    "nose wrinkler",
    "upper lip raiser",
    "lip corner puller",
    "dimpler",
    "lip corner depressor",
    "lip stretcher",
    "lips part",
    "jaw drop",
    "chin raiser",
    "lip tightener",
]


def _calibration_ready(threshold_info: dict[str, Any]) -> bool:
    threshold = float(threshold_info.get("threshold", 0.0) or 0.0)
    precision = float(threshold_info.get("precision", 0.0) or 0.0)
    recall = float(threshold_info.get("recall", 0.0) or 0.0)
    f1 = float(threshold_info.get("f1", 0.0) or 0.0)
    return threshold > 0.0 and max(precision, recall, f1) > 0.0


def _uses_geometry_metrics(dataset: str) -> bool:
    if get_active_ablation().disable_geometry_calibration:
        return False
    return dataset in {"samm", "casme2"}


def _geometry_enabled_for_au(dataset: str, au_name: str) -> bool:
    if get_active_ablation().disable_geometry_calibration:
        return False
    if dataset == "samm":
        return True
    if dataset == "casme2":
        return au_name in CASME2_GEOMETRY_AUS
    return False


def numeric_features_available() -> bool:
    return _MEDIAPIPE_AVAILABLE and _resolve_model_asset() is not None


def _resolve_model_asset() -> Path | None:
    candidates = [
        Path(__file__).resolve().parents[2] / "face_landmarker.task",
        Path.cwd() / "face_landmarker.task",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def ensure_numeric_feature_dependencies() -> Path:
    if not _MEDIAPIPE_AVAILABLE:
        raise RuntimeError(
            "MediaPipe is required for numeric features. Install the `mediapipe` dependency before running the batch runner."
        )
    model_asset = _resolve_model_asset()
    if model_asset is None:
        raise RuntimeError(
            "face_landmarker.task is required for numeric features but was not found next to the repo or in the current working directory."
        )
    return model_asset


def _circular_distance(angle_a: float, angle_b: float) -> float:
    delta = abs(float(angle_a) - float(angle_b)) % 360.0
    return min(delta, 360.0 - delta)


def _angle_distance_to_ranges(angle: float, ranges: list[tuple[float, float]]) -> float:
    if not ranges:
        return 0.0
    best = 180.0
    for start, end in ranges:
        if start <= angle <= end:
            return 0.0
        best = min(best, _circular_distance(angle, start), _circular_distance(angle, end))
    return best


def _direction_compatibility(info: dict[str, Any], ranges: list[tuple[float, float]]) -> float:
    if not info:
        return 0.0
    active = float(info.get("active", 0.0) or 0.0)
    mag = float(info.get("mag", 0.0) or 0.0)
    if active <= 0.0 or mag <= 0.0:
        return 0.0
    angle = float(info.get("angle", 0.0) or 0.0) % 360.0
    dist = _angle_distance_to_ranges(angle, ranges)
    if dist <= 0.0:
        return 1.0
    return max(0.0, 1.0 - dist / 90.0)


def _scaled_region_support(info: dict[str, Any], *, residual: bool) -> float:
    if not info:
        return 0.0
    scale = 18.0 if residual else 1.0
    return (
        scale
        * float(info.get("active", 0.0) or 0.0)
        * float(info.get("mag", 0.0) or 0.0)
    )


def _head_motion_penalty(head_motion: dict[str, Any], residual_support: float) -> float:
    quality = str((head_motion or {}).get("quality", "unknown") or "unknown").lower()
    if quality == "dominated":
        return 0.55 if residual_support < 0.10 else 0.75
    if quality == "moderate":
        return 0.75 if residual_support < 0.05 else 0.9
    return 1.0


def _region_support_summary(
    region: str,
    *,
    ranges: list[tuple[float, float]],
    raw_regions: dict[str, Any],
    residual_regions: dict[str, Any],
    head_motion: dict[str, Any],
) -> dict[str, float]:
    raw_info = raw_regions.get(region) or {}
    residual_info = residual_regions.get(region) or {}

    raw_base = _scaled_region_support(raw_info, residual=False)
    residual_base = _scaled_region_support(residual_info, residual=True)
    raw_dir = _direction_compatibility(raw_info, ranges)
    residual_dir = _direction_compatibility(residual_info, ranges)

    raw_support = raw_base * (0.45 + 0.55 * raw_dir)
    residual_support = residual_base * (0.55 + 0.45 * residual_dir)
    raw_penalty = _head_motion_penalty(head_motion, residual_support)
    combined = max(
        residual_support * 1.20,
        residual_support + 0.35 * raw_support * raw_penalty,
        raw_support * 0.60 * raw_penalty,
    )
    return {
        "raw_support": round(raw_support, 4),
        "residual_support": round(residual_support, 4),
        "combined_support": round(combined, 4),
        "direction": round(max(raw_dir, residual_dir), 4),
    }


def _balance_score(values: list[float]) -> float:
    nonzero = [float(value) for value in values if float(value) > 0.0]
    if len(nonzero) <= 1:
        return 1.0 if nonzero else 0.0
    high = max(nonzero)
    low = min(nonzero)
    if high <= 1e-6:
        return 0.0
    return round(max(0.0, low / high), 4)


def _ratio(current: float, reference: float, *, neutral: float = 1.0, min_reference: float = 1.0) -> float:
    if reference <= min_reference:
        return neutral
    return float(current) / float(reference)


def _named_point_map(lm, W: int, H: int, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: landmark_xy(lm, LM_IDX[name], W, H) for name in names}


def _named_point_map_xyz(lm, W: int, H: int, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: landmark_xyz(lm, LM_IDX[name], W, H) for name in names}


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
    matrix3d = np.concatenate(
        [scale * rotation, translation.reshape(3, 1)],
        axis=1,
    ).astype(np.float32)
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
        transformed[name] = _transform_points(np.array([point], dtype=np.float32), matrix)[0]
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
    onset_lm,
    apex_lm,
    *,
    dataset: str = "",
    W: int,
    H: int,
    iod: float,
    transform_info: dict[str, Any] | None,
) -> dict[str, float]:
    if onset_lm is None or apex_lm is None:
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
    ref_points = _transform_point_map(_named_point_map(onset_lm, W, H, names), transform_info)
    ref_points_xyz = _transform_point_map_xyz(_named_point_map_xyz(onset_lm, W, H, names), transform_info)
    brow_ref_points = (
        {name: point[:2].astype(np.float32) for name, point in ref_points_xyz.items()}
        if dataset == "samm"
        else ref_points
    )
    apex_points = _named_point_map(apex_lm, W, H, names)
    apex_points_xyz = _named_point_map_xyz(apex_lm, W, H, names)

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
    onset_lm,
    apex_lm,
    *,
    W: int,
    H: int,
    transform_info: dict[str, Any] | None,
) -> dict[str, float]:
    if onset_lm is None or apex_lm is None:
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
    ref_points = _transform_point_map(_named_point_map(onset_lm, W, H, names), transform_info)
    ref_points_xyz = _transform_point_map_xyz(_named_point_map_xyz(onset_lm, W, H, names), transform_info)
    apex_points = _named_point_map(apex_lm, W, H, names)
    apex_points_xyz = _named_point_map_xyz(apex_lm, W, H, names)

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
        float(apex_points["lower_lip_bot"][1] - apex_nose[1]) - float(ref_points["lower_lip_bot"][1] - ref_nose[1]),
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


def _geometry_au_support(
    au_name: str,
    *,
    dataset: str,
    eye_metrics: dict[str, Any],
    mouth_metrics: dict[str, Any],
    calibration_profiles: dict[str, Any] | None = None,
) -> tuple[float, float, float]:
    eye_metrics = eye_metrics or {}
    mouth_metrics = mouth_metrics or {}
    profile = (calibration_profiles or {}).get(au_name) or None

    if dataset == "samm" and au_name == "inner brow raiser":
        gap_ratio = float(eye_metrics.get("mean_gap_ratio", 1.0) or 1.0)
        inner_raise = float(eye_metrics.get("inner_brow_raise_mean", 0.0) or 0.0)
        relative_raise = float(eye_metrics.get("inner_brow_relative_raise_mean", 0.0) or 0.0)
        forward_relative = float(eye_metrics.get("inner_brow_forward_relative_mean", 0.0) or 0.0)
        brow_drop = float(eye_metrics.get("brow_center_drop_norm", 0.0) or 0.0)
        pose_rmse = float(eye_metrics.get("pose_landmark_fit_rmse_3d", 0.0) or 0.0)
        support = max(0.0, inner_raise - 0.0040) * 18.0
        support += max(0.0, relative_raise - 0.0022) * 170.0
        support += max(0.0, forward_relative - 0.0008) * 52.0
        if relative_raise < 0.0012 and forward_relative < 0.0007:
            support *= 0.28
        support = max(0.0, support - max(0.0, gap_ratio - 1.03) * 16.0)
        support = max(0.0, support - brow_drop * 42.0)
        support = max(0.0, support - max(0.0, pose_rmse - 2.2) * 0.08)
        balance = float(eye_metrics.get("inner_brow_raise_balance", 0.0) or 0.0)
        return support, 1.0 if support > 0.0 else 0.0, balance

    if dataset == "samm" and au_name == "outer brow raiser":
        outer_raise = float(eye_metrics.get("outer_brow_raise_mean", 0.0) or 0.0)
        relative_raise = float(eye_metrics.get("outer_brow_relative_raise_mean", 0.0) or 0.0)
        forward_relative = float(eye_metrics.get("outer_brow_forward_relative_mean", 0.0) or 0.0)
        brow_drop = float(eye_metrics.get("brow_center_drop_norm", 0.0) or 0.0)
        inner_raise = float(eye_metrics.get("inner_brow_relative_raise_mean", 0.0) or 0.0)
        pose_rmse = float(eye_metrics.get("pose_landmark_fit_rmse_3d", 0.0) or 0.0)
        support = max(0.0, outer_raise - 0.0040) * 18.0
        support += max(0.0, relative_raise - 0.0022) * 160.0
        support += max(0.0, forward_relative - 0.0008) * 46.0
        support = max(0.0, support - max(0.0, inner_raise - relative_raise) * 42.0)
        support = max(0.0, support - brow_drop * 36.0)
        if relative_raise < 0.0011 and forward_relative < 0.0007:
            support *= 0.30
        support = max(0.0, support - max(0.0, pose_rmse - 2.2) * 0.08)
        balance = float(eye_metrics.get("outer_brow_raise_balance", 0.0) or 0.0)
        return support, 1.0 if support > 0.0 else 0.0, balance

    if dataset in {"samm", "casme2"} and au_name == "brow lowerer":
        gap_shrink = float(eye_metrics.get("mean_gap_shrink", 0.0) or 0.0)
        inner_drop = float(eye_metrics.get("inner_brow_relative_drop_mean", 0.0) or 0.0)
        outer_drop = float(eye_metrics.get("outer_brow_relative_drop_mean", 0.0) or 0.0)
        brow_drop = float(eye_metrics.get("brow_center_drop_norm", 0.0) or 0.0)
        mean_ratio = float(eye_metrics.get("mean_ratio", 1.0) or 1.0)
        pose_rmse = float(eye_metrics.get("pose_landmark_fit_rmse_3d", 0.0) or 0.0)
        support = max(0.0, gap_shrink - 0.008) * 28.0
        support += max(0.0, inner_drop - 0.0022) * 150.0
        support += max(0.0, outer_drop - 0.0018) * 92.0
        support += max(0.0, brow_drop - 0.0025) * 120.0
        support = max(0.0, support - max(0.0, mean_ratio - 1.02) * 18.0)
        support = max(0.0, support - max(0.0, pose_rmse - 2.6) * 0.05)
        balance = min(
            max(float(eye_metrics.get("inner_brow_raise_balance", 0.0) or 0.0), 0.0),
            max(float(eye_metrics.get("outer_brow_raise_balance", 0.0) or 0.0), 0.0) or 1.0,
        )
        return support, 1.0 if support > 0.0 else 0.0, max(balance, 0.55)

    if au_name == "upper lid raiser":
        mean_ratio = float(eye_metrics.get("mean_ratio", 1.0) or 1.0)
        max_ratio = float(eye_metrics.get("max_ratio", mean_ratio) or mean_ratio)
        asymmetry = float(eye_metrics.get("asymmetry", 0.0) or 0.0)
        if dataset == "samm":
            gap_shrink = float(eye_metrics.get("mean_gap_shrink", 0.0) or 0.0)
            pose_rmse = float(eye_metrics.get("pose_landmark_fit_rmse_3d", 0.0) or 0.0)
            if mean_ratio > 1.18 or max_ratio > 1.22 or asymmetry > 0.35 or pose_rmse > 3.0:
                return 0.0, 0.0, max(0.0, 1.0 - min(asymmetry, 1.0))
            support = max(0.0, mean_ratio - 1.023) * 13.0
            support += max(0.0, max_ratio - 1.030) * 9.0
            support += max(0.0, min(max_ratio, 1.20) - max(mean_ratio, 1.0) - 0.004) * 4.0
            support = max(0.0, support - max(0.0, gap_shrink - 0.035) * 3.5)
            support = max(0.0, support - max(0.0, gap_shrink - 0.065) * 9.0)
            support = max(0.0, support - max(0.0, pose_rmse - 2.4) * 0.08)
        else:
            support = max(0.0, mean_ratio - 1.03) * 8.0 + max(0.0, max_ratio - 1.05) * 4.0
        balance = max(0.0, 1.0 - asymmetry)
        return support, 1.0 if support > 0.0 else 0.0, balance

    if au_name == "lid tightener":
        mean_ratio = float(eye_metrics.get("mean_ratio", 1.0) or 1.0)
        min_ratio = float(eye_metrics.get("min_ratio", mean_ratio) or mean_ratio)
        apex_norm = float(eye_metrics.get("apex_min_height_norm", 1.0) or 1.0)
        narrowing = max(0.0, 0.98 - mean_ratio) * 8.5
        closure_penalty = max(0.0, 0.86 - min_ratio) * 9.0 + max(0.0, 0.032 - apex_norm) * 32.0
        support = max(0.0, narrowing - 0.45 * closure_penalty)
        balance = max(0.0, 1.0 - float(eye_metrics.get("asymmetry", 0.0) or 0.0))
        return support, 1.0 if support > 0.0 else 0.0, balance

    if au_name == "eye closure":
        min_ratio = float(eye_metrics.get("min_ratio", 1.0) or 1.0)
        apex_norm = float(eye_metrics.get("apex_min_height_norm", 1.0) or 1.0)
        support = max(0.0, 0.90 - min_ratio) * 12.0 + max(0.0, 0.034 - apex_norm) * 42.0
        balance = max(0.0, 1.0 - float(eye_metrics.get("asymmetry", 0.0) or 0.0))
        return support, 1.0 if support > 0.0 else 0.0, balance

    if dataset == "samm" and au_name == "nose wrinkler":
        nostril_raise = float(mouth_metrics.get("nostril_raise_mean", 0.0) or 0.0)
        bridge_shrink = float(mouth_metrics.get("nose_bridge_shrink_norm", 0.0) or 0.0)
        upper_lip_raise = float(mouth_metrics.get("upper_lip_raise_relative_norm", 0.0) or 0.0)
        upper_lip_raise_abs = float(mouth_metrics.get("upper_lip_raise_norm", 0.0) or 0.0)
        upper_lip_forward = float(mouth_metrics.get("upper_lip_forward_norm", 0.0) or 0.0)
        center_raise = float(mouth_metrics.get("mouth_center_raise_norm", 0.0) or 0.0)
        corner_raise = float(mouth_metrics.get("corner_raise_relative_mean", 0.0) or 0.0)
        support = max(0.0, nostril_raise - 0.0010) * 54.0
        support += max(0.0, bridge_shrink - 0.0010) * 56.0
        support += max(0.0, upper_lip_raise - 0.0012) * 20.0
        support += max(0.0, upper_lip_raise_abs - 0.0018) * 10.0
        support += max(0.0, upper_lip_forward - 0.0008) * 8.0
        support = max(0.0, support - max(0.0, center_raise - 0.0020) * 22.0)
        support = max(0.0, support - max(0.0, corner_raise - 0.0020) * 14.0)
        balance = min(
            max(float(mouth_metrics.get("nostril_raise_balance", 0.0) or 0.0), 0.0),
            1.0,
        )
        return support, 1.0 if support > 0.0 else 0.0, max(balance, 0.55)

    if dataset == "samm" and au_name == "upper lip raiser":
        ratio = float(mouth_metrics.get("upper_lip_nose_ratio", 1.0) or 1.0)
        raise_norm = float(mouth_metrics.get("upper_lip_raise_relative_norm", 0.0) or 0.0)
        forward = float(mouth_metrics.get("upper_lip_forward_norm", 0.0) or 0.0)
        bridge_shrink = float(mouth_metrics.get("nose_bridge_shrink_norm", 0.0) or 0.0)
        nostril_raise = float(mouth_metrics.get("nostril_raise_mean", 0.0) or 0.0)
        corner_raise = float(mouth_metrics.get("corner_raise_relative_mean", 0.0) or 0.0)
        corner_lift = float(mouth_metrics.get("corner_lift_mean", 0.0) or 0.0)
        support = max(0.0, raise_norm - 0.0014) * 58.0
        support += max(0.0, 0.998 - ratio) * 18.0
        support += max(0.0, forward - 0.0008) * 12.0
        support += max(0.0, bridge_shrink - 0.0012) * 8.0
        support = max(0.0, support - max(0.0, nostril_raise - 0.0014) * 10.0)
        support = max(0.0, support - max(0.0, corner_raise - 0.0025) * 14.0)
        support = max(0.0, support - max(0.0, corner_lift - 0.0040) * 10.0)
        return support, 1.0 if support > 0.0 else 0.0, 1.0

    if au_name == "upper lip raiser":
        ratio = float(mouth_metrics.get("upper_lip_nose_ratio", 1.0) or 1.0)
        support = max(0.0, 0.98 - ratio) * 12.0
        return support, 1.0 if support > 0.0 else 0.0, 1.0

    if dataset in DATA_DRIVEN_GEOMETRY_DATASETS and au_name in DATA_DRIVEN_GEOMETRY_AUS:
        if profile:
            support, balance = compute_geometry_profile_support(
                profile,
                mouth_metrics=mouth_metrics,
                eye_metrics=eye_metrics,
            )
            return support, 1.0 if support > 0.0 else 0.0, max(balance, 0.0)
        return 0.0, 0.0, 1.0

    if au_name == "lip corner puller":
        if dataset == "samm":
            raise_mean = float(mouth_metrics.get("corner_raise_relative_mean", 0.0) or 0.0)
            lift_mean = float(mouth_metrics.get("corner_lift_mean", 0.0) or 0.0)
            outward_mean = float(mouth_metrics.get("corner_outward_mean", 0.0) or 0.0)
            center_raise = float(mouth_metrics.get("mouth_center_raise_norm", 0.0) or 0.0)
            lower_raise = float(mouth_metrics.get("lower_lip_raise_norm", 0.0) or 0.0)
            chin_raise = float(mouth_metrics.get("chin_raise_norm", 0.0) or 0.0)
            center_forward = float(mouth_metrics.get("mouth_center_forward_norm", 0.0) or 0.0)
            corner_forward = float(mouth_metrics.get("corner_forward_mean", 0.0) or 0.0)
            balance = min(
                float(mouth_metrics.get("corner_raise_relative_balance", 0.0) or 0.0),
                max(float(mouth_metrics.get("corner_lift_balance", 0.0) or 0.0), 0.6),
            )
            support = max(0.0, raise_mean - 0.0015) * 62.0
            support += max(0.0, lift_mean - 0.0035) * 18.0
            support += max(0.0, outward_mean - 0.0025) * 10.0
            support += max(0.0, corner_forward - 0.5 * center_forward - 0.0008) * 14.0
            support = max(0.0, support - max(0.0, center_raise - 0.0025) * 26.0)
            support = max(0.0, support - max(0.0, lower_raise - 0.0030) * 20.0)
            support = max(0.0, support - max(0.0, chin_raise - 0.0025) * 16.0)
            return support, 1.0 if support > 0.0 else 0.0, balance
        raise_mean = float(mouth_metrics.get("corner_raise_mean", 0.0) or 0.0)
        lift_mean = float(mouth_metrics.get("corner_lift_mean", 0.0) or 0.0)
        balance = float(mouth_metrics.get("corner_raise_balance", 0.0) or 0.0)
        support = max(0.0, raise_mean - 0.004) * 55.0 + max(0.0, lift_mean - 0.003) * 18.0
        return support, 1.0 if support > 0.0 else 0.0, balance

    if au_name == "dimpler":
        if dataset == "samm":
            inward_mean = float(mouth_metrics.get("corner_inward_mean", 0.0) or 0.0)
            backward_mean = float(mouth_metrics.get("corner_backward_mean", 0.0) or 0.0)
            width_ratio = float(mouth_metrics.get("mouth_width_ratio", 1.0) or 1.0)
            outward_mean = float(mouth_metrics.get("corner_outward_mean", 0.0) or 0.0)
            raise_mean = float(mouth_metrics.get("corner_raise_relative_mean", 0.0) or 0.0)
            lift_mean = float(mouth_metrics.get("corner_lift_mean", 0.0) or 0.0)
            open_ratio = float(mouth_metrics.get("mouth_open_ratio", 1.0) or 1.0)
            drop_mean = float(mouth_metrics.get("corner_drop_mean", 0.0) or 0.0)
            forward_mean = float(mouth_metrics.get("corner_forward_mean", 0.0) or 0.0)
            lower_drop = float(mouth_metrics.get("lower_lip_drop_norm", 0.0) or 0.0)
            center_drop = float(mouth_metrics.get("mouth_center_drop_norm", 0.0) or 0.0)
            inward_balance = float(mouth_metrics.get("corner_inward_balance", 0.0) or 0.0)
            backward_balance = float(mouth_metrics.get("corner_backward_balance", 0.0) or 0.0)
            core_inward = max(0.0, inward_mean - 0.0018)
            core_backward = max(0.0, backward_mean - 0.0010)
            core_compress = max(0.0, 0.9985 - width_ratio)
            support = (
                core_inward * 72.0
                + core_backward * 62.0
                + core_compress * 18.0
            ) * (0.58 + 0.42 * min(max(inward_balance, 0.0), max(backward_balance, 0.0)))
            if core_inward > 0.0 or core_backward > 0.0:
                support += max(0.0, lift_mean - 0.0040) * 8.0
                support += max(0.0, raise_mean - 0.0045) * 4.0
            else:
                support *= 0.12

            support = max(0.0, support - max(0.0, outward_mean - 0.0012) * 48.0)
            support = max(0.0, support - max(0.0, raise_mean - 0.0048) * 24.0)
            support = max(0.0, support - max(0.0, lift_mean - 0.0065) * 14.0)
            support = max(0.0, support - max(0.0, width_ratio - 1.0015) * 18.0)
            support = max(0.0, support - max(0.0, open_ratio - 1.04) * 14.0)
            support = max(0.0, support - max(0.0, lower_drop - 0.012) * 36.0)
            support = max(0.0, support - max(0.0, center_drop - 0.008) * 18.0)
            support = max(0.0, support - drop_mean * 10.0)
            support = max(0.0, support - max(0.0, forward_mean - 0.0012) * 12.0)
            balance = max(
                min(max(inward_balance, 0.0), max(backward_balance, 0.0)),
                max(inward_balance, 0.0) * 0.90,
                max(backward_balance, 0.0) * 0.80,
            )
            return support, 1.0 if support > 0.0 else 0.0, max(balance, 0.45)
        raise_mean = float(mouth_metrics.get("corner_raise_mean", 0.0) or 0.0)
        lift_mean = float(mouth_metrics.get("corner_lift_mean", 0.0) or 0.0)
        outward_mean = float(mouth_metrics.get("corner_outward_mean", 0.0) or 0.0)
        open_ratio = float(mouth_metrics.get("mouth_open_ratio", 1.0) or 1.0)
        drop_mean = float(mouth_metrics.get("corner_drop_mean", 0.0) or 0.0)
        asymmetry = float(mouth_metrics.get("corner_asymmetry", 0.0) or 0.0)
        support = max(0.0, raise_mean - 0.0035) * 34.0
        support += max(0.0, lift_mean - 0.003) * 22.0
        support = max(0.0, support - max(0.0, outward_mean - 0.004) * 25.0)
        support = max(0.0, support - max(0.0, open_ratio - 1.03) * 10.0)
        support = max(0.0, support - drop_mean * 18.0)
        balance = max(0.45, 1.0 - asymmetry * 20.0)
        return support, 1.0 if support > 0.0 else 0.0, balance

    if au_name == "lip corner depressor":
        drop_mean = float(mouth_metrics.get("corner_drop_mean", 0.0) or 0.0)
        depress_mean = float(mouth_metrics.get("corner_depress_mean", 0.0) or 0.0)
        lift_mean = float(mouth_metrics.get("corner_raise_mean", 0.0) or 0.0)
        balance = float(mouth_metrics.get("corner_drop_balance", 0.0) or 0.0)
        support = max(0.0, drop_mean - 0.003) * 45.0 + max(0.0, depress_mean - 0.002) * 18.0
        support = max(0.0, support - lift_mean * 20.0)
        return support, 1.0 if support > 0.0 else 0.0, balance

    if au_name == "lip stretcher":
        if dataset == "samm":
            outward_mean = float(mouth_metrics.get("corner_outward_mean", 0.0) or 0.0)
            backward_mean = float(mouth_metrics.get("corner_backward_mean", 0.0) or 0.0)
            inward_mean = float(mouth_metrics.get("corner_inward_mean", 0.0) or 0.0)
            width_ratio = float(mouth_metrics.get("mouth_width_ratio", 1.0) or 1.0)
            drop_mean = float(mouth_metrics.get("corner_drop_mean", 0.0) or 0.0)
            raise_mean = float(mouth_metrics.get("corner_raise_relative_mean", 0.0) or 0.0)
            open_ratio = float(mouth_metrics.get("mouth_open_ratio", 1.0) or 1.0)
            balance = max(
                float(mouth_metrics.get("corner_outward_balance", 0.0) or 0.0),
                float(mouth_metrics.get("corner_backward_balance", 0.0) or 0.0) * 0.85,
            )
            support = max(0.0, outward_mean - 0.0013) * 36.0
            support += max(0.0, backward_mean - 0.0006) * 16.0
            support += max(0.0, width_ratio - 1.004) * 14.0
            support = max(0.0, support - max(0.0, inward_mean - 0.0010) * 28.0)
            support = max(0.0, support - max(0.0, raise_mean - 0.0018) * 16.0)
            support = max(0.0, support - max(0.0, drop_mean - 0.0055) * 8.0)
            support = max(0.0, support - max(0.0, 1.0 - open_ratio) * 6.0)
            return support, 1.0 if support > 0.0 else 0.0, max(balance, 0.4)
        outward_mean = float(mouth_metrics.get("corner_outward_mean", 0.0) or 0.0)
        raise_mean = float(mouth_metrics.get("corner_raise_mean", 0.0) or 0.0)
        drop_mean = float(mouth_metrics.get("corner_drop_mean", 0.0) or 0.0)
        support = max(0.0, outward_mean - 0.003) * 50.0
        support += max(0.0, float(mouth_metrics.get("mouth_width_ratio", 1.0) or 1.0) - 1.02) * 8.0
        support = max(0.0, support - max(raise_mean, drop_mean) * 10.0)
        balance = float(mouth_metrics.get("corner_outward_balance", 0.0) or 0.0)
        return support, 1.0 if support > 0.0 else 0.0, balance

    if au_name == "lips part":
        open_ratio = float(mouth_metrics.get("mouth_open_ratio", 1.0) or 1.0)
        lower_ratio = float(mouth_metrics.get("lower_lip_nose_ratio", 1.0) or 1.0)
        lower_drop = float(mouth_metrics.get("lower_lip_drop_norm", 0.0) or 0.0)
        support = max(0.0, open_ratio - 1.03) * 6.0
        support += max(0.0, lower_ratio - 1.015) * 10.0
        support += max(0.0, lower_drop - 0.005) * 22.0
        return support, 1.0 if support > 0.0 else 0.0, 1.0

    if au_name == "jaw drop":
        open_ratio = float(mouth_metrics.get("mouth_open_ratio", 1.0) or 1.0)
        lower_ratio = float(mouth_metrics.get("lower_lip_nose_ratio", 1.0) or 1.0)
        lower_drop = float(mouth_metrics.get("lower_lip_drop_norm", 0.0) or 0.0)
        support = max(0.0, open_ratio - 1.06) * 7.0
        support += max(0.0, lower_ratio - 1.04) * 10.0
        support += max(0.0, lower_drop - 0.01) * 42.0
        return support, 1.0 if support > 0.0 else 0.0, 1.0

    if au_name == "lip tightener":
        if dataset == "samm":
            width_ratio = float(mouth_metrics.get("mouth_width_ratio", 1.0) or 1.0)
            open_ratio = float(mouth_metrics.get("mouth_open_ratio", 1.0) or 1.0)
            outward_mean = float(mouth_metrics.get("corner_outward_mean", 0.0) or 0.0)
            inward_mean = float(mouth_metrics.get("corner_inward_mean", 0.0) or 0.0)
            corner_lift = float(mouth_metrics.get("corner_lift_mean", 0.0) or 0.0)
            corner_raise = float(mouth_metrics.get("corner_raise_mean", 0.0) or 0.0)
            corner_drop = float(mouth_metrics.get("corner_drop_mean", 0.0) or 0.0)
            support = 0.0
            if width_ratio <= 0.997 and inward_mean >= 0.0030:
                support = max(0.0, 0.995 - width_ratio) * 18.0
                support += max(0.0, inward_mean - 0.0030) * 10.0
                support += max(0.0, 0.995 - open_ratio) * 6.0
            support = max(0.0, support - max(0.0, outward_mean - 0.0008) * 36.0)
            support = max(0.0, support - corner_lift * 22.0)
            support = max(0.0, support - max(corner_raise, corner_drop) * 18.0)
            if open_ratio > 1.02:
                support = 0.0
            return support, 1.0 if support > 0.0 else 0.0, 1.0
        width_ratio = float(mouth_metrics.get("mouth_width_ratio", 1.0) or 1.0)
        open_ratio = float(mouth_metrics.get("mouth_open_ratio", 1.0) or 1.0)
        corner_lift = float(mouth_metrics.get("corner_lift_mean", 0.0) or 0.0)
        support = max(0.0, width_ratio - 1.01) * 4.0 + max(0.0, 1.0 - open_ratio) * 6.0
        support = max(0.0, support - corner_lift * 10.0)
        return support, 1.0 if support > 0.0 else 0.0, 1.0

    return 0.0, 0.0, 1.0


def build_au_evidence(
    *,
    dataset: str,
    raw_regions: dict[str, Any],
    residual_regions: dict[str, Any],
    eye_ratio: float | None,
    eye_metrics: dict[str, Any] | None,
    mouth_metrics: dict[str, Any] | None,
    head_motion: dict[str, Any],
) -> dict[str, Any]:
    calibration = load_motion_feature_calibration()
    dataset_calibration = (calibration.get("datasets") or {}).get(dataset, {})
    au_thresholds = dataset_calibration.get("au_thresholds") or {}
    au_geometry_profiles = dataset_calibration.get("au_geometry_profiles") or {}
    evidence: dict[str, Any] = {}
    use_samm_geometry = dataset == "samm"
    use_geometry_metrics = _uses_geometry_metrics(dataset)
    geometry_eye_metrics = (eye_metrics or ({"mean_ratio": eye_ratio} if eye_ratio is not None else {})) if use_geometry_metrics else {}
    geometry_mouth_metrics = (mouth_metrics or {}) if use_geometry_metrics else {}

    for au_name in EVIDENCE_AUS:
        if not use_samm_geometry and au_name in SAMM_GEOMETRY_ONLY_AUS:
            continue
        threshold_info = au_thresholds.get(au_name) or {}
        threshold = float(threshold_info.get("threshold", 0.0) or 0.0)
        precision = float(threshold_info.get("precision", 0.0) or 0.0)
        recall = float(threshold_info.get("recall", 0.0) or 0.0)
        f1 = float(threshold_info.get("f1", 0.0) or 0.0)
        calibrated = _calibration_ready(threshold_info)
        if dataset == "samm" and au_name == "dimpler" and not calibrated and max(precision, recall, f1) > 0.0:
            threshold = 0.10
            calibrated = True
        score = 0.0
        normalized_support = 0.0
        raw_support = 0.0
        residual_support = 0.0
        combined_support = 0.0
        direction_score = 0.0
        balance_score = 1.0
        floor = 0.12

        geometry_enabled = _geometry_enabled_for_au(dataset, au_name)
        geom_support, geom_direction, geom_balance = _geometry_au_support(
            au_name,
            dataset=dataset,
            eye_metrics=geometry_eye_metrics if geometry_enabled else {},
            mouth_metrics=geometry_mouth_metrics if geometry_enabled else {},
            calibration_profiles=au_geometry_profiles,
        )

        if use_samm_geometry and au_name in {"upper lid raiser", "eye closure"}:
            support = geom_support
            floor = 0.14 if au_name == "upper lid raiser" else 0.12
            normalized_support = support / max(threshold, floor) if support > 0.0 else 0.0
            score = normalized_support
            raw_support = support
            residual_support = support
            combined_support = support
            direction_score = geom_direction
            balance_score = geom_balance
        else:
            ranges = AU_DIRECTION_RANGES.get(au_name, [])
            regions = AU_REGION_MAP.get(au_name, ())
            summaries = [
                _region_support_summary(
                    region,
                    ranges=ranges,
                    raw_regions=raw_regions,
                    residual_regions=residual_regions,
                    head_motion=head_motion,
                )
                for region in regions
            ]
            if au_name in PAIR_AUS:
                combined_values = [item["combined_support"] for item in summaries]
                raw_values = [item["raw_support"] for item in summaries]
                residual_values = [item["residual_support"] for item in summaries]
                direction_values = [item["direction"] for item in summaries]
                balance_score = _balance_score(combined_values)
                raw_support = min(raw_values) if raw_values else 0.0
                residual_support = min(residual_values) if residual_values else 0.0
                combined_support = min(combined_values) if combined_values else 0.0
                direction_score = min(direction_values) if direction_values else 0.0
            else:
                raw_support = max((item["raw_support"] for item in summaries), default=0.0)
                residual_support = max((item["residual_support"] for item in summaries), default=0.0)
                combined_support = max((item["combined_support"] for item in summaries), default=0.0)
                direction_score = max((item["direction"] for item in summaries), default=0.0)
                balance_score = 1.0

            floor = 0.08 if au_name in {"brow lowerer", "nose wrinkler", "upper lip raiser", "chin raiser", "lip tightener"} else 0.12
            normalized_support = combined_support / max(threshold, floor) if combined_support > 0.0 else 0.0
            score = normalized_support * (0.50 + 0.50 * direction_score) * (0.65 + 0.35 * balance_score)
            if precision > 0.0:
                score *= 0.90 + 0.20 * precision

            geom_score = 0.0
            geom_normalized = 0.0
            if geom_support > 0.0 and geometry_enabled:
                geom_normalized = geom_support / max(threshold, floor)
                geom_score = geom_normalized * (0.50 + 0.50 * geom_direction) * (0.65 + 0.35 * geom_balance)
                raw_support = max(raw_support, geom_support)
                residual_support = max(residual_support, geom_support)
                normalized_support = max(normalized_support, geom_normalized)
                direction_score = max(direction_score, geom_direction)
                if balance_score <= 0.0:
                    balance_score = geom_balance
                else:
                    balance_score = max(balance_score, geom_balance)
                score = max(score, geom_score)
                if precision > 0.0:
                    score *= 0.90 + 0.20 * precision

            if dataset == "samm" and au_name in {"inner brow raiser", "outer brow raiser", "brow lowerer", "lip corner puller", "chin raiser", "nose wrinkler", "upper lip raiser", "lip stretcher", "lip tightener"}:
                if au_name == "inner brow raiser":
                    pose_noise_penalty = max(
                        0.0,
                        float(geometry_eye_metrics.get("pose_landmark_fit_rmse_3d", 0.0) or 0.0) - 2.0,
                    ) * 0.10
                    brow_drop_penalty = float(geometry_eye_metrics.get("brow_center_drop_norm", 0.0) or 0.0) * 18.0
                    if geom_score > 0.0:
                        score = max(0.0, 0.12 * score + 1.08 * geom_score - pose_noise_penalty - brow_drop_penalty)
                    else:
                        score = max(0.0, 0.10 * score - pose_noise_penalty - brow_drop_penalty)
                elif au_name == "outer brow raiser":
                    pose_noise_penalty = max(
                        0.0,
                        float(geometry_eye_metrics.get("pose_landmark_fit_rmse_3d", 0.0) or 0.0) - 2.0,
                    ) * 0.10
                    brow_drop_penalty = float(geometry_eye_metrics.get("brow_center_drop_norm", 0.0) or 0.0) * 15.0
                    if geom_score > 0.0:
                        score = max(0.0, 0.15 * score + 1.05 * geom_score - pose_noise_penalty - brow_drop_penalty)
                    else:
                        score = max(0.0, 0.08 * score - pose_noise_penalty - brow_drop_penalty)
                elif au_name == "brow lowerer":
                    raise_penalty = (
                        float(geometry_eye_metrics.get("inner_brow_relative_raise_mean", 0.0) or 0.0)
                        + float(geometry_eye_metrics.get("outer_brow_relative_raise_mean", 0.0) or 0.0)
                    ) * 22.0
                    if geom_score > 0.0:
                        score = max(0.0, 0.25 * score + 1.00 * geom_score - raise_penalty)
                    else:
                        score = max(0.0, 0.22 * score - raise_penalty)
                elif au_name == "lip corner puller":
                    lower_face_shift = (
                        float(geometry_mouth_metrics.get("mouth_center_raise_norm", 0.0) or 0.0)
                        + float(geometry_mouth_metrics.get("lower_lip_raise_norm", 0.0) or 0.0)
                        + float(geometry_mouth_metrics.get("chin_raise_norm", 0.0) or 0.0)
                    ) * 8.0
                    if geom_score > 0.0:
                        score = max(0.0, 0.45 * score + 0.80 * geom_score - lower_face_shift)
                    else:
                        score = max(0.0, 0.30 * score - lower_face_shift)
                elif au_name == "nose wrinkler":
                    if geom_score > 0.0:
                        score = max(0.0, 0.70 * score + 0.85 * geom_score)
                    else:
                        score *= 0.80
                elif au_name == "upper lip raiser":
                    if geom_score > 0.0:
                        score = max(0.0, 0.65 * score + 0.90 * geom_score)
                    else:
                        score *= 0.75
                elif au_name == "lip stretcher":
                    if geom_score > 0.0:
                        score = max(0.0, 0.55 * score + 0.85 * geom_score)
                    else:
                        score *= 0.55
                elif au_name == "lip tightener":
                    if geom_score > 0.0:
                        score = max(0.0, 0.25 * score + 0.55 * geom_score)
                    else:
                        score *= 0.15
                else:
                    if geom_score > 0.0:
                        score = max(0.0, 0.25 * score + 0.95 * geom_score)
                    else:
                        score *= 0.2

        if dataset == "samm" and au_name == "lip tightener" and precision < 0.05 and f1 < 0.05:
            score *= 0.08
            normalized_support *= 0.12

        if use_samm_geometry and au_name in SAMM_GEOMETRY_ONLY_AUS and not calibrated:
            score = 0.0
            normalized_support = 0.0

        present = bool(
            score >= 1.0
            or (
                normalized_support >= 0.85
                and direction_score >= 0.80
                and balance_score >= 0.70
            )
        )
        if dataset == "samm" and au_name in SAMM_BROW_AGENT_AUS:
            # SAMM brow evidence stays visible for downstream specialists, but this
            # stage never upgrades it into a direct present/not-present decision.
            present = False
        if use_samm_geometry and au_name in SAMM_GEOMETRY_ONLY_AUS and not calibrated:
            strength = "uncalibrated"
            present = False
        elif score >= 1.35:
            strength = "strong"
        elif score >= 0.85:
            strength = "moderate"
        elif score > 0.0:
            strength = "weak"
        else:
            strength = "none"

        evidence[au_name] = {
            "score": round(score, 4),
            "normalized_support": round(normalized_support, 4),
            "raw_support": round(raw_support, 4),
            "residual_support": round(residual_support, 4),
            "combined_support": round(combined_support, 4),
            "geometry_support": round(geom_support, 4),
            "direction_score": round(direction_score, 4),
            "balance_score": round(balance_score, 4),
            "threshold": round(threshold, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "calibrated": calibrated,
            "present": present,
            "strength": strength,
            "agent_only": bool(dataset == "samm" and au_name in SAMM_BROW_AGENT_AUS),
        }

    return evidence


def ensure_au_evidence(numeric: dict[str, Any] | None, dataset: str = "") -> dict[str, Any]:
    numeric = numeric or {}
    if not numeric.get("available"):
        numeric.setdefault("au_evidence", {})
        numeric.setdefault("landmark_aus", [])
        numeric.setdefault("eye_metrics", {})
        numeric.setdefault("mouth_metrics", {})
        return numeric
    if numeric.get("au_evidence"):
        if "landmark_aus" not in numeric:
            ordered = sorted(
                (numeric.get("au_evidence") or {}).items(),
                key=lambda item: (
                    bool(item[1].get("present")),
                    float(item[1].get("score", 0.0) or 0.0),
                    float(item[1].get("normalized_support", 0.0) or 0.0),
                ),
                reverse=True,
            )
            numeric["landmark_aus"] = [name for name, info in ordered if bool(info.get("present"))]
        return numeric
    evidence = build_au_evidence(
        dataset=dataset or "",
        raw_regions=numeric.get("landmark_au_regions") or {},
        residual_regions=numeric.get("landmark_residual_au_regions") or {},
        eye_ratio=numeric.get("eye_ratio"),
        eye_metrics=numeric.get("eye_metrics") or {},
        mouth_metrics=numeric.get("mouth_metrics") or {},
        head_motion=numeric.get("head_motion") or {},
    )
    numeric["au_evidence"] = evidence
    ordered = sorted(
        evidence.items(),
        key=lambda item: (
            bool(item[1].get("present")),
            float(item[1].get("score", 0.0) or 0.0),
            float(item[1].get("normalized_support", 0.0) or 0.0),
        ),
        reverse=True,
    )
    numeric["landmark_aus"] = [name for name, info in ordered if bool(info.get("present"))]
    return numeric


def get_numeric_features(clip: dict) -> dict[str, Any]:
    model_asset = ensure_numeric_feature_dependencies()
    onset_path, apex_path = resolve_onset_apex_paths(Path(clip["path"]))
    result: dict[str, Any] = {
        "available": False,
        "onset_path": str(onset_path) if onset_path else "",
        "apex_path": str(apex_path) if apex_path else "",
        "landmark_au_regions": {},
        "landmark_residual_method": LANDMARK_RESIDUAL_METHOD,
        "landmark_residual_au_regions": {},
        "head_motion_transform": {},
        "head_motion": {},
        "eye_ratio": None,
        "eye_metrics": {},
        "eyelid_signal": "unknown",
        "mouth_metrics": {},
        "landmark_aus": [],
        "au_evidence": {},
        "compensation_applied": False,
        "notes": [],
    }
    if onset_path is None or apex_path is None:
        result["notes"].append("missing_onset_or_apex")
        return result
    if not Path(clip["path"]).exists():
        result["notes"].append("missing_flow")
        return result

    try:
        au_regions, hm, iod, onset_lm, apex_lm, H, W = extract_landmark_au_regions(
            onset_path=str(onset_path),
            flow_npy_path=str(clip["path"]),
            model_path=str(model_asset),
        )
        transform_info = build_head_motion_transform(
            onset_lm,
            apex_lm,
            W=W,
            H=H,
            method=LANDMARK_RESIDUAL_METHOD,
        ) if onset_lm is not None and apex_lm is not None else None
        use_geometry_metrics = _uses_geometry_metrics(clip.get("dataset", ""))
        eye_metrics = compute_eye_geometry_metrics(
            onset_lm,
            apex_lm,
            dataset=clip.get("dataset", ""),
            W=W,
            H=H,
            iod=iod,
            transform_info=transform_info,
        ) if use_geometry_metrics and onset_lm is not None and apex_lm is not None else {}
        eye_ratio = eye_metrics.get("mean_ratio")
        if eye_ratio is None:
            eye_ratio = compute_eye_height_ratio(str(onset_path), str(apex_path), model_path=str(model_asset))
        eyelid_signal = infer_eyelid_signal(eye_ratio, clip.get("dataset", ""))
        mouth_metrics = compute_mouth_geometry_metrics(
            onset_lm,
            apex_lm,
            W=W,
            H=H,
            transform_info=transform_info,
        ) if use_geometry_metrics and onset_lm is not None and apex_lm is not None else {}
        residual_regions = compute_landmark_residual_au_regions(
            onset_lm,
            apex_lm,
            dataset=clip.get("dataset", ""),
            W=W,
            H=H,
            iod=iod,
            transform_info=transform_info,
        ) if onset_lm is not None and apex_lm is not None else {}
        landmark_aus = infer_aus_from_numeric(
            merge_au_region_evidence(residual_regions, au_regions),
            eye_ratio=eye_ratio,
            dataset=clip.get("dataset", ""),
            eye_metrics=eye_metrics,
            mouth_metrics=mouth_metrics,
        )
        result.update(
            {
                "available": True,
                "landmark_au_regions": au_regions or {},
                "landmark_residual_au_regions": residual_regions or {},
                "head_motion_transform": serialize_transform_info(transform_info),
                "head_motion": hm or {},
                "eye_ratio": eye_ratio,
                "eye_metrics": eye_metrics,
                "eyelid_signal": eyelid_signal,
                "mouth_metrics": mouth_metrics,
                "landmark_aus": landmark_aus,
                "au_evidence": {},
                "compensation_applied": True,
            }
        )
        ensure_au_evidence(result, clip.get("dataset", ""))
    except Exception as exc:
        result["notes"].append(f"numeric_feature_error:{exc}")
    return result


def resolve_onset_apex_paths(flow_npy_path: Path) -> tuple[Path | None, Path | None]:
    base = flow_npy_path.parent.parent
    frames_dir = base / "frames_used"
    candidates = [
        (frames_dir / "onset.jpg", frames_dir / "apex.jpg"),
        (frames_dir / "onset.png", frames_dir / "apex.png"),
    ]
    for onset, apex in candidates:
        if onset.exists() and apex.exists():
            return onset, apex
    return None, None


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


def landmark_xy(landmarks, idx: int, W: int, H: int) -> np.ndarray:
    return np.array([landmarks[idx].x * W, landmarks[idx].y * H], dtype=np.float32)


def landmark_xyz(landmarks, idx: int, W: int, H: int) -> np.ndarray:
    point = landmarks[idx]
    return np.array([point.x * W, point.y * H, point.z * W], dtype=np.float32)


def landmarks_to_points(landmarks, names: list[str], W: int, H: int) -> np.ndarray:
    return np.array([landmark_xy(landmarks, LM_IDX[name], W, H) for name in names], dtype=np.float32)


def _rect_mask(H: int, W: int, cx: float, cy: float, hw: float, hh: float) -> np.ndarray:
    r0 = max(0, int(cy - hh))
    r1 = min(H, int(cy + hh))
    c0 = max(0, int(cx - hw))
    c1 = min(W, int(cx + hw))
    mask = np.zeros((H, W), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def _flow_stats(u: np.ndarray, v: np.ndarray, mag: np.ndarray, mask_2d: np.ndarray, mag_threshold: float = 0.05) -> dict[str, Any] | None:
    m = mag[mask_2d]
    uu = u[mask_2d]
    vv = v[mask_2d]
    active = m > mag_threshold
    if active.sum() < 4:
        return None
    uw = (uu * m)[active]
    vw = (vv * m)[active]
    angle = float(np.degrees(np.arctan2(-vw.mean(), uw.mean())) % 360)
    return {
        "angle": round(angle, 1),
        "dir": angle_to_direction(angle),
        "mag": round(float(m[active].mean()), 3),
        "active": round(float(active.sum() / max(mask_2d.sum(), 1)), 2),
    }


def _get_detector():
    global _lm_detector
    if _lm_detector is None:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=bo.BaseOptions(),
            num_faces=1,
            min_face_detection_confidence=0.2,
        )
        _lm_detector = mp_vision.FaceLandmarker.create_from_options(opts)
    return _lm_detector


def _get_detector_with_model(model_path: str):
    global _lm_detector
    if _lm_detector is None:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=bo.BaseOptions(model_asset_path=model_path),
            num_faces=1,
            min_face_detection_confidence=0.2,
        )
        _lm_detector = mp_vision.FaceLandmarker.create_from_options(opts)
    return _lm_detector


def _detect_landmarks(image_path: str):
    img_np = np.array(Image.open(image_path).convert("RGB"))
    H, W = img_np.shape[:2]
    model_asset = ensure_numeric_feature_dependencies()
    detector = _get_detector_with_model(str(model_asset))
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np))
    if not res.face_landmarks:
        return None, H, W
    return res.face_landmarks[0], H, W


def build_head_motion_transform(onset_lm, apex_lm, W: int, H: int, method: str = "similarity") -> dict[str, Any] | None:
    if onset_lm is None or apex_lm is None:
        return None
    src = landmarks_to_points(onset_lm, RIGID_LANDMARKS, W, H)
    dst = landmarks_to_points(apex_lm, RIGID_LANDMARKS, W, H)
    src_xyz = np.array([landmark_xyz(onset_lm, LM_IDX[name], W, H) for name in RIGID_LANDMARKS], dtype=np.float32)
    dst_xyz = np.array([landmark_xyz(apex_lm, LM_IDX[name], W, H) for name in RIGID_LANDMARKS], dtype=np.float32)
    if method in {"similarity", "affine"} and not _CV2_AVAILABLE:
        method = "translation"
    if method == "translation":
        delta = np.median(dst - src, axis=0)
        matrix = np.array([[1.0, 0.0, float(delta[0])], [0.0, 1.0, float(delta[1])]], dtype=np.float32)
    elif method == "similarity":
        matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS, ransacReprojThreshold=3.0)
    elif method == "affine":
        matrix, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS, ransacReprojThreshold=3.0)
    else:
        raise ValueError(f"Unsupported method: {method}")
    if matrix is None:
        return None
    residuals = []
    linear = matrix[:, :2]
    translation = matrix[:, 2]
    for src_pt, dst_pt in zip(src, dst):
        pred = linear @ src_pt + translation
        residuals.append(float(np.linalg.norm(dst_pt - pred)))
    transform_info: dict[str, Any] = {
        "method": method,
        "matrix": matrix.astype(np.float32),
        "rotation_deg": round(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))), 4),
        "landmark_fit_rmse": round(float(np.sqrt(np.mean(np.square(residuals)))), 4),
    }
    info3d = _estimate_similarity_transform_3d(src_xyz, dst_xyz)
    if info3d is not None:
        transform_info.update(info3d)
    return transform_info


def serialize_transform_info(transform_info: dict[str, Any] | None) -> dict[str, Any]:
    if transform_info is None:
        return {}
    payload = {
        "method": transform_info["method"],
        "rotation_deg": transform_info["rotation_deg"],
        "landmark_fit_rmse": transform_info["landmark_fit_rmse"],
    }
    if "rotation3d_deg" in transform_info:
        payload["rotation3d_deg"] = transform_info["rotation3d_deg"]
    if "scale3d" in transform_info:
        payload["scale3d"] = transform_info["scale3d"]
    if "landmark_fit_rmse_3d" in transform_info:
        payload["landmark_fit_rmse_3d"] = transform_info["landmark_fit_rmse_3d"]
    return payload


def _derived_au_points(lm, W: int, H: int) -> dict[str, np.ndarray]:
    def pt(name: str) -> np.ndarray:
        return landmark_xy(lm, LM_IDX[name], W, H)

    def mid(*names: str) -> np.ndarray:
        return np.mean([pt(name) for name in names], axis=0)

    return {
        "AU1_inner_brow_subR": np.array([pt("left_brow_inner")], dtype=np.float32),
        "AU1_inner_brow_subL": np.array([pt("right_brow_inner")], dtype=np.float32),
        "AU4_brow_lowerer": np.array([mid("left_brow_inner", "right_brow_inner")], dtype=np.float32),
        "AU2_brow_outer_subR": np.array([mid("left_brow_outer", "left_brow_peak")], dtype=np.float32),
        "AU2_brow_outer_subL": np.array([mid("right_brow_outer", "right_brow_peak")], dtype=np.float32),
        "AU9_nose_bridge": np.array([mid("nose_top", "left_nostril", "right_nostril")], dtype=np.float32),
        "AU10_upper_lip": np.array([mid("left_nostril", "right_nostril", "upper_lip_top")], dtype=np.float32),
        "AU12_corner_subR": np.array([pt("left_lip_corner")], dtype=np.float32),
        "AU12_corner_subL": np.array([pt("right_lip_corner")], dtype=np.float32),
        "AU23_lip_center": np.array([mid("upper_lip_top", "lower_lip_bot")], dtype=np.float32),
        "AU17_chin": np.array([mid("lower_lip_bot", "chin_bottom")], dtype=np.float32),
    }


def _derived_au_points_xyz(lm, W: int, H: int) -> dict[str, np.ndarray]:
    def pt(name: str) -> np.ndarray:
        return landmark_xyz(lm, LM_IDX[name], W, H)

    def mid(*names: str) -> np.ndarray:
        return np.mean([pt(name) for name in names], axis=0)

    return {
        "AU1_inner_brow_subR": np.array([pt("left_brow_inner")], dtype=np.float32),
        "AU1_inner_brow_subL": np.array([pt("right_brow_inner")], dtype=np.float32),
        "AU4_brow_lowerer": np.array([mid("left_brow_inner", "right_brow_inner")], dtype=np.float32),
        "AU2_brow_outer_subR": np.array([mid("left_brow_outer", "left_brow_peak")], dtype=np.float32),
        "AU2_brow_outer_subL": np.array([mid("right_brow_outer", "right_brow_peak")], dtype=np.float32),
        "AU9_nose_bridge": np.array([mid("nose_top", "left_nostril", "right_nostril")], dtype=np.float32),
        "AU10_upper_lip": np.array([mid("left_nostril", "right_nostril", "upper_lip_top")], dtype=np.float32),
        "AU12_corner_subR": np.array([pt("left_lip_corner")], dtype=np.float32),
        "AU12_corner_subL": np.array([pt("right_lip_corner")], dtype=np.float32),
        "AU23_lip_center": np.array([mid("upper_lip_top", "lower_lip_bot")], dtype=np.float32),
        "AU17_chin": np.array([mid("lower_lip_bot", "chin_bottom")], dtype=np.float32),
    }


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homog = np.concatenate([points.astype(np.float32), ones], axis=1)
    return (homog @ matrix.T).astype(np.float32)


def _transform_points_xyz(points: np.ndarray, matrix3d: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homog = np.concatenate([points.astype(np.float32), ones], axis=1)
    return (homog @ matrix3d.T).astype(np.float32)


def _residual_stat(delta: np.ndarray, iod: float) -> dict[str, Any]:
    dx = float(delta[0])
    dy = float(delta[1])
    mag_px = float(np.linalg.norm(delta))
    mag = mag_px / iod if iod > 1e-6 else 0.0
    angle = float(np.degrees(np.arctan2(-dy, dx)) % 360)
    return {
        "dx": round(dx, 4),
        "dy": round(dy, 4),
        "mag_px": round(mag_px, 4),
        "mag": round(mag, 4),
        "angle": round(angle, 1),
        "dir": angle_to_direction(angle),
        "active": round(1.0 if mag >= 0.01 else 0.0, 2),
    }


def compute_landmark_residual_au_regions(
    onset_lm,
    apex_lm,
    dataset: str,
    W: int,
    H: int,
    iod: float,
    transform_info: dict[str, Any] | None,
) -> dict[str, Any]:
    if onset_lm is None or apex_lm is None:
        return {}
    onset_points = _derived_au_points(onset_lm, W, H)
    onset_points_xyz = _derived_au_points_xyz(onset_lm, W, H)
    apex_points = _derived_au_points(apex_lm, W, H)
    predicted = onset_points
    if transform_info is not None:
        predicted = {name: _transform_points(points, transform_info["matrix"]) for name, points in onset_points.items()}
        matrix3d = transform_info.get("matrix3d")
        if dataset == "samm" and matrix3d is not None:
            brow_regions = {
                "AU1_inner_brow_subR",
                "AU1_inner_brow_subL",
                "AU4_brow_lowerer",
                "AU2_brow_outer_subR",
                "AU2_brow_outer_subL",
            }
            predicted_xyz = {name: _transform_points_xyz(points, matrix3d) for name, points in onset_points_xyz.items()}
            for name in brow_regions:
                if name in predicted_xyz:
                    predicted[name] = predicted_xyz[name][:, :2]
    result = {}
    for name, pred_pts in predicted.items():
        delta = np.mean(apex_points[name] - pred_pts, axis=0)
        result[name] = _residual_stat(delta, iod)
    return result


def estimate_head_motion(H: int, W: int, u: np.ndarray, v: np.ndarray, mag: np.ndarray, lm) -> dict[str, Any]:
    def px(name: str) -> np.ndarray:
        return np.array([lm[LM_IDX[name]].x * W, lm[LM_IDX[name]].y * H])

    iod = float(np.linalg.norm(px("right_eye_outer") - px("left_eye_outer")))
    nose_tip = px("nose_tip")
    mask = _rect_mask(H, W, nose_tip[0], nose_tip[1], iod * 0.10, iod * 0.10)
    m_vals = mag[mask]
    active = m_vals > 0.03
    if active.sum() < 4:
        return {"tx": 0.0, "ty": 0.0, "head_mag": 0.0, "angle": 0.0, "dir": "none", "quality": "unknown"}
    m = m_vals[active]
    uu = u[mask][active]
    vv = v[mask][active]
    tx = float((uu * m).sum() / m.sum())
    ty = float((vv * m).sum() / m.sum())
    hmag = float(np.sqrt(tx**2 + ty**2))
    hangle = float(np.degrees(np.arctan2(-ty, tx)) % 360)
    if hmag < 0.06:
        quality = "minimal"
    elif hmag < 0.8:
        quality = "clean"
    elif hmag < 2.5:
        quality = "moderate"
    else:
        quality = "dominated"
    return {
        "tx": round(tx, 4),
        "ty": round(ty, 4),
        "head_mag": round(hmag, 4),
        "angle": round(hangle, 1),
        "dir": angle_to_direction(hangle),
        "quality": quality,
    }


def compute_eye_height_ratio(onset_path: str, apex_path: str, model_path: str | None = None) -> float | None:
    def eye_heights(path: str):
        lm, H, _ = _detect_landmarks(path)
        if lm is None:
            return None
        left_h = abs(lm[LM_IDX["left_lid_bot"]].y - lm[LM_IDX["left_lid_top"]].y) * H
        right_h = abs(lm[LM_IDX["right_lid_bot"]].y - lm[LM_IDX["right_lid_top"]].y) * H
        return left_h, right_h

    onset = eye_heights(onset_path)
    apex = eye_heights(apex_path)
    if onset is None or apex is None:
        return None
    lh = apex[0] / onset[0] if onset[0] > 1.0 else 1.0
    rh = apex[1] / onset[1] if onset[1] > 1.0 else 1.0
    return round((lh + rh) / 2, 3)


def infer_eyelid_signal(eye_ratio: float | None, dataset: str = "") -> str:
    if eye_ratio is None:
        return "unknown"
    if dataset == "samm" and eye_ratio < 0.88:
        return "closure"
    if eye_ratio < 0.97:
        return "narrowing"
    if eye_ratio > 1.03:
        return "widening"
    return "stable"


def extract_landmark_au_regions(onset_path: str, flow_npy_path: str, mag_threshold: float = 0.05, model_path: str | None = None):
    lm, H, W = _detect_landmarks(onset_path)
    _, apex_path = resolve_onset_apex_paths(Path(flow_npy_path))
    apex_lm, _, _ = _detect_landmarks(str(apex_path)) if apex_path else (None, None, None)
    if lm is None:
        return None, None, 0.0, None, None, H, W

    def px(name: str) -> np.ndarray:
        return np.array([lm[LM_IDX[name]].x * W, lm[LM_IDX[name]].y * H])

    def mid(*names: str) -> np.ndarray:
        return np.mean([px(name) for name in names], axis=0)

    iod = float(np.linalg.norm(px("right_eye_outer") - px("left_eye_outer")))
    bw = float(np.linalg.norm(px("right_brow_outer") - px("left_brow_outer")))
    arr = np.load(flow_npy_path).astype(np.float32)
    u = arr[:, :, 0].copy()
    v = arr[:, :, 1].copy()
    mag = np.sqrt(u**2 + v**2).astype(np.float32)
    Hf, Wf = arr.shape[:2]
    if Hf != H or Wf != W:
        sx, sy = Wf / W, Hf / H
        orig_px = px

        def px(name: str) -> np.ndarray:
            return orig_px(name) * np.array([sx, sy])

        def mid(*names: str) -> np.ndarray:
            return np.mean([px(name) for name in names], axis=0)

        iod = float(np.linalg.norm(px("right_eye_outer") - px("left_eye_outer")))
        bw = float(np.linalg.norm(px("right_brow_outer") - px("left_brow_outer")))
        H, W = Hf, Wf

    hm = estimate_head_motion(H, W, u, v, mag, lm)

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
        "AU1_inner_brow_subR": _rect_mask(H, W, inner_r[0], inner_r[1], iod * 0.16, iod * 0.16),
        "AU1_inner_brow_subL": _rect_mask(H, W, inner_l[0], inner_l[1], iod * 0.16, iod * 0.16),
        "AU4_brow_lowerer": _rect_mask(H, W, brow_mid[0], brow_mid[1], bw * 0.52, iod * 0.18),
        "AU2_brow_outer_subR": _rect_mask(H, W, left_out[0], left_out[1], iod * 0.22, iod * 0.16),
        "AU2_brow_outer_subL": _rect_mask(H, W, right_out[0], right_out[1], iod * 0.22, iod * 0.16),
        "AU9_nose_bridge": _rect_mask(H, W, nose_mid[0], nose_mid[1], iod * 0.28, iod * 0.20),
        "AU10_upper_lip": _rect_mask(H, W, upper_lip[0], upper_lip[1], iod * 0.44, iod * 0.20),
        "AU12_corner_subR": _rect_mask(H, W, px("left_lip_corner")[0], px("left_lip_corner")[1], iod * 0.21, iod * 0.17),
        "AU12_corner_subL": _rect_mask(H, W, px("right_lip_corner")[0], px("right_lip_corner")[1], iod * 0.21, iod * 0.17),
        "AU23_lip_center": _rect_mask(H, W, lip_ctr[0], lip_ctr[1], lip_hw, iod * 0.14),
        "AU17_chin": _rect_mask(H, W, chin_ctr[0], chin_ctr[1], iod * 0.32, iod * 0.16),
    }
    return {key: _flow_stats(u, v, mag, mask, mag_threshold) for key, mask in regions.items()}, hm, iod, lm, apex_lm, H, W


def merge_au_region_evidence(
    primary_regions: dict[str, Any] | None,
    fallback_regions: dict[str, Any] | None,
    *,
    min_mag: float = 0.04,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    primary_regions = primary_regions or {}
    fallback_regions = fallback_regions or {}
    for key in set(primary_regions) | set(fallback_regions):
        primary = primary_regions.get(key) or {}
        fallback = fallback_regions.get(key) or {}
        if primary and float(primary.get("mag", 0.0) or 0.0) >= min_mag:
            merged[key] = primary
        elif fallback:
            merged[key] = fallback
        else:
            merged[key] = primary
    return merged


def infer_aus_from_numeric(
    au_regions: dict[str, Any],
    eye_ratio: float | None = None,
    *,
    dataset: str = "",
    eye_metrics: dict[str, Any] | None = None,
    mouth_metrics: dict[str, Any] | None = None,
) -> list[str]:
    aus: list[str] = []
    use_samm_geometry = dataset == "samm"
    use_geometry_metrics = _uses_geometry_metrics(dataset)
    eye_metrics = eye_metrics or {}
    mouth_metrics = mouth_metrics or {}
    inner_brow_geom, _, _ = _geometry_au_support(
        "inner brow raiser",
        dataset=dataset,
        eye_metrics=eye_metrics if use_geometry_metrics else {},
        mouth_metrics=mouth_metrics if use_geometry_metrics else {},
    )
    outer_brow_geom, _, _ = _geometry_au_support(
        "outer brow raiser",
        dataset=dataset,
        eye_metrics=eye_metrics if use_geometry_metrics else {},
        mouth_metrics=mouth_metrics if use_geometry_metrics else {},
    )
    brow_lowerer_geom, _, _ = _geometry_au_support(
        "brow lowerer",
        dataset=dataset,
        eye_metrics=eye_metrics if use_geometry_metrics else {},
        mouth_metrics=mouth_metrics if use_geometry_metrics else {},
    )
    lip_corner_geom, _, _ = _geometry_au_support(
        "lip corner puller",
        dataset=dataset,
        eye_metrics=eye_metrics if use_geometry_metrics else {},
        mouth_metrics=mouth_metrics if use_geometry_metrics else {},
    )
    dimpler_geom, _, _ = _geometry_au_support(
        "dimpler",
        dataset=dataset,
        eye_metrics=eye_metrics if use_geometry_metrics else {},
        mouth_metrics=mouth_metrics if use_geometry_metrics else {},
    )
    nose_geom, _, _ = _geometry_au_support(
        "nose wrinkler",
        dataset=dataset,
        eye_metrics=eye_metrics if use_geometry_metrics else {},
        mouth_metrics=mouth_metrics if use_geometry_metrics else {},
    )
    upper_lip_geom, _, _ = _geometry_au_support(
        "upper lip raiser",
        dataset=dataset,
        eye_metrics=eye_metrics if use_geometry_metrics else {},
        mouth_metrics=mouth_metrics if use_geometry_metrics else {},
    )
    lip_stretcher_geom, _, _ = _geometry_au_support(
        "lip stretcher",
        dataset=dataset,
        eye_metrics=eye_metrics if use_geometry_metrics else {},
        mouth_metrics=mouth_metrics if use_geometry_metrics else {},
    )
    lip_tightener_geom, _, _ = _geometry_au_support(
        "lip tightener",
        dataset=dataset,
        eye_metrics=eye_metrics if use_geometry_metrics else {},
        mouth_metrics=mouth_metrics if use_geometry_metrics else {},
    )
    chin_geom, _, _ = _geometry_au_support(
        "chin raiser",
        dataset=dataset,
        eye_metrics=eye_metrics if use_geometry_metrics else {},
        mouth_metrics=mouth_metrics if use_geometry_metrics else {},
    )

    def present_upward(region: str, active: float = 0.08, mag: float = 0.04) -> bool:
        info = au_regions.get(region)
        if not info or info.get("active", 0) < active or info.get("mag", 0) < mag:
            return False
        angle = info["angle"] % 360
        return 40 <= angle <= 130

    def present_downward(region: str, active: float = 0.08, mag: float = 0.04) -> bool:
        info = au_regions.get(region)
        if not info or info.get("active", 0) < active or info.get("mag", 0) < mag:
            return False
        angle = info["angle"] % 360
        return 215 <= angle <= 325

    def present_horizontal(region: str, active: float = 0.08, mag: float = 0.04) -> bool:
        info = au_regions.get(region)
        if not info or info.get("active", 0) < active or info.get("mag", 0) < mag:
            return False
        angle = info["angle"] % 360
        return angle <= 25 or angle >= 335 or 155 <= angle <= 205

    if use_samm_geometry:
        if (
            present_upward("AU1_inner_brow_subR")
            and present_upward("AU1_inner_brow_subL")
            and inner_brow_geom > 0.28
        ) or inner_brow_geom > 0.58:
            aus.append("inner brow raiser")
    elif present_upward("AU1_inner_brow_subR") and present_upward("AU1_inner_brow_subL"):
        aus.append("inner brow raiser")
    if use_samm_geometry:
        if (
            present_upward("AU2_brow_outer_subR")
            and present_upward("AU2_brow_outer_subL")
            and outer_brow_geom > 0.24
        ) or outer_brow_geom > 0.54:
            aus.append("outer brow raiser")
    elif present_upward("AU2_brow_outer_subR") and present_upward("AU2_brow_outer_subL"):
        aus.append("outer brow raiser")
    if use_samm_geometry:
        if (
            present_downward("AU4_brow_lowerer", active=0.10, mag=0.05)
            and brow_lowerer_geom > 0.06
        ) or brow_lowerer_geom > 0.36:
            aus.append("brow lowerer")
    elif present_downward("AU4_brow_lowerer", active=0.10, mag=0.05):
        aus.append("brow lowerer")
    eye_ratio = float(eye_metrics.get("mean_ratio", eye_ratio) or eye_ratio or 0.0) if (use_geometry_metrics and (eye_metrics or eye_ratio is not None)) else eye_ratio
    if eye_ratio is not None:
        if use_samm_geometry and float(eye_metrics.get("min_ratio", eye_ratio) or eye_ratio) < 0.88:
            aus.append("eye closure")
        elif eye_ratio < 0.97:
            aus.append("lid tightener")
        elif eye_ratio > 1.03:
            aus.append("upper lid raiser")
    if use_samm_geometry:
        if (present_upward("AU9_nose_bridge", active=0.04, mag=0.035) and nose_geom > 0.08) or nose_geom > 0.24:
            aus.append("nose wrinkler")
    elif present_upward("AU9_nose_bridge", active=0.05, mag=0.04):
        aus.append("nose wrinkler")
    if use_samm_geometry:
        if (present_upward("AU10_upper_lip", active=0.04, mag=0.035) and upper_lip_geom > 0.10) or upper_lip_geom > 0.32:
            aus.append("upper lip raiser")
    elif present_upward("AU10_upper_lip", active=0.05, mag=0.04) or (_geometry_enabled_for_au(dataset, "upper lip raiser") and float(mouth_metrics.get("upper_lip_nose_ratio", 1.0) or 1.0) < 0.98):
        aus.append("upper lip raiser")
    if use_samm_geometry:
        if (
            present_upward("AU12_corner_subR", active=0.05, mag=0.04)
            and present_upward("AU12_corner_subL", active=0.05, mag=0.04)
            and lip_corner_geom > 0.18
        ) or lip_corner_geom > 0.5:
            aus.append("lip corner puller")
    elif present_upward("AU12_corner_subR", active=0.05, mag=0.04) and present_upward("AU12_corner_subL", active=0.05, mag=0.04):
        aus.append("lip corner puller")
    elif use_samm_geometry and float(mouth_metrics.get("corner_lift_mean", 0.0) or 0.0) > 0.01:
        aus.append("lip corner puller")
    if use_samm_geometry and dimpler_geom > 0.28:
        aus.append("dimpler")
    if use_samm_geometry and float(mouth_metrics.get("corner_depress_mean", 0.0) or 0.0) > 0.01:
        aus.append("lip corner depressor")
    if use_samm_geometry:
        if chin_geom > 0.45 or (
            present_upward("AU17_chin", active=0.05, mag=0.04)
            and chin_geom > 0.15
            and float(mouth_metrics.get("mouth_open_ratio", 1.0) or 1.0) < 1.03
        ):
            aus.append("chin raiser")
    elif present_upward("AU17_chin", active=0.05, mag=0.04):
        aus.append("chin raiser")
    lip_center = au_regions.get("AU23_lip_center")
    if use_samm_geometry:
        if (
            lip_center
            and present_horizontal("AU23_lip_center", active=0.05, mag=0.04)
            and lip_tightener_geom > 0.05
            and float(mouth_metrics.get("corner_outward_mean", 0.0) or 0.0) < 0.0045
        ) or lip_tightener_geom > 0.25:
            aus.append("lip tightener")
        if (
            present_horizontal("AU20_corner_subR", active=0.05, mag=0.04)
            and present_horizontal("AU20_corner_subL", active=0.05, mag=0.04)
            and lip_stretcher_geom > 0.04
        ) or lip_stretcher_geom > 0.18:
            aus.append("lip stretcher")
    else:
        if lip_center and lip_center.get("active", 0) >= 0.05 and lip_center.get("mag", 0) >= 0.04:
            angle = lip_center["angle"] % 360
            if angle <= 25 or angle >= 335 or 155 <= angle <= 205:
                aus.append("lip tightener")
        elif float(mouth_metrics.get("mouth_width_ratio", 1.0) or 1.0) > 1.01 and float(mouth_metrics.get("mouth_open_ratio", 1.0) or 1.0) < 1.0:
            aus.append("lip tightener")
        if float(mouth_metrics.get("mouth_width_ratio", 1.0) or 1.0) > 1.04:
            aus.append("lip stretcher")
    if use_samm_geometry and float(mouth_metrics.get("mouth_open_ratio", 1.0) or 1.0) > 1.06:
        aus.append("lips part")
    if use_samm_geometry and float(mouth_metrics.get("mouth_open_ratio", 1.0) or 1.0) > 1.12 and float(mouth_metrics.get("lower_lip_nose_ratio", 1.0) or 1.0) > 1.05:
        aus.append("jaw drop")
    return list(dict.fromkeys(aus))
