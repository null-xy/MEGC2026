from __future__ import annotations

import json
import math
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any

try:
    from .config import ROOT_DIR, get_active_ablation
except ImportError:
    from config import ROOT_DIR, get_active_ablation


CALIBRATION_PATH = ROOT_DIR / "outputs" / "casme2_samm_motion_features.jsonl"
_ACTIVE_CALIBRATION_PATH: Path | None = None
_ACTIVE_EXCLUDED_SUBJECTS: tuple[tuple[str, tuple[str, ...]], ...] = ()

AU_REGION_MAP = {
    "inner brow raiser": ["AU1_inner_brow_subR", "AU1_inner_brow_subL"],
    "outer brow raiser": ["AU2_brow_outer_subR", "AU2_brow_outer_subL"],
    "brow lowerer": ["AU4_brow_lowerer"],
    "cheek raiser": ["AU6_cheek_subR", "AU6_cheek_subL"],
    "nose wrinkler": ["AU9_nose_bridge"],
    "upper lip raiser": ["AU10_upper_lip"],
    "lip corner puller": ["AU12_corner_subR", "AU12_corner_subL"],
    "lip corner depressor": ["AU20_corner_subR", "AU20_corner_subL"],
    "lip stretcher": ["AU20_corner_subR", "AU20_corner_subL"],
    "chin raiser": ["AU17_chin"],
    "lip tightener": ["AU23_lip_center"],
    "jaw drop": ["AU26_jaw"],
}
UPWARD_RANGES = [(40.0, 130.0)]
DOWNWARD_RANGES = [(215.0, 325.0)]
HORIZONTAL_RANGES = [(0.0, 30.0), (150.0, 210.0), (330.0, 360.0)]
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
    "jaw drop": DOWNWARD_RANGES,
}

SUPPORTED_AUS = [
    "inner brow raiser",
    "outer brow raiser",
    "brow lowerer",
    "upper lid raiser",
    "lid tightener",
    "cheek raiser",
    "nose wrinkler",
    "upper lip raiser",
    "lip corner puller",
    "chin raiser",
    "lip tightener",
]
SAMM_EXTRA_SUPPORTED_AUS = [
    "dimpler",
    "eye closure",
    "lip corner depressor",
    "lip stretcher",
    "lips part",
    "jaw drop",
]
AU_GEOMETRY_PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "chin raiser": {
        "datasets": ["samm"],
        "source": "mouth_metrics",
        "features": [
            "mouth_center_raise_norm",
            "lower_lip_raise_norm",
            "chin_raise_norm",
            "upper_lip_raise_norm",
            "upper_lip_nose_ratio",
            "lower_lip_nose_ratio",
            "mouth_center_drop_norm",
            "corner_drop_mean",
        ],
        "min_positive": 5,
        "min_negative": 20,
        "min_effect": 0.35,
        "top_k": 8,
    },
    "lip corner puller": {
        "datasets": ["samm"],
        "source": "mouth_metrics",
        "features": [
            "corner_raise_relative_mean",
            "corner_raise_mean",
            "corner_lift_mean",
            "corner_outward_mean",
            "mouth_open_ratio",
            "corner_drop_mean",
            "mouth_center_raise_norm",
            "lower_lip_raise_norm",
            "corner_forward_mean",
        ],
        "min_positive": 8,
        "min_negative": 24,
        "min_effect": 0.25,
        "top_k": 9,
    },
    "dimpler": {
        "datasets": ["samm"],
        "source": "mouth_metrics",
        "features": [
            "corner_drop_mean",
            "corner_depress_mean",
            "mouth_center_drop_norm",
            "corner_lift_mean",
            "corner_asymmetry",
            "abs_corner_lift_diff",
            "abs_corner_raise_diff",
            "abs_corner_raise_relative_diff",
            "corner_inward_mean",
            "upper_lip_nose_ratio",
            "corner_raise_mean",
            "lower_lip_drop_norm",
            "lower_lip_nose_ratio",
            "corner_raise_relative_mean",
        ],
        "min_positive": 8,
        "min_negative": 24,
        "min_effect": 0.25,
        "top_k": 12,
    },
    "lip stretcher": {
        "datasets": ["samm"],
        "source": "mouth_metrics",
        "features": [
            "corner_drop_mean",
            "mouth_width_ratio",
            "corner_outward_mean",
            "mouth_center_raise_norm",
            "lower_lip_raise_norm",
            "mouth_center_drop_norm",
            "lower_lip_nose_ratio",
            "lower_lip_drop_norm",
            "corner_inward_mean",
            "corner_raise_mean",
        ],
        "min_positive": 5,
        "min_negative": 24,
        "min_effect": 0.25,
        "top_k": 10,
    },
}
PROFILE_ONLY_AUS = frozenset(AU_GEOMETRY_PROFILE_SPECS.keys())


_DERIVED_PROFILE_FEATURES: dict[str, tuple[str, ...]] = {
    "abs_corner_lift_diff": ("left_corner_lift", "right_corner_lift"),
    "abs_corner_raise_diff": ("left_corner_raise", "right_corner_raise"),
    "abs_corner_raise_relative_diff": ("left_corner_raise_relative", "right_corner_raise_relative"),
    "abs_corner_inward_diff": ("left_corner_inward", "right_corner_inward"),
    "abs_corner_drop_diff": ("left_corner_drop", "right_corner_drop"),
    "abs_nostril_raise_diff": ("left_nostril_raise_norm", "right_nostril_raise_norm"),
}


def supported_aus_for_dataset(dataset: str) -> list[str]:
    if dataset == "samm":
        return [*SUPPORTED_AUS, *SAMM_EXTRA_SUPPORTED_AUS]
    return SUPPORTED_AUS[:]


def _feature_default(name: str) -> float:
    if name in _DERIVED_PROFILE_FEATURES:
        return 0.0
    return 1.0 if "ratio" in name else 0.0


def _profile_enabled_for_dataset(dataset: str, au_name: str) -> bool:
    spec = AU_GEOMETRY_PROFILE_SPECS.get(au_name) or {}
    allowed_datasets = spec.get("datasets") or []
    return not allowed_datasets or dataset in allowed_datasets


def _derived_profile_value(source: dict[str, Any], name: str) -> float | None:
    deps = _DERIVED_PROFILE_FEATURES.get(name)
    if not deps:
        return None
    if len(deps) != 2:
        return None
    left_name, right_name = deps
    left = source.get(left_name, None)
    right = source.get(right_name, None)
    if left is None:
        left = _feature_default(left_name)
    if right is None:
        right = _feature_default(right_name)
    return abs(float(left) - float(right))


def _profile_value(source: dict[str, Any], name: str) -> float:
    derived = _derived_profile_value(source, name)
    if derived is not None:
        return derived
    value = source.get(name, None)
    if value is None:
        return _feature_default(name)
    return float(value)


def _build_geometry_profile(rows: list[dict[str, Any]], *, dataset: str, au_name: str) -> dict[str, Any] | None:
    spec = AU_GEOMETRY_PROFILE_SPECS.get(au_name) or None
    if not spec:
        return None
    if not _profile_enabled_for_dataset(dataset, au_name):
        return None
    positives = [row for row in rows if au_name in (row.get("gt_au_names") or [])]
    negatives = [row for row in rows if au_name not in (row.get("gt_au_names") or [])]
    if len(positives) < int(spec.get("min_positive", 5) or 5):
        return None
    if len(negatives) < int(spec.get("min_negative", 20) or 20):
        return None

    source_name = str(spec.get("source") or "")
    feature_keys = list(spec.get("features") or [])
    if not source_name or not feature_keys:
        return None

    features: list[dict[str, Any]] = []
    min_effect = float(spec.get("min_effect", 0.25) or 0.25)
    for key in feature_keys:
        pos_values = [_profile_value(row.get(source_name) or {}, key) for row in positives]
        neg_values = [_profile_value(row.get(source_name) or {}, key) for row in negatives]
        pos_mean = sum(pos_values) / len(pos_values)
        neg_mean = sum(neg_values) / len(neg_values)
        pos_var = sum((value - pos_mean) ** 2 for value in pos_values) / max(1, len(pos_values) - 1)
        neg_var = sum((value - neg_mean) ** 2 for value in neg_values) / max(1, len(neg_values) - 1)
        pooled_std = math.sqrt(max((pos_var + neg_var) / 2.0, 0.0))
        if pooled_std <= 1e-6:
            continue
        effect = (pos_mean - neg_mean) / pooled_std
        if abs(effect) < min_effect:
            continue
        features.append(
            {
                "name": key,
                "default": _feature_default(key),
                "pos_mean": round(pos_mean, 6),
                "neg_mean": round(neg_mean, 6),
                "midpoint": round((pos_mean + neg_mean) / 2.0, 6),
                "scale": round(max(pooled_std, abs(pos_mean - neg_mean) / 2.0, 1e-4), 6),
                "direction": 1.0 if pos_mean >= neg_mean else -1.0,
                "weight": round(min(abs(effect), 2.5), 6),
                "effect": round(effect, 6),
            }
        )

    features.sort(key=lambda item: abs(float(item.get("effect", 0.0) or 0.0)), reverse=True)
    features = features[: int(spec.get("top_k", len(features)) or len(features))]
    if len(features) < 3:
        return None
    return {
        "kind": "centroid_margin_v1",
        "source": source_name,
        "dataset": dataset,
        "au_name": au_name,
        "features": features,
    }


def compute_geometry_profile_support(
    profile: dict[str, Any] | None,
    *,
    mouth_metrics: dict[str, Any] | None = None,
    eye_metrics: dict[str, Any] | None = None,
) -> tuple[float, float]:
    if not profile:
        return 0.0, 0.0
    source_name = str(profile.get("source") or "")
    if source_name == "mouth_metrics":
        source = mouth_metrics or {}
    elif source_name == "eye_metrics":
        source = eye_metrics or {}
    else:
        source = {}

    features = profile.get("features") or []
    if not features:
        return 0.0, 0.0

    weighted_margin = 0.0
    total_weight = 0.0
    positive_weight = 0.0
    for feature in features:
        key = str(feature.get("name") or "")
        if not key:
            continue
        default = float(feature.get("default", _feature_default(key)) or _feature_default(key))
        midpoint = float(feature.get("midpoint", 0.0) or 0.0)
        scale = max(float(feature.get("scale", 1.0) or 1.0), 1e-6)
        direction = float(feature.get("direction", 1.0) or 1.0)
        weight = max(float(feature.get("weight", 0.0) or 0.0), 0.0)
        if weight <= 0.0:
            continue
        raw_value = source.get(key, None)
        value = default if raw_value is None else float(raw_value)
        margin = direction * (value - midpoint) / scale
        margin = max(-2.5, min(2.5, margin))
        contribution = weight * margin
        weighted_margin += contribution
        total_weight += weight
        if contribution > 0.0:
            positive_weight += weight

    if total_weight <= 0.0:
        return 0.0, 0.0
    support = max(0.0, weighted_margin / total_weight)
    balance = positive_weight / total_weight
    return support, balance


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
    if not info or not ranges:
        return 1.0
    active = float(info.get("active", 0.0) or 0.0)
    mag = float(info.get("mag", 0.0) or 0.0)
    if active <= 0.0 or mag <= 0.0:
        return 0.0
    angle = float(info.get("angle", 0.0) or 0.0) % 360.0
    dist = _angle_distance_to_ranges(angle, ranges)
    if dist <= 0.0:
        return 1.0
    return max(0.0, 1.0 - dist / 90.0)


def _region_score(row: dict[str, Any], region: str, ranges: list[tuple[float, float]] | None = None) -> float:
    score = 0.0
    for source, multiplier in (
        (row.get("au_regions_head_motion_removed_similarity") or {}, 1.0),
        (row.get("au_regions_raw") or {}, 0.7),
        (row.get("au_landmark_residual_similarity") or {}, 18.0),
    ):
        info = source.get(region) or {}
        if not info:
            continue
        direction = _direction_compatibility(info, ranges or [])
        score = max(
            score,
            multiplier
            * float(info.get("active", 0.0) or 0.0)
            * float(info.get("mag", 0.0) or 0.0)
            * direction,
        )
    return score


def calibration_score(
    row: dict[str, Any],
    au_name: str,
    *,
    geometry_profiles: dict[str, Any] | None = None,
) -> float:
    values = [_region_score(row, region, AU_DIRECTION_RANGES.get(au_name, [])) for region in AU_REGION_MAP.get(au_name, [])]
    values = [value for value in values if value > 0.0]

    eye_ratio = row.get("eye_height_ratio")
    eye_metrics = row.get("eye_metrics") or {}
    mouth_metrics = row.get("mouth_metrics") or {}

    mean_ratio = eye_metrics.get("mean_ratio", eye_ratio)
    min_ratio = eye_metrics.get("min_ratio", mean_ratio)
    max_ratio = eye_metrics.get("max_ratio", mean_ratio)
    apex_norm = eye_metrics.get("apex_min_height_norm")
    asymmetry = float(eye_metrics.get("asymmetry", 0.0) or 0.0)
    dataset = (row.get("dataset") or "").strip().lower()
    profile = (geometry_profiles or {}).get(au_name) or None

    if au_name == "inner brow raiser" and dataset == "samm":
        gap_ratio = float(eye_metrics.get("mean_gap_ratio", 1.0) or 1.0)
        inner_raise = float(eye_metrics.get("inner_brow_raise_mean", 0.0) or 0.0)
        relative_raise = float(eye_metrics.get("inner_brow_relative_raise_mean", 0.0) or 0.0)
        forward_relative = float(eye_metrics.get("inner_brow_forward_relative_mean", 0.0) or 0.0)
        brow_drop = float(eye_metrics.get("brow_center_drop_norm", 0.0) or 0.0)
        pose_rmse = float(eye_metrics.get("pose_landmark_fit_rmse_3d", 0.0) or 0.0)
        geom_score = max(0.0, inner_raise - 0.0040) * 18.0
        geom_score += max(0.0, relative_raise - 0.0022) * 170.0
        geom_score += max(0.0, forward_relative - 0.0008) * 52.0
        if relative_raise < 0.0012 and forward_relative < 0.0007:
            geom_score *= 0.28
        geom_score = max(0.0, geom_score - max(0.0, gap_ratio - 1.03) * 16.0)
        geom_score = max(0.0, geom_score - brow_drop * 42.0)
        geom_score = max(0.0, geom_score - max(0.0, pose_rmse - 2.2) * 0.08)
        flow_score = min(values[:2]) if len(values) >= 2 else (values[0] if values else 0.0)
        if geom_score > 0.0:
            return max(0.0, 0.12 * flow_score + 1.08 * geom_score)
        return max(0.0, 0.10 * flow_score)

    if au_name == "outer brow raiser" and dataset == "samm":
        outer_raise = float(eye_metrics.get("outer_brow_raise_mean", 0.0) or 0.0)
        relative_raise = float(eye_metrics.get("outer_brow_relative_raise_mean", 0.0) or 0.0)
        forward_relative = float(eye_metrics.get("outer_brow_forward_relative_mean", 0.0) or 0.0)
        brow_drop = float(eye_metrics.get("brow_center_drop_norm", 0.0) or 0.0)
        inner_raise = float(eye_metrics.get("inner_brow_relative_raise_mean", 0.0) or 0.0)
        pose_rmse = float(eye_metrics.get("pose_landmark_fit_rmse_3d", 0.0) or 0.0)
        geom_score = max(0.0, outer_raise - 0.0040) * 18.0
        geom_score += max(0.0, relative_raise - 0.0022) * 160.0
        geom_score += max(0.0, forward_relative - 0.0008) * 46.0
        geom_score = max(0.0, geom_score - max(0.0, inner_raise - relative_raise) * 42.0)
        geom_score = max(0.0, geom_score - brow_drop * 36.0)
        if relative_raise < 0.0011 and forward_relative < 0.0007:
            geom_score *= 0.30
        geom_score = max(0.0, geom_score - max(0.0, pose_rmse - 2.2) * 0.08)
        flow_score = min(values[:2]) if len(values) >= 2 else (values[0] if values else 0.0)
        if geom_score > 0.0:
            return max(0.0, 0.15 * flow_score + 1.05 * geom_score)
        return max(0.0, 0.08 * flow_score)

    if au_name == "brow lowerer" and dataset == "samm":
        gap_shrink = float(eye_metrics.get("mean_gap_shrink", 0.0) or 0.0)
        inner_drop = float(eye_metrics.get("inner_brow_relative_drop_mean", 0.0) or 0.0)
        outer_drop = float(eye_metrics.get("outer_brow_relative_drop_mean", 0.0) or 0.0)
        brow_drop = float(eye_metrics.get("brow_center_drop_norm", 0.0) or 0.0)
        pose_rmse = float(eye_metrics.get("pose_landmark_fit_rmse_3d", 0.0) or 0.0)
        geom_score = max(0.0, gap_shrink - 0.008) * 28.0
        geom_score += max(0.0, inner_drop - 0.0022) * 150.0
        geom_score += max(0.0, outer_drop - 0.0018) * 92.0
        geom_score += max(0.0, brow_drop - 0.0025) * 120.0
        geom_score = max(0.0, geom_score - max(0.0, float(mean_ratio or 1.0) - 1.02) * 18.0)
        geom_score = max(0.0, geom_score - max(0.0, pose_rmse - 2.6) * 0.05)
        flow_score = values[0] if values else 0.0
        if geom_score > 0.0:
            return max(0.0, 0.25 * flow_score + 1.00 * geom_score)
        return max(0.0, 0.22 * flow_score)

    if au_name == "upper lid raiser" and mean_ratio is not None:
        if dataset == "samm":
            gap_shrink = float(eye_metrics.get("mean_gap_shrink", 0.0) or 0.0)
            pose_rmse = float(eye_metrics.get("pose_landmark_fit_rmse_3d", 0.0) or 0.0)
            if float(mean_ratio) > 1.18 or float(max_ratio or mean_ratio) > 1.22 or asymmetry > 0.35 or pose_rmse > 3.0:
                score = 0.0
            else:
                score = max(0.0, float(mean_ratio) - 1.023) * 13.0
                if max_ratio is not None:
                    score += max(0.0, float(max_ratio) - 1.030) * 9.0
                    score += max(
                        0.0,
                        min(float(max_ratio), 1.20) - max(float(mean_ratio), 1.0) - 0.004,
                    ) * 4.0
                score = max(0.0, score - max(0.0, gap_shrink - 0.035) * 3.5)
                score = max(0.0, score - max(0.0, gap_shrink - 0.065) * 9.0)
                score = max(0.0, score - max(0.0, pose_rmse - 2.4) * 0.08)
        else:
            score = max(0.0, float(mean_ratio) - 1.03) * 8.0
            if max_ratio is not None:
                score += max(0.0, float(max_ratio) - 1.05) * 4.0
        values.append(score)
    if au_name == "lid tightener" and mean_ratio is not None:
        narrowing = max(0.0, 0.98 - float(mean_ratio)) * 8.5
        closure_penalty = 0.0
        if min_ratio is not None:
            closure_penalty += max(0.0, 0.86 - float(min_ratio)) * 9.0
        if apex_norm is not None:
            closure_penalty += max(0.0, 0.032 - float(apex_norm)) * 32.0
        values.append(max(0.0, narrowing - 0.45 * closure_penalty) * max(0.0, 1.0 - asymmetry))
    if au_name == "eye closure":
        score = 0.0
        if min_ratio is not None:
            score += max(0.0, 0.90 - float(min_ratio)) * 12.0
        if apex_norm is not None:
            score += max(0.0, 0.034 - float(apex_norm)) * 42.0
        elif eye_ratio is not None:
            score = max(score, max(0.0, 0.90 - float(eye_ratio)) * 12.0)
        values.append(score * max(0.0, 1.0 - asymmetry))

    if au_name == "nose wrinkler" and dataset == "samm":
        nostril_raise = float(mouth_metrics.get("nostril_raise_mean", 0.0) or 0.0)
        bridge_shrink = float(mouth_metrics.get("nose_bridge_shrink_norm", 0.0) or 0.0)
        upper_lip_raise = float(mouth_metrics.get("upper_lip_raise_relative_norm", 0.0) or 0.0)
        upper_lip_raise_abs = float(mouth_metrics.get("upper_lip_raise_norm", 0.0) or 0.0)
        upper_lip_forward = float(mouth_metrics.get("upper_lip_forward_norm", 0.0) or 0.0)
        center_raise = float(mouth_metrics.get("mouth_center_raise_norm", 0.0) or 0.0)
        corner_raise = float(mouth_metrics.get("corner_raise_relative_mean", 0.0) or 0.0)
        geom_score = max(0.0, nostril_raise - 0.0010) * 54.0
        geom_score += max(0.0, bridge_shrink - 0.0010) * 56.0
        geom_score += max(0.0, upper_lip_raise - 0.0012) * 20.0
        geom_score += max(0.0, upper_lip_raise_abs - 0.0018) * 10.0
        geom_score += max(0.0, upper_lip_forward - 0.0008) * 8.0
        geom_score = max(0.0, geom_score - max(0.0, center_raise - 0.0020) * 22.0)
        geom_score = max(0.0, geom_score - max(0.0, corner_raise - 0.0020) * 14.0)
        flow_score = values[0] if values else 0.0
        if geom_score > 0.0:
            return max(0.0, 0.70 * flow_score + 0.85 * geom_score)
        return max(0.0, 0.80 * flow_score)

    if au_name == "upper lip raiser" and dataset == "samm":
        ratio = float(mouth_metrics.get("upper_lip_nose_ratio", 1.0) or 1.0)
        raise_norm = float(mouth_metrics.get("upper_lip_raise_relative_norm", 0.0) or 0.0)
        forward = float(mouth_metrics.get("upper_lip_forward_norm", 0.0) or 0.0)
        bridge_shrink = float(mouth_metrics.get("nose_bridge_shrink_norm", 0.0) or 0.0)
        nostril_raise = float(mouth_metrics.get("nostril_raise_mean", 0.0) or 0.0)
        corner_raise = float(mouth_metrics.get("corner_raise_relative_mean", 0.0) or 0.0)
        corner_lift = float(mouth_metrics.get("corner_lift_mean", 0.0) or 0.0)
        geom_score = max(0.0, raise_norm - 0.0014) * 58.0
        geom_score += max(0.0, 0.998 - ratio) * 18.0
        geom_score += max(0.0, forward - 0.0008) * 12.0
        geom_score += max(0.0, bridge_shrink - 0.0012) * 8.0
        geom_score = max(0.0, geom_score - max(0.0, nostril_raise - 0.0014) * 10.0)
        geom_score = max(0.0, geom_score - max(0.0, corner_raise - 0.0025) * 14.0)
        geom_score = max(0.0, geom_score - max(0.0, corner_lift - 0.0040) * 10.0)
        flow_score = values[0] if values else 0.0
        if geom_score > 0.0:
            return max(0.0, 0.65 * flow_score + 0.90 * geom_score)
        return max(0.0, 0.75 * flow_score)

    if au_name == "upper lip raiser":
        ratio = mouth_metrics.get("upper_lip_nose_ratio")
        if ratio is not None:
            values.append(max(0.0, 0.98 - float(ratio)) * 12.0)

    if au_name in PROFILE_ONLY_AUS and _profile_enabled_for_dataset(dataset, au_name):
        if profile:
            support, _ = compute_geometry_profile_support(profile, mouth_metrics=mouth_metrics, eye_metrics=eye_metrics)
            return support
        return 0.0

    if au_name == "lip corner puller":
        if dataset == "samm":
            raise_mean = mouth_metrics.get("corner_raise_relative_mean")
            lift_mean = mouth_metrics.get("corner_lift_mean")
            outward_mean = float(mouth_metrics.get("corner_outward_mean", 0.0) or 0.0)
            center_raise = float(mouth_metrics.get("mouth_center_raise_norm", 0.0) or 0.0)
            lower_raise = float(mouth_metrics.get("lower_lip_raise_norm", 0.0) or 0.0)
            chin_raise = float(mouth_metrics.get("chin_raise_norm", 0.0) or 0.0)
            center_forward = float(mouth_metrics.get("mouth_center_forward_norm", 0.0) or 0.0)
            corner_forward = float(mouth_metrics.get("corner_forward_mean", 0.0) or 0.0)
            if raise_mean is not None:
                balance = min(
                    float(mouth_metrics.get("corner_raise_relative_balance", 0.0) or 0.0),
                    max(float(mouth_metrics.get("corner_lift_balance", 0.0) or 0.0), 0.6),
                )
                score = max(0.0, float(raise_mean) - 0.0015) * 62.0
                if lift_mean is not None:
                    score += max(0.0, float(lift_mean) - 0.0035) * 18.0
                score += max(0.0, outward_mean - 0.0025) * 10.0
                score += max(0.0, corner_forward - 0.5 * center_forward - 0.0008) * 14.0
                score = max(0.0, score - max(0.0, center_raise - 0.0025) * 26.0)
                score = max(0.0, score - max(0.0, lower_raise - 0.0030) * 20.0)
                score = max(0.0, score - max(0.0, chin_raise - 0.0025) * 16.0)
                flow_score = min(values[:2]) if len(values) >= 2 else (values[0] if values else 0.0)
                return max(0.0, 0.45 * flow_score + 0.80 * score * max(balance, 0.6))
        raise_mean = mouth_metrics.get("corner_raise_mean")
        lift_mean = mouth_metrics.get("corner_lift_mean")
        if raise_mean is not None:
            balance = float(mouth_metrics.get("corner_raise_balance", 0.0) or 0.0)
            score = max(0.0, float(raise_mean) - 0.004) * 55.0
            if lift_mean is not None:
                score += max(0.0, float(lift_mean) - 0.003) * 18.0
            values.append(score * max(balance, 0.6))
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
            score = (
                core_inward * 72.0
                + core_backward * 62.0
                + core_compress * 18.0
            ) * (0.58 + 0.42 * min(max(inward_balance, 0.0), max(backward_balance, 0.0)))
            if core_inward > 0.0 or core_backward > 0.0:
                score += max(0.0, lift_mean - 0.0040) * 8.0
                score += max(0.0, raise_mean - 0.0045) * 4.0
            else:
                score *= 0.12

            score = max(0.0, score - max(0.0, outward_mean - 0.0012) * 48.0)
            score = max(0.0, score - max(0.0, raise_mean - 0.0048) * 24.0)
            score = max(0.0, score - max(0.0, lift_mean - 0.0065) * 14.0)
            score = max(0.0, score - max(0.0, width_ratio - 1.0015) * 18.0)
            score = max(0.0, score - max(0.0, open_ratio - 1.04) * 14.0)
            score = max(0.0, score - max(0.0, lower_drop - 0.012) * 36.0)
            score = max(0.0, score - max(0.0, center_drop - 0.008) * 18.0)
            score = max(0.0, score - drop_mean * 10.0)
            score = max(0.0, score - max(0.0, forward_mean - 0.0012) * 12.0)
            balance = max(
                min(max(inward_balance, 0.0), max(backward_balance, 0.0)),
                max(inward_balance, 0.0) * 0.90,
                max(backward_balance, 0.0) * 0.80,
            )
            values.append(score * max(balance, 0.45))
        else:
            raise_mean = mouth_metrics.get("corner_raise_mean")
            if raise_mean is not None:
                lift_mean = float(mouth_metrics.get("corner_lift_mean", 0.0) or 0.0)
                outward_mean = float(mouth_metrics.get("corner_outward_mean", 0.0) or 0.0)
                open_ratio = float(mouth_metrics.get("mouth_open_ratio", 1.0) or 1.0)
                drop_mean = float(mouth_metrics.get("corner_drop_mean", 0.0) or 0.0)
                asymmetry = float(mouth_metrics.get("corner_asymmetry", 0.0) or 0.0)
                score = max(0.0, float(raise_mean) - 0.0035) * 34.0
                score += max(0.0, lift_mean - 0.003) * 22.0
                score = max(0.0, score - max(0.0, outward_mean - 0.004) * 25.0)
                score = max(0.0, score - max(0.0, open_ratio - 1.03) * 10.0)
                score = max(0.0, score - drop_mean * 18.0)
                values.append(score * max(0.45, 1.0 - asymmetry * 20.0))
    if au_name == "lip corner depressor":
        drop_mean = mouth_metrics.get("corner_drop_mean")
        depress_mean = mouth_metrics.get("corner_depress_mean")
        raise_mean = mouth_metrics.get("corner_raise_mean")
        if drop_mean is not None:
            balance = float(mouth_metrics.get("corner_drop_balance", 0.0) or 0.0)
            score = max(0.0, float(drop_mean) - 0.003) * 45.0
            if depress_mean is not None:
                score += max(0.0, float(depress_mean) - 0.002) * 18.0
            if raise_mean is not None:
                score = max(0.0, score - float(raise_mean) * 20.0)
            values.append(score * max(balance, 0.55))
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
            geom_score = max(0.0, outward_mean - 0.0013) * 36.0
            geom_score += max(0.0, backward_mean - 0.0006) * 16.0
            geom_score += max(0.0, width_ratio - 1.004) * 14.0
            geom_score = max(0.0, geom_score - max(0.0, inward_mean - 0.0010) * 28.0)
            geom_score = max(0.0, geom_score - max(0.0, raise_mean - 0.0018) * 16.0)
            geom_score = max(0.0, geom_score - max(0.0, drop_mean - 0.0055) * 8.0)
            geom_score = max(0.0, geom_score - max(0.0, 1.0 - open_ratio) * 6.0)
            flow_score = min(values[:2]) if len(values) >= 2 else (values[0] if values else 0.0)
            if geom_score > 0.0:
                return max(0.0, 0.55 * flow_score + 0.85 * geom_score * max(balance, 0.4))
            return max(0.0, 0.55 * flow_score)
        outward_mean = mouth_metrics.get("corner_outward_mean")
        if outward_mean is not None:
            balance = float(mouth_metrics.get("corner_outward_balance", 0.0) or 0.0)
            raise_mean = float(mouth_metrics.get("corner_raise_mean", 0.0) or 0.0)
            drop_mean = float(mouth_metrics.get("corner_drop_mean", 0.0) or 0.0)
            score = max(0.0, float(outward_mean) - 0.003) * 50.0
            width_ratio = mouth_metrics.get("mouth_width_ratio")
            if width_ratio is not None:
                score += max(0.0, float(width_ratio) - 1.02) * 8.0
            score = max(0.0, score - max(raise_mean, drop_mean) * 10.0)
            values.append(score * max(balance, 0.55))
    if au_name == "lips part":
        score = 0.0
        open_ratio = mouth_metrics.get("mouth_open_ratio")
        lower_ratio = mouth_metrics.get("lower_lip_nose_ratio")
        lower_drop = mouth_metrics.get("lower_lip_drop_norm")
        if open_ratio is not None:
            score += max(0.0, float(open_ratio) - 1.03) * 6.0
        if lower_ratio is not None:
            score += max(0.0, float(lower_ratio) - 1.015) * 10.0
        if lower_drop is not None:
            score += max(0.0, float(lower_drop) - 0.005) * 22.0
        values.append(score)
    if au_name == "jaw drop":
        open_ratio = mouth_metrics.get("mouth_open_ratio")
        lower_ratio = mouth_metrics.get("lower_lip_nose_ratio")
        lower_drop = mouth_metrics.get("lower_lip_drop_norm")
        score = 0.0
        if open_ratio is not None:
            score += max(0.0, float(open_ratio) - 1.06) * 7.0
        if lower_ratio is not None:
            score += max(0.0, float(lower_ratio) - 1.04) * 10.0
        if lower_drop is not None:
            score += max(0.0, float(lower_drop) - 0.01) * 42.0
        values.append(score)
    if au_name == "lip tightener":
        if dataset == "samm":
            width_ratio = float(mouth_metrics.get("mouth_width_ratio", 1.0) or 1.0)
            open_ratio = float(mouth_metrics.get("mouth_open_ratio", 1.0) or 1.0)
            outward_mean = float(mouth_metrics.get("corner_outward_mean", 0.0) or 0.0)
            inward_mean = float(mouth_metrics.get("corner_inward_mean", 0.0) or 0.0)
            corner_lift = float(mouth_metrics.get("corner_lift_mean", 0.0) or 0.0)
            corner_raise = float(mouth_metrics.get("corner_raise_mean", 0.0) or 0.0)
            corner_drop = float(mouth_metrics.get("corner_drop_mean", 0.0) or 0.0)
            geom_score = 0.0
            if width_ratio <= 0.997 and inward_mean >= 0.0030:
                geom_score = max(0.0, 0.995 - width_ratio) * 18.0
                geom_score += max(0.0, inward_mean - 0.0030) * 10.0
                geom_score += max(0.0, 0.995 - open_ratio) * 6.0
            geom_score = max(0.0, geom_score - max(0.0, outward_mean - 0.0008) * 36.0)
            geom_score = max(0.0, geom_score - corner_lift * 22.0)
            geom_score = max(0.0, geom_score - max(corner_raise, corner_drop) * 18.0)
            if open_ratio > 1.02:
                geom_score = 0.0
            flow_score = values[0] if values else 0.0
            if geom_score > 0.0:
                return max(0.0, 0.25 * flow_score + 0.55 * geom_score)
            return max(0.0, 0.15 * flow_score)
        width_ratio = mouth_metrics.get("mouth_width_ratio")
        open_ratio = mouth_metrics.get("mouth_open_ratio")
        corner_lift = mouth_metrics.get("corner_lift_mean")
        score = 0.0
        if width_ratio is not None:
            score += max(0.0, float(width_ratio) - 1.01) * 4.0
        if open_ratio is not None:
            score += max(0.0, 1.0 - float(open_ratio)) * 6.0
        if corner_lift is not None:
            score = max(0.0, score - float(corner_lift) * 10.0)
        values.append(score)

    if not values:
        return 0.0
    if au_name in {"inner brow raiser", "outer brow raiser", "lip corner puller", "cheek raiser"} and len(values) >= 2:
        top_two = sorted(values, reverse=True)[:2]
        return min(top_two)
    return max(values)


def _best_threshold(
    rows: list[dict[str, Any]],
    au_name: str,
    *,
    geometry_profiles: dict[str, Any] | None = None,
) -> dict[str, float]:
    scores = [calibration_score(row, au_name, geometry_profiles=geometry_profiles) for row in rows]
    candidates = sorted({round(score, 3) for score in scores if score > 0.0})
    if not candidates:
        return {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    best = {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in candidates:
        tp = fp = fn = 0
        for row, score in zip(rows, scores):
            predicted = score >= threshold
            actual = au_name in (row.get("gt_au_names") or [])
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif actual:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if (f1, precision, recall) > (best["f1"], best["precision"], best["recall"]):
            best = {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
    return best


def _normalize_excluded_subjects(
    exclude_subjects: dict[str, Any] | tuple[tuple[str, tuple[str, ...]], ...] | None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not exclude_subjects:
        return ()
    if isinstance(exclude_subjects, tuple):
        normalized = []
        for dataset, subjects in exclude_subjects:
            dataset_key = str(dataset or "").strip().lower()
            if not dataset_key:
                continue
            subject_values = sorted(
                {
                    str(subject or "").strip().lower()
                    for subject in subjects
                    if str(subject or "").strip()
                }
            )
            if subject_values:
                normalized.append((dataset_key, tuple(subject_values)))
        return tuple(sorted(normalized))

    normalized = []
    for dataset, subjects in dict(exclude_subjects).items():
        dataset_key = str(dataset or "").strip().lower()
        if not dataset_key:
            continue
        if isinstance(subjects, (str, bytes)):
            subject_iterable = [subjects]
        else:
            subject_iterable = list(subjects or [])
        subject_values = sorted(
            {
                str(subject or "").strip().lower()
                for subject in subject_iterable
                if str(subject or "").strip()
            }
        )
        if subject_values:
            normalized.append((dataset_key, tuple(subject_values)))
    return tuple(sorted(normalized))


def _excluded_subject_map(
    exclude_subjects: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict[str, set[str]]:
    return {
        dataset: {subject for subject in subjects if subject}
        for dataset, subjects in exclude_subjects
        if dataset and subjects
    }


def _filter_rows_by_excluded_subjects(
    rows: list[dict[str, Any]],
    exclude_subjects: tuple[tuple[str, tuple[str, ...]], ...],
) -> list[dict[str, Any]]:
    excluded = _excluded_subject_map(exclude_subjects)
    if not excluded:
        return rows
    filtered = []
    for row in rows:
        dataset = str(row.get("dataset", "") or "").strip().lower()
        subject = str(row.get("subject", "") or "").strip().lower()
        if dataset and subject and subject in excluded.get(dataset, set()):
            continue
        filtered.append(row)
    return filtered


@contextmanager
def motion_feature_calibration_context(
    *,
    path: str | Path | None = None,
    exclude_subjects: dict[str, Any] | None = None,
):
    global _ACTIVE_CALIBRATION_PATH
    global _ACTIVE_EXCLUDED_SUBJECTS

    previous_path = _ACTIVE_CALIBRATION_PATH
    previous_excluded = _ACTIVE_EXCLUDED_SUBJECTS
    _ACTIVE_CALIBRATION_PATH = Path(path) if path else previous_path
    if exclude_subjects is None:
        _ACTIVE_EXCLUDED_SUBJECTS = previous_excluded
    else:
        _ACTIVE_EXCLUDED_SUBJECTS = _normalize_excluded_subjects(exclude_subjects)
    try:
        yield
    finally:
        _ACTIVE_CALIBRATION_PATH = previous_path
        _ACTIVE_EXCLUDED_SUBJECTS = previous_excluded


@lru_cache(maxsize=64)
def _load_motion_feature_calibration_cached(
    path_str: str,
    mtime_ns: int,
    file_size: int,
    exclude_subjects: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict[str, Any]:
    path = Path(path_str)
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    rows = _filter_rows_by_excluded_subjects(rows, exclude_subjects)

    datasets: dict[str, Any] = {}
    for dataset in sorted({row.get("dataset", "") for row in rows if row.get("dataset")}):
        subset = [row for row in rows if row.get("dataset") == dataset]
        dataset_supported_aus = supported_aus_for_dataset(dataset)
        emotion_counts: dict[str, int] = defaultdict(int)
        emotion_au_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        emotion_pair_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        au_thresholds: dict[str, Any] = {}
        au_geometry_profiles: dict[str, Any] = {}

        for row in subset:
            emotion = (row.get("gt_fine_emotion") or "").strip().lower()
            if not emotion:
                continue
            emotion_counts[emotion] += 1
            observed_aus = [au_name for au_name in row.get("gt_au_names") or [] if au_name in dataset_supported_aus]
            for au_name in observed_aus:
                emotion_au_counts[emotion][au_name] += 1
            for pair in combinations(sorted(set(observed_aus)), 2):
                emotion_pair_counts[emotion]["|".join(pair)] += 1

        for au_name in dataset_supported_aus:
            profile = _build_geometry_profile(subset, dataset=dataset, au_name=au_name)
            if profile:
                au_geometry_profiles[au_name] = profile

        for au_name in dataset_supported_aus:
            metrics = _best_threshold(subset, au_name, geometry_profiles=au_geometry_profiles)
            au_thresholds[au_name] = {
                key: round(float(value), 4) for key, value in metrics.items()
            }

        emotion_au_rates: dict[str, dict[str, float]] = {}
        emotion_pair_rates: dict[str, dict[str, float]] = {}
        for emotion, counts in emotion_au_counts.items():
            denom = max(emotion_counts.get(emotion, 0), 1)
            emotion_au_rates[emotion] = {
                au_name: round(count / denom, 4)
                for au_name, count in counts.items()
            }
            emotion_pair_rates[emotion] = {
                pair_name: round(count / denom, 4)
                for pair_name, count in emotion_pair_counts.get(emotion, {}).items()
            }

        datasets[dataset] = {
            "au_thresholds": au_thresholds,
            "au_geometry_profiles": au_geometry_profiles,
            "emotion_au_rates": emotion_au_rates,
            "emotion_pair_rates": emotion_pair_rates,
            "emotion_counts": dict(emotion_counts),
        }

    return {
        "available": True,
        "path": str(path),
        "file_mtime_ns": int(mtime_ns),
        "file_size": int(file_size),
        "excluded_subjects": {
            dataset: list(subjects)
            for dataset, subjects in exclude_subjects
        },
        "datasets": datasets,
    }


def load_motion_feature_calibration(
    path: str | Path | None = None,
    *,
    exclude_subjects: dict[str, Any] | tuple[tuple[str, tuple[str, ...]], ...] | None = None,
) -> dict[str, Any]:
    calibration_path = Path(path) if path else (_ACTIVE_CALIBRATION_PATH or CALIBRATION_PATH)
    normalized_excluded = (
        _normalize_excluded_subjects(exclude_subjects)
        if exclude_subjects is not None
        else _ACTIVE_EXCLUDED_SUBJECTS
    )
    if not calibration_path.exists():
        return {"available": False, "datasets": {}}
    stat = calibration_path.stat()
    calibration = _load_motion_feature_calibration_cached(
        str(calibration_path),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        normalized_excluded,
    )
    if not get_active_ablation().disable_geometry_calibration:
        return calibration
    datasets = {}
    for dataset, info in (calibration.get("datasets") or {}).items():
        dataset_info = dict(info or {})
        dataset_info["au_thresholds"] = {}
        dataset_info["au_geometry_profiles"] = {}
        datasets[dataset] = dataset_info
    return {
        **calibration,
        "datasets": datasets,
    }
