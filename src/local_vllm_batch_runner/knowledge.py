from __future__ import annotations

import math
from itertools import combinations
from typing import Any

try:
    from .calibration import load_motion_feature_calibration
except ImportError:
    from calibration import load_motion_feature_calibration

try:
    from .numeric_features import ensure_au_evidence
except ImportError:
    from numeric_features import ensure_au_evidence


DATASET_PRIOR_POLICY = {
    "default": {
        "single_weight": 1.6,
        "pair_weight": 2.2,
        "unsupported_penalty": 0.08,
        "coverage_bonus": 0.02,
        "score_scale": 1.0,
        "min_top_score": 0.15,
        "min_margin": 0.12,
    },
    "casme2": {
        "single_weight": 1.6,
        "pair_weight": 2.2,
        "unsupported_penalty": 0.08,
        "coverage_bonus": 0.02,
        "score_scale": 1.0,
        "min_top_score": 0.15,
        "min_margin": 0.12,
    },
    "samm": {
        "single_weight": 1.15,
        "pair_weight": 1.45,
        "unsupported_penalty": 0.05,
        "coverage_bonus": 0.01,
        "score_scale": 0.65,
        "min_top_score": 0.3,
        "min_margin": 0.28,
    },
}
SAMM_GEOMETRY_EXTRA_AUS = {
    "dimpler",
    "eye closure",
    "lip corner depressor",
    "lip stretcher",
    "lips part",
    "jaw drop",
}
SAMM_NONSMILE_CORNER_DIRECTIONS = {
    "leftward",
    "rightward",
    "downward",
    "downward-left",
    "downward-right",
}


def _normalized_landmark_aus(numeric: dict[str, Any]) -> list[str]:
    numeric = ensure_au_evidence(dict(numeric or {}), "")
    evidence = numeric.get("au_evidence") or {}
    ordered_scores = sorted(
        (
            (
                (name or "").strip().lower(),
                _knowledge_evidence_value(info),
                bool(info.get("present")),
            )
            for name, info in evidence.items()
        ),
        key=lambda item: (item[2], item[1]),
        reverse=True,
    )
    seen = set()
    ordered: list[str] = []
    for name, score, present in ordered_scores:
        if not name or name in seen or (not present and score < 0.85):
            continue
        seen.add(name)
        ordered.append(name)
    for name in numeric.get("landmark_aus") or []:
        key = (name or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _knowledge_evidence_value(info: dict[str, Any] | None) -> float:
    info = info or {}
    score = float(info.get("score", 0.0) or 0.0)
    normalized_support = float(info.get("normalized_support", 0.0) or 0.0)
    value = max(score, 0.8 * normalized_support)

    if not bool(info.get("agent_only")):
        return value

    threshold = max(float(info.get("threshold", 0.0) or 0.0), 0.08)
    combined_support = float(info.get("combined_support", 0.0) or 0.0)
    geometry_support = float(info.get("geometry_support", 0.0) or 0.0)
    raw_support = float(info.get("raw_support", 0.0) or 0.0)
    direction_score = float(info.get("direction_score", 0.0) or 0.0)
    balance_score = float(info.get("balance_score", 0.0) or 0.0)
    reliability = max(
        float(info.get("precision", 0.0) or 0.0),
        float(info.get("f1", 0.0) or 0.0),
    )

    proxy_support = max(combined_support, geometry_support, 0.45 * raw_support)
    if proxy_support <= 0.0:
        return value

    proxy_normalized = min(6.0, proxy_support / threshold)
    proxy_quality = (
        (0.45 + 0.55 * direction_score)
        * (0.55 + 0.45 * balance_score)
        * (0.70 + 0.30 * reliability)
    )
    return max(value, proxy_normalized * proxy_quality)


def _au_evidence_scores(numeric: dict[str, Any]) -> dict[str, float]:
    numeric = ensure_au_evidence(dict(numeric or {}), "")
    scores: dict[str, float] = {}
    for name, info in (numeric.get("au_evidence") or {}).items():
        key = (name or "").strip().lower()
        if not key:
            continue
        value = _knowledge_evidence_value(info)
        if value > 0.0:
            scores[key] = round(value, 4)
    return scores


def _soft_numeric_tokens(numeric: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    eye_ratio = numeric.get("eye_ratio")
    if eye_ratio is not None:
        if float(eye_ratio) > 1.03:
            tokens.append("upper lid raiser")
        elif float(eye_ratio) < 0.97:
            tokens.append("lid tightener")
    return tokens


def _samm_non_smile_corner_retraction(numeric: dict[str, Any]) -> bool:
    mouth_metrics = numeric.get("mouth_metrics") or {}
    regions = numeric.get("landmark_au_regions") or {}
    right = regions.get("AU12_corner_subR") or {}
    left = regions.get("AU12_corner_subL") or {}
    if not right or not left:
        return False

    right_dir = str(right.get("dir") or "").strip().lower()
    left_dir = str(left.get("dir") or "").strip().lower()
    active_min = min(
        float(right.get("active", 0.0) or 0.0),
        float(left.get("active", 0.0) or 0.0),
    )
    mag_max = max(
        float(right.get("mag", 0.0) or 0.0),
        float(left.get("mag", 0.0) or 0.0),
    )
    corner_raise = float(mouth_metrics.get("corner_raise_relative_mean", 0.0) or 0.0)
    width_ratio = float(mouth_metrics.get("mouth_width_ratio", 1.0) or 1.0)
    outward_mean = float(mouth_metrics.get("corner_outward_mean", 0.0) or 0.0)
    backward_mean = float(mouth_metrics.get("corner_backward_mean", 0.0) or 0.0)
    if active_min < 0.9 or mag_max < 0.2:
        return False
    if corner_raise > 0.0016:
        return False
    if (
        right_dir in SAMM_NONSMILE_CORNER_DIRECTIONS
        and left_dir in SAMM_NONSMILE_CORNER_DIRECTIONS
    ):
        return True
    return width_ratio < 0.995 and (outward_mean > 0.0010 or backward_mean > 0.0015)


def _dataset_pattern_observations(
    dataset: str,
    numeric: dict[str, Any],
    evidence_scores: dict[str, float],
) -> dict[str, Any]:
    if dataset != "samm":
        return {
            "notes": [],
            "observations": [],
            "pattern_candidate_aus": [],
        }

    pattern_candidate_aus: list[str] = []
    observations: list[str] = []
    notes: list[str] = []
    mouth_metrics = numeric.get("mouth_metrics") or {}
    non_smile_retraction = _samm_non_smile_corner_retraction(numeric)
    lower_lip_nose_ratio = float(mouth_metrics.get("lower_lip_nose_ratio", 1.0) or 1.0)
    mouth_open_ratio = float(mouth_metrics.get("mouth_open_ratio", 1.0) or 1.0)
    corner_raise = float(mouth_metrics.get("corner_raise_relative_mean", 0.0) or 0.0)
    width_ratio = float(mouth_metrics.get("mouth_width_ratio", 1.0) or 1.0)
    lip_corner_score = float(evidence_scores.get("lip corner puller", 0.0) or 0.0)
    jaw_drop_score = float(evidence_scores.get("jaw drop", 0.0) or 0.0)
    lips_part_score = float(evidence_scores.get("lips part", 0.0) or 0.0)
    upper_lid_score = float(evidence_scores.get("upper lid raiser", 0.0) or 0.0)
    strong_retraction = non_smile_retraction and corner_raise <= 0.0005 and width_ratio < 0.99

    if non_smile_retraction:
        pattern_candidate_aus.append("lip stretcher")
        notes.append("samm_non_smile_corner_retraction")
        observations.append(
            "SAMM data pattern: bilateral corner motion that looks retracted, horizontal, or downward is often closer to AU20 lip stretcher than to AU12 smile-like corner pull."
        )
        if lip_corner_score >= 1.0:
            notes.append("samm_retraction_compare_fear_vs_smile_contempt")
            observations.append(
                "When non-smile corner retraction coexists with AU12-like score, compare fear and other tension/opening interpretations before accepting happiness or contempt."
            )
        if strong_retraction:
            notes.append("samm_strong_retraction_pattern")
            observations.append(
                "SAMM data pattern: strong low-raise corner retraction with mild width compression is a high-risk fear-like lower-face pattern, not a default smile/contempt cue."
            )

    if mouth_open_ratio >= 1.0 and lower_lip_nose_ratio >= 1.03:
        pattern_candidate_aus.append("jaw drop")
        notes.append("samm_jaw_drop_pattern_candidate")
        observations.append(
            "SAMM data pattern: when mouth opening and lower-lip-to-nose expansion are both elevated, compare AU26 jaw drop and AU25 lips part instead of reading the opening as generic smile support."
        )

    if (
        non_smile_retraction
        and (
            jaw_drop_score >= 0.8
            or lips_part_score >= 0.8
            or float(evidence_scores.get("jaw drop", 0.0) or 0.0) >= 0.8
            or upper_lid_score >= 0.8
        )
    ):
        notes.append("samm_fear_signature_pattern")
        observations.append(
            "SAMM data pattern: corner retraction together with opening or upper-lid widening often marks fear-like clips; ask whether the local evidence supports tension/retraction better than contempt."
        )

    return {
        "notes": list(dict.fromkeys(notes)),
        "observations": list(dict.fromkeys(observations)),
        "pattern_candidate_aus": list(dict.fromkeys(pattern_candidate_aus)),
    }


def derive_knowledge_prior(dataset: str, numeric: dict[str, Any]) -> dict[str, Any]:
    numeric = ensure_au_evidence(dict(numeric or {}), dataset)
    if not numeric or not numeric.get("available"):
        return {
            "available": False,
            "suggested_emotion": "",
            "suggested_aus": [],
            "scores": {},
            "notes": ["numeric_unavailable"],
        }

    calibration = load_motion_feature_calibration()
    dataset_calibration = (calibration.get("datasets") or {}).get(dataset, {})
    if not dataset_calibration:
        return {
            "available": False,
            "suggested_emotion": "",
            "suggested_aus": _normalized_landmark_aus(numeric),
            "scores": {},
            "notes": ["calibration_unavailable"],
        }

    landmark_aus = _normalized_landmark_aus(numeric)
    evidence_scores = _au_evidence_scores(numeric)
    observed_aus = [
        name
        for name, score in sorted(
            evidence_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if score >= 0.55
    ]
    if not observed_aus:
        observed_aus = landmark_aus[:]
    for token in _soft_numeric_tokens(numeric):
        if token not in observed_aus:
            observed_aus.append(token)
        evidence_scores[token] = max(evidence_scores.get(token, 0.0), 1.0)
    dataset_patterns = _dataset_pattern_observations(
        dataset,
        numeric,
        evidence_scores,
    )

    if len(observed_aus) > 6:
        observed_aus = observed_aus[:6]

    emotion_au_rates = dataset_calibration.get("emotion_au_rates") or {}
    emotion_pair_rates = dataset_calibration.get("emotion_pair_rates") or {}
    emotions = sorted(emotion_au_rates.keys())
    globally_supported_aus = {
        au_name
        for rates in emotion_au_rates.values()
        for au_name in (rates or {}).keys()
    }
    scores: dict[str, float] = {}
    policy = {
        **DATASET_PRIOR_POLICY["default"],
        **DATASET_PRIOR_POLICY.get(dataset, {}),
    }
    notes = ["likelihood_prior_from_calibration", f"prior_policy={dataset or 'default'}"]
    notes.extend(dataset_patterns.get("notes") or [])

    for emotion in emotions:
        single_rates = emotion_au_rates.get(emotion) or {}
        pair_rates = emotion_pair_rates.get(emotion) or {}
        score = 0.0

        for au_name in observed_aus:
            rate = float(single_rates.get(au_name, 0.0) or 0.0)
            strength = max(0.35, float(evidence_scores.get(au_name, 0.0) or 0.0))
            if rate > 0:
                score += float(policy["single_weight"]) * rate * strength
            elif not (
                dataset == "samm"
                and au_name in SAMM_GEOMETRY_EXTRA_AUS
                and au_name not in globally_supported_aus
            ):
                # Light penalty for unsupported observed AUs so the prior stays soft.
                score -= float(policy["unsupported_penalty"]) * min(1.25, 0.65 * strength)

        for pair in combinations(sorted(set(observed_aus)), 2):
            rate = float(pair_rates.get("|".join(pair), 0.0) or 0.0)
            if rate > 0:
                pair_strength = math.sqrt(
                    max(float(evidence_scores.get(pair[0], 0.0) or 0.0), 0.05)
                    * max(float(evidence_scores.get(pair[1], 0.0) or 0.0), 0.05)
                )
                score += float(policy["pair_weight"]) * rate * pair_strength

        # Tiny prior toward emotions that have better AU coverage in the calibration set.
        score += float(policy["coverage_bonus"]) * len(single_rates)
        score *= float(policy["score_scale"])
        scores[emotion] = round(score, 3)

    suggested_emotion = ""
    score_margin = 0.0
    if scores:
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_name, top_score = ordered[0]
        next_score = ordered[1][1] if len(ordered) > 1 else 0.0
        score_margin = top_score - next_score
        if top_score > float(policy["min_top_score"]) and score_margin >= float(policy["min_margin"]):
            suggested_emotion = top_name

    suggested_aus = observed_aus
    if suggested_emotion:
        rates = emotion_au_rates.get(suggested_emotion) or {}
        prioritized = sorted(
            observed_aus,
            key=lambda name: (
                float(rates.get(name, 0.0) or 0.0) * float(evidence_scores.get(name, 0.0) or 0.0),
                name in landmark_aus,
                float(evidence_scores.get(name, 0.0) or 0.0),
            ),
            reverse=True,
        )
        suggested_aus = []
        for name in prioritized:
            if name not in suggested_aus:
                suggested_aus.append(name)

    return {
        "available": True,
        "suggested_emotion": suggested_emotion,
        "suggested_aus": suggested_aus,
        "evidence_scores": {name: round(score, 3) for name, score in evidence_scores.items()},
        "scores": scores,
        "score_margin": round(score_margin, 3),
        "notes": notes,
        "dataset_observations": list(dataset_patterns.get("observations") or []),
        "pattern_candidate_aus": list(dataset_patterns.get("pattern_candidate_aus") or []),
    }
