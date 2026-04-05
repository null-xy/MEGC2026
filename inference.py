from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from .config import AblationConfig, ablation_context, get_active_ablation
except ImportError:
    from config import AblationConfig, ablation_context, get_active_ablation

try:
    from .formatting import (
        DATASET_ALLOWED_EMOTIONS,
        FINE_TO_COARSE,
        au_names_to_pred_string,
        aus_f1,
        aus_mismatch,
        dedupe_keep_order,
        extract_au_names,
        extract_emotion,
        normalize_au_name,
        normalize_emotion_for_dataset,
    )
except ImportError:
    from formatting import (
        DATASET_ALLOWED_EMOTIONS,
        FINE_TO_COARSE,
        au_names_to_pred_string,
        aus_f1,
        aus_mismatch,
        dedupe_keep_order,
        extract_au_names,
        extract_emotion,
        normalize_au_name,
        normalize_emotion_for_dataset,
    )

try:
    from .numeric_features import ensure_au_evidence, get_numeric_features
except ImportError:
    from numeric_features import ensure_au_evidence, get_numeric_features

try:
    from .knowledge import derive_knowledge_prior
except ImportError:
    from knowledge import derive_knowledge_prior

try:
    from .calibration import load_motion_feature_calibration
except ImportError:
    from calibration import load_motion_feature_calibration

try:
    from .local_pipeline import SpatioFunctionalMASPipeline, compute_exact_match as local_compute_exact_match, normalize_answer_text as local_normalize_answer_text
except ImportError:
    from local_pipeline import SpatioFunctionalMASPipeline, compute_exact_match as local_compute_exact_match, normalize_answer_text as local_normalize_answer_text


@dataclass
class BackendConfig:
    model: str = ""
    reasoning_model: str = ""
    gpu_mem: float = 0.90
    reasoning_gpu_mem: float = 0.90
    reasoning_device: str = ""
    reasoning_tensor_parallel_size: int = 1
    vision_device: str = ""
    memory: str = ""
    scope: str = ""
    no_vision: bool = False
    debug: bool = False
    ablation: AblationConfig = field(default_factory=AblationConfig)


class LocalMASBackend:
    def __init__(self, config: BackendConfig):
        self.ablation = config.ablation or AblationConfig()
        model_name = config.model or config.reasoning_model or ""
        self.pipeline = SpatioFunctionalMASPipeline(
            model_name=model_name,
            gpu_memory_utilization=config.gpu_mem,
            reasoning_model_name=config.reasoning_model or "",
            reasoning_gpu_memory_utilization=config.reasoning_gpu_mem,
            reasoning_device=config.reasoning_device or "",
            reasoning_tensor_parallel_size=config.reasoning_tensor_parallel_size,
            vision_device=config.vision_device or "",
            memory_path=config.memory or "",
            scope_guidelines_path=config.scope or "",
            no_vision=config.no_vision,
            debug=config.debug,
            ablation=self.ablation,
        )

    def analyze_batch(self, clip_infos: list[dict]) -> tuple[dict[tuple, dict], dict[tuple, str]]:
        unique_records = []
        question_records = []
        for clip in clip_infos:
            clip_record = {
                "video": clip["clip_id"],
                "video_name": clip["clip_id"],
                "image_path": str(clip["path"]),
                "flow_format": "npy",
                "dataset": clip["dataset"],
            }
            unique_records.append(clip_record)
            for idx, qa in enumerate(clip.get("vqa_questions", []), start=1):
                question_records.append(
                    {
                        "video_id": f"{clip['clip_id']}_{idx}",
                        "video": clip["clip_id"],
                        "video_name": clip["clip_id"],
                        "image_path": str(clip["path"]),
                        "flow_format": "npy",
                        "dataset": clip["dataset"],
                        "question": qa["q"],
                        "answer": qa.get("a", ""),
                    }
                )

        with ablation_context(self.ablation):
            clip_cache = self.pipeline.analyze_of_clips_batch_mas(unique_records)
            for clip in clip_infos:
                key = (clip["clip_id"], str(clip["path"]))
                clip_cache.setdefault(key, {})
                clip_cache[key]["numeric_features"] = get_numeric_features(clip)
            answers = self.pipeline.answer_records_batch(question_records, clip_cache)
        return clip_cache, answers

    def exact_match(self, question: str, gt: str, pred: str) -> bool:
        return local_compute_exact_match(question, gt, pred)


def _pre_fusion_emotion(
    agent_preferred_emotion: str,
    *,
    fine_answer: str,
    analysis_answer: str,
    dataset: str,
    allow_answer_fallback: bool = True,
) -> str:
    preferred = normalize_emotion_for_dataset(agent_preferred_emotion, dataset)
    if preferred and preferred not in _AGENT_UNCERTAIN_EMOTIONS:
        return preferred
    if not allow_answer_fallback:
        return ""
    fallback_text = fine_answer or analysis_answer
    return normalize_emotion_for_dataset(extract_emotion(fallback_text), dataset)


def _numeric_au_evidence_map(numeric: dict[str, Any] | None, dataset: str) -> dict[str, dict[str, Any]]:
    numeric_local = ensure_au_evidence(dict(numeric or {}), dataset)
    evidence: dict[str, dict[str, Any]] = {}
    for name, info in (numeric_local.get("au_evidence") or {}).items():
        key = normalize_au_name(name)
        if key:
            evidence[key] = info or {}
    return evidence


def _ranked_numeric_aus(numeric: dict[str, Any] | None, dataset: str) -> list[str]:
    evidence = _numeric_au_evidence_map(numeric, dataset)
    ranked = sorted(
        evidence.items(),
        key=lambda item: _rank_key(item[1]),
        reverse=True,
    )
    return [
        name
        for name, info in ranked
        if _support_gate(info, threshold=0.75)
    ]


def _evidence_reliability(info: dict[str, Any] | None) -> float:
    info = info or {}
    precision = float(info.get("precision", 0.0) or 0.0)
    f1 = float(info.get("f1", 0.0) or 0.0)
    if precision <= 0.0 and f1 <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 0.55 * precision + 0.45 * f1))


def _samm_specialist_proxy_strength(info: dict[str, Any] | None) -> float:
    info = info or {}
    if not bool(info.get("agent_only")):
        return 0.0
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
        return 0.0
    proxy_normalized = min(6.0, proxy_support / threshold)
    proxy_quality = (
        (0.45 + 0.55 * direction_score)
        * (0.55 + 0.45 * balance_score)
        * (0.70 + 0.30 * reliability)
    )
    return max(0.0, proxy_normalized * proxy_quality)


def _uses_calibrated_reliability_regularization(dataset: str) -> bool:
    return dataset in {"samm", "casme2"}


def _present_gate_enabled() -> bool:
    return not get_active_ablation().disable_present_gate


def _continuous_support(info: dict[str, Any] | None) -> float:
    info = info or {}
    return max(
        float(info.get("score", 0.0) or 0.0),
        float(info.get("normalized_support", 0.0) or 0.0),
        float(info.get("combined_support", 0.0) or 0.0),
        float(info.get("geometry_support", 0.0) or 0.0),
        float(info.get("raw_support", 0.0) or 0.0),
        float(info.get("residual_support", 0.0) or 0.0),
    )


def _present_bonus(info: dict[str, Any] | None, *, value: float = 1.0) -> float:
    if not _present_gate_enabled():
        return 0.0
    return value if bool((info or {}).get("present")) else 0.0


def _support_gate(info: dict[str, Any] | None, *, threshold: float) -> bool:
    info = info or {}
    if _present_gate_enabled() and bool(info.get("present")):
        return True
    return _continuous_support(info) >= threshold


def _override_gate(info: dict[str, Any] | None, *, threshold: float) -> bool:
    info = info or {}
    strength = _agent_override_strength(info)
    if _present_gate_enabled():
        return bool(info.get("present")) and strength >= threshold
    return strength >= threshold


def _rank_key(info: dict[str, Any] | None) -> tuple[float, float, float]:
    info = info or {}
    if _present_gate_enabled():
        return (
            1.0 if bool(info.get("present")) else 0.0,
            float(info.get("score", 0.0) or 0.0),
            float(info.get("normalized_support", 0.0) or 0.0),
        )
    return (
        _continuous_support(info),
        float(info.get("score", 0.0) or 0.0),
        float(info.get("normalized_support", 0.0) or 0.0),
    )


_AGENT_UNCERTAIN_EMOTIONS = {
    "",
    "unknown",
    "other",
    "uncertain",
    "uncertainty",
    "neutral",
    "none",
}

_SAMM_BROW_AGENT_AUS = {
    "inner brow raiser",
    "outer brow raiser",
    "brow lowerer",
}
_SAMM_EYE_AGENT_AUS = {
    "upper lid raiser",
    "lid tightener",
    "eye closure",
}
_SAMM_MOUTH_AGENT_AUS = {
    "nose wrinkler",
    "upper lip raiser",
    "lip corner puller",
    "dimpler",
    "lip corner depressor",
    "lip stretcher",
    "chin raiser",
    "lip tightener",
    "lips part",
    "jaw drop",
}


def _strip_samm_specialist_aux_aus(
    names: list[str] | set[str] | tuple[str, ...] | None,
    *,
    dataset: str,
    agent_trace: dict[str, Any] | None = None,
) -> list[str]:
    cleaned = dedupe_keep_order(list(names or []))
    if dataset != "samm":
        return cleaned
    agent_guidance = _agent_guidance(agent_trace, dataset)
    owned = set()
    if agent_guidance.get("brow_agent_present"):
        owned |= _SAMM_BROW_AGENT_AUS
    if agent_guidance.get("eye_agent_present"):
        owned |= _SAMM_EYE_AGENT_AUS
    if agent_guidance.get("mouth_agent_present"):
        owned |= _SAMM_MOUTH_AGENT_AUS
    if not owned:
        return cleaned
    return [name for name in cleaned if normalize_au_name(name) not in owned]


def _sanitize_knowledge_prior_for_specialists(
    knowledge_prior: dict[str, Any],
    *,
    dataset: str,
    agent_trace: dict[str, Any] | None = None,
    numeric: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if dataset != "samm":
        return knowledge_prior
    agent_guidance = _agent_guidance(agent_trace, dataset)
    if not (
        agent_guidance.get("brow_agent_present")
        or agent_guidance.get("eye_agent_present")
        or agent_guidance.get("mouth_agent_present")
    ):
        return knowledge_prior
    owned = set()
    if agent_guidance.get("brow_agent_present"):
        owned |= _SAMM_BROW_AGENT_AUS
    if agent_guidance.get("eye_agent_present"):
        owned |= _SAMM_EYE_AGENT_AUS
    if agent_guidance.get("mouth_agent_present"):
        owned |= _SAMM_MOUTH_AGENT_AUS
    sanitized = dict(knowledge_prior or {})
    sanitized["suggested_aus"] = _strip_samm_specialist_aux_aus(
        sanitized.get("suggested_aus") or [],
        dataset=dataset,
        agent_trace=agent_trace,
    )
    if not owned or not numeric:
        return sanitized

    sanitized_numeric = dict(numeric or {})
    sanitized_numeric["landmark_aus"] = [
        name
        for name in (numeric.get("landmark_aus") or [])
        if normalize_au_name(name) not in owned
    ]
    sanitized_numeric["au_evidence"] = {
        name: info
        for name, info in (numeric.get("au_evidence") or {}).items()
        if normalize_au_name(name) not in owned
    }
    return derive_knowledge_prior(dataset, sanitized_numeric)


def _agent_guidance(agent_trace: dict[str, Any] | None, dataset: str) -> dict[str, Any]:
    trace = agent_trace or {}
    brow_trace = trace.get("brow_agent") or {}
    eye_trace = trace.get("eye_agent") or {}
    mouth_trace = trace.get("mouth_agent") or {}
    brow_arbiter_trace = trace.get("brow_arbiter") or {}
    eye_arbiter_trace = trace.get("eye_arbiter") or {}
    mouth_arbiter_trace = trace.get("mouth_arbiter") or {}
    motion_trace = trace.get("motion_agent") or {}
    critic_trace = trace.get("critic_agent") or {}
    emotion_trace = trace.get("emotion_agent") or {}
    integrator_trace = trace.get("integrator_agent") or {}

    def canonicalize(names: Any) -> list[str]:
        if not names:
            return []
        if isinstance(names, str):
            names = extract_au_names(names)
        canonical = []
        for name in names:
            if not isinstance(name, str):
                continue
            normalized = normalize_au_name(name)
            if not normalized:
                continue
            if not au_names_to_pred_string([normalized]):
                continue
            canonical.append(normalized)
        return dedupe_keep_order(canonical)

    motion_supported = canonicalize(
        trace.get("motion_supported_aus") or motion_trace.get("supported_aus")
    )
    motion_rejected = canonicalize(
        trace.get("motion_rejected_aus") or motion_trace.get("rejected_aus")
    )
    critic_keep_raw = canonicalize(critic_trace.get("keep_aus"))
    critic_drop_raw = canonicalize(critic_trace.get("drop_aus"))
    critic_keep = critic_keep_raw[:]
    critic_drop = critic_drop_raw[:]
    critic_specialist_fallback = canonicalize(
        trace.get("critic_specialist_fallback_aus")
        or critic_trace.get("specialist_fallback_aus")
        or critic_trace.get("fallback_aus")
        or critic_trace.get("keep_aus")
    )
    integrator_aus = canonicalize(integrator_trace.get("aus"))
    brow_keep = canonicalize(trace.get("brow_keep_aus") or brow_trace.get("keep_aus"))
    brow_drop = canonicalize(trace.get("brow_drop_aus") or brow_trace.get("drop_aus"))
    eye_keep = canonicalize(trace.get("eye_keep_aus") or eye_trace.get("keep_aus"))
    eye_drop = canonicalize(trace.get("eye_drop_aus") or eye_trace.get("drop_aus"))
    mouth_keep = canonicalize(trace.get("mouth_keep_aus") or mouth_trace.get("keep_aus"))
    mouth_drop = canonicalize(trace.get("mouth_drop_aus") or mouth_trace.get("drop_aus"))
    brow_agent_present = bool(brow_trace) or bool(brow_keep) or bool(brow_drop)
    eye_agent_present = bool(eye_trace) or bool(eye_keep) or bool(eye_drop)
    mouth_agent_present = bool(mouth_trace) or bool(mouth_keep) or bool(mouth_drop)

    if dataset == "samm" and brow_agent_present:
        motion_supported = [name for name in motion_supported if name not in _SAMM_BROW_AGENT_AUS]
        motion_rejected = [name for name in motion_rejected if name not in _SAMM_BROW_AGENT_AUS]
        critic_keep = [name for name in critic_keep if name not in _SAMM_BROW_AGENT_AUS]
        critic_drop = [name for name in critic_drop if name not in _SAMM_BROW_AGENT_AUS]
        integrator_aus = [name for name in integrator_aus if name not in _SAMM_BROW_AGENT_AUS]
    if dataset == "samm" and eye_agent_present:
        motion_supported = [name for name in motion_supported if name not in _SAMM_EYE_AGENT_AUS]
        motion_rejected = [name for name in motion_rejected if name not in _SAMM_EYE_AGENT_AUS]
        critic_keep = [name for name in critic_keep if name not in _SAMM_EYE_AGENT_AUS]
        critic_drop = [name for name in critic_drop if name not in _SAMM_EYE_AGENT_AUS]
        integrator_aus = [name for name in integrator_aus if name not in _SAMM_EYE_AGENT_AUS]
    if dataset == "samm" and mouth_agent_present:
        motion_supported = [name for name in motion_supported if name not in _SAMM_MOUTH_AGENT_AUS]
        motion_rejected = [name for name in motion_rejected if name not in _SAMM_MOUTH_AGENT_AUS]
        critic_keep = [name for name in critic_keep if name not in _SAMM_MOUTH_AGENT_AUS]
        critic_drop = [name for name in critic_drop if name not in _SAMM_MOUTH_AGENT_AUS]
        integrator_aus = [name for name in integrator_aus if name not in _SAMM_MOUTH_AGENT_AUS]
    if dataset == "samm" and (brow_agent_present or eye_agent_present or mouth_agent_present):
        specialist_owned = set()
        if brow_agent_present:
            specialist_owned |= _SAMM_BROW_AGENT_AUS
        if eye_agent_present:
            specialist_owned |= _SAMM_EYE_AGENT_AUS
        if mouth_agent_present:
            specialist_owned |= _SAMM_MOUTH_AGENT_AUS
        critic_specialist_fallback = [
            name for name in critic_specialist_fallback if name in specialist_owned
        ][:1]

    keep = integrator_aus[:]
    if not keep:
        keep = dedupe_keep_order([*critic_keep, *motion_supported])
    if dataset == "samm" and brow_keep:
        keep = dedupe_keep_order([*brow_keep, *keep])
    if dataset == "samm" and eye_keep:
        keep = dedupe_keep_order([*keep, *eye_keep])
    if dataset == "samm" and mouth_keep:
        keep = dedupe_keep_order([*keep, *mouth_keep])
    reject = [
        name
        for name in dedupe_keep_order([*critic_drop, *motion_rejected, *brow_drop, *eye_drop, *mouth_drop])
        if name not in keep
    ]

    raw_emotions = {
        "integrator": str(integrator_trace.get("emotion", "") or "").strip().lower(),
        "critic": str(critic_trace.get("revised_emotion", "") or "").strip().lower(),
        "emotion_agent": str(emotion_trace.get("emotion", "") or "").strip().lower(),
    }

    return {
        "keep_aus": keep,
        "reject_aus": reject,
        "brow_keep_aus": [name for name in brow_keep if name in _SAMM_BROW_AGENT_AUS],
        "brow_reject_aus": [name for name in brow_drop if name in _SAMM_BROW_AGENT_AUS],
        "eye_keep_aus": [name for name in eye_keep if name in _SAMM_EYE_AGENT_AUS],
        "eye_reject_aus": [name for name in eye_drop if name in _SAMM_EYE_AGENT_AUS],
        "mouth_keep_aus": [name for name in mouth_keep if name in _SAMM_MOUTH_AGENT_AUS],
        "mouth_reject_aus": [name for name in mouth_drop if name in _SAMM_MOUTH_AGENT_AUS],
        "critic_specialist_fallback_aus": critic_specialist_fallback,
        "brow_agent_present": brow_agent_present,
        "eye_agent_present": eye_agent_present,
        "mouth_agent_present": mouth_agent_present,
        "brow_state": local_normalize_answer_text(
            str(brow_arbiter_trace.get("brow_state") or brow_arbiter_trace.get("state") or "")
        ).replace(" ", "_"),
        "eye_state": local_normalize_answer_text(
            str(eye_arbiter_trace.get("eye_state") or eye_arbiter_trace.get("state") or "")
        ).replace(" ", "_"),
        "mouth_state": local_normalize_answer_text(
            str(mouth_arbiter_trace.get("mouth_state") or mouth_arbiter_trace.get("state") or "")
        ).replace(" ", "_"),
        "brow_numeric_support_assessment": local_normalize_answer_text(
            str(brow_arbiter_trace.get("numeric_support_assessment") or "")
        ).replace(" ", "_"),
        "eye_numeric_support_assessment": local_normalize_answer_text(
            str(eye_arbiter_trace.get("numeric_support_assessment") or "")
        ).replace(" ", "_"),
        "mouth_numeric_support_assessment": local_normalize_answer_text(
            str(mouth_arbiter_trace.get("numeric_support_assessment") or "")
        ).replace(" ", "_"),
        "brow_local_evidence_status": local_normalize_answer_text(
            str(brow_arbiter_trace.get("local_evidence_status") or "")
        ).replace(" ", "_"),
        "eye_local_evidence_status": local_normalize_answer_text(
            str(eye_arbiter_trace.get("local_evidence_status") or "")
        ).replace(" ", "_"),
        "mouth_local_evidence_status": local_normalize_answer_text(
            str(mouth_arbiter_trace.get("local_evidence_status") or "")
        ).replace(" ", "_"),
        "preferred_emotion": "",
        "emotion_uncertain": False,
        "raw_emotions": raw_emotions,
    }


def _agent_override_strength(info: dict[str, Any] | None) -> float:
    info = info or {}
    return max(
        float(info.get("score", 0.0) or 0.0),
        float(info.get("normalized_support", 0.0) or 0.0),
    )


def _samm_outer_brow_corroborated(info: dict[str, Any] | None) -> bool:
    info = info or {}
    score = float(info.get("score", 0.0) or 0.0)
    normalized = float(info.get("normalized_support", 0.0) or 0.0)
    geometry = float(info.get("geometry_support", 0.0) or 0.0)
    if _present_bonus(info):
        return True
    if geometry >= 0.08:
        return True
    if score >= 1.20 or normalized >= 0.90:
        return True
    return False


def _resolve_anatomical_conflicts(
    names: list[str],
    *,
    dataset: str,
    evidence_map: dict[str, dict[str, Any]],
    visual_set: set[str],
    agent_keep_set: set[str],
    agent_reject_set: set[str],
    numeric: dict[str, Any],
) -> list[str]:
    resolved = dedupe_keep_order(names)
    if dataset != "samm":
        return resolved

    eyelid_pair = {"lid tightener", "eye closure"}
    if eyelid_pair.issubset(set(resolved)):
        preferred = ""
        keep_candidates = [name for name in resolved if name in eyelid_pair and name in agent_keep_set]
        reject_candidates = [name for name in resolved if name in eyelid_pair and name in agent_reject_set]
        if len(keep_candidates) == 1:
            preferred = keep_candidates[0]
        elif len(reject_candidates) == 1:
            preferred = next(name for name in eyelid_pair if name not in reject_candidates)
        else:
            eyelid_signal = str((numeric or {}).get("eyelid_signal", "") or "").strip().lower()
            eye_metrics = (numeric or {}).get("eye_metrics") or {}
            mean_ratio = float(eye_metrics.get("mean_ratio", (numeric or {}).get("eye_ratio", 1.0)) or 1.0)
            min_ratio = float(eye_metrics.get("min_ratio", mean_ratio) or mean_ratio)
            if eyelid_signal == "closure" or min_ratio < 0.88:
                preferred = "eye closure"
            elif eyelid_signal == "narrowing" or mean_ratio < 0.97:
                preferred = "lid tightener"
            else:
                pair_scores = {
                    name: max(
                        float((evidence_map.get(name) or {}).get("score", 0.0) or 0.0),
                        float((evidence_map.get(name) or {}).get("normalized_support", 0.0) or 0.0),
                        float((evidence_map.get(name) or {}).get("raw_support", 0.0) or 0.0),
                        _present_bonus(evidence_map.get(name)),
                    )
                    for name in eyelid_pair
                }
                if pair_scores["eye closure"] > pair_scores["lid tightener"] + 0.08:
                    preferred = "eye closure"
                elif pair_scores["lid tightener"] > pair_scores["eye closure"] + 0.08:
                    preferred = "lid tightener"
                elif "eye closure" in visual_set and "lid tightener" not in visual_set:
                    preferred = "eye closure"
                elif "lid tightener" in visual_set and "eye closure" not in visual_set:
                    preferred = "lid tightener"
                else:
                    preferred = "eye closure" if min_ratio < 0.90 else "lid tightener"
        if preferred:
            resolved = [name for name in resolved if name not in eyelid_pair or name == preferred]

    if "outer brow raiser" in resolved:
        outer_info = evidence_map.get("outer brow raiser") or {}
        if "outer brow raiser" in visual_set and not _samm_outer_brow_corroborated(outer_info):
            resolved = [name for name in resolved if name != "outer brow raiser"]
    return dedupe_keep_order(resolved)


def _visual_au_uncertain(
    pred_au_names: list[str],
    au_answer: str,
    analysis_answer: str,
    *,
    numeric: dict[str, Any] | None,
    dataset: str,
) -> bool:
    if not pred_au_names:
        return True
    source_text = " ".join(part for part in [au_answer, analysis_answer] if part).lower()
    if any(token in source_text for token in ("unknown", "unclear", "not sure")):
        return True
    evidence = _numeric_au_evidence_map(numeric, dataset)
    if not evidence:
        return False

    visual_set = {normalize_au_name(name) for name in pred_au_names if name}
    if not visual_set:
        return True

    strong_numeric = {
        name
        for name, info in evidence.items()
        if _support_gate(info, threshold=0.9)
    }
    overlap = len(visual_set & strong_numeric)
    visual_scores = [
        max(
            float((evidence.get(name) or {}).get("score", 0.0) or 0.0),
            float((evidence.get(name) or {}).get("normalized_support", 0.0) or 0.0),
        )
        for name in visual_set
    ]
    avg_visual_score = sum(visual_scores) / len(visual_scores) if visual_scores else 0.0
    max_nonvisual_score = max(
        (
            max(
                float(info.get("score", 0.0) or 0.0),
                float(info.get("normalized_support", 0.0) or 0.0),
            )
            for name, info in evidence.items()
            if name not in visual_set
        ),
        default=0.0,
    )
    if dataset == "samm":
        if overlap == 0 and avg_visual_score < 0.45:
            return True
        if overlap == 0 and max_nonvisual_score >= avg_visual_score + 0.35:
            return True
        if any(name in {"brow lowerer", "upper lip raiser", "nose wrinkler"} for name in visual_set) and avg_visual_score < 0.55:
            return True
        if "outer brow raiser" in visual_set and not _samm_outer_brow_corroborated(evidence.get("outer brow raiser")):
            return True
    return False


def _dataset_blocks_au(
    *,
    dataset: str,
    au_name: str,
    info: dict[str, Any] | None,
) -> bool:
    info = info or {}
    if dataset == "casme2" and au_name == "lip tightener":
        # CASME2 AU23 evidence is currently uncalibrated and behaves as a consistent false positive.
        return not bool(info.get("calibrated"))
    return False


def _score_pre_fusion_emotion_candidate(
    candidate: str,
    *,
    pre_fusion_emotion: str,
    emotion_uncertain: bool,
) -> float:
    candidate = (candidate or "").strip().lower()
    pre_fusion_emotion = (pre_fusion_emotion or "").strip().lower()
    if not candidate or candidate != pre_fusion_emotion:
        return 0.0
    return 0.55 if emotion_uncertain else 1.0


def _normalize_emotion_prior_scores(prior_scores: dict[str, float]) -> dict[str, float]:
    finite_scores = {
        (name or "").strip().lower(): float(score)
        for name, score in (prior_scores or {}).items()
        if isinstance(score, (int, float)) and math.isfinite(float(score))
    }
    if not finite_scores:
        return {}

    values = list(finite_scores.values())
    low = min(values)
    high = max(values)
    if high - low <= 1e-8:
        return {name: 0.5 for name in finite_scores}
    return {
        name: max(0.0, min(1.0, (score - low) / (high - low)))
        for name, score in finite_scores.items()
    }


def _pre_fusion_consistency_scale(
    pre_fusion_emotion: str,
    *,
    numeric_emotion_scores: dict[str, float],
    authoritative_keep: set[str],
) -> float:
    pre_fusion_emotion = (pre_fusion_emotion or "").strip().lower()
    if not pre_fusion_emotion or pre_fusion_emotion not in numeric_emotion_scores:
        return 1.0
    if not numeric_emotion_scores:
        return 1.0

    ordered = sorted(
        numeric_emotion_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    top_name, top_score = ordered[0]
    pre_score = float(numeric_emotion_scores.get(pre_fusion_emotion, 0.0) or 0.0)
    if top_name == pre_fusion_emotion:
        return 1.0

    margin = top_score - pre_score
    if margin >= 0.55:
        return 0.65 if authoritative_keep else 0.35
    if margin >= 0.30:
        return 0.78 if authoritative_keep else 0.55
    if margin >= 0.18:
        return 0.88 if authoritative_keep else 0.72
    return 1.0


def _emotion_scoring_support(au_name: str, info: dict[str, Any] | None) -> float:
    info = info or {}
    support = max(
        float(info.get("score", 0.0) or 0.0),
        float(info.get("normalized_support", 0.0) or 0.0),
        float(info.get("combined_support", 0.0) or 0.0),
        float(info.get("geometry_support", 0.0) or 0.0),
        _present_bonus(info),
    )
    if bool(info.get("agent_only")):
        support = max(support, _samm_specialist_proxy_strength(info))
    return min(2.5, support)


def _samm_lip_corner_puller_rescue_allowed(
    *,
    numeric_evidence: dict[str, dict[str, Any]],
    agent_guidance: dict[str, Any],
    emotion_scores: dict[str, float] | None = None,
) -> bool:
    info = numeric_evidence.get("lip corner puller") or {}
    if bool(info.get("agent_only")):
        return False
    support = _agent_override_strength(info)
    present = _support_gate(info, threshold=1.0)
    normalized_support = float(info.get("normalized_support", 0.0) or 0.0)
    combined = max(
        float(info.get("combined_support", 0.0) or 0.0),
        float(info.get("geometry_support", 0.0) or 0.0),
        float(info.get("raw_support", 0.0) or 0.0),
        float(info.get("residual_support", 0.0) or 0.0),
    )
    if support <= 0.0:
        return False

    normalized_scores: dict[str, float] = {}
    for name, value in (emotion_scores or {}).items():
        normalized = normalize_emotion_for_dataset(str(name or ""), "samm")
        if not normalized:
            continue
        normalized_scores[normalized] = float(value or 0.0)
    ordered_scores = sorted(
        normalized_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    top_emotion = ordered_scores[0][0] if ordered_scores else ""
    top_score = ordered_scores[0][1] if ordered_scores else 0.0
    second_score = ordered_scores[1][1] if len(ordered_scores) > 1 else 0.0
    happiness_score = float(normalized_scores.get("happiness", 0.0) or 0.0)
    anger_score = float(normalized_scores.get("anger", 0.0) or 0.0)

    upper_lip_info = numeric_evidence.get("upper lip raiser") or {}
    cheek_info = numeric_evidence.get("cheek raiser") or {}
    positive_aux_present = _support_gate(upper_lip_info, threshold=1.0) or _support_gate(
        cheek_info,
        threshold=1.0,
    )
    mouth_state = str(agent_guidance.get("mouth_state") or "").strip().lower()
    mouth_numeric = str(agent_guidance.get("mouth_numeric_support_assessment") or "").strip().lower()
    authoritative_negative = bool(
        set(agent_guidance.get("eye_keep_aus") or []) & {"lid tightener", "eye closure"}
    ) or bool(
        set(agent_guidance.get("brow_keep_aus") or []) & {"brow lowerer"}
    )

    strong_hidden_support = (
        support >= 3.0
        and (
            combined >= 0.25
            or normalized_support >= 1.8
        )
    )
    positive_aux_bridge = (
        present
        and support >= 1.3
        and combined >= 0.25
        and positive_aux_present
    )
    confident_happy_prior = (
        top_emotion == "happiness"
        and top_score - second_score >= 0.5
        and present
        and support >= 2.7
        and combined >= 0.35
    )
    strong_arbiter_numeric = (
        mouth_numeric == "strong"
        and present
        and support >= 1.0
        and combined >= 0.20
    )
    moderate_local_support = (
        mouth_numeric in {"moderate", "strong"}
        and present
        and support >= 1.15
        and combined >= 0.20
    )

    if mouth_state == "local_mouth_action":
        if authoritative_negative:
            return (
                strong_hidden_support
                or (positive_aux_bridge and happiness_score >= anger_score + 0.25)
                or strong_arbiter_numeric
            )
        return (
            strong_hidden_support
            or positive_aux_bridge
            or strong_arbiter_numeric
            or moderate_local_support
            or confident_happy_prior
        )

    if mouth_state == "no_reliable_mouth_au":
        if authoritative_negative:
            return (
                strong_hidden_support
                or (
                    strong_arbiter_numeric
                    and (positive_aux_present or happiness_score >= anger_score + 0.35)
                )
            )
        return (
            strong_hidden_support
            or positive_aux_bridge
            or strong_arbiter_numeric
            or confident_happy_prior
        )

    return (
        strong_hidden_support
        or positive_aux_bridge
        or strong_arbiter_numeric
    )


def _samm_strong_mouth_override_candidates(
    *,
    numeric_evidence: dict[str, dict[str, Any]],
    agent_guidance: dict[str, Any],
    emotion_hint: str = "",
    emotion_scores: dict[str, float] | None = None,
) -> list[str]:
    if not agent_guidance.get("mouth_agent_present"):
        return []
    hinted_emotion = normalize_emotion_for_dataset(emotion_hint, "samm")
    normalized_scores: dict[str, float] = {}
    for name, value in (emotion_scores or {}).items():
        normalized = normalize_emotion_for_dataset(str(name or ""), "samm")
        if not normalized:
            continue
        normalized_scores[normalized] = float(value or 0.0)
    ordered_scores = sorted(
        normalized_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    top_emotion = ordered_scores[0][0] if ordered_scores else ""
    if hinted_emotion != "happiness" and top_emotion != "happiness":
        return []
    happiness_score = float(normalized_scores.get("happiness", 0.0) or 0.0)
    anger_score = float(normalized_scores.get("anger", 0.0) or 0.0)
    authoritative_eye = bool(set(agent_guidance.get("eye_keep_aus") or []) & _SAMM_EYE_AGENT_AUS)
    mouth_keep = set(agent_guidance.get("mouth_keep_aus") or [])
    mouth_reject = set(agent_guidance.get("mouth_reject_aus") or [])
    upper_lip_info = numeric_evidence.get("upper lip raiser") or {}
    cheek_info = numeric_evidence.get("cheek raiser") or {}
    positive_aux_present = _support_gate(upper_lip_info, threshold=1.0) or _support_gate(
        cheek_info,
        threshold=1.0,
    )
    candidates: list[tuple[float, str]] = []
    for name in ["lip corner puller"]:
        if name in mouth_keep or name not in mouth_reject:
            continue
        info = numeric_evidence.get(name) or {}
        if not _samm_lip_corner_puller_rescue_allowed(
            numeric_evidence=numeric_evidence,
            agent_guidance=agent_guidance,
            emotion_scores=normalized_scores,
        ):
            continue
        support = _agent_override_strength(info)
        present = _support_gate(info, threshold=1.0)
        combined = max(
            float(info.get("combined_support", 0.0) or 0.0),
            float(info.get("geometry_support", 0.0) or 0.0),
            float(info.get("raw_support", 0.0) or 0.0),
            float(info.get("residual_support", 0.0) or 0.0),
        )
        normalized_support = float(info.get("normalized_support", 0.0) or 0.0)
        score = (
            support
            + 0.35 * combined
            + (0.25 if present and _present_gate_enabled() else 0.0)
            + (0.12 if positive_aux_present else 0.0)
            + 0.08 * max(0.0, happiness_score - anger_score)
        )
        candidates.append((score, name))
    candidates.sort(reverse=True)
    return [name for _score, name in candidates[:2]]


def _score_emotion_from_numeric_evidence(
    candidate: str,
    *,
    dataset: str,
    numeric_evidence: dict[str, dict[str, Any]],
    agent_guidance: dict[str, Any],
) -> float:
    candidate = (candidate or "").strip().lower()
    if not candidate:
        return 0.0

    calibration = load_motion_feature_calibration()
    dataset_calibration = (calibration.get("datasets") or {}).get(dataset, {})
    emotion_au_rates = dataset_calibration.get("emotion_au_rates") or {}
    candidate_rates = emotion_au_rates.get(candidate) or {}
    if not candidate_rates:
        return 0.0

    allowed = [
        name
        for name in (DATASET_ALLOWED_EMOTIONS.get(dataset) or [])
        if name in emotion_au_rates
    ]
    other_emotions = [name for name in allowed if name != candidate]
    if not other_emotions:
        other_emotions = [name for name in emotion_au_rates if name != candidate]
    if not other_emotions:
        return 0.0

    authoritative_keep = {
        name
        for name in [
            *(agent_guidance.get("keep_aus") or []),
            *(agent_guidance.get("brow_keep_aus") or []),
            *(agent_guidance.get("eye_keep_aus") or []),
            *(agent_guidance.get("mouth_keep_aus") or []),
        ]
        if name
    }
    authoritative_reject = {
        name
        for name in [
            *(agent_guidance.get("reject_aus") or []),
            *(agent_guidance.get("brow_reject_aus") or []),
            *(agent_guidance.get("eye_reject_aus") or []),
            *(agent_guidance.get("mouth_reject_aus") or []),
        ]
        if name and name not in authoritative_keep
    }

    support_map: dict[str, float] = {}
    for au_name, info in numeric_evidence.items():
        if au_name in authoritative_reject:
            continue
        support = _emotion_scoring_support(au_name, info)
        if support <= 0.0:
            continue
        support_map[au_name] = support

    for au_name in authoritative_keep:
        support_map[au_name] = max(support_map.get(au_name, 0.0), 1.0)

    observed = {
        au_name: support
        for au_name, support in support_map.items()
        if support >= 0.25
    }
    if not observed:
        return 0.0

    raw_score = 0.0
    for au_name, support in observed.items():
        rate = float(candidate_rates.get(au_name, 0.0) or 0.0)
        other_mean = sum(
            float((emotion_au_rates.get(other) or {}).get(au_name, 0.0) or 0.0)
            for other in other_emotions
        ) / max(1, len(other_emotions))
        raw_score += support * (rate - other_mean)

    missing_penalty = 0.0
    for au_name, raw_rate in candidate_rates.items():
        if au_name in authoritative_reject:
            continue
        rate = float(raw_rate or 0.0)
        if rate < 0.18:
            continue
        if observed.get(au_name, 0.0) >= 0.25:
            continue
        other_mean = sum(
            float((emotion_au_rates.get(other) or {}).get(au_name, 0.0) or 0.0)
            for other in other_emotions
        ) / max(1, len(other_emotions))
        missing_penalty += max(0.0, rate - other_mean) * rate

    raw_score -= 0.45 * missing_penalty
    return max(0.0, min(1.0, 0.5 + 0.5 * math.tanh(raw_score / 0.9)))


def _best_agent_emotion_fallback(
    agent_guidance: dict[str, Any],
    *,
    dataset: str,
) -> str:
    raw_emotions = agent_guidance.get("raw_emotions") or {}
    for key in ("critic", "emotion_agent", "integrator"):
        candidate = normalize_emotion_for_dataset(str(raw_emotions.get(key, "") or ""), dataset)
        if candidate and candidate not in _AGENT_UNCERTAIN_EMOTIONS:
            return candidate
    return ""


def _post_prune_emotion_scores(
    *,
    dataset: str,
    pred_au_names: list[str],
    numeric: dict[str, Any],
    agent_trace: dict[str, Any] | None = None,
) -> dict[str, float]:
    final_aus = dedupe_keep_order(normalize_au_name(name) for name in (pred_au_names or []) if name)
    if not final_aus:
        return {}

    agent_guidance = dict(_agent_guidance(agent_trace, dataset))
    reject = [
        name
        for name in (agent_guidance.get("reject_aus") or [])
        if name not in final_aus
    ]
    agent_guidance["keep_aus"] = final_aus
    agent_guidance["reject_aus"] = reject
    if dataset == "samm":
        agent_guidance["brow_keep_aus"] = [name for name in final_aus if name in _SAMM_BROW_AGENT_AUS]
        agent_guidance["eye_keep_aus"] = [name for name in final_aus if name in _SAMM_EYE_AGENT_AUS]
        agent_guidance["mouth_keep_aus"] = [name for name in final_aus if name in _SAMM_MOUTH_AGENT_AUS]
        agent_guidance["brow_reject_aus"] = [
            name for name in reject if name in _SAMM_BROW_AGENT_AUS
        ]
        agent_guidance["eye_reject_aus"] = [
            name for name in reject if name in _SAMM_EYE_AGENT_AUS
        ]
        agent_guidance["mouth_reject_aus"] = [
            name for name in reject if name in _SAMM_MOUTH_AGENT_AUS
        ]

    numeric_evidence = _numeric_au_evidence_map(numeric, dataset)
    scores = {
        emotion: _score_emotion_from_numeric_evidence(
            emotion,
            dataset=dataset,
            numeric_evidence=numeric_evidence,
            agent_guidance=agent_guidance,
        )
        for emotion in (DATASET_ALLOWED_EMOTIONS.get(dataset) or [])
    }
    return {
        emotion: round(score, 3)
        for emotion, score in scores.items()
        if score > 0.0
    }


def fuse_predictions(
    *,
    dataset: str,
    fine_answer: str,
    au_answer: str,
    analysis_answer: str,
    visual_au_names: list[str],
    knowledge_prior: dict[str, Any],
    numeric: dict[str, Any],
    agent_trace: dict[str, Any] | None = None,
    allow_answer_emotion_fallback: bool = True,
) -> tuple[str, list[str], dict[str, Any]]:
    fused_au_names = list(visual_au_names)
    agent_guidance = _agent_guidance(agent_trace, dataset)
    agent_keep_set = set(agent_guidance.get("keep_aus") or [])
    agent_reject_set = set(agent_guidance.get("reject_aus") or [])
    brow_keep_set = set(agent_guidance.get("brow_keep_aus") or [])
    brow_reject_set = set(agent_guidance.get("brow_reject_aus") or [])
    eye_keep_set = set(agent_guidance.get("eye_keep_aus") or [])
    eye_reject_set = set(agent_guidance.get("eye_reject_aus") or [])
    mouth_keep_set = set(agent_guidance.get("mouth_keep_aus") or [])
    mouth_reject_set = set(agent_guidance.get("mouth_reject_aus") or [])
    critic_specialist_fallback = dedupe_keep_order(
        agent_guidance.get("critic_specialist_fallback_aus") or []
    )[:1]
    brow_agent_present = bool(agent_guidance.get("brow_agent_present"))
    eye_agent_present = bool(agent_guidance.get("eye_agent_present"))
    mouth_agent_present = bool(agent_guidance.get("mouth_agent_present"))
    authoritative_brow = [
        name
        for name in (agent_guidance.get("brow_keep_aus") or [])
        if name in _SAMM_BROW_AGENT_AUS
    ]
    authoritative_eye = [
        name
        for name in (agent_guidance.get("eye_keep_aus") or [])
        if name in _SAMM_EYE_AGENT_AUS
    ]
    authoritative_mouth = [
        name
        for name in (agent_guidance.get("mouth_keep_aus") or [])
        if name in _SAMM_MOUTH_AGENT_AUS
    ]
    pre_fusion_emotion = _pre_fusion_emotion(
        "",
        fine_answer=fine_answer,
        analysis_answer=analysis_answer,
        dataset=dataset,
        allow_answer_fallback=allow_answer_emotion_fallback,
    )
    fused_emotion = ""
    policy = {
        "mode": "evidence_first_hybrid" if numeric and numeric.get("available") else "agent_first_numeric_verifier",
        "emotion_source": "unset",
        "au_source": "visual",
        "emotion_override": False,
        "au_override": False,
    }

    visual_au_is_uncertain = _visual_au_uncertain(
        visual_au_names,
        au_answer,
        analysis_answer,
        numeric=numeric,
        dataset=dataset,
    )

    suggested_emotion = (knowledge_prior.get("suggested_emotion") or "").strip().lower()
    suggested_aus = list(knowledge_prior.get("suggested_aus") or [])
    prior_scores = {
        (name or "").strip().lower(): float(score)
        for name, score in (knowledge_prior.get("scores") or {}).items()
    }
    top_prior_emotion = ""
    if prior_scores:
        top_prior_emotion = normalize_emotion_for_dataset(
            max(prior_scores.items(), key=lambda item: item[1])[0],
            dataset,
        )
    combined_emotion_scores: dict[str, float] = {}
    numeric_emotion_scores: dict[str, float] = {}

    calibration = load_motion_feature_calibration()
    dataset_calibration = (calibration.get("datasets") or {}).get(dataset, {})
    emotion_au_rates = dataset_calibration.get("emotion_au_rates") or {}
    numeric_evidence = _numeric_au_evidence_map(numeric, dataset)
    mouth_override_candidates = (
        _samm_strong_mouth_override_candidates(
            numeric_evidence=numeric_evidence,
            agent_guidance=agent_guidance,
            emotion_hint=(suggested_emotion or top_prior_emotion or ""),
            emotion_scores=prior_scores,
        )
        if dataset == "samm"
        else []
    )
    authoritative_keep = set([*authoritative_brow, *authoritative_eye, *authoritative_mouth])
    prior_emotion_scores = _normalize_emotion_prior_scores(prior_scores)
    allowed_emotions = [
        emotion
        for emotion in (DATASET_ALLOWED_EMOTIONS.get(dataset) or [])
        if emotion in emotion_au_rates or emotion in prior_emotion_scores
    ]
    for emotion in allowed_emotions:
        numeric_score = _score_emotion_from_numeric_evidence(
            emotion,
            dataset=dataset,
            numeric_evidence=numeric_evidence,
            agent_guidance=agent_guidance,
        )
        numeric_emotion_scores[emotion] = round(numeric_score, 3)
        prior_score = float(prior_emotion_scores.get(emotion, 0.0) or 0.0)
        if authoritative_keep:
            combined = 0.75 * numeric_score + 0.25 * prior_score
        else:
            combined = 0.45 * numeric_score + 0.55 * prior_score
        combined_emotion_scores[emotion] = round(combined, 3)

    numeric_ranked = _ranked_numeric_aus(numeric, dataset)
    if not fused_emotion and authoritative_keep and combined_emotion_scores:
        ordered_emotions = sorted(
            combined_emotion_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_emotion, top_score = ordered_emotions[0]
        second_score = ordered_emotions[1][1] if len(ordered_emotions) > 1 else 0.0
        if top_score >= 0.52 and (top_score - second_score >= 0.05 or top_score >= 0.62):
            fused_emotion = normalize_emotion_for_dataset(top_emotion, dataset)
            policy["emotion_source"] = "authority_consistent_numeric"
            policy["emotion_override"] = bool(
                pre_fusion_emotion
                and pre_fusion_emotion not in _AGENT_UNCERTAIN_EMOTIONS
                and fused_emotion != pre_fusion_emotion
            )
    if not fused_emotion and suggested_emotion:
        fused_emotion = normalize_emotion_for_dataset(suggested_emotion, dataset)
        policy["emotion_source"] = "knowledge_prior"
        policy["emotion_override"] = bool(
            pre_fusion_emotion
            and pre_fusion_emotion not in _AGENT_UNCERTAIN_EMOTIONS
            and fused_emotion != pre_fusion_emotion
        )
    elif not fused_emotion and top_prior_emotion:
        fused_emotion = top_prior_emotion
        policy["emotion_source"] = "knowledge_prior_top1"
        policy["emotion_override"] = bool(
            pre_fusion_emotion
            and pre_fusion_emotion not in _AGENT_UNCERTAIN_EMOTIONS
            and fused_emotion != pre_fusion_emotion
        )
    elif not fused_emotion and allow_answer_emotion_fallback:
        fallback_text = fine_answer or analysis_answer
        fused_emotion = normalize_emotion_for_dataset(extract_emotion(fallback_text), dataset)
        policy["emotion_source"] = "answer_fallback" if fused_emotion else "unset"
        policy["emotion_override"] = bool(
            pre_fusion_emotion
            and pre_fusion_emotion not in _AGENT_UNCERTAIN_EMOTIONS
            and fused_emotion
            and fused_emotion != pre_fusion_emotion
        )

    visual_au_set = set(visual_au_names)
    independent_visual_au_set = set(visual_au_set)
    if dataset == "samm" and brow_agent_present:
        independent_visual_au_set -= _SAMM_BROW_AGENT_AUS
    if dataset == "samm" and eye_agent_present:
        independent_visual_au_set -= _SAMM_EYE_AGENT_AUS
    if dataset == "samm" and mouth_agent_present:
        independent_visual_au_set -= _SAMM_MOUTH_AGENT_AUS
    suggested_au_set = set(suggested_aus)
    candidate_names = dedupe_keep_order(
        [*numeric_ranked, *suggested_aus, *visual_au_names, *(agent_guidance.get("keep_aus") or [])]
    )
    combined_au_scores: dict[str, float] = {}
    for name in candidate_names:
        info = numeric_evidence.get(name, {})
        if _dataset_blocks_au(dataset=dataset, au_name=name, info=info):
            continue
        evidence_score = min(float(info.get("score", 0.0) or 0.0), 2.0)
        normalized_support = float(info.get("normalized_support", 0.0) or 0.0)
        present = _support_gate(info, threshold=1.0)
        calibrated = bool(info.get("calibrated"))
        reliability = _evidence_reliability(info)
        in_visual = name in independent_visual_au_set
        in_prior = name in suggested_au_set
        in_agent_keep = name in agent_keep_set
        in_agent_reject = name in agent_reject_set
        score = (0.75 if dataset == "samm" else 0.60) * evidence_score
        if present and _present_gate_enabled():
            score += 0.18
        if in_visual:
            score += 0.14
        if in_prior:
            score += 0.10
        if in_agent_keep:
            if _uses_calibrated_reliability_regularization(dataset) and calibrated:
                keep_scale = 0.25 + 0.75 * min(1.0, reliability / 0.35)
                score += 0.18 * keep_scale
            else:
                score += 0.18
        if in_agent_reject:
            score -= 0.36 if dataset == "samm" else 0.24
        if _uses_calibrated_reliability_regularization(dataset) and calibrated:
            score *= 0.55 + 0.45 * min(1.0, reliability / 0.35)
            if reliability < 0.18 and not in_visual:
                score -= 0.10
        if (
            _uses_calibrated_reliability_regularization(dataset)
            and in_visual
            and not present
            and normalized_support < 0.55
        ):
            score -= 0.18
        if dataset == "samm" and name in _SAMM_BROW_AGENT_AUS and brow_agent_present:
            if name in brow_keep_set:
                score += 0.28
            elif name in brow_reject_set:
                score -= 0.50
            else:
                score -= 0.26
        if dataset == "samm" and name in _SAMM_EYE_AGENT_AUS and eye_agent_present:
            if name in eye_keep_set:
                score += 0.28
            elif name in eye_reject_set:
                score -= 0.50
            else:
                score -= 0.26
        if dataset == "samm" and name in _SAMM_MOUTH_AGENT_AUS and mouth_agent_present:
            if name in mouth_keep_set:
                score += 0.28
            elif name in mouth_reject_set:
                score -= 0.50
            else:
                score -= 0.26
        combined_au_scores[name] = round(score, 3)

    if combined_au_scores:
        ordered_candidates = [
            name
            for name, _ in sorted(
                combined_au_scores.items(),
                key=lambda item: (
                    item[1],
                    _continuous_support(numeric_evidence.get(item[0])),
                    item[0] in independent_visual_au_set,
                ),
                reverse=True,
            )
        ]
        cutoff = 0.72 if dataset == "samm" else 0.55
        fused_au_names = []
        for name in ordered_candidates:
            info = numeric_evidence.get(name) or {}
            if _dataset_blocks_au(dataset=dataset, au_name=name, info=info):
                continue
            present = _support_gate(info, threshold=1.0)
            if name in agent_reject_set and name not in agent_keep_set:
                if not _override_gate(info, threshold=1.55 if dataset == "samm" else 1.35):
                    continue
            if dataset == "samm" and name in _SAMM_BROW_AGENT_AUS and brow_agent_present:
                if name in brow_reject_set and name not in brow_keep_set:
                    if not _override_gate(info, threshold=1.95):
                        continue
                if name not in brow_keep_set:
                    if not _override_gate(info, threshold=1.95):
                        continue
            if dataset == "samm" and name in _SAMM_EYE_AGENT_AUS and eye_agent_present:
                if name in eye_reject_set and name not in eye_keep_set:
                    if not _override_gate(info, threshold=1.95):
                        continue
                if name not in eye_keep_set:
                    if not _override_gate(info, threshold=1.95):
                        continue
            if dataset == "samm" and name in _SAMM_MOUTH_AGENT_AUS and mouth_agent_present:
                if name in mouth_reject_set and name not in mouth_keep_set:
                    if not _override_gate(info, threshold=1.95):
                        continue
                if name not in mouth_keep_set:
                    if not _override_gate(info, threshold=1.95):
                        continue
            if agent_keep_set and name not in agent_keep_set:
                if not _override_gate(info, threshold=1.55 if dataset == "samm" else 1.35):
                    continue
            if name in agent_keep_set:
                if (
                    combined_au_scores[name] >= cutoff - 0.18
                    or present
                    or name in independent_visual_au_set
                ):
                    fused_au_names.append(name)
                continue
            if combined_au_scores[name] >= cutoff or present:
                fused_au_names.append(name)
        if not fused_au_names:
            if agent_keep_set:
                fused_au_names = [
                    name
                    for name in ordered_candidates
                    if name in agent_keep_set
                ][: max(1, min(2, len(agent_keep_set)))]
            elif not visual_au_is_uncertain and visual_au_names:
                fused_au_names = ordered_candidates[: max(1, min(2, len(visual_au_names)))]
            else:
                fused_au_names = ordered_candidates[:1]
    if dataset == "samm" and (brow_agent_present or eye_agent_present or mouth_agent_present):
        fused_au_names = [
            name
            for name in fused_au_names
            if name not in (_SAMM_BROW_AGENT_AUS | _SAMM_EYE_AGENT_AUS | _SAMM_MOUTH_AGENT_AUS)
        ]
        fused_au_names = dedupe_keep_order([
            *authoritative_brow,
            *authoritative_eye,
            *authoritative_mouth,
            *mouth_override_candidates,
            *fused_au_names,
        ])
        if fused_au_names != visual_au_names:
            policy["au_source"] = "evidence_hybrid" if visual_au_names else "numeric_evidence"
            policy["au_override"] = True
    elif visual_au_is_uncertain and suggested_aus:
        fused_au_names = suggested_aus
        policy["au_source"] = "knowledge_prior"
        policy["au_override"] = True

    if dataset == "samm" and (brow_agent_present or eye_agent_present or mouth_agent_present):
        fused_au_names = [
            name
            for name in fused_au_names
            if name not in (_SAMM_BROW_AGENT_AUS | _SAMM_EYE_AGENT_AUS | _SAMM_MOUTH_AGENT_AUS)
        ]
        fused_au_names = dedupe_keep_order([
            *authoritative_brow,
            *authoritative_eye,
            *authoritative_mouth,
            *mouth_override_candidates,
            *fused_au_names,
        ])
        if not fused_au_names and not authoritative_keep:
            if critic_specialist_fallback:
                fused_au_names = critic_specialist_fallback[:]
                policy["au_source"] = "critic_specialist_fallback"
                policy["au_override"] = True
            else:
                fallback_emotion = _best_agent_emotion_fallback(agent_guidance, dataset=dataset)
                if fallback_emotion:
                    fused_emotion = fallback_emotion
                    policy["emotion_source"] = "empty_au_agent_fallback"
                    policy["emotion_override"] = bool(
                        pre_fusion_emotion
                        and pre_fusion_emotion not in _AGENT_UNCERTAIN_EMOTIONS
                        and fused_emotion != pre_fusion_emotion
                    )

    policy["combined_emotion_scores"] = combined_emotion_scores
    policy["numeric_emotion_scores"] = numeric_emotion_scores
    policy["combined_au_scores"] = combined_au_scores
    policy["numeric_supported_aus"] = numeric_ranked
    policy["agent_keep_aus"] = list(agent_keep_set)
    policy["agent_reject_aus"] = list(agent_reject_set)
    policy["pre_fusion_emotion"] = pre_fusion_emotion
    policy["agent_preferred_emotion"] = ""
    policy["agent_emotion_uncertain"] = False

    return fused_emotion, dedupe_keep_order(fused_au_names), policy


def _join_names(names: list[str]) -> str:
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _split_aus(pred_au_names: list[str]) -> tuple[list[str], list[str]]:
    upper_face = {
        "inner brow raiser",
        "outer brow raiser",
        "brow lowerer",
        "upper lid raiser",
        "lid tightener",
        "cheek raiser",
        "eye closure",
    }
    brow_aus = [name for name in pred_au_names if name in upper_face]
    mouth_aus = [name for name in pred_au_names if name not in upper_face]
    return brow_aus, mouth_aus


def _au_numeric_support(name: str, numeric: dict[str, Any]) -> float:
    evidence = _numeric_au_evidence_map(numeric, "")
    info = evidence.get(normalize_au_name(name)) or {}
    if info:
        return max(
            float(info.get("score", 0.0) or 0.0),
            float(info.get("normalized_support", 0.0) or 0.0),
        )
    raw = numeric.get("landmark_au_regions") or {}
    residual = numeric.get("landmark_residual_au_regions") or {}

    def region_score(region: str) -> float:
        score = 0.0
        info = raw.get(region) or {}
        if info:
            score = max(score, float(info.get("active", 0.0) or 0.0) * float(info.get("mag", 0.0) or 0.0))
        info = residual.get(region) or {}
        if info:
            score = max(score, 18.0 * float(info.get("active", 0.0) or 0.0) * float(info.get("mag", 0.0) or 0.0))
        return score

    canonical = normalize_au_name(name)
    if canonical == "inner brow raiser":
        return min(region_score("AU1_inner_brow_subR"), region_score("AU1_inner_brow_subL"))
    if canonical == "outer brow raiser":
        return min(region_score("AU2_brow_outer_subR"), region_score("AU2_brow_outer_subL"))
    if canonical == "brow lowerer":
        return region_score("AU4_brow_lowerer")
    if canonical == "upper lid raiser":
        eye_ratio = numeric.get("eye_ratio")
        if eye_ratio is None:
            return 0.0
        return max(0.0, float(eye_ratio) - 1.03) * 8.0
    if canonical == "lid tightener":
        eye_ratio = numeric.get("eye_ratio")
        if eye_ratio is None:
            return 0.0
        return max(0.0, 0.97 - float(eye_ratio)) * 8.0
    if canonical == "nose wrinkler":
        return region_score("AU9_nose_bridge")
    if canonical == "upper lip raiser":
        return region_score("AU10_upper_lip")
    if canonical == "lip corner puller":
        return min(region_score("AU12_corner_subR"), region_score("AU12_corner_subL"))
    if canonical == "chin raiser":
        return region_score("AU17_chin")
    if canonical == "lip tightener":
        return region_score("AU23_lip_center")
    return 0.0


def _prune_predicted_aus(
    pred_au_names: list[str],
    *,
    dataset: str,
    final_emotion: str,
    visual_au_names: list[str],
    knowledge_prior: dict[str, Any],
    numeric: dict[str, Any],
    agent_trace: dict[str, Any] | None = None,
) -> list[str]:
    canonical = dedupe_keep_order(normalize_au_name(name) for name in pred_au_names if name)
    original_canonical = canonical[:]
    numeric = ensure_au_evidence(dict(numeric or {}), dataset)
    evidence_map = _numeric_au_evidence_map(numeric, dataset)
    agent_guidance = _agent_guidance(agent_trace, dataset)
    prior_scores = {
        (name or "").strip().lower(): float(score)
        for name, score in (knowledge_prior.get("scores") or {}).items()
    }
    top_prior_emotion = ""
    if prior_scores:
        top_prior_emotion = normalize_emotion_for_dataset(
            max(prior_scores.items(), key=lambda item: item[1])[0],
            dataset,
        )
    mouth_override_candidates = (
        _samm_strong_mouth_override_candidates(
            numeric_evidence=evidence_map,
            agent_guidance=agent_guidance,
            emotion_hint=(
                normalize_emotion_for_dataset(
                    str(knowledge_prior.get("suggested_emotion") or ""),
                    dataset,
                )
                or top_prior_emotion
                or final_emotion
            ),
            emotion_scores=prior_scores,
        )
        if dataset == "samm"
        else []
    )
    mouth_override_set = set(mouth_override_candidates)
    critic_specialist_fallback = dedupe_keep_order(
        agent_guidance.get("critic_specialist_fallback_aus") or []
    )[:1]
    critic_specialist_fallback_set = set(critic_specialist_fallback)
    agent_keep = dedupe_keep_order(agent_guidance.get("keep_aus") or [])
    agent_keep_set = set(agent_keep)
    agent_reject_set = set(agent_guidance.get("reject_aus") or [])
    brow_keep_set = set(agent_guidance.get("brow_keep_aus") or [])
    brow_reject_set = set(agent_guidance.get("brow_reject_aus") or [])
    eye_keep_set = set(agent_guidance.get("eye_keep_aus") or [])
    eye_reject_set = set(agent_guidance.get("eye_reject_aus") or [])
    mouth_keep_set = set(agent_guidance.get("mouth_keep_aus") or [])
    mouth_reject_set = set(agent_guidance.get("mouth_reject_aus") or [])
    brow_agent_present = bool(agent_guidance.get("brow_agent_present"))
    eye_agent_present = bool(agent_guidance.get("eye_agent_present"))
    mouth_agent_present = bool(agent_guidance.get("mouth_agent_present"))
    authoritative_brow = [
        name
        for name in (agent_guidance.get("brow_keep_aus") or [])
        if name in _SAMM_BROW_AGENT_AUS
    ]
    authoritative_eye = [
        name
        for name in (agent_guidance.get("eye_keep_aus") or [])
        if name in _SAMM_EYE_AGENT_AUS
    ]
    authoritative_mouth = [
        name
        for name in (agent_guidance.get("mouth_keep_aus") or [])
        if name in _SAMM_MOUTH_AGENT_AUS
    ]
    specialist_owned = set()
    if dataset == "samm" and brow_agent_present:
        specialist_owned |= _SAMM_BROW_AGENT_AUS
    if dataset == "samm" and eye_agent_present:
        specialist_owned |= _SAMM_EYE_AGENT_AUS
    if dataset == "samm" and mouth_agent_present:
        specialist_owned |= _SAMM_MOUTH_AGENT_AUS
    if specialist_owned:
        canonical = [
            name
            for name in canonical
            if name not in specialist_owned
            or name in mouth_override_set
            or name in critic_specialist_fallback_set
        ]
        agent_keep = [
            name
            for name in agent_keep
            if name not in specialist_owned
            or name in mouth_override_set
            or name in critic_specialist_fallback_set
        ]
        agent_keep_set = set(agent_keep)
        agent_reject_set = {
            name
            for name in agent_reject_set
            if name not in specialist_owned
            or name in mouth_override_set
            or name in critic_specialist_fallback_set
        }
    visual_set = {normalize_au_name(name) for name in visual_au_names if name}
    if specialist_owned:
        visual_set = {
            name
            for name in visual_set
            if name not in specialist_owned
            or name in mouth_override_set
            or name in critic_specialist_fallback_set
        }
    prior_set = {normalize_au_name(name) for name in (knowledge_prior.get("suggested_aus") or []) if name}
    if specialist_owned:
        prior_set = {
            name
            for name in prior_set
            if name not in specialist_owned
            or name in mouth_override_set
            or name in critic_specialist_fallback_set
        }
    mouth_reject_set = {
        name for name in mouth_reject_set if name not in mouth_override_set
    }
    agent_reject_set = {
        name for name in agent_reject_set if name not in mouth_override_set
    }

    def rescue_empty_result() -> list[str]:
        fallback_pool = dedupe_keep_order([
            *critic_specialist_fallback,
            *authoritative_brow,
            *authoritative_eye,
            *authoritative_mouth,
            *mouth_override_candidates,
            *original_canonical,
            *(knowledge_prior.get("suggested_aus") or []),
            *visual_au_names,
            *[
                name
                for name, info in sorted(
                    evidence_map.items(),
                    key=lambda item: _rank_key(item[1]),
                    reverse=True,
                )
            ],
        ])
        scored: list[tuple[float, str]] = []
        specialist_rejects = brow_reject_set | eye_reject_set | mouth_reject_set
        specialist_keeps = brow_keep_set | eye_keep_set | mouth_keep_set
        calibration = load_motion_feature_calibration()
        dataset_calibration = (calibration.get("datasets") or {}).get(dataset, {})
        emotion_rates = (dataset_calibration.get("emotion_au_rates") or {}).get(final_emotion, {})
        for raw_name in fallback_pool:
            name = normalize_au_name(raw_name)
            if not name:
                continue
            if name in specialist_rejects and name not in specialist_keeps:
                continue
            info = evidence_map.get(name) or {}
            present = _support_gate(info, threshold=1.0)
            score = max(
                float(info.get("score", 0.0) or 0.0),
                float(info.get("normalized_support", 0.0) or 0.0),
                float(info.get("raw_support", 0.0) or 0.0),
                float(info.get("combined_support", 0.0) or 0.0),
                float(info.get("geometry_support", 0.0) or 0.0),
                _present_bonus(info),
            )
            if present and _present_gate_enabled():
                score += 1.35
            if name in prior_set:
                score += 0.10
            if name in visual_set:
                score += 0.08
            if name in agent_keep_set or name in specialist_keeps:
                score += 0.16
            score += 0.12 * float(emotion_rates.get(name, 0.0) or 0.0)
            if name in agent_reject_set:
                score -= 0.18 if present else 0.42
            if score <= 0.0:
                continue
            scored.append((score, name))
        if not scored:
            if critic_specialist_fallback:
                rescued = _resolve_anatomical_conflicts(
                    critic_specialist_fallback[:1],
                    dataset=dataset,
                    evidence_map=evidence_map,
                    visual_set=visual_set,
                    agent_keep_set=agent_keep_set,
                    agent_reject_set=set(),
                    numeric=numeric,
                )
                if rescued:
                    return dedupe_keep_order([
                        *authoritative_brow,
                        *authoritative_eye,
                        *authoritative_mouth,
                        *rescued,
                    ])
            if dataset == "samm":
                eyelid_signal = str(numeric.get("eyelid_signal") or "").strip().lower()
                eye_metrics = numeric.get("eye_metrics") or {}
                mean_ratio = float(eye_metrics.get("mean_ratio", 1.0) or 1.0)
                min_ratio = float(eye_metrics.get("min_ratio", mean_ratio) or mean_ratio)
                gap_shrink = float(eye_metrics.get("mean_gap_shrink", 0.0) or 0.0)
                eye_backoff: list[tuple[float, str]] = []
                for name in ["upper lid raiser", "lid tightener", "eye closure"]:
                    info = evidence_map.get(name) or {}
                    support = max(
                        float(info.get("score", 0.0) or 0.0),
                        float(info.get("normalized_support", 0.0) or 0.0),
                        float(info.get("combined_support", 0.0) or 0.0),
                        float(info.get("geometry_support", 0.0) or 0.0),
                        _present_bonus(info),
                    )
                    score = support + 0.30 * float(emotion_rates.get(name, 0.0) or 0.0)
                    if name == "upper lid raiser":
                        if eyelid_signal == "widening":
                            score += 0.85
                            score += max(0.0, mean_ratio - 1.0) * 12.0
                            score += max(0.0, min_ratio - 1.0) * 8.0
                        elif mean_ratio >= 1.03 or min_ratio >= 1.02:
                            score += max(0.0, mean_ratio - 1.0) * 10.0
                            score += max(0.0, min_ratio - 1.0) * 6.0
                    elif name == "lid tightener":
                        if eyelid_signal == "narrowing":
                            score += 0.85
                        elif eyelid_signal == "stable" and mean_ratio <= 1.025 and gap_shrink >= 0.015:
                            score += 0.45
                        score += max(0.0, 1.0 - mean_ratio) * 12.0
                        score += max(0.0, 1.0 - min_ratio) * 16.0
                        score += 6.0 * gap_shrink
                    else:
                        if eyelid_signal == "closure":
                            score += 1.0
                        score += max(0.0, 0.93 - min_ratio) * 20.0
                    eye_backoff.append((score, name))
                eye_backoff.sort(reverse=True)
                if eye_backoff and eye_backoff[0][0] > 0.18:
                    rescued = _resolve_anatomical_conflicts(
                        [eye_backoff[0][1]],
                        dataset=dataset,
                        evidence_map=evidence_map,
                        visual_set=visual_set,
                        agent_keep_set=agent_keep_set,
                        agent_reject_set=set(),
                        numeric=numeric,
                    )
                    if rescued:
                        return dedupe_keep_order([*authoritative_brow, *authoritative_eye, *authoritative_mouth, *rescued])
            emotion_backoff = [
                name
                for name, _ in sorted(
                    emotion_rates.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                if name not in specialist_owned and name not in agent_reject_set
            ]
            if emotion_backoff:
                rescued = _resolve_anatomical_conflicts(
                    [emotion_backoff[0]],
                    dataset=dataset,
                    evidence_map=evidence_map,
                    visual_set=visual_set,
                    agent_keep_set=agent_keep_set,
                    agent_reject_set=agent_reject_set,
                    numeric=numeric,
                )
                return dedupe_keep_order([*authoritative_brow, *authoritative_eye, *authoritative_mouth, *rescued])
            return dedupe_keep_order([*authoritative_brow, *authoritative_eye, *authoritative_mouth])
        rescued = _resolve_anatomical_conflicts(
            [max(scored)[1]],
            dataset=dataset,
            evidence_map=evidence_map,
            visual_set=visual_set,
            agent_keep_set=agent_keep_set,
            agent_reject_set=agent_reject_set,
            numeric=numeric,
        )
        return dedupe_keep_order([*authoritative_brow, *authoritative_eye, *authoritative_mouth, *rescued])

    if len(canonical) <= 1:
        if not canonical:
            rescued = rescue_empty_result()
            if rescued:
                return rescued
            return dedupe_keep_order([*authoritative_brow, *authoritative_mouth])
        only_name = canonical[0]
        info = evidence_map.get(only_name) or {}
        override_strength = _agent_override_strength(info)
        if only_name in agent_reject_set and only_name not in agent_keep_set and agent_keep:
            ranked_keep = sorted(
                agent_keep,
                key=lambda name: (
                    _agent_override_strength(evidence_map.get(name) or {}),
                    name in visual_set,
                ),
                reverse=True,
            )
            if ranked_keep:
                return ranked_keep[:1]
        if only_name in agent_reject_set and only_name not in agent_keep_set:
            if not _override_gate(info, threshold=1.55 if dataset == "samm" else 1.35):
                return dedupe_keep_order([*authoritative_brow, *authoritative_eye, *authoritative_mouth])
        return dedupe_keep_order([*authoritative_brow, *authoritative_eye, *authoritative_mouth, *canonical])
    upper_face = {
        "inner brow raiser",
        "outer brow raiser",
        "brow lowerer",
        "upper lid raiser",
        "lid tightener",
        "cheek raiser",
        "eye closure",
    }
    calibration = load_motion_feature_calibration()
    dataset_calibration = (calibration.get("datasets") or {}).get(dataset, {})
    au_thresholds = dataset_calibration.get("au_thresholds") or {}
    emotion_au_rates = (dataset_calibration.get("emotion_au_rates") or {}).get(final_emotion, {})
    emotion_pair_rates = (dataset_calibration.get("emotion_pair_rates") or {}).get(final_emotion, {})

    upper_candidates: list[tuple[float, str, bool]] = []
    lower_candidates: list[tuple[float, str, bool]] = []
    for name in canonical:
        info = evidence_map.get(name) or {}
        if _dataset_blocks_au(dataset=dataset, au_name=name, info=info):
            continue
        support = _au_numeric_support(name, numeric)
        normalized_support = float(info.get("normalized_support", 0.0) or 0.0)
        present = _support_gate(info, threshold=1.0)
        calibrated = bool(info.get("calibrated"))
        reliability = _evidence_reliability(info)
        in_visual = name in visual_set
        in_prior = name in prior_set
        in_agent_keep = name in agent_keep_set
        in_agent_reject = name in agent_reject_set
        emotion_rate = float(emotion_au_rates.get(name, 0.0) or 0.0)
        threshold_info = au_thresholds.get(name) or {}
        precision = float(threshold_info.get("precision", 0.0) or 0.0)
        retention = 0.72 * min(support, 2.0) + 0.18 * emotion_rate
        if present and _present_gate_enabled():
            retention += 0.10
        if in_visual:
            retention += 0.09
        if in_prior:
            retention += 0.12
        if in_agent_keep:
            if _uses_calibrated_reliability_regularization(dataset) and calibrated:
                keep_scale = 0.25 + 0.75 * min(1.0, reliability / 0.35)
                retention += 0.18 * keep_scale
            else:
                retention += 0.18
        if in_agent_reject:
            retention -= 0.36 if dataset == "samm" else 0.24
        if _uses_calibrated_reliability_regularization(dataset) and calibrated:
            retention *= 0.55 + 0.45 * min(1.0, reliability / 0.35)
            if reliability < 0.18 and not in_visual:
                retention -= 0.10
        if _uses_calibrated_reliability_regularization(dataset) and in_visual and not present and normalized_support < 0.60:
            retention -= 0.20
        if dataset == "samm" and name in {"brow lowerer", "upper lip raiser", "nose wrinkler"} and not present and not in_prior:
            retention -= 0.18
        if dataset == "samm" and name == "outer brow raiser":
            corroborated = _samm_outer_brow_corroborated(info)
            if in_visual and not corroborated:
                retention -= 0.30
            if in_agent_keep and not corroborated:
                retention -= 0.18
        if precision > 0.0:
            retention += 0.04 * precision
        if in_agent_reject and not in_agent_keep:
            if not _override_gate(info, threshold=1.55 if dataset == "samm" else 1.35):
                continue

        if name in upper_face:
            if dataset == "samm" and name == "outer brow raiser":
                corroborated = _samm_outer_brow_corroborated(info)
                if in_visual and not corroborated:
                    continue
            gate = (
                present
                or normalized_support >= 0.95
                or (in_visual and in_prior and normalized_support >= 0.60)
                or (emotion_rate >= 0.25 and normalized_support >= 0.70)
            )
            if gate and retention >= (0.72 if dataset == "samm" else 0.60):
                upper_candidates.append((retention, name, present))
            continue
        gate = (
            (
                present
                and (
                    emotion_rate >= 0.10
                    or in_prior
                    or normalized_support >= 1.15
                )
            )
            or normalized_support >= 1.00
            or (in_visual and in_prior and normalized_support >= 0.72)
            or (emotion_rate >= 0.18 and normalized_support >= 0.82)
        )
        if name == "lip corner puller":
            gate = (
                (
                    present
                    and (
                        emotion_rate >= 0.16
                        or in_prior
                        or normalized_support >= 1.25
                    )
                )
                or normalized_support >= 1.10
                or (in_visual and in_prior and normalized_support >= 0.95)
            )
        if in_agent_keep:
            gate = gate or present or name in visual_set
        if gate and retention >= (0.76 if dataset == "samm" else 0.62):
            lower_candidates.append((retention, name, present))

    keep: list[str] = [name for _, name, _ in sorted(upper_candidates, reverse=True)]
    selected_lower: list[str] = []
    for retention, name, present in sorted(lower_candidates, reverse=True):
        if not selected_lower:
            selected_lower.append(name)
            continue
        if len(selected_lower) >= 2:
            continue
        pair_key = "|".join(sorted((selected_lower[0], name)))
        pair_rate = float(emotion_pair_rates.get(pair_key, 0.0) or 0.0)
        if present and (pair_rate >= 0.04 or retention >= 1.05):
            selected_lower.append(name)
        elif pair_rate >= 0.08 and retention >= 0.88:
            selected_lower.append(name)

    keep.extend(selected_lower)
    pruned = _resolve_anatomical_conflicts(
        dedupe_keep_order(keep),
        dataset=dataset,
        evidence_map=evidence_map,
        visual_set=visual_set,
        agent_keep_set=agent_keep_set,
        agent_reject_set=agent_reject_set,
        numeric=numeric,
    )
    if pruned:
        return dedupe_keep_order([*authoritative_brow, *authoritative_eye, *authoritative_mouth, *mouth_override_candidates, *pruned])
    if agent_keep:
        ranked_keep = sorted(
            agent_keep,
            key=lambda name: (
                _agent_override_strength(evidence_map.get(name) or {}),
                name in visual_set,
                name in prior_set,
            ),
            reverse=True,
        )
        if ranked_keep:
            rescued = _resolve_anatomical_conflicts(
                ranked_keep[: max(1, min(2, len(ranked_keep)))],
                dataset=dataset,
                evidence_map=evidence_map,
                visual_set=visual_set,
                agent_keep_set=agent_keep_set,
                agent_reject_set=agent_reject_set,
                numeric=numeric,
            )
            if rescued:
                return dedupe_keep_order([*authoritative_brow, *authoritative_eye, *authoritative_mouth, *mouth_override_candidates, *rescued])

    fallback_scores = []
    for name in canonical:
        info = evidence_map.get(name) or {}
        value = max(
            float(info.get("score", 0.0) or 0.0),
            float(info.get("normalized_support", 0.0) or 0.0),
            0.15 if name in visual_set else 0.0,
            0.10 if name in prior_set else 0.0,
        )
        fallback_scores.append((value, name))
    fallback_scores.sort(reverse=True)
    fallback = [fallback_scores[0][1]] if fallback_scores else canonical[:1]
    rescued = _resolve_anatomical_conflicts(
        fallback,
        dataset=dataset,
        evidence_map=evidence_map,
        visual_set=visual_set,
        agent_keep_set=agent_keep_set,
        agent_reject_set=agent_reject_set,
        numeric=numeric,
    )
    if rescued:
        return dedupe_keep_order([*authoritative_brow, *authoritative_eye, *authoritative_mouth, *mouth_override_candidates, *rescued])
    rescued = rescue_empty_result()
    if rescued:
        return rescued
    return dedupe_keep_order([*authoritative_brow, *authoritative_eye, *authoritative_mouth, *mouth_override_candidates])


def _build_postfusion_reasoning(
    *,
    final_emotion: str,
    pred_au_names: list[str],
    pre_fusion_emotion: str,
    visual_au_names: list[str],
    knowledge_prior: dict[str, Any],
    fusion_policy: dict[str, Any],
    numeric: dict[str, Any],
) -> tuple[str, str]:
    brow_aus, mouth_aus = _split_aus(pred_au_names)
    numeric_evidence = _numeric_au_evidence_map(numeric, "")
    landmark_aus = {
        name
        for name, info in numeric_evidence.items()
        if _support_gate(info, threshold=1.0)
    }
    prior_aus = set(knowledge_prior.get("suggested_aus") or [])
    visual_aus = set(visual_au_names or [])

    def support_text(names: list[str]) -> str:
        if not names:
            return ""
        numeric_supported = [name for name in names if name in landmark_aus]
        visual_supported = [name for name in names if name in visual_aus]
        prior_supported = [name for name in names if name in prior_aus]
        parts = []
        if numeric_supported:
            parts.append(f"numeric cues support {_join_names(numeric_supported)}")
        if visual_supported and visual_supported != numeric_supported:
            parts.append(f"the upstream AU answer also pointed to {_join_names(visual_supported)}")
        if prior_supported and prior_supported not in (numeric_supported, visual_supported):
            parts.append(f"the knowledge prior suggested {_join_names(prior_supported)}")
        return "; ".join(parts)

    emotion_source = fusion_policy.get("emotion_source", "fusion")
    au_source = fusion_policy.get("au_source", "visual")
    brow_reasoning = ""
    mouth_reasoning = ""

    if brow_aus:
        brow_reasoning = (
            f"The final upper-face interpretation is {final_emotion}, with {_join_names(brow_aus)} retained after fusion."
        )
        support = support_text(brow_aus)
        if support:
            brow_reasoning += f" In particular, {support}."
        if au_source != "visual":
            brow_reasoning += f" The final AU set came from {au_source.replace('_', ' ')} evidence rather than the raw AU answer alone."
    else:
        brow_reasoning = (
            f"No upper-face AU was kept in the final output for {final_emotion}."
        )
        if visual_au_names:
            brow_reasoning += " This avoids claiming brow evidence that did not survive fusion."

    if mouth_aus:
        mouth_reasoning = (
            f"The final mouth-region interpretation keeps {_join_names(mouth_aus)} for the {final_emotion} label."
        )
        support = support_text(mouth_aus)
        if support:
            mouth_reasoning += f" Here, {support}."
        if emotion_source != "answer_fallback" or final_emotion != pre_fusion_emotion:
            mouth_reasoning += f" The final emotion stayed at {final_emotion} after {emotion_source.replace('_', ' ')} scoring."
    else:
        mouth_reasoning = (
            f"No mouth AU was kept in the final output for {final_emotion}."
        )
        if knowledge_prior.get("suggested_emotion"):
            mouth_reasoning += " The explanation is therefore kept conservative instead of expanding unsupported mouth cues."

    if not pred_au_names:
        brow_reasoning = (
            f"The final emotion is {final_emotion}, but no AU was retained with enough support after fusion."
        )
        mouth_reasoning = "The explanation is intentionally conservative because the fused AU evidence is weak."

    return brow_reasoning, mouth_reasoning


def summarize_clip(
    clip: dict,
    clip_cache: dict[tuple, dict],
    answers: dict[tuple, str],
    *,
    ablation: AblationConfig | None = None,
) -> dict[str, Any]:
    ablation = ablation or AblationConfig()
    with ablation_context(ablation):
        fine_answer = ""
        au_answer = ""
        analysis_answer = ""
        vqa_rows = []
        answer_map: dict[str, str] = {}
        dataset = clip["dataset"]
        cache = clip_cache.get((clip["clip_id"], str(clip["path"])), {})
        numeric = ensure_au_evidence(cache.get("numeric_features", {}), dataset)
        agent_trace = cache.get("agent_trace", {})
        has_agent_trace = any(bool(value) for value in agent_trace.values()) if isinstance(agent_trace, dict) else False
        knowledge_prior = {} if ablation.disable_prior else derive_knowledge_prior(dataset, numeric)

        for qa in clip.get("vqa_questions", []):
            key = (clip["clip_id"], qa["q"])
            pred = answers.get(key, "other")
            q_lower = qa["q"].lower()
            if "fine-grained expression" in q_lower and not fine_answer:
                fine_answer = pred
            elif qa["q"] in ("What is the action unit?", "What are the action units?") and len(pred) > len(au_answer):
                au_answer = pred
            elif any(token in q_lower for token in ("analys", "describe", "comprehensive", "provide")) and not analysis_answer:
                analysis_answer = pred

        if not fine_answer and analysis_answer:
            fine_answer = analysis_answer

        visual_au_names = extract_au_names(au_answer or analysis_answer)
        if not visual_au_names:
            visual_au_names = clip_reasoner_aus(cache)
        visual_au_names = _strip_samm_specialist_aux_aus(
            visual_au_names,
            dataset=dataset,
            agent_trace=agent_trace,
        )
        if not ablation.disable_prior:
            knowledge_prior = _sanitize_knowledge_prior_for_specialists(
                knowledge_prior,
                dataset=dataset,
                agent_trace=agent_trace,
                numeric=numeric,
            )

        final_emotion, pred_au_names, fusion_policy = fuse_predictions(
            dataset=dataset,
            fine_answer=fine_answer,
            au_answer=au_answer,
            analysis_answer=analysis_answer,
            visual_au_names=visual_au_names,
            knowledge_prior=knowledge_prior,
            numeric=numeric,
            agent_trace=agent_trace,
            allow_answer_emotion_fallback=not has_agent_trace,
        )
        pre_fusion_emotion = str(fusion_policy.get("pre_fusion_emotion") or "").strip().lower()
        pred_au_names = _prune_predicted_aus(
            dedupe_keep_order(pred_au_names),
            dataset=dataset,
            final_emotion=final_emotion,
            visual_au_names=visual_au_names,
            knowledge_prior=knowledge_prior,
            numeric=numeric,
            agent_trace=agent_trace,
        )
        post_prune_scores = _post_prune_emotion_scores(
            dataset=dataset,
            pred_au_names=pred_au_names,
            numeric=numeric,
            agent_trace=agent_trace,
        )
        if post_prune_scores:
            post_prune_numeric_evidence = _numeric_au_evidence_map(numeric, dataset)
            post_prune_agent_guidance = _agent_guidance(agent_trace, dataset)
            post_prune_block_override = False
            if dataset == "samm" and pred_au_names == ["lid tightener"] and final_emotion == "happiness":
                prior_scores = {
                    normalize_emotion_for_dataset(str(name or ""), dataset): float(score)
                    for name, score in (knowledge_prior.get("scores") or {}).items()
                    if normalize_emotion_for_dataset(str(name or ""), dataset)
                }
                top_prior_emotion = ""
                if prior_scores:
                    top_prior_emotion = max(prior_scores.items(), key=lambda item: item[1])[0]
                eye_keep = set((_agent_guidance(agent_trace, dataset).get("eye_keep_aus") or []))
                if top_prior_emotion == "happiness" and not (eye_keep & _SAMM_EYE_AGENT_AUS):
                    post_prune_block_override = True
            if (
                dataset == "samm"
                and final_emotion == "happiness"
                and "lip corner puller" in pred_au_names
                and not _samm_lip_corner_puller_rescue_allowed(
                    numeric_evidence=post_prune_numeric_evidence,
                    agent_guidance=post_prune_agent_guidance,
                    emotion_scores=post_prune_scores,
                )
            ):
                post_prune_block_override = True
            ordered_post_prune = sorted(
                post_prune_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            top_emotion, top_score = ordered_post_prune[0]
            current_score = float(post_prune_scores.get(final_emotion, 0.0) or 0.0)
            second_score = ordered_post_prune[1][1] if len(ordered_post_prune) > 1 else 0.0
            relaxed_happiness_override = (
                dataset == "samm"
                and top_emotion == "happiness"
                and top_emotion != final_emotion
                and "lip corner puller" in pred_au_names
                and top_score >= 0.90
                and top_score - current_score >= 0.05
            )
            if not post_prune_block_override and top_score >= 0.58 and (
                relaxed_happiness_override
                or (
                    top_emotion != final_emotion
                    and (top_score - current_score >= 0.12 or top_score - second_score >= 0.08)
                )
            ):
                final_emotion = top_emotion
                fusion_policy["emotion_source"] = "post_prune_au_consistency"
                fusion_policy["emotion_override"] = True
        fusion_policy["post_prune_emotion_scores"] = post_prune_scores
        pred_aus = au_names_to_pred_string(pred_au_names)
        coarse = FINE_TO_COARSE.get(final_emotion, "negative")
        brow_reasoning, mouth_reasoning = _build_postfusion_reasoning(
            final_emotion=final_emotion,
            pred_au_names=pred_au_names,
            pre_fusion_emotion=pre_fusion_emotion,
            visual_au_names=visual_au_names,
            knowledge_prior=knowledge_prior,
            fusion_policy=fusion_policy,
            numeric=numeric,
        )

        combined = {
            "final_emotion": final_emotion,
            "pred_aus": pred_aus,
            "coarse": coarse,
            "numeric_features": numeric,
            "reasoning": brow_reasoning[:300] + "\n" + mouth_reasoning[:300],
        }
        vqa_rows = vqa_probe(clip, combined)
        answer_map = {row["question"]: row["pred"] for row in vqa_rows}

        return {
            "clip_id": clip["clip_id"],
            "dataset": dataset,
            "pred": final_emotion,
            "final_emotion": final_emotion,
            "pred_aus": pred_aus,
            "pred_au_names": pred_au_names,
            "coarse": coarse,
            "brow_reasoning": brow_reasoning,
            "mouth_reasoning": mouth_reasoning,
            "numeric_features": numeric,
            "knowledge_prior": knowledge_prior,
            "fusion_policy": fusion_policy,
            "pre_fusion_emotion": pre_fusion_emotion,
            "visual_pred_au_names": visual_au_names,
            "agent_trace": agent_trace,
            "vqa_rows": vqa_rows,
            "answer_map": answer_map,
        }


def clip_reasoner_aus(cache: dict[str, Any]) -> list[str]:
    names = []
    for text in (cache.get("brow_reasoning", ""), cache.get("mouth_reasoning", "")):
        names.extend(extract_au_names(text))
    ordered = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _to_builtin_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_builtin_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin_jsonable(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return _to_builtin_jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_builtin_jsonable(row), ensure_ascii=False) + "\n")


def load_done_clip_ids(results_file: Path) -> set[str]:
    done = set()
    if not results_file.exists():
        return done
    with open(results_file, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(json.loads(line)["clip_id"])
            except Exception:
                continue
    return done


def build_training_output_rows(summary: dict, clip: dict) -> tuple[dict, dict | None]:
    pred = summary["pred"]
    gt = clip["gt_fine"].lower()
    pred_aus = summary["pred_aus"]
    ok = pred == gt
    au_wrong = aus_mismatch(pred_aus, clip.get("gt_aus", []))
    _, au_recall, au_f1_value = aus_f1(pred_aus, clip.get("gt_aus", []))
    result_row = {
        "clip_id": clip["clip_id"],
        "dataset": clip.get("dataset", ""),
        "subject": clip.get("subject", ""),
        "gt": gt,
        "pred": pred,
        "pred_aus": pred_aus,
        "correct": ok,
        "au_mismatch": au_wrong,
        "au_recall": round(au_recall, 3),
        "au_f1": round(au_f1_value, 3),
        "final_emotion": summary["final_emotion"],
        "critic_changed": False,
        "numeric_features": summary.get("numeric_features", {}),
        "knowledge_prior": summary.get("knowledge_prior", {}),
        "fusion_policy": summary.get("fusion_policy", {}),
        "pre_fusion_emotion": summary.get("pre_fusion_emotion", ""),
        "visual_pred_au_names": summary.get("visual_pred_au_names", []),
        "agent_trace": summary.get("agent_trace", {}),
        "batch": 0,
    }
    wrong_row = None
    if not ok or au_wrong:
        wrong_row = {
            "clip_id": clip["clip_id"],
            "dataset": clip.get("dataset", ""),
            "subject": clip.get("subject", ""),
            "gt_fine": gt,
            "gt_coarse": clip.get("gt_coarse", ""),
            "gt_aus": clip.get("gt_aus", []),
            "pred": pred,
            "correct": ok,
            "au_mismatch": au_wrong,
            "final_emotion": summary["final_emotion"],
            "pred_aus": pred_aus,
            "stage1_emotion": "",
            "critic_changed": False,
            "critic_note": "",
            "brow_reasoning": summary.get("brow_reasoning", ""),
            "mouth_reasoning": summary.get("mouth_reasoning", ""),
            "numeric_features": summary.get("numeric_features", {}),
            "knowledge_prior": summary.get("knowledge_prior", {}),
            "fusion_policy": summary.get("fusion_policy", {}),
            "pre_fusion_emotion": summary.get("pre_fusion_emotion", ""),
            "visual_pred_au_names": summary.get("visual_pred_au_names", []),
            "agent_trace": summary.get("agent_trace", {}),
        }
    return result_row, wrong_row


def normalize_answer_for_question(
    question: str,
    raw_pred: str,
    final_emotion: str,
    coarse: str,
    pred_au_names: list[str],
    numeric: dict[str, Any] | None = None,
) -> str:
    def side_score(region: str, regions: dict[str, dict[str, Any]], residual: dict[str, dict[str, Any]]) -> float:
        score = 0.0
        info = regions.get(region) or {}
        if info:
            score = max(score, float(info.get("active", 0.0) or 0.0) * float(info.get("mag", 0.0) or 0.0))
        info = residual.get(region) or {}
        if info:
            score = max(score, 18.0 * float(info.get("active", 0.0) or 0.0) * float(info.get("mag", 0.0) or 0.0))
        return score

    def infer_au_side(au_name: str) -> str:
        numeric_local = numeric or {}
        regions = numeric_local.get("landmark_au_regions") or {}
        residual = numeric_local.get("landmark_residual_au_regions") or {}
        name = normalize_au_name(au_name)
        pair_map = {
            "inner brow raiser": ("AU1_inner_brow_subL", "AU1_inner_brow_subR"),
            "outer brow raiser": ("AU2_brow_outer_subL", "AU2_brow_outer_subR"),
            "lip corner puller": ("AU12_corner_subL", "AU12_corner_subR"),
        }
        if name in pair_map:
            left_region, right_region = pair_map[name]
            left_score = side_score(left_region, regions, residual)
            right_score = side_score(right_region, regions, residual)
        elif name in {"upper lip raiser", "lip tightener", "chin raiser", "jaw drop", "jaw clencher", "lips part"}:
            left_score = side_score("AU12_corner_subL", regions, residual)
            right_score = side_score("AU12_corner_subR", regions, residual)
        else:
            left_score = side_score("AU1_inner_brow_subL", regions, residual) + side_score("AU2_brow_outer_subL", regions, residual)
            right_score = side_score("AU1_inner_brow_subR", regions, residual) + side_score("AU2_brow_outer_subR", regions, residual)
        return "left" if left_score >= right_score else "right"

    q_lower = (question or "").lower()
    au_list = ", ".join(pred_au_names)
    primary_au = pred_au_names[0] if pred_au_names else ""
    if "what are the action units present" in q_lower:
        return (
            f"The action units present are: {au_list}. Therefore, the fine-grained "
            f"expression class is {final_emotion}, and the coarse expression class is {coarse}."
            if pred_au_names
            else f"The action units present are: none. Therefore, the fine-grained expression class is {final_emotion}, and the coarse expression class is {coarse}."
        )
    if "what is the action unit present" in q_lower:
        primary = primary_au or "none"
        return (
            f"The action unit present is: {primary}. Therefore, the fine-grained "
            f"expression class is {final_emotion}, and the coarse expression class is {coarse}."
        )
    if "located on the left or right face" in q_lower:
        m = re.search(r"is the action unit (.+?) located on the left or right face", q_lower)
        asked = normalize_au_name(m.group(1).strip()) if m else ""
        if asked and asked in {normalize_au_name(name) for name in pred_au_names}:
            return infer_au_side(asked)
        return "left"
    if "coarse expression" in q_lower:
        return coarse
    if "fine-grained expression" in q_lower:
        return final_emotion
    if question == "What is the action unit?":
        return primary_au
    if question == "What are the action units?":
        return au_list
    if "is the action unit" in q_lower and "shown on the face" in q_lower:
        asked = ""
        m = re.search(r"is the action unit (.+?) shown", q_lower)
        if m:
            asked = normalize_au_name(m.group(1).strip())
        return "yes" if asked and asked in {normalize_au_name(name) for name in pred_au_names} else "no"
    if any(token in q_lower for token in ("analys", "describe", "provide", "comprehensive")):
        return (
            f"The action units present are: {au_list}. Therefore, the fine-grained "
            f"expression class is {final_emotion}, and the coarse expression class is {coarse}."
            if pred_au_names
            else f"The action units present are: none. Therefore, the fine-grained expression class is {final_emotion}, and the coarse expression class is {coarse}."
        )
    return raw_pred


def vqa_probe(info: dict, combined: dict) -> list[dict]:
    questions = info.get("vqa_questions", [])
    if not questions:
        return []
    pred_emo = normalize_emotion_for_dataset(
        extract_emotion(combined.get("final_emotion", "")),
        info["dataset"],
    )
    pred_aus = combined.get("pred_aus", "")
    pred_au_names = extract_au_names(pred_aus)
    coarse = combined.get("coarse", FINE_TO_COARSE.get(pred_emo, "negative"))
    rows = []
    for qa in questions:
        pred_ans = normalize_answer_for_question(
            question=qa["q"],
            raw_pred="other",
            final_emotion=pred_emo,
            coarse=coarse,
            pred_au_names=pred_au_names,
            numeric=combined.get("numeric_features", {}),
        )
        rows.append(
            {
                "clip_id": info["clip_id"],
                "question": qa["q"],
                "pred": pred_ans,
                "gt": qa.get("a", ""),
            }
        )
    return rows
