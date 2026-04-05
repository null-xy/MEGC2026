from __future__ import annotations

import inspect
import json
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

try:
    from .config import AblationConfig
except ImportError:
    from config import AblationConfig

try:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover
    AutoTokenizer = None
    LLM = None
    SamplingParams = None

try:
    from .formatting import DATASET_ALLOWED_EMOTIONS, FINE_TO_COARSE, normalize_emotion_for_dataset
except ImportError:
    from formatting import DATASET_ALLOWED_EMOTIONS, FINE_TO_COARSE, normalize_emotion_for_dataset

try:
    from .knowledge import derive_knowledge_prior
except ImportError:
    from knowledge import derive_knowledge_prior

try:
    from .numeric_features import ensure_au_evidence
except ImportError:
    from numeric_features import ensure_au_evidence

try:
    from .calibration import load_motion_feature_calibration
except ImportError:
    from calibration import load_motion_feature_calibration


@contextmanager
def _temporary_visible_devices(device: str):
    import os

    if not device:
        yield
        return
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def normalize_answer_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s,]", "", text)
    return text.strip()


def _normalize_au_token(text: str) -> str:
    token = normalize_answer_text(text)
    token = re.sub(r"^au\s*\d+[_\s-]*", "", token)
    token = token.replace("_", " ")
    token = token.replace(" and based on it", "").replace(" and based on them", "")
    return token.strip(" ,")


_AGENT_AU_ALIASES = {
    "brow lowerer": "brow lowerer",
    "inner brow raiser": "inner brow raiser",
    "inner brow subr": "inner brow raiser",
    "inner brow subl": "inner brow raiser",
    "outer brow raiser": "outer brow raiser",
    "brow outer subr": "outer brow raiser",
    "brow outer subl": "outer brow raiser",
    "brow outer sub": "outer brow raiser",
    "upper lid raiser": "upper lid raiser",
    "lid tightener": "lid tightener",
    "eye closure": "eye closure",
    "cheek raiser": "cheek raiser",
    "nose wrinkler": "nose wrinkler",
    "upper lip raiser": "upper lip raiser",
    "upper lip": "upper lip raiser",
    "lip corner puller": "lip corner puller",
    "corner subr": "lip corner puller",
    "corner subl": "lip corner puller",
    "lip corner depressor": "lip corner depressor",
    "lip stretcher": "lip stretcher",
    "dimpler": "dimpler",
    "chin raiser": "chin raiser",
    "chin": "chin raiser",
    "lip tightener": "lip tightener",
    "lip center": "lip tightener",
    "lips part": "lips part",
    "jaw drop": "jaw drop",
}

_BROW_AUS = {"inner brow raiser", "outer brow raiser", "brow lowerer"}
_BROW_FAMILY_WINNERS = {"raise_family", "lower_family", "no_reliable_brow_au"}
_BROW_AUDIT_DECISIONS = {"approve", "downgrade"}
_MOUTH_AUS = {
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
_ORDERED_MOUTH_AUS = [
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
]
_EYE_AUS = {"upper lid raiser", "lid tightener", "eye closure"}
_EYE_STATES = {"widening", "narrowing", "closure", "no_reliable_eye_au"}
_HEAD_MOTION_PRESENCE = {"yes", "no", "uncertain"}
_MOUTH_AU_NUMBER_ALIASES = {
    "9": "nose wrinkler",
    "10": "upper lip raiser",
    "12": "lip corner puller",
    "14": "dimpler",
    "15": "lip corner depressor",
    "17": "chin raiser",
    "20": "lip stretcher",
    "23": "lip tightener",
    "25": "lips part",
    "26": "jaw drop",
}
_HEAD_MOTION_MAGNITUDES = {"small", "moderate", "large", "uncertain"}
_HEAD_MOTION_DIRECTIONS = {"upward", "downward", "none", "uncertain"}
_HEAD_MOTION_RISKS = {"low", "medium", "high", "uncertain"}


def _canonicalize_agent_au_name(text: str) -> str:
    token = _normalize_au_token(text)
    if not token:
        return ""
    if token in _AGENT_AU_ALIASES:
        return _AGENT_AU_ALIASES[token]
    simplified = re.sub(r"\s+", " ", token.replace("-", " ")).strip()
    if simplified in _AGENT_AU_ALIASES:
        return _AGENT_AU_ALIASES[simplified]
    return token


def _extract_au_list(text: str) -> list[str]:
    text = normalize_answer_text(text)
    if "the action units present are" in text:
        text = text.split("the action units present are", 1)[1]
    elif "the action unit present is" in text:
        text = text.split("the action unit present is", 1)[1]
    if "therefore the finegrained expression class is" in text:
        text = text.split("therefore the finegrained expression class is", 1)[0]
    items = []
    for part in text.split(","):
        token = _normalize_au_token(part)
        if token and token not in items:
            items.append(token)
    return items


def _extract_compound_parts(text: str) -> tuple[list[str], str, str]:
    text = normalize_answer_text(text)
    aus = _extract_au_list(text)
    emotion = ""
    coarse = ""
    m = re.search(r"finegrained expression class is ([a-z]+)", text)
    if m:
        emotion = m.group(1)
    m = re.search(r"coarse expression class is ([a-z]+)", text)
    if m:
        coarse = m.group(1)
    return aus, emotion, coarse


def compute_exact_match(question: str, gt: str, pred: str) -> bool:
    q_lower = normalize_answer_text(question)
    gt_n = normalize_answer_text(gt)
    pred_n = normalize_answer_text(pred)

    if "located on the left or right face" in q_lower:
        return gt_n in {"left", "right"} and pred_n == gt_n
    if "shown on the face" in q_lower:
        return pred_n == gt_n
    if question == "What are the action units?":
        return set(_extract_au_list(gt)) == set(_extract_au_list(pred))
    if question == "What is the action unit?":
        gt_aus = _extract_au_list(gt)
        pred_aus = _extract_au_list(pred)
        if gt_aus or pred_aus:
            return bool(gt_aus) and bool(pred_aus) and gt_aus[0] == pred_aus[0]
        return gt_n == pred_n
    if "what are the action units present" in q_lower or "what is the action unit present" in q_lower:
        gt_aus, gt_emotion, gt_coarse = _extract_compound_parts(gt)
        pred_aus, pred_emotion, pred_coarse = _extract_compound_parts(pred)
        return set(gt_aus) == set(pred_aus) and gt_emotion == pred_emotion and gt_coarse == pred_coarse
    if any(token in q_lower for token in ("analys", "describe", "provide", "comprehensive")):
        gt_aus, gt_emotion, gt_coarse = _extract_compound_parts(gt)
        pred_aus, pred_emotion, pred_coarse = _extract_compound_parts(pred)
        if gt_emotion or pred_emotion or gt_aus or pred_aus:
            return set(gt_aus) == set(pred_aus) and gt_emotion == pred_emotion and gt_coarse == pred_coarse
    return gt_n == pred_n


def _safe_json_loads(text: str) -> dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def _json_dumps_compact(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return "{}"


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for item in items:
        key = _canonicalize_agent_au_name(item)
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _extract_agent_aus(value: Any) -> list[str]:
    if isinstance(value, list):
        return _dedupe_keep_order([str(item) for item in value])
    if isinstance(value, str):
        parts = re.split(r"[,\n;]+", value)
        return _dedupe_keep_order(parts)
    return []


def _sanitize_brow_family_winner(value: Any) -> str:
    text = normalize_answer_text(str(value or "")).replace(" ", "_")
    if text in _BROW_FAMILY_WINNERS:
        return text
    if "raise" in text and "lower" not in text:
        return "raise_family"
    if "lower" in text or "corrugator" in text:
        return "lower_family"
    if any(token in text for token in ("null", "no_reliable", "no_brow", "none", "uncertain")):
        return "no_reliable_brow_au"
    return ""


def _sanitize_brow_evidence_status(value: Any) -> str:
    text = normalize_answer_text(str(value or "")).replace(" ", "_")
    if text in {"confirmed", "ambiguous", "pose_confounded", "absent"}:
        return text
    if any(token in text for token in ("pose", "motion", "confound", "drift")):
        return "pose_confounded"
    if any(token in text for token in ("ambiguous", "uncertain", "mixed", "weak", "unclear")):
        return "ambiguous"
    if any(token in text for token in ("absent", "none", "no_evidence", "not_present")):
        return "absent"
    if any(token in text for token in ("confirmed", "supported", "reliable")):
        return "confirmed"
    return ""


def _sanitize_brow_numeric_support(value: Any) -> str:
    text = normalize_answer_text(str(value or "")).replace(" ", "_")
    if text in {"strong", "moderate", "weak", "none"}:
        return text
    if any(token in text for token in ("strong", "clear", "robust")):
        return "strong"
    if any(token in text for token in ("moderate", "partial", "some")):
        return "moderate"
    if any(token in text for token in ("weak", "slight", "limited", "subtle")):
        return "weak"
    if any(token in text for token in ("none", "absent", "no_evidence", "no_support")):
        return "none"
    return ""


def _sanitize_brow_pose_risk(value: Any) -> str:
    text = normalize_answer_text(str(value or "")).replace(" ", "_")
    if text in {"low", "medium", "high"}:
        return text
    if any(token in text for token in ("high", "severe", "large")):
        return "high"
    if any(token in text for token in ("medium", "moderate")):
        return "medium"
    if any(token in text for token in ("low", "small", "minimal")):
        return "low"
    return ""


def _sanitize_brow_arbiter_output(value: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(value or {})
    family_winner = _sanitize_brow_family_winner(value.get("family_winner"))
    evidence_status = _sanitize_brow_evidence_status(
        value.get("local_evidence_status") or value.get("evidence_status")
    )
    numeric_support = _sanitize_brow_numeric_support(
        value.get("numeric_support_assessment") or value.get("support_assessment")
    )
    pose_risk = _sanitize_brow_pose_risk(value.get("pose_risk"))
    keep = [au for au in _extract_agent_aus(value.get("brow_keep_aus") or value.get("keep_aus")) if au in _BROW_AUS]
    drop = [au for au in _extract_agent_aus(value.get("brow_drop_aus") or value.get("drop_aus")) if au in _BROW_AUS]
    state = normalize_answer_text(str(value.get("brow_state") or value.get("state") or "")).replace(" ", "_")
    normalized_state_label = _normalize_au_token(state.replace("_", " "))
    if normalized_state_label in _BROW_AUS and normalized_state_label not in keep:
        keep = [normalized_state_label, *keep]
        state = "local_brow_action"
    if state not in {"local_brow_action", "no_reliable_brow_au"}:
        if keep:
            state = "local_brow_action"
        elif family_winner == "no_reliable_brow_au":
            state = "no_reliable_brow_au"
        else:
            state = "no_reliable_brow_au"
    if evidence_status in {"ambiguous", "pose_confounded", "absent"}:
        state = "no_reliable_brow_au"
    if state == "no_reliable_brow_au":
        keep = []
        if not drop:
            drop = sorted(_BROW_AUS)
    elif not keep and family_winner == "no_reliable_brow_au":
        state = "no_reliable_brow_au"
        drop = sorted(_BROW_AUS)
    if state == "local_brow_action" and evidence_status == "absent":
        state = "no_reliable_brow_au"
        keep = []
        if not drop:
            drop = sorted(_BROW_AUS)
    summary = str(value.get("brow_summary") or value.get("summary") or "").strip()
    confirmation_basis = str(value.get("confirmation_basis") or "").strip()
    why_not = str(value.get("why_not_no_reliable_brow_au") or "").strip()
    if keep and not confirmation_basis and summary:
        confirmation_basis = summary.split(". ", 1)[0].strip().rstrip(".")
    if keep and not why_not and confirmation_basis:
        why_not = (
            "The cited brow-card evidence is specific enough to survive a no_reliable_brow_au reading."
        )
    return {
        "family_winner": family_winner or ("no_reliable_brow_au" if not keep else ""),
        "local_evidence_status": evidence_status or ("confirmed" if keep else "absent"),
        "numeric_support_assessment": numeric_support or ("moderate" if keep else "none"),
        "pose_risk": pose_risk or "medium",
        "brow_state": state,
        "brow_keep_aus": keep,
        "brow_drop_aus": drop,
        "confirmation_basis": confirmation_basis,
        "why_not_no_reliable_brow_au": why_not,
        "brow_summary": summary,
        "confidence": str(value.get("confidence") or "").strip(),
    }


def _sanitize_brow_skeptic_output(value: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(value or {})
    decision = normalize_answer_text(str(value.get("audit_decision") or "")).replace(" ", "_")
    if decision not in _BROW_AUDIT_DECISIONS:
        if any(token in decision for token in ("downgrade", "reject", "demote", "no_reliable", "skeptical")):
            decision = "downgrade"
        else:
            decision = "approve"
    return {
        "audit_decision": decision,
        "main_issue": str(value.get("main_issue") or "").strip(),
        "critic_summary": str(value.get("critic_summary") or value.get("summary") or "").strip(),
        "confidence": str(value.get("confidence") or "").strip(),
    }


def _apply_brow_skeptic_review(
    brow_arbiter: dict[str, Any] | None,
    brow_skeptic: dict[str, Any] | None,
) -> dict[str, Any]:
    reviewed = dict(brow_arbiter or {})
    skeptic = dict(brow_skeptic or {})
    audit_decision = normalize_answer_text(str(skeptic.get("audit_decision") or "")).replace(" ", "_")
    if audit_decision not in _BROW_AUDIT_DECISIONS:
        audit_decision = "approve"
    main_issue = str(skeptic.get("main_issue") or "").strip()
    critic_summary = str(skeptic.get("critic_summary") or "").strip()
    reviewed["brow_audit_decision"] = audit_decision
    reviewed["brow_audit_issue"] = main_issue
    reviewed["brow_audit_summary"] = critic_summary
    if audit_decision != "downgrade":
        return reviewed

    existing_summary = str(reviewed.get("brow_summary") or "").strip()
    downgrade_note = critic_summary or main_issue or "Brow Skeptic downgraded the brow decision."
    reviewed["family_winner"] = "no_reliable_brow_au"
    reviewed["local_evidence_status"] = "ambiguous"
    reviewed["numeric_support_assessment"] = "weak"
    reviewed["pose_risk"] = reviewed.get("pose_risk") or "high"
    reviewed["brow_state"] = "no_reliable_brow_au"
    reviewed["brow_keep_aus"] = []
    reviewed["brow_drop_aus"] = sorted(_BROW_AUS)
    reviewed["confirmation_basis"] = ""
    reviewed["why_not_no_reliable_brow_au"] = ""
    reviewed["brow_summary"] = (
        f"{existing_summary} Brow Skeptic downgrade: {downgrade_note}".strip()
        if existing_summary
        else f"Brow Skeptic downgrade: {downgrade_note}"
    )
    return reviewed


def _sanitize_mouth_arbiter_output(value: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(value or {})
    evidence_status = _sanitize_brow_evidence_status(
        value.get("local_evidence_status") or value.get("evidence_status")
    )
    numeric_support = _sanitize_brow_numeric_support(
        value.get("numeric_support_assessment") or value.get("support_assessment")
    )
    pose_risk = _sanitize_brow_pose_risk(value.get("pose_risk"))
    keep = [
        au for au in _extract_agent_aus(value.get("mouth_keep_aus") or value.get("keep_aus"))
        if au in _MOUTH_AUS
    ]
    drop = [
        au for au in _extract_agent_aus(value.get("mouth_drop_aus") or value.get("drop_aus"))
        if au in _MOUTH_AUS
    ]
    state = normalize_answer_text(str(value.get("mouth_state") or value.get("state") or "")).replace(" ", "_")
    normalized_state_label = _normalize_au_token(state.replace("_", " "))
    if normalized_state_label in _MOUTH_AUS and normalized_state_label not in keep:
        keep = [normalized_state_label, *keep]
        state = "local_mouth_action"
    if state not in {"local_mouth_action", "no_reliable_mouth_au"}:
        state = "local_mouth_action" if keep else "no_reliable_mouth_au"
    summary = str(value.get("mouth_summary") or value.get("summary") or "").strip()
    confirmation_basis = str(value.get("confirmation_basis") or "").strip()
    why_not = str(value.get("why_not_no_reliable_mouth_au") or "").strip()

    def _extract_textual_mouth_aus(*texts: str) -> list[str]:
        recovered: list[str] = []
        for text in texts:
            normalized = normalize_answer_text(text)
            if not normalized:
                continue
            for number, canonical in _MOUTH_AU_NUMBER_ALIASES.items():
                if re.search(rf"\bau\s*{number}\b", normalized):
                    recovered.append(canonical)
            for canonical in _ORDERED_MOUTH_AUS:
                if canonical in normalized:
                    recovered.append(canonical)
        return _dedupe_keep_order(recovered)

    def _supports_local_mouth_action(text: str) -> bool:
        normalized = normalize_answer_text(text)
        if not normalized:
            return False
        negative_markers = (
            "not strong enough",
            "not enough",
            "no clear",
            "not clear",
            "does not clearly",
            "do not clearly",
            "not high enough",
            "not specific",
            "not strongly support",
            "not strong enough to confirm",
            "without additional",
            "ambiguous",
            "unclear",
            "generic",
            "weak",
        )
        if any(marker in normalized for marker in negative_markers):
            return False
        positive_markers = (
            "strong",
            "clear",
            "specific",
            "concrete",
            "strongest",
            "primary",
            "dominant",
            "survive a skeptical",
        )
        return any(marker in normalized for marker in positive_markers)

    textual_keep = _extract_textual_mouth_aus(confirmation_basis, why_not, summary)
    text_supports_keep = (
        _supports_local_mouth_action(confirmation_basis)
        or _supports_local_mouth_action(why_not)
    )
    if not keep and textual_keep and numeric_support in {"strong", "moderate"} and text_supports_keep:
        keep = textual_keep[:1]
        state = "local_mouth_action"
    keep_has_basis = bool(keep) and (
        text_supports_keep
        or (numeric_support == "strong" and confirmation_basis and any(au in textual_keep for au in keep))
    )
    if evidence_status in {"ambiguous", "pose_confounded", "absent"}:
        if keep_has_basis:
            evidence_status = "confirmed"
            state = "local_mouth_action"
        else:
            state = "no_reliable_mouth_au"
    if state == "no_reliable_mouth_au":
        keep = []
        if not drop:
            drop = sorted(_MOUTH_AUS)
    elif not keep:
        state = "no_reliable_mouth_au"
        drop = sorted(_MOUTH_AUS)
    drop = [au for au in drop if au not in keep]
    if state == "no_reliable_mouth_au" and evidence_status == "confirmed":
        evidence_status = "ambiguous"
    if state == "local_mouth_action" and keep and evidence_status in {"ambiguous", "pose_confounded", "absent"}:
        evidence_status = "confirmed"
    if keep and not confirmation_basis and summary:
        confirmation_basis = summary.split(". ", 1)[0].strip().rstrip(".")
    if keep and not why_not and confirmation_basis:
        why_not = (
            "The cited mouth-card evidence is specific enough to survive a no_reliable_mouth_au reading."
        )
    return {
        "local_evidence_status": evidence_status or ("confirmed" if keep else "absent"),
        "numeric_support_assessment": numeric_support or ("moderate" if keep else "none"),
        "pose_risk": pose_risk or "medium",
        "mouth_state": state,
        "mouth_keep_aus": keep,
        "mouth_drop_aus": drop,
        "confirmation_basis": confirmation_basis,
        "why_not_no_reliable_mouth_au": why_not,
        "mouth_summary": summary,
        "confidence": str(value.get("confidence") or "").strip(),
    }


def _sanitize_eye_state(value: Any) -> str:
    text = normalize_answer_text(str(value or "")).replace(" ", "_")
    if text in _EYE_STATES:
        return text
    if any(token in text for token in ("no_reliable", "no_eye", "none", "neutral", "uncertain", "absent")):
        return "no_reliable_eye_au"
    if any(token in text for token in ("closure", "closed", "blink", "shut")):
        return "closure"
    if any(token in text for token in ("tight", "narrow", "squint")):
        return "narrowing"
    if any(token in text for token in ("widen", "wide", "raise", "open")):
        return "widening"
    return ""


def _sanitize_eye_arbiter_output(value: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(value or {})
    evidence_status = _sanitize_brow_evidence_status(
        value.get("local_evidence_status") or value.get("evidence_status")
    )
    numeric_support = _sanitize_brow_numeric_support(
        value.get("numeric_support_assessment") or value.get("support_assessment")
    )
    pose_risk = _sanitize_brow_pose_risk(value.get("pose_risk"))
    keep = [
        au for au in _extract_agent_aus(value.get("eye_keep_aus") or value.get("keep_aus"))
        if au in _EYE_AUS
    ]
    drop = [
        au for au in _extract_agent_aus(value.get("eye_drop_aus") or value.get("drop_aus"))
        if au in _EYE_AUS
    ]
    raw_state = value.get("eye_state") or value.get("state") or ""
    state = _sanitize_eye_state(raw_state)
    normalized_state_label = _normalize_au_token(str(raw_state).replace("_", " "))
    state_to_au = {
        "widening": "upper lid raiser",
        "narrowing": "lid tightener",
        "closure": "eye closure",
    }
    au_to_state = {au: eye_state for eye_state, au in state_to_au.items()}
    if normalized_state_label in _EYE_AUS:
        preferred_au = normalized_state_label
        keep = [preferred_au, *[au for au in keep if au != preferred_au]]
        state = au_to_state[preferred_au]
    elif keep and not state:
        state = au_to_state.get(keep[0], "")
    if state in state_to_au:
        preferred_au = state_to_au[state]
        keep = [preferred_au]
        drop = [au for au in drop if au != preferred_au]
    elif len(keep) > 1:
        preferred_au = "eye closure" if "eye closure" in keep else keep[0]
        keep = [preferred_au]
        drop = [au for au in drop if au != preferred_au]
        state = au_to_state.get(preferred_au, "")
    if evidence_status in {"ambiguous", "pose_confounded", "absent"}:
        state = "no_reliable_eye_au"
    if state == "no_reliable_eye_au":
        keep = []
        if not drop:
            drop = sorted(_EYE_AUS)
    elif not keep:
        state = "no_reliable_eye_au"
        drop = sorted(_EYE_AUS)
    if state == "no_reliable_eye_au" and evidence_status == "confirmed":
        evidence_status = "ambiguous"
    if state in state_to_au and evidence_status in {"ambiguous", "pose_confounded", "absent"}:
        evidence_status = "confirmed"
    summary = str(value.get("eye_summary") or value.get("summary") or "").strip()
    confirmation_basis = str(value.get("confirmation_basis") or "").strip()
    why_not = str(value.get("why_not_no_reliable_eye_au") or "").strip()
    if keep and not confirmation_basis and summary:
        confirmation_basis = summary.split(". ", 1)[0].strip().rstrip(".")
    if keep and not why_not and confirmation_basis:
        why_not = (
            "The cited eyelid evidence is specific enough to survive a no_reliable_eye_au reading."
        )
    return {
        "local_evidence_status": evidence_status or ("confirmed" if keep else "absent"),
        "numeric_support_assessment": numeric_support or ("moderate" if keep else "none"),
        "pose_risk": pose_risk or "medium",
        "eye_state": state or ("no_reliable_eye_au" if not keep else ""),
        "eye_keep_aus": keep[:1],
        "eye_drop_aus": drop,
        "confirmation_basis": confirmation_basis,
        "why_not_no_reliable_eye_au": why_not,
        "eye_summary": summary,
        "confidence": str(value.get("confidence") or "").strip(),
    }


def _sanitize_head_motion_agent_output(value: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(value or {})
    motion_present = normalize_answer_text(str(value.get("motion_present") or "")).replace(" ", "_")
    if motion_present not in _HEAD_MOTION_PRESENCE:
        motion_present = "uncertain"

    primary_direction = normalize_answer_text(str(value.get("primary_direction") or "")).replace(" ", "_")
    primary_direction = primary_direction.replace("-", "_")
    if primary_direction not in _HEAD_MOTION_DIRECTIONS:
        if "up" in primary_direction:
            primary_direction = "upward"
        elif "down" in primary_direction:
            primary_direction = "downward"
        elif any(token in primary_direction for token in ("right", "left", "lateral", "sideways", "horizontal")):
            primary_direction = "none"
        else:
            primary_direction = "uncertain"

    magnitude = normalize_answer_text(str(value.get("magnitude") or "")).replace(" ", "_")
    if magnitude not in _HEAD_MOTION_MAGNITUDES:
        if "small" in magnitude:
            magnitude = "small"
        elif "moderate" in magnitude or "medium" in magnitude:
            magnitude = "moderate"
        elif "large" in magnitude or "strong" in magnitude:
            magnitude = "large"
        else:
            magnitude = "uncertain"

    def _normalize_risk(raw: Any) -> str:
        text = normalize_answer_text(str(raw or "")).replace(" ", "_")
        if text in _HEAD_MOTION_RISKS:
            return text
        if "high" in text or "strong" in text:
            return "high"
        if "med" in text or "moderate" in text:
            return "medium"
        if "low" in text or "small" in text or "weak" in text:
            return "low"
        return "uncertain"

    brow_summary_value = value.get("brow_summary")
    if isinstance(brow_summary_value, str):
        brow_summary_text = brow_summary_value.strip()
    else:
        brow_summary = brow_summary_value if isinstance(brow_summary_value, dict) else {}
        frontalis_summary = str(
            brow_summary.get("frontalis_summary")
            or ""
        ).strip()
        corrugator_summary = str(
            brow_summary.get("corrugator_procerus_summary")
            or ""
        ).strip()
        brow_summary_text = " ".join(
            part for part in (frontalis_summary, corrugator_summary) if part
        ).strip()
        if not brow_summary_text:
            brow_summary_text = str(
                brow_summary.get("summary")
                or brow_summary.get("notes")
                or ""
            ).strip()

    return {
        "motion_present": motion_present,
        "primary_direction": primary_direction,
        "magnitude": magnitude,
        "brow_summary": brow_summary_text,
        "global_drift_risk": _normalize_risk(
            (brow_summary_value or {}).get("global_drift_risk")
            if isinstance(brow_summary_value, dict)
            else value.get("global_drift_risk")
        ),
        "possible_brow_effects": str(value.get("possible_brow_effects") or "").strip(),
        "raw_reasoning": str(value.get("raw_reasoning") or value.get("reasoning") or "").strip(),
        "confidence": str(value.get("confidence") or "").strip(),
    }


def _stringify_region(name: str, info: dict[str, Any] | None) -> str:
    if not info:
        return f"{name}: none"
    parts = []
    for key in ("mag", "angle", "dir", "active"):
        if key in info:
            parts.append(f"{key}={info[key]}")
    return f"{name}: " + ", ".join(parts)


def _stringify_au_evidence(name: str, info: dict[str, Any] | None) -> str:
    if not info:
        return f"{name}: none"
    return (
        f"{name}: score={info.get('score', 'na')}, "
        f"norm={info.get('normalized_support', 'na')}, "
        f"raw={info.get('raw_support', 'na')}, "
        f"residual={info.get('residual_support', 'na')}, "
        f"geometry={info.get('geometry_support', 'na')}, "
        f"present={info.get('present', False)}, "
        f"strength={info.get('strength', 'unknown')}, "
        f"dir={info.get('direction_score', 'na')}, "
        f"balance={info.get('balance_score', 'na')}"
    )


def _stringify_au_reliability(name: str, info: dict[str, Any] | None) -> str:
    if not info:
        return f"{name}: none"
    precision = float(info.get("precision", 0.0) or 0.0)
    f1 = float(info.get("f1", 0.0) or 0.0)
    calibrated = bool(info.get("calibrated"))
    if not calibrated:
        reliability = "uncalibrated"
    elif precision >= 0.45 and f1 >= 0.35:
        reliability = "reliable"
    elif precision >= 0.25 or f1 >= 0.25:
        reliability = "mixed"
    else:
        reliability = "risky"
    return (
        f"{name}: score={info.get('score', 'na')}, "
        f"norm={info.get('normalized_support', 'na')}, "
        f"precision={round(precision, 4)}, "
        f"f1={round(f1, 4)}, "
        f"reliability={reliability}, "
        f"agent_only={bool(info.get('agent_only'))}"
    )


def _build_contrastive_hallmark_text(
    allowed_emotions: list[str],
    emotion_au_rates: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    if not allowed_emotions or not emotion_au_rates or not evidence:
        return "none"

    observed: list[tuple[str, float, bool]] = []
    for name, info in evidence.items():
        key = str(name or "").strip().lower()
        if not key:
            continue
        score = float((info or {}).get("score", 0.0) or 0.0)
        combined = float((info or {}).get("combined_support", 0.0) or 0.0)
        present = bool((info or {}).get("present"))
        support = max(score, combined)
        if present or support >= 0.35:
            observed.append((key, support, present))
    if not observed:
        return "none"

    epsilon = 1e-4
    lines: list[str] = []
    observed_names = {name for name, _support, _present in observed}
    for emotion in allowed_emotions:
        rates = emotion_au_rates.get(emotion) or {}
        other_emotions = [name for name in allowed_emotions if name != emotion]
        hits: list[tuple[float, str]] = []
        conflicts: list[tuple[float, str]] = []

        for name, support, present in observed:
            rate = float(rates.get(name, 0.0) or 0.0)
            other_rate_pairs = [
                (other, float((emotion_au_rates.get(other) or {}).get(name, 0.0) or 0.0))
                for other in other_emotions
            ]
            other_mean = (
                sum(value for _other, value in other_rate_pairs) / max(1, len(other_rate_pairs))
            )
            best_other, best_other_rate = max(
                other_rate_pairs,
                key=lambda item: item[1],
                default=("", 0.0),
            )
            contrast = math.log((rate + epsilon) / (other_mean + epsilon))
            weighted_contrast = contrast * math.log1p(max(support, 0.0))
            if rate >= 0.05 and contrast >= 0.35:
                hits.append(
                    (
                        weighted_contrast,
                        f"{name}(contrast={round(contrast, 2)},support={round(support, 3)},present={present})",
                    )
                )
            elif best_other and best_other_rate >= 0.1 and contrast <= -0.35:
                conflicts.append(
                    (
                        -weighted_contrast,
                        f"{name}->{best_other}(contrast={round(-contrast, 2)},support={round(support, 3)})",
                    )
                )

        misses: list[tuple[float, str]] = []
        for name, raw_rate in rates.items():
            rate = float(raw_rate or 0.0)
            if name in observed_names or rate < 0.2:
                continue
            other_rates = [
                float((emotion_au_rates.get(other) or {}).get(name, 0.0) or 0.0)
                for other in other_emotions
            ]
            other_mean = sum(other_rates) / max(1, len(other_rates))
            contrast = math.log((rate + epsilon) / (other_mean + epsilon))
            if contrast >= 0.35:
                misses.append(
                    (
                        contrast * rate,
                        f"{name}(rate={round(rate, 3)},contrast={round(contrast, 2)})",
                    )
                )

        hits_text = ", ".join(text for _score, text in sorted(hits, reverse=True)[:3]) or "none"
        misses_text = ", ".join(text for _score, text in sorted(misses, reverse=True)[:2]) or "none"
        conflicts_text = ", ".join(text for _score, text in sorted(conflicts, reverse=True)[:2]) or "none"
        lines.append(f"- {emotion}: hits={hits_text}; misses={misses_text}; conflicts={conflicts_text}")
    return "\n".join(lines) if lines else "none"


def _stringify_brow_signal(name: str, info: dict[str, Any] | None) -> str:
    if not info:
        return f"{name}: none"
    return (
        f"{name}: score={info.get('score', 'na')}, "
        f"norm={info.get('normalized_support', 'na')}, "
        f"raw={info.get('raw_support', 'na')}, "
        f"residual={info.get('residual_support', 'na')}, "
        f"combined={info.get('combined_support', 'na')}, "
        f"geom={info.get('geometry_support', 'na')}, "
        f"present={info.get('present', False)}, "
        f"strength={info.get('strength', 'unknown')}, "
        f"agent_only={bool(info.get('agent_only'))}, "
        f"dir={info.get('direction_score', 'na')}, "
        f"bal={info.get('balance_score', 'na')}, "
        f"threshold={info.get('threshold', 'na')}, "
        f"precision={info.get('precision', 'na')}, "
        f"f1={info.get('f1', 'na')}"
    )


def _stringify_mouth_signal(name: str, info: dict[str, Any] | None) -> str:
    if not info:
        return f"{name}: none"
    return (
        f"{name}: score={info.get('score', 'na')}, "
        f"norm={info.get('normalized_support', 'na')}, "
        f"raw={info.get('raw_support', 'na')}, "
        f"residual={info.get('residual_support', 'na')}, "
        f"combined={info.get('combined_support', 'na')}, "
        f"geom={info.get('geometry_support', 'na')}, "
        f"present={info.get('present', False)}, "
        f"strength={info.get('strength', 'unknown')}, "
        f"dir={info.get('direction_score', 'na')}, "
        f"bal={info.get('balance_score', 'na')}, "
        f"threshold={info.get('threshold', 'na')}, "
        f"precision={info.get('precision', 'na')}, "
        f"f1={info.get('f1', 'na')}"
    )


def _stringify_metric_triplets(pairs: list[tuple[str, Any]]) -> str:
    return ", ".join(
        f"{label}={value}"
        for label, value in pairs
        if value is not None
    ) or "none"


def _clip_numeric_prompt_parts(
    clip_cache: dict[str, Any],
    dataset: str,
    *,
    ablation: AblationConfig | None = None,
) -> dict[str, Any]:
    ablation = ablation or AblationConfig()
    numeric = ensure_au_evidence(clip_cache.get("numeric_features", {}) or {}, dataset)
    prior = {} if ablation.disable_prior else derive_knowledge_prior(dataset, numeric)
    allowed = sorted(DATASET_ALLOWED_EMOTIONS.get(dataset, []))
    raw = numeric.get("landmark_au_regions") or {}
    residual = numeric.get("landmark_residual_au_regions") or {}
    head_motion = numeric.get("head_motion") or {}
    eye_ratio = numeric.get("eye_ratio")
    eye_metrics = numeric.get("eye_metrics") or {}
    eyelid_signal = numeric.get("eyelid_signal", "unknown")
    mouth_metrics = numeric.get("mouth_metrics") or {}
    landmark_aus = ", ".join(numeric.get("landmark_aus") or []) or "none"

    key_regions = [
        "AU4_brow_lowerer",
        "AU1_inner_brow_subR",
        "AU1_inner_brow_subL",
        "AU2_brow_outer_subR",
        "AU2_brow_outer_subL",
        "AU9_nose_bridge",
        "AU10_upper_lip",
        "AU12_corner_subR",
        "AU12_corner_subL",
        "AU23_lip_center",
        "AU17_chin",
    ]
    raw_text = "\n".join(_stringify_region(name, raw.get(name)) for name in key_regions)
    residual_text = "\n".join(_stringify_region(name, residual.get(name)) for name in key_regions)
    prior_scores = ", ".join(
        f"{name}={score}" for name, score in sorted((prior.get("scores") or {}).items(), key=lambda item: item[1], reverse=True)
    ) or "none"
    evidence = numeric.get("au_evidence") or {}
    present_gate_enabled = not ablation.disable_present_gate

    def rank_key(info: dict[str, Any] | None) -> tuple[float, float, float]:
        info = info or {}
        if present_gate_enabled:
            return (
                1.0 if bool(info.get("present")) else 0.0,
                float(info.get("score", 0.0) or 0.0),
                float(info.get("normalized_support", 0.0) or 0.0),
            )
        continuous = max(
            float(info.get("score", 0.0) or 0.0),
            float(info.get("normalized_support", 0.0) or 0.0),
            float(info.get("combined_support", 0.0) or 0.0),
            float(info.get("geometry_support", 0.0) or 0.0),
            float(info.get("raw_support", 0.0) or 0.0),
            float(info.get("residual_support", 0.0) or 0.0),
        )
        return (
            continuous,
            float(info.get("score", 0.0) or 0.0),
            float(info.get("normalized_support", 0.0) or 0.0),
        )

    def support_gate(info: dict[str, Any] | None, *, threshold: float) -> bool:
        info = info or {}
        if present_gate_enabled and bool(info.get("present")):
            return True
        return max(
            float(info.get("score", 0.0) or 0.0),
            float(info.get("normalized_support", 0.0) or 0.0),
            float(info.get("combined_support", 0.0) or 0.0),
            float(info.get("geometry_support", 0.0) or 0.0),
            float(info.get("raw_support", 0.0) or 0.0),
            float(info.get("residual_support", 0.0) or 0.0),
        ) >= threshold

    evidence_rows = sorted(
        evidence.items(),
        key=lambda item: rank_key(item[1]),
        reverse=True,
    )
    evidence_text = "\n".join(
        _stringify_au_evidence(name, info)
        for name, info in evidence_rows[:10]
    ) or "none"
    reliability_text = "\n".join(
        _stringify_au_reliability(name, info)
        for name, info in evidence_rows[:8]
        if support_gate(info, threshold=0.35)
    ) or "none"
    eye_metrics_text = ", ".join(
        f"{key}={value}"
        for key, value in [
            ("left_ratio", eye_metrics.get("left_ratio")),
            ("right_ratio", eye_metrics.get("right_ratio")),
            ("mean_ratio", eye_metrics.get("mean_ratio")),
            ("min_ratio", eye_metrics.get("min_ratio")),
            ("gap_ratio", eye_metrics.get("mean_gap_ratio")),
            ("gap_shrink", eye_metrics.get("mean_gap_shrink")),
            ("inner_rel_raise", eye_metrics.get("inner_brow_relative_raise_mean")),
            ("inner_rel_drop", eye_metrics.get("inner_brow_relative_drop_mean")),
            ("outer_rel_raise", eye_metrics.get("outer_brow_relative_raise_mean")),
            ("outer_rel_drop", eye_metrics.get("outer_brow_relative_drop_mean")),
            ("inner_fwd_rel", eye_metrics.get("inner_brow_forward_relative_mean")),
            ("outer_fwd_rel", eye_metrics.get("outer_brow_forward_relative_mean")),
            ("brow_drop", eye_metrics.get("brow_center_drop_norm")),
            ("apex_min_norm", eye_metrics.get("apex_min_height_norm")),
            ("asym", eye_metrics.get("asymmetry")),
        ]
        if value is not None
    ) or "none"
    mouth_metrics_text = ", ".join(
        f"{key}={value}"
        for key, value in [
            ("open_ratio", mouth_metrics.get("mouth_open_ratio")),
            ("width_ratio", mouth_metrics.get("mouth_width_ratio")),
            ("upper_lip_nose_ratio", mouth_metrics.get("upper_lip_nose_ratio")),
            ("lower_lip_nose_ratio", mouth_metrics.get("lower_lip_nose_ratio")),
            ("nostril_raise", mouth_metrics.get("nostril_raise_mean")),
            ("nose_bridge_shrink", mouth_metrics.get("nose_bridge_shrink_norm")),
            ("upper_lip_raise", mouth_metrics.get("upper_lip_raise_norm")),
            ("upper_lip_raise_rel", mouth_metrics.get("upper_lip_raise_relative_norm")),
            ("upper_lip_fwd", mouth_metrics.get("upper_lip_forward_norm")),
            ("left_corner_lift", mouth_metrics.get("left_corner_lift")),
            ("right_corner_lift", mouth_metrics.get("right_corner_lift")),
            ("corner_lift_mean", mouth_metrics.get("corner_lift_mean")),
            ("corner_raise_rel", mouth_metrics.get("corner_raise_relative_mean")),
            ("corner_fwd", mouth_metrics.get("corner_forward_mean")),
            ("mouth_center_fwd", mouth_metrics.get("mouth_center_forward_norm")),
            ("mouth_center_raise", mouth_metrics.get("mouth_center_raise_norm")),
            ("lower_lip_raise", mouth_metrics.get("lower_lip_raise_norm")),
            ("chin_raise", mouth_metrics.get("chin_raise_norm")),
            ("left_corner_depress", mouth_metrics.get("left_corner_depress")),
            ("right_corner_depress", mouth_metrics.get("right_corner_depress")),
            ("corner_depress_mean", mouth_metrics.get("corner_depress_mean")),
            ("lower_lip_drop_norm", mouth_metrics.get("lower_lip_drop_norm")),
        ]
        if value is not None
    ) or "none"
    extra_geometry_lines = ""
    if eye_metrics:
        extra_geometry_lines += f"Eye geometry metrics: {eye_metrics_text}\n"
    if mouth_metrics:
        extra_geometry_lines += f"Mouth geometry metrics: {mouth_metrics_text}\n"
    suggested_emotion = prior.get("suggested_emotion") or "none"
    suggested_aus = ", ".join(prior.get("suggested_aus") or []) or "none"
    prior_notes = ", ".join(prior.get("notes") or []) or "none"
    pattern_candidate_aus = ", ".join(prior.get("pattern_candidate_aus") or []) or "none"
    prior_observations_list = list(prior.get("dataset_observations") or [])
    prior_observations = (
        "\n".join(f"  - {item}" for item in prior_observations_list)
        if prior_observations_list
        else "  - none"
    )
    calibration = load_motion_feature_calibration()
    dataset_calibration = (calibration.get("datasets") or {}).get(dataset, {}) if calibration.get("available") else {}
    threshold_count = len(dataset_calibration.get("au_thresholds") or {})
    emotion_au_rates = dataset_calibration.get("emotion_au_rates") or {}
    emotion_alignment_text = (
        _build_contrastive_hallmark_text(
            allowed,
            emotion_au_rates,
            evidence,
        )
        if not ablation.disable_prior
        else "none"
    )

    brow_focus_regions = [
        "AU4_brow_lowerer",
        "AU1_inner_brow_subR",
        "AU1_inner_brow_subL",
        "AU2_brow_outer_subR",
        "AU2_brow_outer_subL",
    ]
    brow_raw_text = "\n".join(_stringify_region(name, raw.get(name)) for name in brow_focus_regions) or "none"
    brow_residual_text = "\n".join(_stringify_region(name, residual.get(name)) for name in brow_focus_regions) or "none"
    brow_focus_aus = [
        "inner brow raiser",
        "outer brow raiser",
        "brow lowerer",
        "upper lid raiser",
        "lid tightener",
        "eye closure",
    ]
    brow_signal_text = "\n".join(
        _stringify_brow_signal(name, evidence.get(name))
        for name in ["inner brow raiser", "outer brow raiser", "brow lowerer"]
    ) or "none"
    brow_reliability_text = "\n".join(
        _stringify_au_reliability(name, evidence.get(name))
        for name in brow_focus_aus
    ) or "none"
    brow_prior_lines = []
    for au_name in ["inner brow raiser", "outer brow raiser", "brow lowerer", "upper lid raiser", "lid tightener"]:
        ranked = sorted(
            (
                (emotion, float((rates or {}).get(au_name, 0.0) or 0.0))
                for emotion, rates in emotion_au_rates.items()
                if float((rates or {}).get(au_name, 0.0) or 0.0) > 0.0
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        summary = ", ".join(f"{emotion}={round(rate, 4)}" for emotion, rate in ranked[:3]) or "none"
        brow_prior_lines.append(f"{au_name}: {summary}")
    brow_prior_text = (
        "Brow prior card (weak context only, never clip-level proof):\n"
        f"- Clip-level suggested emotion: {suggested_emotion}\n"
        f"- Clip-level suggested AUs: {suggested_aus}\n"
        f"- Clip-level prior notes: {prior_notes}\n"
        f"- Data-learned observations:\n{prior_observations}\n"
        f"- Contrastive hallmark card:\n{emotion_alignment_text}\n"
        "- Dataset brow-related AU rates by emotion:\n"
        + "\n".join(f"  - {line}" for line in brow_prior_lines)
    )
    brow_metrics_text = _stringify_metric_triplets(
        [
            ("mean_gap_ratio", eye_metrics.get("mean_gap_ratio")),
            ("mean_gap_shrink", eye_metrics.get("mean_gap_shrink")),
            ("inner_raise", eye_metrics.get("inner_brow_raise_mean")),
            ("inner_rel_raise", eye_metrics.get("inner_brow_relative_raise_mean")),
            ("inner_rel_drop", eye_metrics.get("inner_brow_relative_drop_mean")),
            ("inner_fwd_rel", eye_metrics.get("inner_brow_forward_relative_mean")),
            ("outer_raise", eye_metrics.get("outer_brow_raise_mean")),
            ("outer_rel_raise", eye_metrics.get("outer_brow_relative_raise_mean")),
            ("outer_rel_drop", eye_metrics.get("outer_brow_relative_drop_mean")),
            ("outer_fwd_rel", eye_metrics.get("outer_brow_forward_relative_mean")),
            ("brow_center_drop", eye_metrics.get("brow_center_drop_norm")),
            ("brow_center_raise", eye_metrics.get("brow_center_raise_norm")),
            ("eye_mean_ratio", eye_metrics.get("mean_ratio")),
            ("eye_min_ratio", eye_metrics.get("min_ratio")),
            ("pose_rmse_3d", eye_metrics.get("pose_landmark_fit_rmse_3d")),
        ]
    )
    transform = numeric.get("head_motion_transform") or {}
    brow_pose_text = _stringify_metric_triplets(
        [
            ("method", transform.get("method")),
            ("rot2d_deg", transform.get("rotation_deg")),
            ("rot3d_deg", transform.get("rotation3d_deg")),
            ("scale3d", transform.get("scale3d")),
            ("rmse2d", transform.get("landmark_fit_rmse")),
            ("rmse3d", transform.get("landmark_fit_rmse_3d")),
        ]
    )
    brow_card_text = (
        "Brow evidence card:\n"
        f"- Head motion quality: {head_motion.get('quality', 'na')} (mag={head_motion.get('head_mag', 'na')}, angle={head_motion.get('angle', 'na')})\n"
        f"- Head-motion transform summary: {brow_pose_text}\n"
        f"- Brow geometry metrics: {brow_metrics_text}\n"
        "- SAMM brow AUs are evidence-only here: treat score/norm/present/strength as context, not as automatic final decisions.\n"
        "- In 3D head pitch/yaw, brow regions farther from the rotation center can show larger apparent image-plane flow even without a true AU.\n"
        f"- Brow raw/residual/geometry signals:\n{brow_signal_text}\n"
        f"- Historical brow detector reliability (secondary context only):\n{brow_reliability_text}\n"
        f"- Brow residual regions:\n{brow_residual_text}\n"
        f"- Brow raw-flow regions:\n{brow_raw_text}\n"
    )
    eye_focus_aus = [
        "upper lid raiser",
        "lid tightener",
        "eye closure",
    ]
    eye_signal_text = "\n".join(
        _stringify_brow_signal(name, evidence.get(name))
        for name in eye_focus_aus
    ) or "none"
    eye_reliability_text = "\n".join(
        _stringify_au_reliability(name, evidence.get(name))
        for name in eye_focus_aus
    ) or "none"
    eye_prior_lines = []
    for au_name in eye_focus_aus:
        ranked = sorted(
            (
                (emotion, float((rates or {}).get(au_name, 0.0) or 0.0))
                for emotion, rates in emotion_au_rates.items()
                if float((rates or {}).get(au_name, 0.0) or 0.0) > 0.0
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        summary = ", ".join(f"{emotion}={round(rate, 4)}" for emotion, rate in ranked[:3]) or "none"
        eye_prior_lines.append(f"{au_name}: {summary}")
    eye_metrics_focus_text = _stringify_metric_triplets(
        [
            ("left_ratio", eye_metrics.get("left_ratio")),
            ("right_ratio", eye_metrics.get("right_ratio")),
            ("mean_ratio", eye_metrics.get("mean_ratio")),
            ("min_ratio", eye_metrics.get("min_ratio")),
            ("gap_ratio", eye_metrics.get("mean_gap_ratio")),
            ("gap_shrink", eye_metrics.get("mean_gap_shrink")),
            ("apex_min_norm", eye_metrics.get("apex_min_height_norm")),
            ("asym", eye_metrics.get("asymmetry")),
            ("pose_rmse_3d", eye_metrics.get("pose_landmark_fit_rmse_3d")),
        ]
    )
    eye_prior_text = (
        "Eye prior card (weak context only, never clip-level proof):\n"
        f"- Clip-level suggested emotion: {suggested_emotion}\n"
        f"- Clip-level suggested AUs: {suggested_aus}\n"
        f"- Clip-level prior notes: {prior_notes}\n"
        f"- Data-learned observations:\n{prior_observations}\n"
        f"- Contrastive hallmark card:\n{emotion_alignment_text}\n"
        "- Dataset eyelid-related AU rates by emotion:\n"
        + "\n".join(f"  - {line}" for line in eye_prior_lines)
    )
    eye_card_text = (
        "Eye evidence card:\n"
        f"- Head motion quality: {head_motion.get('quality', 'na')} (mag={head_motion.get('head_mag', 'na')}, angle={head_motion.get('angle', 'na')})\n"
        f"- Eyelid categorical cue: {eyelid_signal}\n"
        f"- Eye geometry metrics: {eye_metrics_focus_text}\n"
        "- Eye specialist AUs are evidence-only here: treat score/norm/present/strength as context, not as automatic final decisions.\n"
        "- AU5 upper lid raiser needs widened aperture or upper-lid lifting that is not explained by brow-lid gap alone.\n"
        "- AU7 lid tightener needs narrowing / tightening rather than near-closure.\n"
        "- AU46 eye closure needs near-closure or blink-like closure; when closure dominates, do not keep AU7 as a second automatic eyelid AU.\n"
        f"- Eye raw/residual/geometry signals:\n{eye_signal_text}\n"
        f"- Historical eye detector reliability (secondary context only):\n{eye_reliability_text}\n"
    )
    mouth_focus_regions = [
        "AU9_nose_bridge",
        "AU10_upper_lip",
        "AU12_corner_subR",
        "AU12_corner_subL",
        "AU20_corner_subR",
        "AU20_corner_subL",
        "AU23_lip_center",
        "AU17_chin",
        "AU26_jaw",
    ]
    mouth_raw_text = "\n".join(_stringify_region(name, raw.get(name)) for name in mouth_focus_regions) or "none"
    mouth_residual_text = "\n".join(
        _stringify_region(name, residual.get(name)) for name in mouth_focus_regions
    ) or "none"
    mouth_focus_aus = [
        "lip corner puller",
        "dimpler",
        "lip corner depressor",
        "lip stretcher",
        "chin raiser",
        "lip tightener",
        "lips part",
        "jaw drop",
        "upper lip raiser",
        "nose wrinkler",
    ]
    mouth_signal_text = "\n".join(
        _stringify_mouth_signal(name, evidence.get(name))
        for name in [
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
        ]
    ) or "none"
    mouth_reliability_text = "\n".join(
        _stringify_au_reliability(name, evidence.get(name))
        for name in mouth_focus_aus
    ) or "none"
    mouth_prior_lines = []
    for au_name in [
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
    ]:
        ranked = sorted(
            (
                (emotion, float((rates or {}).get(au_name, 0.0) or 0.0))
                for emotion, rates in emotion_au_rates.items()
                if float((rates or {}).get(au_name, 0.0) or 0.0) > 0.0
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        summary = ", ".join(f"{emotion}={round(rate, 4)}" for emotion, rate in ranked[:3]) or "none"
        mouth_prior_lines.append(f"{au_name}: {summary}")
    mouth_prior_text = (
        "Mouth prior card (weak context only, never proof):\n"
        f"- Clip-level suggested emotion: {suggested_emotion}\n"
        f"- Clip-level suggested AUs: {suggested_aus}\n"
        f"- Clip-level prior notes: {prior_notes}\n"
        f"- Data-learned observations:\n{prior_observations}\n"
        f"- Contrastive hallmark card:\n{emotion_alignment_text}\n"
        "- Historical lower-face AU rates by emotion:\n"
        + "\n".join(f"  - {line}" for line in mouth_prior_lines)
    )
    mouth_card_text = (
        "Mouth evidence card:\n"
        f"- Head motion quality: {head_motion.get('quality', 'na')} (mag={head_motion.get('head_mag', 'na')}, angle={head_motion.get('angle', 'na')})\n"
        f"- Mouth geometry metrics: {mouth_metrics_text}\n"
        "- Lower-face specialist AUs are evidence-only here: treat score/norm/present/strength as context, not as automatic final decisions.\n"
        "- AU12 needs real lip-corner raise/lift/back/lateral pull. Do not infer it from generic mouth opening, center raise, or chin raise alone.\n"
        "- AU14 dimpler is a pinched or compressed corner pattern, often with reduced corner drop and sometimes slight lift. Do not let AU12 absorb that pattern.\n"
        "- AU9 nose wrinkler needs direct nose/nostril evidence. AU10 upper lip raiser needs upper-lip lift or compression. Do not let AU12 absorb those patterns either.\n"
        "- AU20 lip stretcher looks tense, retracted, horizontal, or downward rather than smile-like. AU17 is lower-lip/chin push-up. AU23 is lip tightening or thinning rather than generic mouth closure.\n"
        "- AU25 lips part and AU26 jaw drop may co-occur, but jaw drop requires clear jaw-lowering evidence rather than lip separation alone.\n"
        f"- Mouth raw/residual/geometry signals:\n{mouth_signal_text}\n"
        f"- Historical mouth detector reliability (secondary context only):\n{mouth_reliability_text}\n"
        f"- Mouth residual regions:\n{mouth_residual_text}\n"
        f"- Mouth raw-flow regions:\n{mouth_raw_text}\n"
    )

    return {
        "prior_enabled": not ablation.disable_prior,
        "numeric": numeric,
        "allowed_text": ", ".join(allowed),
        "head_motion": head_motion,
        "eye_ratio": eye_ratio,
        "eyelid_signal": eyelid_signal,
        "extra_geometry_lines": extra_geometry_lines,
        "landmark_aus": landmark_aus,
        "evidence_text": evidence_text,
        "reliability_text": reliability_text,
        "suggested_emotion": suggested_emotion,
        "suggested_aus": suggested_aus,
        "pattern_candidate_aus": pattern_candidate_aus,
        "prior_scores": prior_scores,
        "prior_notes": prior_notes,
        "prior_observations": prior_observations,
        "emotion_alignment_text": emotion_alignment_text,
        "residual_text": residual_text,
        "raw_text": raw_text,
        "calibration_threshold_count": threshold_count,
        "brow_card_text": brow_card_text,
        "brow_prior_text": brow_prior_text,
        "eye_card_text": eye_card_text,
        "eye_prior_text": eye_prior_text,
        "mouth_card_text": mouth_card_text,
        "mouth_prior_text": mouth_prior_text,
    }


def _render_clip_numeric_summary(parts: dict[str, Any]) -> str:
    head_motion = parts["head_motion"]
    prior_section = ""
    if parts.get("prior_enabled"):
        prior_section = (
            f"Knowledge prior suggested emotion: {parts['suggested_emotion']}\n"
            f"Knowledge prior suggested AUs: {parts['suggested_aus']}\n"
            f"Knowledge prior pattern candidate AUs: {parts['pattern_candidate_aus']}\n"
            f"Knowledge prior notes: {parts['prior_notes']}\n"
            f"Knowledge prior data-learned observations:\n{parts['prior_observations']}\n"
            f"Contrastive hallmark card:\n{parts['emotion_alignment_text']}\n"
        )

    return (
        f"Allowed emotions: {parts['allowed_text']}\n"
        f"Head motion: mag={head_motion.get('head_mag', 'na')}, angle={head_motion.get('angle', 'na')}, "
        f"dir={head_motion.get('dir', 'na')}, quality={head_motion.get('quality', 'na')}\n"
        f"Eye ratio: {parts['eye_ratio']}\n"
        f"Eyelid signal: {parts['eyelid_signal']}\n"
        f"{parts['extra_geometry_lines']}"
        "Important: reason primarily from the continuous numeric evidence below "
        "(mag, angle, active, residual motion, eye ratio, and head-motion quality), "
        "not from any hard AU activation.\n"
        "Treat any AU names below only as weak candidate hints for attention, not as facts.\n"
        f"Weak landmark AU hints: {parts['landmark_aus']}\n"
        f"Calibrated AU evidence:\n{parts['evidence_text']}\n"
        f"{prior_section}"
        f"Residual landmark regions:\n{parts['residual_text']}\n"
        f"Raw flow regions:\n{parts['raw_text']}\n"
    )


def _clip_numeric_summary(
    clip_cache: dict[str, Any],
    dataset: str,
    *,
    ablation: AblationConfig | None = None,
) -> str:
    parts = _clip_numeric_prompt_parts(clip_cache, dataset, ablation=ablation)
    return _render_clip_numeric_summary(parts)


def _clip_numeric_views(
    clip_cache: dict[str, Any],
    dataset: str,
    *,
    ablation: AblationConfig | None = None,
) -> dict[str, str]:
    parts = _clip_numeric_prompt_parts(clip_cache, dataset, ablation=ablation)
    legacy = _render_clip_numeric_summary(parts)
    if dataset != "samm":
        return {
            "head_motion_agent": legacy,
            "brow_agent": legacy,
            "eye_agent": legacy,
            "mouth_agent": legacy,
            "motion_agent": legacy,
            "emotion_agent": legacy,
            "critic_agent": legacy,
            "legacy": legacy,
        }
    head_motion = parts["head_motion"]
    shared_header = (
        f"Allowed emotions: {parts['allowed_text']}\n"
        f"Head motion: mag={head_motion.get('head_mag', 'na')}, angle={head_motion.get('angle', 'na')}, "
        f"dir={head_motion.get('dir', 'na')}, quality={head_motion.get('quality', 'na')}\n"
        f"Eye ratio: {parts['eye_ratio']}\n"
        f"Eyelid signal: {parts['eyelid_signal']}\n"
        f"{parts['extra_geometry_lines']}"
        "Anatomical consistency notes: AU7 lid tightener and AU46 eye closure should not both survive as final AUs unless there is a very strong explicit reason; "
        "a larger brow-lid gap caused by eyelid motion is not positive evidence for inner brow raiser.\n"
    )
    motion_view = (
        f"{shared_header}"
        "Evidence view: continuous motion only.\n"
        "Use head-motion quality, residual motion, raw flow regions, and continuous geometry values.\n"
        "Do not assume any AU is present just because a prior or a landmark hint would suggest it.\n"
        f"Residual landmark regions:\n{parts['residual_text']}\n"
        f"Raw flow regions:\n{parts['raw_text']}\n"
    )
    head_motion_view = (
        f"{shared_header}"
        "Evidence view: head-motion specialist.\n"
        "Focus on the direction, magnitude, and likely image-plane consequences of head motion for the brow region.\n"
        "Do not decide final brow AUs; only explain what the pose motion could plausibly make look exaggerated, underestimated, or only minimally affected.\n"
        f"{parts['brow_card_text']}"
    )
    if parts.get("prior_enabled"):
        emotion_view = (
            f"{shared_header}"
            "Evidence view: geometry and motion synthesis.\n"
            "Treat Agent A as the motion specialist and use eye/mouth geometry as an independent second view.\n"
            "Do not use weak landmark AU hints or top-prior label names as deciding evidence.\n"
            "Use the contrastive hallmark card below instead of raw prior scores.\n"
            f"Knowledge prior notes: {parts['prior_notes']}\n"
            f"Knowledge prior data-learned observations:\n{parts['prior_observations']}\n"
            f"Contrastive hallmark card:\n{parts['emotion_alignment_text']}\n"
        )
        critic_view = (
            f"{shared_header}"
            "Evidence view: reliability audit.\n"
            "Agent A and the calibrated motion scores are not independent if they come from the same motion stream.\n"
            "Downweight any AU whose calibration reliability is risky unless both motion and geometry support it.\n"
            f"Calibrated AU reliability:\n{parts['reliability_text']}\n"
            "Use the contrastive hallmark card below instead of raw prior scores.\n"
            f"Knowledge prior notes: {parts['prior_notes']}\n"
            f"Knowledge prior data-learned observations:\n{parts['prior_observations']}\n"
            f"Contrastive hallmark card:\n{parts['emotion_alignment_text']}\n"
            f"Calibration thresholds loaded: {parts['calibration_threshold_count']}\n"
        )
    else:
        emotion_view = (
            f"{shared_header}"
            "Evidence view: geometry and motion synthesis.\n"
            "Treat Agent A as the motion specialist and use eye/mouth geometry as an independent second view.\n"
            "No external prior or contrastive hallmark card is available in this ablation.\n"
        )
        critic_view = (
            f"{shared_header}"
            "Evidence view: reliability audit.\n"
            "Agent A and the calibrated motion scores are not independent if they come from the same motion stream.\n"
            "Downweight any AU whose calibration reliability is risky unless both motion and geometry support it.\n"
            f"Calibrated AU reliability:\n{parts['reliability_text']}\n"
            f"Calibration thresholds loaded: {parts['calibration_threshold_count']}\n"
        )
    brow_prior_suffix = f"{parts['brow_prior_text']}\n" if parts.get("prior_enabled") else ""
    brow_view = (
        f"{shared_header}"
        "Evidence view: brow specialist.\n"
        "Decide only whether the brow evidence is more consistent with inner brow raiser, outer brow raiser, brow lowerer, or none.\n"
        "Use the brow evidence card below as the primary evidence source.\n"
        "Use the brow prior card only as weak tie-breaker context after local evidence already looks plausible.\n"
        "Do not treat generic upward raw flow or a surprise-like impression as sufficient.\n"
        "Do not let eyelid motion alone create brow-raise evidence.\n"
        f"{parts['brow_card_text']}"
        f"{brow_prior_suffix}"
    )
    eye_prior_suffix = f"{parts['eye_prior_text']}\n" if parts.get("prior_enabled") else ""
    eye_view = (
        f"{shared_header}"
        "Evidence view: eye specialist.\n"
        "Decide only whether the eyelid evidence is more consistent with upper lid raiser, lid tightener, eye closure, or no reliable local eye AU.\n"
        "Use the eye evidence card below as the primary evidence source.\n"
        "Use the eye prior card only as weak tie-breaker context after local evidence already looks plausible.\n"
        "Do not keep both lid tightener and eye closure; choose the dominant local eyelid state or no reliable eye AU.\n"
        "Do not infer upper lid raiser from brow-lid gap growth alone.\n"
        f"{parts['eye_card_text']}"
        f"{eye_prior_suffix}"
    )
    mouth_view = (
        f"{shared_header}"
        "Evidence view: mouth specialist.\n"
        "Decide only whether the lower-face evidence is more consistent with the mouth/jaw AUs in the mouth evidence card, or with no reliable local mouth AU.\n"
        "Use the mouth evidence card below as the primary evidence source.\n"
        "Ignore historical priors unless a local tie remains after reading the evidence card.\n"
        "Do not treat generic smile-like appearance, mouth opening, or lower-face drift as sufficient by themselves.\n"
        f"{parts['mouth_card_text']}"
    )
    return {
        "head_motion_agent": head_motion_view,
        "brow_agent": brow_view,
        "eye_agent": eye_view,
        "mouth_agent": mouth_view,
        "motion_agent": motion_view,
        "emotion_agent": emotion_view,
        "critic_agent": critic_view,
        "legacy": legacy,
    }


def _dataset_reasoning_guidance(dataset: str) -> str:
    if dataset == "samm":
        return (
            "SAMM-specific reasoning guidance:\n"
            "- Do not default to disgust unless the evidence clearly supports disgust-like cues such as nose wrinkler or upper lip raiser.\n"
            "- If brow lowerer or lid tightener is stronger than disgust mouth evidence, actively compare anger against disgust.\n"
            "- If lip corner puller or cheek raiser is stronger than disgust cues, actively compare happiness against disgust.\n"
            "- If inner brow raiser, outer brow raiser, or upper lid raiser dominate, actively compare surprise against disgust.\n"
            "- Treat contempt and sadness as valid options when the prior or the visible AUs support them; do not collapse them into disgust by default.\n"
            "- When the top prior and the visible pattern disagree, explain which cues break the tie instead of copying the prior.\n"
            "- Weak single-region mouth motion is not enough by itself to assert lip tightener, upper lip raiser, nose wrinkler, or chin raiser.\n"
            "- If only one weak numeric mouth cue is present, prefer uncertainty or a smaller AU set over forcing an AU label.\n"
        )
    if dataset == "casme2":
        return (
            "CASME2-specific reasoning guidance:\n"
            "- Repression is a valid label and should be compared carefully against disgust, happiness, and surprise.\n"
            "- Do not replace repression with disgust unless the observed AU pattern clearly supports disgust better.\n"
            "- When surprise is considered, check whether brow-raising evidence is truly stronger than repression or disgust cues.\n"
            "- Resolve eyelid-state consistency explicitly: AU7 lid tightener and AU46 eye closure should not be treated as automatic duplicate confirmation.\n"
            "- Do not use a larger brow-lid gap by itself as evidence for brow raising; when brow geometry shows true downward motion, prefer brow lowerer.\n"
            "- Keep chin raiser only when lower-lip or chin geometry supports it; do not infer AU17 from generic lower-face drift alone.\n"
            "- Use eyelid geometry as supporting evidence for lid tightener or upper lid raiser when the raw eye-flow cue is weak.\n"
            "- Use upper-lip-to-nose compression as supporting evidence for upper lip raiser when the mouth flow is subtle.\n"
            "- Do not assert lip tightener from CASME2 mouth motion unless it has strong calibrated support; otherwise prefer a smaller AU set.\n"
        )
    return ""


@dataclass
class _ClipAnalysis:
    emotion: str
    aus: list[str]
    brow_reasoning: str
    mouth_reasoning: str
    agent_trace: dict[str, Any]


class SpatioFunctionalMASPipeline:
    def __init__(
        self,
        model_name: str = "",
        gpu_memory_utilization: float = 0.90,
        reasoning_model_name: str = "",
        reasoning_gpu_memory_utilization: float = 0.90,
        reasoning_device: str = "",
        reasoning_tensor_parallel_size: int = 1,
        vision_device: str = "",
        no_vision: bool = False,
        debug: bool = False,
        facs_mode: bool = False,
        memory_path: str = "",
        scope_guidelines_path: str = "",
        api_provider: str = "",
        api_model: str = "",
        api_key: str = "",
        api_base_url: str = "",
        api_concurrency: int = 8,
        ablation: AblationConfig | None = None,
    ):
        del gpu_memory_utilization, vision_device, facs_mode, memory_path, scope_guidelines_path
        del api_provider, api_model, api_key, api_base_url, api_concurrency
        self.debug = debug
        self.no_vision = no_vision
        self.ablation = ablation or AblationConfig()
        self.model_name = reasoning_model_name or model_name
        if LLM is None or AutoTokenizer is None or SamplingParams is None:
            raise RuntimeError("transformers and vllm are required for local_pipeline")
        if reasoning_tensor_parallel_size < 1:
            raise ValueError("reasoning_tensor_parallel_size must be at least 1")

        if not no_vision:
            print("[local_pipeline] Vision is not loaded in the simplified local pipeline; using text + numeric cues.")

        self.reasoning_tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        if reasoning_device:
            print(f"[local_pipeline] Reasoning model pinned to CUDA device {reasoning_device}")
        visible_devices = [part.strip() for part in str(reasoning_device).split(",") if part.strip()]
        if visible_devices and reasoning_tensor_parallel_size > len(visible_devices):
            raise ValueError(
                "reasoning_tensor_parallel_size cannot exceed the number of reasoning devices"
            )
        llm_kwargs = {
            "model": self.model_name,
            "gpu_memory_utilization": reasoning_gpu_memory_utilization,
            "max_model_len": 8192,
        }
        if reasoning_tensor_parallel_size > 1:
            llm_kwargs["tensor_parallel_size"] = reasoning_tensor_parallel_size
            print(
                "[local_pipeline] Reasoning tensor parallelism enabled: "
                f"tp={reasoning_tensor_parallel_size}"
            )
        try:
            if "enable_prefix_caching" in inspect.signature(LLM).parameters:
                llm_kwargs["enable_prefix_caching"] = True
                if self.debug:
                    print("[local_pipeline] vLLM prefix caching enabled for reasoning prompts.")
        except Exception:
            pass
        with _temporary_visible_devices(reasoning_device):
            self.reasoning_llm = LLM(**llm_kwargs)

    def analyze_of_clips_batch_mas(self, unique_records: list[dict]) -> dict[tuple, dict]:
        cache: dict[tuple, dict] = {}
        for record in unique_records:
            key = (record["video"], record["image_path"])
            cache[key] = {
                "observer_note": "",
                "brow_reasoning": "",
                "mouth_reasoning": "",
            }
        return cache

    def _build_chat_prompt(self, prompt: str, *, system_prompt: str | None = None) -> str:
        return self.reasoning_tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": system_prompt
                    or (
                        "You are a careful micro-expression reasoner working in a multi-agent setting. "
                        + (
                            "Treat the knowledge prior and landmark AU suggestions as useful memory, not ground truth. "
                            if not self.ablation.disable_prior
                            else "No external prior is available in this ablation; rely only on current evidence. "
                        )
                        + "Use them to guide attention, but only keep emotions and AUs that are supported by the current evidence. "
                        + "Prefer conservative final predictions over speculative extra AUs. "
                        + "Return valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )

    def _generate_json_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 400,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        if not prompts:
            return []
        prompt_texts = [
            self._build_chat_prompt(prompt, system_prompt=system_prompt) for prompt in prompts
        ]
        if self.debug:
            print(
                f"[local_pipeline] Batched reasoning call: prompts={len(prompt_texts)} max_tokens={max_tokens}"
            )
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = self.reasoning_llm.generate(prompt_texts, params)
        results: list[dict[str, Any]] = []
        for output in outputs:
            text = output.outputs[0].text if output.outputs else ""
            results.append(_safe_json_loads(text))
        return results

    def _generate_json(self, prompt: str, *, max_tokens: int = 400, system_prompt: str | None = None) -> dict[str, Any]:
        results = self._generate_json_batch(
            [prompt],
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        return results[0] if results else {}

    def _build_head_motion_prompt(
        self,
        *,
        dataset: str,
        numeric_views: dict[str, str],
        dataset_guidance: str,
    ) -> tuple[str, str]:
        prompt = (
            f"Dataset: {dataset}\n"
            f"{numeric_views['head_motion_agent']}\n"
            f"{dataset_guidance}\n"
            "Role: Head Motion Agent.\n"
            "Analyze only the head motion and its possible influence on brow evidence.\n"
            "Use the numeric head-motion summary and brow evidence card together.\n"
            "Your job is not to decide final brow AUs.\n"
            "Return one JSON object only with keys: motion_present, primary_direction, magnitude, brow_summary, possible_brow_effects, raw_reasoning, confidence.\n"
            "motion_present must be one of {yes, no, uncertain}.\n"
            "primary_direction must be one of {upward, downward, none, uncertain}.\n"
            "Project diagonal motion onto its vertical component: upward-left and upward-right should both be reported as upward; downward-left and downward-right should both be reported as downward; purely left-right motion should be reported as none.\n"
            "magnitude must be one of {small, moderate, large, uncertain}.\n"
            "brow_summary must be two short sentences: one about how the current head motion most likely changes optical flow around frontalis-related regions at this motion magnitude, and one about how it most likely changes optical flow around corrugator/procerus-related regions at this motion magnitude.\n"
            "It is valid to say that the current head motion is not likely to noticeably affect one or both of these regions.\n"
            "Write those two sentences as neutral pose-analysis context, not as final AU instructions.\n"
            "possible_brow_effects should briefly explain how the pose motion could make optical flow around frontalis-related or corrugator/procerus-related regions look exaggerated, underestimated, or only minimally affected.\n"
            "raw_reasoning should be a compact prose explanation of what the pose is doing and why that matters for brow interpretation.\n"
            "Do not output brow_keep_aus, brow_drop_aus, family_winner, or final brow labels."
        )
        system_prompt = (
            "You are a head-motion specialist in a multi-agent micro-expression workflow. "
            "You only analyze pose motion and its possible effects on brow evidence. "
            "Do not infer emotion or decide final brow AUs. "
            "Return valid JSON only."
        )
        return prompt, system_prompt

    def _build_brow_prompt(
        self,
        *,
        dataset: str,
        numeric_views: dict[str, str],
        dataset_guidance: str,
        head_motion_analysis: dict[str, Any],
    ) -> tuple[str, str]:
        prompt = (
            f"Dataset: {dataset}\n"
            f"{numeric_views['brow_agent']}\n"
            f"{dataset_guidance}\n"
            f"Head Motion Agent report:\n{_json_dumps_compact(head_motion_analysis)}\n"
            "Role: Brow Analyst.\n"
            "Use the brow evidence card as the primary source and use the Head Motion Agent report as pose-analysis context.\n"
            "Key knowledge:\n"
            "- AU1 inner brow raiser belongs to the frontalis-family brow raises, especially the medial portion of frontalis.\n"
            "- AU1 action: inner brow raiser lifts the medial brow.\n"
            "- AU2 outer brow raiser belongs to the frontalis-family brow raises, especially the lateral portion of frontalis.\n"
            "- AU2 action: outer brow raiser lifts the lateral brow.\n"
            "- AU4 brow lowerer belongs to the brow-lowering / knitting family linked to corrugator supercilii, depressor supercilii, and/or procerus.\n"
            "- AU4 action: brow lowerer knits and lowers the brow area.\n"
            "- AU4 can appear through different visible subpatterns; it does not need every subtype at once.\n"
            "- A larger brow-lid gap by itself is not enough evidence for AU1.\n"
            "Analyze the evidence in your own words, but do not make the final AU keep/drop decision yourself.\n"
            "Treat clip-level prior suggestions and dataset-level AU rates as weak context only; they can guide attention or break a tie, but they must not rescue weak local brow evidence.\n"
            "When brow score/norm are low and present=false for AU1/AU2/AU4, say that the brow evidence remains unconfirmed instead of forcing a brow AU.\n"
            "When you argue for or against AU1/AU2/AU4, ground the explanation in concrete brow-card fields or residual-flow evidence rather than generic appearance alone.\n"
            "Return one JSON object only with key: raw_reasoning.\n"
            "raw_reasoning should be concrete and informative. Use the brow evidence card and the anatomy notes, mention uncertainty when needed, and avoid turning the response into a numbered protocol or a restatement of the prompt.\n"
            "When pose matters, rely on the Head Motion Agent report instead of re-deriving pose theory yourself. Explain how the reported pose context does or does not change the interpretation of the local brow evidence.\n"
            "Do not output keep_aus, drop_aus, family_winner, or confidence."
        )
        system_prompt = (
            "You are a brow analyst in a multi-agent micro-expression workflow. "
            "Your job is to interpret structured brow evidence and produce an exploratory brow analysis for downstream review. "
            "Reason from the brow evidence card, the supplied anatomy knowledge, and the Head Motion Agent report, then return the requested JSON schema. "
            "Do not infer emotion. Return valid JSON only."
        )
        return prompt, system_prompt

    def _build_brow_arbiter_prompt(
        self,
        *,
        dataset: str,
        numeric_views: dict[str, str],
        dataset_guidance: str,
        head_motion_analysis: dict[str, Any],
        brow_analysis: dict[str, Any],
    ) -> tuple[str, str]:
        prompt = (
            f"Dataset: {dataset}\n"
            f"{numeric_views['brow_agent']}\n"
            f"{dataset_guidance}\n"
            f"Head Motion Agent report:\n{_json_dumps_compact(head_motion_analysis)}\n"
            f"Brow analyst report:\n{_json_dumps_compact(brow_analysis)}\n"
            "Role: Brow Arbiter.\n"
            "Use the brow evidence card, the Head Motion Agent report, and the Brow Analyst raw reasoning together, but make the final brow AU decision yourself.\n"
            "Rules:\n"
            "1. You may keep only canonical brow AUs from {inner brow raiser, outer brow raiser, brow lowerer}. Any other label is invalid.\n"
            "2. brow_state must be either local_brow_action or no_reliable_brow_au.\n"
            "3. It is valid to keep no brow AU when the evidence remains ambiguous.\n"
            "4. Prefer one brow AU unless the evidence clearly supports simultaneous inner and outer brow raising.\n"
            "5. Treat the Brow Analyst as a useful hypothesis generator, not as an unquestionable authority.\n"
            "6. Audit the Brow Analyst instead of echoing it. If the raw reasoning sounds plausible but the evidence card does not clearly support a local brow action, choose no_reliable_brow_au.\n"
            "7. Make the final brow decision from local brow evidence, not from generic surprise-like appearance, prior labels, or persuasive wording alone.\n"
            "8. Use the evidence card and the Brow Analyst reasoning together, but make your own final judgment.\n"
            "9. Use the Head Motion Agent report as the pose reference when deciding whether observed brow flow is more likely local action or pose-driven drift.\n"
            "10. When the Brow Analyst provides a weak or hand-wavy argument, trust the evidence card over the narrative.\n"
            "11. Treat clip-level prior suggestions and dataset-level AU rates as weak context only; they can break a tie between already plausible brow interpretations, but they must not rescue weak local evidence.\n"
            "12. If AU1/AU2/AU4 still look unconfirmed in the brow evidence card, especially when present=false and the Brow Analyst relies mainly on appearance or pose narrative, choose no_reliable_brow_au.\n"
            "13. For AU4 specifically, if brow lowerer is clearly stronger than AU1 and AU2 across the brow card and the local evidence shows concentrated brow-drop / corrugator-like support, you may keep AU4 even when present=false; explain that concrete dominance explicitly.\n"
            "13b. Dominant AU4 raw/combined/residual evidence from the brow card counts as concrete local evidence; you do not need a separate geometry narrative when the local AU4 evidence is already clearly strongest.\n"
            "14. You must explicitly report local_evidence_status as one of {confirmed, ambiguous, pose_confounded, absent}.\n"
            "15. If local_evidence_status is ambiguous, pose_confounded, or absent, then brow_state must be no_reliable_brow_au and brow_keep_aus must be empty.\n"
            "16. If you keep a brow AU, local_evidence_status must be confirmed and brow_summary must explain the local evidence rather than saying the evidence is unconfirmed.\n"
            "17. You must explicitly report numeric_support_assessment as one of {strong, moderate, weak, none}.\n"
            "18. You must explicitly report pose_risk as one of {low, medium, high}.\n"
            "19. confirmation_basis must be a short sentence that cites the concrete local evidence you relied on from the brow card or residual-flow view. Generic statements like 'looks surprise-like' or 'fits the pattern' are not sufficient.\n"
            "20. why_not_no_reliable_brow_au must be a short sentence explaining why the local evidence is strong enough to survive a skeptical no-reliable-brow-AU decision.\n"
            "21. If numeric_support_assessment is weak or none, or if the only argument is appearance, prior, or pose narrative, the default decision should be ambiguous or pose_confounded rather than confirmed.\n"
            "22. If you keep a brow AU while present=false or agent_only=true in the evidence card, explicitly justify why the local brow card still supports that AU instead of no_reliable_brow_au, and cite the dominant raw/combined/residual evidence instead of generic pose language.\n"
            "23. Never leave confirmation_basis blank when you keep a brow AU; name the dominant local evidence directly.\n"
            "Return one JSON object only with keys: family_winner, local_evidence_status, numeric_support_assessment, pose_risk, brow_state, brow_keep_aus, brow_drop_aus, confirmation_basis, why_not_no_reliable_brow_au, brow_summary, confidence.\n"
            "family_winner must be one of {raise_family, lower_family, no_reliable_brow_au}."
        )
        system_prompt = (
            "You are a brow arbiter in a multi-agent micro-expression workflow. "
            "Your job is to convert structured brow evidence, the Head Motion Agent report, and the Brow Analyst raw reasoning into a final brow AU decision. "
            "Return valid JSON only."
        )
        return prompt, system_prompt

    def _build_brow_skeptic_prompt(
        self,
        *,
        dataset: str,
        numeric_views: dict[str, str],
        dataset_guidance: str,
        head_motion_analysis: dict[str, Any],
        brow_analysis: dict[str, Any],
        brow_arbiter: dict[str, Any],
    ) -> tuple[str, str]:
        prompt = (
            f"Dataset: {dataset}\n"
            f"{numeric_views['brow_agent']}\n"
            f"{dataset_guidance}\n"
            f"Head Motion Agent report:\n{_json_dumps_compact(head_motion_analysis)}\n"
            f"Brow analyst report:\n{_json_dumps_compact(brow_analysis)}\n"
            f"Brow arbiter report:\n{_json_dumps_compact(brow_arbiter)}\n"
            "Role: Brow Skeptic.\n"
            "Your only job is to audit whether the Brow Arbiter has over-claimed a brow AU from weak local evidence.\n"
            "Rules:\n"
            "1. audit_decision must be one of {approve, downgrade}.\n"
            "2. Head motion can explain or discount brow appearance, but it must not rescue weak local evidence into confirmed.\n"
            "3. A relative winner among AU1/AU2/AU4 is not enough by itself to justify confirmed.\n"
            "4. If the Brow Analyst or Brow Arbiter uses words like unconfirmed, weak, uncertain, ambiguous, or pose-driven, default to downgrade unless concrete local evidence clearly overrides that language.\n"
            "5. If the brow evidence card says present=false or agent_only=true, require concrete local evidence in confirmation_basis. Generic phrases like looks surprise-like, matches the pattern, or more consistent than another brow AU are not enough.\n"
            "6. Do not treat present=false or agent_only=true as automatic downgrade when the brow card still shows clearly stronger local raw/geometry evidence for one brow AU than for the alternatives; in that case, audit whether the confirmation_basis cites that concrete evidence.\n"
            "7. Strong brow-drop geometry, strong residual-flow concentration, or clearly dominant AU4 raw/combined evidence can justify approving brow lowerer even when the SAMM brow detector leaves present=false.\n"
            "8. Do not downgrade AU4 just because the confirmation_basis is concise if the Brow Arbiter still cites clearly dominant local brow evidence from the card; concise is acceptable when it is concrete.\n"
            "8b. Dominant AU4 raw/combined/residual evidence itself counts as concrete local evidence; do not insist on separate geometry wording when the card already shows a clearly strongest AU4 pattern.\n"
            "9. If confirmation_basis relies mainly on head motion, prior, appearance, or tie-breaking narrative instead of local brow evidence, downgrade.\n"
            "10. If numeric_support_assessment is weak or none, confirmed is high risk and usually should be downgraded, but a clearly dominant AU4 local pattern can still survive if the basis is concrete.\n"
            "11. Approve only when the Brow Arbiter clearly cites local brow evidence that survives a skeptical no_reliable_brow_au alternative.\n"
            "Return one JSON object only with keys: audit_decision, main_issue, critic_summary, confidence.\n"
            "critic_summary should be short and specific about why the brow decision should survive or be downgraded."
        )
        system_prompt = (
            "You are a skeptical brow auditor in a multi-agent micro-expression workflow. "
            "Your job is to stop the Brow Arbiter from upgrading weak brow evidence into a confirmed AU. "
            "Return valid JSON only."
        )
        return prompt, system_prompt

    def _build_eye_prompt(
        self,
        *,
        dataset: str,
        numeric_views: dict[str, str],
        dataset_guidance: str,
        head_motion_analysis: dict[str, Any],
    ) -> tuple[str, str]:
        prompt = (
            f"Dataset: {dataset}\n"
            f"{numeric_views['eye_agent']}\n"
            f"{dataset_guidance}\n"
            f"Head Motion Agent report:\n{_json_dumps_compact(head_motion_analysis)}\n"
            "Role: Eye Analyst.\n"
            "Use the eye evidence card as the primary source and use the Head Motion Agent report as pose-analysis context.\n"
            "Key knowledge:\n"
            "- AU5 upper lid raiser widens the eye opening through upper-lid lift; do not infer it from brow raise or brow-lid gap change alone.\n"
            "- AU7 lid tightener narrows and tightens the eyelids without requiring full closure.\n"
            "- AU46 eye closure indicates near-closure or blink-like closure rather than simple narrowing.\n"
            "- AU7 and AU46 are competing eyelid-state explanations here, not automatic double confirmation.\n"
            "Analyze the evidence in your own words, but do not make the final AU keep/drop decision yourself.\n"
            "Treat clip-level prior suggestions and dataset-level AU rates as weak context only; they can guide attention or break a tie, but they must not rescue weak local eye evidence.\n"
            "When eye score/norm are low and present=false across AU5/AU7/AU46, say that the local eye evidence remains unconfirmed instead of forcing an eye AU.\n"
            "Ground the explanation in concrete eye-card fields or geometry rather than generic surprise-like or tension-like appearance.\n"
            "Return one JSON object only with key: raw_reasoning.\n"
            "raw_reasoning should be concrete and informative, mention uncertainty when needed, and avoid turning the response into a numbered protocol or a restatement of the prompt.\n"
            "Do not output keep_aus, drop_aus, eye_state, or confidence."
        )
        system_prompt = (
            "You are an eye specialist in a multi-agent micro-expression workflow. "
            "Your job is to interpret structured eyelid evidence and produce an exploratory eye analysis for downstream review. "
            "Reason from the eye evidence card, the supplied anatomy knowledge, and the Head Motion Agent report, then return the requested JSON schema. "
            "Do not infer emotion. Return valid JSON only."
        )
        return prompt, system_prompt

    def _build_eye_arbiter_prompt(
        self,
        *,
        dataset: str,
        numeric_views: dict[str, str],
        dataset_guidance: str,
        head_motion_analysis: dict[str, Any],
        eye_analysis: dict[str, Any],
    ) -> tuple[str, str]:
        prompt = (
            f"Dataset: {dataset}\n"
            f"{numeric_views['eye_agent']}\n"
            f"{dataset_guidance}\n"
            f"Head Motion Agent report:\n{_json_dumps_compact(head_motion_analysis)}\n"
            f"Eye analyst report:\n{_json_dumps_compact(eye_analysis)}\n"
            "Role: Eye Arbiter.\n"
            "Use the eye evidence card, the Head Motion Agent report, and the Eye Analyst raw reasoning together, but make the final eye AU decision yourself.\n"
            "Rules:\n"
            "1. You may keep only canonical local eye AUs from {upper lid raiser, lid tightener, eye closure}. Any other label is invalid.\n"
            "2. eye_state must be one of {widening, narrowing, closure, no_reliable_eye_au}.\n"
            "3. It is valid to keep no eye AU when the evidence remains ambiguous.\n"
            "4. Keep at most one eye AU. AU7 lid tightener and AU46 eye closure must not both remain in eye_keep_aus.\n"
            "5. widening maps to upper lid raiser, narrowing maps to lid tightener, and closure maps to eye closure.\n"
            "6. Audit the Eye Analyst instead of echoing it. If the raw reasoning sounds plausible but the eye evidence card does not clearly support a local eyelid action, choose no_reliable_eye_au.\n"
            "7. Make the final eye decision from local eyelid evidence, not from generic surprise-like appearance, prior labels, or persuasive wording alone.\n"
            "8. If eyelid_signal or the eye geometry indicate near-closure, prefer eye closure over lid tightener unless the closure evidence is clearly weak.\n"
            "9. If the eye is narrowed without near-closure, prefer lid tightener over eye closure.\n"
            "10. Do not infer upper lid raiser from brow-lid gap increase alone; require local widening evidence from the eye card.\n"
            "11. Use the Head Motion Agent report as pose context when deciding whether the observed eyelid pattern is local or confounded.\n"
            "12. You must explicitly report local_evidence_status as one of {confirmed, ambiguous, pose_confounded, absent}.\n"
            "13. If local_evidence_status is ambiguous, pose_confounded, or absent, then eye_state must be no_reliable_eye_au and eye_keep_aus must be empty.\n"
            "14. If you keep an eye AU, local_evidence_status must be confirmed and eye_summary must explain the local evidence rather than saying the evidence is unconfirmed.\n"
            "15. You must explicitly report numeric_support_assessment as one of {strong, moderate, weak, none}.\n"
            "16. You must explicitly report pose_risk as one of {low, medium, high}.\n"
            "17. confirmation_basis must be a short sentence that cites the concrete local eye evidence you relied on. Generic statements like 'looks surprised' or 'looks tense' are not sufficient.\n"
            "18. why_not_no_reliable_eye_au must be a short sentence explaining why the local evidence is strong enough to survive a skeptical no_reliable_eye_au decision.\n"
            "19. If numeric_support_assessment is weak or none, or if the only argument is appearance, prior, or vague narrative, the default decision should be ambiguous or pose_confounded rather than confirmed.\n"
            "20. If you keep an eye AU while present=false or agent_only=true in the evidence card, explicitly justify why the local eye card still supports that AU instead of no_reliable_eye_au.\n"
            "Return one JSON object only with keys: local_evidence_status, numeric_support_assessment, pose_risk, eye_state, eye_keep_aus, eye_drop_aus, confirmation_basis, why_not_no_reliable_eye_au, eye_summary, confidence.\n"
        )
        system_prompt = (
            "You are an eye arbiter in a multi-agent micro-expression workflow. "
            "Your job is to convert structured eyelid evidence, the Head Motion Agent report, and the Eye Analyst raw reasoning into a final eye AU decision. "
            "Return valid JSON only."
        )
        return prompt, system_prompt

    def _build_mouth_prompt(
        self,
        *,
        dataset: str,
        numeric_views: dict[str, str],
        dataset_guidance: str,
    ) -> tuple[str, str]:
        del dataset, dataset_guidance
        prompt = (
            f"{numeric_views['mouth_agent']}\n"
            "Role: Mouth Analyst.\n"
            "Use the mouth evidence card as the primary source.\n"
            "Focus on a few key distinctions:\n"
            "- AU9 needs direct nose or nostril raise evidence, not generic upper-face tension.\n"
            "- AU10 needs upper-lip lift or compression, not generic mouth opening.\n"
            "- AU12 needs real lip-corner raise/lift/back/lateral pull. Subtle or asymmetric AU12 is allowed only when that corner evidence itself is concrete.\n"
            "- If the corners look pinched, compressed, inward, or reduced-corner-drop rather than clearly smile-like, compare AU14 before AU12.\n"
            "- AU17 is lower-lip/chin push-up. Keep it only when that push-up is primary.\n"
            "- If the corners look retracted, horizontal, or downward, especially with mouth opening or jaw lowering, compare AU20 before AU12.\n"
            "- AU25 is lip separation; AU26 needs clear jaw lowering.\n"
            "- Prefer the most specific local explanation, not the most common smile-like label.\n"
            "Analyze the evidence in your own words, but do not make the final AU keep/drop decision yourself.\n"
            "Treat prior suggestions and historical AU rates as weak context only; they may guide attention or break a tie, but they must not rescue weak local mouth evidence.\n"
            "When mouth score/norm are low and present=false across the candidate mouth AUs, say that the local mouth evidence remains unconfirmed instead of forcing a mouth AU.\n"
            "Ground the explanation in concrete mouth-card fields or residual-flow evidence rather than generic smile-like or tension-like appearance.\n"
            "Return one JSON object only with key: raw_reasoning.\n"
            "raw_reasoning should be concrete, concise, and mention uncertainty when needed.\n"
            "Do not output keep_aus, drop_aus, or confidence."
        )
        system_prompt = (
            "You are a mouth specialist in a multi-agent micro-expression workflow. "
            "Your job is to interpret structured lower-face evidence and produce an exploratory mouth analysis for downstream review. "
            "Reason from the mouth evidence card and the supplied anatomy knowledge, then return the requested JSON schema. "
            "Do not infer emotion. Return valid JSON only."
        )
        return prompt, system_prompt

    def _build_mouth_arbiter_prompt(
        self,
        *,
        dataset: str,
        numeric_views: dict[str, str],
        dataset_guidance: str,
        mouth_analysis: dict[str, Any],
    ) -> tuple[str, str]:
        del dataset, dataset_guidance
        prompt = (
            f"{numeric_views['mouth_agent']}\n"
            f"Mouth analyst report:\n{_json_dumps_compact(mouth_analysis)}\n"
            "Role: Mouth Arbiter.\n"
            "Make the final lower-face AU decision from the local mouth evidence card, using the Mouth Analyst only as a second opinion.\n"
            "Rules:\n"
            "1. You may keep only canonical mouth/nasal AUs from {nose wrinkler, upper lip raiser, lip corner puller, dimpler, lip corner depressor, lip stretcher, chin raiser, lip tightener, lips part, jaw drop}.\n"
            "2. mouth_state must be either local_mouth_action or no_reliable_mouth_au.\n"
            "3. Use local evidence, not generic appearance, as the deciding factor.\n"
            "4. Prefer the most specific reliable AU, not the most generic fallback. A small set is best, but the kept AU should explain the actual local pattern.\n"
            "5. AU12 requires real lip-corner raise/lift/back/lateral pull. Do not create it from opening, center raise, chin raise, or a vague smile-like impression. If inward pull, drop, pinching, or tension is as strong as the upward corner pull, do not default to AU12.\n"
            "6. Keep AU9 or AU10 when direct nose or upper-lip evidence is real, even if AU12 is also tempting. Do not let AU12 absorb a clearer AU9 or AU10 pattern.\n"
            "7. Keep AU14 when the corners look pinched, compressed, inward, or reduced-corner-drop rather than clearly smile-like. Keep AU20 when retraction, tension, horizontal pull, or non-smile opening dominates over upward corner pull. Keep AU17 only when chin or lower-lip push-up is the primary action.\n"
            "8. Separate AU25 and AU26: AU25 is lip separation; AU26 requires clear jaw lowering. When opening is strong but the corners are not smile-like, compare AU20/AU25/AU26 before AU12.\n"
            "9. Report local_evidence_status as one of {confirmed, ambiguous, pose_confounded, absent}; numeric_support_assessment as one of {strong, moderate, weak, none}; and pose_risk as one of {low, medium, high}. confirmed requires a non-empty mouth_keep_aus; otherwise mouth_keep_aus must be empty.\n"
            "10. confirmation_basis and why_not_no_reliable_mouth_au must cite the concrete local evidence you used. If the evidence is weak, generic, or pose-driven, choose no_reliable_mouth_au.\n"
            "Return one JSON object only with keys: local_evidence_status, numeric_support_assessment, pose_risk, mouth_state, mouth_keep_aus, mouth_drop_aus, confirmation_basis, why_not_no_reliable_mouth_au, mouth_summary, confidence.\n"
        )
        system_prompt = (
            "You are a mouth arbiter in a multi-agent micro-expression workflow. "
            "Your job is to convert structured mouth evidence and the Mouth Analyst raw reasoning into a final lower-face AU decision. "
            "Return valid JSON only."
        )
        return prompt, system_prompt

    def _build_motion_prompt(
        self,
        *,
        dataset: str,
        numeric_views: dict[str, str],
        dataset_guidance: str,
        brow_analysis: dict[str, Any],
        brow_arbiter: dict[str, Any],
        eye_analysis: dict[str, Any],
        eye_arbiter: dict[str, Any],
        mouth_analysis: dict[str, Any],
        mouth_arbiter: dict[str, Any],
    ) -> tuple[str, str]:
        prompt = (
            f"Dataset: {dataset}\n"
            f"{numeric_views['motion_agent']}\n"
            f"{dataset_guidance}\n"
            f"Brow analyst report:\n{_json_dumps_compact(brow_analysis)}\n"
            f"Brow arbiter report:\n{_json_dumps_compact(brow_arbiter)}\n"
            f"Eye analyst report:\n{_json_dumps_compact(eye_analysis)}\n"
            f"Eye arbiter report:\n{_json_dumps_compact(eye_arbiter)}\n"
            f"Mouth analyst report:\n{_json_dumps_compact(mouth_analysis)}\n"
            f"Mouth arbiter report:\n{_json_dumps_compact(mouth_arbiter)}\n"
            "Role: Agent A (Action-Capturer). Your job is only to interpret facial motion evidence and decide which AUs are directly supported.\n"
            "Protocol:\n"
            "1. Work from head-motion quality, residual motion, raw flow regions, and continuous eye/mouth geometry values.\n"
            "2. Separate upper-face and lower-face evidence.\n"
            "3. Mark any AU as rejected if it is weak, isolated, contradictory, or likely caused by global motion.\n"
            "4. Do not guess emotion yet.\n"
            "5. Be especially skeptical of smile-like mouth AUs when they appear alone.\n"
            "6. Eye AUs AU5/AU7/AU46 are owned by the Eye Arbiter. Do not include upper lid raiser, lid tightener, or eye closure in supported_aus or rejected_aus.\n"
            "7. Do not treat a larger brow-lid gap by itself as evidence for inner brow raiser; require actual brow raise relative to lid motion.\n"
            "8. Do not keep outer brow raiser from a surprise-like visual impression alone; require lateral-brow numeric or geometry corroboration.\n"
            "9. Brow AUs AU1/AU2/AU4 are owned by the Brow Arbiter. Do not include inner brow raiser, outer brow raiser, or brow lowerer in supported_aus or rejected_aus.\n"
            "10. Mouth/nasal AUs {nose wrinkler, upper lip raiser, lip corner puller, dimpler, lip corner depressor, lip stretcher, chin raiser, lip tightener, lips part, jaw drop} are owned by the Mouth Arbiter. Do not include them in supported_aus or rejected_aus.\n"
            "11. You may discuss owned brow or eye evidence only in upper_face_summary or motion_risks, and you may discuss owned mouth evidence only in lower_face_summary or motion_risks.\n"
            "12. In supported_aus and rejected_aus, output canonical AU labels only, never raw region names like corner_subr, lip_center, upper_lip, or chin.\n"
            "Return JSON with keys: supported_aus, rejected_aus, upper_face_summary, lower_face_summary, confidence, motion_risks.\n"
            'Example: {"supported_aus":["cheek raiser"],"rejected_aus":[],"upper_face_summary":"...",'
            '"lower_face_summary":"...","confidence":"medium","motion_risks":["leftward head motion"]}'
        )
        system_prompt = (
            "You are Agent A in a multi-agent micro-expression workflow. "
            "You specialize in action-unit evidence extraction from motion. "
            "Do not infer emotion. "
            "Continuous numeric evidence is more trustworthy than any pre-binarized AU hint. "
            "Return valid JSON only."
        )
        return prompt, system_prompt

    def _build_emotion_prompt(
        self,
        *,
        dataset: str,
        numeric_views: dict[str, str],
        dataset_guidance: str,
        brow_analysis: dict[str, Any],
        brow_arbiter: dict[str, Any],
        eye_analysis: dict[str, Any],
        eye_arbiter: dict[str, Any],
        mouth_analysis: dict[str, Any],
        mouth_arbiter: dict[str, Any],
        motion: dict[str, Any],
    ) -> tuple[str, str]:
        specialist_focus = _BROW_AUS | _EYE_AUS | _MOUTH_AUS if dataset == "samm" else set()
        motion_supported = [
            au for au in _extract_agent_aus(motion.get("supported_aus") or motion.get("aus"))
            if au not in specialist_focus
        ]
        motion_rejected = [
            au for au in _extract_agent_aus(motion.get("rejected_aus"))
            if au not in specialist_focus
        ]
        authority_snapshot = {
            "motion_supported_aus": motion_supported,
            "motion_rejected_aus": motion_rejected,
            "motion_risks": motion.get("motion_risks") or [],
            "brow_state": brow_arbiter.get("brow_state"),
            "brow_local_evidence_status": brow_arbiter.get("local_evidence_status"),
            "brow_numeric_support_assessment": brow_arbiter.get("numeric_support_assessment"),
            "brow_keep_aus": _extract_agent_aus(
                brow_arbiter.get("brow_keep_aus") or brow_arbiter.get("keep_aus")
            ),
            "brow_drop_aus": _extract_agent_aus(
                brow_arbiter.get("brow_drop_aus") or brow_arbiter.get("drop_aus")
            ),
            "eye_state": eye_arbiter.get("eye_state"),
            "eye_local_evidence_status": eye_arbiter.get("local_evidence_status"),
            "eye_numeric_support_assessment": eye_arbiter.get("numeric_support_assessment"),
            "eye_keep_aus": _extract_agent_aus(
                eye_arbiter.get("eye_keep_aus") or eye_arbiter.get("keep_aus")
            ),
            "eye_drop_aus": _extract_agent_aus(
                eye_arbiter.get("eye_drop_aus") or eye_arbiter.get("drop_aus")
            ),
            "mouth_state": mouth_arbiter.get("mouth_state"),
            "mouth_local_evidence_status": mouth_arbiter.get("local_evidence_status"),
            "mouth_numeric_support_assessment": mouth_arbiter.get("numeric_support_assessment"),
            "mouth_keep_aus": _extract_agent_aus(
                mouth_arbiter.get("mouth_keep_aus") or mouth_arbiter.get("keep_aus")
            ),
            "mouth_drop_aus": _extract_agent_aus(
                mouth_arbiter.get("mouth_drop_aus") or mouth_arbiter.get("drop_aus")
            ),
        }
        hallmark_guidance = (
            "6b. Treat data-learned pattern observations as weak hypotheses to test against the current clip, not as automatic votes.\n"
            "6c. Use the contrastive hallmark card as a structured comparison aid: it tells you which observed AUs align with each candidate emotion, which hallmark AUs are missing, and which observed AUs conflict toward another emotion, but you must still judge the current clip yourself.\n"
            if not self.ablation.disable_prior
            else "6b. No knowledge prior or contrastive hallmark card is available in this ablation; compare candidate emotions directly from the current clip evidence.\n"
        )
        prompt = (
            f"Dataset: {dataset}\n"
            f"{numeric_views['emotion_agent']}\n"
            f"{dataset_guidance}\n"
            f"Authoritative local-action snapshot:\n{_json_dumps_compact(authority_snapshot)}\n"
            f"Secondary specialist context (never override the authoritative snapshot):\n"
            f"- Brow analyst report: {_json_dumps_compact(brow_analysis)}\n"
            f"- Brow arbiter report: {_json_dumps_compact(brow_arbiter)}\n"
            f"- Eye analyst report: {_json_dumps_compact(eye_analysis)}\n"
            f"- Eye arbiter report: {_json_dumps_compact(eye_arbiter)}\n"
            f"- Mouth analyst report: {_json_dumps_compact(mouth_analysis)}\n"
            f"- Mouth arbiter report: {_json_dumps_compact(mouth_arbiter)}\n"
            f"- Agent A motion report: {_json_dumps_compact(motion)}\n"
            "Role: Agent B (Emotion-Diagnostician). Infer the most likely fine-grained emotion from Agent A's AU evidence and the geometry-aware numeric view.\n"
            "Protocol:\n"
            "1. Use the Authoritative local-action snapshot as the local-action view, then cross-check it against the eye/mouth geometry view.\n"
            "2. Compare at least two competing emotions before deciding.\n"
            "3. Explain why the best candidate beats the closest alternative.\n"
            "4. If the evidence is too weak for a smile-like interpretation, do not default to happiness.\n"
            "5. If upper-face evidence is stronger than mouth evidence, let that dominate the diagnosis.\n"
            "6. Use the Eye Arbiter's resolved eyelid state as the authoritative eye view; do not treat eyelid cues as duplicate confirmation just because multiple eye-related hypotheses were mentioned upstream.\n"
            f"{hallmark_guidance}"
            "6d. If Brow Arbiter says no_reliable_brow_au or drops a brow AU, do not describe AU1/AU2/AU4 as confirmed or use them as the main reason for the emotion. You may mention only weak, unconfirmed brow tendencies in free text.\n"
            "6e. If Eye Arbiter says no_reliable_eye_au or drops an eye AU, do not describe AU5/AU7/AU46 as confirmed or use them as the main reason for the emotion. You may mention only weak, unconfirmed eyelid tendencies in free text.\n"
            "6f. If Mouth Arbiter says no_reliable_mouth_au or drops a mouth/nasal AU, do not describe AU9/AU10/AU12/AU14/AU15/AU17/AU20/AU23/AU25/AU26 as confirmed or use them as the main reason for the emotion. You may mention only weak, unconfirmed mouth tendencies in free text.\n"
            "7. Fear deserves an explicit comparison when the evidence shows mouth tension or retraction rather than a clear smile: look for lip stretcher, jaw drop, lips part, jaw clencher, upper-lid widening, or non-smile corner retraction.\n"
            "8. Do not map lip-corner motion to happiness unless the corners are clearly lifted like a smile. If the corners look retracted, horizontal, or downward, especially with mouth opening or upper-face widening, compare fear before happiness or contempt.\n"
            "9. If anger, disgust, and contempt cues are all weak or contradictory in a negative clip, do not default to contempt; compare fear explicitly.\n"
            "9b. If lid tightener is strong and the brow card still shows brow-lowerer dominance over AU1/AU2, keep anger in the comparison before defaulting to sadness or fear, even if the brow specialist declines to confirm AU4.\n"
            "10. supporting_aus should not include AU1/AU2/AU4, AU5/AU7/AU46, or the Mouth-Arbiter-owned mouth/jaw AUs; the specialists own those AUs and you may refer to them only in free-text reasoning.\n"
            "11. If the snapshot says brow_keep_aus=[] and brow_state=no_reliable_brow_au, you may still mention weak brow tendencies, but you must not make them the main evidence for the chosen emotion.\n"
            "12. If the snapshot says eye_keep_aus=[] and eye_state=no_reliable_eye_au, you may still mention weak eyelid tendencies, but you must not make them the main evidence for the chosen emotion.\n"
            "13. If the snapshot says mouth_keep_aus=[] and mouth_state=no_reliable_mouth_au, you may still mention weak mouth tendencies, but you must not make them the main evidence for the chosen emotion.\n"
            "14. Use the secondary specialist context only to explain nuance, local ambiguity, or pose confounds; when it conflicts with the authoritative snapshot, follow the snapshot.\n"
            "Return JSON with keys: emotion, supporting_aus, alternatives, reasoning, confidence."
        )
        system_prompt = (
            "You are Agent B in a multi-agent micro-expression workflow. "
            "You specialize in mapping AU evidence to emotion. "
            "Do not invent AUs that are unsupported by Agent A or by the local specialist arbiters. "
            "Never turn a dropped or no_reliable specialist AU into confirmed emotion evidence. "
            "Be especially cautious about happiness when only a single mouth AU is present. "
            "Return valid JSON only."
        )
        return prompt, system_prompt

    def _build_critic_prompt(
        self,
        *,
        dataset: str,
        numeric_views: dict[str, str],
        dataset_guidance: str,
        brow_analysis: dict[str, Any],
        brow_arbiter: dict[str, Any],
        eye_analysis: dict[str, Any],
        eye_arbiter: dict[str, Any],
        mouth_analysis: dict[str, Any],
        mouth_arbiter: dict[str, Any],
        motion: dict[str, Any],
        emotion_hypothesis: dict[str, Any],
    ) -> tuple[str, str]:
        hallmark_guidance = (
            "1. Start from Brow/Eye/Mouth Arbiter as the local authorities. Compare all three together, then use the contrastive hallmark card as a global cross-check.\n"
            "2. Prefer hallmark and strong conflict cues over auxiliary cues. Do not let a weak smile-like mouth cue override stronger upper-face evidence.\n"
            if not self.ablation.disable_prior
            else "1. Start from Brow/Eye/Mouth Arbiter as the local authorities and compare all three together directly from current evidence; no hallmark prior is available in this ablation.\n"
        )
        prompt = (
            f"Dataset: {dataset}\n"
            f"{numeric_views['critic_agent']}\n"
            f"{dataset_guidance}\n"
            f"Brow analyst report:\n{_json_dumps_compact(brow_analysis)}\n"
            f"Brow arbiter report:\n{_json_dumps_compact(brow_arbiter)}\n"
            f"Eye analyst report:\n{_json_dumps_compact(eye_analysis)}\n"
            f"Eye arbiter report:\n{_json_dumps_compact(eye_arbiter)}\n"
            f"Mouth analyst report:\n{_json_dumps_compact(mouth_analysis)}\n"
            f"Mouth arbiter report:\n{_json_dumps_compact(mouth_arbiter)}\n"
            f"Agent A motion report:\n{_json_dumps_compact(motion)}\n"
            f"Agent B emotion report:\n{_json_dumps_compact(emotion_hypothesis)}\n"
            "Role: Agent C (Adversarial-Critic). Audit Agent A and Agent B, then give your own best final judgment.\n"
            "Protocol:\n"
            f"{hallmark_guidance}"
            "3. Treat low-reliability calibrated AUs as risky unless both motion and geometry support them.\n"
            "4. If Agent B relies on dropped or no_reliable specialist AUs, call that out and revise the emotion.\n"
            "5. revised_emotion must be your own best final emotion label and must never be blank.\n"
            "6. keep_aus and drop_aus are for non-specialist AUs only. Use canonical AU labels only.\n"
            "7. Brow Arbiter owns {inner brow raiser, outer brow raiser, brow lowerer}; Eye Arbiter owns {upper lid raiser, lid tightener, eye closure}; Mouth Arbiter owns {nose wrinkler, upper lip raiser, lip corner puller, dimpler, lip corner depressor, lip stretcher, chin raiser, lip tightener, lips part, jaw drop}. Do not output those AUs in keep_aus or drop_aus.\n"
            "8. If SAMM specialist keep_aus are all empty, you may output at most one canonical specialist_fallback_aus. Choose it only when one AU is clearly the single most coherent cross-specialist explanation; otherwise return [].\n"
            "9. Prefer a smaller, higher-confidence AU set over a larger speculative one.\n"
            "Return JSON with keys: revised_emotion, keep_aus, drop_aus, specialist_fallback_aus, brow_summary, main_issues, critic_summary, confidence. revised_emotion must never be blank."
        )
        system_prompt = (
            "You are Agent C in a multi-agent micro-expression workflow. "
            "You are a skeptical cross-specialist reviewer. "
            "Return valid JSON only."
        )
        return prompt, system_prompt

    def _finalize_clip_analysis(
        self,
        *,
        dataset: str,
        head_motion_analysis: dict[str, Any],
        brow_analysis: dict[str, Any],
        brow_arbiter: dict[str, Any],
        brow_skeptic: dict[str, Any],
        eye_analysis: dict[str, Any],
        eye_arbiter: dict[str, Any],
        mouth_analysis: dict[str, Any],
        mouth_arbiter: dict[str, Any],
        motion: dict[str, Any],
        emotion_hypothesis: dict[str, Any],
        critic: dict[str, Any],
    ) -> _ClipAnalysis:
        critic_enabled = not self.ablation.disable_critic
        brow_keep: list[str] = []
        brow_drop: list[str] = []
        brow_focus = set(_BROW_AUS)
        eye_keep: list[str] = []
        eye_drop: list[str] = []
        eye_focus = set(_EYE_AUS)
        mouth_keep: list[str] = []
        mouth_drop: list[str] = []
        mouth_focus = set(_MOUTH_AUS)
        if dataset == "samm":
            family_winner = str(brow_arbiter.get("family_winner") or "").strip().lower()
            brow_keep = [
                au for au in _extract_agent_aus(
                    brow_arbiter.get("brow_keep_aus") or brow_arbiter.get("keep_aus")
                ) if au in brow_focus
            ]
            brow_drop = [
                au for au in _extract_agent_aus(
                    brow_arbiter.get("brow_drop_aus") or brow_arbiter.get("drop_aus")
                ) if au in brow_focus
            ]
            arbiter_brow_state = str(
                brow_arbiter.get("brow_state")
                or brow_arbiter.get("state")
                or ""
            ).strip().lower()
            normalized_state_label = _normalize_au_token(arbiter_brow_state.replace("_", " "))
            if normalized_state_label in brow_focus:
                if normalized_state_label not in brow_keep:
                    brow_keep = [normalized_state_label, *brow_keep]
                arbiter_brow_state = "local_brow_action"
            if arbiter_brow_state in {"no_reliable_brow_au", "no_brow_au", "no_brow", "null_explanation"}:
                brow_keep = []
                brow_drop = sorted(brow_focus)
            elif not brow_keep and not brow_drop and family_winner == "no_reliable_brow_au":
                brow_drop = sorted(brow_focus)
            if not brow_keep and not brow_drop:
                brow_drop = sorted(brow_focus)
            eye_keep = [
                au for au in _extract_agent_aus(
                    eye_arbiter.get("eye_keep_aus") or eye_arbiter.get("keep_aus")
                ) if au in eye_focus
            ]
            eye_drop = [
                au for au in _extract_agent_aus(
                    eye_arbiter.get("eye_drop_aus") or eye_arbiter.get("drop_aus")
                ) if au in eye_focus
            ]
            arbiter_eye_state = _sanitize_eye_state(
                eye_arbiter.get("eye_state")
                or eye_arbiter.get("state")
                or ""
            )
            eye_state_to_au = {
                "widening": "upper lid raiser",
                "narrowing": "lid tightener",
                "closure": "eye closure",
            }
            if arbiter_eye_state in eye_state_to_au:
                preferred_eye = eye_state_to_au[arbiter_eye_state]
                eye_keep = [preferred_eye]
                eye_drop = [au for au in eye_drop if au != preferred_eye]
            elif eye_keep:
                eye_keep = [eye_keep[0]]
            if arbiter_eye_state == "no_reliable_eye_au":
                eye_keep = []
                eye_drop = sorted(eye_focus)
            elif not eye_keep and not eye_drop:
                eye_drop = sorted(eye_focus)
            mouth_keep = [
                au for au in _extract_agent_aus(
                    mouth_arbiter.get("mouth_keep_aus") or mouth_arbiter.get("keep_aus")
                ) if au in mouth_focus
            ]
            mouth_drop = [
                au for au in _extract_agent_aus(
                    mouth_arbiter.get("mouth_drop_aus") or mouth_arbiter.get("drop_aus")
                ) if au in mouth_focus
            ]
            arbiter_mouth_state = str(
                mouth_arbiter.get("mouth_state")
                or mouth_arbiter.get("state")
                or ""
            ).strip().lower()
            normalized_mouth_state = _normalize_au_token(arbiter_mouth_state.replace("_", " "))
            if normalized_mouth_state in mouth_focus:
                if normalized_mouth_state not in mouth_keep:
                    mouth_keep = [normalized_mouth_state, *mouth_keep]
                arbiter_mouth_state = "local_mouth_action"
            if arbiter_mouth_state in {"no_reliable_mouth_au", "no_mouth_au", "no_mouth", "null_explanation"}:
                mouth_keep = []
                mouth_drop = sorted(mouth_focus)
            elif not mouth_keep and not mouth_drop:
                mouth_drop = sorted(mouth_focus)

        motion_supported = _extract_agent_aus(motion.get("supported_aus") or motion.get("aus"))
        motion_rejected = _extract_agent_aus(motion.get("rejected_aus"))
        critic_keep_raw = _extract_agent_aus(critic.get("keep_aus"))
        critic_drop = set(_extract_agent_aus(critic.get("drop_aus")))
        critic_specialist_fallback = _extract_agent_aus(
            critic.get("specialist_fallback_aus") or critic.get("fallback_aus")
        )
        emotion_supporting = _extract_agent_aus(emotion_hypothesis.get("supporting_aus"))
        if dataset == "samm":
            specialist_focus = brow_focus | eye_focus | mouth_focus
            critic_specialist_fallback = _dedupe_keep_order([
                *critic_specialist_fallback,
                *[name for name in critic_keep_raw if name in specialist_focus],
            ])
            motion_supported = [name for name in motion_supported if name not in specialist_focus]
            motion_rejected = [name for name in motion_rejected if name not in specialist_focus]
            critic_keep = [name for name in critic_keep_raw if name not in specialist_focus]
            critic_drop = {name for name in critic_drop if name not in specialist_focus}
            emotion_supporting = [name for name in emotion_supporting if name not in specialist_focus]
        else:
            critic_keep = critic_keep_raw[:]
        fallback_emotion = normalize_emotion_for_dataset(
            str(emotion_hypothesis.get("emotion", "unknown")), dataset
        )
        critic_raw_emotion = str(critic.get("revised_emotion", "")).strip().lower()
        revised_emotion = normalize_emotion_for_dataset(
            str(critic.get("revised_emotion", "")), dataset
        )
        emotion = revised_emotion or fallback_emotion
        if emotion in {"", "unknown", "other"}:
            emotion = fallback_emotion

        aus = critic_keep[:]
        if not aus:
            aus = _dedupe_keep_order(
                [*motion_supported, *emotion_supporting]
            )
        if critic_drop:
            aus = [name for name in aus if name not in critic_drop]
        if not aus:
            aus = motion_supported[:]
        if dataset == "samm":
            brow_keep_focus = [name for name in brow_keep if name in brow_focus]
            brow_drop_focus = {name for name in brow_drop if name in brow_focus}
            eye_keep_focus = [name for name in eye_keep if name in eye_focus]
            eye_drop_focus = {name for name in eye_drop if name in eye_focus}
            mouth_keep_focus = [name for name in mouth_keep if name in mouth_focus]
            mouth_drop_focus = {name for name in mouth_drop if name in mouth_focus}
            aus = [name for name in aus if name not in (brow_focus | eye_focus | mouth_focus)]
            authoritative_upper = [*brow_keep_focus, *eye_keep_focus]
            if authoritative_upper:
                aus = _dedupe_keep_order([*authoritative_upper, *aus])
            if mouth_keep_focus:
                aus = _dedupe_keep_order([*aus, *mouth_keep_focus])
            if brow_drop_focus:
                aus = [name for name in aus if name not in brow_drop_focus]
            if eye_drop_focus:
                aus = [name for name in aus if name not in eye_drop_focus]
            if mouth_drop_focus:
                aus = [name for name in aus if name not in mouth_drop_focus]
            if not brow_keep_focus and not eye_keep_focus and not mouth_keep_focus:
                chosen_fallback = ""
                for name in critic_specialist_fallback:
                    if name in specialist_focus:
                        chosen_fallback = name
                        break
                if chosen_fallback in brow_focus:
                    brow_keep_focus = [chosen_fallback]
                    brow_drop_focus.discard(chosen_fallback)
                elif chosen_fallback in eye_focus:
                    eye_keep_focus = [chosen_fallback]
                    eye_drop_focus.discard(chosen_fallback)
                elif chosen_fallback in mouth_focus:
                    mouth_keep_focus = [chosen_fallback]
                    mouth_drop_focus.discard(chosen_fallback)
                if brow_keep_focus or eye_keep_focus:
                    aus = _dedupe_keep_order([*brow_keep_focus, *eye_keep_focus, *aus])
                if mouth_keep_focus:
                    aus = _dedupe_keep_order([*aus, *mouth_keep_focus])
                brow_keep = brow_keep_focus
                eye_keep = eye_keep_focus
                mouth_keep = mouth_keep_focus
                brow_drop = sorted(brow_drop_focus)
                eye_drop = sorted(eye_drop_focus)
                mouth_drop = sorted(mouth_drop_focus)

        upper_face = {
            "inner brow raiser",
            "outer brow raiser",
            "brow lowerer",
            "upper lid raiser",
            "lid tightener",
            "cheek raiser",
            "eye closure",
        }
        upper_final = [name for name in aus if name in upper_face]
        lower_final = [name for name in aus if name not in upper_face]
        upper_summary = str(motion.get("upper_face_summary", "")).strip()
        lower_summary = str(motion.get("lower_face_summary", "")).strip()
        critic_summary = str(critic.get("critic_summary", "")).strip()
        emotion_reasoning = str(emotion_hypothesis.get("reasoning", "")).strip()
        brow_raw_reasoning = str(
            brow_analysis.get("raw_reasoning") or brow_analysis.get("reasoning", "")
        ).strip()
        eye_raw_reasoning = str(
            eye_analysis.get("raw_reasoning") or eye_analysis.get("reasoning", "")
        ).strip()
        mouth_raw_reasoning = str(
            mouth_analysis.get("raw_reasoning") or mouth_analysis.get("reasoning", "")
        ).strip()
        arbiter_brow_summary = str(brow_arbiter.get("brow_summary", "")).strip()
        arbiter_eye_summary = str(eye_arbiter.get("eye_summary", "")).strip()
        arbiter_mouth_summary = str(mouth_arbiter.get("mouth_summary", "")).strip()
        critic_brow_summary = str(critic.get("brow_summary", "")).strip()
        upper_reasoning_parts = [
            part
            for part in [
                brow_raw_reasoning,
                arbiter_brow_summary,
                eye_raw_reasoning,
                arbiter_eye_summary,
                critic_brow_summary,
            ]
            if part
        ]
        brow_reasoning = " ".join(upper_reasoning_parts).strip() or upper_summary
        if not brow_reasoning:
            if upper_final:
                brow_reasoning = (
                    f"Final upper-face AUs after {'critic review' if critic_enabled else 'fusion'}: {', '.join(upper_final)}."
                )
            else:
                brow_reasoning = critic_summary or (
                    "No upper-face AU survived critic review."
                    if critic_enabled
                    else "No upper-face AU survived fusion."
                )

        mouth_reasoning = mouth_raw_reasoning or arbiter_mouth_summary or lower_summary
        if not mouth_reasoning:
            if lower_final:
                mouth_reasoning = (
                    f"Final lower-face AUs after {'critic review' if critic_enabled else 'fusion'}: {', '.join(lower_final)}."
                )
            else:
                mouth_reasoning = (
                    emotion_reasoning
                    or critic_summary
                    or (
                        "No lower-face AU survived critic review."
                        if critic_enabled
                        else "No lower-face AU survived fusion."
                    )
                )

        parsed = {
            "emotion": emotion,
            "aus": aus,
            "brow_reasoning": brow_reasoning,
            "mouth_reasoning": mouth_reasoning,
            "confidence": critic.get("confidence")
            or emotion_hypothesis.get("confidence")
            or motion.get("confidence", ""),
            "protocol_note": (
                "Synthesized from Agent A/B/C without an extra integrator generation."
                if critic_enabled
                else "Synthesized from Agent A/B without critic review."
            ),
            "source": "critic_consensus" if critic_enabled else "emotion_consensus",
            "critic_raw_emotion": critic_raw_emotion,
        }

        return _ClipAnalysis(
            emotion=emotion,
            aus=aus,
            brow_reasoning=brow_reasoning,
            mouth_reasoning=mouth_reasoning,
            agent_trace={
                "head_motion_agent": head_motion_analysis,
                "brow_agent": brow_analysis,
                "brow_arbiter": brow_arbiter,
                "brow_skeptic": brow_skeptic,
                "eye_agent": eye_analysis,
                "eye_arbiter": eye_arbiter,
                "mouth_agent": mouth_analysis,
                "mouth_arbiter": mouth_arbiter,
                "motion_agent": motion,
                "emotion_agent": emotion_hypothesis,
                "critic_agent": critic,
                "integrator_agent": parsed,
                "brow_keep_aus": brow_keep,
                "brow_drop_aus": brow_drop,
                "eye_keep_aus": eye_keep,
                "eye_drop_aus": eye_drop,
                "mouth_keep_aus": mouth_keep,
                "mouth_drop_aus": mouth_drop,
                "motion_supported_aus": motion_supported,
                "motion_rejected_aus": motion_rejected,
                "critic_specialist_fallback_aus": critic_specialist_fallback[:1],
                "evidence_protocol": (
                    "split_numeric_views"
                    if dataset == "samm" and not self.ablation.disable_specialists
                    else "shared_numeric_summary"
                ),
            },
        )

    def _analyze_clips_batch(
        self,
        clip_requests: list[dict[str, Any]],
    ) -> dict[tuple[str, str], _ClipAnalysis]:
        if not clip_requests:
            return {}

        states: list[dict[str, Any]] = []
        for request in clip_requests:
            dataset = request["dataset"]
            cache = request["cache"]
            states.append(
                {
                    "clip_id": request["clip_id"],
                    "image_path": request["image_path"],
                    "dataset": dataset,
                    "cache": cache,
                    "numeric_views": _clip_numeric_views(cache, dataset, ablation=self.ablation),
                    "dataset_guidance": _dataset_reasoning_guidance(dataset),
                    "head_motion_analysis": {},
                    "brow_analysis": {},
                    "brow_arbiter": {},
                    "brow_skeptic": {},
                    "eye_analysis": {},
                    "eye_arbiter": {},
                    "mouth_analysis": {},
                    "mouth_arbiter": {},
                    "motion": {},
                    "emotion_hypothesis": {},
                    "critic": {},
                }
            )

        samm_states = [state for state in states if state["dataset"] == "samm"]
        if samm_states and not self.ablation.disable_specialists:
            head_motion_prompts: list[str] = []
            head_motion_system_prompt = ""
            for state in samm_states:
                prompt, system_prompt = self._build_head_motion_prompt(
                    dataset=state["dataset"],
                    numeric_views=state["numeric_views"],
                    dataset_guidance=state["dataset_guidance"],
                )
                head_motion_prompts.append(prompt)
                if not head_motion_system_prompt:
                    head_motion_system_prompt = system_prompt
            head_motion_outputs = self._generate_json_batch(
                head_motion_prompts,
                max_tokens=260,
                system_prompt=head_motion_system_prompt,
            )
            for state, head_motion_analysis in zip(samm_states, head_motion_outputs):
                state["head_motion_analysis"] = _sanitize_head_motion_agent_output(head_motion_analysis)

            brow_prompts: list[str] = []
            brow_system_prompt = ""
            for state in samm_states:
                prompt, system_prompt = self._build_brow_prompt(
                    dataset=state["dataset"],
                    numeric_views=state["numeric_views"],
                    dataset_guidance=state["dataset_guidance"],
                    head_motion_analysis=state["head_motion_analysis"],
                )
                brow_prompts.append(prompt)
                if not brow_system_prompt:
                    brow_system_prompt = system_prompt
            brow_outputs = self._generate_json_batch(
                brow_prompts,
                max_tokens=700,
                system_prompt=brow_system_prompt,
            )
            for state, brow in zip(samm_states, brow_outputs):
                state["brow_analysis"] = brow

            brow_arbiter_prompts: list[str] = []
            brow_arbiter_system_prompt = ""
            for state in samm_states:
                prompt, system_prompt = self._build_brow_arbiter_prompt(
                    dataset=state["dataset"],
                    numeric_views=state["numeric_views"],
                    dataset_guidance=state["dataset_guidance"],
                    head_motion_analysis=state["head_motion_analysis"],
                    brow_analysis=state["brow_analysis"],
                )
                brow_arbiter_prompts.append(prompt)
                if not brow_arbiter_system_prompt:
                    brow_arbiter_system_prompt = system_prompt
            brow_arbiter_outputs = self._generate_json_batch(
                brow_arbiter_prompts,
                max_tokens=320,
                system_prompt=brow_arbiter_system_prompt,
            )
            for state, brow_arbiter in zip(samm_states, brow_arbiter_outputs):
                state["brow_arbiter"] = _sanitize_brow_arbiter_output(brow_arbiter)

            brow_skeptic_prompts: list[str] = []
            brow_skeptic_system_prompt = ""
            for state in samm_states:
                prompt, system_prompt = self._build_brow_skeptic_prompt(
                    dataset=state["dataset"],
                    numeric_views=state["numeric_views"],
                    dataset_guidance=state["dataset_guidance"],
                    head_motion_analysis=state["head_motion_analysis"],
                    brow_analysis=state["brow_analysis"],
                    brow_arbiter=state["brow_arbiter"],
                )
                brow_skeptic_prompts.append(prompt)
                if not brow_skeptic_system_prompt:
                    brow_skeptic_system_prompt = system_prompt
            brow_skeptic_outputs = self._generate_json_batch(
                brow_skeptic_prompts,
                max_tokens=240,
                system_prompt=brow_skeptic_system_prompt,
            )
            for state, brow_skeptic in zip(samm_states, brow_skeptic_outputs):
                state["brow_skeptic"] = _sanitize_brow_skeptic_output(brow_skeptic)
                state["brow_arbiter"] = _apply_brow_skeptic_review(
                    state["brow_arbiter"],
                    state["brow_skeptic"],
                )

            eye_prompts: list[str] = []
            eye_system_prompt = ""
            for state in samm_states:
                prompt, system_prompt = self._build_eye_prompt(
                    dataset=state["dataset"],
                    numeric_views=state["numeric_views"],
                    dataset_guidance=state["dataset_guidance"],
                    head_motion_analysis=state["head_motion_analysis"],
                )
                eye_prompts.append(prompt)
                if not eye_system_prompt:
                    eye_system_prompt = system_prompt
            eye_outputs = self._generate_json_batch(
                eye_prompts,
                max_tokens=560,
                system_prompt=eye_system_prompt,
            )
            for state, eye in zip(samm_states, eye_outputs):
                state["eye_analysis"] = eye

            eye_arbiter_prompts: list[str] = []
            eye_arbiter_system_prompt = ""
            for state in samm_states:
                prompt, system_prompt = self._build_eye_arbiter_prompt(
                    dataset=state["dataset"],
                    numeric_views=state["numeric_views"],
                    dataset_guidance=state["dataset_guidance"],
                    head_motion_analysis=state["head_motion_analysis"],
                    eye_analysis=state["eye_analysis"],
                )
                eye_arbiter_prompts.append(prompt)
                if not eye_arbiter_system_prompt:
                    eye_arbiter_system_prompt = system_prompt
            eye_arbiter_outputs = self._generate_json_batch(
                eye_arbiter_prompts,
                max_tokens=320,
                system_prompt=eye_arbiter_system_prompt,
            )
            for state, eye_arbiter in zip(samm_states, eye_arbiter_outputs):
                state["eye_arbiter"] = _sanitize_eye_arbiter_output(eye_arbiter)

            mouth_prompts: list[str] = []
            mouth_system_prompt = ""
            for state in samm_states:
                prompt, system_prompt = self._build_mouth_prompt(
                    dataset=state["dataset"],
                    numeric_views=state["numeric_views"],
                    dataset_guidance=state["dataset_guidance"],
                )
                mouth_prompts.append(prompt)
                if not mouth_system_prompt:
                    mouth_system_prompt = system_prompt
            mouth_outputs = self._generate_json_batch(
                mouth_prompts,
                max_tokens=700,
                system_prompt=mouth_system_prompt,
            )
            for state, mouth in zip(samm_states, mouth_outputs):
                state["mouth_analysis"] = mouth

            mouth_arbiter_prompts: list[str] = []
            mouth_arbiter_system_prompt = ""
            for state in samm_states:
                prompt, system_prompt = self._build_mouth_arbiter_prompt(
                    dataset=state["dataset"],
                    numeric_views=state["numeric_views"],
                    dataset_guidance=state["dataset_guidance"],
                    mouth_analysis=state["mouth_analysis"],
                )
                mouth_arbiter_prompts.append(prompt)
                if not mouth_arbiter_system_prompt:
                    mouth_arbiter_system_prompt = system_prompt
            mouth_arbiter_outputs = self._generate_json_batch(
                mouth_arbiter_prompts,
                max_tokens=320,
                system_prompt=mouth_arbiter_system_prompt,
            )
            for state, mouth_arbiter in zip(samm_states, mouth_arbiter_outputs):
                state["mouth_arbiter"] = _sanitize_mouth_arbiter_output(mouth_arbiter)

        motion_prompts: list[str] = []
        motion_system_prompt = ""
        for state in states:
            prompt, system_prompt = self._build_motion_prompt(
                dataset=state["dataset"],
                numeric_views=state["numeric_views"],
                dataset_guidance=state["dataset_guidance"],
                brow_analysis=state["brow_analysis"],
                brow_arbiter=state["brow_arbiter"],
                eye_analysis=state["eye_analysis"],
                eye_arbiter=state["eye_arbiter"],
                mouth_analysis=state["mouth_analysis"],
                mouth_arbiter=state["mouth_arbiter"],
            )
            motion_prompts.append(prompt)
            if not motion_system_prompt:
                motion_system_prompt = system_prompt
        motion_outputs = self._generate_json_batch(
            motion_prompts,
            max_tokens=420,
            system_prompt=motion_system_prompt,
        )
        for state, motion in zip(states, motion_outputs):
            state["motion"] = motion

        emotion_prompts: list[str] = []
        emotion_system_prompt = ""
        for state in states:
            prompt, system_prompt = self._build_emotion_prompt(
                dataset=state["dataset"],
                numeric_views=state["numeric_views"],
                dataset_guidance=state["dataset_guidance"],
                brow_analysis=state["brow_analysis"],
                brow_arbiter=state["brow_arbiter"],
                eye_analysis=state["eye_analysis"],
                eye_arbiter=state["eye_arbiter"],
                mouth_analysis=state["mouth_analysis"],
                mouth_arbiter=state["mouth_arbiter"],
                motion=state["motion"],
            )
            emotion_prompts.append(prompt)
            if not emotion_system_prompt:
                emotion_system_prompt = system_prompt
        emotion_outputs = self._generate_json_batch(
            emotion_prompts,
            max_tokens=420,
            system_prompt=emotion_system_prompt,
        )
        for state, emotion_hypothesis in zip(states, emotion_outputs):
            state["emotion_hypothesis"] = emotion_hypothesis

        if not self.ablation.disable_critic:
            critic_prompts: list[str] = []
            critic_system_prompt = ""
            for state in states:
                prompt, system_prompt = self._build_critic_prompt(
                    dataset=state["dataset"],
                    numeric_views=state["numeric_views"],
                    dataset_guidance=state["dataset_guidance"],
                    brow_analysis=state["brow_analysis"],
                    brow_arbiter=state["brow_arbiter"],
                    eye_analysis=state["eye_analysis"],
                    eye_arbiter=state["eye_arbiter"],
                    mouth_analysis=state["mouth_analysis"],
                    mouth_arbiter=state["mouth_arbiter"],
                    motion=state["motion"],
                    emotion_hypothesis=state["emotion_hypothesis"],
                )
                critic_prompts.append(prompt)
                if not critic_system_prompt:
                    critic_system_prompt = system_prompt
            critic_outputs = self._generate_json_batch(
                critic_prompts,
                max_tokens=420,
                system_prompt=critic_system_prompt,
            )
            for state, critic in zip(states, critic_outputs):
                state["critic"] = critic

        analyses: dict[tuple[str, str], _ClipAnalysis] = {}
        for state in states:
            analyses[(state["clip_id"], state["image_path"])] = self._finalize_clip_analysis(
                dataset=state["dataset"],
                head_motion_analysis=state["head_motion_analysis"],
                brow_analysis=state["brow_analysis"],
                brow_arbiter=state["brow_arbiter"],
                brow_skeptic=state["brow_skeptic"],
                eye_analysis=state["eye_analysis"],
                eye_arbiter=state["eye_arbiter"],
                mouth_analysis=state["mouth_analysis"],
                mouth_arbiter=state["mouth_arbiter"],
                motion=state["motion"],
                emotion_hypothesis=state["emotion_hypothesis"],
                critic=state["critic"],
            )
        return analyses

    def _analyze_clip(self, dataset: str, clip_cache: dict[str, Any]) -> _ClipAnalysis:
        numeric_views = _clip_numeric_views(clip_cache, dataset, ablation=self.ablation)
        dataset_guidance = _dataset_reasoning_guidance(dataset)
        head_motion_analysis = {}
        brow_analysis = {}
        brow_arbiter = {}
        brow_skeptic = {}
        eye_analysis = {}
        eye_arbiter = {}
        mouth_analysis = {}
        mouth_arbiter = {}
        if dataset == "samm" and not self.ablation.disable_specialists:
            head_motion_prompt, head_motion_system = self._build_head_motion_prompt(
                dataset=dataset,
                numeric_views=numeric_views,
                dataset_guidance=dataset_guidance,
            )
            head_motion_analysis = self._generate_json(
                head_motion_prompt,
                max_tokens=260,
                system_prompt=head_motion_system,
            )
            head_motion_analysis = _sanitize_head_motion_agent_output(head_motion_analysis)
            brow_prompt, brow_system = self._build_brow_prompt(
                dataset=dataset,
                numeric_views=numeric_views,
                dataset_guidance=dataset_guidance,
                head_motion_analysis=head_motion_analysis,
            )
            brow_analysis = self._generate_json(brow_prompt, max_tokens=700, system_prompt=brow_system)
            brow_arbiter_prompt, brow_arbiter_system = self._build_brow_arbiter_prompt(
                dataset=dataset,
                numeric_views=numeric_views,
                dataset_guidance=dataset_guidance,
                head_motion_analysis=head_motion_analysis,
                brow_analysis=brow_analysis,
            )
            brow_arbiter = self._generate_json(
                brow_arbiter_prompt,
                max_tokens=320,
                system_prompt=brow_arbiter_system,
            )
            brow_arbiter = _sanitize_brow_arbiter_output(brow_arbiter)
            brow_skeptic_prompt, brow_skeptic_system = self._build_brow_skeptic_prompt(
                dataset=dataset,
                numeric_views=numeric_views,
                dataset_guidance=dataset_guidance,
                head_motion_analysis=head_motion_analysis,
                brow_analysis=brow_analysis,
                brow_arbiter=brow_arbiter,
            )
            brow_skeptic = self._generate_json(
                brow_skeptic_prompt,
                max_tokens=240,
                system_prompt=brow_skeptic_system,
            )
            brow_skeptic = _sanitize_brow_skeptic_output(brow_skeptic)
            brow_arbiter = _apply_brow_skeptic_review(brow_arbiter, brow_skeptic)
            eye_prompt, eye_system = self._build_eye_prompt(
                dataset=dataset,
                numeric_views=numeric_views,
                dataset_guidance=dataset_guidance,
                head_motion_analysis=head_motion_analysis,
            )
            eye_analysis = self._generate_json(
                eye_prompt,
                max_tokens=560,
                system_prompt=eye_system,
            )
            eye_arbiter_prompt, eye_arbiter_system = self._build_eye_arbiter_prompt(
                dataset=dataset,
                numeric_views=numeric_views,
                dataset_guidance=dataset_guidance,
                head_motion_analysis=head_motion_analysis,
                eye_analysis=eye_analysis,
            )
            eye_arbiter = self._generate_json(
                eye_arbiter_prompt,
                max_tokens=320,
                system_prompt=eye_arbiter_system,
            )
            eye_arbiter = _sanitize_eye_arbiter_output(eye_arbiter)
            mouth_prompt, mouth_system = self._build_mouth_prompt(
                dataset=dataset,
                numeric_views=numeric_views,
                dataset_guidance=dataset_guidance,
            )
            mouth_analysis = self._generate_json(
                mouth_prompt,
                max_tokens=700,
                system_prompt=mouth_system,
            )
            mouth_arbiter_prompt, mouth_arbiter_system = self._build_mouth_arbiter_prompt(
                dataset=dataset,
                numeric_views=numeric_views,
                dataset_guidance=dataset_guidance,
                mouth_analysis=mouth_analysis,
            )
            mouth_arbiter = self._generate_json(
                mouth_arbiter_prompt,
                max_tokens=320,
                system_prompt=mouth_arbiter_system,
            )
            mouth_arbiter = _sanitize_mouth_arbiter_output(mouth_arbiter)

        motion_prompt, motion_system = self._build_motion_prompt(
            dataset=dataset,
            numeric_views=numeric_views,
            dataset_guidance=dataset_guidance,
            brow_analysis=brow_analysis,
            brow_arbiter=brow_arbiter,
            eye_analysis=eye_analysis,
            eye_arbiter=eye_arbiter,
            mouth_analysis=mouth_analysis,
            mouth_arbiter=mouth_arbiter,
        )
        motion = self._generate_json(motion_prompt, max_tokens=420, system_prompt=motion_system)
        emotion_prompt, emotion_system = self._build_emotion_prompt(
            dataset=dataset,
            numeric_views=numeric_views,
            dataset_guidance=dataset_guidance,
            brow_analysis=brow_analysis,
            brow_arbiter=brow_arbiter,
            eye_analysis=eye_analysis,
            eye_arbiter=eye_arbiter,
            mouth_analysis=mouth_analysis,
            mouth_arbiter=mouth_arbiter,
            motion=motion,
        )
        emotion_hypothesis = self._generate_json(emotion_prompt, max_tokens=420, system_prompt=emotion_system)
        critic = {}
        if not self.ablation.disable_critic:
            critic_prompt, critic_system = self._build_critic_prompt(
                dataset=dataset,
                numeric_views=numeric_views,
                dataset_guidance=dataset_guidance,
                brow_analysis=brow_analysis,
                brow_arbiter=brow_arbiter,
                eye_analysis=eye_analysis,
                eye_arbiter=eye_arbiter,
                mouth_analysis=mouth_analysis,
                mouth_arbiter=mouth_arbiter,
                motion=motion,
                emotion_hypothesis=emotion_hypothesis,
            )
            critic = self._generate_json(critic_prompt, max_tokens=420, system_prompt=critic_system)
        return self._finalize_clip_analysis(
            dataset=dataset,
            head_motion_analysis=head_motion_analysis,
            brow_analysis=brow_analysis,
            brow_arbiter=brow_arbiter,
            brow_skeptic=brow_skeptic,
            eye_analysis=eye_analysis,
            eye_arbiter=eye_arbiter,
            mouth_analysis=mouth_analysis,
            mouth_arbiter=mouth_arbiter,
            motion=motion,
            emotion_hypothesis=emotion_hypothesis,
            critic=critic,
        )

    def answer_records_batch(self, question_records: list[dict], clip_cache: dict[tuple, dict]) -> dict[tuple, str]:
        grouped: dict[tuple[str, str, str], list[dict]] = {}
        for record in question_records:
            key = (record["video"], record["image_path"], record["dataset"])
            grouped.setdefault(key, []).append(record)

        analyses = self._analyze_clips_batch(
            [
                {
                    "clip_id": clip_id,
                    "image_path": image_path,
                    "dataset": dataset,
                    "cache": clip_cache.setdefault((clip_id, image_path), {}),
                }
                for (clip_id, image_path, dataset) in grouped
            ]
        )

        answers: dict[tuple, str] = {}
        for (clip_id, image_path, dataset), records in grouped.items():
            cache_key = (clip_id, image_path)
            cache = clip_cache.setdefault(cache_key, {})
            analysis = analyses[cache_key]
            cache["brow_reasoning"] = analysis.brow_reasoning
            cache["mouth_reasoning"] = analysis.mouth_reasoning
            cache["simple_local_analysis"] = {
                "emotion": analysis.emotion,
                "aus": analysis.aus,
            }
            cache["agent_trace"] = analysis.agent_trace

            coarse = FINE_TO_COARSE.get(analysis.emotion, "negative")
            au_list = ", ".join(analysis.aus)
            primary_au = analysis.aus[0] if analysis.aus else ""
            analysis_answer = (
                f"The action units present are: {au_list}. Therefore, the fine-grained "
                f"expression class is {analysis.emotion}, and the coarse expression class is {coarse}."
                if analysis.aus
                else f"No clear action units are present. Therefore, the fine-grained expression class is {analysis.emotion}, and the coarse expression class is {coarse}."
            )

            for record in records:
                question = record["question"]
                q_lower = question.lower()
                if "fine-grained expression" in q_lower:
                    pred = analysis.emotion
                elif "coarse expression" in q_lower:
                    pred = coarse
                elif question == "What is the action unit?":
                    pred = primary_au
                elif question == "What are the action units?":
                    pred = au_list
                elif "is the action unit" in q_lower and "shown on the face" in q_lower:
                    match = re.search(r"is the action unit (.+?) shown", q_lower)
                    asked = match.group(1).strip() if match else ""
                    pred = "yes" if asked and asked in set(analysis.aus) else "no"
                elif any(token in q_lower for token in ("analys", "describe", "comprehensive", "provide")):
                    pred = analysis_answer
                else:
                    pred = analysis_answer
                answers[(clip_id, question)] = pred
        return answers
