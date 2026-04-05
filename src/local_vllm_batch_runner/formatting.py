from __future__ import annotations

import re
from typing import Iterable, List


AU_NAME_TO_NUM = {
    "inner brow raiser": 1,
    "outer brow raiser": 2,
    "brow lowerer": 4,
    "upper lid raiser": 5,
    "lid raiser": 5,
    "cheek raiser": 6,
    "lid tightener": 7,
    "nose wrinkler": 9,
    "upper lip raiser": 10,
    "lip corner puller": 12,
    "dimpler": 14,
    "lip corner depressor": 15,
    "lower lip depressor": 16,
    "chin raiser": 17,
    "lip stretcher": 20,
    "lip tightener": 23,
    "lips part": 25,
    "jaw drop": 26,
    "jaw clencher": 31,
    "neck tightener": 21,
    "nostril dilater": 38,
    "nostril compress": 39,
    "eyes left": 61,
    "eye closure": 46,
}

AU_NUM_TO_NAME = {
    1: "inner brow raiser",
    2: "outer brow raiser",
    4: "brow lowerer",
    5: "upper lid raiser",
    6: "cheek raiser",
    7: "lid tightener",
    9: "nose wrinkler",
    10: "upper lip raiser",
    12: "lip corner puller",
    14: "dimpler",
    15: "lip corner depressor",
    16: "lower lip depressor",
    17: "chin raiser",
    20: "lip stretcher",
    21: "neck tightener",
    23: "lip tightener",
    24: "lip tightener",
    25: "lips part",
    26: "jaw drop",
    31: "jaw clencher",
    38: "nostril dilater",
    39: "nostril compress",
    61: "eyes left",
    46: "eye closure",
}

AU_ALIASES = {
    "lid raiser": "upper lid raiser",
    "eye closer": "eye closure",
    "eye closure": "eye closure",
    "lip pressor": "lip tightener",
    "lip presser": "lip tightener",
    "lip corner depress": "lip corner depressor",
    "nostril dilator": "nostril dilater",
}

FINE_TO_COARSE = {
    "happiness": "positive",
    "surprise": "surprise",
    "fear": "negative",
    "disgust": "negative",
    "anger": "negative",
    "sadness": "negative",
    "repression": "negative",
    "contempt": "negative",
    "unknown": "negative",
    "other": "negative",
}

DATASET_ALLOWED_EMOTIONS = {
    "casme2": {"happiness", "surprise", "disgust", "repression"},
    "samm": {"happiness", "surprise", "disgust", "fear", "anger", "sadness", "contempt"},
    "casme3": {"happiness", "surprise", "disgust", "fear", "anger", "sadness"},
}

_ORDERED_AU_NAMES = sorted(AU_NAME_TO_NUM, key=len, reverse=True)


def extract_emotion(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"fine-grained expression class is (\w+)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    match = re.search(r"\b(happiness|surprise|fear|disgust|anger|sadness|repression|contempt|other|unknown)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return text.lower().split()[0] if text else "unknown"


def normalize_au_name(name: str) -> str:
    normalized = re.sub(r"\s+", " ", (name or "").strip().lower())
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return AU_ALIASES.get(normalized, normalized)


def extract_au_names(text: str) -> List[str]:
    text_l = (text or "").lower()
    found: List[str] = []
    for au_name in _ORDERED_AU_NAMES:
        if re.search(rf"\b{re.escape(au_name)}\b", text_l):
            found.append(normalize_au_name(au_name))
    if found:
        return dedupe_keep_order(found)

    names: List[str] = []
    for number in re.findall(r"\bau\s*(\d+)\b", text_l, re.IGNORECASE):
        name = AU_NUM_TO_NAME.get(int(number))
        if name:
            names.append(name)
    return dedupe_keep_order(names)


def au_names_to_pred_string(names: Iterable[str]) -> str:
    items = []
    for name in dedupe_keep_order(normalize_au_name(n) for n in names if n):
        number = AU_NAME_TO_NUM.get(name)
        if number is None:
            continue
        items.append(f"AU{number} {name}")
    return ", ".join(items) if items else ""


def gt_aus_to_set(gt_aus_list: Iterable[str]) -> set[int]:
    gt = set()
    for name in gt_aus_list or []:
        number = AU_NAME_TO_NUM.get(normalize_au_name(name))
        if number is not None:
            gt.add(number)
    return gt


def pred_aus_to_set(pred_aus_str: str) -> set[int]:
    canonical = set()
    for match in re.findall(r"\bau\s*(\d+)\b", pred_aus_str or "", re.IGNORECASE):
        number = int(match)
        if number == 24:
            number = 23
        canonical.add(number)
    return canonical


def aus_mismatch(pred_aus_str: str, gt_aus_list: Iterable[str]) -> bool:
    return pred_aus_to_set(pred_aus_str) != gt_aus_to_set(gt_aus_list)


def aus_f1(pred_aus_str: str, gt_aus_list: Iterable[str]) -> tuple[float, float, float]:
    pred = pred_aus_to_set(pred_aus_str)
    gt = gt_aus_to_set(gt_aus_list)
    if not pred and not gt:
        return 1.0, 1.0, 1.0
    tp = len(pred & gt)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def normalize_emotion_for_dataset(emotion: str, dataset: str) -> str:
    normalized = (emotion or "unknown").strip().lower()
    allowed = DATASET_ALLOWED_EMOTIONS.get(dataset, set())
    if normalized in allowed:
        return normalized
    if normalized in {"", "other", "unknown"}:
        return "unknown"
    return "unknown"
