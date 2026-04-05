from __future__ import annotations

import json
from pathlib import Path

try:
    from .formatting import extract_emotion
except ImportError:
    from formatting import extract_emotion


def casme2_flow_path(mame_dir: Path, clip_id: str) -> Path:
    return mame_dir / "trainset_flow" / "casme2" / f"casme2_{clip_id}" / "tv_l1_direct" / "flow_npy" / "onset_to_apex.npy"


def samm_flow_path(mame_dir: Path, clip_id: str) -> Path:
    return mame_dir / "trainset_flow" / "samm" / f"samm_{clip_id}" / "tv_l1_direct" / "flow_npy" / "onset_to_apex.npy"


def infer_subject_id(dataset: str, clip_id: str, explicit_subject: str = "") -> str:
    subject = str(explicit_subject or "").strip()
    if subject:
        return subject
    clip_id = str(clip_id or "").strip()
    if not clip_id:
        return ""
    if dataset in {"casme2", "samm", "casme3"}:
        return clip_id.split("_", 1)[0].strip()
    return clip_id.split("_", 1)[0].strip()


def load_training_clips(megc_jsonl: Path, dataset: str, mame_dir: Path) -> dict[str, dict]:
    clips: dict[str, dict] = {}
    with open(megc_jsonl, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("dataset") != dataset:
                continue
            clip_id = record["image_id"]
            question = record.get("question", "")
            answer = record.get("answer", "")
            if clip_id not in clips:
                path = casme2_flow_path(mame_dir, clip_id) if dataset == "casme2" else samm_flow_path(mame_dir, clip_id)
                clips[clip_id] = {
                    "clip_id": clip_id,
                    "dataset": dataset,
                    "subject": infer_subject_id(dataset, clip_id, record.get("subject", "")),
                    "filename": str(record.get("filename", "") or ""),
                    "path": path,
                    "gt_fine": None,
                    "gt_coarse": None,
                    "gt_aus": [],
                    "gt_detail": "",
                    "vqa_questions": [],
                }
            clip = clips[clip_id]
            if "fine-grained expression" in question:
                clip["gt_fine"] = extract_emotion(answer)
            elif "coarse expression" in question:
                clip["gt_coarse"] = answer.strip()
            elif question in ("What is the action unit?", "What are the action units?"):
                clip["gt_aus"] = [part.strip() for part in answer.split(",") if part.strip()]
            elif "analyse the micro-expression" in question:
                clip["gt_detail"] = answer.strip()
            clip["vqa_questions"].append({"q": question, "a": answer})

    return {
        clip_id: clip
        for clip_id, clip in clips.items()
        if clip.get("gt_fine") and Path(clip["path"]).exists()
    }


def load_test_clips(mame_dir: Path) -> dict[str, dict]:
    crop_root = mame_dir / "data" / "ME_VQA_MEGC_2025_Test_Crop_0329"
    flow_root = mame_dir / "trainset_flow" / "testset"
    clips: dict[str, dict] = {}
    for clip_dir in sorted(crop_root.iterdir()):
        if not clip_dir.is_dir() or clip_dir.name == "cache":
            continue
        clip_id = clip_dir.name
        path = flow_root / clip_id / "tv_l1_direct" / "flow_npy" / "onset_to_apex.npy"
        if not path.exists():
            continue
        is_samm = clip_id.startswith("SAMM")
        dataset = "samm" if is_samm else "casme2"
        template_dataset = "samm" if is_samm else "casme3"
        submission_split = "samm" if is_samm else "cas"
        clips[clip_id] = {
            "clip_id": clip_id,
            "dataset": dataset,
            "template_dataset": template_dataset,
            "submission_split": submission_split,
            "subject": infer_subject_id(dataset, clip_id),
            "path": path,
            "gt_fine": "unknown",
            "gt_coarse": "unknown",
            "gt_aus": [],
            "gt_detail": "",
            "vqa_questions": [],
        }
    return clips


def load_question_templates(path: Path) -> dict[str, list[dict]]:
    templates: dict[str, list[dict]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            templates.setdefault(record["video"], []).append(record)
    return templates
