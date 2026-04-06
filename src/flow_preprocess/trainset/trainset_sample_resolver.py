"""Resolve trainset samples from JSONL and annotation files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from flow_preprocess.utils import is_image_file, natural_sort_key


@dataclass
class ResolvedTrainSample:
    """Resolved frame paths and metadata for one train sample."""

    sample_id: str
    dataset: str
    subject: str
    video_name: str
    onset: int
    apex: int
    offset: int
    onset_path: Path
    apex_path: Path
    offset_path: Path


def load_unique_train_samples(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Load unique sample descriptors from the JSONL, deduplicated per sample."""
    jsonl_path = Path(jsonl_path)
    unique_items: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    with open(jsonl_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            item = json.loads(line)
            dataset = str(item.get("dataset", "")).strip().lower()
            image_id = str(item.get("image_id") or item.get("filename") or "").strip()
            subject = str(item.get("subject") or "").strip()
            key = (dataset, image_id, subject)
            if not dataset or not image_id or key in seen:
                continue
            seen.add(key)
            unique_items.append(item)
    return sorted(
        unique_items,
        key=lambda item: natural_sort_key(
            f"{item.get('dataset','')}_{item.get('subject','')}_{item.get('filename','')}"
        ),
    )


def _normalize_subject_token(dataset: str, subject: str) -> str:
    if dataset == "samm":
        return f"{int(subject):03d}"
    if dataset == "casme2":
        if subject.lower().startswith("sub"):
            return f"sub{int(subject[3:]):02d}"
        return f"sub{int(subject):02d}"
    return subject


def _normalize_column_name(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _find_matching_column(columns: list[Any], candidates: list[str]) -> Any:
    normalized_to_original = {
        _normalize_column_name(column): column
        for column in columns
    }
    for candidate in candidates:
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]
    raise KeyError(f"Required SAMM annotation column not found. Expected one of: {candidates}")


def _load_samm_annotation_dataframe(annotation_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_excel(annotation_path)
    normalized_columns = {_normalize_column_name(column) for column in dataframe.columns}
    old_schema = {"subject", "filename", "onset", "apex frame", "offset"}
    new_schema = {"subject", "filename", "onset frame", "apex frame", "offset frame"}
    if old_schema.issubset(normalized_columns) or new_schema.issubset(normalized_columns):
        return dataframe

    workbook = pd.ExcelFile(annotation_path)
    sheet_name = "MICRO_ONLY" if "MICRO_ONLY" in workbook.sheet_names else workbook.sheet_names[0]
    raw_dataframe = pd.read_excel(annotation_path, sheet_name=sheet_name, header=None)
    header_row_index = None
    for row_index in range(min(30, len(raw_dataframe))):
        normalized_row = {
            _normalize_column_name(value)
            for value in raw_dataframe.iloc[row_index].tolist()
            if not pd.isna(value)
        }
        if {"subject", "filename", "apex frame"}.issubset(normalized_row) and (
            {"onset", "offset"}.issubset(normalized_row) or {"onset frame", "offset frame"}.issubset(normalized_row)
        ):
            header_row_index = row_index
            break
    if header_row_index is None:
        raise KeyError(f"Unable to locate a SAMM annotation header row in: {annotation_path}")
    return pd.read_excel(annotation_path, sheet_name=sheet_name, header=header_row_index)


def _index_samm_annotations(annotation_path: str | Path) -> dict[str, dict[str, Any]]:
    dataframe = _load_samm_annotation_dataframe(annotation_path)
    subject_column = _find_matching_column(list(dataframe.columns), ["subject"])
    filename_column = _find_matching_column(list(dataframe.columns), ["filename"])
    onset_column = _find_matching_column(list(dataframe.columns), ["onset", "onset frame"])
    apex_column = _find_matching_column(list(dataframe.columns), ["apex frame", "apex"])
    offset_column = _find_matching_column(list(dataframe.columns), ["offset", "offset frame"])
    records: dict[str, dict[str, Any]] = {}
    for _, row in dataframe.iterrows():
        filename = str(row[filename_column]).strip()
        if not filename or filename.lower() == "nan":
            continue
        onset = _safe_int(row[onset_column])
        apex = _safe_int(row[apex_column])
        offset = _safe_int(row[offset_column])
        if onset is None or apex is None or offset is None:
            continue
        records[filename] = {
            "subject": f"{int(row[subject_column]):03d}",
            "video_name": filename,
            "onset": onset,
            "apex": apex,
            "offset": offset,
        }
    return records


def _index_casme2_annotations(annotation_path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    dataframe = pd.read_excel(annotation_path)
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in dataframe.iterrows():
        if pd.isna(row["Subject"]) or pd.isna(row["Filename"]):
            continue
        subject = f"sub{int(row['Subject']):02d}"
        filename = str(row["Filename"]).strip()
        onset = _safe_int(row["OnsetFrame"])
        apex = _safe_int(row["ApexFrame"])
        offset = _safe_int(row["OffsetFrame"])
        if onset is None or apex is None or offset is None:
            continue
        records[(subject, filename)] = {
            "subject": subject,
            "video_name": filename,
            "onset": onset,
            "apex": apex,
            "offset": offset,
        }
    return records


def _safe_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text == "/" or text.lower() == "nan":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _find_existing_frame_by_patterns(directory: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        for match in sorted(directory.glob(pattern), key=natural_sort_key):
            if match.is_file() and is_image_file(match):
                return match
    return None


def _resolve_samm_frame(sample_dir: Path, subject_token: str, frame_index: int) -> Path:
    patterns = [
        f"{subject_token}_{int(frame_index):05d}.*",
        f"{subject_token}_{int(frame_index):06d}.*",
        f"*{int(frame_index):05d}.*",
        f"*{int(frame_index):06d}.*",
    ]
    match = _find_existing_frame_by_patterns(sample_dir, patterns)
    if match is None:
        raise FileNotFoundError(f"SAMM frame {frame_index} not found in {sample_dir}")
    return match


def _resolve_casme2_frame(sample_dir: Path, frame_index: int) -> Path:
    patterns = [
        f"reg_img{int(frame_index)}.*",
        f"img{int(frame_index)}.*",
        f"*{int(frame_index)}.*",
    ]
    match = _find_existing_frame_by_patterns(sample_dir, patterns)
    if match is None:
        raise FileNotFoundError(f"CASME II frame {frame_index} not found in {sample_dir}")
    return match


def resolve_train_sample(
    sample: dict[str, Any],
    *,
    samm_root: str | Path,
    casme2_root: str | Path,
    samm_annotations: dict[str, dict[str, Any]],
    casme2_annotations: dict[tuple[str, str], dict[str, Any]],
) -> ResolvedTrainSample:
    """Resolve one JSONL sample into onset/apex/offset frame paths."""
    dataset = str(sample.get("dataset", "")).strip().lower()
    filename = str(sample.get("filename") or sample.get("image_id") or "").strip()
    subject = str(sample.get("subject") or "").strip()
    sample_id = str(sample.get("id") or f"{dataset}_{filename}").rsplit("_", 1)[0]

    if dataset == "samm":
        subject_token = _normalize_subject_token(dataset, subject)
        annotation = samm_annotations.get(filename)
        if annotation is None:
            raise KeyError(f"SAMM annotation row not found for sample: {filename}")
        sample_dir = Path(samm_root) / subject_token / annotation["video_name"]
        if not sample_dir.exists():
            raise FileNotFoundError(f"SAMM sample directory not found: {sample_dir}")
        onset = int(annotation["onset"])
        apex = int(annotation["apex"])
        offset = int(annotation["offset"])
        return ResolvedTrainSample(
            sample_id=sample_id,
            dataset=dataset,
            subject=subject_token,
            video_name=annotation["video_name"],
            onset=onset,
            apex=apex,
            offset=offset,
            onset_path=_resolve_samm_frame(sample_dir, subject_token, onset),
            apex_path=_resolve_samm_frame(sample_dir, subject_token, apex),
            offset_path=_resolve_samm_frame(sample_dir, subject_token, offset),
        )

    if dataset == "casme2":
        subject_token = _normalize_subject_token(dataset, subject)
        annotation = casme2_annotations.get((subject_token, filename))
        if annotation is None:
            raise KeyError(f"CASME II annotation row not found for sample: {subject_token}/{filename}")
        sample_dir = Path(casme2_root) / subject_token / annotation["video_name"]
        if not sample_dir.exists():
            raise FileNotFoundError(f"CASME II sample directory not found: {sample_dir}")
        onset = int(annotation["onset"])
        apex = int(annotation["apex"])
        offset = int(annotation["offset"])
        return ResolvedTrainSample(
            sample_id=sample_id,
            dataset=dataset,
            subject=subject_token,
            video_name=annotation["video_name"],
            onset=onset,
            apex=apex,
            offset=offset,
            onset_path=_resolve_casme2_frame(sample_dir, onset),
            apex_path=_resolve_casme2_frame(sample_dir, apex),
            offset_path=_resolve_casme2_frame(sample_dir, offset),
        )

    raise ValueError(f"Unsupported dataset for this pipeline: {dataset}")


def load_annotation_indexes(
    samm_anno_path: str | Path,
    casme2_anno_path: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Load and index both supported annotation files."""
    return _index_samm_annotations(samm_anno_path), _index_casme2_annotations(casme2_anno_path)
