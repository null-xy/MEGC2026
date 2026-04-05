from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from .formatting import extract_emotion
except ImportError:
    from formatting import extract_emotion


COARSE_LABELS = ("positive", "negative", "surprise")
FINE_LABELS = (
    "happiness",
    "surprise",
    "fear",
    "disgust",
    "anger",
    "sadness",
    "repression",
    "contempt",
)


@dataclass(frozen=True)
class QARecord:
    dataset: str
    sample_id: str
    question: str
    pred: str
    gt: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MEGC2025 VQA outputs with UF1/UAR for coarse and "
            "fine-grained emotion classes plus BLEU/ROUGE-1."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help=(
            "Self-contained jsonl with fields like clip_id/question/pred/gt, "
            "e.g. outputs/batch_results_vqa_casme2.jsonl. Can be repeated."
        ),
    )
    parser.add_argument(
        "--pair",
        action="append",
        nargs=3,
        metavar=("DATASET", "PRED_JSONL", "REF_JSONL"),
        default=[],
        help=(
            "Prediction/reference pair. DATASET should be cas or samm. "
            "PRED/REF are jsonl files with question-answer rows."
        ),
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to save the computed metrics as JSON.",
    )
    return parser


def normalize_dataset_name(value: str) -> str:
    text = (value or "").strip().lower()
    if "samm" in text:
        return "SAMM"
    if "cas" in text:
        return "CAS"
    raise ValueError(f"Cannot infer dataset from: {value}")


def infer_dataset_from_path(path: Path) -> str:
    return normalize_dataset_name(path.name)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return rows


def record_sample_id(row: dict) -> str:
    for key in ("clip_id", "video", "image_id", "video_id"):
        value = row.get(key)
        if value:
            return str(value)
    raise ValueError(f"Cannot find sample id in row: {row}")


def record_question(row: dict) -> str:
    question = row.get("question")
    if question is None:
        raise ValueError(f"Cannot find question field in row: {row}")
    return str(question)


def load_self_contained(path: Path) -> list[QARecord]:
    rows = read_jsonl(path)
    dataset = None
    for row in rows:
        row_dataset = row.get("dataset")
        if row_dataset:
            dataset = normalize_dataset_name(str(row_dataset))
            break
    if dataset is None:
        dataset = infer_dataset_from_path(path)
    records: list[QARecord] = []
    for row in rows:
        pred = row.get("pred", row.get("answer", ""))
        gt = row.get("gt", row.get("reference", ""))
        records.append(
            QARecord(
                dataset=dataset,
                sample_id=record_sample_id(row),
                question=record_question(row),
                pred=str(pred or ""),
                gt=str(gt or ""),
            )
        )
    return records


def load_prediction_pair(dataset: str, pred_path: Path, ref_path: Path) -> list[QARecord]:
    dataset_name = normalize_dataset_name(dataset)
    pred_rows = read_jsonl(pred_path)
    ref_rows = read_jsonl(ref_path)
    pred_map: dict[tuple[str, str], dict] = {}
    for row in pred_rows:
        key = (record_sample_id(row), record_question(row))
        pred_map[key] = row

    records: list[QARecord] = []
    for row in ref_rows:
        key = (record_sample_id(row), record_question(row))
        pred_row = pred_map.get(key, {})
        pred = pred_row.get("pred", pred_row.get("answer", ""))
        gt = row.get("gt", row.get("answer", row.get("reference", "")))
        records.append(
            QARecord(
                dataset=dataset_name,
                sample_id=key[0],
                question=key[1],
                pred=str(pred or ""),
                gt=str(gt or ""),
            )
        )
    return records


def is_coarse_question(question: str) -> bool:
    return normalize_text(question) == "what is the coarse expression class?"


def is_fine_question(question: str) -> bool:
    return normalize_text(question) == "what is the fine-grained expression class?"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", normalize_text(text))


def extract_coarse_label(text: str) -> str | None:
    text_l = normalize_text(text)
    match = re.search(r"coarse expression class is (\w+)", text_l)
    if match:
        candidate = match.group(1)
        if candidate in COARSE_LABELS:
            return candidate
    for candidate in COARSE_LABELS:
        if re.search(rf"\b{candidate}\b", text_l):
            return candidate
    return None


def extract_fine_label(text: str) -> str | None:
    candidate = extract_emotion(text)
    return candidate if candidate in FINE_LABELS else None


def one_vs_rest_f1(gt_labels: list[str | None], pred_labels: list[str | None], label: str) -> float:
    tp = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == label and pred == label)
    fp = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt != label and pred == label)
    fn = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == label and pred != label)
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)


def one_vs_rest_recall(gt_labels: list[str | None], pred_labels: list[str | None], label: str) -> float:
    tp = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == label and pred == label)
    fn = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == label and pred != label)
    return tp / (tp + fn) if (tp + fn) else 0.0


def compute_uf1_uar(gt_labels: list[str | None], pred_labels: list[str | None]) -> tuple[float, float]:
    label_set = sorted({label for label in gt_labels if label})
    if not label_set:
        return 0.0, 0.0
    uf1 = sum(one_vs_rest_f1(gt_labels, pred_labels, label) for label in label_set) / len(label_set)
    uar = sum(one_vs_rest_recall(gt_labels, pred_labels, label) for label in label_set) / len(label_set)
    return uf1, uar


def ngrams(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1))


def compute_corpus_bleu(pred_texts: Iterable[str], ref_texts: Iterable[str], max_order: int = 4) -> float:
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    pred_length = 0
    ref_length = 0

    for pred_text, ref_text in zip(pred_texts, ref_texts):
        pred_tokens = tokenize(pred_text)
        ref_tokens = tokenize(ref_text)
        pred_length += len(pred_tokens)
        ref_length += len(ref_tokens)

        for order in range(1, max_order + 1):
            pred_ngrams = ngrams(pred_tokens, order)
            ref_ngrams = ngrams(ref_tokens, order)
            overlap = pred_ngrams & ref_ngrams
            matches_by_order[order - 1] += sum(overlap.values())
            possible_matches_by_order[order - 1] += max(len(pred_tokens) - order + 1, 0)

    precisions = []
    for matches, possible in zip(matches_by_order, possible_matches_by_order):
        if possible == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / possible)

    if pred_length == 0:
        return 0.0
    if min(precisions) == 0.0:
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    brevity_penalty = 1.0 if pred_length > ref_length else math.exp(1 - (ref_length / pred_length))
    return brevity_penalty * geo_mean


def rouge1_f1(pred_text: str, ref_text: str) -> float:
    pred_counts = Counter(tokenize(pred_text))
    ref_counts = Counter(tokenize(ref_text))
    if not pred_counts and not ref_counts:
        return 1.0
    overlap = sum((pred_counts & ref_counts).values())
    pred_total = sum(pred_counts.values())
    ref_total = sum(ref_counts.values())
    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    return 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)


def compute_average_rouge1(pred_texts: Iterable[str], ref_texts: Iterable[str]) -> float:
    scores = [rouge1_f1(pred, ref) for pred, ref in zip(pred_texts, ref_texts)]
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_partition(records: list[QARecord]) -> dict[str, float | int]:
    coarse_rows = [row for row in records if is_coarse_question(row.question)]
    fine_rows = [row for row in records if is_fine_question(row.question)]

    coarse_gt = [extract_coarse_label(row.gt) for row in coarse_rows]
    coarse_pred = [extract_coarse_label(row.pred) for row in coarse_rows]
    fine_gt = [extract_fine_label(row.gt) for row in fine_rows]
    fine_pred = [extract_fine_label(row.pred) for row in fine_rows]

    uf1_coarse, uar_coarse = compute_uf1_uar(coarse_gt, coarse_pred)
    uf1_fine, uar_fine = compute_uf1_uar(fine_gt, fine_pred)
    bleu = compute_corpus_bleu((row.pred for row in records), (row.gt for row in records))
    rouge = compute_average_rouge1((row.pred for row in records), (row.gt for row in records))

    return {
        "num_rows": len(records),
        "num_coarse_rows": len(coarse_rows),
        "num_fine_rows": len(fine_rows),
        "UF1 Coarse": uf1_coarse,
        "UAR Coarse": uar_coarse,
        "UF1 Fine-Grained": uf1_fine,
        "UAR Fine-Grained": uar_fine,
        "BLEU": bleu,
        "ROUGE": rouge,
    }


def render_table(results: dict[str, dict[str, float | int]]) -> str:
    columns = [
        ("Split", 8),
        ("Rows", 6),
        ("Coarse", 8),
        ("Fine", 8),
        ("UF1-C", 10),
        ("UAR-C", 10),
        ("UF1-F", 10),
        ("UAR-F", 10),
        ("BLEU", 10),
        ("ROUGE", 10),
    ]
    header = " ".join(name.ljust(width) for name, width in columns)
    lines = [header, "-" * len(header)]
    order = ["Overall", "SAMM", "CAS"]
    for split in order:
        values = results.get(split, {})
        lines.append(
            " ".join(
                [
                    split.ljust(8),
                    str(values.get("num_rows", 0)).ljust(6),
                    str(values.get("num_coarse_rows", 0)).ljust(8),
                    str(values.get("num_fine_rows", 0)).ljust(8),
                    f"{values.get('UF1 Coarse', 0.0):.4f}".ljust(10),
                    f"{values.get('UAR Coarse', 0.0):.4f}".ljust(10),
                    f"{values.get('UF1 Fine-Grained', 0.0):.4f}".ljust(10),
                    f"{values.get('UAR Fine-Grained', 0.0):.4f}".ljust(10),
                    f"{values.get('BLEU', 0.0):.4f}".ljust(10),
                    f"{values.get('ROUGE', 0.0):.4f}".ljust(10),
                ]
            )
        )
    return "\n".join(lines)


def build_flat_metric_dict(results: dict[str, dict[str, float | int]]) -> dict[str, float]:
    flat: dict[str, float] = {}
    metric_names = (
        "UF1 Coarse",
        "UAR Coarse",
        "UF1 Fine-Grained",
        "UAR Fine-Grained",
        "BLEU",
        "ROUGE",
    )
    for split in ("Overall", "SAMM", "CAS"):
        values = results.get(split, {})
        for metric_name in metric_names:
            flat[f"{metric_name} ({split})"] = float(values.get(metric_name, 0.0))
    return flat


def main() -> None:
    args = build_parser().parse_args()
    records: list[QARecord] = []

    for input_path in args.input:
        records.extend(load_self_contained(Path(input_path)))
    for dataset, pred_path, ref_path in args.pair:
        records.extend(load_prediction_pair(dataset, Path(pred_path), Path(ref_path)))

    if not records:
        raise SystemExit("No evaluation data provided. Use --input and/or --pair.")

    grouped = {
        "Overall": records,
        "SAMM": [row for row in records if row.dataset == "SAMM"],
        "CAS": [row for row in records if row.dataset == "CAS"],
    }
    results = {split: evaluate_partition(split_rows) for split, split_rows in grouped.items()}

    print(render_table(results))
    print()
    for key, value in build_flat_metric_dict(results).items():
        print(f"{key}\t{value:.6f}")

    if args.json_out:
        payload = {
            "summary": results,
            "flat_metrics": build_flat_metric_dict(results),
        }
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
