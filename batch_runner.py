from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from .calibration import motion_feature_calibration_context
    from .config import AblationConfig, DEFAULT_SAMPLE_ANSWER_DIR, add_common_args
    from .datasets import load_question_templates, load_test_clips, load_training_clips
    from .inference import (
        BackendConfig,
        LocalMASBackend,
        build_training_output_rows,
        load_done_clip_ids,
        summarize_clip,
        write_jsonl,
    )
    from .numeric_features import ensure_numeric_feature_dependencies
except ImportError:
    from calibration import motion_feature_calibration_context
    from config import AblationConfig, DEFAULT_SAMPLE_ANSWER_DIR, add_common_args
    from datasets import load_question_templates, load_test_clips, load_training_clips
    from inference import (
        BackendConfig,
        LocalMASBackend,
        build_training_output_rows,
        load_done_clip_ids,
        summarize_clip,
        write_jsonl,
    )
    from numeric_features import ensure_numeric_feature_dependencies


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(
        argparse.ArgumentParser(description="Unified local vLLM batch runner")
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["casme2", "samm", "testset"],
        help="Which dataset pipeline to run.",
    )
    parser.add_argument("--sample-answer-dir", default=str(DEFAULT_SAMPLE_ANSWER_DIR))
    parser.add_argument(
        "--loso",
        action="store_true",
        help="Run leave-one-subject-out evaluation for a training dataset.",
    )
    parser.add_argument(
        "--loso-subject",
        default="",
        help="Optional held-out subject id. If set, only that LOSO fold is executed.",
    )
    parser.add_argument(
        "--loso-output-dir",
        default="",
        help="Optional directory for LOSO outputs. Defaults to outputs/loso/<dataset>/.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    mame_dir = Path(args.mame_dir)
    model_asset = ensure_numeric_feature_dependencies()
    print(f"[numeric] MediaPipe ready with face landmarker: {model_asset}")
    backend = _build_backend(args)

    if args.dataset == "testset":
        if args.loso:
            parser.error("--loso is only supported for casme2 and samm.")
        run_testset(args, mame_dir, backend)
        return

    if args.loso:
        run_training_dataset_loso(args, mame_dir, backend)
        return

    run_training_dataset(args, mame_dir, backend)


def resolve_output_dir(mame_dir: Path) -> Path:
    return mame_dir.parent / "outputs"


def _build_backend(args: argparse.Namespace) -> LocalMASBackend:
    model_dir = args.vllm_directory
    reasoning_dir = args.llm_directory or args.vllm_directory
    ablation = AblationConfig.from_name(getattr(args, "ablation", "full"))
    return LocalMASBackend(
        BackendConfig(
            model=model_dir,
            reasoning_model=reasoning_dir,
            gpu_mem=args.gpu_mem,
            reasoning_gpu_mem=args.reasoning_gpu_mem,
            reasoning_device=args.reasoning_device,
            reasoning_tensor_parallel_size=args.reasoning_tensor_parallel_size,
            vision_device=args.vision_device,
            memory=args.memory,
            scope=args.scope,
            no_vision=args.no_vision,
            debug=args.debug,
            ablation=ablation,
        )
    )


def run_training_dataset(args: argparse.Namespace, mame_dir: Path, backend: LocalMASBackend) -> None:
    dataset = args.dataset
    output_dir = resolve_output_dir(mame_dir)
    results_file = output_dir / f"batch_results_{dataset}.jsonl"
    wrong_file = output_dir / f"batch_wrong_{dataset}.jsonl"
    vqa_file = output_dir / f"batch_results_vqa_{dataset}.jsonl"

    clips = load_training_clips(Path(args.megc_jsonl), dataset, mame_dir)
    done = load_done_clip_ids(results_file)
    pending = [clip for clip in clips.values() if clip["clip_id"] not in done]
    print(f"[{dataset}] clips={len(clips)} pending={len(pending)}")
    if not pending:
        return

    _process_training_batches(
        dataset=dataset,
        clips=pending,
        batch_size=args.batch_size,
        backend=backend,
        results_file=results_file,
        wrong_file=wrong_file,
        vqa_file=vqa_file,
    )


def run_training_dataset_loso(args: argparse.Namespace, mame_dir: Path, backend: LocalMASBackend) -> None:
    dataset = args.dataset
    output_root = _resolve_loso_output_dir(args, resolve_output_dir(mame_dir), dataset)
    folds_root = output_root / "folds"
    clips = load_training_clips(Path(args.megc_jsonl), dataset, mame_dir)
    grouped = _group_training_clips_by_subject(clips.values())
    requested_subject = str(args.loso_subject or "").strip()
    if requested_subject:
        grouped = {
            subject: subject_clips
            for subject, subject_clips in grouped.items()
            if subject == requested_subject
        }
        if not grouped:
            raise ValueError(f"Subject '{requested_subject}' was not found in {dataset}.")

    total_subjects = len(grouped)
    total_clips = sum(len(subject_clips) for subject_clips in grouped.values())
    print(f"[{dataset}][loso] subjects={total_subjects} clips={total_clips}")
    if not grouped:
        return

    for fold_index, (subject, subject_clips) in enumerate(grouped.items(), start=1):
        fold_dir = folds_root / subject
        results_file = fold_dir / "batch_results.jsonl"
        wrong_file = fold_dir / "batch_wrong.jsonl"
        vqa_file = fold_dir / "batch_results_vqa.jsonl"
        done = load_done_clip_ids(results_file)
        pending = [clip for clip in subject_clips if clip["clip_id"] not in done]
        print(
            f"[{dataset}][loso] fold {fold_index}/{total_subjects} "
            f"holdout={subject} clips={len(subject_clips)} pending={len(pending)}"
        )
        if pending:
            with motion_feature_calibration_context(exclude_subjects={dataset: [subject]}):
                _process_training_batches(
                    dataset=dataset,
                    clips=pending,
                    batch_size=args.batch_size,
                    backend=backend,
                    results_file=results_file,
                    wrong_file=wrong_file,
                    vqa_file=vqa_file,
                    loso_subject=subject,
                )

    _rebuild_loso_aggregate_outputs(output_root, dataset)


def _process_training_batches(
    *,
    dataset: str,
    clips: list[dict],
    batch_size: int,
    backend: LocalMASBackend,
    results_file: Path,
    wrong_file: Path,
    vqa_file: Path,
    loso_subject: str = "",
) -> None:
    for start in range(0, len(clips), batch_size):
        batch = clips[start : start + batch_size]
        batch_number = start // batch_size + 1
        if loso_subject:
            print(f"[{dataset}][loso:{loso_subject}] batch {batch_number}: {len(batch)} clips")
        else:
            print(f"[{dataset}] batch {batch_number}: {len(batch)} clips")
        clip_cache, answers = backend.analyze_batch(batch)
        result_rows = []
        wrong_rows = []
        vqa_rows = []
        for clip in batch:
            summary = summarize_clip(clip, clip_cache, answers, ablation=backend.ablation)
            result_row, wrong_row = build_training_output_rows(summary, clip)
            _annotate_training_row(
                result_row,
                clip=clip,
                dataset=dataset,
                batch_number=batch_number,
                loso_subject=loso_subject,
            )
            result_rows.append(result_row)
            if wrong_row is not None:
                _annotate_training_row(
                    wrong_row,
                    clip=clip,
                    dataset=dataset,
                    batch_number=batch_number,
                    loso_subject=loso_subject,
                )
                wrong_rows.append(wrong_row)
            for row in summary["vqa_rows"]:
                row["dataset"] = dataset
                if clip.get("subject"):
                    row["subject"] = clip["subject"]
                row["correct"] = backend.exact_match(row["question"], row["gt"], row["pred"])
                if loso_subject:
                    row["loso"] = True
                    row["loso_fold"] = loso_subject
                vqa_rows.append(row)
        write_jsonl(results_file, result_rows)
        if wrong_rows:
            write_jsonl(wrong_file, wrong_rows)
        if vqa_rows:
            write_jsonl(vqa_file, vqa_rows)


def _annotate_training_row(
    row: dict,
    *,
    clip: dict,
    dataset: str,
    batch_number: int,
    loso_subject: str = "",
) -> None:
    row["dataset"] = dataset
    row["batch"] = batch_number
    if clip.get("subject"):
        row["subject"] = clip["subject"]
    if dataset == "samm":
        row["recon_changed"] = False
        row["anger_gate"] = False
    if loso_subject:
        row["loso"] = True
        row["loso_fold"] = loso_subject


def _group_training_clips_by_subject(clips: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for clip in clips:
        subject = str(clip.get("subject", "") or "").strip()
        if not subject:
            raise ValueError(f"Missing subject for clip {clip.get('clip_id', '')}. LOSO requires subject ids.")
        grouped[subject].append(clip)
    return {
        subject: sorted(subject_clips, key=lambda clip: str(clip.get("clip_id", "")))
        for subject, subject_clips in sorted(grouped.items())
    }


def _resolve_loso_output_dir(args: argparse.Namespace, output_dir: Path, dataset: str) -> Path:
    custom_dir = str(args.loso_output_dir or "").strip()
    if custom_dir:
        return Path(custom_dir)
    return output_dir / "loso" / dataset


def _rebuild_loso_aggregate_outputs(output_root: Path, dataset: str) -> None:
    fold_dirs = sorted(
        path
        for path in (output_root / "folds").iterdir()
        if path.is_dir()
    ) if (output_root / "folds").exists() else []
    _rewrite_jsonl_from_folds(
        output_root / f"batch_results_{dataset}_loso.jsonl",
        [fold_dir / "batch_results.jsonl" for fold_dir in fold_dirs],
    )
    _rewrite_jsonl_from_folds(
        output_root / f"batch_wrong_{dataset}_loso.jsonl",
        [fold_dir / "batch_wrong.jsonl" for fold_dir in fold_dirs],
    )
    _rewrite_jsonl_from_folds(
        output_root / f"batch_results_vqa_{dataset}_loso.jsonl",
        [fold_dir / "batch_results_vqa.jsonl" for fold_dir in fold_dirs],
    )


def _rewrite_jsonl_from_folds(output_file: Path, input_files: list[Path]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        for input_file in input_files:
            if not input_file.exists():
                continue
            with open(input_file, "r", encoding="utf-8") as input_handle:
                for line in input_handle:
                    line = line.rstrip("\n")
                    if line:
                        handle.write(line + "\n")


def run_testset(args: argparse.Namespace, mame_dir: Path, backend: LocalMASBackend) -> None:
    sample_answer_dir = Path(args.sample_answer_dir)
    output_dir = resolve_output_dir(mame_dir)
    cas_results = output_dir / "batch_results_testset_cas.jsonl"
    samm_results = output_dir / "batch_results_testset_samm.jsonl"
    cas_pred = output_dir / "me_vqa_casme3_v2_test_pred.jsonl"
    samm_pred = output_dir / "me_vqa_samm_v2_test_pred.jsonl"

    clips = load_test_clips(mame_dir)
    cas_templates = load_question_templates(sample_answer_dir / "me_vqa_casme3_v2_test_to_answer.jsonl")
    samm_templates = load_question_templates(sample_answer_dir / "me_vqa_samm_v2_test_to_answer.jsonl")

    for clip_id, clip in clips.items():
        clip["vqa_questions"] = [
            {"q": row["question"], "a": row.get("answer", "")}
            for row in (
                cas_templates.get(clip_id, [])
                if clip.get("template_dataset") == "casme3"
                else samm_templates.get(clip_id, [])
            )
        ]

    done = load_done_clip_ids(cas_results) | load_done_clip_ids(samm_results)
    pending = [clip for clip in clips.values() if clip["clip_id"] not in done]
    print(f"[testset] clips={len(clips)} pending={len(pending)}")

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        batch_number = start // args.batch_size + 1
        print(f"[testset] batch {batch_number}: {len(batch)} clips")
        clip_cache, answers = backend.analyze_batch(batch)
        cas_rows = []
        samm_rows = []
        new_cas_answers = []
        new_samm_answers = []
        for clip in batch:
            summary = summarize_clip(clip, clip_cache, answers, ablation=backend.ablation)
            result_row = {
                "clip_id": clip["clip_id"],
                "dataset": clip["dataset"],
                "template_dataset": clip.get("template_dataset", clip["dataset"]),
                "pred": summary["pred"],
                "pred_aus": summary["pred_aus"],
                "final_emotion": summary["final_emotion"],
                "critic_changed": False,
                "anger_gate": False,
            }
            template_rows = (
                cas_templates.get(clip["clip_id"], [])
                if clip.get("template_dataset") == "casme3"
                else samm_templates.get(clip["clip_id"], [])
            )
            answer_rows = [
                {
                    "video_id": row["video_id"],
                    "video": row["video"],
                    "question": row["question"],
                    "answer": summary["answer_map"].get(row["question"], "other"),
                }
                for row in template_rows
            ]
            if clip.get("submission_split") == "cas":
                cas_rows.append(result_row)
                new_cas_answers.extend(answer_rows)
            else:
                samm_rows.append(result_row)
                new_samm_answers.extend(answer_rows)
        if cas_rows:
            write_jsonl(cas_results, cas_rows)
        if samm_rows:
            write_jsonl(samm_results, samm_rows)
        if new_cas_answers:
            write_jsonl(cas_pred, new_cas_answers)
        if new_samm_answers:
            write_jsonl(samm_pred, new_samm_answers)

    _rewrite_submission_from_existing(cas_pred, cas_templates)
    _rewrite_submission_from_existing(samm_pred, samm_templates)


def _rewrite_submission_from_existing(output_file: Path, templates: dict[str, list[dict]]) -> None:
    existing_answers = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    existing_answers[(row["video"], row["question"])] = row["answer"]
                except Exception:
                    continue
    rebuilt = []
    for clip_id in sorted(templates):
        for row in templates[clip_id]:
            rebuilt.append(
                {
                    "video_id": row["video_id"],
                    "video": row["video"],
                    "question": row["question"],
                    "answer": existing_answers.get((row["video"], row["question"]), row.get("answer", "")),
                }
            )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        for row in rebuilt:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
