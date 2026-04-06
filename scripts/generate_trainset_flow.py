"""Generate onset/apex/offset trainset TV-L1 flow directly from resolved original frames."""

from __future__ import annotations

import argparse
import csv
import shutil
import traceback
from pathlib import Path

import cv2

from flow_preprocess.flow.optical_flow_export import compute_flow, describe_flow_tensor_channels, save_flow_bundle
from flow_preprocess.trainset.trainset_sample_resolver import (
    ResolvedTrainSample,
    _index_samm_annotations,
    load_annotation_indexes,
    load_unique_train_samples,
    resolve_train_sample,
)
from flow_preprocess.utils import ensure_directory, utc_timestamp, write_json


FLOW_METHOD_CHOICES = ("tv_l1",)
TRAINSET_PREPROCESS_MODE = "direct_original_frames"
TRAINSET_FLOW_VARIANT = "tv_l1_direct"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_samm_corp_sample(
    sample: dict,
    samm_root: Path,
    samm_annotations: dict,
) -> ResolvedTrainSample:
    """Resolve a SAMM sample from samm_Crop_0329-style layout: {samm_root}/{video_name}/cropped/."""
    filename = str(sample.get("filename") or sample.get("image_id") or "").strip()
    subject = str(sample.get("subject") or "").strip()
    sample_id = str(sample.get("id") or f"samm_{filename}").rsplit("_", 1)[0]
    subject_token = f"{int(subject):03d}"

    annotation = samm_annotations.get(filename)
    if annotation is None:
        raise KeyError(f"SAMM annotation row not found for sample: {filename}")

    cropped_dir = samm_root / filename / "cropped"
    if not cropped_dir.exists():
        raise FileNotFoundError(f"samm_corp cropped directory not found: {cropped_dir}")

    def find_frame(frame_index: int) -> Path:
        # samm_Crop_0329 naming: {subject}_{frame_index}_crop.png (no zero-padding)
        for pattern in [
            f"{subject_token}_{frame_index}_crop.*",
            f"*_{frame_index}_crop.*",
            f"*{frame_index}_crop.*",
        ]:
            matches = sorted(cropped_dir.glob(pattern))
            if matches:
                return matches[0]
        raise FileNotFoundError(
            f"Frame {frame_index} not found in {cropped_dir}"
        )

    onset = int(annotation["onset"])
    apex = int(annotation["apex"])
    offset = int(annotation["offset"])
    return ResolvedTrainSample(
        sample_id=sample_id,
        dataset="samm",
        subject=subject_token,
        video_name=filename,
        onset=onset,
        apex=apex,
        offset=offset,
        onset_path=find_frame(onset),
        apex_path=find_frame(apex),
        offset_path=find_frame(offset),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trainset TV-L1 optical flow directly from resolved original frames."
    )
    parser.add_argument("--jsonl", default=str(REPO_ROOT / "me_vqa_samm_casme2_smic_v2.jsonl"))
    parser.add_argument("--samm-root", default="")
    parser.add_argument("--casme2-root", default="")
    parser.add_argument("--samm-anno", default="")
    parser.add_argument("--casme2-anno", default="")
    parser.add_argument("--output-root", default="outputs/trainset_flow")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--flow-method", choices=FLOW_METHOD_CHOICES, default="tv_l1")
    parser.add_argument("--skip-casme2", action="store_true", help="Skip loading CASME II annotations and samples.")
    parser.add_argument("--samm-corp-layout", action="store_true",
                        help="Use samm_Crop_0329-style layout: {samm_root}/{video_name}/cropped/ instead of {samm_root}/{subject}/{video_name}/.")
    return parser.parse_args()


def _copy_frame(path: Path, target: Path) -> str:
    ensure_directory(target.parent)
    shutil.copy2(path, target)
    return str(target)

def _write_failures_csv(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_directory(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["sample_id", "dataset", "reason"])
        writer.writeheader()
        writer.writerows(rows)


def _load_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to read resolved trainset frame: {path}")
    return image


def _resize_image_to_shape(image, target_shape: tuple[int, int]):
    target_height, target_width = target_shape
    interpolation = cv2.INTER_AREA if target_height < image.shape[0] or target_width < image.shape[1] else cv2.INTER_CUBIC
    return cv2.resize(image, (int(target_width), int(target_height)), interpolation=interpolation)


def _prepare_pair_images(
    start_image,
    end_image,
    *,
    pair_name: str,
    start_label: str,
    end_label: str,
) -> tuple[object, object, dict[str, object] | None]:
    if start_image.shape[:2] == end_image.shape[:2]:
        return start_image, end_image, None

    resized_end_image = _resize_image_to_shape(end_image, start_image.shape[:2])
    return start_image, resized_end_image, {
        "pair_name": pair_name,
        "status": "applied",
        "reason": "pair_frame_shapes_differed",
        "source_shape": list(end_image.shape),
        "target_shape": list(start_image.shape),
        "resized_image_label": end_label,
        "reference_image_label": start_label,
    }


def main() -> None:
    args = parse_args()
    samples = load_unique_train_samples(args.jsonl)
    if args.skip_casme2:
        samples = [s for s in samples if str(s.get("dataset", "")).strip().lower() != "casme2"]
    if args.limit is not None:
        samples = samples[: args.limit]

    output_root = ensure_directory(args.output_root)
    if args.skip_casme2:
        samm_annotations = _index_samm_annotations(args.samm_anno)
        casme2_annotations = {}
    else:
        samm_annotations, casme2_annotations = load_annotation_indexes(args.samm_anno, args.casme2_anno)
    failures: list[dict[str, str]] = []
    total_items = len(samples)
    success_items = 0
    failed_items = 0
    partial_items = 0

    for sample in samples:
        sample_id = str(sample.get("id") or "unknown").rsplit("_", 1)[0]
        dataset = str(sample.get("dataset", "")).strip().lower()
        try:
            if args.samm_corp_layout and dataset == "samm":
                resolved = _resolve_samm_corp_sample(sample, Path(args.samm_root), samm_annotations)
            else:
                resolved = resolve_train_sample(
                    sample,
                    samm_root=args.samm_root,
                    casme2_root=args.casme2_root,
                    samm_annotations=samm_annotations,
                    casme2_annotations=casme2_annotations,
                )
            sample_root_dir = output_root / resolved.dataset / resolved.sample_id
            sample_output_dir = sample_root_dir / TRAINSET_FLOW_VARIANT
            frames_used_dir = sample_output_dir / "frames_used"
            flow_npy_dir = sample_output_dir / "flow_npy"
            flow_rgb_dir = sample_output_dir / "flow_rgb"
            flow_hsv_dir = sample_output_dir / "flow_hsv"

            original_paths = {
                "onset": resolved.onset_path,
                "apex": resolved.apex_path,
                "offset": resolved.offset_path,
            }
            for label, frame_path in original_paths.items():
                _copy_frame(frame_path, frames_used_dir / f"{label}{frame_path.suffix.lower()}")

            frame_images = {
                label: _load_image(original_paths[label])
                for label in ("onset", "apex", "offset")
            }

            pair_results = {}
            pair_failures = {}
            resize_for_shape_compatibility: dict[str, dict[str, object]] = {}
            pair_specs = {
                "onset_to_apex": ("onset", "apex"),
                "apex_to_offset": ("apex", "offset"),
            }
            for pair_name, (start_key, end_key) in pair_specs.items():
                try:
                    start_image, end_image, resize_record = _prepare_pair_images(
                        frame_images[start_key],
                        frame_images[end_key],
                        pair_name=pair_name,
                        start_label=start_key,
                        end_label=end_key,
                    )
                    if resize_record is not None:
                        resize_for_shape_compatibility[pair_name] = resize_record
                    flow = compute_flow(
                        start_image,
                        end_image,
                        method=args.flow_method,
                        image_size=None,
                    )
                    pair_results[pair_name] = save_flow_bundle(
                        flow,
                        flow_npy_dir / f"{pair_name}.npy",
                        flow_rgb_dir / f"{pair_name}.png",
                        flow_hsv_dir / f"{pair_name}.png",
                    )
                except Exception as pair_error:
                    pair_failures[pair_name] = str(pair_error)

            metadata = {
                "sample_id": resolved.sample_id,
                "dataset": resolved.dataset,
                "subject": resolved.subject,
                "video_name": resolved.video_name,
                "onset": resolved.onset,
                "apex": resolved.apex,
                "offset": resolved.offset,
                "onset_path": str(resolved.onset_path),
                "apex_path": str(resolved.apex_path),
                "offset_path": str(resolved.offset_path),
                "flow_method": args.flow_method,
                "flow_variant": TRAINSET_FLOW_VARIANT,
                "trainset_preprocess": TRAINSET_PREPROCESS_MODE,
                "flow_tensor_channels": describe_flow_tensor_channels(args.flow_method),
                "onset_shape": list(frame_images["onset"].shape),
                "apex_shape": list(frame_images["apex"].shape),
                "offset_shape": list(frame_images["offset"].shape),
                "pair_1": "onset_to_apex",
                "pair_2": "apex_to_offset",
                "saved_files": pair_results,
                "pair_failures": pair_failures,
                "resize_for_shape_compatibility": resize_for_shape_compatibility,
                "face_crop_alignment_used": False,
                "note": "Trainset flow uses resolved original frames directly. No face crop or facial alignment is applied.",
            }
            write_json(metadata, sample_output_dir / "metadata.json")

            if len(pair_results) == len(pair_specs):
                success_items += 1
            elif pair_results:
                partial_items += 1
            else:
                failed_items += 1
                failures.append(
                    {
                        "sample_id": sample_id,
                        "dataset": dataset,
                        "reason": "No flow pairs were written.",
                    }
                )
        except Exception as error:
            failed_items += 1
            failures.append({"sample_id": sample_id, "dataset": dataset, "reason": str(error)})
            if args.debug:
                traceback.print_exc()

    _write_failures_csv(output_root / "failures.csv", failures)
    summary = {
        "total_items": total_items,
        "success_items": success_items,
        "failed_items": failed_items,
        "partial_items": partial_items,
        "timestamp": utc_timestamp(),
        "config_used": {
            "jsonl": args.jsonl,
            "samm_root": args.samm_root,
            "casme2_root": args.casme2_root,
            "samm_anno": args.samm_anno,
            "casme2_anno": args.casme2_anno,
            "output_root": args.output_root,
            "limit": args.limit,
            "overwrite": args.overwrite,
            "flow_method": args.flow_method,
            "flow_variant": TRAINSET_FLOW_VARIANT,
            "trainset_preprocess": TRAINSET_PREPROCESS_MODE,
        },
    }
    write_json(summary, output_root / "run_summary.json")


if __name__ == "__main__":
    main()
