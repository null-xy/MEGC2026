#!/usr/bin/env python3
"""
Generate TV-L1 optical flow for MEGC 2025 test set from pre-cropped selected_frames.

Input structure:
    <test_root>/CAS-1/selected_frames/first_*.png, middle_*.png, last_*.png

Output structure (matches batch_runner path convention):
    <output_root>/<clip_id>/tv_l1_direct/flow_npy/onset_to_apex.npy

Run:
    python testset_flow_from_crops.py \
        --input /mnt/d/thesis/mame/data/ME_VQA_MEGC_2025_Test_Crop_0329 \
        --output-root /mnt/d/thesis/mame/trainset_flow/testset \
        --image-size 224
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from flow_preprocess.flow.optical_flow_export import compute_tv_l1_flow, save_flow_bundle


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='')
    p.add_argument('--output-root', default='outputs/trainset_flow/testset')
    p.add_argument('--image-size', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def find_selected_frame(frames_dir: Path, prefix: str) -> Path | None:
    matches = sorted(frames_dir.glob(f'{prefix}_*.png'))
    return matches[0] if matches else None


def main():
    args = parse_args()
    test_root = Path(args.input)
    out_root = Path(args.output_root)
    size = args.image_size

    clip_dirs = sorted(test_root.iterdir())
    clip_dirs = [d for d in clip_dirs if d.is_dir()]
    print(f'Found {len(clip_dirs)} clips under {test_root}')

    success, failed = 0, 0
    for clip_dir in clip_dirs:
        clip_id = clip_dir.name
        frames_dir = clip_dir / 'selected_frames'
        if not frames_dir.exists():
            print(f'  [SKIP] {clip_id}: no selected_frames/')
            failed += 1
            continue

        first_path  = find_selected_frame(frames_dir, 'first')
        middle_path = find_selected_frame(frames_dir, 'middle')
        if first_path is None or middle_path is None:
            print(f'  [SKIP] {clip_id}: missing first/middle frame')
            failed += 1
            continue

        npy_path = out_root / clip_id / 'tv_l1_direct' / 'flow_npy' / 'onset_to_apex.npy'
        if npy_path.exists() and not args.overwrite:
            print(f'  [SKIP] {clip_id}: already exists')
            success += 1
            continue

        img1 = cv2.imread(str(first_path))
        img2 = cv2.imread(str(middle_path))
        if img1 is None or img2 is None:
            print(f'  [FAIL] {clip_id}: failed to read images')
            failed += 1
            continue

        try:
            flow = compute_tv_l1_flow(img1, img2, image_size=size)
            base_dir = npy_path.parent.parent  # .../tv_l1_direct
            rgb_path = base_dir / 'flow_rgb' / 'onset_to_apex.png'
            hsv_path = base_dir / 'flow_hsv' / 'onset_to_apex.png'
            save_flow_bundle(flow, npy_path, rgb_path, hsv_path)

            # Create frames_used/ so probe_clip can find onset/apex images
            frames_used_dir = base_dir / 'frames_used'
            frames_used_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(first_path,  frames_used_dir / 'onset.png')
            shutil.copy2(middle_path, frames_used_dir / 'apex.png')

            print(f'  [OK]   {clip_id}  shape={flow.shape}  '
                  f'first={first_path.name}  middle={middle_path.name}')
            success += 1
        except Exception as e:
            print(f'  [FAIL] {clip_id}: {e}')
            failed += 1

    print(f'\nDone: {success} success, {failed} failed')
    print(f'Output: {out_root}')


if __name__ == '__main__':
    main()
