#!/usr/bin/env python3
"""Build per-frame area/motion signals from predicted PNG masks and extracted PNG frames.

Expected folder structure:
  masks_root/<video_id>/frame_0.png, frame_1.png, ...
  frames_root/<video_id>/frame_0.png, frame_1.png, ...

Output columns: video_id, frame, area, motion

Farneback parameters match the supplied manuscript script:
  pyr_scale=0.5, levels=3, winsize=15, iterations=3,
  poly_n=5, poly_sigma=1.2, flags=0.
"""
import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FRAME_RE = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)


def frame_files(folder: Path):
    """Return {frame_index: path} for filenames such as frame_12.png."""
    found = {}
    if not folder.is_dir():
        return found
    for path in folder.iterdir():
        if not path.is_file():
            continue
        match = FRAME_RE.match(path.name)
        if match:
            found[int(match.group(1))] = path
    return found


def binary_mask(mask_img: np.ndarray, threshold: float) -> np.ndarray:
    """Support binary PNGs encoded as 0/255, 0/1, or grayscale/probability-like images."""
    if mask_img.size == 0:
        return np.zeros(mask_img.shape, dtype=bool)
    max_value = float(np.max(mask_img))
    if max_value <= 1.0:
        return mask_img > 0.5
    return mask_img > threshold


def main():
    parser = argparse.ArgumentParser(
        description="Create frame-wise LV area and optical-flow motion CSV from PNG masks."
    )
    parser.add_argument("--masks", required=True,
                        help="Root containing one predicted-mask subfolder per video ID")
    parser.add_argument("--frames", required=True,
                        help="Root containing the corresponding original extracted-frame subfolders")
    parser.add_argument("--out", default="signals.csv", help="Output CSV path")
    parser.add_argument("--thr", type=float, default=127.0,
                        help="Threshold for 0-255 grayscale mask PNGs (default: 127)")
    args = parser.parse_args()

    masks_root = Path(args.masks)
    frames_root = Path(args.frames)
    if not masks_root.is_dir():
        raise SystemExit(f"Mask root not found: {masks_root}")
    if not frames_root.is_dir():
        raise SystemExit(f"Original-frame root not found: {frames_root}")

    video_dirs = sorted(p for p in masks_root.iterdir() if p.is_dir())
    if not video_dirs:
        raise SystemExit(
            f"No video subfolders found under {masks_root}. Expected masks/<video_id>/frame_0.png, ..."
        )

    rows = []
    total_missing_masks = 0
    total_missing_frames = 0
    videos_with_rows = 0

    for k, mask_video_dir in enumerate(video_dirs, start=1):
        video_id = mask_video_dir.name
        mask_map = frame_files(mask_video_dir)
        frame_map = frame_files(frames_root / video_id)

        if not mask_map:
            print(f"[WARN] No frame_*.png masks found for {video_id}; skipping")
            continue
        if not frame_map:
            print(f"[WARN] No original frame_*.png files found for {video_id} at {frames_root / video_id}; skipping")
            continue

        common_ids = sorted(set(mask_map) & set(frame_map))
        total_missing_masks += len(set(frame_map) - set(mask_map))
        total_missing_frames += len(set(mask_map) - set(frame_map))

        if not common_ids:
            print(f"[WARN] No matching mask/frame filenames for {video_id}; skipping")
            continue

        prev_idx = None
        prev_gray = None
        video_rows = []

        for frame_idx in common_ids:
            mask_img = cv2.imread(str(mask_map[frame_idx]), cv2.IMREAD_GRAYSCALE)
            gray = cv2.imread(str(frame_map[frame_idx]), cv2.IMREAD_GRAYSCALE)
            if mask_img is None or gray is None:
                print(f"[WARN] Could not read {video_id} frame {frame_idx}; skipping frame")
                continue

            area = int(np.count_nonzero(binary_mask(mask_img, args.thr)))
            motion = 0.0

            if prev_gray is not None and prev_idx is not None and frame_idx == prev_idx + 1:
                if gray.shape != prev_gray.shape:
                    gray_for_flow = cv2.resize(gray, (prev_gray.shape[1], prev_gray.shape[0]))
                else:
                    gray_for_flow = gray
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray_for_flow, None,
                    0.5, 3, 15, 3, 5, 1.2, 0
                )
                motion = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))

            video_rows.append({
                "video_id": video_id,
                "frame": int(frame_idx),
                "area": area,
                "motion": motion,
            })

            prev_gray = gray.copy()
            prev_idx = frame_idx

        rows.extend(video_rows)
        if video_rows:
            videos_with_rows += 1
        if k % 100 == 0 or k == len(video_dirs):
            print(f"Processed {k}/{len(video_dirs)} mask folders")

    if not rows:
        raise SystemExit(
            "No signals were produced. Check that mask and frame subfolders use matching video IDs "
            "and filenames such as frame_0.png, frame_1.png."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values(["video_id", "frame"])
    result.to_csv(out_path, index=False)

    print(f"\nWrote: {out_path}")
    print(f"Videos with signals: {videos_with_rows}")
    print(f"Rows written: {len(result)}")
    print(f"Mask frames without matching original frames: {total_missing_frames}")
    print(f"Original frames without matching masks: {total_missing_masks}")


if __name__ == "__main__":
    main()
