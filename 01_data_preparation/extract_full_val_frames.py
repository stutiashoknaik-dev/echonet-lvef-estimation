
import cv2
import os
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE = Path(os.environ.get("ECHONET_ROOT", "./EchoNet_LV"))

VIDEO_DIR = BASE / "data" / "videos_split" / "val"
OUTPUT_DIR = BASE / "data" / "frames_full" / "val"

IMG_SIZE = 112

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

video_files = sorted(VIDEO_DIR.glob("*.avi"))

print(f"Validation videos found: {len(video_files)}")

total_frames = 0
failed_videos = []

for video_path in video_files:
    video_id = video_path.stem
    save_dir = OUTPUT_DIR / video_id
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path.name}")
        failed_videos.append(video_path.name)
        cap.release()
        continue

    frame_idx = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        # Match the 112x112 input size used by the
        # existing segmentation pipeline.
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        # Save grayscale frames using the filenames
        # expected by extract_signals_from_png_masks.py.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        output_path = save_dir / f"frame_{frame_idx}.png"

        if cv2.imwrite(str(output_path), gray):
            frame_idx += 1
        else:
            print(f"[WARN] Could not save {output_path}")

    cap.release()
    total_frames += frame_idx

    print(f"{video_path.name}: {frame_idx} frames")

print("\n========== EXTRACTION SUMMARY ==========")
print(f"Videos found       : {len(video_files)}")
print(f"Videos failed      : {len(failed_videos)}")
print(f"Total frames saved : {total_frames}")
print(f"Output directory   : {OUTPUT_DIR}")

if failed_videos:
    print("Failed videos:")
    for name in failed_videos:
        print(f"  {name}")
