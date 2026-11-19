# ============================================================
# CORRECT FRAME EXTRACTION PIPELINE (FINAL - 112x112)
# ============================================================

import os
import cv2
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

VIDEO_PATH = os.path.join(BASE_PATH, "data/videos_split")
CSV_PATH   = os.path.join(BASE_PATH, "data/VolumeTracings.csv")
SAVE_PATH  = os.path.join(BASE_PATH, "data/frames")

IMG_SIZE = 112  # 🔥 IMPORTANT

# ============================================================
# STEP 1: COMPUTE ED / ES USING AREA
# ============================================================

print("Computing ED/ES frames...")

df = pd.read_csv(CSV_PATH)

frame_dict = {}

grouped = df.groupby(["FileName", "Frame"])

for (file, frame), group in grouped:

    pts = group[["X1", "Y1"]].values

    x = pts[:, 0]
    y = pts[:, 1]

    # polygon area (Shoelace formula)
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    if file not in frame_dict:
        frame_dict[file] = []

    frame_dict[file].append((frame, area))


final_frames = {}

for file, values in frame_dict.items():

    values = sorted(values, key=lambda x: x[1])

    es = values[0][0]   # smallest area
    ed = values[-1][0]  # largest area

    final_frames[file] = {"ED": ed, "ES": es}

print("ED/ES ready!")

# ============================================================
# AUGMENTATION (TRAIN ONLY)
# ============================================================

def augment(img, save_dir, base_name):

    # Flip
    flip = cv2.flip(img, 1)
    cv2.imwrite(os.path.join(save_dir, base_name + "_flip.png"), flip)

    # Rotation (+10 deg)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), 10, 1)
    rot = cv2.warpAffine(img, M, (w, h))
    cv2.imwrite(os.path.join(save_dir, base_name + "_rot.png"), rot)


# ============================================================
# FRAME EXTRACTION FUNCTION
# ============================================================

def extract_frames(video_path, save_dir, ed, es, split):

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 RESIZE (CRITICAL)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        # ---------------- TRAIN ----------------
        if split == "train":

            target_frames = [ed-1, ed, ed+1, es-1, es, es+1]

            if frame_id in target_frames:

                # ✅ SAFE labeling
                if frame_id in [ed-1, ed, ed+1]:
                    label = "ED"
                else:
                    label = "ES"

                name = f"{label}_{frame_id}.png"

                cv2.imwrite(os.path.join(save_dir, name), frame)

                # augmentation only for train
                augment(frame, save_dir, name.replace(".png", ""))

        # ---------------- VALIDATION ----------------
        elif split == "val":

            if frame_id == ed or frame_id == es:

                label = "ED" if frame_id == ed else "ES"
                name = f"{label}_{frame_id}.png"

                cv2.imwrite(os.path.join(save_dir, name), frame)

        # ---------------- TEST ----------------
        elif split == "test":

            name = f"frame_{frame_id}.png"
            cv2.imwrite(os.path.join(save_dir, name), frame)

        frame_id += 1

    cap.release()


# ============================================================
# MAIN LOOP
# ============================================================

splits = ["train", "val", "test"]

for split in splits:

    print(f"\nProcessing {split}...")

    split_path = os.path.join(VIDEO_PATH, split)

    for video in os.listdir(split_path):

        video_path = os.path.join(split_path, video)

        if video not in final_frames:
            continue

        ed = final_frames[video]["ED"]
        es = final_frames[video]["ES"]

        save_dir = os.path.join(SAVE_PATH, split, video.replace(".avi", ""))
        os.makedirs(save_dir, exist_ok=True)

        extract_frames(video_path, save_dir, ed, es, split)

print("\n✅ DONE: Frame extraction completed successfully!")
