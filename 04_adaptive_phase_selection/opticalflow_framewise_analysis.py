# ============================================================
# OPTICAL FLOW FRAMEWISE ANALYSIS
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd

# ============================================================
# BASE PATH
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# ============================================================
# INPUT FRAMES
# ============================================================

FRAME_ROOT = os.path.join(
    BASE_PATH,
    "data",
    "frames",
    "test"
)

# ============================================================
# OUTPUT
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "opticalflow_phase_detection"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

OUTPUT_CSV = os.path.join(
    SAVE_DIR,
    "opticalflow_framewise_analysis.csv"
)

# ============================================================
# STORAGE
# ============================================================

results = []

# ============================================================
# VIDEOS
# ============================================================

videos = sorted([
    d for d in os.listdir(FRAME_ROOT)
    if os.path.isdir(
        os.path.join(FRAME_ROOT, d)
    )
])

print("Total videos:", len(videos))

# ============================================================
# PROCESS
# ============================================================

for video in videos:

    print(
        "Processing:",
        video
    )

    video_dir = os.path.join(
        FRAME_ROOT,
        video
    )

    frame_files = [
        f for f in os.listdir(video_dir)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]

    def frame_number(filename):

        digits = "".join(
            c for c in os.path.splitext(filename)[0]
            if c.isdigit()
        )

        return int(digits) if digits else -1

    frame_files = sorted(
        frame_files,
        key=frame_number
    )

    previous = None

    for frame_file in frame_files:

        frame_idx = frame_number(
            frame_file
        )

        path = os.path.join(
            video_dir,
            frame_file
        )

        current = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

        if current is None:
            continue

        # ----------------------------------------------------
        # FIRST FRAME
        # ----------------------------------------------------

        if previous is None:

            motion = 0.0

        else:

            # ------------------------------------------------
            # OPTICAL FLOW
            # ------------------------------------------------

            flow = cv2.calcOpticalFlowFarneback(

                previous,
                current,

                None,

                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            # ------------------------------------------------
            # FLOW COMPONENTS
            # ------------------------------------------------

            fx = flow[..., 0]
            fy = flow[..., 1]

            magnitude = np.sqrt(
                fx ** 2 +
                fy ** 2
            )

            motion = float(
                np.mean(magnitude)
            )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        results.append({

            "Video": video,

            "Frame": int(frame_idx),

            "Motion_Magnitude": motion
        })

        previous = current

# ============================================================
# SAVE
# ============================================================

out_df = pd.DataFrame(
    results
)

out_df = out_df.sort_values(
    ["Video", "Frame"]
)

out_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print("============================================")
print("OPTICAL FLOW ANALYSIS COMPLETE")
print("============================================")
print("Saved:")
print(OUTPUT_CSV)
print("Rows:", len(out_df))
