# # ============================================================
# # IMPROVED TEST CONTOUR EXTRACTION
# # VIDEO-WISE ED / ES STRUCTURE
# # ============================================================

# import os
# import cv2
# import numpy as np
# import pandas as pd

# # ============================================================
# # BASE PATH
# # ============================================================

# BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

# # ============================================================
# # ROOT DIRECTORY
# # ============================================================

# ROOT_DIR = os.path.join(
#     BASE_PATH,
#     "outputs/adaptive_phase_selection"
# )

# # ============================================================
# # SAVE DIRECTORY
# # ============================================================

# SAVE_DIR = os.path.join(
#     BASE_PATH,
#     "outputs/test_predicted_contours"
# )

# os.makedirs(SAVE_DIR, exist_ok=True)

# # ============================================================
# # SAVE CSV
# # ============================================================

# SAVE_CSV = os.path.join(
#     SAVE_DIR,
#     "Improved_Predicted_Test_VolumeTracings.csv"
# )

# # ============================================================
# # STORAGE
# # ============================================================

# results = []

# # ============================================================
# # GET VIDEO FOLDERS
# # ============================================================

# video_folders = os.listdir(ROOT_DIR)

# print(f"\nTotal Videos Found : {len(video_folders)}")

# # ============================================================
# # MAIN LOOP
# # ============================================================

# for video_name in video_folders:

#     video_path = os.path.join(
#         ROOT_DIR,
#         video_name
#     )

#     if not os.path.isdir(video_path):
#         continue

#     print(f"\nProcessing : {video_name}")

#     # ========================================================
#     # ED + ES PATHS
#     # ========================================================

#     ED_DIR = os.path.join(
#         video_path,
#         "ED"
#     )

#     ES_DIR = os.path.join(
#         video_path,
#         "ES"
#     )

#     phase_dirs = [

#         ("ED", ED_DIR),

#         ("ES", ES_DIR)

#     ]

#     # ========================================================
#     # LOOP THROUGH ED / ES
#     # ========================================================

#     for phase_name, phase_path in phase_dirs:

#         if not os.path.exists(phase_path):
#             continue

#         # ====================================================
#         # GET PNG FILES
#         # ====================================================

#         mask_files = [

#             f for f in os.listdir(phase_path)

#             if (
#                 f.endswith(".png")
#                 and "mask" in f.lower()
#             )

#         ]

#         # ====================================================
#         # LOOP THROUGH MASKS
#         # ====================================================

#         for mask_file in mask_files:

#             try:

#                 # --------------------------------------------
#                 # Example:
#                 # ED_mask_213.png
#                 # --------------------------------------------

#                 frame_num = int(

#                     mask_file.split("_")[-1]
#                     .replace(".png", "")

#                 )

#             except:
#                 continue

#             # =================================================
#             # LOAD MASK
#             # =================================================

#             mask_path = os.path.join(
#                 phase_path,
#                 mask_file
#             )

#             mask = cv2.imread(
#                 mask_path,
#                 cv2.IMREAD_GRAYSCALE
#             )

#             if mask is None:
#                 continue

#             # =================================================
#             # BINARY
#             # =================================================

#             _, binary = cv2.threshold(

#                 mask,

#                 127,

#                 255,

#                 cv2.THRESH_BINARY
#             )

#             # =================================================
#             # FIND CONTOURS
#             # =================================================

#             contours, _ = cv2.findContours(

#                 binary,

#                 cv2.RETR_EXTERNAL,

#                 cv2.CHAIN_APPROX_NONE
#             )

#             if len(contours) == 0:
#                 continue

#             # =================================================
#             # LARGEST CONTOUR
#             # =================================================

#             contour = max(
#                 contours,
#                 key=cv2.contourArea
#             )

#             contour = contour.squeeze()

#             if len(contour.shape) != 2:
#                 continue

#             # =================================================
#             # KEEP ORIGINAL ORDER
#             # =================================================

#             n = len(contour)

#             half = n // 2

#             left_pts = contour[:half]

#             right_pts = contour[half:]

#             # =================================================
#             # REVERSE RIGHT SIDE
#             # =================================================

#             right_pts = right_pts[::-1]

#             # =================================================
#             # MATCH LENGTH
#             # =================================================

#             n_points = min(
#                 len(left_pts),
#                 len(right_pts)
#             )

#             if n_points < 5:
#                 continue

#             left_pts = left_pts[:n_points]
#             right_pts = right_pts[:n_points]

#             # =================================================
#             # STORE
#             # =================================================

#             for i in range(n_points):

#                 x1, y1 = left_pts[i]
#                 x2, y2 = right_pts[i]

#                 results.append({

#                     "FileName": video_name + ".avi",

#                     "X1": float(x1),
#                     "Y1": float(y1),

#                     "X2": float(x2),
#                     "Y2": float(y2),

#                     "Frame": int(frame_num)
#                 })

# # ============================================================
# # SAVE CSV
# # ============================================================

# df = pd.DataFrame(results)

# df.to_csv(
#     SAVE_CSV,
#     index=False
# )

# # ============================================================
# # FINAL
# # ============================================================

# print("\n======================================")
# print("TEST CONTOUR EXTRACTION COMPLETE")
# print("======================================")

# print(f"\nTotal Points Extracted : {len(df)}")

# print(f"\nSaved CSV:\n{SAVE_CSV}")

# ============================================================
# IMPROVED TEST CONTOUR EXTRACTION
# SEPARATE ED / ES CSV GENERATION
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
# ROOT DIRECTORY
# ============================================================

ROOT_DIR = os.path.join(
    BASE_PATH,
    "outputs/adaptive_phase_selection"
)

# ============================================================
# SAVE DIRECTORY
# ============================================================

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/test_predicted_contours"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# SAVE CSV FILES
# ============================================================

ED_SAVE_CSV = os.path.join(
    SAVE_DIR,
    "Predicted_ED_Test_VolumeTracings.csv"
)

ES_SAVE_CSV = os.path.join(
    SAVE_DIR,
    "Predicted_ES_Test_VolumeTracings.csv"
)

# ============================================================
# STORAGE
# ============================================================

ed_results = []
es_results = []

# ============================================================
# GET VIDEO FOLDERS
# ============================================================

video_folders = os.listdir(ROOT_DIR)

print(f"\nTotal Videos Found : {len(video_folders)}")

# ============================================================
# MAIN LOOP
# ============================================================

for video_name in video_folders:

    video_path = os.path.join(
        ROOT_DIR,
        video_name
    )

    if not os.path.isdir(video_path):
        continue

    print(f"\nProcessing : {video_name}")

    # ========================================================
    # ED + ES PATHS
    # ========================================================

    ED_DIR = os.path.join(
        video_path,
        "ED"
    )

    ES_DIR = os.path.join(
        video_path,
        "ES"
    )

    phase_dirs = [

        ("ED", ED_DIR),

        ("ES", ES_DIR)

    ]

    # ========================================================
    # LOOP THROUGH ED / ES
    # ========================================================

    for phase_name, phase_path in phase_dirs:

        if not os.path.exists(phase_path):
            continue

        # ====================================================
        # GET MASK FILES
        # ====================================================

        mask_files = [

            f for f in os.listdir(phase_path)

            if (
                f.endswith(".png")
                and "mask" in f.lower()
            )

        ]

        # ====================================================
        # LOOP THROUGH MASKS
        # ====================================================

        for mask_file in mask_files:

            try:

                # Example:
                # ED_mask_213.png

                frame_num = int(

                    mask_file.split("_")[-1]
                    .replace(".png", "")

                )

            except:
                continue

            # =================================================
            # LOAD MASK
            # =================================================

            mask_path = os.path.join(
                phase_path,
                mask_file
            )

            mask = cv2.imread(
                mask_path,
                cv2.IMREAD_GRAYSCALE
            )

            if mask is None:
                continue

            # =================================================
            # BINARY MASK
            # =================================================

            _, binary = cv2.threshold(

                mask,

                127,

                255,

                cv2.THRESH_BINARY
            )

            # =================================================
            # FIND CONTOURS
            # =================================================

            contours, _ = cv2.findContours(

                binary,

                cv2.RETR_EXTERNAL,

                cv2.CHAIN_APPROX_NONE
            )

            if len(contours) == 0:
                continue

            # =================================================
            # LARGEST CONTOUR
            # =================================================

            contour = max(
                contours,
                key=cv2.contourArea
            )

            contour = contour.squeeze()

            if len(contour.shape) != 2:
                continue

            # =================================================
            # KEEP ORIGINAL ORDER
            # =================================================

            n = len(contour)

            half = n // 2

            left_pts = contour[:half]

            right_pts = contour[half:]

            # =================================================
            # REVERSE RIGHT SIDE
            # =================================================

            right_pts = right_pts[::-1]

            # =================================================
            # MATCH LENGTHS
            # =================================================

            n_points = min(
                len(left_pts),
                len(right_pts)
            )

            if n_points < 5:
                continue

            left_pts = left_pts[:n_points]
            right_pts = right_pts[:n_points]

            # =================================================
            # STORE RESULTS
            # =================================================

            for i in range(n_points):

                x1, y1 = left_pts[i]
                x2, y2 = right_pts[i]

                row = {

                    "FileName": video_name + ".avi",

                    "X1": float(x1),
                    "Y1": float(y1),

                    "X2": float(x2),
                    "Y2": float(y2),

                    "Frame": int(frame_num)
                }

                # =============================================
                # STORE BASED ON PHASE
                # =============================================

                if phase_name == "ED":
                    ed_results.append(row)

                elif phase_name == "ES":
                    es_results.append(row)

# ============================================================
# CREATE DATAFRAMES
# ============================================================

ed_df = pd.DataFrame(ed_results)
es_df = pd.DataFrame(es_results)

# ============================================================
# SAVE CSV FILES
# ============================================================

ed_df.to_csv(
    ED_SAVE_CSV,
    index=False
)

es_df.to_csv(
    ES_SAVE_CSV,
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n======================================")
print("SEPARATE ED / ES CONTOUR EXTRACTION COMPLETE")
print("======================================")

print(f"\nED Points Extracted : {len(ed_df)}")
print(f"ES Points Extracted : {len(es_df)}")

print(f"\nED CSV Saved:\n{ED_SAVE_CSV}")
print(f"\nES CSV Saved:\n{ES_SAVE_CSV}")
