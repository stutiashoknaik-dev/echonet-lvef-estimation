# ============================================================
# COMPARE TRAIN / VAL / TEST FEATURE DISTRIBUTIONS
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

TRAIN_CSV = os.path.join(
    BASE_PATH,
    "outputs",
    "mask_features",
    "train_mask_features.csv"
)

VAL_CSV = os.path.join(
    BASE_PATH,
    "outputs",
    "mask_features",
    "val_mask_features.csv"
)

TEST_CSV = os.path.join(
    BASE_PATH,
    "outputs",
    "mask_features",
    "test_mask_features.csv"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs",
    "analysis",
    "Feature_Distributions"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

# ============================================================
# LOAD CSVs
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

print("\n===================================")
print("DATA LOADED")
print("===================================")

print("Train :", train_df.shape)
print("Val   :", val_df.shape)
print("Test  :", test_df.shape)

# ============================================================
# IMPORTANT FEATURES
# ============================================================

features = [

    "ED_Area",
    "ES_Area",

    "ED_Perimeter",
    "ES_Perimeter",

    "ED_MajorAxis",
    "ES_MajorAxis",

    "ED_MinorAxis",
    "ES_MinorAxis",

    "Area_Change",
    "Area_Ratio",

    "Perimeter_Change",
    "Perimeter_Ratio"

]

# ============================================================
# PLOT EACH FEATURE
# ============================================================

for feature in features:

    if feature not in train_df.columns:
        print(f"Skipping {feature}")
        continue

    plt.figure(figsize=(8,6))

    plt.boxplot(

        [

            train_df[feature].dropna(),

            val_df[feature].dropna(),

            test_df[feature].dropna()

        ],

        labels=[

            "Train",

            "Validation",

            "Test"

        ],

        showfliers=False

    )

    plt.title(feature)

    plt.ylabel(feature)

    plt.grid(alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(

        SAVE_DIR,

        feature + ".png"

    )

    plt.savefig(

        save_path,

        dpi=300

    )

    plt.close()

print()

print("===================================")
print("ALL FEATURE PLOTS SAVED")
print("===================================")

print()

print(SAVE_DIR)
