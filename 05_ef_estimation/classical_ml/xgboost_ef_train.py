import os
import numpy as np
import pandas as pd
import joblib

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from scipy.stats import pearsonr

# ============================================================
# PATHS
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

TRAIN_CSV = os.path.join(
    BASE_PATH,
    "outputs/ef_features/train_ef_features.csv"
)

VAL_CSV = os.path.join(
    BASE_PATH,
    "outputs/ef_features/val_ef_features.csv"
)

SAVE_DIR = os.path.join(
    BASE_PATH,
    "outputs/ef_model"
)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("Train Shape:", train_df.shape)
print("Val Shape  :", val_df.shape)

# ============================================================
# REMOVE NON-FEATURE COLUMNS
# ============================================================

DROP_COLUMNS = [
    "FileName",
    "VideoKey",
    "Split",
    "GT_EF",
    "ED_MaskFile",
    "ES_MaskFile"
]

X_train = train_df.drop(columns=DROP_COLUMNS)

y_train = train_df["GT_EF"]

X_val = val_df.drop(columns=DROP_COLUMNS)

y_val = val_df["GT_EF"]

print("\nFeature Count:", X_train.shape[1])

# ============================================================
# XGBOOST REGRESSOR
# ============================================================

model = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training Completed!")

# ============================================================
# VALIDATION PREDICTION
# ============================================================

y_pred = model.predict(X_val)

# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_val,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_val,
        y_pred
    )
)

r2 = r2_score(
    y_val,
    y_pred
)

pearson_r, p_value = pearsonr(
    y_val,
    y_pred
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n====================================")
print("VALIDATION RESULTS")
print("====================================")

print(f"MAE        : {mae:.4f}")
print(f"RMSE       : {rmse:.4f}")
print(f"R2 Score   : {r2:.4f}")
print(f"Pearson r  : {pearson_r:.4f}")

# ============================================================
# SAVE PREDICTIONS
# ============================================================

pred_df = pd.DataFrame({

    "FileName": val_df["FileName"],

    "GT_EF": y_val,

    "Predicted_EF": y_pred,

    "Absolute_Error":
        np.abs(
            y_val - y_pred
        )
})

pred_csv = os.path.join(
    SAVE_DIR,
    "Validation_Predictions.csv"
)

pred_df.to_csv(
    pred_csv,
    index=False
)

# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    SAVE_DIR,
    "xgboost_ef_model.pkl"
)

joblib.dump(
    model,
    model_path
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({

    "Feature":
        X_train.columns,

    "Importance":
        model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_csv = os.path.join(
    SAVE_DIR,
    "Feature_Importance.csv"
)

importance_df.to_csv(
    importance_csv,
    index=False
)

# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([{

    "MAE": mae,

    "RMSE": rmse,

    "R2": r2,

    "Pearson_r": pearson_r
}])

metrics_csv = os.path.join(
    SAVE_DIR,
    "Validation_Metrics.csv"
)

metrics_df.to_csv(
    metrics_csv,
    index=False
)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nSaved Files:")
print(pred_csv)
print(model_path)
print(importance_csv)
print(metrics_csv)

print("\n====================================")
print("EF TRAINING + VALIDATION COMPLETE")
print("====================================")
