# ============================================================
# EDGE-AWARE BOUNDARY FINE-TUNING
# OPTIMIZED ATTENTION U-NET
# MEMORY-EFFICIENT VERSION
# ============================================================

import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)

# ============================================================
# CONFIG
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

IMG_SIZE = 112

BATCH_SIZE = 8
EPOCHS = 10

LEARNING_RATE = 1e-5

# ============================================================
# DATA PATHS
# ============================================================

TRAIN_FRAME_DIR = os.path.join(
    BASE_PATH,
    "data/frames/train"
)

TRAIN_MASK_DIR = os.path.join(
    BASE_PATH,
    "data/masks/train"
)

VAL_FRAME_DIR = os.path.join(
    BASE_PATH,
    "data/frames/val"
)

VAL_MASK_DIR = os.path.join(
    BASE_PATH,
    "data/masks/val"
)

# ============================================================
# PRETRAINED MODEL
# ============================================================

PRETRAINED_MODEL = "attention_unet_edge_finetuned.keras"

# ============================================================
# SAVE MODEL
# ============================================================

SAVE_MODEL = "attention_unet_edge_finetuned1.keras"

# ============================================================
# DATA GENERATOR
# ============================================================

class EchoDataset(tf.keras.utils.Sequence):

    def __init__(
        self,
        frame_dir,
        mask_dir,
        batch_size=8,
        shuffle=True
    ):

        self.frame_dir = frame_dir
        self.mask_dir = mask_dir

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.samples = []

        # ====================================================
        # COLLECT FILES
        # ====================================================

        for video in os.listdir(frame_dir):

            video_frame = os.path.join(
                frame_dir,
                video
            )

            video_mask = os.path.join(
                mask_dir,
                video
            )

            if not os.path.exists(video_mask):
                continue

            for file in os.listdir(video_frame):

                img_path = os.path.join(
                    video_frame,
                    file
                )

                mask_path = os.path.join(

                    video_mask,

                    file.replace(
                        ".png",
                        "_mask.png"
                    )
                )

                if os.path.exists(mask_path):

                    self.samples.append(
                        (img_path, mask_path)
                    )

        self.on_epoch_end()

    # ========================================================
    # TOTAL BATCHES
    # ========================================================

    def __len__(self):

        return int(
            np.ceil(
                len(self.samples) / self.batch_size
            )
        )

    # ========================================================
    # GET BATCH
    # ========================================================

    def __getitem__(self, idx):

        batch = self.samples[
            idx * self.batch_size :
            (idx + 1) * self.batch_size
        ]

        images = []
        masks = []

        for img_path, mask_path in batch:

            # =================================================
            # IMAGE
            # =================================================

            img = cv2.imread(
                img_path,
                cv2.IMREAD_GRAYSCALE
            )

            img = cv2.resize(
                img,
                (IMG_SIZE, IMG_SIZE)
            )

            img = img.astype(np.float32) / 255.0

            img = np.expand_dims(
                img,
                axis=-1
            )

            # =================================================
            # MASK
            # =================================================

            mask = cv2.imread(
                mask_path,
                cv2.IMREAD_GRAYSCALE
            )

            mask = cv2.resize(
                mask,
                (IMG_SIZE, IMG_SIZE)
            )

            mask = (mask > 0).astype(np.float32)

            mask = np.expand_dims(
                mask,
                axis=-1
            )

            images.append(img)
            masks.append(mask)

        return (
            np.array(images),
            np.array(masks)
        )

    # ========================================================
    # SHUFFLE AFTER EPOCH
    # ========================================================

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.samples)

# ============================================================
# CREATE GENERATORS
# ============================================================

print("\nCreating Training Generator...\n")

train_dataset = EchoDataset(

    TRAIN_FRAME_DIR,
    TRAIN_MASK_DIR,

    batch_size=BATCH_SIZE
)

print("Training Generator Ready")

print("\nCreating Validation Generator...\n")

val_dataset = EchoDataset(

    VAL_FRAME_DIR,
    VAL_MASK_DIR,

    batch_size=BATCH_SIZE,

    shuffle=False
)

print("Validation Generator Ready")

# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(y_true, y_pred):

    smooth = 1e-6

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=[1,2,3]
    )

    union = (
        tf.reduce_sum(y_true, axis=[1,2,3]) +
        tf.reduce_sum(y_pred, axis=[1,2,3])
    )

    dice = (
        2. * intersection + smooth
    ) / (
        union + smooth
    )

    return 1 - tf.reduce_mean(dice)

# ============================================================
# IOU LOSS
# ============================================================

def iou_loss(y_true, y_pred):

    smooth = 1e-6

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=[1,2,3]
    )

    union = (
        tf.reduce_sum(y_true, axis=[1,2,3]) +
        tf.reduce_sum(y_pred, axis=[1,2,3]) -
        intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return 1 - tf.reduce_mean(iou)

# ============================================================
# EDGE-AWARE BOUNDARY LOSS
# ============================================================

def edge_loss(y_true, y_pred):

    # ========================================================
    # SOBEL EDGE MAPS
    # ========================================================

    gt_edges = tf.image.sobel_edges(y_true)

    pred_edges = tf.image.sobel_edges(y_pred)

    # ========================================================
    # EDGE DIFFERENCE
    # ========================================================

    loss = tf.reduce_mean(
        tf.abs(gt_edges - pred_edges)
    )

    return loss

# ============================================================
# FINAL COMBINED LOSS
# ============================================================

def combined_boundary_loss(y_true, y_pred):

    dl = dice_loss(y_true, y_pred)

    il = iou_loss(y_true, y_pred)

    el = edge_loss(y_true, y_pred)

    total = (

        0.5 * dl +

        0.3 * il +

        0.2 * el

    )

    return total

# ============================================================
# METRICS
# ============================================================

def dice_coef(y_true, y_pred):

    smooth = 1e-6

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    return (
        2. * intersection + smooth
    ) / (
        tf.reduce_sum(y_true) +
        tf.reduce_sum(y_pred) +
        smooth
    )

# ------------------------------------------------------------

def iou_metric(y_true, y_pred):

    smooth = 1e-6

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    union = (
        tf.reduce_sum(y_true) +
        tf.reduce_sum(y_pred) -
        intersection
    )

    return (
        intersection + smooth
    ) / (
        union + smooth
    )

# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================

print("\nLoading Optimized Attention U-Net...\n")

model = load_model(

    PRETRAINED_MODEL,

    custom_objects={

        "dice_coef": dice_coef,
        "iou_metric": iou_metric

    },

    compile=False
)

print("Model Loaded!")

# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=combined_boundary_loss,

    metrics=[
        dice_coef,
        iou_metric
    ]
)

# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(

    SAVE_MODEL,

    monitor="val_dice_coef",

    mode="max",

    save_best_only=True,

    verbose=1
)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=1e-7,

    verbose=1
)

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=4,

    restore_best_weights=True,

    verbose=1
)

# ============================================================
# TRAINING
# ============================================================

print("\nStarting Edge-Aware Boundary Fine-Tuning...\n")

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    initial_epoch=5,

    callbacks=[

        checkpoint,
        reduce_lr,
        early_stop
    ]
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(SAVE_MODEL)

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n======================================")
print("EDGE-AWARE FINE-TUNING COMPLETE")
print("======================================")

print(f"\nModel Saved : {SAVE_MODEL}")
