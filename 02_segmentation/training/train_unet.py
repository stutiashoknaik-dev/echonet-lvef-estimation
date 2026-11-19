import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# CONFIG
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")
IMG_SIZE = 112
BATCH_SIZE = 8
EPOCHS = 40

# ============================================================
# LOAD PATHS (NO MEMORY LOAD)
# ============================================================

def load_paths(split):

    frame_dir = os.path.join(BASE_PATH, f"data/frames/{split}")
    mask_dir  = os.path.join(BASE_PATH, f"data/masks/{split}")

    img_paths, mask_paths = [], []

    for video in os.listdir(frame_dir):

        v_frame = os.path.join(frame_dir, video)
        v_mask  = os.path.join(mask_dir, video)

        if not os.path.exists(v_mask):
            continue

        for file in os.listdir(v_frame):

            img_path = os.path.join(v_frame, file)
            mask_path = os.path.join(v_mask, file.replace(".png", "_mask.png"))

            if os.path.exists(mask_path):
                img_paths.append(img_path)
                mask_paths.append(mask_path)

    print(f"{split} samples: {len(img_paths)}")

    return img_paths, mask_paths

# ============================================================
# GENERATOR (IMPORTANT PART)
# ============================================================

def data_generator(img_paths, mask_paths):

    while True:

        # 🔥 shuffle every epoch
        idx = np.random.permutation(len(img_paths))

        for i in range(0, len(idx), BATCH_SIZE):

            batch_idx = idx[i:i+BATCH_SIZE]

            X, Y = [], []

            for j in batch_idx:

                # IMAGE
                img = cv2.imread(img_paths[j], cv2.IMREAD_GRAYSCALE)
                img = img.astype(np.float32) / 255.0

                # MASK
                mask = cv2.imread(mask_paths[j], cv2.IMREAD_GRAYSCALE)
                mask = (mask > 0).astype(np.float32)

                X.append(img)
                Y.append(mask)

            yield np.array(X)[..., np.newaxis], np.array(Y)[..., np.newaxis]

# ============================================================
# U-NET (OPTIMIZED)
# ============================================================

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x

def build_unet():

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1))

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    b = conv_block(p3, 256)

    u1 = layers.UpSampling2D()(b)
    u1 = layers.concatenate([u1, c3])
    c4 = conv_block(u1, 128)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = conv_block(u2, 64)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.concatenate([u3, c1])
    c6 = conv_block(u3, 32)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c6)

    return models.Model(inputs, outputs)

# ============================================================
# LOSS + METRIC
# ============================================================

def dice_coef(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

def loss_fn(y_true, y_pred):

    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
    dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)

    return tf.reduce_mean(bce) + dice_loss

# ============================================================
# PREPARE DATA
# ============================================================

train_imgs, train_masks = load_paths("train")
val_imgs, val_masks     = load_paths("val")

train_gen = data_generator(train_imgs, train_masks)
val_gen   = data_generator(val_imgs, val_masks)

steps_train = len(train_imgs) // BATCH_SIZE
steps_val   = len(val_imgs) // BATCH_SIZE

# ============================================================
# BUILD MODEL
# ============================================================

model = build_unet()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=loss_fn,
    metrics=[dice_coef]
)

model.summary()

# ============================================================
# TRAIN
# ============================================================

history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=steps_train,
    validation_steps=steps_val,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3),
        tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
    ]
)

model.save("final_model.keras")

print("✅ TRAINING DONE")
