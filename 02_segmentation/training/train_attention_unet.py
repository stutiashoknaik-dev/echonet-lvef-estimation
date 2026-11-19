import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

# ============================================================
# CONFIG
# ============================================================

BASE_PATH = os.environ.get("ECHONET_ROOT", "./EchoNet_LV")

IMG_SIZE = 112
BATCH_SIZE = 8
EPOCHS = 40

# ============================================================
# LOAD PATHS
# ============================================================

def load_paths(split):

    frame_dir = os.path.join(BASE_PATH, f"data/frames/{split}")
    mask_dir  = os.path.join(BASE_PATH, f"data/masks/{split}")

    img_paths = []
    mask_paths = []

    for video in os.listdir(frame_dir):

        v_frame = os.path.join(frame_dir, video)
        v_mask  = os.path.join(mask_dir, video)

        if not os.path.exists(v_mask):
            continue

        for file in os.listdir(v_frame):

            img_path = os.path.join(v_frame, file)

            mask_name = file.replace(".png", "_mask.png")
            mask_path = os.path.join(v_mask, mask_name)

            if os.path.exists(mask_path):

                img_paths.append(img_path)
                mask_paths.append(mask_path)

    print(f"{split} samples : {len(img_paths)}")

    return img_paths, mask_paths

# ============================================================
# DATA GENERATOR
# ============================================================

def data_generator(img_paths, mask_paths):

    while True:

        idx = np.random.permutation(len(img_paths))

        for i in range(0, len(idx), BATCH_SIZE):

            batch_idx = idx[i:i+BATCH_SIZE]

            X = []
            Y = []

            for j in batch_idx:

                # ------------------------------------------------
                # IMAGE
                # ------------------------------------------------

                img = cv2.imread(
                    img_paths[j],
                    cv2.IMREAD_GRAYSCALE
                )

                img = img.astype(np.float32) / 255.0

                # ------------------------------------------------
                # MASK
                # ------------------------------------------------

                mask = cv2.imread(
                    mask_paths[j],
                    cv2.IMREAD_GRAYSCALE
                )

                mask = (mask > 0).astype(np.float32)

                X.append(img)
                Y.append(mask)

            X = np.array(X)[..., np.newaxis]
            Y = np.array(Y)[..., np.newaxis]

            yield X, Y

# ============================================================
# ATTENTION BLOCK
# ============================================================

def attention_block(g, x, filters):

    g1 = layers.Conv2D(filters, 1, padding='same')(g)
    g1 = layers.BatchNormalization()(g1)

    x1 = layers.Conv2D(filters, 1, padding='same')(x)
    x1 = layers.BatchNormalization()(x1)

    psi = layers.Activation('relu')(g1 + x1)

    psi = layers.Conv2D(1, 1, padding='same')(psi)
    psi = layers.BatchNormalization()(psi)

    psi = layers.Activation('sigmoid')(psi)

    return x * psi

# ============================================================
# CONV BLOCK
# ============================================================

def conv_block(x, filters):

    x = layers.Conv2D(
        filters,
        3,
        padding='same'
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters,
        3,
        padding='same'
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x

# ============================================================
# ATTENTION U-NET
# ============================================================

def build_attention_unet():

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1))

    # ========================================================
    # ENCODER
    # ========================================================

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    # ========================================================
    # BOTTLENECK
    # ========================================================

    b = conv_block(p3, 256)

    # ========================================================
    # DECODER
    # ========================================================

    u1 = layers.UpSampling2D()(b)

    a1 = attention_block(u1, c3, 128)

    u1 = layers.concatenate([u1, a1])

    c4 = conv_block(u1, 128)

    # --------------------------------------------------------

    u2 = layers.UpSampling2D()(c4)

    a2 = attention_block(u2, c2, 64)

    u2 = layers.concatenate([u2, a2])

    c5 = conv_block(u2, 64)

    # --------------------------------------------------------

    u3 = layers.UpSampling2D()(c5)

    a3 = attention_block(u3, c1, 32)

    u3 = layers.concatenate([u3, a3])

    c6 = conv_block(u3, 32)

    # ========================================================
    # OUTPUT
    # ========================================================

    outputs = layers.Conv2D(
        1,
        1,
        activation='sigmoid'
    )(c6)

    return models.Model(inputs, outputs)

# ============================================================
# METRICS
# ============================================================

def dice_coef(y_true, y_pred):

    smooth = 1e-6

    intersection = tf.reduce_sum(y_true * y_pred)

    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true) +
        tf.reduce_sum(y_pred) +
        smooth
    )

# ============================================================
# LOSS
# ============================================================

def loss_fn(y_true, y_pred):

    bce = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred
    )

    smooth = 1e-6

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=[1,2,3]
    )

    union = (
        tf.reduce_sum(y_true, axis=[1,2,3]) +
        tf.reduce_sum(y_pred, axis=[1,2,3])
    )

    dice_loss = 1 - (
        (2. * intersection + smooth) /
        (union + smooth)
    )

    return tf.reduce_mean(bce) + dice_loss

# ============================================================
# LOAD DATA
# ============================================================

train_imgs, train_masks = load_paths("train")
val_imgs, val_masks     = load_paths("val")

train_gen = data_generator(
    train_imgs,
    train_masks
)

val_gen = data_generator(
    val_imgs,
    val_masks
)

steps_train = len(train_imgs) // BATCH_SIZE
steps_val   = len(val_imgs) // BATCH_SIZE

# ============================================================
# BUILD MODEL
# ============================================================

model = build_attention_unet()

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

        tf.keras.callbacks.EarlyStopping(
            patience=6,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            patience=3
        ),

        tf.keras.callbacks.ModelCheckpoint(
            "attention_best_model.keras",
            save_best_only=True
        )
    ]
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save("attention_final_model.keras")

print("\n✅ ATTENTION U-NET TRAINING DONE")

# ============================================================
# PLOT TRAIN VS VAL
# ============================================================

plt.figure(figsize=(12,5))

# ------------------------------------------------------------
# LOSS
# ------------------------------------------------------------

plt.subplot(1,2,1)

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Val Loss'
)

plt.title("Train vs Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

# ------------------------------------------------------------
# DICE
# ------------------------------------------------------------

plt.subplot(1,2,2)

plt.plot(
    history.history['dice_coef'],
    label='Train Dice'
)

plt.plot(
    history.history['val_dice_coef'],
    label='Val Dice'
)

plt.title("Train vs Validation Dice")

plt.xlabel("Epoch")
plt.ylabel("Dice")

plt.legend()

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

plt.tight_layout()

plt.savefig("attention_unet_training_plot.png")

plt.show()

print("\n✅ Training plots saved")
