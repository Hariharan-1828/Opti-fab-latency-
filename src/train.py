# Model training using MobileNetV2 for wafer defect classification

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

from config import (
    TRAIN_DIR, VAL_DIR, MODEL_KERAS,
    IMG_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    DROPOUT_RATE, FINE_TUNE_FROM,
    AUG_ROTATION, AUG_WIDTH_SHIFT, AUG_HEIGHT_SHIFT,
    AUG_ZOOM, AUG_SHEAR,
    EARLY_STOP_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
    get_logger,
)

log = get_logger(__name__)

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=AUG_ROTATION,
    width_shift_range=AUG_WIDTH_SHIFT,
    height_shift_range=AUG_HEIGHT_SHIFT,
    zoom_range=AUG_ZOOM,
    shear_range=AUG_SHEAR,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

log.info(f"Classes: {train_generator.class_indices}")
log.info(f"Train samples: {train_generator.samples}")
log.info(f"Val samples: {val_generator.samples}")

NUM_CLASSES = train_generator.num_classes

# Compute class weights to handle imbalance
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_generator.classes),
    y=train_generator.classes,
)
class_weights = dict(enumerate(class_weights))
log.info(f"Class weights: {class_weights}")

# Build model
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input")

# Map grayscale to 3 channels for MobileNetV2
x = layers.Conv2D(3, (1, 1), padding="same", name="gray_to_rgb")(inputs)

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = True

# Freeze base layers up to FINE_TUNE_FROM
for layer in base_model.layers[:FINE_TUNE_FROM]:
    layer.trainable = False

log.info(
    f"Trainable layers: {sum(1 for l in base_model.layers if l.trainable)} / "
    f"{len(base_model.layers)}"
)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(DROPOUT_RATE)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary(print_fn=log.info)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=REDUCE_LR_FACTOR,
        patience=REDUCE_LR_PATIENCE,
        verbose=1,
    ),
]

# Train
log.info("Starting training...")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

# Save
model.save(MODEL_KERAS)
log.info(f"Model saved to {MODEL_KERAS}")

