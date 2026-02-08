import tensorflow as tf

NUM_CLASSES = 8
IMG_SIZE = 128

def build_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="input")

    # Convert grayscale → RGB
    x = tf.keras.layers.Conv2D(3, (1,1), name="gray_to_rgb")(inputs)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    model = build_model()

    # LOAD ONLY WEIGHTS (this works even if .h5 model is broken)
    model.load_weights("../models/opti_fab_model.h5")

    # SAVE CLEAN MODEL
    model.save("../models/opti_fab_model_clean.keras")

    print("✅ Model rebuilt and saved as opti_fab_model_clean.keras")
