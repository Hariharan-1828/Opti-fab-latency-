import tensorflow as tf

# Load model WITHOUT compiling
model = tf.keras.models.load_model(
    "../models/opti_fab_model.h5",
    compile=False
)

# Re-save in modern Keras format
model.save("../models/opti_fab_model.keras")

print("✅ Model re-saved in .keras format")
