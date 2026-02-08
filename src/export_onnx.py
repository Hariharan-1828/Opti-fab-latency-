import tensorflow as tf
import tf2onnx
import os

# ======================
# CONFIG
# ======================
MODEL_PATH = "../models/opti_fab_model.keras"
ONNX_PATH = "../models/opti_fab_model.onnx"
IMG_SIZE = 160  # IMPORTANT: must match training size

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ======================
# DEFINE INPUT SIGNATURE
# ======================
spec = (tf.TensorSpec((1, IMG_SIZE, IMG_SIZE, 1), tf.float32, name="input"),)

# ======================
# EXPORT TO ONNX
# ======================
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path=ONNX_PATH
)

print("✅ ONNX export successful")
print("📦 Saved at:", ONNX_PATH)
