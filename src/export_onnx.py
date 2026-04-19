"""
OPTI-FAB — ONNX Export
Converts the trained Keras model to ONNX format for edge deployment.
Compatible with NXP eIQ Toolkit and TensorRT workflows.
"""

import tensorflow as tf
import tf2onnx

from config import (
    MODEL_KERAS, MODEL_ONNX,
    IMG_SIZE, ONNX_OPSET, INPUT_SPEC_NAME,
    get_logger,
)

log = get_logger(__name__)

# =============================================================================
# LOAD MODEL
# =============================================================================

log.info(f"Loading model from {MODEL_KERAS}")
model = tf.keras.models.load_model(MODEL_KERAS, compile=False)
log.info("Model loaded successfully")

# =============================================================================
# DEFINE INPUT SIGNATURE
# =============================================================================

# Input shape: (batch=1, height, width, channels=1 grayscale)
spec = (
    tf.TensorSpec(
        (1, IMG_SIZE, IMG_SIZE, 1),
        tf.float32,
        name=INPUT_SPEC_NAME,
    ),
)

log.info(f"Input spec: {spec}")

# =============================================================================
# EXPORT TO ONNX
# =============================================================================

log.info(f"Exporting to ONNX (opset {ONNX_OPSET})...")

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=ONNX_OPSET,
    output_path=str(MODEL_ONNX),
)

log.info(f"ONNX export successful")
log.info(f"Model saved: {MODEL_ONNX}")
log.info(f"Model size: {MODEL_ONNX.stat().st_size / 1e6:.2f} MB")
