# Export Keras model to ONNX format

import tensorflow as tf
import tf2onnx

from config import (
    MODEL_KERAS, MODEL_ONNX,
    IMG_SIZE, ONNX_OPSET, INPUT_SPEC_NAME,
    get_logger,
)

log = get_logger(__name__)

# Load trained model
log.info(f"Loading Keras model from {MODEL_KERAS}")
model = tf.keras.models.load_model(MODEL_KERAS, compile=False)

# Define input spec (batch_size=1, height, width, channels=1)
spec = (
    tf.TensorSpec(
        (1, IMG_SIZE, IMG_SIZE, 1),
        tf.float32,
        name=INPUT_SPEC_NAME,
    ),
)

log.info(f"Input spec: {spec}")

# Export using tf2onnx
log.info(f"Converting model to ONNX format (opset {ONNX_OPSET})...")

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=ONNX_OPSET,
    output_path=str(MODEL_ONNX),
)

log.info("ONNX export complete")
log.info(f"Saved: {MODEL_ONNX} ({MODEL_ONNX.stat().st_size / 1e6:.2f} MB)")

