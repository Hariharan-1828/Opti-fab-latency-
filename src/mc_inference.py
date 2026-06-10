# Monte-Carlo Dropout inference and decision gating

import time
import numpy as np
import tensorflow as tf

from config import (
    MODEL_KERAS, IMG_SIZE,
    MC_PASSES, CONFIDENCE_THRESHOLD, UNCERTAINTY_MAX,
    get_logger,
)

log = get_logger(__name__)

def load_mc_model(model_path=None):
    """
    Load Keras model and enable dropout at inference time.
    """
    path = model_path or MODEL_KERAS
    log.info(f"Loading model for MC Dropout from {path}")

    base_model = tf.keras.models.load_model(str(path))

    # Force dropout layers to stay active during inference
    inputs = base_model.input
    x = inputs
    for layer in base_model.layers[1:]:
        if isinstance(layer, tf.keras.layers.Dropout):
            x = layer(x, training=True)
        else:
            x = layer(x)

    mc_model = tf.keras.Model(inputs=inputs, outputs=x)
    return mc_model

def predict_with_uncertainty(model, img_array, num_passes=None):
    """
    Run multiple forward passes and return class prediction, confidence, and variance.
    """
    n = num_passes or MC_PASSES

    # Batched forward pass for efficiency
    batch = np.repeat(img_array, n, axis=0)
    predictions = model.predict(batch, verbose=0)

    # Calculate statistics
    mean_preds = np.mean(predictions, axis=0)
    variance_preds = np.var(predictions, axis=0)

    pred_class = int(np.argmax(mean_preds))
    confidence = float(mean_preds[pred_class])
    uncertainty = float(variance_preds[pred_class])

    return pred_class, confidence, uncertainty

def apply_decision_gate(pred_class, confidence, uncertainty, defect_classes):
    """
    Threshold decision logic based on confidence and prediction variance.
    """
    if confidence >= CONFIDENCE_THRESHOLD:
        if uncertainty <= UNCERTAINTY_MAX:
            if pred_class in defect_classes:
                return "REJECT"
            else:
                return "ACCEPT"
        else:
            return "DEFER"
    return "CONTINUE"

if __name__ == "__main__":
    model = load_mc_model()
    dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)

    # Warmup
    predict_with_uncertainty(model, dummy_input, num_passes=1)

    start = time.perf_counter()
    cls, conf, unc = predict_with_uncertainty(model, dummy_input)
    elapsed = time.perf_counter() - start

    log.info(f"Inference time ({MC_PASSES} passes): {elapsed * 1000:.1f} ms")
    log.info(f"Class: {cls} | Conf: {conf:.4f} | Var: {unc:.6f}")

