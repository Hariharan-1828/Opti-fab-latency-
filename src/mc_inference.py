"""
OPTI-FAB — Monte-Carlo Dropout Inference
Uncertainty-aware inference using MC Dropout.

Usage:
    from mc_inference import load_mc_model, predict_with_uncertainty

Decision logic:
    High confidence + low uncertainty  → immediate inline decision
    High confidence + high uncertainty → deferred review
    Low confidence                     → fallback full-frame inference
"""

import time
import numpy as np
import tensorflow as tf

from config import (
    MODEL_KERAS, IMG_SIZE,
    MC_PASSES, CONFIDENCE_THRESHOLD, UNCERTAINTY_MAX,
    get_logger,
)

log = get_logger(__name__)


# =============================================================================
# MODEL LOADER
# =============================================================================

def load_mc_model(model_path=None):
    """
    Loads a Keras model with all Dropout layers forced active (training=True).
    This enables Monte-Carlo Dropout for uncertainty estimation at inference time.

    Args:
        model_path: Path to .keras model file. Defaults to config.MODEL_KERAS.

    Returns:
        A Keras model with MC Dropout enabled.
    """
    path = model_path or MODEL_KERAS
    log.info(f"Loading MC model from {path}")

    base_model = tf.keras.models.load_model(str(path))

    # Rebuild inference graph with Dropout forced ON
    inputs = base_model.input

    x = inputs
    for layer in base_model.layers[1:]:          # skip Input layer
        if isinstance(layer, tf.keras.layers.Dropout):
            x = layer(x, training=True)           # force dropout active
        else:
            x = layer(x)

    mc_model = tf.keras.Model(inputs=inputs, outputs=x)

    log.info(f"MC model ready — dropout active during inference")
    return mc_model


# =============================================================================
# UNCERTAINTY ESTIMATION
# =============================================================================

def predict_with_uncertainty(model, img_array, num_passes=None):
    """
    Runs N stochastic forward passes and returns mean prediction + variance.

    Args:
        model:      MC-enabled Keras model from load_mc_model()
        img_array:  Input array of shape (1, H, W, 1), float32, range [0, 1]
        num_passes: Number of MC passes. Defaults to config.MC_PASSES.

    Returns:
        pred_class  (int)   — index of predicted class
        confidence  (float) — mean softmax probability of predicted class
        uncertainty (float) — variance of predicted class across passes
    """
    n = num_passes or MC_PASSES

    # Tile the single input into a batch of N identical copies
    batch = np.repeat(img_array, n, axis=0)

    # Run all N passes in one batched call (efficient on GPU)
    predictions = model.predict(batch, verbose=0)   # shape: (N, num_classes)

    # Aggregate across passes
    mean_preds     = np.mean(predictions, axis=0)   # shape: (num_classes,)
    variance_preds = np.var(predictions,  axis=0)   # shape: (num_classes,)

    pred_class  = int(np.argmax(mean_preds))
    confidence  = float(mean_preds[pred_class])
    uncertainty = float(variance_preds[pred_class])

    return pred_class, confidence, uncertainty


# =============================================================================
# DECISION GATE
# =============================================================================

def apply_decision_gate(pred_class, confidence, uncertainty, defect_classes):
    """
    Applies confidence-gated decision logic.

    Returns:
        "REJECT"   — defect detected with high confidence, low uncertainty
        "DEFER"    — high confidence but high uncertainty, needs review
        "CONTINUE" — confidence too low, keep streaming
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


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    model = load_mc_model()
    log.info("Model loaded with MC Dropout enabled")

    dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)

    # Warm-up pass (first call is slower due to TF graph compilation)
    predict_with_uncertainty(model, dummy_input, num_passes=1)

    start = time.perf_counter()
    cls, conf, unc = predict_with_uncertainty(model, dummy_input)
    elapsed = time.perf_counter() - start

    log.info(f"Inference time ({MC_PASSES} passes): {elapsed * 1000:.1f} ms")
    log.info(f"Predicted class : {cls}")
    log.info(f"Confidence      : {conf:.4f}")
    log.info(f"Uncertainty     : {unc:.6f}")
    log.info(f"Threshold check : conf>={CONFIDENCE_THRESHOLD} → {conf >= CONFIDENCE_THRESHOLD}")
    log.info(f"Uncertainty check: unc<={UNCERTAINTY_MAX} → {unc <= UNCERTAINTY_MAX}")
