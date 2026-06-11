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

class PermanentDropout(tf.keras.layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

def load_mc_model(model_path=None):
    """
    Load Keras model and enable dropout at inference time.
    """
    path = model_path or MODEL_KERAS
    log.info(f"Loading model for MC Dropout from {path}")

    base_model = tf.keras.models.load_model(str(path))

    # Rebuild inference graph replacing standard Dropout with PermanentDropout
    inputs = base_model.input
    x = inputs
    for layer in base_model.layers[1:]:
        if isinstance(layer, tf.keras.layers.Dropout):
            x = PermanentDropout(rate=layer.rate)(x)
        else:
            x = layer(x)

    mc_model = tf.keras.Model(inputs=inputs, outputs=x)
    return mc_model

_MC_FUNCTIONS = {}

def predict_with_uncertainty(model, img_array, num_passes=None):
    """
    Run multiple forward passes and return class prediction, confidence, and variance.
    """
    n = num_passes or MC_PASSES

    model_id = id(model)
    if model_id not in _MC_FUNCTIONS:
        log.info("Compiling tf.function for MC Dropout dynamic sequential passes...")
        
        # Specify input signature with passes as a Tensor to prevent re-compilation / unrolling
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, IMG_SIZE, IMG_SIZE, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32)
        ])
        def run_mc_passes_compiled(x, passes):
            preds = tf.TensorArray(dtype=tf.float32, size=passes)
            for i in tf.range(passes):
                p = model(x, training=False)  # BN runs in inference mode
                preds = preds.write(i, p[0])
            return preds.stack()
            
        _MC_FUNCTIONS[model_id] = run_mc_passes_compiled

    compiled_fn = _MC_FUNCTIONS[model_id]
    passes_tensor = tf.constant(n, dtype=tf.int32)
    P = compiled_fn(img_array, passes_tensor).numpy()  # shape: (N, K)

    # Calculate statistics
    mean_preds = np.mean(P, axis=0)
    variance_preds = np.var(P, axis=0)

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

