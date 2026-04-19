"""
OPTI-FAB — MC Dropout ONNX Export
Exports the Keras model to ONNX with Dropout nodes preserved and active.

Standard export optimizes away Dropout entirely.
This script disables that optimization and injects stochastic Dropout
nodes manually into the graph for MC uncertainty estimation.

Output:
    models/opti_fab_model_mc.onnx

Usage:
    python tools/export_mc_onnx.py
"""

import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from onnx import TensorProto
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_KERAS, MODEL_DIR, IMG_SIZE, DROPOUT_RATE, get_logger

log = get_logger(__name__)

MC_MODEL_PATH = MODEL_DIR / "opti_fab_model_mc.onnx"


# =============================================================================
# STEP 1: Export with NO graph optimization (preserves Dropout)
# =============================================================================

def export_unoptimized():
    """
    Exports the Keras model to ONNX with optimizations disabled.
    This preserves Dropout nodes in the graph.
    """
    log.info(f"Loading Keras model: {MODEL_KERAS}")
    model = tf.keras.models.load_model(str(MODEL_KERAS), compile=False)

    spec = (tf.TensorSpec((1, IMG_SIZE, IMG_SIZE, 1), tf.float32, name="input"),)

    log.info("Exporting to ONNX with optimizations DISABLED...")

    # Disable optimizations to preserve Dropout
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=None,        # Don't save yet
        extra_opset=None,
        custom_ops=None,
        custom_op_handlers=None,
        custom_rewriter=None,
        inputs_as_nchw=None,
        large_model=False,
    )

    log.info(f"Unoptimized graph has {len(model_proto.graph.node)} nodes")

    # Check what we got
    op_counts = {}
    for node in model_proto.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    log.info("Node types in unoptimized graph:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        log.info(f"  {op:<30} {count}")

    return model_proto


# =============================================================================
# STEP 2: Inject stochastic Dropout into the graph
# =============================================================================

def inject_dropout_nodes(model_proto, dropout_rate=DROPOUT_RATE):
    """
    Finds Identity/Dropout-like nodes and replaces them with
    stochastic Dropout nodes (training_mode=True).
    """
    graph = model_proto.graph

    # Find candidates: Identity nodes OR any node whose name suggests Dropout
    candidates = []
    for node in graph.node:
        name_lower = node.name.lower()
        if node.op_type == "Identity":
            candidates.append(node)
        elif node.op_type == "Dropout":
            candidates.append(node)
        elif "dropout" in name_lower:
            candidates.append(node)

    log.info(f"Found {len(candidates)} Dropout candidate nodes")

    if not candidates:
        log.warning("No candidates found — will inject Dropout after Dense layer")
        return inject_after_dense(model_proto, dropout_rate)

    new_initializers = []
    dropout_replacements = []

    for idx, node in enumerate(candidates):
        suffix = f"_mc_{idx}"

        rate_name     = f"mc_rate{suffix}"
        training_name = f"mc_training{suffix}"

        rate_init = numpy_helper.from_array(
            np.array(dropout_rate, dtype=np.float64), name=rate_name
        )
        training_init = numpy_helper.from_array(
            np.array(True, dtype=bool), name=training_name
        )

        new_initializers.extend([rate_init, training_init])

        new_node = helper.make_node(
            "Dropout",
            inputs=[node.input[0], rate_name, training_name],
            outputs=[node.output[0]],
            name=f"mc_stochastic_dropout{suffix}",
            seed=int(np.random.randint(0, 65535)),
        )

        dropout_replacements.append((node, new_node))
        log.info(f"  Replacing {node.op_type} '{node.name}' with stochastic Dropout")

    # Apply replacements
    for old_node, new_node in dropout_replacements:
        idx = list(graph.node).index(old_node)
        graph.node.remove(old_node)
        graph.node.insert(idx, new_node)

    graph.initializer.extend(new_initializers)

    return model_proto


# =============================================================================
# STEP 3: Fallback — inject Dropout after the Dense(256) layer
# =============================================================================

def inject_after_dense(model_proto, dropout_rate=DROPOUT_RATE):
    """
    Fallback: directly inserts a Dropout node after the first Dense (Gemm) layer.
    This is the layer that had Dropout(0.4) during training.
    """
    graph = model_proto.graph

    # Find Gemm nodes (Dense layers in ONNX)
    gemm_nodes = [n for n in graph.node if n.op_type == "Gemm"]
    log.info(f"Found {len(gemm_nodes)} Gemm (Dense) nodes")

    if not gemm_nodes:
        log.error("No Gemm nodes found — cannot inject Dropout")
        return model_proto

    # Target: first Gemm node (Dense(256)) — inject Dropout after it
    target_gemm = gemm_nodes[0]
    gemm_output = target_gemm.output[0]

    log.info(f"Injecting Dropout after: {target_gemm.name} (output: {gemm_output})")

    # New intermediate tensor name
    dropout_output = f"{gemm_output}_after_mc_dropout"

    # Rate and training mode constants
    rate_init = numpy_helper.from_array(
        np.array(dropout_rate, dtype=np.float64), name="mc_injected_rate"
    )
    training_init = numpy_helper.from_array(
        np.array(True, dtype=bool), name="mc_injected_training"
    )

    # Create Dropout node
    dropout_node = helper.make_node(
        "Dropout",
        inputs=[gemm_output, "mc_injected_rate", "mc_injected_training"],
        outputs=[dropout_output],
        name="mc_injected_dropout",
        seed=42,
    )

    # Remap: any node that took gemm_output now takes dropout_output
    for node in graph.node:
        if node == target_gemm:
            continue
        new_inputs = [
            dropout_output if inp == gemm_output else inp
            for inp in node.input
        ]
        del node.input[:]
        node.input.extend(new_inputs)

    # Insert Dropout node right after target_gemm
    gemm_idx = list(graph.node).index(target_gemm)
    graph.node.insert(gemm_idx + 1, dropout_node)
    graph.initializer.extend([rate_init, training_init])

    log.info(f"Dropout injected after Dense(256) layer")
    return model_proto


# =============================================================================
# STEP 4: Verify variance > 0
# =============================================================================

def verify_variance(model_path, num_passes=15):
    import onnxruntime as ort

    log.info(f"\nVerifying MC Dropout variance...")
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    input_name  = session.get_inputs()[0].name
    dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)

    outputs = []
    for _ in range(num_passes):
        out = session.run(None, {input_name: dummy_input})[0]
        outputs.append(out)

    outputs  = np.vstack(outputs)
    mean_var = float(np.mean(np.var(outputs, axis=0)))

    log.info(f"  Mean variance across {num_passes} passes: {mean_var:.8f}")

    if mean_var > 1e-8:
        log.info("  MC Dropout WORKING -- variance confirmed > 0")
        return True
    else:
        log.warning("  Variance still 0 -- Dropout not stochastic in ONNX runtime")
        log.warning("  ONNX runtime ignores training_mode on CPU for some opsets")
        log.warning("  Recommendation: use Keras model directly for MC inference")
        return False


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Export unoptimized
    model_proto = export_unoptimized()

    # Inject stochastic Dropout
    model_proto = inject_dropout_nodes(model_proto)

    # Save
    log.info(f"Saving MC model to {MC_MODEL_PATH}...")
    onnx.save(model_proto, str(MC_MODEL_PATH))
    log.info(f"Saved ({MC_MODEL_PATH.stat().st_size / 1e6:.2f} MB)")

    # Verify
    works = verify_variance(MC_MODEL_PATH)

    if not works:
        log.info("\nFalling back to Keras-based MC inference...")
        log.info("See mc_inference.py -- load_mc_model() uses Keras with training=True")
        log.info("This is the correct production approach for uncertainty estimation")
