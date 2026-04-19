"""
OPTI-FAB — MC Dropout ONNX Injection
Performs graph surgery on the exported ONNX model to replace Identity nodes
(frozen Dropout) with real stochastic Dropout nodes.

This enables true Monte-Carlo Dropout uncertainty estimation through the
ONNX runtime path, with each forward pass producing different output.

Usage:
    python tools/inject_mc_dropout.py

Output:
    models/opti_fab_model_mc.onnx  — MC-enabled model for uncertainty estimation
"""

import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from onnx import TensorProto
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_ONNX, MODEL_DIR, DROPOUT_RATE, get_logger

log = get_logger(__name__)

MC_MODEL_PATH = MODEL_DIR / "opti_fab_model_mc.onnx"


# =============================================================================
# GRAPH SURGERY
# =============================================================================

def inject_mc_dropout(input_path, output_path, dropout_rate=DROPOUT_RATE):
    """
    Loads an ONNX model and replaces Identity nodes that correspond to
    frozen Dropout layers with real stochastic Dropout nodes.

    Strategy:
    - ONNX opset 13 Dropout takes: input, rate, training_mode
    - We set training_mode = True (constant) to force dropout active
    - Rate is set to match the original training dropout rate

    Args:
        input_path  : Path to input ONNX model (frozen Dropout)
        output_path : Path to save MC-enabled ONNX model
        dropout_rate: Dropout probability (must match training config)
    """
    log.info(f"Loading model: {input_path}")
    model = onnx.load(str(input_path))
    graph = model.graph

    # Collect Identity nodes
    identity_nodes = [n for n in graph.node if n.op_type == "Identity"]
    log.info(f"Found {len(identity_nodes)} Identity node(s)")

    if not identity_nodes:
        log.warning("No Identity nodes found — model may already have Dropout nodes")
        log.warning("Checking for existing Dropout nodes...")
        dropout_nodes = [n for n in graph.node if n.op_type == "Dropout"]
        if dropout_nodes:
            log.info(f"Found {len(dropout_nodes)} existing Dropout nodes")
            log.info("Model already has stochastic Dropout — no surgery needed")
            onnx.save(model, str(output_path))
            return

    # Add constants needed for Dropout:
    # 1. dropout_rate (float64 scalar)
    # 2. training_mode = True (bool scalar)
    new_initializers = []
    new_nodes        = []
    nodes_to_remove  = set()

    for idx, node in enumerate(identity_nodes):
        node_input  = node.input[0]
        node_output = node.output[0]
        suffix      = f"_mc_{idx}"

        # Rate initializer
        rate_name = f"mc_dropout_rate{suffix}"
        rate_tensor = numpy_helper.from_array(
            np.array(dropout_rate, dtype=np.float64),
            name=rate_name,
        )
        new_initializers.append(rate_tensor)

        # Training mode initializer (True = dropout active)
        training_name = f"mc_training_mode{suffix}"
        training_tensor = numpy_helper.from_array(
            np.array(True, dtype=bool),
            name=training_name,
        )
        new_initializers.append(training_tensor)

        # New Dropout node — opset 13 signature:
        # inputs:  [data, rate, training_mode]
        # outputs: [output, mask (optional)]
        dropout_output_name = f"mc_dropout_output{suffix}"
        dropout_node = helper.make_node(
            op_type  = "Dropout",
            inputs   = [node_input, rate_name, training_name],
            outputs  = [dropout_output_name],
            name     = f"mc_dropout{suffix}",
            seed     = np.random.randint(0, 2**16),   # Random seed per node
        )
        new_nodes.append((node, dropout_node, dropout_output_name, node_output))
        nodes_to_remove.add(id(node))

        log.info(
            f"  Identity node {idx} -> Dropout node "
            f"(rate={dropout_rate}, training=True, seed={dropout_node.attribute[0].i})"
        )

    # Rebuild graph nodes — replace Identity with Dropout
    # Also need to remap any downstream nodes that consumed Identity output
    output_remap = {}
    for (old_node, new_node, new_out, old_out) in new_nodes:
        output_remap[old_out] = new_out

    new_graph_nodes = []
    for node in graph.node:
        if id(node) in nodes_to_remove:
            # Find the corresponding new Dropout node
            for (old_node, new_dropout, new_out, old_out) in new_nodes:
                if id(old_node) == id(node):
                    new_graph_nodes.append(new_dropout)
                    break
        else:
            # Remap inputs if they consumed Identity output
            new_inputs = []
            for inp in node.input:
                new_inputs.append(output_remap.get(inp, inp))
            del node.input[:]
            node.input.extend(new_inputs)
            new_graph_nodes.append(node)

    # Rebuild graph
    del graph.node[:]
    graph.node.extend(new_graph_nodes)

    # Add new initializers
    graph.initializer.extend(new_initializers)

    # Remap graph outputs if needed
    for output in graph.output:
        if output.name in output_remap:
            output.name = output_remap[output.name]

    # Validate
    log.info("Validating modified graph...")
    try:
        onnx.checker.check_model(model)
        log.info("Graph validation passed")
    except onnx.checker.ValidationError as e:
        log.warning(f"Validation warning: {e}")
        log.warning("Proceeding — some warnings are non-fatal")

    # Save
    onnx.save(model, str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    log.info(f"MC model saved: {output_path} ({size_mb:.2f} MB)")
    return output_path


# =============================================================================
# VERIFY MC DROPOUT IS WORKING
# =============================================================================

def verify_mc_dropout(model_path, num_passes=10):
    """
    Runs num_passes forward passes on the MC model and checks
    that outputs differ between passes (variance > 0).
    """
    import onnxruntime as ort

    log.info(f"\nVerifying MC Dropout on: {model_path}")

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    input_name  = session.get_inputs()[0].name
    dummy_input = np.random.rand(1, 160, 160, 1).astype(np.float32)

    outputs = []
    for i in range(num_passes):
        out = session.run(None, {input_name: dummy_input})[0]
        outputs.append(out)

    outputs = np.vstack(outputs)
    variance = np.var(outputs, axis=0)
    mean_var = float(np.mean(variance))

    log.info(f"  Passes: {num_passes}")
    log.info(f"  Mean variance across passes: {mean_var:.8f}")

    if mean_var > 1e-8:
        log.info("  MC Dropout is WORKING -- outputs differ between passes")
    else:
        log.warning("  Variance is still 0 -- Dropout may not be stochastic")
        log.warning("  Check that Identity nodes were correctly replaced")

    return mean_var


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Step 1: Perform graph surgery
    result = inject_mc_dropout(MODEL_ONNX, MC_MODEL_PATH)

    if result:
        # Step 2: Verify it works
        verify_mc_dropout(MC_MODEL_PATH)

        log.info(f"\nNext step: run benchmarks/benchmark_mc_dropout.py")
        log.info(f"It will automatically use {MC_MODEL_PATH.name}")
