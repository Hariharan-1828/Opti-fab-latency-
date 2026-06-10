# Inject stochastic Dropout nodes into ONNX graph to replace Identity nodes

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

def inject_mc_dropout(input_path, output_path, dropout_rate=DROPOUT_RATE):
    """
    Replace Identity nodes from Keras export with real Dropout nodes.
    """
    log.info(f"Loading ONNX model from {input_path}")
    model = onnx.load(str(input_path))
    graph = model.graph

    identity_nodes = [n for n in graph.node if n.op_type == "Identity"]
    log.info(f"Found {len(identity_nodes)} Identity nodes")

    if not identity_nodes:
        log.warning("No Identity nodes found — model may already have Dropout nodes")
        dropout_nodes = [n for n in graph.node if n.op_type == "Dropout"]
        if dropout_nodes:
            log.info(f"Found {len(dropout_nodes)} existing Dropout nodes")
            onnx.save(model, str(output_path))
            return

    new_initializers = []
    new_nodes        = []
    nodes_to_remove  = set()

    for idx, node in enumerate(identity_nodes):
        node_input  = node.input[0]
        node_output = node.output[0]
        suffix      = f"_mc_{idx}"

        # Rate constant
        rate_name = f"mc_dropout_rate{suffix}"
        rate_tensor = numpy_helper.from_array(
            np.array(dropout_rate, dtype=np.float64),
            name=rate_name,
        )
        new_initializers.append(rate_tensor)

        # Training mode constant (True = active dropout)
        training_name = f"mc_training_mode{suffix}"
        training_tensor = numpy_helper.from_array(
            np.array(True, dtype=bool),
            name=training_name,
        )
        new_initializers.append(training_tensor)

        # Dropout node
        dropout_output_name = f"mc_dropout_output{suffix}"
        dropout_node = helper.make_node(
            op_type  = "Dropout",
            inputs   = [node_input, rate_name, training_name],
            outputs  = [dropout_output_name],
            name     = f"mc_dropout{suffix}",
            seed     = np.random.randint(0, 2**16),
        )
        new_nodes.append((node, dropout_node, dropout_output_name, node_output))
        nodes_to_remove.add(id(node))

    # Rebuild graph nodes remapping connections
    output_remap = {}
    for (old_node, new_node, new_out, old_out) in new_nodes:
        output_remap[old_out] = new_out

    new_graph_nodes = []
    for node in graph.node:
        if id(node) in nodes_to_remove:
            for (old_node, new_dropout, new_out, old_out) in new_nodes:
                if id(old_node) == id(node):
                    new_graph_nodes.append(new_dropout)
                    break
        else:
            new_inputs = []
            for inp in node.input:
                new_inputs.append(output_remap.get(inp, inp))
            del node.input[:]
            node.input.extend(new_inputs)
            new_graph_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_graph_nodes)
    graph.initializer.extend(new_initializers)

    for output in graph.output:
        if output.name in output_remap:
            output.name = output_remap[output.name]

    log.info("Checking model...")
    onnx.checker.check_model(model)

    onnx.save(model, str(output_path))
    log.info(f"MC model saved to {output_path}")
    return output_path

def verify_mc_dropout(model_path, num_passes=10):
    import onnxruntime as ort
    log.info(f"Verifying variance of MC Dropout on: {model_path}")

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    input_name  = session.get_inputs()[0].name
    dummy_input = np.random.rand(1, 160, 160, 1).astype(np.float32)

    outputs = []
    for _ in range(num_passes):
        out = session.run(None, {input_name: dummy_input})[0]
        outputs.append(out)

    outputs = np.vstack(outputs)
    variance = np.var(outputs, axis=0)
    mean_var = float(np.mean(variance))

    log.info(f"Mean variance: {mean_var:.8f}")
    return mean_var

if __name__ == "__main__":
    result = inject_mc_dropout(MODEL_ONNX, MC_MODEL_PATH)
    if result:
        verify_mc_dropout(MC_MODEL_PATH)

