# Export Keras model to ONNX with Dropout active for MC uncertainty estimation

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

def export_unoptimized():
    """
    Export without optimization to keep Dropout layers.
    """
    log.info(f"Loading Keras model: {MODEL_KERAS}")
    model = tf.keras.models.load_model(str(MODEL_KERAS), compile=False)

    spec = (tf.TensorSpec((1, IMG_SIZE, IMG_SIZE, 1), tf.float32, name="input"),)

    log.info("Exporting unoptimized ONNX...")
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=None,
    )

    log.info(f"Nodes in unoptimized graph: {len(model_proto.graph.node)}")
    return model_proto

def inject_dropout_nodes(model_proto, dropout_rate=DROPOUT_RATE):
    """
    Replace Identity nodes with active Dropout nodes.
    """
    graph = model_proto.graph

    candidates = []
    for node in graph.node:
        name_lower = node.name.lower()
        if node.op_type in ("Identity", "Dropout") or "dropout" in name_lower:
            candidates.append(node)

    log.info(f"Found {len(candidates)} Dropout candidate nodes")

    if not candidates:
        log.warning("No candidates found, injecting after Dense layer")
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
        log.info(f"Replacing {node.op_type} '{node.name}' with active Dropout")

    for old_node, new_node in dropout_replacements:
        idx = list(graph.node).index(old_node)
        graph.node.remove(old_node)
        graph.node.insert(idx, new_node)

    graph.initializer.extend(new_initializers)
    return model_proto

def inject_after_dense(model_proto, dropout_rate=DROPOUT_RATE):
    """
    Inject active Dropout node after first Gemm (Dense) layer as fallback.
    """
    graph = model_proto.graph

    gemm_nodes = [n for n in graph.node if n.op_type == "Gemm"]
    log.info(f"Found {len(gemm_nodes)} Gemm nodes")

    if not gemm_nodes:
        log.error("No Gemm nodes found")
        return model_proto

    target_gemm = gemm_nodes[0]
    gemm_output = target_gemm.output[0]

    log.info(f"Injecting Dropout after {target_gemm.name}")
    dropout_output = f"{gemm_output}_after_mc_dropout"

    rate_init = numpy_helper.from_array(
        np.array(dropout_rate, dtype=np.float64), name="mc_injected_rate"
    )
    training_init = numpy_helper.from_array(
        np.array(True, dtype=bool), name="mc_injected_training"
    )

    dropout_node = helper.make_node(
        "Dropout",
        inputs=[gemm_output, "mc_injected_rate", "mc_injected_training"],
        outputs=[dropout_output],
        name="mc_injected_dropout",
        seed=42,
    )

    for node in graph.node:
        if node == target_gemm:
            continue
        new_inputs = [
            dropout_output if inp == gemm_output else inp
            for inp in node.input
        ]
        del node.input[:]
        node.input.extend(new_inputs)

    gemm_idx = list(graph.node).index(target_gemm)
    graph.node.insert(gemm_idx + 1, dropout_node)
    graph.initializer.extend([rate_init, training_init])

    return model_proto

def verify_variance(model_path, num_passes=15):
    import onnxruntime as ort

    log.info(f"Verifying variance on: {model_path}")
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

    log.info(f"Mean variance: {mean_var:.8f}")
    if mean_var > 1e-8:
        log.info("MC Dropout active in ONNX runtime")
        return True
    else:
        log.warning("Variance is 0 - ORT might ignore training_mode on CPU")
        return False

if __name__ == "__main__":
    model_proto = export_unoptimized()
    model_proto = inject_dropout_nodes(model_proto)

    log.info(f"Saving to {MC_MODEL_PATH}...")
    onnx.save(model_proto, str(MC_MODEL_PATH))

    verify_variance(MC_MODEL_PATH)

