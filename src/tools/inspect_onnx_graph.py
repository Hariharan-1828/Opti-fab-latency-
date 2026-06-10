# Helper script to inspect ONNX model nodes

import onnx
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_ONNX, get_logger

log = get_logger(__name__)

def inspect_graph(model_path):
    log.info(f"Loading ONNX model: {model_path}")
    model = onnx.load(str(model_path))
    graph = model.graph

    log.info(f"Total nodes: {len(graph.node)}")
    log.info(f"Inputs: {[i.name for i in graph.input]}")
    log.info(f"Outputs: {[o.name for o in graph.output]}")

    # Count ops
    op_counts = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    print("\n--- Op Counts ---")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"  {op:<20} {count}")

    # Find Identity nodes (likely dropout candidates)
    print("\n--- Identity Nodes ---")
    identity_nodes = [n for n in graph.node if n.op_type == "Identity"]
    for node in identity_nodes:
        print(f"  Name: {node.name} | Input: {list(node.input)} | Output: {list(node.output)}")

    # Check for Dropout nodes
    print("\n--- Dropout Nodes ---")
    dropout_nodes = [n for n in graph.node if n.op_type == "Dropout"]
    for node in dropout_nodes:
        print(f"  Name: {node.name} | Input: {list(node.input)} | Output: {list(node.output)}")

    return identity_nodes, dropout_nodes

if __name__ == "__main__":
    identity_nodes, dropout_nodes = inspect_graph(MODEL_ONNX)

