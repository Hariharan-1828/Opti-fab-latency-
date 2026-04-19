"""
OPTI-FAB — ONNX Graph Inspector
Inspects the exported ONNX model to find nodes that replaced Dropout layers.
Run this first to understand the graph structure before surgery.

Usage:
    python tools/inspect_onnx_graph.py
"""

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

    log.info(f"Total nodes in graph: {len(graph.node)}")
    log.info(f"Inputs : {[i.name for i in graph.input]}")
    log.info(f"Outputs: {[o.name for o in graph.output]}")

    # Count node types
    op_counts = {}
    for node in graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    log.info("\n--- Node type counts ---")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        log.info(f"  {op:<30} {count}")

    # Find all Identity nodes (likely Dropout replacements)
    log.info("\n--- Identity nodes (potential Dropout replacements) ---")
    identity_nodes = [n for n in graph.node if n.op_type == "Identity"]
    for node in identity_nodes:
        log.info(f"  Name: {node.name}")
        log.info(f"  Input : {list(node.input)}")
        log.info(f"  Output: {list(node.output)}")

    # Find any existing Dropout nodes
    log.info("\n--- Existing Dropout nodes ---")
    dropout_nodes = [n for n in graph.node if n.op_type == "Dropout"]
    if dropout_nodes:
        for node in dropout_nodes:
            log.info(f"  Name: {node.name}")
            log.info(f"  Input : {list(node.input)}")
            log.info(f"  Output: {list(node.output)}")
            for attr in node.attribute:
                log.info(f"  Attr  : {attr.name} = {attr.f}")
    else:
        log.info("  None found — Dropout was replaced by Identity during export")

    # Find initializers named with 'dropout' or 'keep'
    log.info("\n--- Initializers with dropout-related names ---")
    found = False
    for init in graph.initializer:
        if any(k in init.name.lower() for k in ["dropout", "keep", "rate"]):
            log.info(f"  {init.name}")
            found = True
    if not found:
        log.info("  None found")

    return identity_nodes, dropout_nodes


if __name__ == "__main__":
    identity_nodes, dropout_nodes = inspect_graph(MODEL_ONNX)
    log.info(f"\nSummary: {len(identity_nodes)} Identity nodes, "
             f"{len(dropout_nodes)} Dropout nodes")
    log.info("Run inject_mc_dropout.py next to perform graph surgery.")
