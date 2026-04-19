"""
OPTI-FAB — Stream Simulator
Simulates a line-scan camera feeding pixel data row-by-row into a circular
buffer. Runs MC inference on partial frames and applies confidence-gated
early-exit logic.

This is the core demonstration of OPTI-FAB's latency-first architecture:
decisions are made before the full frame is acquired.

Usage:
    python stream_simulator.py
    python stream_simulator.py --img_path dataset/test/crack/img001.png
"""

import argparse
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from config import (
    MODEL_KERAS, TEST_DIR, RESULTS_DIR,
    IMG_SIZE, SCAN_SPEED, CLASS_NAMES,
    MC_PASSES_FAST, CONFIDENCE_THRESHOLD,
    UNCERTAINTY_MAX, DEFECT_CLASSES,
    get_logger,
)
from mc_inference import load_mc_model, predict_with_uncertainty, apply_decision_gate

log = get_logger(__name__)


# =============================================================================
# STREAM SIMULATION
# =============================================================================

def run_stream_simulation(img_path: str) -> dict:
    """
    Simulates line-scan acquisition and stream-aware inference on a single image.

    Args:
        img_path: Path to a grayscale test image.

    Returns:
        A dict containing timing, decision, and per-tick inference results.
    """
    log.info(f"Loading image: {img_path}")
    full_image     = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
    full_img_array = img_to_array(full_image) / 255.0   # shape: (H, W, 1)

    # Circular buffer — holds the current sliding window of pixel rows
    circular_buffer = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    results = {
        "img_path"      : img_path,
        "ticks"         : [],
        "decision"      : "NONE",
        "decision_row"  : None,
        "total_time_ms" : None,
    }

    start_time  = time.perf_counter()
    tick_count  = 0

    log.info(f"Stream simulation started — scan speed: {SCAN_SPEED} rows/tick")

    for row in range(0, IMG_SIZE, SCAN_SPEED):
        tick_count += 1

        # Ingest the next chunk of pixel rows
        chunk = full_img_array[row : row + SCAN_SPEED, :, :]

        # Roll buffer up and write new rows at the bottom
        circular_buffer          = np.roll(circular_buffer, -SCAN_SPEED, axis=0)
        circular_buffer[-SCAN_SPEED:, :, :] = chunk

        # Wait until the buffer is at least half filled before inferring
        if row < IMG_SIZE // 2:
            log.debug(f"[Tick {tick_count:02d}] Buffering... ({row}/{IMG_SIZE} rows)")
            continue

        # Run MC inference on the current buffer state
        input_tensor = np.expand_dims(circular_buffer, axis=0)
        pred_class, conf, unc = predict_with_uncertainty(
            model, input_tensor, num_passes=MC_PASSES_FAST
        )

        decision = apply_decision_gate(pred_class, conf, unc, DEFECT_CLASSES)

        tick_result = {
            "tick"       : tick_count,
            "row"        : row,
            "class_idx"  : pred_class,
            "class_name" : CLASS_NAMES[pred_class],
            "confidence" : round(conf, 4),
            "uncertainty": round(unc, 6),
            "decision"   : decision,
        }
        results["ticks"].append(tick_result)

        log.info(
            f"[Tick {tick_count:02d} | Row {row:03d}/{IMG_SIZE}] "
            f"Class: {CLASS_NAMES[pred_class]:<12} "
            f"Conf: {conf:.4f}  "
            f"Unc: {unc:.6f}  "
            f"→ {decision}"
        )

        # Early exit on REJECT or ACCEPT
        if decision in ("REJECT", "ACCEPT"):
            results["decision"]     = decision
            results["decision_row"] = row
            log.info(
                f"\n{'='*55}\n"
                f"  EARLY EXIT at row {row}/{IMG_SIZE} "
                f"({100 * row // IMG_SIZE}% of frame scanned)\n"
                f"  Decision  : {decision}\n"
                f"  Class     : {CLASS_NAMES[pred_class]}\n"
                f"  Confidence: {conf:.4f}\n"
                f"  Uncertainty: {unc:.6f}\n"
                f"{'='*55}"
            )
            break

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    results["total_time_ms"] = round(elapsed_ms, 2)

    if results["decision"] == "NONE":
        log.info("No early exit — full frame processed. Standard decision path.")

    log.info(f"Stream simulation complete in {elapsed_ms:.1f} ms")
    return results


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OPTI-FAB Stream Simulator")
    parser.add_argument(
        "--img_path",
        type=str,
        default=None,
        help="Path to a test image. If not provided, picks the first image from TEST_DIR/crack/",
    )
    args = parser.parse_args()

    # Resolve image path
    if args.img_path:
        img_path = args.img_path
    else:
        default_dir = TEST_DIR / "crack"
        candidates  = [
            f for f in os.listdir(default_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not candidates:
            log.error(f"No images found in {default_dir}")
            exit(1)
        img_path = str(default_dir / candidates[0])

    # Load model once
    model = load_mc_model(MODEL_KERAS)

    # Run simulation
    results = run_stream_simulation(img_path)

    # Summary
    log.info("\n--- SUMMARY ---")
    log.info(f"Image       : {results['img_path']}")
    log.info(f"Decision    : {results['decision']}")
    log.info(f"Decision row: {results['decision_row']}")
    log.info(f"Total time  : {results['total_time_ms']} ms")
    log.info(f"Ticks run   : {len(results['ticks'])}")
