# Stream simulator for line-scan camera and early-exit inference

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

def run_stream_simulation(img_path: str) -> dict:
    """
    Simulate row-by-row pixel ingestion and early-exit decision logic.
    """
    log.info(f"Loading image: {img_path}")
    full_image     = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
    full_img_array = img_to_array(full_image) / 255.0

    # Ring buffer for sliding window of scanned rows
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

        # Read rows and update sliding window buffer
        chunk = full_img_array[row : row + SCAN_SPEED, :, :]
        circular_buffer = np.roll(circular_buffer, -SCAN_SPEED, axis=0)
        circular_buffer[-SCAN_SPEED:, :, :] = chunk

        # Wait until buffer has enough rows before running inference
        if row < IMG_SIZE // 2:
            continue

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
            f"Var: {unc:.6f}  "
            f"→ {decision}"
        )

        if decision in ("REJECT", "ACCEPT"):
            results["decision"]     = decision
            results["decision_row"] = row
            log.info(
                f"\nEarly exit at row {row}/{IMG_SIZE} "
                f"({100 * row // IMG_SIZE}% scanned) "
                f"→ Decision: {decision} | Class: {CLASS_NAMES[pred_class]}"
            )
            break

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    results["total_time_ms"] = round(elapsed_ms, 2)

    log.info(f"Stream simulation done in {elapsed_ms:.1f} ms")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OPTI-FAB Stream Simulator")
    parser.add_argument(
        "--img_path",
        type=str,
        default=None,
        help="Path to a test image",
    )
    args = parser.parse_args()

    # Find test image if not specified
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

    model = load_mc_model(MODEL_KERAS)
    results = run_stream_simulation(img_path)

    print("\n--- SUMMARY ---")
    print(f"Image       : {results['img_path']}")
    print(f"Decision    : {results['decision']}")
    print(f"Decision row: {results['decision_row']}")
    print(f"Total time  : {results['total_time_ms']} ms")

