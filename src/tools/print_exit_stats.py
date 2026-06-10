import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    MODEL_KERAS, TEST_DIR, IMG_SIZE, CLASS_NAMES,
    MC_PASSES_FAST, CONFIDENCE_THRESHOLD, UNCERTAINTY_MAX,
    DEFECT_CLASSES, get_logger
)
from mc_inference import load_mc_model, predict_with_uncertainty, apply_decision_gate

def main():
    print("Loading model for stream simulation...")
    model = load_mc_model(MODEL_KERAS)
    
    total_images = 0
    correct_predictions = 0
    early_exit_count = 0
    early_exit_rows = []
    
    y_true = []
    y_pred = []
    
    classes = CLASS_NAMES
    
    # We will simulate row-by-row pixel ingestion for each test image
    for class_idx, class_name in enumerate(classes):
        class_dir = TEST_DIR / class_name
        if not class_dir.exists():
            continue
            
        print(f"Simulating class: {class_name}...")
        img_names = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in img_names:
            img_path = class_dir / img_name
            full_image = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
            full_img_array = img_to_array(full_image) / 255.0
            
            circular_buffer = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
            
            final_pred_class = None
            exited_early = False
            decision_row = IMG_SIZE
            
            SCAN_SPEED = 10
            for row in range(0, IMG_SIZE, SCAN_SPEED):
                # Update circular buffer
                chunk = full_img_array[row : row + SCAN_SPEED, :, :]
                circular_buffer = np.roll(circular_buffer, -SCAN_SPEED, axis=0)
                circular_buffer[-SCAN_SPEED:, :, :] = chunk
                
                # Check early exit after 50% scanned
                if row >= IMG_SIZE // 2:
                    input_tensor = np.expand_dims(circular_buffer, axis=0)
                    pred_class, conf, unc = predict_with_uncertainty(model, input_tensor, num_passes=MC_PASSES_FAST)
                    decision = apply_decision_gate(pred_class, conf, unc, DEFECT_CLASSES)
                    
                    if decision in ("REJECT", "ACCEPT"):
                        final_pred_class = pred_class
                        exited_early = True
                        decision_row = row
                        break
            
            # If no early exit, run final inference on the full frame
            if not exited_early:
                input_tensor = np.expand_dims(full_img_array, axis=0)
                pred_class, conf, unc = predict_with_uncertainty(model, input_tensor, num_passes=MC_PASSES_FAST)
                final_pred_class = pred_class
            
            y_true.append(class_idx)
            y_pred.append(final_pred_class)
            
            total_images += 1
            if final_pred_class == class_idx:
                correct_predictions += 1
            
            if exited_early:
                early_exit_count += 1
                early_exit_rows.append(decision_row)
                
    accuracy = correct_predictions / total_images
    early_exit_pct = (early_exit_count / total_images) * 100
    mean_exit_row = np.mean(early_exit_rows) if early_exit_rows else 0
    mean_exit_pct = (mean_exit_row / IMG_SIZE) * 100
    
    print("\n" + "="*50)
    print("STREAMING SIMULATION RESULTS ON TEST SET")
    print(f"Total images evaluated : {total_images}")
    print(f"Overall Accuracy       : {accuracy:.4f}")
    print(f"Early Exit Rate        : {early_exit_pct:.2f}% ({early_exit_count}/{total_images})")
    print(f"Mean Early Exit Row    : {mean_exit_row:.2f} ({mean_exit_pct:.2f}% scanned)")
    print("="*50)
    
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nLaTeX table rows:")
    for i, row_label in enumerate(classes):
        row_str = " & ".join(str(val) for val in cm[i])
        print(f"\\textbf{{{row_label}}} & {row_str} \\\\")
        
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

if __name__ == "__main__":
    main()
