import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import accuracy_score
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_KERAS, TEST_DIR, IMG_SIZE, CLASS_NAMES, MC_PASSES_FAST
from mc_inference import load_mc_model

@tf.function
def get_mc_predictions(model, x, n: int):
    preds = []
    for _ in range(n):
        p = model(x, training=False)
        preds.append(p[0])
    return tf.stack(preds)

def simulate_with_thresholds(model, test_images, conf_thresh, entropy_thresh):
    y_true = []
    y_pred = []
    early_exits = 0
    exit_rows = []
    
    for class_idx, class_name, img_path in test_images:
        full_image = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
        full_img_array = img_to_array(full_image) / 255.0
        
        circular_buffer = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        exited_early = False
        final_pred_class = None
        decision_row = IMG_SIZE
        
        SCAN_SPEED = 10
        for row in range(0, IMG_SIZE, SCAN_SPEED):
            # Update circular buffer
            chunk = full_img_array[row : row + SCAN_SPEED, :, :]
            circular_buffer = np.roll(circular_buffer, -SCAN_SPEED, axis=0)
            circular_buffer[-SCAN_SPEED:, :, :] = chunk
            
            # Run inference after 50% scanned
            if row >= IMG_SIZE // 2:
                input_tensor = np.expand_dims(circular_buffer, axis=0)
                
                # Single compiled call for all passes
                P = get_mc_predictions(model, input_tensor, MC_PASSES_FAST).numpy()  # shape: (N, K)
                
                mean_preds = np.mean(P, axis=0)
                pred_class = int(np.argmax(mean_preds))
                conf = float(mean_preds[pred_class])
                
                # Calculate Shannon entropy from mean predictions
                p = np.clip(mean_preds, 1e-9, 1.0)
                p = p / np.sum(p)
                entropy = -np.sum(p * np.log(p)) / np.log(len(p))
                
                if conf >= conf_thresh and entropy <= entropy_thresh:
                    final_pred_class = pred_class
                    exited_early = True
                    decision_row = row
                    break
        
        if not exited_early:
            input_tensor = np.expand_dims(full_img_array, axis=0)
            P = get_mc_predictions(model, input_tensor, MC_PASSES_FAST).numpy()
            mean_preds = np.mean(P, axis=0)
            final_pred_class = int(np.argmax(mean_preds))
            
        y_true.append(class_idx)
        y_pred.append(final_pred_class)
        if exited_early:
            early_exits += 1
            exit_rows.append(decision_row)
            
    accuracy = accuracy_score(y_true, y_pred)
    exit_rate = (early_exits / len(test_images)) * 100
    mean_exit_row = np.mean(exit_rows) if exit_rows else 0
    mean_exit_pct = (mean_exit_row / IMG_SIZE) * 100
    
    return accuracy, exit_rate, mean_exit_pct

def main():
    print("Loading model...")
    model = load_mc_model(MODEL_KERAS)
    
    # Pre-load all test image paths
    test_images = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = TEST_DIR / class_name
        if class_dir.exists():
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    test_images.append((class_idx, class_name, class_dir / f))
                    
    print(f"Loaded {len(test_images)} test images.")
    
    # Sweep thresholds
    conf_thresholds = [0.85, 0.90, 0.95, 0.98, 0.99]
    entropy_thresholds = [0.35, 0.20, 0.10, 0.05, 0.02]
    
    print("\nSweep results:")
    print(f"{'Conf Thresh':<12} {'Entropy Thresh':<15} {'Accuracy':<10} {'Early Exit Rate':<18} {'Mean Scan Completion':<22}")
    print("-" * 80)
    
    for ct, et in zip(conf_thresholds, entropy_thresholds):
        acc, er, esc = simulate_with_thresholds(model, test_images, ct, et)
        print(f"{ct:<12.2f} {et:<15.2f} {acc:<10.4f} {er:<18.2f}% {esc:<22.2f}%")

if __name__ == "__main__":
    main()
