import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_KERAS, TEST_DIR, IMG_SIZE, BATCH_SIZE

def main():
    model = tf.keras.models.load_model(MODEL_KERAS)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_data = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="grayscale",
        shuffle=False,
    )
    predictions = model.predict(test_data, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_data.classes
    cm = confusion_matrix(y_true, y_pred)
    
    classes = list(test_data.class_indices.keys())
    print("Classes:", classes)
    print("Confusion Matrix:")
    print(cm)
    
    # Print in latex format
    print("\nLaTeX table rows:")
    for i, row_label in enumerate(classes):
        row_str = " & ".join(str(val) for val in cm[i])
        print(f"\\textbf{{{row_label}}} & {row_str} \\\\")

if __name__ == "__main__":
    main()
