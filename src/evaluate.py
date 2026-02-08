import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = "../models/opti_fab_model.keras"
TEST_DIR = "../dataset/test"
RESULTS_DIR = "../results"

IMG_SIZE = 128
BATCH_SIZE = 16

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="grayscale",
    shuffle=False
)

# Predictions
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# Classification Report
report = classification_report(
    y_true,
    y_pred,
    target_names=list(test_data.class_indices.keys()),
    zero_division=0
)

print(report)

# Save metrics
with open("../results/metrics.txt", "w") as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(range(len(test_data.class_indices)), test_data.class_indices.keys(), rotation=45)
plt.yticks(range(len(test_data.class_indices)), test_data.class_indices.keys())
plt.tight_layout()
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig("../results/confusion_matrix.png")

print("✅ Evaluation complete. Results saved.")
