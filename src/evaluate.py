"""
OPTI-FAB — Model Evaluation
Generates classification report and confusion matrix on the test set.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    MODEL_KERAS, TEST_DIR, RESULTS_DIR,
    IMG_SIZE, BATCH_SIZE, CLASS_NAMES,
    get_logger,
)

log = get_logger(__name__)

# =============================================================================
# LOAD MODEL & DATA
# =============================================================================

log.info(f"Loading model from {MODEL_KERAS}")
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

log.info(f"Test samples: {test_data.samples}")
log.info(f"Class indices: {test_data.class_indices}")

# =============================================================================
# PREDICTIONS
# =============================================================================

log.info("Running predictions...")
predictions = model.predict(test_data, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# =============================================================================
# CLASSIFICATION REPORT
# =============================================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=list(test_data.class_indices.keys()),
    zero_division=0,
)

print(report)

metrics_path = RESULTS_DIR / "metrics.txt"
with open(metrics_path, "w") as f:
    f.write(report)

log.info(f"Metrics saved: {metrics_path}")

# =============================================================================
# CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(y_true, y_pred)
class_labels = list(test_data.class_indices.keys())

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
fig.colorbar(im, ax=ax)

ax.set(
    title="Confusion Matrix — OPTI-FAB",
    xlabel="Predicted Label",
    ylabel="True Label",
    xticks=range(len(class_labels)),
    yticks=range(len(class_labels)),
    xticklabels=class_labels,
    yticklabels=class_labels,
)
plt.xticks(rotation=45, ha="right")

# Annotate cells with counts
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, format(cm[i, j], "d"),
            ha="center", va="center",
            color="white" if cm[i, j] < thresh else "black",
            fontsize=9,
        )

plt.tight_layout()

cm_path = RESULTS_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=150)
log.info(f"Confusion matrix saved: {cm_path}")

plt.show()
log.info("Evaluation complete.")
