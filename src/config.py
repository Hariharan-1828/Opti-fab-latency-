"""
Central config for paths, model, training and inference parameters
"""

import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR   = ROOT_DIR / "dataset"
TRAIN_DIR     = DATASET_DIR / "train"
VAL_DIR       = DATASET_DIR / "validation"
TEST_DIR      = DATASET_DIR / "test"

MODEL_DIR     = ROOT_DIR / "models"
RESULTS_DIR   = ROOT_DIR / "results"
LOGS_DIR      = ROOT_DIR / "logs"

MODEL_KERAS   = MODEL_DIR / "opti_fab_model.keras"
MODEL_ONNX    = MODEL_DIR / "opti_fab_model.onnx"
MODEL_TRTENGINE = MODEL_DIR / "opti_fab_model.trt"

# Ensure dirs exist
for d in [MODEL_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Model configuration
IMG_SIZE    = 160
NUM_CLASSES = 8
CLASS_NAMES = [
    "clean",
    "crack",
    "edge_defect",
    "open",
    "other",
    "scratch",
    "short",
    "spot",
]

# Training parameters
BATCH_SIZE      = 16
EPOCHS          = 25
LEARNING_RATE   = 1e-4
DROPOUT_RATE    = 0.4
FINE_TUNE_FROM  = 100      # freeze layers before this index

# Augmentation params
AUG_ROTATION    = 15
AUG_WIDTH_SHIFT = 0.1
AUG_HEIGHT_SHIFT= 0.1
AUG_ZOOM        = 0.15
AUG_SHEAR       = 0.1

# Callbacks patience
EARLY_STOP_PATIENCE  = 5
REDUCE_LR_PATIENCE   = 3
REDUCE_LR_FACTOR     = 0.5

# Inference & streaming
MC_PASSES           = 15
MC_PASSES_FAST      = 5
CONFIDENCE_THRESHOLD = 0.85
UNCERTAINTY_MAX      = 0.05
SCAN_SPEED          = 10
DEFECT_CLASSES      = [i for i, c in enumerate(CLASS_NAMES) if c != "clean"]

# ONNX / TRT Settings
ONNX_OPSET      = 13
INPUT_SPEC_NAME = "input"
TRT_PRECISION   = "fp16"
TRT_WORKSPACE   = 1 << 30

# Logging config
LOG_LEVEL   = "INFO"
LOG_FORMAT  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

def get_logger(name: str):
    import logging
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / f"{name}.log", mode="a"),
        ],
    )
    return logging.getLogger(name)
