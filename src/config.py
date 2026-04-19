"""
OPTI-FAB — Central Configuration
All scripts import from here. Change a value once, it updates everywhere.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

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

# Auto-create directories so scripts never fail on missing folders
for _dir in [MODEL_DIR, RESULTS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MODEL
# =============================================================================

IMG_SIZE    = 160          # Single source of truth — was inconsistent across scripts
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

# =============================================================================
# TRAINING
# =============================================================================

BATCH_SIZE      = 16
EPOCHS          = 25
LEARNING_RATE   = 1e-4
DROPOUT_RATE    = 0.4
FINE_TUNE_FROM  = 100      # Freeze MobileNetV2 layers before this index

# Augmentation
AUG_ROTATION    = 15
AUG_WIDTH_SHIFT = 0.1
AUG_HEIGHT_SHIFT= 0.1
AUG_ZOOM        = 0.15
AUG_SHEAR       = 0.1

# Callbacks
EARLY_STOP_PATIENCE  = 5
REDUCE_LR_PATIENCE   = 3
REDUCE_LR_FACTOR     = 0.5

# =============================================================================
# INFERENCE
# =============================================================================

# Monte-Carlo Dropout
MC_PASSES           = 15       # Forward passes for uncertainty estimation
MC_PASSES_FAST      = 5        # Reduced passes for stream simulation

# Confidence-gated decision thresholds
CONFIDENCE_THRESHOLD = 0.85
UNCERTAINTY_MAX      = 0.05

# Stream simulator
SCAN_SPEED          = 10       # Pixel rows per hardware clock tick
DEFECT_CLASSES      = [i for i, c in enumerate(CLASS_NAMES) if c != "clean"]

# =============================================================================
# EXPORT
# =============================================================================

ONNX_OPSET      = 13
INPUT_SPEC_NAME = "input"

# TensorRT
TRT_PRECISION   = "fp16"       # Options: fp32, fp16, int8
TRT_WORKSPACE   = 1 << 30      # 1 GB workspace

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL   = "INFO"
LOG_FORMAT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str):
    """
    Returns a configured logger. Use in every script:

        from config import get_logger
        log = get_logger(__name__)
        log.info("Training started")
    """
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
