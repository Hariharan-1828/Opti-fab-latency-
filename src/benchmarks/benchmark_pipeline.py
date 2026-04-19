"""
OPTI-FAB — Pipeline Benchmark
Measures and compares end-to-end latency between:
  1. File-based pipeline  (traditional: save → load → infer)
  2. Stream pipeline      (OPTI-FAB: circular buffer → infer)

This is the core evidence for the 41% latency reduction claim.
Run this AFTER benchmark_trt.py so TRT engine is already cached.

Usage:
    python benchmarks/benchmark_pipeline.py
"""

import os

# Load NVIDIA DLLs before importing onnxruntime
_PYTHON = r"C:\Users\harih\AppData\Local\Programs\Python\Python310\Lib\site-packages"
os.add_dll_directory(rf"{_PYTHON}\tensorrt_libs")
os.add_dll_directory(rf"{_PYTHON}\nvidia\cudnn\bin")
os.add_dll_directory(rf"{_PYTHON}\nvidia\cublas\bin")
os.add_dll_directory(rf"{_PYTHON}\nvidia\cuda_nvrtc\bin")

import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from PIL import Image
import tempfile
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    MODEL_ONNX, TEST_DIR, IMG_SIZE,
    SCAN_SPEED, CONFIDENCE_THRESHOLD, UNCERTAINTY_MAX,
    get_logger,
)

log        = get_logger(__name__)
BENCH_RUNS = 100
RESULTS_DIR  = Path(__file__).resolve().parent
RESULTS_FILE = RESULTS_DIR / "pipeline_results.txt"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SHARED: ORT SESSION
# =============================================================================

def get_session():
    """Returns a TensorRT-accelerated onnxruntime session."""
    trt_cache = str(RESULTS_DIR / "trt_engine_cache")
    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_max_workspace_size" : 1 << 30,
                "trt_fp16_enable"        : True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path"  : trt_cache,
            },
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    return ort.InferenceSession(str(MODEL_ONNX), providers=providers)


# =============================================================================
# PIPELINE 1: FILE-BASED (traditional)
# =============================================================================

def file_based_pipeline(session, img_array: np.ndarray) -> float:
    """
    Simulates traditional file-based pipeline:
      1. Save image to disk as PNG
      2. OS flush (simulated with os.fsync)
      3. Read image back from disk
      4. Preprocess
      5. Run inference
    Returns end-to-end time in ms.
    """
    input_name = session.get_inputs()[0].name

    t0 = time.perf_counter()

    # Step 1: Save to disk
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    img_pil = Image.fromarray((img_array[:, :, 0] * 255).astype(np.uint8), mode="L")
    img_pil.save(tmp_path)

    # Step 2: OS flush
    with open(tmp_path, "ab") as f:
        f.flush()
        os.fsync(f.fileno())

    # Step 3: Read back from disk
    img_loaded = Image.open(tmp_path).convert("L")
    img_np     = np.array(img_loaded, dtype=np.float32) / 255.0
    img_np     = img_np[np.newaxis, :, :, np.newaxis]  # (1, H, W, 1)

    # Step 4: Inference
    session.run(None, {input_name: img_np})

    t1 = time.perf_counter()

    # Cleanup
    os.unlink(tmp_path)

    return (t1 - t0) * 1000  # ms


# =============================================================================
# PIPELINE 2: STREAM-BASED (OPTI-FAB)
# =============================================================================

def stream_pipeline(session, img_array: np.ndarray) -> float:
    """
    Simulates OPTI-FAB stream pipeline:
      1. Feed pixel rows into circular buffer as they arrive
      2. Begin inference once buffer is half full
      3. Apply confidence gate — exit early if threshold met
    Returns end-to-end time in ms.
    """
    input_name      = session.get_inputs()[0].name
    circular_buffer = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    t0 = time.perf_counter()

    for row in range(0, IMG_SIZE, SCAN_SPEED):
        chunk = img_array[row: row + SCAN_SPEED, :, :]
        circular_buffer          = np.roll(circular_buffer, -SCAN_SPEED, axis=0)
        circular_buffer[-SCAN_SPEED:, :, :] = chunk

        if row < IMG_SIZE // 2:
            continue

        input_tensor = circular_buffer[np.newaxis, ...]
        preds        = session.run(None, {input_name: input_tensor})[0][0]

        confidence = float(np.max(preds))
        if confidence >= CONFIDENCE_THRESHOLD:
            break   # Early exit

    t1 = time.perf_counter()
    return (t1 - t0) * 1000  # ms


# =============================================================================
# MAIN
# =============================================================================

def run_benchmark():
    if not MODEL_ONNX.exists():
        log.error(f"ONNX model not found: {MODEL_ONNX}")
        return

    # Find test images
    test_images = []
    for class_dir in TEST_DIR.iterdir():
        if class_dir.is_dir():
            for img_file in class_dir.glob("*.png"):
                test_images.append(img_file)
            for img_file in class_dir.glob("*.jpg"):
                test_images.append(img_file)

    if not test_images:
        log.error(f"No test images found in {TEST_DIR}")
        return

    log.info(f"Found {len(test_images)} test images")
    log.info(f"Benchmark runs per pipeline: {BENCH_RUNS}")

    session = get_session()
    log.info("Session ready")

    # Use a fixed dummy array if not enough real images
    dummy = np.random.rand(IMG_SIZE, IMG_SIZE, 1).astype(np.float32)

    def get_img(idx):
        if idx < len(test_images):
            img = Image.open(test_images[idx % len(test_images)]).convert("L")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img, dtype=np.float32) / 255.0
            return arr[:, :, np.newaxis]
        return dummy

    # Warm up both pipelines
    log.info("Warming up...")
    for i in range(10):
        img = get_img(i)
        file_based_pipeline(session, img)
        stream_pipeline(session, img)

    # Benchmark file-based
    log.info("Benchmarking file-based pipeline...")
    file_latencies = []
    for i in range(BENCH_RUNS):
        img = get_img(i)
        file_latencies.append(file_based_pipeline(session, img))

    # Benchmark stream
    log.info("Benchmarking stream pipeline...")
    stream_latencies = []
    for i in range(BENCH_RUNS):
        img = get_img(i)
        stream_latencies.append(stream_pipeline(session, img))

    file_latencies   = np.array(file_latencies)
    stream_latencies = np.array(stream_latencies)

    file_mean   = round(float(np.mean(file_latencies)),   2)
    stream_mean = round(float(np.mean(stream_latencies)), 2)
    reduction   = round((1 - stream_mean / file_mean) * 100, 1)
    speedup     = round(file_mean / stream_mean, 2)

    # ==========================================================================
    # RESULTS TABLE
    # ==========================================================================

    output = (
        f"\n{'='*65}\n"
        f"  OPTI-FAB — Pipeline Latency Benchmark\n"
        f"  Hardware : NVIDIA RTX 4050 Laptop GPU\n"
        f"  Model    : MobileNetV2 ONNX + TensorRT FP16\n"
        f"  Runs     : {BENCH_RUNS} per pipeline\n"
        f"{'='*65}\n\n"
        f"  {'Metric':<30} {'File-Based':>12} {'OPTI-FAB':>12}\n"
        f"  {'-'*54}\n"
        f"  {'Mean latency (ms)':<30} {file_mean:>12} {stream_mean:>12}\n"
        f"  {'Median latency (ms)':<30} {round(float(np.median(file_latencies)),2):>12} {round(float(np.median(stream_latencies)),2):>12}\n"
        f"  {'P95 latency (ms)':<30} {round(float(np.percentile(file_latencies,95)),2):>12} {round(float(np.percentile(stream_latencies,95)),2):>12}\n"
        f"  {'Min latency (ms)':<30} {round(float(np.min(file_latencies)),2):>12} {round(float(np.min(stream_latencies)),2):>12}\n"
        f"  {'Throughput (fps)':<30} {round(1000/file_mean,1):>12} {round(1000/stream_mean,1):>12}\n"
        f"\n"
        f"  Latency reduction : {reduction}%\n"
        f"  Speedup           : {speedup}x\n"
        f"{'='*65}\n"
    )

    print(output)

    with open(RESULTS_FILE, "w") as f:
        f.write(output)

    log.info(f"Results saved: {RESULTS_FILE}")


if __name__ == "__main__":
    run_benchmark()
