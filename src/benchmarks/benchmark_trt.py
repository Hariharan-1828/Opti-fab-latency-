# Benchmark model inference latency across CPU, CUDA, and TensorRT

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
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_ONNX, IMG_SIZE, NUM_CLASSES, get_logger

log = get_logger(__name__)

WARMUP_RUNS  = 50
BENCH_RUNS   = 500
RESULTS_DIR  = Path(__file__).resolve().parent
RESULTS_FILE = RESULTS_DIR / "trt_results.txt"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def benchmark_provider(provider: str, session_options=None) -> dict:
    """
    Inference latency benchmarking for a given ORT provider.
    """
    log.info(f"Benchmarking provider: {provider}")

    providers = [provider, "CPUExecutionProvider"]

    opts = session_options or ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    try:
        session = ort.InferenceSession(
            str(MODEL_ONNX),
            sess_options=opts,
            providers=providers,
        )
    except Exception as e:
        log.error(f"Failed to create session for {provider}: {e}")
        return None

    input_name  = session.get_inputs()[0].name
    dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)

    # Warmup
    log.info(f"  Warming up ({WARMUP_RUNS} runs)...")
    for _ in range(WARMUP_RUNS):
        session.run(None, {input_name: dummy_input})

    # Benchmark
    log.info(f"  Benchmarking ({BENCH_RUNS} runs)...")
    latencies = []
    for _ in range(BENCH_RUNS):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    latencies = np.array(latencies)

    result = {
        "provider"   : provider,
        "mean_ms"    : round(float(np.mean(latencies)),   2),
        "median_ms"  : round(float(np.median(latencies)), 2),
        "p95_ms"     : round(float(np.percentile(latencies, 95)), 2),
        "p99_ms"     : round(float(np.percentile(latencies, 99)), 2),
        "min_ms"     : round(float(np.min(latencies)),    2),
        "max_ms"     : round(float(np.max(latencies)),    2),
        "throughput" : round(1000.0 / float(np.mean(latencies)), 1),
    }

    log.info(
        f"  Done — mean: {result['mean_ms']} ms | "
        f"p95: {result['p95_ms']} ms | "
        f"throughput: {result['throughput']} fps"
    )
    return result

def get_trt_provider_options():
    trt_cache = str(RESULTS_DIR / "trt_engine_cache")
    Path(trt_cache).mkdir(exist_ok=True)
    return {
        "trt_max_workspace_size"    : 1 << 30,
        "trt_fp16_enable"           : True,
        "trt_engine_cache_enable"   : True,
        "trt_engine_cache_path"     : trt_cache,
        "trt_max_batch_size"        : 1,
    }

def run_benchmark():
    if not MODEL_ONNX.exists():
        log.error(f"ONNX model not found at {MODEL_ONNX}")
        return

    log.info(f"Model: {MODEL_ONNX}")
    log.info(f"Input: (1, {IMG_SIZE}, {IMG_SIZE}, 1)")
    log.info(f"Warm-up runs: {WARMUP_RUNS} | Benchmark runs: {BENCH_RUNS}")

    results = []

    # 1. CPU
    cpu_result = benchmark_provider("CPUExecutionProvider")
    if cpu_result:
        results.append(cpu_result)

    # 2. CUDA
    cuda_result = benchmark_provider("CUDAExecutionProvider")
    if cuda_result:
        results.append(cuda_result)

    # 3. TensorRT
    log.info("TensorRT compilation (first run builds engine cache)...")
    trt_opts = ort.SessionOptions()
    trt_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    trt_result = benchmark_provider(
        "TensorrtExecutionProvider",
        session_options=trt_opts,
    )
    if trt_result:
        results.append(trt_result)

    # Print & Save Results
    header = (
        f"\n{'='*70}\n"
        f"  Inference Benchmark Results\n"
        f"  Hardware : NVIDIA RTX 4050 Laptop GPU\n"
        f"  Model    : MobileNetV2 (ONNX, {IMG_SIZE}x{IMG_SIZE} grayscale)\n"
        f"  Runs     : {BENCH_RUNS} (after {WARMUP_RUNS} warm-up)\n"
        f"{'='*70}\n"
    )

    col_w = 22
    table  = f"  {'Provider':<{col_w}} {'Mean':>8} {'Median':>8} {'P95':>8} {'P99':>8} {'FPS':>8}\n"
    table += f"  {'-'*col_w} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'':>8}\n"

    for r in results:
        name = r["provider"].replace("ExecutionProvider", "")
        table += (
            f"  {name:<{col_w}} "
            f"{r['mean_ms']:>8} "
            f"{r['median_ms']:>8} "
            f"{r['p95_ms']:>8} "
            f"{r['p99_ms']:>8} "
            f"{r['throughput']:>8}\n"
        )

    # Speedup vs CPU
    if len(results) >= 2:
        cpu_mean = results[0]["mean_ms"]
        table += f"\n  Speedup vs CPU baseline:\n"
        for r in results[1:]:
            name    = r["provider"].replace("ExecutionProvider", "")
            speedup = round(cpu_mean / r["mean_ms"], 2)
            table  += f"    {name}: {speedup}x faster\n"

    footer = f"\n{'='*70}\n"
    output = header + table + footer

    print(output)

    with open(RESULTS_FILE, "w") as f:
        f.write(output)

    log.info(f"Results saved: {RESULTS_FILE}")

if __name__ == "__main__":
    run_benchmark()

