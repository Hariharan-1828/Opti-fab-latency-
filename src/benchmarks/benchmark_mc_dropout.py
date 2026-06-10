# Profile Monte-Carlo Dropout latency/uncertainty tradeoff using Keras model

import os
os.add_dll_directory(r"C:\Users\harih\AppData\Local\Programs\Python\Python310\Lib\site-packages\tensorrt_libs")
os.add_dll_directory(r"C:\Users\harih\AppData\Local\Programs\Python\Python310\Lib\site-packages\nvidia\cudnn\bin")
os.add_dll_directory(r"C:\Users\harih\AppData\Local\Programs\Python\Python310\Lib\site-packages\nvidia\cublas\bin")
os.add_dll_directory(r"C:\Users\harih\AppData\Local\Programs\Python\Python310\Lib\site-packages\nvidia\cuda_nvrtc\bin")

import time
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MODEL_KERAS, IMG_SIZE, get_logger
from mc_inference import run_mc_passes

log = get_logger(__name__)

BENCH_RUNS   = 50
PASS_COUNTS  = [1, 5, 10, 15, 20, 30]
RESULTS_DIR  = Path(__file__).resolve().parent
RESULTS_FILE = RESULTS_DIR / "mc_dropout_results.txt"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_mc_model():
    log.info(f"Loading Keras model: {MODEL_KERAS}")
    base_model = tf.keras.models.load_model(str(MODEL_KERAS))

    inputs = base_model.input
    x = inputs
    for layer in base_model.layers[1:]:
        if isinstance(layer, tf.keras.layers.Dropout):
            x = layer(x, training=True)
            log.info(f"  Dropout layer active: {layer.name}")
        else:
            x = layer(x)

    mc_model = tf.keras.Model(inputs=inputs, outputs=x)
    return mc_model

def benchmark_mc_passes(model, num_passes: int) -> dict:
    dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 1).astype(np.float32)

    # Warmup
    run_mc_passes(model, dummy_input, num_passes)

    latencies = []
    variances = []

    for _ in range(BENCH_RUNS):
        t0    = time.perf_counter()
        preds = run_mc_passes(model, dummy_input, num_passes).numpy()
        t1 = time.perf_counter()

        mean_var = float(np.mean(np.var(preds, axis=0)))
        latencies.append((t1 - t0) * 1000)
        variances.append(mean_var)

    return {
        "passes"       : num_passes,
        "mean_ms"      : round(float(np.mean(latencies)),   2),
        "p95_ms"       : round(float(np.percentile(latencies, 95)), 2),
        "mean_variance": round(float(np.mean(variances)),   8),
        "throughput"   : round(1000.0 / float(np.mean(latencies)), 1),
    }

def run_benchmark():
    model = load_mc_model()

    results = []
    for n in PASS_COUNTS:
        log.info(f"Benchmarking {n} passes...")
        r = benchmark_mc_passes(model, n)
        results.append(r)
        log.info(f"  {n} passes -> {r['mean_ms']} ms | variance: {r['mean_variance']:.8f}")

    base_variance = results[-1]["mean_variance"]

    output = (
        f"\n{'='*70}\n"
        f"  MC Dropout Latency / Uncertainty Tradeoff\n"
        f"  Hardware : NVIDIA RTX 4050 Laptop GPU\n"
        f"  Model    : MobileNetV2 Keras (Dropout active)\n"
        f"  Runs     : {BENCH_RUNS} per pass count\n"
        f"{'='*70}\n\n"
        f"  {'Passes':<10} {'Mean (ms)':>12} {'P95 (ms)':>10} {'FPS':>8} {'Variance':>14} {'Stability':>12}\n"
        f"  {'-'*68}\n"
    )

    recommended = None
    for r in results:
        stability = round(r["mean_variance"] / base_variance * 100, 1) if base_variance > 0 else 0.0
        output += (
            f"  {r['passes']:<10} "
            f"{r['mean_ms']:>12} "
            f"{r['p95_ms']:>10} "
            f"{r['throughput']:>8} "
            f"{r['mean_variance']:>14.8f} "
            f"{stability:>11}%\n"
        )
        if recommended is None and stability >= 90.0 and r["passes"] > 1:
            recommended = r

    if recommended:
        output += (
            f"\n  Recommended: {recommended['passes']} passes\n"
            f"  Latency    : {recommended['mean_ms']} ms\n"
            f"  Throughput : {recommended['throughput']} estimates/sec\n"
        )

    output += f"\n{'='*70}\n"

    print(output)
    with open(RESULTS_FILE, "w") as f:
        f.write(output)
    log.info(f"Results saved: {RESULTS_FILE}")

if __name__ == "__main__":
    run_benchmark()

