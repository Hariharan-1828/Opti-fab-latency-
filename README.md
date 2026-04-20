# OPTI-FAB
### Latency-First, Uncertainty-Aware Edge-AI Defect Classification

**Hariharan M &nbsp;·&nbsp; Asmitha M**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange)](https://tensorflow.org)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.3-green)](https://developer.nvidia.com/tensorrt)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

> Reframing semiconductor wafer inspection as a **real-time systems problem** — not just a vision accuracy problem.

---

## Demo

<!-- After pushing to GitHub, drag the demo video into this README via the editor to get the real URL -->

*Stream inference running live on real test images — early exit triggered at 50–68% frame completion on RTX 4050.*

---

## The Core Idea

Most AI inspection systems follow a serial pipeline:

```
Capture → Buffer → Save File → OS I/O → Load → Infer → Decide
```

Over **70% of end-to-end latency lives outside the model** — in file I/O, OS buffering, and memory copies. Optimizing the model alone cannot fix this.

OPTI-FAB processes inspection data as a **continuous pixel stream**, overlapping data acquisition with inference:

```
Camera Stream → Circular Buffer ↔ Infer → Confidence Gate → Decide
```

Decisions are made **while the wafer is still under the scanner** — typically at 50–68% frame completion.

---

## Measured Results

> All benchmarks measured on **NVIDIA RTX 4050 Laptop GPU**, Windows 11, CUDA 12.6, TensorRT 10.3.

### Inference Latency

| Provider | Mean (ms) | P95 (ms) | Throughput |
|---|---|---|---|
| CPU baseline | 2.64 | 3.12 | 379 fps |
| CUDA | 2.74 | 4.01 | 365 fps |
| **TensorRT FP16** | **0.79** | **1.05** | **1,262 fps** |

**TensorRT is 3.34x faster than CPU** — 500 runs after 50 warm-up.

### Pipeline Latency — Stream vs File-Based

| Metric | File-Based Pipeline | OPTI-FAB Stream | Improvement |
|---|---|---|---|
| Mean latency | 8.30 ms | 3.00 ms | **−63.9%** |
| Median latency | 7.74 ms | 2.47 ms | −68.1% |
| P95 latency | 11.41 ms | 4.83 ms | −57.7% |
| Min latency | 6.27 ms | 2.02 ms | −67.8% |
| Throughput | 120.5 fps | 333.3 fps | **+176.6%** |

**63.9% latency reduction — measured on real hardware, not simulated.**

### Classification Performance — Test Set (n=150)

| Metric | Value |
|---|---|
| Accuracy | 90% |
| Weighted Precision | 0.93 |
| Weighted Recall | 0.90 |
| Weighted F1 | 0.89 |
| Macro F1 | 0.90 |

### Why Latency Matters in Manufacturing

At wafer transport speed of **100 mm/s**:

| Pipeline | Latency | Wafer movement | Inline rejection |
|---|---|---|---|
| File-based | 8.3 ms | 0.83 mm | Borderline |
| **OPTI-FAB** | **3.0 ms** | **0.30 mm** | **Feasible** |

---

## Key Contributions

**1. Stream-aware inference pipeline**
Inference begins on partial frames as pixel data arrives. No waiting for full image capture.

**2. Circular buffer sliding-window**
A ring buffer maintains visual context without full-frame memory allocation, reducing data ingress overhead by 79%.

**3. Confidence-gated early exit**
Decisions triggered once prediction confidence exceeds threshold — typically at 50–68% frame completion, saving the remaining scan time entirely.

**4. Entropy-based uncertainty estimation**
Normalized prediction entropy distinguishes high-confidence decisions from borderline cases that need deferred review.

**5. TensorRT FP16 optimization**
ONNX export + TensorRT compilation achieves **0.79 ms inference** — 3.34x faster than CPU baseline, 1,262 frames per second.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      OPTI-FAB Pipeline                       │
│                                                              │
│  Inspection    Circular      MobileNetV2     Confidence      │
│  Camera     →  Buffer     →  TRT FP16     →  Gate         →  Decision
│  (pixel        (sliding      (0.79 ms)       conf ≥ 0.85     │
│   stream)       window)                      unc  ≤ 0.35)    │
└──────────────────────────────────────────────────────────────┘
```

**Decision logic:**

| Condition | Action |
|---|---|
| `conf ≥ 0.85` AND `entropy ≤ 0.35` | Immediate inline decision |
| `conf ≥ 0.85` AND `entropy > 0.35` | Defer for human review |
| `conf < 0.85` | Continue streaming next rows |

**Model:** MobileNetV2 backbone, grayscale → RGB adapter, GlobalAveragePooling, Dense(256), Dropout(0.4), Softmax(8)

---

## Defect Classes

| Class | Description |
|---|---|
| `clean` | No defect — accept |
| `crack` | Surface or structural crack |
| `edge_defect` | Defect at die or wafer edge |
| `open` | Open circuit pattern defect |
| `other` | Unclassified anomaly |
| `scratch` | Linear surface scratch |
| `short` | Short circuit pattern defect |
| `spot` | Contamination spot |

---

## Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| clean | 0.95 | 1.00 | **0.98** | 20 |
| crack | 0.95 | 1.00 | **0.98** | 20 |
| open | 1.00 | 1.00 | **1.00** | 15 |
| short | 1.00 | 1.00 | **1.00** | 15 |
| scratch | 0.87 | 1.00 | 0.93 | 20 |
| spot | 1.00 | 0.75 | 0.86 | 20 |
| edge_defect | 0.67 | 1.00 | 0.80 | 20 |
| other | 1.00 | 0.50 | 0.67 | 20 |

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare dataset
```
dataset/
├── train/
│   ├── clean/
│   ├── crack/
│   ├── edge_defect/
│   ├── open/
│   ├── other/
│   ├── scratch/
│   ├── short/
│   └── spot/
├── validation/
└── test/
```

### 3. Train
```bash
python src/train.py
```

### 4. Evaluate
```bash
python src/evaluate.py
```

### 5. Export to ONNX
```bash
python src/export_onnx.py
```

### 6. Run live stream demo
```bash
# Normal speed
python src/demo/stream_demo.py

# Slower — easier to follow
python src/demo/stream_demo.py --speed slow

# Specific image
python src/demo/stream_demo.py --img_path dataset/test/crack/img001.png
```

**Demo controls:** `SPACE` = pause/resume · `R` = next image · `Q` = quit

### 7. Run benchmarks
```bash
# TensorRT vs CUDA vs CPU
python src/benchmarks/benchmark_trt.py

# Stream pipeline vs file-based
python src/benchmarks/benchmark_pipeline.py

# MC Dropout latency profiling
python src/benchmarks/benchmark_mc_dropout.py
```

---

## Project Structure

```
OPTI-FAB/
├── src/
│   ├── config.py                    # Central config — one source of truth
│   ├── train.py                     # Model training
│   ├── evaluate.py                  # Evaluation + confusion matrix
│   ├── export_onnx.py               # ONNX export
│   ├── mc_inference.py              # MC Dropout inference + decision gate
│   ├── stream_simulator.py          # Stream pipeline simulation
│   ├── benchmarks/
│   │   ├── benchmark_trt.py         # TensorRT latency benchmark
│   │   ├── benchmark_pipeline.py    # Pipeline comparison benchmark
│   │   ├── benchmark_mc_dropout.py  # MC Dropout profiling
│   │   └── BENCHMARK_RESULTS.md     # Full results table
│   ├── demo/
│   │   └── stream_demo.py           # Pygame real-time visualization
│   └── tools/
│       ├── inspect_onnx_graph.py    # ONNX graph inspector
│       └── export_mc_onnx.py        # MC Dropout ONNX export tool
├── models/                          # Trained models (see Releases)
├── dataset/                         # Train / validation / test splits
├── results/                         # Metrics, confusion matrix
├── requirements.txt
└── README.md
```

---

## Environment

| Component | Version |
|---|---|
| Python | 3.10.9 |
| TensorFlow | 2.12.0 |
| TensorRT | 10.3.0 |
| onnxruntime-gpu | 1.19.2 |
| CUDA | 12.6 |
| cuDNN | 9.3.0 |
| pygame | 2.5.2 |

**Benchmark hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM), Windows 11

---

## Roadmap

- [x] Stream-aware inference pipeline
- [x] Circular buffer sliding-window processing
- [x] Confidence-gated early exit
- [x] Entropy-based uncertainty estimation
- [x] ONNX export + TensorRT FP16 optimization
- [x] Real-time pygame visualization demo
- [x] Measured benchmarks on RTX 4050
- [ ] Hardware validation on NXP i.MX board
- [ ] Retraining on NEU-DET real wafer surface dataset
- [ ] Grad-CAM explainability visualizations
- [ ] Ablation study with larger dataset
- [ ] Federated learning for cross-fab deployment

---

## Authors

**Hariharan M** &nbsp;·&nbsp; **Asmitha M**

*Built for IESA DeepTech Hackathon 2026 — continued as an open edge-AI inference research project.*
