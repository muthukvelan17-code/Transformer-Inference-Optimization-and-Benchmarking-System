# Transformer Inference Optimization and Benchmarking System

A production-grade, empirical AI infrastructure benchmarking platform built to evaluate how precision modes, batch sizes, sequence lengths, and quantization affect the latency, throughput, memory consumption, and prediction fidelity of pretrained Transformer models.

This system executes **real inference forward passes** using Hugging Face Transformers and PyTorch, measures true steady-state performance across hardware targets, and generates comprehensive reports (CSV, JSON, and executive PDF).

---

## Architecture Overview

```text
transformer-inference-benchmark/
├── app.py                      # Interactive Streamlit Web Dashboard (Dark Theme AI Infra UI)
├── requirements.txt            # Python dependencies (PyTorch, Transformers, Datasets, etc.)
├── README.md                   # Comprehensive technical documentation & runbook
├── .gitignore                  # Git ignore rules for virtual environments, caches, models
├── config.py                   # Central configuration, default parameters, and design tokens
├── models/
│   ├── __init__.py
│   └── model_loader.py         # Model & tokenizer download, in-memory caching, metadata
├── data/
│   ├── __init__.py
│   ├── sample_dataset.csv      # Bundled 30-sample sentiment dataset for offline testing
│   └── dataset_loader.py       # CSV ingestion, data cleaning, validation, HF datasets
├── inference/
│   ├── __init__.py
│   ├── predictor.py            # Real model prediction, forward timing, confidence scores
│   └── tokenizer_utils.py      # Padding, truncation, attention masks, token statistics
├── benchmark/
│   ├── __init__.py
│   ├── benchmark_runner.py     # Steady-state benchmark engine, warmup separation, sweeps
│   ├── metrics.py              # Latency stats (mean, p95, p99), throughput, improvement %
│   ├── device_info.py          # CPU/CUDA detection, memory tracking, precision checks
│   └── optimizations.py        # FP32, FP16, and INT8 dynamic quantization strategies
├── visualization/
│   ├── __init__.py
│   └── charts.py               # Dark-themed Matplotlib analytics plots
├── reports/
│   ├── __init__.py
│   └── report_generator.py     # CSV, JSON, and ReportLab PDF executive export engines
├── utils/
│   ├── __init__.py
│   └── validation.py           # Robust input and dataset validation guards
└── tests/
    ├── __init__.py
    ├── test_metrics.py         # Unit tests for latency, throughput, deltas, fidelity
    ├── test_preprocessing.py   # Unit tests for text cleaning and validation edge cases
    └── test_benchmark.py       # Unit tests for hardware queries, precision logic, exports
```

---

## Key Features

1. **Empirical Benchmarking (Zero Simulated Data):**
   - Pure model inference forward passes timed using high-precision counters (`time.perf_counter()`).
   - Strict separation between model initialization, warmup passes, and steady-state measurement iterations.
   - Proper CUDA device synchronization (`torch.cuda.synchronize()`) when executing on GPUs.

2. **Supported Optimization Techniques & Precision Modes:**
   - **FP32 (Full Precision):** Standard baseline utilizing IEEE 754 single-precision floating point.
   - **FP16 (Half Precision):** Native 16-bit floating point acceleration enabled for CUDA GPUs with Tensor Cores.
   - **INT8 (Dynamic Quantization):** 8-bit integer quantization of `torch.nn.Linear` layers via PyTorch AO for CPU memory bandwidth reduction and compute acceleration.

3. **Honest Capability Checks:**
   - The engine **never silently claims an unsupported optimization succeeded** or substitutes FP32.
   - If an optimization is incompatible with the detected device (e.g. FP16 on CPU, which lacks LayerNorm kernel implementations in PyTorch), the platform clearly flags it as `UNSUPPORTED` with the exact technical rationale.

4. **Prediction Fidelity Tracking:**
   - Compares output class predictions and confidence scores between the FP32 baseline and quantized/optimized models to measure numerical drift.

5. **Batch Size & Sequence Length Sweeps:**
   - Run scaling sweeps across batch sizes `[1, 2, 4, 8, 16]` and sequence lengths `[32, 64, 128, 256]` to identify optimal throughput saturation points.

6. **Interactive Modern Web Dashboard:**
   - Dark theme AI infrastructure monitoring interface built with Streamlit.
   - Interactive inference testing (single prompt and batch text).
   - High-contrast Matplotlib telemetry charts (Latency, Throughput, Memory RSS, Variance boxplots).
   - Instant downloadable reports in **CSV**, **JSON**, and **Executive PDF**.

---

## Hardware & Precision Compatibility Matrix

| Precision Mode | CPU Support | CUDA GPU Support | Notes |
| :--- | :---: | :---: | :--- |
| **FP32** | Supported | Supported | Universal full-precision baseline. |
| **FP16** | Unsupported | Supported | PyTorch CPU lacks float16 kernels for LayerNorm/Attention. Requires CUDA GPU. |
| **INT8 (Dynamic)** | Supported | Unsupported (PyTorch AO) | PyTorch Dynamic Quantization targets CPU linear layers. GPU INT8 requires TensorRT/bitsandbytes. |

---

## Getting Started

### Prerequisites
- Python **3.11** or compatible installed Python (e.g. 3.10–3.13)
- Internet connection (initial run only, to download the default ~260MB model weights and tokenizer)
- Git (optional, for cloning)

---

### Installation on Windows PowerShell

1. Open PowerShell and navigate to the project directory:
   ```powershell
   cd "c:\Transformer Inference Optimization and Benchmarking"
   ```

2. Create a virtual environment:
   ```powershell
   python -m venv .venv
   ```

3. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   > **Note on PowerShell Script Execution Policy:**  
   > If Windows PowerShell restricts script execution with `PSSecurityException`, run:
   > ```powershell
   > Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   > ```
   > Or directly invoke Python without activating the shell:
   > ```powershell
   > .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   > .\.venv\Scripts\python.exe -m streamlit run app.py
   > ```

4. Upgrade pip and install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Launch the Streamlit application:
   ```powershell
   streamlit run app.py
   ```

---

### Installation on Linux / macOS

1. Navigate to the repository directory:
   ```bash
   cd transformer-inference-benchmark
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Launch the application:
   ```bash
   streamlit run app.py
   ```

---

## Running the Automated Test Suite

Run unit and integration tests using pytest:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

The test suite validates:
- Metric formulas (mean latency, p95, p99, throughput in samples/s).
- Relative improvement calculations (latency reduction %, throughput gain %, memory reduction %).
- Input validation (batch sizes, sequence bounds, empty inputs, malformed CSVs).
- Text cleaning and Unicode normalization.
- Precision capability verification logic.
- CSV, JSON, and PDF report generation.

---

## How to Use the Application

### 1. System Overview Dashboard
- Inspect detected hardware (CPU cores, total RAM, PyTorch version, CUDA GPU availability).
- Review supported vs. unsupported precision modes.

### 2. Real-Time Model Inference
- Enter a single sentence or multiple lines of text.
- Choose sequence length and precision (`FP32`, `INT8`, `FP16`).
- Click **Run Model Prediction** to view predicted sentiment, confidence percentage, and forward pass latency.

### 3. Benchmarking Engine
- **Precision Comparison Suite:** Executes FP32, INT8, and FP16 consecutively. Calculates exact percentage improvements and output fidelity.
- **Batch Size Scaling Sweep:** Sweeps batch sizes `[1, 2, 4, 8, 16]` to profile throughput scaling vs. memory limits.
- **Custom Experiment:** Customize warm-up iterations, measurement loops, batch sizes, and sequence lengths.

### 4. Performance Analytics
- Inspect high-resolution dark-themed charts:
  - Latency by precision (mean & p95 error bounds).
  - Throughput (samples per second).
  - Process memory RSS (MB).
  - Iteration latency variance boxplots.

### 5. Reports & Exports
- Download all recorded benchmark runs in:
  - **CSV:** Ready for Excel or custom pandas analysis.
  - **JSON:** Complete system environment metadata, configuration details, and run metrics.
  - **Executive PDF:** Formatted benchmark document including hardware specs, results table, optimization gains, and technical notes.

---

## Troubleshooting

- **Model Download Timeout:**  
  The default model (`distilbert/distilbert-base-uncased-finetuned-sst-2-english`) is downloaded automatically on first run and cached in `~/.cache/huggingface`. If downloading behind a corporate proxy or unstable connection, ensure HTTPS access to `huggingface.co`.
- **FP16 Incompatibility on CPU:**  
  If you select FP16 on a machine without a dedicated CUDA GPU, the engine will state: `FP16 half-precision is not supported on CPU. Use a CUDA GPU for FP16 inference.` This is normal and expected; PyTorch does not implement float16 LayerNorm kernels for CPU.
- **CUDA Out of Memory (OOM):**  
  If benchmarking large models with high batch sizes, reduce the batch size or maximum sequence length in the configuration slider.

---

## License

This project is licensed under the MIT License.
