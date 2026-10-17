"""
Transformer Inference Optimization and Benchmarking System
Streamlit Dashboard
"""
import io
import time
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

import config
from benchmark.benchmark_runner import BenchmarkRunner
from benchmark.device_info import check_precision_support, get_current_memory_usage, get_system_hardware_info
from data.dataset_loader import DatasetLoader
from inference.predictor import TransformerPredictor
from models.model_loader import (
    ModelLoadingError,
    UnsupportedOptimizationError,
    load_transformer_model,
    resolve_device,
)
from reports.report_generator import generate_csv_report, generate_json_report, generate_pdf_report
from utils.validation import ValidationError, validate_batch_size, validate_seq_length
from visualization.charts import (
    plot_batch_size_scaling,
    plot_iteration_distribution,
    plot_latency_by_precision,
    plot_memory_by_precision,
    plot_throughput_by_precision,
)

# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Transformer Inference Optimization & Benchmarking",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom CSS for Modern AI Infrastructure Dark UI
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Dark AI Infrastructure Theme */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #131b2e 0%, #1a233a 100%);
        border: 1px solid #233052;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Cards */
    .status-card {
        background: #131b2e;
        border: 1px solid #233052;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    
    /* Badges */
    .badge-success {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        border-color: #38bdf8;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.3);
    }

    /* Streamlit Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "benchmark_history" not in st.session_state:
    st.session_state.benchmark_history = []
if "precision_comparison_results" not in st.session_state:
    st.session_state.precision_comparison_results = None
if "batch_sweep_results" not in st.session_state:
    st.session_state.batch_sweep_results = None
if "dataset_texts" not in st.session_state:
    loader = DatasetLoader()
    texts, stats = loader.load_sample_dataset()
    st.session_state.dataset_texts = texts
    st.session_state.dataset_stats = stats
if "active_model" not in st.session_state:
    st.session_state.active_model = None
    st.session_state.active_tokenizer = None
    st.session_state.active_metadata = None

# Hardware Query
hardware_info = get_system_hardware_info()
live_memory = get_current_memory_usage()


# ---------------------------------------------------------
# Model Loading Helper
# ---------------------------------------------------------
def get_or_load_model(model_name: str, precision: str = "FP32", device: str = "auto"):
    """Cache-aware model retrieval with clear UI error reporting."""
    resolved_dev = resolve_device(device)
    curr_meta = st.session_state.active_metadata

    if (
        st.session_state.active_model is not None
        and curr_meta
        and curr_meta.get("model_name") == model_name
        and curr_meta.get("precision") == precision.upper()
        and curr_meta.get("device") == resolved_dev
    ):
        return (
            st.session_state.active_model,
            st.session_state.active_tokenizer,
            st.session_state.active_metadata,
        )

    with st.spinner(f"Loading '{model_name}' [{precision}] on {resolved_dev}..."):
        model, tokenizer, metadata = load_transformer_model(
            model_name=model_name,
            precision=precision,
            device=resolved_dev,
        )
        st.session_state.active_model = model
        st.session_state.active_tokenizer = tokenizer
        st.session_state.active_metadata = metadata
        return model, tokenizer, metadata


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚡ Benchmarking Engine")
    st.caption("AI Infrastructure & Precision Profiler")
    st.divider()

    st.subheader("Model Configuration")
    model_choice = st.selectbox(
        "Transformer Model",
        config.SUPPORTED_MODELS,
        index=0,
        help="Select a pretrained sequence classification transformer model.",
    )

    custom_model = st.text_input(
        "Or custom Hugging Face model identifier",
        placeholder="e.g. distilbert-base-uncased",
        help="Enter any compatible Hugging Face sequence classification model repo.",
    )
    selected_model_name = custom_model.strip() if custom_model.strip() else model_choice

    st.subheader("Hardware & Device")
    device_options = ["auto", "cpu"]
    if hardware_info.get("cuda_available"):
        device_options.append("cuda")
    selected_device = st.selectbox(
        "Compute Target",
        device_options,
        index=0,
        help="Force inference onto specific hardware or use auto-detection.",
    )

    resolved_device_str = resolve_device(selected_device)
    if "cuda" in resolved_device_str:
        st.markdown(f"<span class='badge-success'>⚡ CUDA Active: {hardware_info.get('gpu_name')}</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge-warning'>🖥️ CPU Mode Active</span>", unsafe_allow_html=True)

    st.divider()

    st.subheader("Dataset Source")
    data_source_mode = st.radio(
        "Dataset Source",
        ["Offline Sample Dataset (30 items)", "Upload Custom CSV", "Hugging Face Dataset"],
        index=0,
    )

    dataset_loader = DatasetLoader()

    if data_source_mode == "Offline Sample Dataset (30 items)":
        if st.button("Reload Built-in Dataset", use_container_width=True):
            texts, stats = dataset_loader.load_sample_dataset()
            st.session_state.dataset_texts = texts
            st.session_state.dataset_stats = stats
            st.success(f"Loaded {len(texts)} samples.")

    elif data_source_mode == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            try:
                preview_df = pd.read_csv(uploaded_file)
                st.write("Columns detected:", list(preview_df.columns))
                text_col = st.selectbox("Select Text Column", preview_df.columns)
                if st.button("Process & Load Uploaded CSV", use_container_width=True):
                    uploaded_file.seek(0)
                    texts, stats = dataset_loader.load_csv(uploaded_file, text_column=text_col)
                    st.session_state.dataset_texts = texts
                    st.session_state.dataset_stats = stats
                    st.success(f"Processed {len(texts)} valid rows ({stats['skipped_samples']} skipped).")
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")

    elif data_source_mode == "Hugging Face Dataset":
        hf_name = st.text_input("Dataset Name", value="glue")
        hf_subset = st.text_input("Subset / Config", value="sst2")
        hf_col = st.text_input("Text Column", value="sentence")
        hf_samples = st.slider("Max Samples", min_value=10, max_value=200, value=50, step=10)
        if st.button("Fetch HF Dataset", use_container_width=True):
            try:
                with st.spinner("Fetching small slice from Hugging Face Hub..."):
                    texts, stats = dataset_loader.load_huggingface_dataset(
                        dataset_name=hf_name,
                        subset=hf_subset,
                        split="validation",
                        text_column=hf_col,
                        max_samples=hf_samples,
                    )
                    st.session_state.dataset_texts = texts
                    st.session_state.dataset_stats = stats
                    st.success(f"Loaded {len(texts)} samples from {hf_name}/{hf_subset}.")
            except Exception as e:
                st.error(f"Failed to fetch dataset: {str(e)}")

    st.caption(f"Active Pool: **{len(st.session_state.dataset_texts)} texts**")
    st.divider()

    # Preload Model Button
    if st.button("Preload Model & Tokenizer", type="primary", use_container_width=True):
        try:
            get_or_load_model(selected_model_name, "FP32", selected_device)
            st.success("Model ready in memory!")
        except Exception as e:
            st.error(str(e))


# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tab_overview, tab_inference, tab_benchmark, tab_analytics, tab_history, tab_reports = st.tabs([
    "🏠 System Overview",
    "⚡ Model Inference",
    "🚀 Benchmark Engine",
    "📊 Performance Analytics",
    "📋 Benchmark History",
    "📄 Reports & Exports",
])


# =========================================================
# TAB 1: OVERVIEW DASHBOARD
# =========================================================
with tab_overview:
    st.header("Transformer Inference Optimization & Benchmarking")
    st.markdown(
        "A real-world AI infrastructure benchmarking platform evaluating the latency, throughput, "
        "memory consumption, and numerical fidelity of pretrained transformer models across precision modes "
        "(**FP32**, **FP16**, **INT8**) and execution parameters."
    )

    # Top KPI Metrics Row
    history_runs = st.session_state.benchmark_history
    success_runs = [r for r in history_runs if r.get("status") == "SUCCESS"]
    latest_run = success_runs[-1] if success_runs else {}

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric(
            "Experiments Done",
            f"{len(success_runs)} / {len(history_runs)}",
            help="Total successful benchmark experiments completed.",
        )
    with col_kpi2:
        st.metric(
            "Latest Latency",
            f"{latest_run.get('mean_latency_ms', 0.0):.1f} ms" if latest_run else "N/A",
            help="Steady-state mean forward latency of the most recent benchmark run.",
        )
    with col_kpi3:
        st.metric(
            "Latest Throughput",
            f"{latest_run.get('throughput_samples_per_sec', 0.0):.1f} s/s" if latest_run else "N/A",
            help="Samples processed per second in the most recent benchmark run.",
        )
    with col_kpi4:
        st.metric(
            "Process Memory RSS",
            f"{live_memory['cpu_rss_mb']:.1f} MB",
            help="Resident Set Size (RAM) memory consumed by this Python process.",
        )

    st.write("")

    # System Status & Precision Compatibility Grid
    col_sys1, col_sys2 = st.columns([1, 1])

    with col_sys1:
        st.subheader("🖥️ Hardware & Platform Telemetry")
        st.markdown(
            f"""
            <div class="status-card">
                <b>Operating System:</b> {hardware_info['os']}<br/>
                <b>Python Environment:</b> Python {hardware_info['python_version']}<br/>
                <b>PyTorch Version:</b> {hardware_info['torch_version']}<br/>
                <b>CPU Cores:</b> {hardware_info['cpu_count_physical']} Physical / {hardware_info['cpu_count_logical']} Logical<br/>
                <b>System RAM:</b> {hardware_info['ram_used_gb']} GB used / {hardware_info['ram_total_gb']} GB total ({hardware_info['ram_percent']}%)<br/>
                <b>CUDA Compute Device:</b> {hardware_info['gpu_name']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_sys2:
        st.subheader("⚙️ Precision Support & Capabilities")
        target_dev = resolve_device(selected_device)
        fp32_ok, fp32_msg = check_precision_support("FP32", target_dev)
        fp16_ok, fp16_msg = check_precision_support("FP16", target_dev)
        int8_ok, int8_msg = check_precision_support("INT8", target_dev)

        def badge(ok: bool) -> str:
            return "<span class='badge-success'>SUPPORTED</span>" if ok else "<span class='badge-danger'>UNSUPPORTED</span>"

        st.markdown(
            f"""
            <div class="status-card">
                <b>FP32 (Full Precision):</b> {badge(fp32_ok)}<br/>
                <small style="color:#94a3b8;">{fp32_msg}</small><br/><br/>
                <b>FP16 (Half Precision):</b> {badge(fp16_ok)}<br/>
                <small style="color:#94a3b8;">{fp16_msg}</small><br/><br/>
                <b>INT8 (Dynamic Quantization):</b> {badge(int8_ok)}<br/>
                <small style="color:#94a3b8;">{int8_msg}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Active Model Metadata
    st.subheader("📦 Active Model Specifications")
    if st.session_state.active_metadata:
        meta = st.session_state.active_metadata
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Architecture", str(meta.get("architecture")))
        c_m2.metric("Parameters", f"{meta.get('parameter_count', 0) / 1e6:.1f}M")
        c_m3.metric("Model Size", f"{meta.get('model_size_mb', 0):.1f} MB")
        c_m4.metric("Model Load Time", f"{meta.get('load_time_sec', 0):.3f} s")
    else:
        st.info("Model is not yet loaded into memory. Click **Preload Model & Tokenizer** in the sidebar or run an inference/benchmark below.")


# =========================================================
# TAB 2: MODEL INFERENCE
# =========================================================
with tab_inference:
    st.header("⚡ Real-Time Model Inference")
    st.markdown("Execute real transformer inference on custom text or batches. Inspect tokenization, output logits, and timing.")

    inf_mode = st.radio("Inference Input Mode", ["Single Text Input", "Batch Text Input"], horizontal=True)

    if inf_mode == "Single Text Input":
        default_sample = "The visual effects and sound design in this film are breathtaking and truly revolutionary."
        user_text = st.text_area("Input Text", value=default_sample, height=100)
        c_seq, c_prec = st.columns(2)
        with c_seq:
            max_seq = st.slider("Max Sequence Length", 16, 256, 128, step=16, key="inf_single_seq")
        with c_prec:
            prec_choice = st.selectbox("Precision Mode", ["FP32", "INT8", "FP16"], index=0, key="inf_single_prec")

        if st.button("Run Model Prediction", type="primary"):
            try:
                model, tokenizer, metadata = get_or_load_model(selected_model_name, prec_choice, selected_device)
                predictor = TransformerPredictor(model, tokenizer, device=metadata["device"])

                with st.spinner("Running inference..."):
                    pred_res = predictor.predict_single(user_text, max_length=max_seq)

                # Results Display
                st.subheader("Prediction Output")
                c_pred1, c_pred2, c_pred3 = st.columns(3)
                label = pred_res["label"]
                conf_pct = pred_res["confidence"] * 100.0

                with c_pred1:
                    st.metric("Predicted Sentiment", label)
                with c_pred2:
                    st.metric("Confidence Score", f"{conf_pct:.2f}%")
                with c_pred3:
                    st.metric("Inference Forward Latency", f"{pred_res.get('forward_latency_ms', 0.0):.2f} ms")

                st.write("")
                st.write("##### Token & Execution Telemetry")
                st.json({
                    "token_count": pred_res.get("token_count"),
                    "raw_logits": pred_res.get("raw_logits"),
                    "probabilities": pred_res.get("probabilities"),
                    "precision": prec_choice,
                    "device": metadata["device"],
                })

            except UnsupportedOptimizationError as uoe:
                st.error(f"Precision Error: {str(uoe)}")
            except Exception as e:
                st.error(f"Inference Failure: {str(e)}")

    else:
        st.subheader("Batch Text Inference")
        default_batch_text = (
            "Brilliant direction and masterful performances make this an unforgettable masterpiece.\n"
            "An utterly disappointing waste of time with a predictable plot and uninspired acting.\n"
            "A genuinely funny and delightfully inventive animation that appeals to all ages.\n"
            "The pacing was agonizingly slow and the character motivations were completely baffling."
        )
        batch_text_area = st.text_area("Input Texts (one per line)", value=default_batch_text, height=140)
        c_b_seq, c_b_prec = st.columns(2)
        with c_b_seq:
            b_max_seq = st.slider("Max Sequence Length", 16, 256, 128, step=16, key="inf_batch_seq")
        with c_b_prec:
            b_prec_choice = st.selectbox("Precision Mode", ["FP32", "INT8", "FP16"], index=0, key="inf_batch_prec")

        if st.button("Run Batch Inference", type="primary"):
            lines = [line.strip() for line in batch_text_area.split("\n") if line.strip()]
            if not lines:
                st.warning("Please provide at least one non-empty line of text.")
            else:
                try:
                    model, tokenizer, metadata = get_or_load_model(selected_model_name, b_prec_choice, selected_device)
                    predictor = TransformerPredictor(model, tokenizer, device=metadata["device"])

                    with st.spinner(f"Evaluating batch of {len(lines)} samples..."):
                        batch_res = predictor.predict_batch(lines, max_length=b_max_seq)

                    # Display Summary
                    st.subheader("Batch Results Summary")
                    c_sum1, c_sum2, c_sum3 = st.columns(3)
                    c_sum1.metric("Batch Size", len(lines))
                    c_sum2.metric("Total Latency", f"{batch_res['total_latency_ms']:.2f} ms")
                    c_sum3.metric("Forward Pass Latency", f"{batch_res['forward_latency_ms']:.2f} ms")

                    # Table of Predictions
                    table_data = []
                    for p in batch_res["predictions"]:
                        table_data.append({
                            "Text": p["text"][:60] + "..." if len(p["text"]) > 60 else p["text"],
                            "Sentiment": p["label"],
                            "Confidence": f"{p['confidence']*100:.1f}%",
                            "Tokens": p["token_count"],
                        })
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

                except Exception as e:
                    st.error(f"Batch Inference Failed: {str(e)}")


# =========================================================
# TAB 3: BENCHMARK ENGINE
# =========================================================
with tab_benchmark:
    st.header("🚀 Benchmarking Engine")
    st.markdown("Measure actual steady-state latency, throughput, and memory consumption under reproducible configurations.")

    bench_suite = st.radio(
        "Benchmark Mode",
        [
            "1. Precision Comparison Suite (FP32 vs INT8 vs FP16)",
            "2. Batch Size Scaling Sweep (1, 2, 4, 8, 16)",
            "3. Custom Single Experiment",
        ],
        horizontal=False,
    )

    st.write("")

    # Global Benchmark Parameters
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        bench_warmup = st.number_input("Warmup Iterations", min_value=1, max_value=20, value=3)
    with col_p2:
        bench_iters = st.number_input("Measured Iterations", min_value=1, max_value=50, value=10)
    with col_p3:
        bench_seq_len = st.select_slider("Sequence Length", options=[32, 64, 128, 256], value=128)
    with col_p4:
        bench_pool_size = st.slider("Dataset Text Pool", min_value=4, max_value=len(st.session_state.dataset_texts), value=min(30, len(st.session_state.dataset_texts)))

    active_texts = st.session_state.dataset_texts[:bench_pool_size]

    st.divider()

    # 1. Precision Comparison Suite
    if bench_suite.startswith("1. Precision"):
        st.subheader("Suite: FP32 Baseline vs. INT8 Dynamic vs. FP16 Half")
        st.markdown(
            "Executes an empirical baseline under FP32, then evaluates INT8 and FP16 where supported by the physical hardware. "
            "Computes relative improvements in latency, throughput, and memory, while verifying output label fidelity."
        )

        suite_batch = st.select_slider("Batch Size for Comparison", options=[1, 2, 4, 8, 16], value=4)

        if st.button("🚀 Execute Precision Comparison Suite", type="primary", use_container_width=True):
            try:
                base_model, tokenizer, metadata = get_or_load_model(selected_model_name, "FP32", selected_device)
                runner = BenchmarkRunner(selected_model_name, base_model, tokenizer)

                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(pct: float, msg: str):
                    progress_bar.progress(pct)
                    status_text.text(msg)

                comparison_results = runner.run_precision_comparison(
                    sample_texts=active_texts,
                    batch_size=suite_batch,
                    max_seq_length=bench_seq_len,
                    warmup_iterations=bench_warmup,
                    benchmark_iterations=bench_iters,
                    device=metadata["device"],
                    progress_callback=update_progress,
                )

                st.session_state.precision_comparison_results = comparison_results
                # Add individual runs to history
                for run_data in comparison_results["runs"].values():
                    st.session_state.benchmark_history.append(run_data)

                st.success("Precision Comparison Suite Completed Successfully!")

            except Exception as e:
                st.error(f"Benchmark Suite Execution Failed: {str(e)}")

        # Display Comparison Results if Available
        if st.session_state.precision_comparison_results:
            comp_res = st.session_state.precision_comparison_results
            runs = comp_res.get("runs", {})
            comparisons = comp_res.get("comparisons", {})

            st.write("### Comparison Results")
            cols = st.columns(len(runs))
            for idx, (prec_name, r) in enumerate(runs.items()):
                with cols[idx]:
                    st.markdown(f"#### {prec_name}")
                    if r.get("status") == "SUCCESS":
                        st.metric("Mean Latency", f"{r.get('mean_latency_ms', 0.0):.1f} ms")
                        st.metric("Throughput", f"{r.get('throughput_samples_per_sec', 0.0):.1f} s/s")
                        st.metric("Memory RSS", f"{r.get('memory_rss_mb', 0.0):.1f} MB")
                    else:
                        st.markdown(f"<span class='badge-danger'>{r.get('status')}</span>", unsafe_allow_html=True)
                        st.caption(f"Reason: {r.get('error_message')}")

            # Improvement Cards
            if comparisons:
                st.write("### Optimization Deltas vs. FP32 Baseline")
                c_cols = st.columns(len(comparisons))
                for idx, (comp_name, delta) in enumerate(comparisons.items()):
                    with c_cols[idx]:
                        st.markdown(f"**{comp_name.replace('_', ' ')}**")
                        lat_imp = delta.get("latency_improvement_pct")
                        thr_imp = delta.get("throughput_improvement_pct")
                        mem_imp = delta.get("memory_reduction_pct")

                        st.metric(
                            "Latency Improvement",
                            f"{lat_imp:+.1f}%" if lat_imp is not None else "N/A",
                            help="Positive means optimized is faster, negative means slower.",
                        )
                        st.metric(
                            "Throughput Improvement",
                            f"{thr_imp:+.1f}%" if thr_imp is not None else "N/A",
                            help="Positive means more samples/second.",
                        )
                        st.metric(
                            "Memory Reduction",
                            f"{mem_imp:+.1f}%" if mem_imp is not None else "N/A",
                            help="Positive means less RAM consumed.",
                        )

    # 2. Batch Size Scaling Sweep
    elif bench_suite.startswith("2. Batch Size"):
        st.subheader("Suite: Batch Size Scaling Sweep")
        st.markdown("Measures latency vs. throughput efficiency across batch sizes [1, 2, 4, 8, 16].")

        sweep_prec = st.selectbox("Precision Mode for Sweep", ["FP32", "INT8"], index=0)
        selected_batches = st.multiselect(
            "Batch Sizes to Benchmark",
            [1, 2, 4, 8, 16],
            default=[1, 2, 4, 8, 16],
        )

        if st.button("🚀 Run Batch Size Scaling Sweep", type="primary", use_container_width=True):
            if not selected_batches:
                st.warning("Please select at least one batch size.")
            else:
                try:
                    base_model, tokenizer, metadata = get_or_load_model(selected_model_name, sweep_prec, selected_device)
                    runner = BenchmarkRunner(selected_model_name, base_model, tokenizer)

                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    def update_progress(pct: float, msg: str):
                        progress_bar.progress(pct)
                        status_text.text(msg)

                    sweep_res = runner.run_batch_size_sweep(
                        sample_texts=active_texts,
                        batch_sizes=sorted(selected_batches),
                        precision=sweep_prec,
                        max_seq_length=bench_seq_len,
                        warmup_iterations=bench_warmup,
                        benchmark_iterations=bench_iters,
                        device=metadata["device"],
                        progress_callback=update_progress,
                    )

                    st.session_state.batch_sweep_results = sweep_res
                    for r in sweep_res:
                        st.session_state.benchmark_history.append(r)

                    st.success("Batch size sweep completed!")

                except Exception as e:
                    st.error(f"Sweep failed: {str(e)}")

        if st.session_state.batch_sweep_results:
            st.write("### Sweep Scaling Chart")
            fig_sweep = plot_batch_size_scaling(st.session_state.batch_sweep_results)
            st.pyplot(fig_sweep)

    # 3. Custom Single Experiment
    else:
        st.subheader("Custom Benchmark Configuration")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            custom_prec = st.selectbox("Precision", ["FP32", "INT8", "FP16"])
        with col_c2:
            custom_batch = st.slider("Batch Size", 1, 32, 4)

        if st.button("🚀 Run Benchmark Experiment", type="primary", use_container_width=True):
            try:
                base_model, tokenizer, metadata = get_or_load_model(selected_model_name, custom_prec, selected_device)
                runner = BenchmarkRunner(selected_model_name, base_model, tokenizer)

                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(pct: float, msg: str):
                    progress_bar.progress(pct)
                    status_text.text(msg)

                res = runner.run_single_benchmark(
                    sample_texts=active_texts,
                    precision=custom_prec,
                    batch_size=custom_batch,
                    max_seq_length=bench_seq_len,
                    warmup_iterations=bench_warmup,
                    benchmark_iterations=bench_iters,
                    device=metadata["device"],
                    progress_callback=update_progress,
                )

                st.session_state.benchmark_history.append(res)

                if res["status"] == "SUCCESS":
                    st.success(f"Benchmark finished: {res['mean_latency_ms']} ms/iter, {res['throughput_samples_per_sec']} samples/s")
                else:
                    st.error(f"Run ended with status {res['status']}: {res['error_message']}")

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")


# =========================================================
# TAB 4: PERFORMANCE ANALYTICS & CHARTS
# =========================================================
with tab_analytics:
    st.header("📊 Performance Analytics & Visualizations")
    st.markdown("Empirical benchmark charts comparing latency, throughput, memory, and scaling dynamics.")

    runs_for_charts = {}
    if st.session_state.precision_comparison_results:
        runs_for_charts = st.session_state.precision_comparison_results.get("runs", {})
    else:
        # Build dictionary from latest successful runs per precision in history
        for r in st.session_state.benchmark_history:
            if r.get("status") == "SUCCESS":
                runs_for_charts[r.get("precision")] = r

    if not runs_for_charts:
        st.info("No benchmark runs available to plot yet. Run an experiment in the **Benchmark Engine** tab to populate charts.")
    else:
        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            st.subheader("Latency Comparison (ms)")
            fig_lat = plot_latency_by_precision(runs_for_charts)
            st.pyplot(fig_lat)

        with col_ch2:
            st.subheader("Throughput (samples / sec)")
            fig_thr = plot_throughput_by_precision(runs_for_charts)
            st.pyplot(fig_thr)

        st.write("")
        col_ch3, col_ch4 = st.columns(2)
        with col_ch3:
            st.subheader("Process Memory RSS (MB)")
            fig_mem = plot_memory_by_precision(runs_for_charts)
            st.pyplot(fig_mem)

        with col_ch4:
            st.subheader("Latency Stability & Variance")
            fig_dist = plot_iteration_distribution(runs_for_charts)
            st.pyplot(fig_dist)


# =========================================================
# TAB 5: BENCHMARK HISTORY & TABLE
# =========================================================
with tab_history:
    st.header("📋 Benchmark Results History")
    st.markdown("Chronological audit log of all benchmark executions, hardware settings, and measured metrics.")

    if not st.session_state.benchmark_history:
        st.info("No experiments have been recorded yet.")
    else:
        # Convert history to DataFrame
        table_records = []
        for r in st.session_state.benchmark_history:
            table_records.append({
                "Experiment ID": r.get("experiment_id", "")[-10:],
                "Timestamp": r.get("timestamp"),
                "Precision": r.get("precision"),
                "Device": r.get("device"),
                "Batch": r.get("batch_size"),
                "SeqLen": r.get("max_seq_length"),
                "Status": r.get("status"),
                "Mean Lat (ms)": r.get("mean_latency_ms") if r.get("status") == "SUCCESS" else "N/A",
                "Median (ms)": r.get("median_latency_ms") if r.get("status") == "SUCCESS" else "N/A",
                "p95 (ms)": r.get("p95_latency_ms") if r.get("status") == "SUCCESS" else "N/A",
                "Throughput (s/s)": r.get("throughput_samples_per_sec") if r.get("status") == "SUCCESS" else "N/A",
                "RAM RSS (MB)": r.get("memory_rss_mb") if r.get("status") == "SUCCESS" else "N/A",
                "Warmup (s)": r.get("warmup_time_sec") if r.get("status") == "SUCCESS" else "N/A",
                "Error / Details": r.get("error_message", "") or "OK",
            })

        df_history = pd.DataFrame(table_records)

        # Filters
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            prec_filter = st.multiselect("Filter by Precision", options=list(df_history["Precision"].unique()), default=list(df_history["Precision"].unique()))
        with c_f2:
            status_filter = st.multiselect("Filter by Status", options=list(df_history["Status"].unique()), default=list(df_history["Status"].unique()))

        filtered_df = df_history[
            df_history["Precision"].isin(prec_filter) & df_history["Status"].isin(status_filter)
        ]

        st.dataframe(filtered_df, use_container_width=True)

        if st.button("Clear Benchmark History"):
            st.session_state.benchmark_history = []
            st.session_state.precision_comparison_results = None
            st.session_state.batch_sweep_results = None
            st.rerun()


# =========================================================
# TAB 6: REPORTS & EXPORTS
# =========================================================
with tab_reports:
    st.header("📄 Benchmark Reports & Exports")
    st.markdown("Download comprehensive benchmark datasets and reports in CSV, JSON, and professional PDF formats.")

    if not st.session_state.benchmark_history:
        st.warning("No benchmark experiments available to export. Run benchmarks to generate reports.")
    else:
        runs_to_export = st.session_state.benchmark_history
        comparisons_to_export = (
            st.session_state.precision_comparison_results.get("comparisons", {})
            if st.session_state.precision_comparison_results
            else {}
        )

        col_rep1, col_rep2, col_rep3 = st.columns(3)

        # 1. CSV Export
        with col_rep1:
            st.subheader("1. CSV Data Export")
            csv_data = generate_csv_report(runs_to_export)
            st.download_button(
                label="📥 Download Benchmark CSV",
                data=csv_data,
                file_name="transformer_benchmark_results.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
            st.caption("Standard spreadsheet format with all metric columns.")

        # 2. JSON Export
        with col_rep2:
            st.subheader("2. JSON Full Telemetry")
            json_data = generate_json_report(hardware_info, runs_to_export, comparisons_to_export)
            st.download_button(
                label="📥 Download Benchmark JSON",
                data=json_data,
                file_name="transformer_benchmark_report.json",
                mime="application/json",
                type="primary",
                use_container_width=True,
            )
            st.caption("Complete system metadata, run history, and configuration details.")

        # 3. PDF Export
        with col_rep3:
            st.subheader("3. Executive PDF Report")
            try:
                pdf_data = generate_pdf_report(hardware_info, runs_to_export, comparisons_to_export)
                st.download_button(
                    label="📥 Download Executive PDF Report",
                    data=pdf_data,
                    file_name="transformer_benchmark_executive_report.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )
                st.caption("Professional formatted PDF ready for stakeholder presentation.")
            except ImportError as ie:
                st.error(str(ie))
            except Exception as pe:
                st.error(f"Failed to compile PDF: {str(pe)}")

        st.divider()
        st.subheader("Report Preview (JSON)")
        st.json(json_data, expanded=False)
