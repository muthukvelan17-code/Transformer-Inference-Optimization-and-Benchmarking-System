"""
Benchmarking Engine for real transformer inference execution, memory tracking, and timing.
"""
import datetime
import gc
import time
import uuid
from typing import Any, Callable, Dict, List, Optional
import torch

from benchmark.device_info import check_precision_support, get_current_memory_usage, get_system_hardware_info
from benchmark.metrics import (
    calculate_latency_metrics,
    calculate_prediction_fidelity,
    calculate_throughput,
    compare_benchmark_runs,
)
from benchmark.optimizations import prepare_optimized_model
from inference.predictor import TransformerPredictor
from inference.tokenizer_utils import tokenize_texts
from models.model_loader import UnsupportedOptimizationError, resolve_device
from utils.validation import validate_batch_size, validate_iterations, validate_seq_length


class BenchmarkRunner:
    """
    Executes reproducible, real inference benchmarks across precision modes and configurations.
    """

    def __init__(self, model_name: str, base_model: torch.nn.Module, tokenizer: Any):
        self.model_name = model_name
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.history: List[Dict[str, Any]] = []

    def run_single_benchmark(
        self,
        sample_texts: List[str],
        precision: str = "FP32",
        batch_size: int = 4,
        max_seq_length: int = 128,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
        device: str = "auto",
        baseline_predictions: Optional[List[Dict[str, Any]]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single benchmark run with precise timing, memory measurement, and capability checks.
        """
        exp_id = f"exp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:6]}"
        target_device = resolve_device(device)
        prec = precision.upper()

        warmup_iters, bench_iters = validate_iterations(warmup_iterations, benchmark_iterations)
        b_size = validate_batch_size(batch_size)
        seq_len = validate_seq_length(max_seq_length)

        result: Dict[str, Any] = {
            "experiment_id": exp_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.model_name,
            "device": target_device,
            "precision": prec,
            "batch_size": b_size,
            "max_seq_length": seq_len,
            "warmup_iterations": warmup_iters,
            "benchmark_iterations": bench_iters,
            "status": "PENDING",
            "error_message": None,
            "mean_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "std_latency_ms": 0.0,
            "throughput_samples_per_sec": 0.0,
            "total_inference_time_sec": 0.0,
            "warmup_time_sec": 0.0,
            "model_prep_time_sec": 0.0,
            "memory_rss_mb": 0.0,
            "gpu_memory_allocated_mb": 0.0,
            "gpu_memory_peak_mb": 0.0,
            "sample_count": len(sample_texts),
            "iteration_latencies_ms": [],
            "predictions_sample": [],
            "fidelity": None,
        }

        # Step 1: Capability Validation
        is_supported, reason = check_precision_support(prec, target_device)
        if not is_supported:
            result["status"] = "UNSUPPORTED"
            result["error_message"] = reason
            self.history.append(result)
            return result

        try:
            if progress_callback:
                progress_callback(0.05, f"Optimizing model for {prec} on {target_device}...")

            # Model preparation & precision transformation
            prep_start = time.perf_counter()
            opt_model, opt_info = prepare_optimized_model(self.base_model, prec, target_device)
            result["model_prep_time_sec"] = round(time.perf_counter() - prep_start, 4)
            result["technique"] = opt_info.get("technique")

            # Slice texts for batch size
            eval_texts = sample_texts[:b_size] if len(sample_texts) >= b_size else sample_texts * (b_size // len(sample_texts) + 1)
            eval_texts = eval_texts[:b_size]

            # Tokenize once to evaluate model pure forward pass and throughput
            tensors, token_stats = tokenize_texts(
                self.tokenizer,
                eval_texts,
                max_length=seq_len,
                device=target_device,
            )

            # Step 2: Warm-up Phase
            if progress_callback:
                progress_callback(0.15, f"Running {warmup_iters} warm-up iterations...")

            warmup_start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(warmup_iters):
                    if "cuda" in target_device and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _ = opt_model(**tensors)
                    if "cuda" in target_device and torch.cuda.is_available():
                        torch.cuda.synchronize()

            result["warmup_time_sec"] = round(time.perf_counter() - warmup_start, 4)

            # Reset peak GPU memory tracking if CUDA
            if "cuda" in target_device and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Step 3: Measured Steady-State Iterations
            iteration_latencies = []
            total_eval_start = time.perf_counter()

            for i in range(bench_iters):
                if progress_callback:
                    pct = 0.25 + (0.65 * (i + 1) / bench_iters)
                    progress_callback(pct, f"Measurement iteration {i+1}/{bench_iters}...")

                if "cuda" in target_device and torch.cuda.is_available():
                    torch.cuda.synchronize()

                iter_start = time.perf_counter()
                with torch.inference_mode():
                    outputs = opt_model(**tensors)

                if "cuda" in target_device and torch.cuda.is_available():
                    torch.cuda.synchronize()

                iter_duration_ms = (time.perf_counter() - iter_start) * 1000.0
                iteration_latencies.append(iter_duration_ms)

            total_inference_time_sec = time.perf_counter() - total_eval_start
            result["total_inference_time_sec"] = round(total_inference_time_sec, 4)
            result["iteration_latencies_ms"] = [round(l, 3) for l in iteration_latencies]

            # Calculate Latency Metrics
            lat_metrics = calculate_latency_metrics(iteration_latencies)
            result.update(lat_metrics)

            # Throughput = (batch_size * benchmark_iterations) / total_time_sec
            total_items = b_size * bench_iters
            result["throughput_samples_per_sec"] = calculate_throughput(total_items, total_inference_time_sec)

            # Memory Telemetry
            mem_snapshot = get_current_memory_usage()
            result["memory_rss_mb"] = mem_snapshot["cpu_rss_mb"]
            result["gpu_memory_allocated_mb"] = mem_snapshot["gpu_allocated_mb"]
            result["gpu_memory_peak_mb"] = mem_snapshot["gpu_peak_mb"]

            # Predictions for fidelity check
            predictor = TransformerPredictor(opt_model, self.tokenizer, device=target_device)
            pred_res = predictor.predict_batch(eval_texts, max_length=seq_len)
            result["predictions_sample"] = pred_res["predictions"]

            # Calculate fidelity if baseline predictions are supplied
            if baseline_predictions:
                result["fidelity"] = calculate_prediction_fidelity(
                    baseline_predictions,
                    pred_res["predictions"]
                )

            result["status"] = "SUCCESS"

            if progress_callback:
                progress_callback(1.0, f"Benchmark completed: {result['mean_latency_ms']} ms/iter, {result['throughput_samples_per_sec']} samples/sec")

        except torch.cuda.OutOfMemoryError as oom:
            result["status"] = "FAILED"
            result["error_message"] = (
                f"CUDA Out of Memory (OOM) encountered: {str(oom)}. "
                "Try decreasing batch size or sequence length."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            result["status"] = "FAILED"
            result["error_message"] = f"Inference execution failed: {str(e)}"

        finally:
            gc.collect()

        self.history.append(result)
        return result

    def run_precision_comparison(
        self,
        sample_texts: List[str],
        batch_size: int = 4,
        max_seq_length: int = 128,
        warmup_iterations: int = 3,
        benchmark_iterations: int = 10,
        device: str = "auto",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run FP32 baseline, then run INT8 and FP16 where supported, and compare.
        """
        results: Dict[str, Any] = {}

        # 1. Run FP32 Baseline
        if progress_callback:
            progress_callback(0.1, "Starting FP32 baseline benchmark...")

        fp32_res = self.run_single_benchmark(
            sample_texts=sample_texts,
            precision="FP32",
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
            device=device,
        )
        results["FP32"] = fp32_res
        baseline_preds = fp32_res.get("predictions_sample", [])

        # 2. Run INT8 Optimization
        if progress_callback:
            progress_callback(0.45, "Running INT8 benchmark...")

        int8_res = self.run_single_benchmark(
            sample_texts=sample_texts,
            precision="INT8",
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
            device=device,
            baseline_predictions=baseline_preds,
        )
        results["INT8"] = int8_res

        # 3. Run FP16 Optimization
        if progress_callback:
            progress_callback(0.75, "Running FP16 benchmark...")

        fp16_res = self.run_single_benchmark(
            sample_texts=sample_texts,
            precision="FP16",
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
            device=device,
            baseline_predictions=baseline_preds,
        )
        results["FP16"] = fp16_res

        # Calculate comparisons
        comparisons = {}
        if fp32_res["status"] == "SUCCESS":
            for mode in ["INT8", "FP16"]:
                if results[mode]["status"] == "SUCCESS":
                    comparisons[f"FP32_vs_{mode}"] = compare_benchmark_runs(fp32_res, results[mode])

        if progress_callback:
            progress_callback(1.0, "Precision comparison complete.")

        return {
            "runs": results,
            "comparisons": comparisons,
            "baseline": fp32_res,
        }

    def run_batch_size_sweep(
        self,
        sample_texts: List[str],
        batch_sizes: List[int],
        precision: str = "FP32",
        max_seq_length: int = 128,
        warmup_iterations: int = 2,
        benchmark_iterations: int = 6,
        device: str = "auto",
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate latency and throughput scaling across various batch sizes.
        """
        sweep_results = []
        total = len(batch_sizes)

        for idx, bs in enumerate(batch_sizes):
            if progress_callback:
                progress_callback((idx / total), f"Testing batch size {bs} ({idx+1}/{total})...")

            res = self.run_single_benchmark(
                sample_texts=sample_texts,
                precision=precision,
                batch_size=bs,
                max_seq_length=max_seq_length,
                warmup_iterations=warmup_iterations,
                benchmark_iterations=benchmark_iterations,
                device=device,
            )
            sweep_results.append(res)

        if progress_callback:
            progress_callback(1.0, "Batch size sweep complete.")

        return sweep_results
