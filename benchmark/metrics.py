"""
Metrics calculation, statistics, and optimization improvement comparison.
"""
from typing import Any, Dict, List, Optional
import numpy as np


def calculate_latency_metrics(latencies_ms: List[float]) -> Dict[str, float]:
    """
    Calculate summary latency statistics from a list of iteration runtimes (in ms).
    """
    if not latencies_ms:
        return {
            "mean_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "std_latency_ms": 0.0,
        }

    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "mean_latency_ms": round(float(np.mean(arr)), 3),
        "median_latency_ms": round(float(np.median(arr)), 3),
        "p95_latency_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(arr, 99)), 3),
        "min_latency_ms": round(float(np.min(arr)), 3),
        "max_latency_ms": round(float(np.max(arr)), 3),
        "std_latency_ms": round(float(np.std(arr)), 3),
    }


def calculate_throughput(total_samples: int, total_time_sec: float) -> float:
    """
    Calculate throughput in samples per second.
    Safe against zero or negative duration.
    """
    if total_time_sec <= 0 or total_samples <= 0:
        return 0.0
    return round(float(total_samples / total_time_sec), 2)


def calculate_improvement_percentage(baseline_val: float, optimized_val: float, higher_is_better: bool = False) -> Optional[float]:
    """
    Calculate relative percentage improvement.
    
    If higher_is_better == False (e.g., Latency, Memory):
        improvement = ((baseline - optimized) / baseline) * 100
        (Positive means optimized is lower/better, negative means optimized is worse)
        
    If higher_is_better == True (e.g., Throughput):
        improvement = ((optimized - baseline) / baseline) * 100
        (Positive means optimized is higher/better, negative means optimized is worse)
        
    Safely handles zero, None, or negative baseline values.
    """
    if baseline_val is None or optimized_val is None:
        return None
    try:
        b = float(baseline_val)
        o = float(optimized_val)
    except (ValueError, TypeError):
        return None

    if b <= 0:
        return None

    if higher_is_better:
        improvement = ((o - b) / b) * 100.0
    else:
        improvement = ((b - o) / b) * 100.0

    return round(float(improvement), 2)


def calculate_prediction_fidelity(
    baseline_predictions: List[Dict[str, Any]],
    optimized_predictions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare predictions between baseline and optimized models.
    Measures:
    - Label agreement rate (fraction of predictions with identical predicted label)
    - Mean absolute confidence difference
    - Maximum confidence drift
    """
    if not baseline_predictions or not optimized_predictions:
        return {
            "label_agreement_pct": 100.0,
            "mean_confidence_drift": 0.0,
            "max_confidence_drift": 0.0,
            "sample_count": 0,
        }

    total = min(len(baseline_predictions), len(optimized_predictions))
    if total == 0:
        return {
            "label_agreement_pct": 100.0,
            "mean_confidence_drift": 0.0,
            "max_confidence_drift": 0.0,
            "sample_count": 0,
        }

    matching_labels = 0
    confidence_diffs = []

    for b, o in zip(baseline_predictions[:total], optimized_predictions[:total]):
        if b.get("label") == o.get("label"):
            matching_labels += 1
        b_conf = float(b.get("confidence", 0.0))
        o_conf = float(o.get("confidence", 0.0))
        confidence_diffs.append(abs(b_conf - o_conf))

    agreement_pct = round((matching_labels / total) * 100.0, 2)
    mean_drift = round(float(np.mean(confidence_diffs)), 4) if confidence_diffs else 0.0
    max_drift = round(float(np.max(confidence_diffs)), 4) if confidence_diffs else 0.0

    return {
        "label_agreement_pct": agreement_pct,
        "mean_confidence_drift": mean_drift,
        "max_confidence_drift": max_drift,
        "sample_count": total,
    }


def compare_benchmark_runs(baseline_run: Dict[str, Any], optimized_run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate complete comparison between a baseline run and an optimized run.
    """
    b_lat = baseline_run.get("mean_latency_ms", 0.0)
    o_lat = optimized_run.get("mean_latency_ms", 0.0)
    lat_imp = calculate_improvement_percentage(b_lat, o_lat, higher_is_better=False)

    b_thr = baseline_run.get("throughput_samples_per_sec", 0.0)
    o_thr = optimized_run.get("throughput_samples_per_sec", 0.0)
    thr_imp = calculate_improvement_percentage(b_thr, o_thr, higher_is_better=True)

    b_mem = baseline_run.get("memory_rss_mb", 0.0)
    o_mem = optimized_run.get("memory_rss_mb", 0.0)
    mem_imp = calculate_improvement_percentage(b_mem, o_mem, higher_is_better=False)

    return {
        "baseline_precision": baseline_run.get("precision", "FP32"),
        "optimized_precision": optimized_run.get("precision", "N/A"),
        "baseline_latency_ms": b_lat,
        "optimized_latency_ms": o_lat,
        "latency_improvement_pct": lat_imp,
        "baseline_throughput": b_thr,
        "optimized_throughput": o_thr,
        "throughput_improvement_pct": thr_imp,
        "baseline_memory_mb": b_mem,
        "optimized_memory_mb": o_mem,
        "memory_reduction_pct": mem_imp,
    }
