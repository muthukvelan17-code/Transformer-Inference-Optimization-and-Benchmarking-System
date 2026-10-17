"""
Unit tests for metric computations, improvement formulas, and fidelity calculations.
"""
import pytest
from benchmark.metrics import (
    calculate_improvement_percentage,
    calculate_latency_metrics,
    calculate_prediction_fidelity,
    calculate_throughput,
    compare_benchmark_runs,
)


def test_latency_metrics_normal():
    latencies = [10.0, 12.0, 14.0, 16.0, 18.0]
    res = calculate_latency_metrics(latencies)
    assert res["mean_latency_ms"] == 14.0
    assert res["median_latency_ms"] == 14.0
    assert res["min_latency_ms"] == 10.0
    assert res["max_latency_ms"] == 18.0
    assert res["p95_latency_ms"] >= 17.0


def test_latency_metrics_empty():
    res = calculate_latency_metrics([])
    assert res["mean_latency_ms"] == 0.0
    assert res["median_latency_ms"] == 0.0


def test_throughput_calculation():
    # 100 samples in 2 seconds = 50 samples/sec
    assert calculate_throughput(100, 2.0) == 50.0
    # Zero or negative handling
    assert calculate_throughput(0, 5.0) == 0.0
    assert calculate_throughput(100, 0.0) == 0.0
    assert calculate_throughput(100, -1.0) == 0.0


def test_improvement_percentage_lower_is_better():
    # Latency: Baseline = 20ms, Optimized = 10ms -> 50% improvement
    res = calculate_improvement_percentage(20.0, 10.0, higher_is_better=False)
    assert res == 50.0

    # Optimization was slower (regressed): Baseline = 20ms, Optimized = 25ms -> -25%
    res_regressed = calculate_improvement_percentage(20.0, 25.0, higher_is_better=False)
    assert res_regressed == -25.0


def test_improvement_percentage_higher_is_better():
    # Throughput: Baseline = 50 samples/s, Optimized = 75 samples/s -> +50% improvement
    res = calculate_improvement_percentage(50.0, 75.0, higher_is_better=True)
    assert res == 50.0

    # Regressed throughput: Baseline = 100, Optimized = 80 -> -20%
    res_regressed = calculate_improvement_percentage(100.0, 80.0, higher_is_better=True)
    assert res_regressed == -20.0


def test_improvement_percentage_safe_zeros():
    assert calculate_improvement_percentage(0.0, 10.0) is None
    assert calculate_improvement_percentage(None, 10.0) is None
    assert calculate_improvement_percentage(10.0, None) is None


def test_prediction_fidelity():
    baseline_preds = [
        {"label": "POSITIVE", "confidence": 0.95},
        {"label": "NEGATIVE", "confidence": 0.85},
    ]
    opt_preds = [
        {"label": "POSITIVE", "confidence": 0.94},
        {"label": "NEGATIVE", "confidence": 0.80},
    ]
    fidelity = calculate_prediction_fidelity(baseline_preds, opt_preds)
    assert fidelity["label_agreement_pct"] == 100.0
    assert round(fidelity["mean_confidence_drift"], 2) == 0.03

    # Disagreeing labels
    opt_preds_diverged = [
        {"label": "POSITIVE", "confidence": 0.94},
        {"label": "POSITIVE", "confidence": 0.60},
    ]
    fidelity2 = calculate_prediction_fidelity(baseline_preds, opt_preds_diverged)
    assert fidelity2["label_agreement_pct"] == 50.0


def test_compare_benchmark_runs():
    b_run = {"precision": "FP32", "mean_latency_ms": 20.0, "throughput_samples_per_sec": 50.0, "memory_rss_mb": 500.0}
    o_run = {"precision": "INT8", "mean_latency_ms": 12.0, "throughput_samples_per_sec": 80.0, "memory_rss_mb": 350.0}
    comp = compare_benchmark_runs(b_run, o_run)

    assert comp["latency_improvement_pct"] == 40.0
    assert comp["throughput_improvement_pct"] == 60.0
    assert comp["memory_reduction_pct"] == 30.0
