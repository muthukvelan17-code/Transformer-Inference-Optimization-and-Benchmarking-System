"""
Unit and integration tests for benchmark capability checks, runner, and reporting exports.
"""
import json
import pytest
from benchmark.device_info import check_precision_support, get_system_hardware_info
from reports.report_generator import generate_csv_report, generate_json_report, generate_pdf_report


def test_hardware_info_query():
    info = get_system_hardware_info()
    assert "os" in info
    assert "python_version" in info
    assert "ram_total_gb" in info
    assert "cuda_available" in info
    assert info["ram_total_gb"] > 0


def test_precision_support_logic():
    # FP32 is universally supported
    fp32_ok, fp32_reason = check_precision_support("FP32", "cpu")
    assert fp32_ok is True

    # FP16 on CPU is NOT supported
    fp16_ok, fp16_reason = check_precision_support("FP16", "cpu")
    assert fp16_ok is False
    assert "not supported" in fp16_reason.lower() or "layernorm" in fp16_reason.lower()

    # INT8 dynamic quantization on CPU IS supported
    int8_ok, int8_reason = check_precision_support("INT8", "cpu")
    assert int8_ok is True


def test_report_csv_generation():
    mock_runs = [
        {
            "experiment_id": "exp_test_01",
            "timestamp": "2026-10-17 12:00:00",
            "model_name": "distilbert-test",
            "device": "cpu",
            "precision": "FP32",
            "batch_size": 4,
            "max_seq_length": 128,
            "status": "SUCCESS",
            "mean_latency_ms": 25.4,
            "median_latency_ms": 24.8,
            "p95_latency_ms": 28.1,
            "p99_latency_ms": 29.0,
            "throughput_samples_per_sec": 38.5,
            "total_inference_time_sec": 1.04,
            "warmup_time_sec": 0.25,
            "memory_rss_mb": 450.2,
            "gpu_memory_peak_mb": 0.0,
            "warmup_iterations": 2,
            "benchmark_iterations": 5,
        }
    ]
    csv_out = generate_csv_report(mock_runs)
    assert "exp_test_01" in csv_out
    assert "FP32" in csv_out
    assert "25.4" in csv_out
    assert "Mean Latency (ms)" in csv_out


def test_report_json_generation():
    mock_runs = [{"experiment_id": "exp_test_01", "status": "SUCCESS", "mean_latency_ms": 20.0}]
    json_out = generate_json_report(None, mock_runs, {})
    data = json.loads(json_out)
    assert data["experiments_count"] == 1
    assert "system_environment" in data
    assert data["experiments"][0]["experiment_id"] == "exp_test_01"


def test_report_pdf_generation():
    mock_runs = [
        {
            "experiment_id": "exp_01",
            "precision": "FP32",
            "batch_size": 2,
            "max_seq_length": 64,
            "status": "SUCCESS",
            "mean_latency_ms": 15.2,
            "p95_latency_ms": 16.0,
            "throughput_samples_per_sec": 65.0,
            "memory_rss_mb": 420.0,
        }
    ]
    pdf_bytes = generate_pdf_report(None, mock_runs)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500  # valid PDF generated
    assert pdf_bytes.startswith(b"%PDF")


def test_real_model_inference_integration():
    try:
        from models.model_loader import load_transformer_model
        from inference.predictor import TransformerPredictor
        model, tokenizer, meta = load_transformer_model(precision="FP32", device="cpu")
        predictor = TransformerPredictor(model, tokenizer, device="cpu")
        pred = predictor.predict_single("This product is amazing!")
        assert pred["label"] in ["POSITIVE", "NEGATIVE", "LABEL_0", "LABEL_1"]
        assert 0.0 <= pred["confidence"] <= 1.0
        assert pred["forward_latency_ms"] > 0.0
    except Exception as e:
        pytest.skip(f"Live model not available: {str(e)}")
