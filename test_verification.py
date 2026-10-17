"""
Integration verification script to test model download, inference, benchmarking, and reporting.
"""
import time
from models.model_loader import load_transformer_model
from inference.predictor import TransformerPredictor
from benchmark.benchmark_runner import BenchmarkRunner
from data.dataset_loader import DatasetLoader
from reports.report_generator import generate_csv_report, generate_json_report, generate_pdf_report

print("=== 1. Loading Pretrained Model ===")
t0 = time.time()
model, tokenizer, meta = load_transformer_model(
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    precision="FP32",
    device="cpu"
)
load_time = time.time() - t0
print(f"Model loaded in {load_time:.2f}s! Architecture: {meta['architecture']}, Params: {meta['parameter_count']}")

print("\n=== 2. Real Model Inference Prediction ===")
predictor = TransformerPredictor(model, tokenizer, device="cpu")
res = predictor.predict_single("Antigravity and transformer optimization are exceptional achievements.")
print(f"Prediction: {res['label']}, Confidence: {res['confidence']*100:.2f}%, Latency: {res['forward_latency_ms']} ms")

batch_test = [
    "This is an extraordinary masterpiece of modern cinema.",
    "Completely dreadful, boring, and an absolute waste of time."
]
batch_res = predictor.predict_batch(batch_test)
print(f"Batch latency: {batch_res['total_latency_ms']} ms, samples: {batch_res['sample_count']}")
for p in batch_res["predictions"]:
    print(f"  - '{p['text']}' -> {p['label']} ({p['confidence']*100:.1f}%)")

print("\n=== 3. Dataset Loading ===")
loader = DatasetLoader()
texts, stats = loader.load_sample_dataset()
print(f"Loaded {len(texts)} samples from offline dataset, text_col={stats['text_column']}")

print("\n=== 4. Benchmarking Engine (Real Inference) ===")
runner = BenchmarkRunner("distilbert/distilbert-base-uncased-finetuned-sst-2-english", model, tokenizer)

print("Running FP32 Benchmark...")
fp32_res = runner.run_single_benchmark(
    texts, precision="FP32", batch_size=4, warmup_iterations=2, benchmark_iterations=3, device="cpu"
)
print(f"FP32 Mean Latency: {fp32_res['mean_latency_ms']} ms, Throughput: {fp32_res['throughput_samples_per_sec']} samples/s")

print("Running INT8 Quantized Benchmark...")
int8_res = runner.run_single_benchmark(
    texts, precision="INT8", batch_size=4, warmup_iterations=2, benchmark_iterations=3, device="cpu"
)
print(f"INT8 Mean Latency: {int8_res['mean_latency_ms']} ms, Throughput: {int8_res['throughput_samples_per_sec']} samples/s")

print("\n=== 5. Reports Verification ===")
csv_out = generate_csv_report([fp32_res, int8_res])
json_out = generate_json_report(None, [fp32_res, int8_res])
pdf_out = generate_pdf_report(None, [fp32_res, int8_res])
print(f"CSV report characters: {len(csv_out)}")
print(f"JSON report characters: {len(json_out)}")
print(f"PDF report bytes: {len(pdf_out)}")

print("\n>>> ALL VERIFICATIONS PASSED SUCCESSFULLY! <<<")
