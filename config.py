"""
Configuration settings for the Transformer Inference Optimization and Benchmarking System.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUTS_DIR = BASE_DIR / "outputs"
SAMPLE_DATASET_PATH = DATA_DIR / "sample_dataset.csv"

# Pretrained Transformer Models
DEFAULT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
SUPPORTED_MODELS = [
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "textattack/bert-base-uncased-SST-2",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "bert-base-uncased",
]

# Supported Precision Modes
PRECISION_MODES = ["FP32", "FP16", "INT8"]

# Default Benchmark Parameters
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16]
DEFAULT_SEQ_LENGTHS = [32, 64, 128, 256]
DEFAULT_WARMUP_ITERATIONS = 3
DEFAULT_BENCHMARK_ITERATIONS = 10
DEFAULT_MAX_SEQ_LENGTH = 128
DEFAULT_BATCH_SIZE = 4

# Device Settings
DEFAULT_DEVICE = "auto"  # "auto", "cpu", "cuda"

# Visualization Theme
DARK_THEME_COLORS = {
    "background": "#0e1117",
    "card_background": "#1e222d",
    "text": "#f1f5f9",
    "muted_text": "#94a3b8",
    "primary": "#0284c7",
    "accent": "#38bdf8",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "purple": "#8b5cf6",
    "grid": "#334155",
}
