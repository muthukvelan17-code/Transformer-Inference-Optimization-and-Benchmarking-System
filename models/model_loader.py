"""
Pretrained Transformer Model Loader with Caching, Timing, and Precision Management.
"""
import gc
import sys
import time
from typing import Any, Dict, Optional, Tuple
import torch

from config import DEFAULT_MODEL
from benchmark.device_info import check_precision_support


class ModelLoadingError(Exception):
    """Raised when model downloading or loading fails."""
    pass


class UnsupportedOptimizationError(Exception):
    """Raised when an unsupported precision or optimization is requested."""
    pass


class ModelCache:
    """
    In-memory model cache to prevent expensive model and tokenizer reloads.
    Keyed by (model_name, precision, device).
    """
    _cache: Dict[Tuple[str, str, str], Tuple[Any, Any, Dict[str, Any]]] = {}

    @classmethod
    def get(cls, key: Tuple[str, str, str]) -> Optional[Tuple[Any, Any, Dict[str, Any]]]:
        return cls._cache.get(key)

    @classmethod
    def put(cls, key: Tuple[str, str, str], value: Tuple[Any, Any, Dict[str, Any]]):
        cls._cache[key] = value

    @classmethod
    def clear(cls):
        cls._cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def resolve_device(requested_device: str = "auto") -> str:
    """
    Resolve device string ('auto', 'cpu', 'cuda') to a valid torch device string.
    """
    req = requested_device.lower().strip()
    if req in ["cuda", "gpu"]:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    elif req == "cpu":
        return "cpu"
    else:  # auto
        return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable/non-trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def estimate_model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model parameter size in Megabytes."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return round(float(size_mb), 2)


def load_transformer_model(
    model_name: str = DEFAULT_MODEL,
    precision: str = "FP32",
    device: str = "auto",
    force_reload: bool = False,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Download and load a pretrained transformer model and tokenizer.
    
    Args:
        model_name: Hugging Face model identifier
        precision: "FP32", "FP16", or "INT8"
        device: "auto", "cpu", or "cuda"
        force_reload: If True, bypass in-memory cache
        
    Returns:
        (model, tokenizer, metadata_dict)
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    target_device = resolve_device(device)
    precision_upper = precision.upper()

    # Honest capability check before attempting
    is_supported, reason = check_precision_support(precision_upper, target_device)
    if not is_supported:
        raise UnsupportedOptimizationError(
            f"Precision '{precision_upper}' is not supported on device '{target_device}'. Reason: {reason}"
        )

    cache_key = (model_name, precision_upper, target_device)
    if not force_reload:
        cached = ModelCache.get(cache_key)
        if cached is not None:
            model, tokenizer, meta = cached
            # Mark that it was served from cache
            meta_copy = dict(meta)
            meta_copy["loaded_from_cache"] = True
            return model, tokenizer, meta_copy

    load_start = time.perf_counter()

    # Step 1: Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        raise ModelLoadingError(
            f"Failed to load tokenizer for '{model_name}'. "
            f"Error: {str(e)}. Please check your internet connection and model name."
        )

    # Step 2: Load Model in standard FP32
    try:
        raw_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as e:
        raise ModelLoadingError(
            f"Failed to load model weights for '{model_name}'. "
            f"Error: {str(e)}. Please verify model name and internet connectivity."
        )

    raw_model.eval()

    # Step 3: Apply Precision Optimization
    if precision_upper == "FP32":
        model = raw_model.float()
        model.to(target_device)
    elif precision_upper == "FP16":
        if target_device == "cuda":
            model = raw_model.half()
            model.to(target_device)
        else:
            raise UnsupportedOptimizationError(
                "FP16 half-precision is not supported on CPU. Use a CUDA GPU for FP16 inference."
            )
    elif precision_upper == "INT8":
        if target_device == "cpu":
            # Dynamic PyTorch Quantization for Linear layers
            try:
                model = torch.ao.quantization.quantize_dynamic(
                    raw_model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            except Exception as q_err:
                raise UnsupportedOptimizationError(
                    f"INT8 dynamic quantization failed: {str(q_err)}"
                )
        else:
            raise UnsupportedOptimizationError(
                "PyTorch dynamic INT8 quantization is designed for CPU. For CUDA INT8, bitsandbytes or TensorRT is required."
            )
    else:
        raise UnsupportedOptimizationError(f"Unsupported precision: {precision}")

    model.eval()
    load_duration = time.perf_counter() - load_start

    # Metadata extraction
    id2label = getattr(model.config, "id2label", {0: "NEGATIVE", 1: "POSITIVE"})
    param_count = count_parameters(model)
    model_size_mb = estimate_model_size_mb(model)

    metadata = {
        "model_name": model_name,
        "precision": precision_upper,
        "device": target_device,
        "load_time_sec": round(load_duration, 4),
        "parameter_count": param_count,
        "model_size_mb": model_size_mb,
        "num_labels": getattr(model.config, "num_labels", 2),
        "id2label": id2label,
        "architecture": model.config.architectures[0] if getattr(model.config, "architectures", None) else type(model).__name__,
        "loaded_from_cache": False,
    }

    # Store in cache
    ModelCache.put(cache_key, (model, tokenizer, metadata))

    return model, tokenizer, metadata
