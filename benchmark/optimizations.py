"""
Inference optimization strategies: FP32, FP16, and INT8 Dynamic Quantization.
"""
import copy
from typing import Any, Dict, Tuple
import torch

from benchmark.device_info import check_precision_support
from models.model_loader import UnsupportedOptimizationError, count_parameters, estimate_model_size_mb


def prepare_optimized_model(
    model: torch.nn.Module,
    precision: str = "FP32",
    device: str = "cpu"
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Applies the requested precision optimization to the PyTorch transformer model.
    Performs capability validation to ensure the optimization is genuinely supported.
    
    Args:
        model: PyTorch transformer model
        precision: "FP32", "FP16", or "INT8"
        device: "cpu" or "cuda"
        
    Returns:
        (optimized_model, optimization_metadata)
    """
    prec = precision.upper()
    dev = device.lower()

    # Honest capability check
    is_supported, reason = check_precision_support(prec, dev)
    if not is_supported:
        raise UnsupportedOptimizationError(
            f"Optimization '{prec}' cannot be executed on device '{dev}': {reason}"
        )

    model.eval()

    if prec == "FP32":
        opt_model = model.float()
        opt_model.to(dev)
        technique = "FP32 Full Precision (Standard Baseline)"
        description = "Standard 32-bit floating point precision. Universal compatibility."

    elif prec == "FP16":
        if "cuda" not in dev:
            raise UnsupportedOptimizationError(
                "FP16 half-precision inference requires a CUDA GPU. PyTorch does not implement float16 "
                "LayerNorm or Attention operations on CPU."
            )
        opt_model = model.half()
        opt_model.to(dev)
        technique = "FP16 Half Precision"
        description = "16-bit floating point precision utilizing CUDA Tensor Cores for reduced memory and accelerated compute."

    elif prec == "INT8":
        if "cpu" not in dev:
            raise UnsupportedOptimizationError(
                "PyTorch dynamic INT8 quantization is designed for CPU execution. "
                "CUDA dynamic quantization is not supported in core PyTorch (requires external libraries like bitsandbytes)."
            )
        # Apply dynamic quantization to Linear layers
        try:
            opt_model = torch.ao.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        except Exception as e:
            raise RuntimeError(f"PyTorch dynamic quantization failed: {str(e)}")

        technique = "INT8 Dynamic Quantization (PyTorch AO)"
        description = "Dynamically quantizes linear layer weights from float32 to 8-bit integers on CPU to halve memory bandwidth and boost throughput."

    else:
        raise UnsupportedOptimizationError(f"Unrecognized precision mode: '{precision}'.")

    opt_model.eval()

    info = {
        "precision": prec,
        "device": dev,
        "technique": technique,
        "description": description,
        "parameter_count": count_parameters(opt_model),
        "estimated_size_mb": estimate_model_size_mb(opt_model),
        "is_quantized": prec == "INT8",
    }

    return opt_model, info
