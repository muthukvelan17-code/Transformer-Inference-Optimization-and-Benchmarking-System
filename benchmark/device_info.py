"""
Hardware device detection, memory monitoring, and precision capability inspection.
"""
import os
import platform
import psutil
from typing import Any, Dict, Optional, Tuple


def get_system_hardware_info() -> Dict[str, Any]:
    """
    Query host CPU, RAM, OS, Python, and CUDA device information.
    """
    info: Dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
        "python_version": platform.python_version(),
        "cpu_count_physical": psutil.cpu_count(logical=False) or 1,
        "cpu_count_logical": psutil.cpu_count(logical=True) or 1,
        "cpu_percent": psutil.cpu_percent(interval=None),
    }

    # RAM Info
    mem = psutil.virtual_memory()
    info["ram_total_gb"] = round(mem.total / (1024 ** 3), 2)
    info["ram_available_gb"] = round(mem.available / (1024 ** 3), 2)
    info["ram_used_gb"] = round(mem.used / (1024 ** 3), 2)
    info["ram_percent"] = mem.percent

    # PyTorch & CUDA Info
    try:
        import torch
        info["torch_version"] = torch.__version__
        cuda_available = torch.cuda.is_available()
        info["cuda_available"] = cuda_available

        if cuda_available:
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            gpu_props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_gb"] = round(gpu_props.total_memory / (1024 ** 3), 2)
            info["gpu_compute_capability"] = f"{gpu_props.major}.{gpu_props.minor}"
        else:
            info["gpu_count"] = 0
            info["gpu_name"] = "No CUDA GPU detected (CPU mode)"
            info["gpu_total_memory_gb"] = 0.0
            info["gpu_compute_capability"] = "N/A"
    except ImportError:
        info["torch_version"] = "Not Installed"
        info["cuda_available"] = False
        info["gpu_count"] = 0
        info["gpu_name"] = "PyTorch not available"
        info["gpu_total_memory_gb"] = 0.0
        info["gpu_compute_capability"] = "N/A"

    return info


def get_current_memory_usage() -> Dict[str, float]:
    """
    Measure current process RSS memory and GPU memory (if CUDA available).
    Returns memory in Megabytes (MB).
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu_rss_mb = round(mem_info.rss / (1024 ** 2), 2)

    usage = {
        "cpu_rss_mb": cpu_rss_mb,
        "gpu_allocated_mb": 0.0,
        "gpu_reserved_mb": 0.0,
        "gpu_peak_mb": 0.0,
    }

    try:
        import torch
        if torch.cuda.is_available():
            usage["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 ** 2), 2)
            usage["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 ** 2), 2)
            usage["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
    except Exception:
        pass

    return usage


def check_precision_support(precision: str, device: str = "cpu") -> Tuple[bool, str]:
    """
    Honestly evaluate whether a specific precision mode is supported on the target device.
    Never silently pretend an unsupported optimization works.
    
    Args:
        precision: "FP32", "FP16", or "INT8"
        device: "cpu", "cuda", or "cuda:0"
        
    Returns:
        (is_supported: bool, reason: str)
    """
    prec = precision.upper()
    dev = device.lower()

    if prec == "FP32":
        return True, "FP32 full precision is universally supported on all devices."

    if prec == "FP16":
        if "cuda" in dev:
            try:
                import torch
                if torch.cuda.is_available():
                    return True, "FP16 half-precision is natively supported on CUDA GPU."
                return False, "CUDA is not available on this system."
            except Exception as e:
                return False, f"Failed to verify CUDA FP16 support: {str(e)}"
        else:
            return (
                False,
                "FP16 half-precision is not supported for Transformer inference on CPU in PyTorch. "
                "PyTorch CPU lacks float16 kernel implementations for LayerNorm and Multihead Attention "
                "(raises 'LayerNormKernelImpl not implemented for Half'). Please use a CUDA GPU for FP16."
            )

    if prec == "INT8":
        if "cpu" in dev:
            try:
                import torch
                # PyTorch dynamic quantization is supported for CPU Linear layers
                return (
                    True,
                    "INT8 dynamic quantization is supported on CPU via PyTorch's torch.ao.quantization."
                )
            except Exception as e:
                return False, f"Failed to verify PyTorch CPU INT8 support: {str(e)}"
        else:
            return (
                False,
                "PyTorch Dynamic Quantization (torch.ao.quantization) is designed for CPU execution. "
                "CUDA dynamic quantization is not supported by PyTorch core (requires external engines "
                "such as bitsandbytes or TensorRT). INT8 benchmarking is available on CPU."
            )

    return False, f"Unknown precision mode: '{precision}'."
