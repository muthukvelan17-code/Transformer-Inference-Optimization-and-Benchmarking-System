"""
Tokenizer utilities: batch tokenization, padding, truncation, and sequence stats.
"""
from typing import Any, Dict, List, Tuple
import torch

from utils.validation import validate_seq_length


def tokenize_texts(
    tokenizer: Any,
    texts: List[str],
    max_length: int = 128,
    padding: bool = True,
    truncation: bool = True,
    device: str = "cpu",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Tokenize a list of texts into PyTorch tensors with padding, truncation, and attention mask.
    
    Args:
        tokenizer: Pretrained Hugging Face tokenizer
        texts: List of input string sentences
        max_length: Max sequence length
        padding: Whether to pad to longest in batch or max_length
        truncation: Whether to truncate to max_length
        device: Target torch device
        
    Returns:
        (encoded_tensors_dict, token_stats_dict)
    """
    max_len = validate_seq_length(max_length)
    if not texts:
        raise ValueError("Cannot tokenize an empty list of texts.")

    # Encode with Hugging Face tokenizer
    encoded = tokenizer(
        texts,
        padding=padding,
        truncation=truncation,
        max_length=max_len,
        return_tensors="pt",
    )

    # Move tensors to destination device
    tensors = {k: v.to(device) for k, v in encoded.items()}

    # Compute sequence statistics
    # Attention mask indicates non-padded tokens (sum of 1s per row)
    attention_mask = encoded["attention_mask"]
    actual_lengths = attention_mask.sum(dim=1).tolist()

    stats = {
        "batch_size": len(texts),
        "max_configured_length": max_len,
        "padded_tensor_shape": list(encoded["input_ids"].shape),
        "min_tokens": min(actual_lengths) if actual_lengths else 0,
        "max_tokens": max(actual_lengths) if actual_lengths else 0,
        "mean_tokens": round(sum(actual_lengths) / len(actual_lengths), 2) if actual_lengths else 0,
        "actual_lengths": actual_lengths,
    }

    return tensors, stats
