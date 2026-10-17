"""
Input validation utilities for the benchmarking system.
"""
from typing import Any, List, Tuple
import pandas as pd


class ValidationError(ValueError):
    """Raised when an input fails validation criteria."""
    pass


def validate_batch_size(batch_size: Any, min_val: int = 1, max_val: int = 256) -> int:
    """
    Validate that batch size is an integer within allowable bounds.
    """
    try:
        val = int(batch_size)
    except (ValueError, TypeError):
        raise ValidationError(f"Batch size must be an integer, got: {batch_size}")

    if val < min_val:
        raise ValidationError(f"Batch size must be at least {min_val}, got: {val}")
    if val > max_val:
        raise ValidationError(f"Batch size cannot exceed {max_val}, got: {val}")
    return val


def validate_seq_length(seq_length: Any, min_val: int = 8, max_val: int = 512) -> int:
    """
    Validate that sequence length is an integer within allowable bounds.
    """
    try:
        val = int(seq_length)
    except (ValueError, TypeError):
        raise ValidationError(f"Sequence length must be an integer, got: {seq_length}")

    if val < min_val:
        raise ValidationError(f"Sequence length must be at least {min_val}, got: {val}")
    if val > max_val:
        raise ValidationError(f"Sequence length cannot exceed {max_val}, got: {val}")
    return val


def validate_iterations(warmup_iters: Any, benchmark_iters: Any) -> Tuple[int, int]:
    """
    Validate warmup and measurement iteration counts.
    """
    try:
        warmup = int(warmup_iters)
        iters = int(benchmark_iters)
    except (ValueError, TypeError):
        raise ValidationError("Warmup and benchmark iterations must be integers.")

    if warmup < 0:
        raise ValidationError(f"Warmup iterations cannot be negative, got: {warmup}")
    if iters < 1:
        raise ValidationError(f"Benchmark iterations must be at least 1, got: {iters}")
    if iters > 1000:
        raise ValidationError(f"Benchmark iterations cannot exceed 1000, got: {iters}")

    return warmup, iters


def validate_text_input(text: Any) -> str:
    """
    Validate single text input.
    """
    if text is None:
        raise ValidationError("Input text cannot be None.")
    text_str = str(text).strip()
    if not text_str:
        raise ValidationError("Input text cannot be empty or whitespace only.")
    return text_str


def validate_precision_mode(precision: str, allowed_modes: List[str] = None) -> str:
    """
    Validate that precision mode is in the allowed list.
    """
    if allowed_modes is None:
        allowed_modes = ["FP32", "FP16", "INT8"]
    p_upper = str(precision).strip().upper()
    if p_upper not in allowed_modes:
        raise ValidationError(
            f"Invalid precision mode '{precision}'. Allowed options: {allowed_modes}"
        )
    return p_upper


def validate_dataset_dataframe(df: pd.DataFrame, text_column: str) -> Tuple[pd.DataFrame, int, int]:
    """
    Validate dataset dataframe and text column.
    Returns:
        (cleaned_df, valid_count, skipped_count)
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValidationError("Dataset must be a valid pandas DataFrame.")

    if df.empty:
        raise ValidationError("Dataset is completely empty.")

    if text_column not in df.columns:
        raise ValidationError(
            f"Column '{text_column}' not found in dataset. Available columns: {list(df.columns)}"
        )

    # Filter out nulls, non-strings, or whitespace-only
    total_rows = len(df)
    valid_mask = (
        df[text_column].notna()
        & df[text_column].astype(str).str.strip().ne("")
    )
    cleaned_df = df[valid_mask].copy()
    cleaned_df[text_column] = cleaned_df[text_column].astype(str).str.strip()

    valid_count = len(cleaned_df)
    skipped_count = total_rows - valid_count

    if valid_count == 0:
        raise ValidationError(f"Column '{text_column}' contains no valid non-empty text rows.")

    return cleaned_df, valid_count, skipped_count
