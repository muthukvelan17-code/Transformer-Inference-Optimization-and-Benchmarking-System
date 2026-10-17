"""
Unit tests for data loading, preprocessing, cleaning, and input validation.
"""
import pandas as pd
import pytest
from data.dataset_loader import DatasetLoader, clean_text
from utils.validation import (
    ValidationError,
    validate_batch_size,
    validate_dataset_dataframe,
    validate_iterations,
    validate_precision_mode,
    validate_seq_length,
    validate_text_input,
)


def test_clean_text():
    raw = "   This   is   a \t test \n movie  review!   "
    cleaned = clean_text(raw)
    assert cleaned == "This is a test movie review!"


def test_validate_batch_size():
    assert validate_batch_size(4) == 4
    assert validate_batch_size("8") == 8
    with pytest.raises(ValidationError):
        validate_batch_size(0)
    with pytest.raises(ValidationError):
        validate_batch_size(-5)
    with pytest.raises(ValidationError):
        validate_batch_size(1000)
    with pytest.raises(ValidationError):
        validate_batch_size("invalid")


def test_validate_seq_length():
    assert validate_seq_length(128) == 128
    with pytest.raises(ValidationError):
        validate_seq_length(4)
    with pytest.raises(ValidationError):
        validate_seq_length(1024)


def test_validate_iterations():
    w, b = validate_iterations(3, 10)
    assert w == 3
    assert b == 10
    with pytest.raises(ValidationError):
        validate_iterations(-1, 10)
    with pytest.raises(ValidationError):
        validate_iterations(3, 0)


def test_validate_text_input():
    assert validate_text_input("Great movie") == "Great movie"
    with pytest.raises(ValidationError):
        validate_text_input("   ")
    with pytest.raises(ValidationError):
        validate_text_input(None)


def test_validate_precision_mode():
    assert validate_precision_mode("fp32") == "FP32"
    assert validate_precision_mode("INT8") == "INT8"
    with pytest.raises(ValidationError):
        validate_precision_mode("FP8")


def test_validate_dataset_dataframe():
    df = pd.DataFrame({
        "text": ["Good film", "", "   ", None, "Bad film"],
        "sentiment": ["positive", "neutral", "neutral", "neutral", "negative"],
    })
    cleaned_df, valid_count, skipped_count = validate_dataset_dataframe(df, "text")
    assert valid_count == 2
    assert skipped_count == 3
    assert len(cleaned_df) == 2


def test_empty_dataset_dataframe():
    df_empty = pd.DataFrame()
    with pytest.raises(ValidationError, match="completely empty"):
        validate_dataset_dataframe(df_empty, "text")


def test_missing_column_dataframe():
    df = pd.DataFrame({"sentence": ["Hello world"]})
    with pytest.raises(ValidationError, match="not found in dataset"):
        validate_dataset_dataframe(df, "text")


def test_dataset_loader_sample():
    loader = DatasetLoader()
    texts, stats = loader.load_sample_dataset()
    assert len(texts) > 0
    assert stats["processed_samples"] == len(texts)
    assert stats["skipped_samples"] == 0
    assert "text_column" in stats


def test_batch_creation():
    texts = ["a", "b", "c", "d", "e"]
    batches = DatasetLoader.create_batches(texts, batch_size=2)
    assert len(batches) == 3
    assert batches[0] == ["a", "b"]
    assert batches[1] == ["c", "d"]
    assert batches[2] == ["e"]
