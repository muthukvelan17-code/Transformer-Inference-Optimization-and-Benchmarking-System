"""
Dataset loading and preprocessing module for transformer inference and benchmarking.
"""
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import pandas as pd

from config import SAMPLE_DATASET_PATH
from utils.validation import ValidationError, validate_batch_size, validate_dataset_dataframe


def clean_text(text: str) -> str:
    """
    Clean input text safely without changing semantic meaning.
    - Normalizes unicode characters (NFKC)
    - Strips leading/trailing whitespace
    - Replaces consecutive whitespace with single space
    """
    if not isinstance(text, str):
        text = str(text)
    # Normalize unicode characters
    normalized = unicodedata.normalize("NFKC", text)
    # Normalize whitespace
    cleaned = " ".join(normalized.split())
    return cleaned


class DatasetLoader:
    """
    Handles loading, validating, cleaning, and batching text datasets.
    """

    def __init__(self):
        self.raw_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.text_column: Optional[str] = None
        self.stats: Dict[str, Any] = {}

    def load_sample_dataset(self, file_path: Optional[Union[str, Path]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Load the bundled offline sample dataset.
        """
        path = Path(file_path) if file_path else SAMPLE_DATASET_PATH
        if not path.exists():
            raise FileNotFoundError(f"Sample dataset file not found at: {path}")

        df = pd.read_csv(path)
        return self.process_dataframe(df, text_column="text")

    def load_csv(self, file_obj_or_path: Any, text_column: str = "text") -> Tuple[List[str], Dict[str, Any]]:
        """
        Load dataset from a user-provided CSV file path or file-like buffer.
        """
        try:
            df = pd.read_csv(file_obj_or_path)
        except Exception as e:
            raise ValidationError(f"Failed to parse CSV file: {str(e)}")

        return self.process_dataframe(df, text_column=text_column)

    def load_huggingface_dataset(
        self,
        dataset_name: str = "glue",
        subset: str = "sst2",
        split: str = "validation",
        text_column: str = "sentence",
        max_samples: int = 100,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Load a small slice from a Hugging Face dataset if internet access is available.
        Does NOT download huge datasets.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required to load Hugging Face datasets. "
                "Install it with `pip install datasets`."
            )

        start_time = time.perf_counter()
        try:
            # Load only the specified small split
            if subset:
                hf_data = load_dataset(dataset_name, subset, split=split)
            else:
                hf_data = load_dataset(dataset_name, split=split)

            # Limit samples to avoid downloading or processing too much
            num_samples = min(len(hf_data), max_samples)
            sliced = hf_data.select(range(num_samples))
            df = sliced.to_pandas()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Hugging Face dataset '{dataset_name}/{subset}': {str(e)}. "
                "Ensure internet connectivity is available or use the offline sample dataset."
            )

        return self.process_dataframe(df, text_column=text_column)

    def process_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> Tuple[List[str], Dict[str, Any]]:
        """
        Validate, clean, and extract texts from a DataFrame.
        """
        start_time = time.perf_counter()

        cleaned_df, valid_count, skipped_count = validate_dataset_dataframe(df, text_column)
        raw_count = len(df)

        # Apply text cleaning
        texts = [clean_text(t) for t in cleaned_df[text_column].tolist()]
        # Filter any that might have become empty after cleaning
        valid_texts = [t for t in texts if len(t.strip()) > 0]
        final_skipped = raw_count - len(valid_texts)

        elapsed = time.perf_counter() - start_time

        self.raw_data = df
        self.cleaned_data = cleaned_df
        self.text_column = text_column
        self.stats = {
            "total_samples": raw_count,
            "processed_samples": len(valid_texts),
            "skipped_samples": final_skipped,
            "preprocessing_duration_sec": round(elapsed, 4),
            "text_column": text_column,
            "columns": list(df.columns),
        }

        return valid_texts, self.stats

    @staticmethod
    def create_batches(texts: List[str], batch_size: int) -> List[List[str]]:
        """
        Split a list of texts into batches of specified size.
        """
        b_size = validate_batch_size(batch_size)
        if not texts:
            return []
        return [texts[i : i + b_size] for i in range(0, len(texts), b_size)]
