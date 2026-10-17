"""
Model Predictor for single text and batch inference with timing and confidence scoring.
"""
import time
from typing import Any, Dict, List, Union
import torch
import torch.nn.functional as F

from inference.tokenizer_utils import tokenize_texts
from utils.validation import validate_text_input


class TransformerPredictor:
    """
    Executes model inference on tokenized inputs, tracks precise forward pass latency,
    and returns sentiment classifications with confidence probabilities.
    """

    def __init__(self, model: torch.nn.Module, tokenizer: Any, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # Extract label mapping
        config = getattr(model, "config", None)
        if config and hasattr(config, "id2label") and config.id2label:
            self.id2label = {int(k): v for k, v in config.id2label.items()}
        else:
            self.id2label = {0: "NEGATIVE", 1: "POSITIVE"}

    def predict_single(self, text: str, max_length: int = 128) -> Dict[str, Any]:
        """
        Run inference on a single text string.
        """
        cleaned = validate_text_input(text)
        results = self.predict_batch([cleaned], max_length=max_length)
        pred = results["predictions"][0]
        pred["forward_latency_ms"] = results["forward_latency_ms"]
        pred["total_latency_ms"] = results["total_latency_ms"]
        return pred

    def predict_batch(self, texts: List[str], max_length: int = 128) -> Dict[str, Any]:
        """
        Run inference on a batch of texts.
        
        Returns:
            Dict containing:
            - predictions: list of dicts with label, confidence, probabilities, token count
            - forward_latency_ms: pure model forward pass duration
            - total_latency_ms: tokenization + forward pass + softmax duration
            - token_stats: token length statistics
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty.")

        total_start = time.perf_counter()

        # Tokenize
        tensors, token_stats = tokenize_texts(
            self.tokenizer,
            texts,
            max_length=max_length,
            device=self.device
        )

        # Synchronize device before timing if CUDA
        if "cuda" in self.device and torch.cuda.is_available():
            torch.cuda.synchronize()

        forward_start = time.perf_counter()

        # Model forward pass in inference mode
        with torch.inference_mode():
            outputs = self.model(**tensors)
            logits = outputs.logits

        if "cuda" in self.device and torch.cuda.is_available():
            torch.cuda.synchronize()

        forward_duration_ms = round((time.perf_counter() - forward_start) * 1000.0, 3)

        # Compute probabilities
        probs = F.softmax(logits, dim=-1).cpu().tolist()
        predicted_classes = torch.argmax(logits, dim=-1).cpu().tolist()
        raw_logits_list = logits.cpu().tolist()

        predictions: List[Dict[str, Any]] = []
        actual_lens = token_stats.get("actual_lengths", [0] * len(texts))

        for idx, (p_class, p_probs, p_raw) in enumerate(zip(predicted_classes, probs, raw_logits_list)):
            label_name = self.id2label.get(p_class, f"LABEL_{p_class}")
            confidence = p_probs[p_class]

            prob_dict = {
                self.id2label.get(i, f"LABEL_{i}"): round(prob, 4)
                for i, prob in enumerate(p_probs)
            }

            predictions.append({
                "text": texts[idx],
                "label": label_name,
                "confidence": round(float(confidence), 4),
                "probabilities": prob_dict,
                "predicted_class_id": p_class,
                "token_count": actual_lens[idx] if idx < len(actual_lens) else len(texts[idx].split()),
                "raw_logits": [round(float(v), 4) for v in p_raw],
            })

        total_duration_ms = round((time.perf_counter() - total_start) * 1000.0, 3)

        return {
            "predictions": predictions,
            "forward_latency_ms": forward_duration_ms,
            "total_latency_ms": total_duration_ms,
            "sample_count": len(texts),
            "token_stats": token_stats,
        }
