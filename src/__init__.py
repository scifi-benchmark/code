"""SciFi-Benchmark: Evaluation tools for the Asimov Dilemmas dataset."""

__version__ = "0.1.0"

from .data import load_scifi_dataset
from .evaluate import evaluate_batch, compute_alignment_score

__all__ = ["load_scifi_dataset", "evaluate_batch", "compute_alignment_score"]
