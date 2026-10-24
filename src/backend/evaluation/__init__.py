"""Evaluation of the RAG pipeline on a labeled dataset.

MetricsEvaluator is not re-exported: evaluation.metrics builds the default
agent, with its OpenAI and Qdrant clients, at import.
"""

from .loader import EvaluationDatasetLoader
from .ragas_eval import RagasEvaluator

__all__ = [
    "EvaluationDatasetLoader",
    "RagasEvaluator",
]
