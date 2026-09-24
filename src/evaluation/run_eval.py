import logging
import os
from dotenv import load_dotenv

from .loader import EvaluationDatasetLoader
from .ragas_eval import RagasEvaluator
from .metrics import MetricsEvaluator
from utils.logging_config import configure_logging

load_dotenv()

logger = logging.getLogger(__name__)


def main():
    """CLI entry point: run RAGAS, Recall@K, latency, and cost evaluation.

    Loads EVALUATION_DATASET_PATH, logs every metric to the console, and
    logs results to LangSmith when tracing is enabled.

    Raises:
        ValueError: If EVALUATION_DATASET_PATH is not set.
    """
    configure_logging()

    dataset_path = os.getenv("EVALUATION_DATASET_PATH")
    if not dataset_path:
        raise ValueError("EVALUATION_DATASET_PATH missing in .env")

    ls_dataset_name = os.getenv("LANGSMITH_EVAL_DATASET_NAME", "codebase-rag-eval")
    ls_experiment_name = os.getenv("LANGSMITH_EVAL_EXPERIMENT_NAME", "codebase-rag-eval")

    logger.info("Loading dataset...")
    records = EvaluationDatasetLoader.load_json(dataset_path)

    evaluator = RagasEvaluator()

    logger.info("Uploading evaluation dataset to LangSmith...")
    evaluator.upload_dataset_to_langsmith(records, ls_dataset_name)

    logger.info("Running RAG evaluation...")
    results_df = evaluator.evaluate(records)

    logger.info("=== PER SAMPLE RESULTS ===\n%s", results_df)

    summary = evaluator.summarize(results_df)

    logger.info("=== FINAL METRICS ===")
    for k, v in summary.items():
        logger.info("%s: %s", k, v)

    logger.info("Logging results to LangSmith...")
    evaluator.log_to_langsmith(records, results_df, ls_dataset_name, ls_experiment_name)

    logger.info("Computing retrieval + performance metrics...")
    metrics_evaluator = MetricsEvaluator()

    retrieval_metrics = metrics_evaluator.recall_at_k(records)

    agent_runs = metrics_evaluator.run_agent(records)
    latency_metrics = metrics_evaluator.evaluate_latency(agent_runs)
    cost_metrics = metrics_evaluator.evaluate_cost(records, agent_runs)
    performance_metrics = {**latency_metrics, **cost_metrics}

    logger.info("=== RETRIEVAL METRICS (Recall@K) ===")
    for name, value in retrieval_metrics.items():
        logger.info("%s: %.3f", name, value)

    logger.info("=== PERFORMANCE METRICS (Latency / Cost) ===")
    for name, value in performance_metrics.items():
        logger.info("%s: %s", name, value)

    all_metrics = {**retrieval_metrics, **performance_metrics}
    metrics_evaluator.log_metrics_to_langsmith(all_metrics, ls_experiment_name)


if __name__ == "__main__":
    main()