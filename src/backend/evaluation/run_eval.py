"""CLI: run the RAGAS and pipeline evaluations on the dataset."""

import logging

from utils.config import settings
from utils.logging_config import configure_logging

from .loader import EvaluationDatasetLoader
from .metrics import MetricsEvaluator
from .ragas_eval import RagasEvaluator

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the RAGAS, Recall@K, latency and cost evaluation (CLI entry).

    Load EVALUATION_DATASET_PATH, log every metric to the console, and send
    the results to LangSmith when tracing is enabled.

    Raises:
        ValueError: If EVALUATION_DATASET_PATH is not set.
    """
    configure_logging()

    dataset_path = settings.evaluation_dataset_path
    if not dataset_path:
        raise ValueError("EVALUATION_DATASET_PATH missing in .env")

    ls_dataset_name = settings.langsmith_eval_dataset_name
    ls_experiment_name = settings.langsmith_eval_experiment_name

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
    evaluator.log_to_langsmith(
        records, results_df, ls_dataset_name, ls_experiment_name
    )

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
