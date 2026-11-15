"""CLI: evaluate the live pipeline (RAGAS, Recall@K, latency, cost)."""

import logging

from evaluation.loader import load_dataset
from evaluation.metrics import MetricsEvaluator
from evaluation.ragas_eval import RagasEvaluator
from utils.config import settings
from utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Run every evaluation on EVALUATION_DATASET_PATH and log the results.

    RAGAS scores the live agent's answers and the live retrieval, judged
    against the dataset's ground truths. Results also go to LangSmith when
    tracing is on.

    Raises:
        ValueError: If EVALUATION_DATASET_PATH is not set.
    """
    configure_logging()

    dataset_path = settings.evaluation_dataset_path
    if not dataset_path:
        raise ValueError("EVALUATION_DATASET_PATH missing in .env")
    dataset_name = settings.langsmith_eval_dataset_name
    experiment_name = settings.langsmith_eval_experiment_name

    records = load_dataset(dataset_path)
    metrics_evaluator = MetricsEvaluator()
    ragas_evaluator = RagasEvaluator()

    logger.info("Running the live agent on %d questions...", len(records))
    agent_runs = metrics_evaluator.run_agent(records)
    contexts = metrics_evaluator.retrieve_contexts(records)
    live_records = [
        {**record, "answer": run["answer"], "contexts": retrieved}
        for record, run, retrieved in zip(
            records, agent_runs, contexts, strict=True
        )
    ]

    ragas_evaluator.upload_dataset_to_langsmith(records, dataset_name)
    results_df = ragas_evaluator.evaluate(live_records)
    logger.info("=== PER SAMPLE RESULTS ===\n%s", results_df)
    ragas_evaluator.log_to_langsmith(
        live_records, results_df, dataset_name, experiment_name
    )

    metrics = {
        **ragas_evaluator.summarize(results_df),
        **metrics_evaluator.recall_at_k(records),
        **metrics_evaluator.evaluate_latency(agent_runs),
        **metrics_evaluator.evaluate_cost(records, agent_runs),
    }
    logger.info("=== METRICS ===")
    for name, value in metrics.items():
        logger.info("%s: %s", name, value)
    metrics_evaluator.log_metrics_to_langsmith(metrics, experiment_name)


if __name__ == "__main__":
    main()
