import os
from dotenv import load_dotenv

from .loader import EvaluationDatasetLoader
from .ragas_eval import RagasEvaluator

load_dotenv()


def main():
    dataset_path = os.getenv("EVALUATION_DATASET_PATH")
    if not dataset_path:
        raise ValueError("EVALUATION_DATASET_PATH missing in .env")

    ls_dataset_name = os.getenv("LANGSMITH_EVAL_DATASET_NAME", "codebase-rag-eval")
    ls_experiment_name = os.getenv("LANGSMITH_EVAL_EXPERIMENT_NAME", "codebase-rag-eval")

    print("Loading dataset...\n")
    records = EvaluationDatasetLoader.load_json(dataset_path)

    evaluator = RagasEvaluator()

    print("Uploading evaluation dataset to LangSmith...\n")
    evaluator.upload_dataset_to_langsmith(records, ls_dataset_name)

    print("Running RAG evaluation...\n")
    results_df = evaluator.evaluate(records)

    print("\n=== PER SAMPLE RESULTS ===")
    print(results_df)

    summary = evaluator.summarize(results_df)

    print("\n=== FINAL METRICS ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nLogging results to LangSmith...\n")
    evaluator.log_to_langsmith(records, results_df, ls_dataset_name, ls_experiment_name)


if __name__ == "__main__":
    main()