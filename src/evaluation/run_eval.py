import os
from dotenv import load_dotenv

from .loader import EvaluationDatasetLoader
from .ragas_eval import RagasEvaluator

load_dotenv()


def main():
    dataset_path = os.getenv("EVALUATION_DATASET_PATH")

    if not dataset_path:
        raise ValueError("EVALUATION_DATASET_PATH missing in .env")

    print("Loading dataset...\n")

    # ✅ Use validated loader
    records = EvaluationDatasetLoader.load_json(dataset_path)

    evaluator = RagasEvaluator()

    print("Running RAG evaluation...\n")

    results_df = evaluator.evaluate(records)

    print("\n=== PER SAMPLE RESULTS ===")
    print(results_df)

    summary = evaluator.summarize(results_df)

    print("\n=== FINAL METRICS ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()