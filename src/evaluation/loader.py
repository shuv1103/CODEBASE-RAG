import json
from pathlib import Path
from typing import Any, Dict, List


class EvaluationDatasetLoader:
    """
    Loads and validates a RAG evaluation dataset from a JSON file.

    Expected schema for each record:
    {
        "question": str,
        "answer": str,
        "contexts": list[str],
        "ground_truth": str
    }
    """

    REQUIRED_FIELDS = {"question", "answer", "contexts", "ground_truth"}

    @classmethod
    def load_json(cls, file_path: str | Path) -> List[Dict[str, Any]]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {path}")

        if not path.is_file():
            raise ValueError(f"Provided path is not a file: {path}")

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("Evaluation dataset must be a JSON array of records.")

        cls._validate(data)
        return data

    @classmethod
    def _validate(cls, data: List[Dict[str, Any]]) -> None:
        for idx, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValueError(f"Record at index {idx} must be a JSON object.")

            missing = cls.REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(
                    f"Record at index {idx} is missing required fields: {sorted(missing)}"
                )

            if not isinstance(row["question"], str) or not row["question"].strip():
                raise ValueError(f"Record {idx}: 'question' must be a non-empty string.")

            if not isinstance(row["answer"], str) or not row["answer"].strip():
                raise ValueError(f"Record {idx}: 'answer' must be a non-empty string.")

            if not isinstance(row["ground_truth"], str) or not row["ground_truth"].strip():
                raise ValueError(f"Record {idx}: 'ground_truth' must be a non-empty string.")

            if not isinstance(row["contexts"], list) or not row["contexts"]:
                raise ValueError(f"Record {idx}: 'contexts' must be a non-empty list.")

            for c_idx, context in enumerate(row["contexts"]):
                if not isinstance(context, str) or not context.strip():
                    raise ValueError(
                        f"Record {idx}: context at index {c_idx} must be a non-empty string."
                    )
