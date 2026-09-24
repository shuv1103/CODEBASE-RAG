import json
from pathlib import Path
from typing import Any, Dict, List


class EvaluationDatasetLoader:
    """Loads and validates a RAG evaluation dataset from a JSON file.

    Expected schema for each record:
    {
        "question": str,
        "answer": str,
        "contexts": list[str],
        "ground_truth": str
    }

    Attributes:
        REQUIRED_FIELDS: Keys every record must contain.

    Example:
        >>> records = EvaluationDatasetLoader.load_json("evaluation/dataset.json")
    """

    REQUIRED_FIELDS = {"question", "answer", "contexts", "ground_truth"}

    @classmethod
    def load_json(cls, file_path: str | Path) -> List[Dict[str, Any]]:
        """Load a dataset file and validate every record.

        Args:
            file_path: Path of the JSON dataset (a top-level array).

        Returns:
            The validated list of records.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the path is not a file, the JSON is not an
                array, or any record fails validation.
        """
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
        """Check each record's required fields, types, and non-emptiness.

        Args:
            data: Parsed dataset records.

        Raises:
            ValueError: On the first invalid record, naming its index.
        """
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
