"""Loader and validator for the evaluation dataset."""

import json
from pathlib import Path
from typing import Any


class EvaluationDatasetLoader:
    """Loader and validator for a RAG evaluation dataset in JSON.

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

    REQUIRED_FIELDS = frozenset(
        {"question", "answer", "contexts", "ground_truth"}
    )

    @classmethod
    def load_json(cls, file_path: str | Path) -> list[dict[str, Any]]:
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
            raise ValueError(
                "Evaluation dataset must be a JSON array of records."
            )

        cls.validate(data)
        return data

    @classmethod
    def validate(cls, data: list[dict[str, Any]]) -> None:
        """Check each record's required fields, types, and non-emptiness.

        Args:
            data: Parsed dataset records.

        Raises:
            ValueError: On the first invalid record, naming its index.
        """
        for idx, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValueError(
                    f"Record at index {idx} must be a JSON object."
                )

            missing = cls.REQUIRED_FIELDS - row.keys()
            if missing:
                raise ValueError(
                    f"Record at index {idx} is missing required fields: "
                    f"{sorted(missing)}"
                )

            for field in ("question", "answer", "ground_truth"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(
                        f"Record {idx}: '{field}' must be a non-empty string."
                    )

            if not isinstance(row["contexts"], list) or not row["contexts"]:
                raise ValueError(
                    f"Record {idx}: 'contexts' must be a non-empty list."
                )

            for c_idx, context in enumerate(row["contexts"]):
                if not isinstance(context, str) or not context.strip():
                    raise ValueError(
                        f"Record {idx}: context at index {c_idx} must be a "
                        "non-empty string."
                    )
