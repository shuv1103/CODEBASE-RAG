"""Loader and validator for the evaluation dataset."""

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints, TypeAdapter

NonBlank = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


class EvalRecord(BaseModel):
    """One labeled evaluation question.

    Attributes:
        contexts: Ground-truth "path: description" strings, used for
            Recall@K.
        ground_truth: Reference answer the RAGAS judge compares against.
    """

    question: NonBlank
    answer: NonBlank
    contexts: list[NonBlank] = Field(min_length=1)
    ground_truth: NonBlank


_DATASET = TypeAdapter(list[EvalRecord])


def load_dataset(file_path: str | Path) -> list[dict[str, Any]]:
    """Load the JSON evaluation dataset and validate every record.

    Args:
        file_path: JSON file holding a top-level array of EvalRecord objects.

    Returns:
        The records as parsed (unstripped) dicts.

    Raises:
        FileNotFoundError: If the path is not an existing file.
        ValueError: If the JSON is invalid, not an array, or any record is
            invalid; pydantic names each failure as "<index>.<field>".
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    _DATASET.validate_python(records)
    return records
