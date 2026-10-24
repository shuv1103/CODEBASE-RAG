"""Iteration helpers."""

from collections.abc import Iterator
from typing import TypeVar

T = TypeVar("T")


def batchify(items: list[T], batch_size: int) -> Iterator[list[T]]:
    """Yield consecutive slices of at most batch_size items.

    Args:
        items: Items to split.
        batch_size: Maximum items per batch.

    Yields:
        Lists of up to batch_size items, in order.

    Example:
        >>> list(batchify([1, 2, 3], 2))
        [[1, 2], [3]]
    """
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]
