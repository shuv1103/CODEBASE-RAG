"""RAGAS evaluation of answers, with optional LangSmith logging."""

import asyncio
import logging
import time
import warnings
from typing import Any

import pandas as pd
from datasets import Dataset

# instructor (a ragas dependency) imports the deprecated
# google.generativeai package; Gemini isn't used, so its warning is noise.
warnings.filterwarnings("ignore", category=FutureWarning, module="instructor")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness

from utils.config import settings

logger = logging.getLogger(__name__)

_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def validate_env() -> None:
    """Fail fast if the RAGAS evaluator's model settings are unset.

    Raises:
        ValueError: If OPENAI_LLM_MODEL or OPENAI_EMBEDDING_MODEL is missing.
    """
    missing = [
        name
        for name, value in (
            ("OPENAI_LLM_MODEL", settings.openai_llm_model),
            ("OPENAI_EMBEDDING_MODEL", settings.openai_embedding_model),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


# Throttling subclasses the LangChain OpenAI classes, because newer ragas
# versions block subclassing its own (deprecated) wrapper classes.

class _ThrottledChatOpenAI(ChatOpenAI):
    """ChatOpenAI with a per-call sleep to honour OpenAI rate limits.

    Attributes:
        sleep_seconds: Delay before every generation call.
    """

    sleep_seconds: float = 2.0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Sleep sleep_seconds, then call ChatOpenAI._generate."""
        time.sleep(self.sleep_seconds)
        return super()._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """Sleep sleep_seconds, then call ChatOpenAI._agenerate."""
        await asyncio.sleep(self.sleep_seconds)
        return await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )


class _ThrottledOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings with a per-call sleep to honour OpenAI rate limits.

    Attributes:
        sleep_seconds: Delay before every embedding call.
    """

    sleep_seconds: float = 1.0

    def embed_documents(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        """Sleep sleep_seconds, then call OpenAIEmbeddings.embed_documents."""
        time.sleep(self.sleep_seconds)
        return super().embed_documents(texts, **kwargs)

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Sleep sleep_seconds, then call OpenAIEmbeddings.embed_query."""
        time.sleep(self.sleep_seconds)
        return super().embed_query(text, **kwargs)

    async def aembed_documents(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        """Sleep sleep_seconds, then call OpenAIEmbeddings.aembed_documents."""
        await asyncio.sleep(self.sleep_seconds)
        return await super().aembed_documents(texts, **kwargs)

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Sleep sleep_seconds, then call OpenAIEmbeddings.aembed_query."""
        await asyncio.sleep(self.sleep_seconds)
        return await super().aembed_query(text, **kwargs)


class RagasEvaluator:
    """RAGAS evaluator of RAG answers, judged by OpenAI models.

    Metrics:
    - faithfulness: whether the answer is grounded (hallucination check).
    - answer_relevancy: generation quality.
    - context_precision: retrieval relevance.
    - context_recall: retrieval completeness.

    Raises:
        ValueError: If OPENAI_LLM_MODEL or OPENAI_EMBEDDING_MODEL is not set.

    Example:
        >>> evaluator = RagasEvaluator()
        >>> results_df = evaluator.evaluate(records)
        >>> print(RagasEvaluator.summarize(results_df))
    """

    def __init__(self) -> None:
        validate_env()

        llm = _ThrottledChatOpenAI(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key.get_secret_value(),
            temperature=0.0,
            sleep_seconds=2.0,
        )
        self._ragas_llm = LangchainLLMWrapper(llm)

        embeddings = _ThrottledOpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key.get_secret_value(),
            sleep_seconds=1.0,
        )
        self._ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    @staticmethod
    def to_dataset(records: list[dict[str, Any]]) -> Dataset:
        """Convert eval records into a column-oriented HuggingFace Dataset.

        Args:
            records: Validated eval dataset records.

        Returns:
            Dataset with question, answer, contexts, and ground_truth columns.
        """
        return Dataset.from_dict({
            "question":     [r["question"]     for r in records],
            "answer":       [r["answer"]       for r in records],
            "contexts":     [r["contexts"]     for r in records],
            "ground_truth": [r["ground_truth"] for r in records],
        })

    def evaluate(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Score every record on the four RAGAS metrics.

        Run serially (one worker, batch size 1) to stay under OpenAI rate
        limits.

        Args:
            records: Validated eval dataset records.

        Returns:
            One row per record with a column per metric.
        """
        dataset = self.to_dataset(records)

        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=self._ragas_llm,
            embeddings=self._ragas_embeddings,
            run_config=RunConfig(max_workers=1, timeout=600),
            batch_size=1,
            raise_exceptions=True,
        )

        return result.to_pandas()

    @staticmethod
    def summarize(results_df: pd.DataFrame) -> dict[str, float]:
        """Average each RAGAS metric column.

        Args:
            results_df: Output of evaluate().

        Returns:
            Metric name -> mean score rounded to 4 places, for metrics
            present in the frame.
        """
        return {
            metric: round(float(results_df[metric].mean()), 4)
            for metric in _METRICS
            if metric in results_df.columns
        }

    def upload_dataset_to_langsmith(
        self,
        records: list[dict[str, Any]],
        dataset_name: str,
    ) -> None:
        """Upload questions and ground truths as LangSmith dataset examples.

        Create the dataset if needed, otherwise append to it. Skip when
        LangSmith tracing is disabled.

        Args:
            records: Validated eval dataset records.
            dataset_name: LangSmith dataset to create or reuse.
        """
        if not settings.tracing_enabled:
            logger.info("LangSmith tracing disabled — skipping dataset upload.")
            return

        import langsmith
        from langsmith.utils import LangSmithConflictError

        client = langsmith.Client()

        try:
            dataset = client.create_dataset(dataset_name=dataset_name)
        except LangSmithConflictError:
            dataset = client.read_dataset(dataset_name=dataset_name)

        client.create_examples(
            inputs=[{"question": r["question"]} for r in records],
            outputs=[{"ground_truth": r["ground_truth"]} for r in records],
            dataset_id=dataset.id,
        )
        logger.info(
            "Uploaded %d examples to LangSmith dataset %r.",
            len(records),
            dataset_name,
        )

    def log_to_langsmith(
        self,
        records: list[dict[str, Any]],
        results_df: pd.DataFrame,
        dataset_name: str,
        experiment_name: str,
    ) -> None:
        """Log one LangSmith run per record with its answer and RAGAS scores.

        Skip when LangSmith tracing is disabled.

        Args:
            records: Eval dataset records, aligned with results_df rows.
            results_df: Output of evaluate().
            dataset_name: Existing LangSmith dataset the runs reference.
            experiment_name: Name given to each logged run.
        """
        if not settings.tracing_enabled:
            logger.info(
                "LangSmith tracing disabled — skipping results logging."
            )
            return

        import langsmith

        client = langsmith.Client()
        dataset = client.read_dataset(dataset_name=dataset_name)

        rows = (row for _, row in results_df.iterrows())
        for record, row in zip(records, rows, strict=True):
            client.create_run(
                name=experiment_name,
                run_type="chain",
                inputs={"question": record["question"]},
                outputs={
                    "answer": record["answer"],
                    **{
                        metric: float(row.get(metric, float("nan")))
                        for metric in _METRICS
                    },
                },
                reference_dataset_id=dataset.id,
            )
        logger.info(
            "Logged %d results to LangSmith experiment %r.",
            len(records),
            experiment_name,
        )
