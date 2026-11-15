"""RAGAS scoring of answers, with optional LangSmith logging."""

import asyncio
import logging
import time
import warnings
from typing import Any

import pandas as pd
from datasets import Dataset

# instructor (via ragas) imports the deprecated google.generativeai package;
# Gemini isn't used, so its warning is noise.
warnings.filterwarnings("ignore", category=FutureWarning, module="instructor")

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness
from ragas.run_config import RunConfig

from utils.config import settings

logger = logging.getLogger(__name__)

_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
_METRIC_NAMES = [metric.name for metric in _METRICS]


def validate_env() -> None:
    """Fail fast if the RAGAS judge models are not configured.

    Raises:
        ValueError: If OPENAI_LLM_MODEL or OPENAI_EMBEDDING_MODEL is unset.
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


# Throttling subclasses LangChain's classes: newer ragas blocks subclassing
# its own (deprecated) wrappers.


class _ThrottledChatOpenAI(ChatOpenAI):
    """ChatOpenAI that sleeps sleep_seconds before each call (rate limits)."""

    sleep_seconds: float = 2.0

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Sleep, then generate."""
        time.sleep(self.sleep_seconds)
        return super()._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Sleep, then generate."""
        await asyncio.sleep(self.sleep_seconds)
        return await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )


class _ThrottledOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings that sleeps sleep_seconds before each call."""

    sleep_seconds: float = 1.0

    def embed_documents(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        """Sleep, then embed."""
        time.sleep(self.sleep_seconds)
        return super().embed_documents(texts, **kwargs)

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Sleep, then embed."""
        time.sleep(self.sleep_seconds)
        return super().embed_query(text, **kwargs)

    async def aembed_documents(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        """Sleep, then embed."""
        await asyncio.sleep(self.sleep_seconds)
        return await super().aembed_documents(texts, **kwargs)

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Sleep, then embed."""
        await asyncio.sleep(self.sleep_seconds)
        return await super().aembed_query(text, **kwargs)


class RagasEvaluator:
    """Scores answers with RAGAS, judged by OPENAI_LLM_MODEL.

    Metrics: faithfulness, answer relevancy, context precision and recall.

    Raises:
        ValueError: If either judge model is not configured.

    Example:
        >>> evaluator = RagasEvaluator()
        >>> RagasEvaluator.summarize(evaluator.evaluate(records))
    """

    def __init__(self) -> None:
        validate_env()
        api_key = settings.openai_api_key.get_secret_value()
        self._ragas_llm = LangchainLLMWrapper(
            _ThrottledChatOpenAI(
                model=settings.openai_llm_model,
                api_key=api_key,
                temperature=0.0,
            )
        )
        self._ragas_embeddings = LangchainEmbeddingsWrapper(
            _ThrottledOpenAIEmbeddings(
                model=settings.openai_embedding_model, api_key=api_key
            )
        )

    @staticmethod
    def to_dataset(records: list[dict[str, Any]]) -> Dataset:
        """Convert records into the column-oriented Dataset RAGAS expects.

        Args:
            records: Dicts with question, answer, contexts and ground_truth.

        Returns:
            A Dataset with one column per field.
        """
        return Dataset.from_dict(
            {
                column: [record[column] for record in records]
                for column in (
                    "question",
                    "answer",
                    "contexts",
                    "ground_truth",
                )
            }
        )

    def evaluate(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Score every record, serially to stay under OpenAI rate limits.

        Args:
            records: Dicts with question, answer, contexts and ground_truth.

        Returns:
            One row per record, with a column per metric.
        """
        result = evaluate(
            dataset=self.to_dataset(records),
            metrics=_METRICS,
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
            Metric name -> mean score, rounded to 4 places.
        """
        return {
            metric: round(float(results_df[metric].mean()), 4)
            for metric in _METRIC_NAMES
            if metric in results_df.columns
        }

    @staticmethod
    def upload_dataset_to_langsmith(
        records: list[dict[str, Any]], dataset_name: str
    ) -> None:
        """Add questions not yet in a LangSmith dataset, with ground truths.

        Creates the dataset if needed; skipped while tracing is off.

        Args:
            records: Evaluation records.
            dataset_name: LangSmith dataset to create or add new questions to.
        """
        if not settings.tracing_enabled:
            logger.info("LangSmith tracing disabled; skipping dataset upload")
            return

        import langsmith
        from langsmith.utils import LangSmithConflictError

        client = langsmith.Client()
        try:
            dataset = client.create_dataset(dataset_name=dataset_name)
        except LangSmithConflictError:
            dataset = client.read_dataset(dataset_name=dataset_name)

        # Re-runs must not append the same examples again.
        existing = {
            example.inputs.get("question")
            for example in client.list_examples(dataset_id=dataset.id)
        }
        new_records = [r for r in records if r["question"] not in existing]
        if new_records:
            client.create_examples(
                inputs=[{"question": r["question"]} for r in new_records],
                outputs=[
                    {"ground_truth": r["ground_truth"]} for r in new_records
                ],
                dataset_id=dataset.id,
            )
        logger.info(
            "Uploaded %d new examples to LangSmith dataset %r",
            len(new_records),
            dataset_name,
        )

    @staticmethod
    def log_to_langsmith(
        records: list[dict[str, Any]],
        results_df: pd.DataFrame,
        dataset_name: str,
        experiment_name: str,
    ) -> None:
        """Log one run per record with its answer and scores.

        Skipped while tracing is off.

        Args:
            records: Scored records, aligned with results_df rows.
            results_df: Output of evaluate().
            dataset_name: Existing LangSmith dataset the runs reference.
            experiment_name: Name of each logged run.
        """
        if not settings.tracing_enabled:
            logger.info("LangSmith tracing disabled; skipping results upload")
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
                        for metric in _METRIC_NAMES
                    },
                },
                reference_dataset_id=dataset.id,
            )
        logger.info(
            "Logged %d results to LangSmith experiment %r",
            len(records),
            experiment_name,
        )
