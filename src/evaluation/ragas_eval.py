import asyncio
import os
import time
import warnings
from typing import List, Dict, Any

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
import logging

# instructor (pulled in by ragas) imports the deprecated google.generativeai package;
# suppress its FutureWarning since we are not using Gemini here.
warnings.filterwarnings("ignore", category=FutureWarning, module="instructor")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

load_dotenv()
logging.basicConfig()

_LANGSMITH_ENABLED = (
    os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    and bool(os.getenv("LANGCHAIN_API_KEY"))
)

_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

_REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "OPENAI_LLM_MODEL",
    "OPENAI_EMBEDDING_MODEL",
]


def _validate_env() -> None:
    missing = [var for var in _REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


# Throttling is applied at the LangChain layer by subclassing the concrete OpenAI
# classes (plain Pydantic models). ragas's LangchainLLMWrapper /
# LangchainEmbeddingsWrapper are deprecated helpers that cannot be subclassed
# in newer ragas versions (DeprecationHelper blocks it).

class _ThrottledChatOpenAI(ChatOpenAI):
    """ChatOpenAI with a per-call sleep to honour OpenAI rate limits."""

    sleep_seconds: float = 2.0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(self.sleep_seconds)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        await asyncio.sleep(self.sleep_seconds)
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


class _ThrottledOpenAIEmbeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings with a per-call sleep to honour OpenAI rate limits."""

    sleep_seconds: float = 1.0

    def embed_documents(self, texts, **kwargs):
        time.sleep(self.sleep_seconds)
        return super().embed_documents(texts, **kwargs)

    def embed_query(self, text, **kwargs):
        time.sleep(self.sleep_seconds)
        return super().embed_query(text, **kwargs)

    async def aembed_documents(self, texts, **kwargs):
        await asyncio.sleep(self.sleep_seconds)
        return await super().aembed_documents(texts, **kwargs)

    async def aembed_query(self, text, **kwargs):
        await asyncio.sleep(self.sleep_seconds)
        return await super().aembed_query(text, **kwargs)


class RagasEvaluator:
    """
    Evaluates RAG system performance using RAGAS with OpenAI.

    Metrics:
    - faithfulness       → hallucination check
    - answer_relevancy   → generation quality
    - context_precision  → retrieval relevance
    - context_recall     → retrieval completeness
    """

    def __init__(self) -> None:
        _validate_env()

        llm = _ThrottledChatOpenAI(
            model=os.getenv("OPENAI_LLM_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.0,
            sleep_seconds=2.0,
        )
        self._ragas_llm = LangchainLLMWrapper(llm)

        embeddings = _ThrottledOpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            sleep_seconds=1.0,
        )
        self._ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # ------------------------------------------------------------------
    # DATASET CONVERSION
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dataset(records: List[Dict[str, Any]]) -> Dataset:
        return Dataset.from_dict({
            "question":     [r["question"]     for r in records],
            "answer":       [r["answer"]       for r in records],
            "contexts":     [r["contexts"]     for r in records],
            "ground_truth": [r["ground_truth"] for r in records],
        })

    # ------------------------------------------------------------------
    # MAIN EVALUATION
    # ------------------------------------------------------------------

    def evaluate(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        dataset = self._to_dataset(records)

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=self._ragas_llm,
            embeddings=self._ragas_embeddings,
            run_config=RunConfig(max_workers=1, timeout=600),
            batch_size=1,
            raise_exceptions=True,
        )

        return result.to_pandas()

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    @staticmethod
    def summarize(results_df: pd.DataFrame) -> Dict[str, float]:
        return {
            metric: round(float(results_df[metric].mean()), 4)
            for metric in _METRICS
            if metric in results_df.columns
        }

    # ------------------------------------------------------------------
    # LANGSMITH — upload evaluation dataset
    # ------------------------------------------------------------------

    def upload_dataset_to_langsmith(
        self,
        records: List[Dict[str, Any]],
        dataset_name: str,
    ) -> None:
        if not _LANGSMITH_ENABLED:
            print("LangSmith tracing disabled — skipping dataset upload.")
            return

        import langsmith

        client = langsmith.Client()

        try:
            dataset = client.create_dataset(dataset_name=dataset_name)
        except Exception:
            dataset = client.read_dataset(dataset_name=dataset_name)

        client.create_examples(
            inputs=[{"question": r["question"]} for r in records],
            outputs=[{"ground_truth": r["ground_truth"]} for r in records],
            dataset_id=dataset.id,
        )
        print(f"Uploaded {len(records)} examples to LangSmith dataset '{dataset_name}'.")

    # ------------------------------------------------------------------
    # LANGSMITH — log evaluation results
    # ------------------------------------------------------------------

    def log_to_langsmith(
        self,
        records: List[Dict[str, Any]],
        results_df: pd.DataFrame,
        dataset_name: str,
        experiment_name: str,
    ) -> None:
        if not _LANGSMITH_ENABLED:
            print("LangSmith tracing disabled — skipping results logging.")
            return

        import langsmith

        client = langsmith.Client()
        dataset = client.read_dataset(dataset_name=dataset_name)

        for idx, record in enumerate(records):
            row = results_df.iloc[idx]
            client.create_run(
                name=experiment_name,
                run_type="chain",
                inputs={"question": record["question"]},
                outputs={
                    "answer":            record["answer"],
                    "faithfulness":      float(row.get("faithfulness",      float("nan"))),
                    "answer_relevancy":  float(row.get("answer_relevancy",  float("nan"))),
                    "context_precision": float(row.get("context_precision", float("nan"))),
                    "context_recall":    float(row.get("context_recall",    float("nan"))),
                },
                reference_dataset_id=dataset.id,
            )
        print(f"Logged {len(records)} results to LangSmith experiment '{experiment_name}'.")
