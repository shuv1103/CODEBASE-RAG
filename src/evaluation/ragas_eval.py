import os
from typing import List, Dict, Any

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
import logging
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
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

class RagasEvaluator:
    """
    Evaluates RAG system performance using RAGAS.

    Metrics:
    - faithfulness → hallucination check
    - answer_relevancy → generation quality
    - context_precision → retrieval relevance
    - context_recall → retrieval completeness
    """

    def __init__(self) -> None:
        self._api_key = os.getenv("GOOGLE_API_KEY")
        self._llm_model = os.getenv("EVALUATION_LLM_MODEL", "gemini-2.0-flash")
        self._embedding_model = os.getenv(
            "EMBEDDING_MODEL", "models/text-embedding-004"
        )

        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY missing in .env")

        # -------------------------------
        # LLM (for evaluation reasoning)
        # -------------------------------
        llm = ChatGoogleGenerativeAI(
            model=self._llm_model,
            google_api_key=self._api_key,
            temperature=0.0,
        )
        self._ragas_llm = LangchainLLMWrapper(llm)

        # -------------------------------
        # Embeddings (CRITICAL FIX)
        # -------------------------------
        embeddings = GoogleGenerativeAIEmbeddings(
            model=self._embedding_model,
            google_api_key=self._api_key,
        )
        self._ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # -------------------------------
    # DATASET CONVERSION
    # -------------------------------
    @staticmethod
    def _to_dataset(records: List[Dict[str, Any]]) -> Dataset:
        return Dataset.from_dict({
            "question": [r["question"] for r in records],
            "answer": [r["answer"] for r in records],
            "contexts": [r["contexts"] for r in records],
            "ground_truth": [r["ground_truth"] for r in records],
        })

    # -------------------------------
    # MAIN EVALUATION
    # -------------------------------
    def evaluate(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        dataset = self._to_dataset(records)

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
            run_config=RunConfig(max_workers = 1, timeout = 300),
            batch_size=1,
            raise_exceptions=True
        )

        return result.to_pandas()

    # -------------------------------
    # SUMMARY
    # -------------------------------
    @staticmethod
    def summarize(results_df: pd.DataFrame) -> Dict[str, float]:
        metrics = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

        return {
            metric: round(float(results_df[metric].mean()), 4)
            for metric in metrics
            if metric in results_df.columns
        }