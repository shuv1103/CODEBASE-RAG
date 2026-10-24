"""Pipeline metrics on the evaluation dataset: Recall@K, latency and cost.

MetricsEvaluator:
- recall_at_k(): retrieval quality (Recall@K).
- run_agent(): one live agent run per question, recording latency and tokens.
- evaluate_latency(): P50/P95/P99 end-to-end latency from run_agent() output.
- evaluate_cost(): mean/P95/total cost per query from run_agent() output.
- log_metrics_to_langsmith(): store the aggregate metrics as a LangSmith run.
"""

import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any

import litellm
import numpy as np
from google.adk.runners import InMemoryRunner
from google.genai import types

from codebase_rag_agent.agent import root_agent
from embeddings.embeddings_formatter import estimate_token_count
from retrieval.retrieval_service import RetrievalService
from utils.config import settings
from utils.model_config import litellm_model_name

logger = logging.getLogger(__name__)


def relevant_files_from_contexts(contexts: list[str]) -> set[str]:
    """Parse ground-truth files from each context's "path: description" prefix.

    Args:
        contexts: A dataset record's context strings.

    Returns:
        The set of file paths; contexts without a ":" are ignored.

    Example:
        >>> relevant_files_from_contexts(["backend/A.java: handles auth", "no colon"])
        {'backend/A.java'}
    """
    return {ctx.split(":", 1)[0].strip() for ctx in contexts if ":" in ctx}


def normalize_path(path: str) -> str:
    """Normalize a path to forward slashes with no leading slash.

    Args:
        path: A Windows- or POSIX-style relative path.

    Returns:
        The normalized path.

    Example:
        >>> normalize_path("\\\\Repo\\\\backend\\\\A.java")
        'Repo/backend/A.java'
    """
    return path.replace("\\", "/").strip().lstrip("/")


def matches_relevant_file(retrieved_path: str, relevant_path: str) -> bool:
    """Check whether a retrieved path refers to a ground-truth relevant file.

    Match by normalized suffix, not equality: stored paths use OS separators
    and start at the folder given to RepoLoader ('Repo\\backend\\A.java' on
    Windows), while dataset paths use forward slashes without that root
    ('backend/A.java'). Exact comparison would never match.

    Args:
        retrieved_path: relative_path stored with a retrieved chunk.
        relevant_path: Path from the eval dataset's contexts.

    Returns:
        True if the normalized paths are equal or the retrieved path ends
        with "/<relevant path>".
    """
    retrieved_norm = normalize_path(retrieved_path)
    relevant_norm = normalize_path(relevant_path)
    return retrieved_norm == relevant_norm or retrieved_norm.endswith(
        f"/{relevant_norm}"
    )


async def run_query(
    runner: InMemoryRunner, user_id: str, session_id: str, question: str
) -> dict[str, float]:
    """Run one question through the live agent, measuring latency and tokens.

    Args:
        runner: Runner wrapping the agent under test.
        user_id: ADK user id for the session.
        session_id: ADK session to run the question in.
        question: The question to ask.

    Returns:
        {"latency_ms", "prompt_tokens", "completion_tokens"}, with tokens
        summed across every model call in the turn.
    """
    start = time.perf_counter()
    prompt_tokens = 0
    completion_tokens = 0

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user", parts=[types.Part(text=question)]
        ),
    ):
        usage = getattr(event, "usage_metadata", None)
        if usage:
            prompt_tokens += usage.prompt_token_count or 0
            completion_tokens += usage.candidates_token_count or 0

    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


class MetricsEvaluator:
    """RAG pipeline metrics on the eval dataset: Recall@K, latency and cost.

    Example:
        >>> evaluator = MetricsEvaluator()
        >>> recall = evaluator.recall_at_k(records)
        >>> runs = evaluator.run_agent(records)
        >>> latency = MetricsEvaluator.evaluate_latency(runs)
        >>> cost = evaluator.evaluate_cost(records, runs)
    """

    def __init__(self) -> None:
        self._retrieval_service: RetrievalService | None = None
        if not settings.agent_model:
            raise ValueError("AGENT_MODEL is required to price agent runs")
        self._agent_model = litellm_model_name(settings.agent_model)
        self._embedding_model = litellm_model_name(settings.embedding_model)

    def get_retrieval_service(self) -> RetrievalService:
        """Return the default RetrievalService, building it on first use.

        Returns:
            The cached RetrievalService.
        """
        if self._retrieval_service is None:
            self._retrieval_service = RetrievalService()
        return self._retrieval_service

    def recall_at_k(
        self,
        records: list[dict[str, Any]],
        k_values: Sequence[int] = (3, 5, 10),
    ) -> dict[str, float]:
        """Compute average Recall@K across the dataset for each K in k_values.

        Make one retrieval per record (top_k=max(k_values)) and score each
        smaller K on a prefix of it. Skip records with no parseable relevant
        files.

        Args:
            records: Eval dataset records.
            k_values: K values to report (default: 3, 5, 10).

        Returns:
            {"recall_at_<k>": mean recall} for each K; 0.0 if no record was
            scorable.
        """
        service = self.get_retrieval_service()
        max_k = max(k_values)
        per_k_scores: dict[int, list[float]] = {k: [] for k in k_values}

        for record in records:
            relevant = relevant_files_from_contexts(record.get("contexts", []))
            if not relevant:
                continue

            response = service.retrieve(query=record["question"], top_k=max_k)
            retrieved_ranked = [c.relative_path for c in response.results]

            for k in k_values:
                retrieved_top_k = retrieved_ranked[:k]
                hits = {
                    rel
                    for rel in relevant
                    if any(
                        matches_relevant_file(rp, rel)
                        for rp in retrieved_top_k
                    )
                }
                per_k_scores[k].append(len(hits) / len(relevant))

        return {
            f"recall_at_{k}": (sum(scores) / len(scores) if scores else 0.0)
            for k, scores in per_k_scores.items()
        }

    async def run_all_queries(
        self, records: list[dict[str, Any]]
    ) -> list[dict[str, float]]:
        """Run every record's question sequentially, each in a fresh session.

        Args:
            records: Eval dataset records.

        Returns:
            One run_query result per record, in order.
        """
        runner = InMemoryRunner(agent=root_agent)
        results = []

        for i, record in enumerate(records):
            user_id = f"metrics-eval-{i}"
            session = await runner.session_service.create_session(
                app_name=runner.app_name, user_id=user_id
            )
            results.append(
                await run_query(
                    runner, user_id, session.id, record["question"]
                )
            )

        return results

    def run_agent(
        self, records: list[dict[str, Any]]
    ) -> list[dict[str, float]]:
        """Run the live agent once per dataset question.

        Feed this same result into both evaluate_latency() and
        evaluate_cost() to avoid a second pass.

        Args:
            records: Eval dataset records.

        Returns:
            Raw per-query {"latency_ms", "prompt_tokens",
            "completion_tokens"} dicts.
        """
        return asyncio.run(self.run_all_queries(records))

    @staticmethod
    def evaluate_latency(
        agent_runs: list[dict[str, float]],
    ) -> dict[str, float]:
        """Compute P50/P95/P99 end-to-end latency (request to full response).

        Args:
            agent_runs: Output of run_agent().

        Returns:
            latency_p50_ms / latency_p95_ms / latency_p99_ms (0.0 when there
            are no runs) and n_queries.
        """
        latencies = [r["latency_ms"] for r in agent_runs]
        p50, p95, p99 = (
            np.percentile(latencies, [50, 95, 99])
            if latencies
            else (0.0, 0.0, 0.0)
        )
        return {
            "latency_p50_ms": float(p50),
            "latency_p95_ms": float(p95),
            "latency_p99_ms": float(p99),
            "n_queries": len(agent_runs),
        }

    def evaluate_cost(
        self,
        records: list[dict[str, Any]],
        agent_runs: list[dict[str, float]],
    ) -> dict[str, float]:
        """Compute mean/P95/total cost per query in USD from litellm's prices.

        Args:
            records: Eval dataset records, aligned with agent_runs.
            agent_runs: Output of run_agent().

        Returns:
            cost_mean_usd, cost_p95_usd, and cost_total_usd.
        """
        costs = [
            self.query_cost_usd(
                run["prompt_tokens"],
                run["completion_tokens"],
                record["question"],
            )
            for run, record in zip(agent_runs, records, strict=True)
        ]
        return {
            "cost_mean_usd": float(sum(costs) / len(costs)) if costs else 0.0,
            "cost_p95_usd": float(np.percentile(costs, 95)) if costs else 0.0,
            "cost_total_usd": float(sum(costs)),
        }

    def query_cost_usd(
        self, prompt_tokens: int, completion_tokens: int, question: str
    ) -> float:
        """Price one query: LLM call cost plus query-embedding cost.

        A model litellm cannot price contributes 0.0 instead of raising.

        Args:
            prompt_tokens: Total prompt tokens across the turn.
            completion_tokens: Total completion tokens across the turn.
            question: Query text, used to estimate embedding tokens.

        Returns:
            Total cost in USD.
        """
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=self._agent_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except Exception:
            prompt_cost = completion_cost = 0.0

        try:
            embedding_cost, _ = litellm.cost_per_token(
                model=self._embedding_model,
                prompt_tokens=estimate_token_count(question),
                completion_tokens=0,
            )
        except Exception:
            embedding_cost = 0.0

        return prompt_cost + completion_cost + embedding_cost

    def log_metrics_to_langsmith(
        self, metrics: dict[str, Any], experiment_name: str
    ) -> None:
        """Persist aggregate pipeline metrics as one LangSmith run.

        Name the run "<experiment_name>-pipeline-metrics". Skip it when
        LangSmith tracing is disabled.

        Args:
            metrics: Metric name -> value, stored as the run's outputs.
            experiment_name: Experiment label used in the run name.
        """
        if not settings.tracing_enabled:
            logger.info(
                "LangSmith tracing disabled — skipping metrics logging."
            )
            return

        import langsmith

        client = langsmith.Client()
        client.create_run(
            name=f"{experiment_name}-pipeline-metrics",
            run_type="chain",
            inputs={
                "experiment_name": experiment_name,
                "n_queries": metrics.get("n_queries"),
            },
            outputs=metrics,
            project_name=settings.langchain_project,
        )
        logger.info(
            "Logged pipeline metrics to LangSmith project %r.",
            settings.langchain_project,
        )
