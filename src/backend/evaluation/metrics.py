"""Live pipeline metrics on the eval dataset: Recall@K, latency and cost."""

import asyncio
import functools
import logging
import time
from collections.abc import Sequence
from typing import Any, TypedDict

import litellm
import numpy as np
from google.adk.runners import InMemoryRunner
from google.genai import types

from codebase_rag_agent.agent import get_root_agent
from embeddings.embeddings_formatter import estimate_token_count
from retrieval.retrieval_service import RetrievalService
from utils.config import settings
from utils.model_config import litellm_model_name

logger = logging.getLogger(__name__)


class AgentRun(TypedDict):
    """One live agent turn.

    Attributes:
        answer: The agent's final answer.
        latency_ms: Request-to-answer time.
        prompt_tokens: Summed over every model call in the turn.
        completion_tokens: Summed over every model call in the turn.
    """

    answer: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


def relevant_files_from_contexts(contexts: list[str]) -> set[str]:
    """Parse ground-truth file paths from "path: description" contexts.

    Args:
        contexts: A record's context strings.

    Returns:
        The paths; contexts without ":" are ignored.

    Example:
        >>> relevant_files_from_contexts(["backend/A.java: auth", "no colon"])
        {'backend/A.java'}
    """
    return {ctx.split(":", 1)[0].strip() for ctx in contexts if ":" in ctx}


def normalize_path(path: str) -> str:
    r"""Normalize a path to forward slashes with no leading slash.

    Args:
        path: A Windows- or POSIX-style relative path.

    Returns:
        The normalized path.

    Example:
        >>> normalize_path("\\Repo\\backend\\A.java")
        'Repo/backend/A.java'
    """
    return path.replace("\\", "/").strip().lstrip("/")


def matches_relevant_file(retrieved_path: str, relevant_path: str) -> bool:
    """Check whether a retrieved path is a ground-truth file, by suffix.

    Stored paths may start with the clone's root folder and use OS
    separators; dataset paths don't.

    Args:
        retrieved_path: relative_path of a retrieved chunk.
        relevant_path: Path from the dataset's contexts.

    Returns:
        True if the normalized paths are equal or one ends with "/<other>".
    """
    retrieved = normalize_path(retrieved_path)
    relevant = normalize_path(relevant_path)
    return retrieved == relevant or retrieved.endswith(f"/{relevant}")


def token_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Price a model call with litellm's price table.

    Args:
        model: litellm model id, e.g. "openai/gpt-4o-mini".
        prompt_tokens: Input tokens.
        completion_tokens: Output tokens.

    Returns:
        Cost in USD; 0.0 for a model litellm can't price.
    """
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    # litellm raises a bare Exception for an unknown model.
    except Exception as exc:  # noqa: BLE001
        logger.warning("No price for %s (%s); counting $0", model, exc)
        return 0.0
    return prompt_cost + completion_cost


async def run_query(
    runner: InMemoryRunner, user_id: str, session_id: str, question: str
) -> AgentRun:
    """Ask the live agent one question, timing it and summing token usage.

    Args:
        runner: Runner wrapping the agent under test.
        user_id: ADK user id of the session.
        session_id: Session to run the question in.
        question: The question to ask.

    Returns:
        AgentRun with the final answer, latency and token counts.
    """
    start = time.perf_counter()
    answer = ""
    prompt_tokens = completion_tokens = 0

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user", parts=[types.Part(text=question)]
        ),
    ):
        if usage := event.usage_metadata:
            prompt_tokens += usage.prompt_token_count or 0
            completion_tokens += usage.candidates_token_count or 0
        if event.is_final_response() and event.content and event.content.parts:
            answer = "".join(part.text or "" for part in event.content.parts)

    return AgentRun(
        answer=answer,
        latency_ms=(time.perf_counter() - start) * 1000,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


class MetricsEvaluator:
    """Recall@K, latency and cost of the live pipeline, searching all repos.

    Raises:
        ValueError: If AGENT_MODEL is not set (needed for pricing).

    Example:
        >>> evaluator = MetricsEvaluator()
        >>> runs = evaluator.run_agent(records)
        >>> evaluator.evaluate_cost(records, runs)
    """

    def __init__(self) -> None:
        if not settings.agent_model:
            raise ValueError("AGENT_MODEL is required to price agent runs")
        self._agent_model = litellm_model_name(settings.agent_model)
        self._embedding_model = litellm_model_name(settings.embedding_model)

    @functools.cached_property
    def retrieval_service(self) -> RetrievalService:
        """The all-repos RetrievalService, built on first use."""
        return RetrievalService(all_repos=True)

    def recall_at_k(
        self,
        records: list[dict[str, Any]],
        k_values: Sequence[int] = (3, 5, 10),
    ) -> dict[str, float]:
        """Compute mean Recall@K over the dataset for each K.

        One retrieval at max(k_values) per record; smaller K use its prefix.
        Records whose contexts name no file are skipped.

        Args:
            records: Evaluation records.
            k_values: K values to report.

        Returns:
            {"recall_at_<k>": mean recall}, 0.0 when no record was scorable.
        """
        service = self.retrieval_service
        scores: dict[int, list[float]] = {k: [] for k in k_values}

        for record in records:
            relevant = relevant_files_from_contexts(record["contexts"])
            if not relevant:
                continue
            response = service.retrieve(
                query=record["question"], top_k=max(k_values)
            )
            ranked = [chunk.relative_path for chunk in response.results]
            for k in k_values:
                found = {
                    path
                    for path in relevant
                    if any(
                        matches_relevant_file(hit, path) for hit in ranked[:k]
                    )
                }
                scores[k].append(len(found) / len(relevant))

        return {
            f"recall_at_{k}": sum(values) / len(values) if values else 0.0
            for k, values in scores.items()
        }

    def retrieve_contexts(
        self, records: list[dict[str, Any]]
    ) -> list[list[str]]:
        """Retrieve TOP_K chunks for each record's question.

        Args:
            records: Evaluation records.

        Returns:
            Per record, the retrieved chunk texts (RAGAS "contexts").
        """
        service = self.retrieval_service
        return [
            [
                chunk.content or ""
                for chunk in service.retrieve(query=record["question"]).results
            ]
            for record in records
        ]

    def run_agent(self, records: list[dict[str, Any]]) -> list[AgentRun]:
        """Ask the live agent every question once, each in a fresh session.

        Args:
            records: Evaluation records.

        Returns:
            One AgentRun per record, in order.
        """
        return asyncio.run(self.run_all_queries(records))

    @staticmethod
    async def run_all_queries(records: list[dict[str, Any]]) -> list[AgentRun]:
        """Run the questions one at a time, so latencies don't interfere.

        Args:
            records: Evaluation records.

        Returns:
            One AgentRun per record, in order.
        """
        runner = InMemoryRunner(agent=get_root_agent())
        runs = []
        for index, record in enumerate(records):
            user_id = f"metrics-eval-{index}"
            session = await runner.session_service.create_session(
                app_name=runner.app_name, user_id=user_id
            )
            runs.append(
                await run_query(
                    runner, user_id, session.id, record["question"]
                )
            )
        return runs

    @staticmethod
    def evaluate_latency(agent_runs: list[AgentRun]) -> dict[str, float]:
        """Compute end-to-end latency percentiles.

        Args:
            agent_runs: Output of run_agent().

        Returns:
            latency_p50_ms, latency_p95_ms, latency_p99_ms (0.0 without runs)
            and n_queries.
        """
        latencies = [run["latency_ms"] for run in agent_runs]
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
        self, records: list[dict[str, Any]], agent_runs: list[AgentRun]
    ) -> dict[str, float]:
        """Compute cost per query: LLM tokens plus the query embedding.

        Args:
            records: Evaluation records, aligned with agent_runs.
            agent_runs: Output of run_agent().

        Returns:
            cost_mean_usd, cost_p95_usd and cost_total_usd.

        Raises:
            ValueError: If records and agent_runs differ in length.
        """
        costs = [
            token_cost_usd(
                self._agent_model,
                run["prompt_tokens"],
                run["completion_tokens"],
            )
            + token_cost_usd(
                self._embedding_model,
                estimate_token_count(record["question"]),
                0,
            )
            for run, record in zip(agent_runs, records, strict=True)
        ]
        return {
            "cost_mean_usd": float(np.mean(costs)) if costs else 0.0,
            "cost_p95_usd": float(np.percentile(costs, 95)) if costs else 0.0,
            "cost_total_usd": float(sum(costs)),
        }

    @staticmethod
    def log_metrics_to_langsmith(
        metrics: dict[str, Any], experiment_name: str
    ) -> None:
        """Store aggregate metrics as one "<experiment>-pipeline-metrics" run.

        Skipped while LangSmith tracing is off.

        Args:
            metrics: Metric name -> value, stored as the run's outputs.
            experiment_name: Experiment label used in the run name.
        """
        if not settings.tracing_enabled:
            logger.info("LangSmith tracing disabled; skipping metrics upload")
            return

        import langsmith

        langsmith.Client().create_run(
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
            "Logged pipeline metrics to LangSmith project %r",
            settings.langchain_project,
        )
