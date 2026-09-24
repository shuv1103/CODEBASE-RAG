import os
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from .tools import CodebaseRetrievalTool
from .tracing import (
    before_agent_callback,
    after_agent_callback,
    before_model_callback,
    after_model_callback,
    before_tool_callback,
    after_tool_callback,
)
from utils.model_config import build_llm_model

load_dotenv()


def _validate_env() -> None:
    """Fail fast if the agent's required environment variables are unset.

    Raises:
        ValueError: If OPENAI_API_KEY or AGENT_MODEL is missing.
    """
    required_env_vars = [
        "OPENAI_API_KEY",
        "AGENT_MODEL",
    ]
    missing = [var for var in required_env_vars if not os.getenv(var)]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


_validate_env()


def build_agent(retrieval_tool: CodebaseRetrievalTool) -> LlmAgent:
    """Build an LlmAgent wired to the given retrieval tool.

    A factory rather than a single module-level instance so the API layer
    can build one agent per repo_id, each scoped to that repo's chunks
    via a differently-configured CodebaseRetrievalTool, without
    disturbing the default `root_agent` singleton below (lazily built, used
    by `adk web`, `run_eval.py`, and `evaluation/metrics.py`).

    Args:
        retrieval_tool: Tool whose retrieve_code_context becomes the agent's
            only function tool.

    Returns:
        The configured LlmAgent, with LangSmith tracing callbacks attached.

    Example:
        >>> tool = CodebaseRetrievalTool(RetrievalService(repo_id="c3460b930023"))
        >>> runner = InMemoryRunner(agent=build_agent(tool))
    """
    function_tool = FunctionTool(func=retrieval_tool.retrieve_code_context)

    return LlmAgent(
        name="codebase_rag_agent",
        model=build_llm_model("AGENT_MODEL", "gpt-4o-mini"),
        instruction="""
        You are a Codebase Agentic RAG Assistant.

        You answer questions about a software codebase using retrieved code chunks.

        Rules:
        1. Use the tool(name = "codebase_retrieval_tool") given to you and base your answer ONLY on the retrieved context.
        2. If the context is insufficient, clearly say so.
        3. Be concise, technical, and accurate.
        4. Mention file paths and line ranges when relevant.
        5. Do not hallucinate functions, classes, files, or logic not present in the retrieved code.

        Preferred answer format:
        - Direct Answer
        - Evidence from Retrieved Code
        - Relevant Files / Lines

        """,
        tools=[function_tool],
        before_agent_callback=before_agent_callback,
        after_agent_callback=after_agent_callback,
        before_model_callback=before_model_callback,
        after_model_callback=after_model_callback,
        before_tool_callback=before_tool_callback,
        after_tool_callback=after_tool_callback,
    )


_root_agent: LlmAgent | None = None


def __getattr__(name: str):
    """Lazily resolve `root_agent` (PEP 562 module __getattr__).

    Args:
        name: Attribute being looked up on the module.

    Returns:
        The default root agent, built on first access.

    Raises:
        AttributeError: For any other attribute name.
    """
    if name == "root_agent":
        global _root_agent
        if _root_agent is None:
            from .tools import codebase_retrieval_tool  # lazy: resolved only here
            _root_agent = build_agent(codebase_retrieval_tool)
        return _root_agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
