"""LangSmith tracing of agent, model, tool and cache events (redacted)."""

import functools
import itertools
import logging
from collections.abc import Callable, Iterable
from typing import Any, TypedDict

import langsmith
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from langsmith.client import RUN_TYPE_T
from langsmith.run_trees import RunTree

from caching.pii_detector import redact_pii
from utils.config import settings

logger = logging.getLogger(__name__)

_TRACING_ENABLED = settings.tracing_enabled
# Counts hook failures, so only the first one is logged at WARNING.
_tracing_failures = itertools.count()
_PROJECT = settings.langchain_project
_client: langsmith.Client | None = (
    langsmith.Client() if _TRACING_ENABLED else None
)


class InvocationRuns(TypedDict, total=False):
    """Open LangSmith runs of one ADK invocation.

    Attributes:
        agent: The invocation's root chain run.
        model_stack: Open "llm" runs, innermost last.
        tool_stack: Open "tool" runs, innermost last.
    """

    agent: RunTree | None
    model_stack: list[RunTree]
    tool_stack: list[RunTree]


# Open runs per invocation, so "after" callbacks close what "before" opened.
_runs: dict[str, InvocationRuns] = {}
# ADK skips after_agent_callback on errors, so cap the open invocations.
_MAX_OPEN_INVOCATIONS = 256
_ABANDONED_RUN_ERROR = "invocation ended without after_agent_callback"


def close_runs(
    state: InvocationRuns, child_error: str, agent_error: str | None = None
) -> None:
    """End an invocation's open child runs, then its agent run.

    Args:
        state: The invocation's open runs.
        child_error: Error recorded on each still-open child run.
        agent_error: Error recorded on the agent run; None ends it normally.
    """
    for run in itertools.chain(
        state.get("model_stack", []), state.get("tool_stack", [])
    ):
        run.end(error=child_error)
        run.patch()
    agent_run: RunTree | None = state.get("agent")
    if agent_run:
        agent_run.end(error=agent_error)
        agent_run.patch()


def invocation_state(invocation_id: str) -> InvocationRuns:
    """Return an invocation's open runs, creating the entry if needed.

    Past _MAX_OPEN_INVOCATIONS entries, the oldest is closed and dropped.

    Args:
        invocation_id: ADK invocation id.

    Returns:
        The invocation's {"agent", "model_stack", "tool_stack"} entry.
    """
    state = _runs.get(invocation_id)
    if state is None:
        state = {"agent": None, "model_stack": [], "tool_stack": []}
        _runs[invocation_id] = state
        while len(_runs) > _MAX_OPEN_INVOCATIONS:
            oldest = _runs.pop(next(iter(_runs)))
            close_runs(oldest, _ABANDONED_RUN_ERROR, _ABANDONED_RUN_ERROR)
    return state


def trace_safely(callback: Callable[..., None]) -> Callable[..., None]:
    """Decorate a tracing hook: a no-op while tracing is off, and never raises.

    Args:
        callback: The tracing hook.

    Returns:
        The wrapped hook, which always returns None (ADK proceeds unchanged).
    """

    @functools.wraps(callback)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        """Call the hook if tracing is on, logging any error."""
        if not _TRACING_ENABLED:
            return
        try:
            callback(*args, **kwargs)
        # Tracing must never break a chat; warn once, then stay quiet.
        except Exception:
            first_failure = next(_tracing_failures) == 0
            logger.log(
                logging.WARNING if first_failure else logging.DEBUG,
                "LangSmith tracing hook %s failed",
                callback.__name__,
                exc_info=True,
            )

    return wrapper


# Traces leave the process, so all user text is PII-redacted first.


def redact_values(values: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a mapping with every string value PII-redacted.

    Args:
        values: Mapping such as tool call arguments.

    Returns:
        The redacted copy; non-string values are unchanged.
    """
    return {
        key: redact_pii(value) if isinstance(value, str) else value
        for key, value in values.items()
    }


def render_part(part: types.Part, with_call_args: bool) -> str | None:
    """Render one content part as trace text.

    Args:
        part: A request or response content part.
        with_call_args: Include a function call's arguments.

    Returns:
        The text, a "[tool_call: name(args)]" / "[tool_result: name]" marker,
        or None for a part with nothing to show.
    """
    if text := getattr(part, "text", None):
        return text
    if call := getattr(part, "function_call", None):
        return (
            f"[tool_call: {call.name}({call.args})]"
            if with_call_args
            else f"[tool_call: {call.name}]"
        )
    if response := getattr(part, "function_response", None):
        return f"[tool_result: {response.name}]"
    return None


def render_parts(
    parts: Iterable[types.Part] | None, with_call_args: bool = False
) -> str:
    """Render and join content parts, PII-redacted.

    Args:
        parts: Content parts, or None.
        with_call_args: Include function call arguments.

    Returns:
        The space-joined text ("" when nothing renders).
    """
    return redact_pii(
        " ".join(
            rendered
            for part in parts or []
            if (rendered := render_part(part, with_call_args)) is not None
        )
    )


def extract_messages(llm_request: LlmRequest) -> list[dict[str, str]]:
    """Flatten an LLM request into trace messages.

    Args:
        llm_request: The request about to be sent to the model.

    Returns:
        One {"role", "content"} dict per request content.
    """
    return [
        {
            "role": content.role or "user",
            "content": render_parts(content.parts),
        }
        for content in llm_request.contents
    ]


def extract_llm_output(llm_response: LlmResponse) -> str:
    """Render an LLM response as trace text.

    Args:
        llm_response: The model's response.

    Returns:
        "[error] ..." for an error, "" for an empty response, otherwise its
        text and tool-call markers.
    """
    if llm_response.error_message:
        return f"[error] {redact_pii(llm_response.error_message)}"
    if not llm_response.content:
        return ""
    return render_parts(llm_response.content.parts, with_call_args=True)


def start_run(
    state: InvocationRuns,
    name: str,
    run_type: RUN_TYPE_T,
    inputs: dict[str, Any],
) -> RunTree:
    """Post a run under the invocation's agent run, else as a root run.

    Args:
        state: The invocation's open runs.
        name: Run name.
        run_type: LangSmith run type, e.g. "llm" or "tool".
        inputs: Run inputs.

    Returns:
        The posted RunTree.
    """
    agent_run: RunTree | None = state.get("agent")
    if agent_run:
        run = agent_run.create_child(
            name=name, run_type=run_type, inputs=inputs
        )
    else:
        run = RunTree(
            name=name,
            run_type=run_type,
            inputs=inputs,
            project_name=_PROJECT,
            ls_client=_client,
        )
    run.post()
    return run


@trace_safely
def log_cache_event(
    repo_id: str,
    outcome: str,
    user_query: str,
    nearest_similarity: float | None = None,
) -> None:
    """Post a cache lookup as its own root chain run.

    Cache hits never reach the ADK callbacks, so without this run they would
    be missing from latency queries.

    Args:
        repo_id: Repo the chat turn was about.
        outcome: "exact_hit", "semantic_hit", "miss" or "blocked".
        user_query: The chat message; PII-redacted before sending.
        nearest_similarity: Matched entry's similarity, if any.
    """
    user_query = redact_pii(user_query)
    # Repeated in outputs so the trace reads on its own in the UI.
    outputs: dict[str, Any] = {
        "cache_outcome": outcome,
        "user_query": user_query,
    }
    if nearest_similarity is not None:
        outputs["nearest_similarity"] = nearest_similarity

    run = start_run(
        {},
        f"semantic_cache_{outcome}",
        "chain",
        {"repo_id": repo_id, "user_query": user_query},
    )
    run.end(outputs=outputs)
    run.patch()


@trace_safely
def before_agent_callback(callback_context: CallbackContext) -> None:
    """Open the invocation's root chain run (ADK callback).

    Args:
        callback_context: ADK context of the agent invocation.
    """
    user_content = callback_context.user_content
    input_text = render_parts(user_content.parts) if user_content else ""
    # A fresh {} state has no agent run, so start_run posts a root run.
    run = start_run(
        {}, callback_context.agent_name, "chain", {"input": input_text}
    )
    invocation_state(callback_context.invocation_id)["agent"] = run


@trace_safely
def after_agent_callback(callback_context: CallbackContext) -> None:
    """Close the invocation's root run and any open children (ADK callback).

    Args:
        callback_context: ADK context of the agent invocation.
    """
    state = _runs.pop(callback_context.invocation_id, {})
    close_runs(state, "closed by after_agent_callback")


@trace_safely
def before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Open an "llm" child run for a model call (ADK callback).

    Args:
        callback_context: ADK context of the agent invocation.
        llm_request: The request about to be sent to the model.
    """
    state = invocation_state(callback_context.invocation_id)
    model_run = start_run(
        state,
        name=llm_request.model or "gemini",
        run_type="llm",
        inputs={"messages": extract_messages(llm_request)},
    )
    state["model_stack"].append(model_run)


@trace_safely
def after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> None:
    """Close the latest "llm" run with the model's output (ADK callback).

    Args:
        callback_context: ADK context of the agent invocation.
        llm_response: The model's response.
    """
    model_stack: list[RunTree] = _runs.get(
        callback_context.invocation_id, {}
    ).get("model_stack", [])
    if model_stack:
        model_run = model_stack.pop()
        model_run.end(outputs={"output": extract_llm_output(llm_response)})
        model_run.patch()


@trace_safely
def before_tool_callback(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
) -> None:
    """Open a "tool" child run for a tool call (ADK callback).

    Args:
        tool: The tool being called.
        args: Arguments the model passed; PII-redacted before sending.
        tool_context: ADK context of the tool call.
    """
    state = invocation_state(tool_context.invocation_id)
    tool_run = start_run(
        state, name=tool.name, run_type="tool", inputs=redact_values(args)
    )
    state["tool_stack"].append(tool_run)


@trace_safely
def after_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict[str, Any],
) -> None:
    """Close the latest "tool" run with the tool's result (ADK callback).

    Args:
        tool: The tool that was called.
        args: Arguments the model passed.
        tool_context: ADK context of the tool call.
        tool_response: The tool's result.
    """
    tool_stack: list[RunTree] = _runs.get(tool_context.invocation_id, {}).get(
        "tool_stack", []
    )
    if tool_stack:
        tool_run = tool_stack.pop()
        # Retrieved code can hold a repo's committed secrets.
        tool_run.end(outputs=redact_values(tool_response or {}))
        tool_run.patch()
