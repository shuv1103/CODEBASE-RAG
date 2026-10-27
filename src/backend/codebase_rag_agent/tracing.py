"""LangSmith tracing callbacks for agent, model, tool and cache events."""

import contextlib
import functools
import itertools
from collections.abc import Callable, Iterable
from typing import Any, TypedDict

import langsmith
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from langsmith.run_trees import RunTree

from caching.pii_detector import redact_pii
from utils.config import settings

_TRACING_ENABLED = settings.tracing_enabled
_PROJECT = settings.langchain_project
_client: langsmith.Client | None = (
    langsmith.Client() if _TRACING_ENABLED else None
)


class InvocationRuns(TypedDict, total=False):
    """The open LangSmith runs of one ADK invocation.

    Attributes:
        agent: The invocation's agent run, once before_agent_callback ran.
        model_stack: Model runs opened and not yet closed, innermost last.
        tool_stack: Tool runs opened and not yet closed, innermost last.
    """

    agent: RunTree | None
    model_stack: list[RunTree]
    tool_stack: list[RunTree]


# Open runs per ADK invocation id, so "after" callbacks can close what
# "before" callbacks opened.
_runs: dict[str, InvocationRuns] = {}
# ADK skips after_agent_callback when a run raises or is cancelled, so cap the
# open invocations; past the cap the oldest is closed and forgotten.
_MAX_OPEN_INVOCATIONS = 256
_ABANDONED_RUN_ERROR = "invocation ended without after_agent_callback"


def close_runs(
    state: InvocationRuns, child_error: str, agent_error: str | None = None
) -> None:
    """End an invocation's open model/tool runs and its agent run.

    Args:
        state: The invocation's entry in _runs.
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
        if agent_error is None:
            agent_run.end()
        else:
            agent_run.end(error=agent_error)
        agent_run.patch()


def invocation_state(invocation_id: str) -> InvocationRuns:
    """Return an invocation's entry in _runs, creating it if needed.

    Creating the entry keeps runs opened for an unknown invocation
    findable by the matching "after" callback. Evicts the oldest entries
    once more than _MAX_OPEN_INVOCATIONS are open.

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
    """Wrap a tracing hook so it is a no-op while tracing is off.

    Tracing must never break a chat, so any exception is swallowed. The
    wrapper returns None, so ADK proceeds with its own request/response.

    Args:
        callback: Tracing hook to wrap.

    Returns:
        The wrapped hook.
    """

    @functools.wraps(callback)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        """Call callback when tracing is on, swallowing any error."""
        if _TRACING_ENABLED:
            with contextlib.suppress(Exception):
                callback(*args, **kwargs)

    return wrapper


# Traces go to a third party and bypass the cache's PII gate, so all user
# text is passed through redact_pii before it is sent.


def redact_values(values: dict[str, Any]) -> dict[str, Any]:
    """Redact PII/credentials in a dict's string values (e.g. tool call args).

    Args:
        values: Mapping whose string values may contain user text.

    Returns:
        A copy with every string value passed through redact_pii.
    """
    return {
        key: redact_pii(value) if isinstance(value, str) else value
        for key, value in values.items()
    }


def render_part(part: types.Part, with_call_args: bool) -> str | None:
    """Render one content part as trace text.

    Args:
        part: A request or response content part.
        with_call_args: Include a function call's arguments in its marker.

    Returns:
        The part's text, a "[tool_call: ...]" / "[tool_result: name]"
        marker, or None for a part with nothing to show.
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
    """Join the rendered form of every part that has something to show.

    Args:
        parts: Content parts, or None.
        with_call_args: Include function call arguments in tool_call markers.

    Returns:
        The joined text ("" when there are no renderable parts).
    """
    return redact_pii(
        " ".join(
            rendered
            for part in parts or []
            if (rendered := render_part(part, with_call_args)) is not None
        )
    )


def extract_messages(llm_request: LlmRequest) -> list[dict[str, str]]:
    """Flatten an LLM request's contents into role/content messages.

    Render tool calls and tool results as "[tool_call: name]" and
    "[tool_result: name]" placeholders.

    Args:
        llm_request: The ADK request about to be sent to the model.

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
    """Render an LLM response's text and tool calls as a single string.

    Args:
        llm_response: The ADK response returned by the model.

    Returns:
        "[error] ..." for an error response, "" for an empty one, otherwise
        the space-joined text parts and "[tool_call: name(args)]" markers.
    """
    if llm_response.error_message:
        return f"[error] {redact_pii(llm_response.error_message)}"
    if not llm_response.content:
        return ""
    return render_parts(llm_response.content.parts, with_call_args=True)


def start_run(
    state: InvocationRuns, name: str, run_type: str, inputs: dict[str, Any]
) -> RunTree:
    """Post a new run: a child of the invocation's agent run, else a root run.

    Args:
        state: The invocation's entry in _runs (empty if unknown).
        name: Run name.
        run_type: LangSmith run type ("llm", "tool", ...).
        inputs: Run inputs.

    Returns:
        The posted run.
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
            client=_client,
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
    """Post a cache hit/miss as its own root LangSmith chain run.

    Cache hits never reach the ADK callbacks, so without this run they
    would be missing from latency queries over root runs.

    Args:
        repo_id: Repo the chat turn was scoped to.
        outcome: "exact_hit" | "semantic_hit" | "miss" | "blocked".
        user_query: The user's chat message.
        nearest_similarity: Matched entry's similarity (1.0 for an exact
            hit); None on a miss or "blocked".
    """
    user_query = redact_pii(user_query)
    # The query is repeated in the outputs so a trace reads on its own in the
    # LangSmith UI.
    outputs: dict[str, Any] = {
        "cache_outcome": outcome,
        "user_query": user_query,
    }
    if nearest_similarity is not None:
        outputs["nearest_similarity"] = nearest_similarity

    run = RunTree(
        name=f"semantic_cache_{outcome}",
        run_type="chain",
        inputs={"repo_id": repo_id, "user_query": user_query},
        project_name=_PROJECT,
        client=_client,
    )
    run.post()
    run.end(outputs=outputs)
    run.patch()


@trace_safely
def before_agent_callback(callback_context: CallbackContext) -> None:
    """Open the root LangSmith chain run for an invocation (ADK callback).

    Args:
        callback_context: ADK context for the agent invocation.

    Returns:
        None, so ADK proceeds with the agent run.
    """
    user_content = callback_context.user_content
    input_text = render_parts(user_content.parts) if user_content else ""
    run = RunTree(
        name=callback_context.agent_name,
        run_type="chain",
        inputs={"input": input_text},
        project_name=_PROJECT,
        client=_client,
    )
    run.post()
    invocation_state(callback_context.invocation_id)["agent"] = run


@trace_safely
def after_agent_callback(callback_context: CallbackContext) -> None:
    """Close an invocation's root run and any open child runs (ADK callback).

    Args:
        callback_context: ADK context for the agent invocation.

    Returns:
        None, so ADK keeps the agent's own output.
    """
    state = _runs.pop(callback_context.invocation_id, {})
    close_runs(state, "closed by after_agent_callback")


@trace_safely
def before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> None:
    """Open an "llm" child run for a model call (ADK callback).

    Args:
        callback_context: ADK context for the agent invocation.
        llm_request: The request about to be sent to the model.

    Returns:
        None, so ADK proceeds with the real model call.
    """
    state = invocation_state(callback_context.invocation_id)
    model_run = start_run(
        state,
        name=llm_request.model or "gemini",
        run_type="llm",
        inputs={"messages": extract_messages(llm_request)},
    )
    state.setdefault("model_stack", []).append(model_run)


@trace_safely
def after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> None:
    """Close the latest "llm" run with the model's output (ADK callback).

    Args:
        callback_context: ADK context for the agent invocation.
        llm_response: The response returned by the model.

    Returns:
        None, so ADK keeps the model's own response.
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
        tool: The tool being invoked.
        args: Arguments the model passed to the tool.
        tool_context: ADK context for the tool call.

    Returns:
        None, so ADK proceeds with the real tool call.
    """
    state = invocation_state(tool_context.invocation_id)
    tool_run = start_run(
        state, name=tool.name, run_type="tool", inputs=redact_values(args)
    )
    state.setdefault("tool_stack", []).append(tool_run)


@trace_safely
def after_tool_callback(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict[str, Any],
) -> None:
    """Close the latest "tool" run with the tool's result (ADK callback).

    Args:
        tool: The tool that was invoked.
        args: Arguments the model passed to the tool.
        tool_context: ADK context for the tool call.
        tool_response: The tool's result.

    Returns:
        None, so ADK keeps the tool's own result.
    """
    tool_stack: list[RunTree] = _runs.get(tool_context.invocation_id, {}).get(
        "tool_stack", []
    )
    if tool_stack:
        tool_run = tool_stack.pop()
        tool_run.end(outputs=tool_response or {})
        tool_run.patch()
