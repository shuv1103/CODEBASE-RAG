import os
from typing import Optional

from langsmith.run_trees import RunTree
import langsmith

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool

_TRACING_ENABLED = (
    os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    and bool(os.getenv("LANGCHAIN_API_KEY"))
)
_PROJECT = os.getenv("LANGCHAIN_PROJECT", "default")
_client: Optional[langsmith.Client] = langsmith.Client() if _TRACING_ENABLED else None

# Keyed by invocation_id → {"agent": RunTree, "model_stack": [...], "tool_stack": [...]}
_runs: dict[str, dict] = {}


def _extract_messages(llm_request: LlmRequest) -> list[dict]:
    """Flatten an LLM request's contents into role/content messages.

    Tool calls and tool results are rendered as "[tool_call: name]" /
    "[tool_result: name]" placeholders.

    Args:
        llm_request: The ADK request about to be sent to the model.

    Returns:
        One {"role", "content"} dict per request content.
    """
    messages = []
    for content in llm_request.contents:
        parts_text = []
        for part in content.parts or []:
            if hasattr(part, "text") and part.text:
                parts_text.append(part.text)
            elif hasattr(part, "function_call") and part.function_call:
                parts_text.append(f"[tool_call: {part.function_call.name}]")
            elif hasattr(part, "function_response") and part.function_response:
                parts_text.append(f"[tool_result: {part.function_response.name}]")
        messages.append({"role": content.role or "user", "content": " ".join(parts_text)})
    return messages


def _extract_llm_output(llm_response: LlmResponse) -> str:
    """Render an LLM response's text and tool calls as a single string.

    Args:
        llm_response: The ADK response returned by the model.

    Returns:
        "[error] ..." for an error response, "" for an empty one, otherwise
        the space-joined text parts and "[tool_call: name(args)]" markers.
    """
    if llm_response.error_message:
        return f"[error] {llm_response.error_message}"
    if not llm_response.content or not llm_response.content.parts:
        return ""
    parts_text = []
    for part in llm_response.content.parts:
        if hasattr(part, "text") and part.text:
            parts_text.append(part.text)
        elif hasattr(part, "function_call") and part.function_call:
            parts_text.append(
                f"[tool_call: {part.function_call.name}({part.function_call.args})]"
            )
    return " ".join(parts_text)


def log_cache_event(
    repo_id: str,
    outcome: str,
    user_query: str,
    nearest_similarity: Optional[float] = None,
) -> None:
    """Post a cache hit/miss as its own root LangSmith chain run.

    No-op when tracing is disabled; tracing errors are swallowed.

    Args:
        repo_id: Repo the chat turn was scoped to.
        outcome: "exact_hit" | "semantic_hit" | "miss" | "blocked".
        user_query: The user's chat message.
        nearest_similarity: Nearest cached query's cosine similarity, when
            the semantic tier was queried.
    """
    # Cache hits/misses happen in api/dependencies.py, before (or instead
    # of) any ADK agent/model/tool callback above ever fires — without this,
    # a cache hit is invisible to the same root-chain, is_root=True queries
    # §5.1 runs for P95 latency, silently skewing those numbers toward
    # cache-miss traffic only.
    #
    # user_query is included in the output (not just inputs) so a hit/miss
    # trace is traceable/debuggable on its own in the LangSmith UI without
    # having to cross-reference the input panel separately.
    #
    # nearest_similarity is the matched cache entry's similarity, rounded to
    # 2 decimal places (see SemanticCacheRepository.lookup /
    # SemanticLookupResult) — 1.0 on "exact_hit", the LangCache similarity on
    # "semantic_hit". None on "miss" (LangCache only returns entries above
    # its similarity threshold) and on "blocked".
    if not _TRACING_ENABLED:
        return
    try:
        outputs = {"cache_outcome": outcome, "user_query": user_query}
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
    except Exception:
        pass


def before_agent_callback(callback_context: CallbackContext) -> None:
    """ADK callback: open the root LangSmith chain run for an invocation.

    Args:
        callback_context: ADK context for the agent invocation.

    Returns:
        None, so ADK proceeds with the agent run.
    """
    if not _TRACING_ENABLED:
        return None
    try:
        invocation_id = callback_context.invocation_id
        user_content = callback_context.user_content
        input_text = ""
        if user_content and user_content.parts:
            input_text = " ".join(
                p.text for p in user_content.parts if hasattr(p, "text") and p.text
            )
        run = RunTree(
            name=callback_context.agent_name,
            run_type="chain",
            inputs={"input": input_text},
            project_name=_PROJECT,
            client=_client,
        )
        run.post()
        _runs[invocation_id] = {"agent": run, "model_stack": [], "tool_stack": []}
    except Exception:
        pass
    return None


def after_agent_callback(callback_context: CallbackContext) -> None:
    """ADK callback: close the invocation's root run and any dangling children.

    Args:
        callback_context: ADK context for the agent invocation.

    Returns:
        None, so ADK keeps the agent's own output.
    """
    if not _TRACING_ENABLED:
        return None
    try:
        invocation_id = callback_context.invocation_id
        state = _runs.pop(invocation_id, {})
        for run in state.get("model_stack", []):
            run.end(error="closed by after_agent_callback")
            run.patch()
        for run in state.get("tool_stack", []):
            run.end(error="closed by after_agent_callback")
            run.patch()
        agent_run: Optional[RunTree] = state.get("agent")
        if agent_run:
            agent_run.end()
            agent_run.patch()
    except Exception:
        pass
    return None


def before_model_callback(callback_context, llm_request: LlmRequest):
    """ADK callback: open an "llm" child run for a model call.

    Args:
        callback_context: ADK context for the agent invocation.
        llm_request: The request about to be sent to the model.

    Returns:
        None, so ADK proceeds with the real model call.
    """
    if not _TRACING_ENABLED:
        return None
    try:
        invocation_id = callback_context.invocation_id
        state = _runs.get(invocation_id, {})
        agent_run: Optional[RunTree] = state.get("agent")
        messages = _extract_messages(llm_request)
        model_run = agent_run.create_child(
            name=llm_request.model or "gemini",
            run_type="llm",
            inputs={"messages": messages},
        ) if agent_run else RunTree(
            name=llm_request.model or "gemini",
            run_type="llm",
            inputs={"messages": messages},
            project_name=_PROJECT,
            client=_client,
        )
        model_run.post()
        state.setdefault("model_stack", []).append(model_run)
    except Exception:
        pass
    return None


def after_model_callback(callback_context, llm_response: LlmResponse):
    """ADK callback: close the most recent "llm" run with the model's output.

    Args:
        callback_context: ADK context for the agent invocation.
        llm_response: The response returned by the model.

    Returns:
        None, so ADK keeps the model's own response.
    """
    if not _TRACING_ENABLED:
        return None
    try:
        invocation_id = callback_context.invocation_id
        state = _runs.get(invocation_id, {})
        model_stack: list = state.get("model_stack", [])
        if model_stack:
            model_run: RunTree = model_stack.pop()
            model_run.end(outputs={"output": _extract_llm_output(llm_response)})
            model_run.patch()
    except Exception:
        pass
    return None


def before_tool_callback(tool: BaseTool, args: dict, tool_context) -> None:
    """ADK callback: open a "tool" child run for a tool call.

    Args:
        tool: The tool being invoked.
        args: Arguments the model passed to the tool.
        tool_context: ADK context for the tool call.

    Returns:
        None, so ADK proceeds with the real tool call.
    """
    if not _TRACING_ENABLED:
        return None
    try:
        invocation_id = tool_context.invocation_id
        state = _runs.get(invocation_id, {})
        agent_run: Optional[RunTree] = state.get("agent")
        tool_run = agent_run.create_child(
            name=tool.name,
            run_type="tool",
            inputs=args,
        ) if agent_run else RunTree(
            name=tool.name,
            run_type="tool",
            inputs=args,
            project_name=_PROJECT,
            client=_client,
        )
        tool_run.post()
        state.setdefault("tool_stack", []).append(tool_run)
    except Exception:
        pass
    return None


def after_tool_callback(tool: BaseTool, args: dict, tool_context, tool_response: dict) -> None:
    """ADK callback: close the most recent "tool" run with the tool's result.

    Args:
        tool: The tool that was invoked.
        args: Arguments the model passed to the tool.
        tool_context: ADK context for the tool call.
        tool_response: The tool's result.

    Returns:
        None, so ADK keeps the tool's own result.
    """
    if not _TRACING_ENABLED:
        return None
    try:
        invocation_id = tool_context.invocation_id
        state = _runs.get(invocation_id, {})
        tool_stack: list = state.get("tool_stack", [])
        if tool_stack:
            tool_run: RunTree = tool_stack.pop()
            tool_run.end(outputs=tool_response or {})
            tool_run.patch()
    except Exception:
        pass
    return None
