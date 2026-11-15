"""Factory for the agent's LiteLLM model."""

from google.adk.models.lite_llm import LiteLlm

from utils.config import settings


def litellm_model_name(model: str) -> str:
    """Build the litellm id for an OpenAI model.

    Args:
        model: OpenAI model name, e.g. "gpt-4o-mini".

    Returns:
        "openai/<model>".
    """
    return f"openai/{model}"


def build_llm_model(model: str | None) -> LiteLlm:
    """Build the agent's LiteLlm model.

    LiteLLM retries itself LLM_NUM_RETRIES times, since the agent's streamed
    run can't be wrapped by ExponentialBackoff.

    Args:
        model: OpenAI model name (AGENT_MODEL).

    Returns:
        A LiteLlm for "openai/<model>".

    Raises:
        ValueError: If model is unset.
    """
    if not model:
        raise ValueError("AGENT_MODEL is required to build the agent's LLM")
    return LiteLlm(
        model=litellm_model_name(model),
        num_retries=settings.llm_num_retries,
    )
