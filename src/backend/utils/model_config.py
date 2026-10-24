"""Factory for the agent's LiteLLM model."""

from google.adk.models.lite_llm import LiteLlm

from .config import settings


def litellm_model_name(model: str) -> str:
    """Build a litellm-style '<provider>/<model>' identifier.

    Every model in this project (agent + embeddings) is served via OpenAI.

    Args:
        model: OpenAI model name, e.g. settings.agent_model.

    Returns:
        The identifier "openai/<model>".

    Example:
        >>> litellm_model_name("gpt-4o-mini")
        'openai/gpt-4o-mini'
    """
    return f"openai/{model}"


def build_llm_model(model: str | None) -> LiteLlm:
    """Build a ready-to-use LiteLlm model instance.

    The model retries itself LLM_NUM_RETRIES times (default 3): LiteLLM
    backs off exponentially on a 429 and retries other API errors at once.
    The agent run streams events, so ExponentialBackoff can't wrap it.

    Args:
        model: OpenAI model name, e.g. settings.agent_model.

    Returns:
        A LiteLlm configured with the resolved "openai/<model>" identifier.

    Raises:
        ValueError: If model is unset (AGENT_MODEL missing).
    """
    if not model:
        raise ValueError("AGENT_MODEL is required to build the agent's LLM")
    return LiteLlm(
        model=litellm_model_name(model),
        num_retries=settings.llm_num_retries,
    )
