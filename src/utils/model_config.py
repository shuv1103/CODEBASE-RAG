import os

from google.adk.models.lite_llm import LiteLlm


def litellm_model_name(env_var: str, default: str) -> str:
    """Build a litellm-style '<provider>/<model>' identifier from an env var.

    Every model in this project (agent + embeddings) is served via OpenAI.

    Args:
        env_var: Name of the environment variable holding the model name.
        default: Model name to use when env_var is unset.

    Returns:
        The identifier "openai/<model>".
    """
    return f"openai/{os.getenv(env_var, default)}"


def build_llm_model(env_var: str, default: str) -> LiteLlm:
    """Build a ready-to-use LiteLlm model instance from an env var.

    Factory Pattern: the single place that turns an env var into a LiteLlm
    instance, instead of each caller building
    LiteLlm(model=litellm_model_name(...)) inline.

    Args:
        env_var: Name of the environment variable holding the model name.
        default: Model name to use when env_var is unset.

    Returns:
        A LiteLlm configured with the resolved "openai/<model>" identifier.
    """
    return LiteLlm(model=litellm_model_name(env_var, default))
