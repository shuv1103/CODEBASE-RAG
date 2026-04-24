import os
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from .tools import codebase_retrieval_tool

load_dotenv()


def _validate_env() -> None:
    """
    Validate required environment variables before agent startup.
    """
    required_env_vars = [
        "GOOGLE_API_KEY",
        "AGENT_MODEL",
    ]

    missing = [var for var in required_env_vars if not os.getenv(var)]
    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )


_validate_env()


retrieve_code_context_tool = FunctionTool(
    func=codebase_retrieval_tool.retrieve_code_context,
)


root_agent = LlmAgent(
    name="codebase_rag_agent",
    model=os.getenv("AGENT_MODEL"),
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
    tools=[retrieve_code_context_tool],
)
