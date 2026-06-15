import os
from functools import lru_cache
from langchain_groq import ChatGroq

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@lru_cache(maxsize=1)
def llm_caller():
    """
    Returns a singleton Groq LLM instance.

    Benefits:
    - Creates only one LLM object
    - Faster performance
    - Lower memory usage
    """

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY environment variable is missing."
        )

    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0,
        max_tokens=2048
    )
