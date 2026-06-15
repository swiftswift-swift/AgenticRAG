from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import PromptTemplate

from config.config import llm_caller


# ==========================================================
# STRUCTURED OUTPUT
# ==========================================================

class RetrievalValidationOutput(BaseModel):

    needs_retrieval: str = Field(
        description="true if more retrieval is required, otherwise false"
    )

    confidence: float = Field(
        description="Confidence score between 0 and 1 indicating how well the retrieved documents answer the query."
    )

    reason: str = Field(
        description="Reason for the retrieval decision."
    )

    new_query: Optional[str] = Field(
        default=None,
        description="Improved retrieval query when additional retrieval is needed."
    )


# ==========================================================
# VALIDATOR
# ==========================================================

def validate_retrieval(state_partial):
    """
    Determines whether the retrieved documents
    sufficiently answer the user's query.

    Returns:
    - needs_retrieval
    - confidence
    - reason
    - new_query
    """

    documents = state_partial["documents"]

    if not documents:
        return RetrievalValidationOutput(
            needs_retrieval="true",
            confidence=0.0,
            reason="No documents were retrieved.",
            new_query=state_partial["main_query"]
        )

    docs_text = "\n\n".join(
        doc.get("text", "")
        for doc in documents
        if isinstance(doc, dict)
    )

    validation_prompt = PromptTemplate.from_template(
        """
You are an expert retrieval validator.

Your job is to determine whether the retrieved documents
contain enough information to answer the user's question.

USER QUERY:
{query}

RETRIEVED DOCUMENTS:
{docs_text}

TASKS:

1. Determine if the documents fully answer the query.

2. Set:
   - needs_retrieval = "false"
     if the documents sufficiently answer the question.

   - needs_retrieval = "true"
     if information is missing, incomplete,
     unrelated, or insufficient.

3. Provide a confidence score
   between 0 and 1.

4. Explain your decision.

5. If retrieval is needed,
   generate a better search query.

IMPORTANT RULES:

- Do NOT answer the user's question.
- Only evaluate retrieval quality.
- Be strict.
- If uncertain, prefer additional retrieval.

Examples:

Good Match:
Query:
"What are symptoms of diabetes?"

Docs:
"Diabetes symptoms include frequent urination,
fatigue, thirst, and blurred vision."

Result:
needs_retrieval = false

Bad Match:
Query:
"What are symptoms of diabetes?"

Docs:
"Diabetes is a metabolic disease."

Result:
needs_retrieval = true
new_query =
"symptoms of diabetes including common clinical signs"
"""
    )

    llm = llm_caller()

    structured_llm = llm.with_structured_output(
        RetrievalValidationOutput
    )

    validation_chain = (
        validation_prompt
        | structured_llm
    )

    result = validation_chain.invoke(
        {
            "query": state_partial["main_query"],
            "docs_text": docs_text
        }
    )

    return result
