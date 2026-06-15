from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import PromptTemplate

from config.config import llm_caller


# ==========================================================
# STRUCTURED OUTPUT
# ==========================================================

class RAGResponseOutput(BaseModel):

    answer: str = Field(
        description="Answer generated from retrieved documents."
    )

    confidence: float = Field(
        description="Confidence score between 0 and 1."
    )

    grounded: bool = Field(
        description="Whether answer is fully supported by documents."
    )

    summary_sources: List[str] = Field(
        description="Sources used to generate answer."
    )


# ==========================================================
# GENERATOR
# ==========================================================

def generate_rag_response(state_partial):

    documents = state_partial["documents"]

    if not documents:
        return RAGResponseOutput(
            answer="I could not find enough information to answer the query.",
            confidence=0.0,
            grounded=False,
            summary_sources=[]
        )

    docs_text = "\n\n".join(
        doc.get("text", "")
        for doc in documents
        if isinstance(doc, dict)
    )

    sources = list(
        {
            doc.get("metadata", {}).get(
                "source",
                "unknown"
            )
            for doc in documents
        }
    )

    rag_prompt = PromptTemplate.from_template(
        """
You are a Retrieval-Augmented Generation assistant.

Your task is to answer the user's question using ONLY the information provided in the retrieved documents.

USER QUESTION:
{main_query}

DOCUMENTS:
{docs_text}

RULES:

1. Use only information from the documents.
2. Do not invent facts.
3. If information is insufficient, clearly say so.
4. Keep the answer concise and informative.
5. Determine whether the answer is fully grounded in the documents.
6. Return confidence between 0 and 1.

AVAILABLE SOURCES:
{sources}
"""
    )

    llm = llm_caller()

    structured_llm = llm.with_structured_output(
        RAGResponseOutput
    )

    rag_chain = rag_prompt | structured_llm

    result = rag_chain.invoke(
        {
            "main_query": state_partial["main_query"],
            "docs_text": docs_text,
            "sources": ", ".join(sources)
        }
    )

    return result
