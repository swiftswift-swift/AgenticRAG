```python
from typing import TypedDict, Literal, NotRequired, List
from langgraph.graph import END, StateGraph, START
from langchain_core.runnables import RunnableLambda

from llm_functions.classifier import classify_domain
from llm_functions.validator import validate_retrieval
from llm_functions.generator import generate_rag_response

from vector_db.chromadb import query_chroma
from vector_db.qdrant import query_qdrant

# ==========================================================
# CONFIG
# ==========================================================

MAX_RETRIES = 3

# ==========================================================
# STATE
# ==========================================================

class ragGraphState(TypedDict):
    domain: NotRequired[Literal["HEALTHCARE", "ENGINEERING"]]

    answer: NotRequired[str]

    main_query: str
    current_query: str

    documents: List[dict]
    sources: List[str]

    needs_retrieval: NotRequired[str]

    retry_count: NotRequired[int]

    confidence: NotRequired[float]

    validation_reason: NotRequired[str]

    retrieval_count: NotRequired[int]


# ==========================================================
# QUERY REWRITER NODE (OPTIONAL)
# ==========================================================

def rewrite_query(state: ragGraphState) -> ragGraphState:
    """
    Future enhancement:
    Use an LLM to rewrite vague user queries.

    Example:
    'heart attack cure'
        ->
    'treatment options for myocardial infarction'
    """

    return {
        **state,
        "current_query": state["current_query"]
    }


# ==========================================================
# DOMAIN CLASSIFICATION
# ==========================================================

def detect_domain(state: ragGraphState) -> ragGraphState:

    state_partial = {
        "snippet_text": state["current_query"]
    }

    parsed = classify_domain(state_partial)

    return {
        **state,
        "domain": parsed.domain
    }


# ==========================================================
# QDRANT RETRIEVAL
# ==========================================================

def query_qdrantdb(state: ragGraphState) -> ragGraphState:

    docs = query_qdrant(state["current_query"])

    print("\nQDRANT RETRIEVAL")
    print(f"Retrieved {len(docs)} documents")

    return {
        **state,
        "documents": state.get("documents", []) + docs,
        "retrieval_count": len(docs)
    }


# ==========================================================
# CHROMA RETRIEVAL
# ==========================================================

def query_chromadb(state: ragGraphState) -> ragGraphState:

    docs = query_chroma(state["current_query"])

    print("\nCHROMADB RETRIEVAL")
    print(f"Retrieved {len(docs)} documents")

    return {
        **state,
        "documents": state.get("documents", []) + docs,
        "retrieval_count": len(docs)
    }


# ==========================================================
# VALIDATION
# ==========================================================

def validate_docs(state: ragGraphState) -> ragGraphState:

    if not state["documents"]:
        return {
            **state,
            "needs_retrieval": "false"
        }

    state_partial = {
        "documents": state["documents"],
        "main_query": state["main_query"]   # FIXED BUG
    }

    parsed = validate_retrieval(state_partial)

    print("\nVALIDATION RESULT")
    print(parsed)

    return {
        **state,
        "needs_retrieval": parsed.needs_retrieval,
        "current_query": getattr(
            parsed,
            "new_query",
            state["current_query"]
        ),
        "confidence": getattr(
            parsed,
            "confidence",
            1.0
        ),
        "validation_reason": getattr(
            parsed,
            "reason",
            "No reason provided"
        ),
        "retry_count": state.get(
            "retry_count",
            0
        ) + 1
    }


# ==========================================================
# RESPONSE GENERATION
# ==========================================================

def generate_response(state: ragGraphState) -> ragGraphState:

    state_partial = {
        "documents": state["documents"],
        "main_query": state["main_query"]  # FIXED BUG
    }

    parsed = generate_rag_response(state_partial)

    print("\nGENERATED RESPONSE")
    print(parsed)

    sources = [
        doc.get("metadata", {}).get(
            "source",
            "unknown"
        )
        for doc in state["documents"]
    ]

    return {
        **state,
        "answer": parsed.answer,
        "sources": list(set(sources))
    }


# ==========================================================
# VECTOR DB ROUTER
# ==========================================================

def vectordb_router(state: ragGraphState):

    if state.get("domain") == "HEALTHCARE":
        return {"next": "qdrant"}

    return {"next": "chromadb"}


# ==========================================================
# VALIDATION ROUTER
# ==========================================================

def validate_router(state: ragGraphState):

    retries = state.get("retry_count", 0)

    if retries >= MAX_RETRIES:
        print("\nMAX RETRIES REACHED")
        return {"next": "false"}

    confidence = state.get(
        "confidence",
        1.0
    )

    if (
        state.get("needs_retrieval") == "true"
        and confidence < 0.80
    ):
        print("\nRETRIEVING MORE CONTEXT")
        return {"next": "true"}

    return {"next": "false"}


# ==========================================================
# GRAPH BUILDER
# ==========================================================

def graph_rag():

    agenticrag_graph = StateGraph(ragGraphState)

    # -----------------------------
    # Nodes
    # -----------------------------

    agenticrag_graph.add_node(
        "rewrite_query",
        RunnableLambda(rewrite_query)
    )

    agenticrag_graph.add_node(
        "detect_domain",
        RunnableLambda(detect_domain)
    )

    agenticrag_graph.add_node(
        "query_qdrantdb",
        RunnableLambda(query_qdrantdb)
    )

    agenticrag_graph.add_node(
        "query_chromadb",
        RunnableLambda(query_chromadb)
    )

    agenticrag_graph.add_node(
        "vectordb_router",
        RunnableLambda(vectordb_router)
    )

    agenticrag_graph.add_node(
        "validate_docs",
        RunnableLambda(validate_docs)
    )

    agenticrag_graph.add_node(
        "validate_router",
        RunnableLambda(validate_router)
    )

    agenticrag_graph.add_node(
        "generate_response",
        RunnableLambda(generate_response)
    )

    # -----------------------------
    # Edges
    # -----------------------------

    agenticrag_graph.add_edge(
        START,
        "rewrite_query"
    )

    agenticrag_graph.add_edge(
        "rewrite_query",
        "detect_domain"
    )

    agenticrag_graph.add_edge(
        "detect_domain",
        "vectordb_router"
    )

    agenticrag_graph.add_conditional_edges(
        "vectordb_router",
        lambda state: state.get("next"),
        {
            "qdrant": "query_qdrantdb",
            "chromadb": "query_chromadb"
        }
    )

    agenticrag_graph.add_edge(
        "query_qdrantdb",
        "validate_docs"
    )

    agenticrag_graph.add_edge(
        "query_chromadb",
        "validate_docs"
    )

    agenticrag_graph.add_edge(
        "validate_docs",
        "validate_router"
    )

    agenticrag_graph.add_conditional_edges(
        "validate_router",
        lambda state: state.get("next"),
        {
            "true": "detect_domain",
            "false": "generate_response"
        }
    )

    agenticrag_graph.add_edge(
        "generate_response",
        END
    )

    return agenticrag_graph.compile()


# ==========================================================
# USAGE
# ==========================================================

if __name__ == "__main__":

    graph = graph_rag()

    query = "What are the symptoms of diabetes?"

    result = graph.invoke(
        {
            "main_query": query,
            "current_query": query,
            "documents": [],
            "sources": [],
            "retry_count": 0
        }
    )

    print("\nFINAL ANSWER")
    print(result["answer"])

    print("\nSOURCES")
    print(result["sources"])
```
