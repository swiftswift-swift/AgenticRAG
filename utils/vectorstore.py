from datetime import datetime

from llm_functions.classifier import classify_domain

from vector_db.chromadb import add_to_chroma
from vector_db.qdrant import add_to_qdrant

from utils.text_splitter import create_chunks


def add_text(document):
    """
    Auto-classify document and store chunks
    in the appropriate vector database.

    Expected document:

    {
        "text": "...",
        "metadata": {
            "source": "file.pdf"
        }
    }
    """

    text = document["text"]

    metadata = document.get("metadata", {})

    # --------------------------------------
    # Domain Classification
    # --------------------------------------

    classification = classify_domain(
        {
            "snippet_text": text[:3000]
        }
    )

    domain = classification.domain

    # --------------------------------------
    # Metadata Enrichment
    # --------------------------------------

    metadata["domain"] = domain

    metadata["ingested_at"] = (
        datetime.utcnow().isoformat()
    )

    # --------------------------------------
    # Chunking
    # --------------------------------------

    chunks = create_chunks(
        {
            "text": text,
            "metadata": metadata
        }
    )

    stored_chunks = 0

    # --------------------------------------
    # Storage
    # --------------------------------------

    if domain == "HEALTHCARE":

        for chunk in chunks:

            add_to_qdrant(
                chunk["text"],
                chunk["metadata"]
            )

            stored_chunks += 1

    elif domain == "ENGINEERING":

        for chunk in chunks:

            add_to_chroma(
                chunk["text"],
                chunk["metadata"]
            )

            stored_chunks += 1

    else:

        raise ValueError(
            f"Unknown domain: {domain}"
        )

    return {
        "status": "SUCCESS",
        "domain": domain,
        "chunks_stored": stored_chunks,
        "source": metadata.get(
            "source",
            "unknown"
        )
    }
