```python
import os
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

from utils.text_splitter import create_chunks
from utils.text_embeddings import (
    get_embeddings,
    embedding_dimension
)

# ==========================================================
# CONFIG
# ==========================================================

QDRANT_PATH = os.getenv(
    "QDRANT_PATH",
    "./qdrant_data"
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "healthcare_docs"
)

# ==========================================================
# CLIENT
# ==========================================================

def get_qdrant_client():
    """
    Returns a Qdrant client.
    """
    return QdrantClient(
        path=QDRANT_PATH
    )

# ==========================================================
# COLLECTION
# ==========================================================

def load_or_create_collection():
    """
    Creates collection if it doesn't exist.
    """

    client = get_qdrant_client()

    if not client.collection_exists(
        collection_name=QDRANT_COLLECTION
    ):

        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=embedding_dimension(),
                distance=Distance.COSINE
            )
        )

        print(
            f"Created collection: "
            f"{QDRANT_COLLECTION}"
        )

    return client

# ==========================================================
# INSERT DOCUMENT
# ==========================================================

def add_to_qdrant(
    text: str,
    metadata: dict = None
):
    """
    Ingests a document into Qdrant.

    Steps:
    1. Chunk text
    2. Generate embeddings
    3. Create payload
    4. Store vectors
    """

    metadata = metadata or {}

    document = {
        "text": text,
        "metadata": metadata
    }

    chunks = create_chunks(document)

    if not chunks:
        print("No chunks created.")
        return 0

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = get_embeddings(
        chunk_texts
    )

    points = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        payload = {
            "text": chunk["text"],
            **chunk["metadata"]
        }

        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=embedding.tolist(),
                payload=payload
            )
        )

    client = load_or_create_collection()

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points
    )

    print(
        f"Added {len(points)} chunks "
        f"to Qdrant collection "
        f"'{QDRANT_COLLECTION}'"
    )

    return len(points)

# ==========================================================
# SEARCH
# ==========================================================

def query_qdrant(
    query: str,
    top_k: int = 5
):
    """
    Retrieve most relevant chunks.
    """

    client = load_or_create_collection()

    query_vector = (
        get_embeddings([query])[0]
        .tolist()
    )

    hits = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k
    )

    results = []

    for hit in hits:

        payload = hit.payload

        text = payload.get(
            "text",
            ""
        )

        metadata = {
            k: v
            for k, v in payload.items()
            if k != "text"
        }

        results.append(
            {
                "text": text,
                "metadata": metadata,
                "score": float(
                    hit.score
                )
            }
        )

    return results

# ==========================================================
# DELETE COLLECTION
# ==========================================================

def delete_collection():
    """
    Deletes collection.
    Use carefully.
    """

    client = get_qdrant_client()

    if client.collection_exists(
        collection_name=QDRANT_COLLECTION
    ):

        client.delete_collection(
            collection_name=QDRANT_COLLECTION
        )

        print(
            f"Deleted collection: "
            f"{QDRANT_COLLECTION}"
        )

# ==========================================================
# COLLECTION INFO
# ==========================================================

def collection_info():
    """
    Returns collection metadata.
    """

    client = load_or_create_collection()

    return client.get_collection(
        collection_name=QDRANT_COLLECTION
    )

# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    sample_text = """
    Diabetes is a chronic disease that occurs when
    blood sugar levels remain elevated.

    Common symptoms include excessive thirst,
    frequent urination, fatigue, and blurred vision.

    Treatment includes diet control, exercise,
    medication, and insulin therapy.
    """

    metadata = {
        "source": "diabetes.pdf",
        "domain": "HEALTHCARE"
    }

    add_to_qdrant(
        sample_text,
        metadata
    )

    results = query_qdrant(
        "What are symptoms of diabetes?"
    )

    print("\nRESULTS\n")

    for result in results:
        print(result)
```
