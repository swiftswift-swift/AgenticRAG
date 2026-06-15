import os
import uuid
import chromadb

from utils.text_splitter import create_chunks
from utils.text_embeddings import get_embeddings


CHROMADB_PATH = os.getenv("CHROMADB_PATH", "./chroma_data")
CHROMADB_COLLECTION = os.getenv(
    "CHROMADB_COLLECTION",
    "engineering_docs"
)


# ==========================================================
# COLLECTION
# ==========================================================

def get_chroma_collection():

    client = chromadb.PersistentClient(
        path=CHROMADB_PATH
    )

    try:

        collection = client.get_collection(
            name=CHROMADB_COLLECTION
        )

    except Exception:

        collection = client.create_collection(
            name=CHROMADB_COLLECTION
        )

    return collection


# ==========================================================
# ADD DOCUMENT
# ==========================================================

def add_to_chroma(text, metadata):

    collection = get_chroma_collection()

    document = {
        "text": text,
        "metadata": metadata
    }

    chunks = create_chunks(document)

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = get_embeddings(
        chunk_texts
    )

    ids = []

    metadatas = []

    for chunk in chunks:

        ids.append(str(uuid.uuid4()))

        metadatas.append(
            chunk["metadata"]
        )

    collection.add(
        documents=chunk_texts,
        embeddings=embeddings.tolist(),
        ids=ids,
        metadatas=metadatas
    )

    print(
        f"Added {len(chunk_texts)} chunks to ChromaDB"
    )

    return len(chunk_texts)


# ==========================================================
# QUERY
# ==========================================================

def query_chroma(
    query,
    top_k=5
):

    collection = get_chroma_collection()

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    output = []

    for doc, meta, distance in zip(
        documents,
        metadatas,
        distances
    ):

        output.append(
            {
                "text": doc,
                "metadata": meta,
                "distance": float(distance)
            }
        )

    return output
