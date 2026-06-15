from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, List


# ==========================================================
# TEXT SPLITTER
# ==========================================================

def split_text(
    text: str,
    chunk_size: int = 700,
    overlap: int = 100
) -> List[str]:
    """
    Split raw text into chunks.

    Args:
        text: Input text
        chunk_size: Max chunk size
        overlap: Overlap between chunks

    Returns:
        List[str]
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_text(text)


# ==========================================================
# DOCUMENT CHUNKER
# ==========================================================

def create_chunks(
    document: Dict,
    chunk_size: int = 700,
    overlap: int = 100
) -> List[Dict]:
    """
    Convert a document into chunk objects.

    Expected input:

    {
        "text": "...",
        "metadata": {
            "source": "diabetes.pdf",
            "file_type": ".pdf"
        }
    }

    Returns:

    [
        {
            "id": 0,
            "text": "...",
            "metadata": {...}
        }
    ]
    """

    text = document.get("text", "")

    metadata = document.get("metadata", {})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_text(text)

    results = []

    for idx, chunk in enumerate(chunks):

        chunk_document = {
            "id": idx,
            "text": chunk,
            "metadata": {
                **metadata,
                "chunk_id": idx,
                "chunk_length": len(chunk)
            }
        }

        results.append(chunk_document)

    return results


# ==========================================================
# CHUNK STATISTICS
# ==========================================================

def get_chunk_stats(chunks: List[Dict]) -> Dict:
    """
    Returns chunk statistics.
    """

    if not chunks:
        return {
            "total_chunks": 0,
            "avg_chunk_length": 0
        }

    lengths = [
        len(chunk["text"])
        for chunk in chunks
    ]

    return {
        "total_chunks": len(chunks),
        "min_chunk_length": min(lengths),
        "max_chunk_length": max(lengths),
        "avg_chunk_length": round(
            sum(lengths) / len(lengths),
            2
        )
    }


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    document = {
        "text": """
        Diabetes is a chronic disease that occurs when
        the body cannot effectively regulate blood sugar.

        Common symptoms include increased thirst,
        frequent urination, fatigue, and blurred vision.

        Treatment options include lifestyle changes,
        medication, and insulin therapy.
        """,
        "metadata": {
            "source": "diabetes.pdf",
            "file_type": ".pdf"
        }
    }

    chunks = create_chunks(document)

    print("TOTAL CHUNKS:")
    print(len(chunks))

    print("\nFIRST CHUNK:")
    print(chunks[0])

    print("\nSTATS:")
    print(get_chunk_stats(chunks))
