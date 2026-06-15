import os
import numpy as np

from functools import lru_cache
from sentence_transformers import SentenceTransformer


# ==========================================================
# CONFIG
# ==========================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)


# ==========================================================
# MODEL LOADER
# ==========================================================

@lru_cache(maxsize=1)
def get_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ==========================================================
# EMBEDDINGS
# ==========================================================

def get_embeddings(
    texts,
    batch_size=32
):

    if isinstance(texts, str):
        texts = [texts]

    model = get_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )
