import asyncio
from typing import Union, List
import numpy as np
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# Model initialized at app startup (ready immediately)
# ------------------------------------------------------------
MODEL_NAME = "BAAI/bge-large-en-v1.5"

try:
    EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Failed to initialize embedding model at startup: {e}")


async def create_embedding(
    text: Union[str, List[str]],
    normalize: bool = True,
) -> np.ndarray:
    """
    Async embedding function (model already initialized)

    Args:
        text: string or list of strings
        normalize: cosine-safe normalization

    Returns:
        np.ndarray (1024,) or (N, 1024)
    """
    try:
        if not text:
            raise ValueError("Input text is empty")

        texts = [text] if isinstance(text, str) else text

        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: EMBEDDING_MODEL.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        return embeddings[0] if isinstance(text, str) else embeddings

    except Exception as e:
        raise RuntimeError(f"Embedding generation failed: {e}")







# import asyncio
# import numpy as np

# # cosine similarity (embeddings are already normalized)
# def cosine(a: np.ndarray, b: np.ndarray) -> float:
#     return float(np.dot(a, b))


# async def test_embedding_basic():
#     context = """
#     User has a 500GB PostgreSQL database.
#     Queries take more than 30 seconds.
#     Full table scans are happening.
#     Indexes already exist on join columns.
#     System uses Redis for caching.
#     Application is built with FastAPI.
#     pgvector is used for embeddings.
#     User prefers concise answers.
#     System is in production.
#     Performance optimization is the goal.
#     """.strip()

#     # Encode full context
#     context_embedding = await create_embedding(context)

#     probes = {
#         "database_size": "large PostgreSQL database",
#         "performance_issue": "slow database queries",
#         "indexing": "indexes on database tables",
#         "caching": "Redis cache usage",
#         "backend_stack": "FastAPI backend service",
#         "preferences": "user prefers concise responses",
#         "irrelevant": "user likes football",
#     }

#     results = {}

#     for key, text in probes.items():
#         probe_emb = await create_embedding(text)
#         results[key] = cosine(context_embedding, probe_emb)

#     print("\nSemantic probe results:\n")
#     for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True):
#         print(f"{k:20s} -> {v:.3f}")


# asyncio.run(test_embedding_basic())
