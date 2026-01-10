from typing import List, Dict
import traceback
import re

from MEMORY_SYSTEM.database.connect.connect import db_manager
from MEMORY_SYSTEM.embeddings.encoder import create_embedding
from MEMORY_SYSTEM.ltm.context_builder import build_ltm_context

# -------------------------------
# Tunables (REALISTIC defaults)
# -------------------------------
VECTOR_LIMIT = 12
DISTANCE_THRESHOLD = 0.80   # cosine distance (similarity >= 0.20)
MIN_CONFIDENCE = 0.6


# -------------------------------
# Utilities
# -------------------------------
def to_pgvector_literal(vec: List[float]) -> str:
    """Convert list[float] to pgvector literal string."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def normalize_text(text: str) -> str:
    """Light canonicalization for deduplication."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def chunk_query(user_query: str) -> List[str]:
    """
    Split multi-fact user input into atomic semantic chunks.
    """
    # Simple rule-based splitter (works well in practice)
    parts = re.split(r"\n|\.|\band\b", user_query)
    chunks = [p.strip() for p in parts if len(p.strip()) > 8]
    return chunks


def is_relevant(row: Dict, query_text: str) -> bool:
    """
    Hybrid relevance check.
    """
    if row["confidence_score"] < MIN_CONFIDENCE:
        return False

    if row["distance"] <= DISTANCE_THRESHOLD:
        return True

    topic = row.get("semantic_topic") or ""
    return topic.lower() in query_text.lower()


# -------------------------------
# Main Retrieval Function
# -------------------------------
async def retrieve_ltm_memories(
    user_id: str,
    user_query: str
) -> List[Dict]:
    """
    Retrieve relevant long-term memories for a given user query.

    - Safe (never crashes agent)
    - Chunk-aware
    - Deduplicated
    - Returns DATA, not prompt text
    """

    print("\n================ FETCHING LT MEMORIES ================\n")

    # -------------------------------------------------
    # 1. Chunk the query (CRITICAL FIX)
    # -------------------------------------------------
    try:
        query_chunks = chunk_query(user_query)
        if not query_chunks:
            return []
    except Exception:
        print("❌ [LTM-RETRIEVE] Failed to chunk query")
        traceback.print_exc()
        return []

    all_rows: List[Dict] = []

    # -------------------------------------------------
    # 2. DB pool
    # -------------------------------------------------
    try:
        pool = await db_manager.get_pool()
    except Exception:
        print("❌ [LTM-RETRIEVE] Failed to acquire DB pool")
        traceback.print_exc()
        return []

    # -------------------------------------------------
    # 3. Multi-pass vector retrieval
    # -------------------------------------------------
    for chunk in query_chunks:
        try:
            embedding = await create_embedding(chunk)

            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()

            query_vector = to_pgvector_literal(embedding)

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        fact,
                        memory_type,
                        semantic_topic,
                        confidence_score,
                        embedding <-> $2::vector AS distance
                    FROM agentic_memory_schema.memories
                    WHERE user_id = $1
                      AND status = 'active'
                    ORDER BY embedding <-> $2::vector
                    LIMIT $3
                    """,
                    user_id,
                    query_vector,
                    VECTOR_LIMIT
                )

            all_rows.extend([dict(r) for r in rows])

        except Exception:
            print("❌ [LTM-RETRIEVE] Vector search failed for chunk:", chunk)
            traceback.print_exc()
            continue

    print("\n================ RAW LT MEMORIES (DB) ================\n")
    print(all_rows)

    # -------------------------------------------------
    # 4. Filter + deduplicate
    # -------------------------------------------------
    try:
        seen = set()
        relevant: List[Dict] = []

        for row in all_rows:
            norm_fact = normalize_text(row["fact"])

            if norm_fact in seen:
                continue

            if not is_relevant(row, user_query):
                continue

            seen.add(norm_fact)
            relevant.append(row)

        print("\n================ FILTERED LT MEMORIES ================\n")
        print(relevant)

        return build_ltm_context(user_query=user_query, memories=relevant)

    except Exception:
        print("❌ [LTM-RETRIEVE] Failed during filtering/deduplication")
        traceback.print_exc()
        return []
