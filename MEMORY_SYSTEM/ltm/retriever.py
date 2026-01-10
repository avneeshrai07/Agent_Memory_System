from typing import List, Dict
import traceback
import re

from MEMORY_SYSTEM.database.connect.connect import db_manager
from MEMORY_SYSTEM.embeddings.encoder import create_embedding


# =====================================================
# Tunables (empirically safe defaults)
# =====================================================
VECTOR_LIMIT = 12
MAX_DISTANCE = 1.05          # allow short-question recall
MIN_CONFIDENCE = 0.6


# =====================================================
# Utilities
# =====================================================
def to_pgvector_literal(vec: List[float]) -> str:
    """Convert list[float] to pgvector literal string."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def normalize_text(text: str) -> str:
    """Canonicalize text for semantic deduplication."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def extract_query_tokens(query: str) -> set[str]:
    """Extract normalized tokens from query."""
    return {
        t.lower()
        for t in re.findall(r"\b[a-zA-Z]{2,}\b", query)
    }


def chunk_query(user_query: str) -> List[str]:
    """
    Split user input into atomic semantic chunks.
    Prevents embedding dilution for multi-fact queries.
    """
    parts = re.split(r"\n|\.|\band\b", user_query)
    return [p.strip() for p in parts if len(p.strip()) > 8]


def is_relevant(row: Dict, query_tokens: set[str]) -> bool:
    """
    Hybrid relevance gate:
    - confidence first
    - topic-intent alignment
    - distance as fallback
    """

    if row["confidence_score"] < MIN_CONFIDENCE:
        return False

    topic = (row.get("semantic_topic") or "").lower()
    distance = row["distance"]

    # 1️⃣ Topic ↔ intent alignment (primary signal)
    if topic and topic in query_tokens:
        return True

    # 2️⃣ Vector fallback (short queries)
    if distance <= MAX_DISTANCE:
        return True

    return False


# =====================================================
# Main Retrieval Function
# =====================================================
async def retrieve_ltm_memories(
    user_id: str,
    user_query: str
) -> List[Dict]:
    """
    Retrieve relevant long-term memories for a user query.

    Guarantees:
    - Never raises
    - Never returns prompt text
    - Returns clean, deduplicated memory records
    """

    print("\n================ FETCHING LT MEMORIES ================\n")

    # -------------------------------------------------
    # 1. Preprocess query
    # -------------------------------------------------
    try:
        query_chunks = chunk_query(user_query)
        if not query_chunks:
            return []

        query_tokens = extract_query_tokens(user_query)

    except Exception:
        print("❌ [LTM-RETRIEVE] Query preprocessing failed")
        traceback.print_exc()
        return []

    # -------------------------------------------------
    # 2. DB pool
    # -------------------------------------------------
    try:
        pool = await db_manager.get_pool()
    except Exception:
        print("❌ [LTM-RETRIEVE] Failed to acquire DB pool")
        traceback.print_exc()
        return []

    all_rows: List[Dict] = []

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

            all_rows.extend(dict(r) for r in rows)

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

            if not is_relevant(row, query_tokens):
                continue

            seen.add(norm_fact)
            relevant.append(row)

        print("\n================ FILTERED LT MEMORIES ================\n")
        print(relevant)
        
        return relevant

    except Exception:
        print("❌ [LTM-RETRIEVE] Filtering/deduplication failed")
        traceback.print_exc()
        return []
