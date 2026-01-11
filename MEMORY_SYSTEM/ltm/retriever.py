from typing import List, Dict
import traceback
import re
import numpy as np

from MEMORY_SYSTEM.database.connect.connect import db_manager
from MEMORY_SYSTEM.embeddings.encoder import create_embedding
from MEMORY_SYSTEM.ltm.retrieve_episodic import retrieve_episodic_context

# =====================================================
# Tunables (production-safe defaults)
# =====================================================
VECTOR_LIMIT = 20
MAX_DISTANCE = 1.05
MIN_CONFIDENCE = 0.65
INTENT_CONFIDENCE_THRESHOLD = 0.25


# =====================================================
# Intent prototypes (static, versioned)
# =====================================================
INTENT_PROTOTYPES = {
    "exploratory": [
        "high level system design and architecture overview",
        "conceptual explanation of how an AI system works",
        "overview of components and interactions",
        "big picture design of an AI agent system",
    ],
    "focused": [
        "how to implement a specific feature",
        "how to debug or fix an issue",
        "step by step implementation guidance",
        "practical backend implementation details",
    ],
    "minimal": [
        "short direct factual answer",
        "quick clarification",
        "concise response without explanation",
    ],
}

# populated at startup
INTENT_EMBEDDINGS: Dict[str, np.ndarray] = {}


# =====================================================
# Intent initialization (run once at app startup)
# =====================================================
async def initialize_intent_embeddings():
    for intent, texts in INTENT_PROTOTYPES.items():
        vectors = []
        for t in texts:
            emb = await create_embedding(t)
            if hasattr(emb, "tolist"):
                emb = emb.tolist()
            vectors.append(emb)

        INTENT_EMBEDDINGS[intent] = np.mean(
            np.array(vectors), axis=0
        )


# =====================================================
# Utilities
# =====================================================
def cosine_similarity(v1, v2) -> float:
    return float(
        np.dot(v1, v2) /
        (np.linalg.norm(v1) * np.linalg.norm(v2))
    )


def to_pgvector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def extract_query_tokens(query: str) -> set[str]:
    return {
        t.lower()
        for t in re.findall(r"\b[a-zA-Z]{2,}\b", query)
    }


def chunk_query(user_query: str) -> List[str]:
    parts = re.split(r"\n|\.|\band\b", user_query)
    return [p.strip() for p in parts if len(p.strip()) > 8]


# =====================================================
# Embedding-based intent detection
# =====================================================
async def detect_query_intent_embedding(query: str) -> str:
    emb = await create_embedding(query)
    if hasattr(emb, "tolist"):
        emb = emb.tolist()

    best_intent = "minimal"
    best_score = -1.0

    for intent, intent_vec in INTENT_EMBEDDINGS.items():
        score = cosine_similarity(emb, intent_vec)
        if score > best_score:
            best_score = score
            best_intent = intent

    if best_score < INTENT_CONFIDENCE_THRESHOLD:
        return "minimal"

    return best_intent


# =====================================================
# Intent-aware caps (aligned with canonical LTM)
# =====================================================
INTENT_LIMITS = {
    "exploratory": {
        "technical_context": 3,
        "problem_domain": 3,
        "constraint": 2,
        "preference": 1,
    },
    "focused": {
        "technical_context": 2,
        "problem_domain": 1,
        "constraint": 1,
    },
    "minimal": {
        "technical_context": 1,
        "constraint": 1,
    },
}


# =====================================================
# Main Retrieval Function
# =====================================================
# =====================================================
# Main Retrieval Function (CORRECT, PRODUCTION-GRADE)
# =====================================================
async def retrieve_ltm_memories(
    user_id: str,
    user_query: str,
    include_supporting: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Canonical LTM retrieval.

    Architecture rules enforced:
    - Episodic memory is ALWAYS retrieved
    - Episodic memory primes reasoning, does not compete
    - Factual memory is ranked deterministically
    - No heuristic gating at retrieval time
    """

    try:
        query_chunks = chunk_query(user_query)
        if not query_chunks:
            return {"episodic": [], "factual": []}

        query_tokens = extract_query_tokens(user_query)
        intent = await detect_query_intent_embedding(user_query)

    except Exception:
        traceback.print_exc()
        return {"episodic": [], "factual": []}

    # -------------------------------------------------
    # 1️⃣ ALWAYS retrieve episodic LTM (NO HEURISTICS)
    # -------------------------------------------------
    try:
        episodic = await retrieve_episodic_context(user_id)
    except Exception:
        traceback.print_exc()
        episodic = []

    # -------------------------------------------------
    # 2️⃣ Retrieve factual LTM (existing logic preserved)
    # -------------------------------------------------
    pool = await db_manager.get_pool()
    all_rows: List[Dict] = []

    status_clause = (
        "status = 'active'"
        if not include_supporting
        else "status IN ('active','supporting')"
    )

    for chunk in query_chunks:
        try:
            emb = await create_embedding(chunk)
            if hasattr(emb, "tolist"):
                emb = emb.tolist()

            vec = to_pgvector_literal(emb)

            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""
                    SELECT
                        memory_id,
                        category,
                        topic,
                        fact,
                        importance,
                        confidence_score,
                        embedding <-> $2::vector AS distance
                    FROM agentic_memory_schema.memories
                    WHERE user_id = $1
                      AND memory_kind = 'factual'
                      AND {status_clause}
                      AND confidence_score >= $4
                    ORDER BY embedding <-> $2::vector
                    LIMIT $3;
                    """,
                    user_id,
                    vec,
                    VECTOR_LIMIT,
                    MIN_CONFIDENCE,
                )

            all_rows.extend(dict(r) for r in rows)

        except Exception:
            traceback.print_exc()
            continue

    # -------------------------------------------------
    # 3️⃣ Rank factual memories (episodic-aware, not episodic-gated)
    # -------------------------------------------------
    seen = set()
    ranked: List[Dict] = []

    for row in all_rows:
        key = (row["category"], row["topic"])
        if key in seen:
            continue
        seen.add(key)

        topic_match = row["topic"].lower() in query_tokens
        vector_match = row["distance"] <= MAX_DISTANCE

        if not (topic_match or vector_match):
            continue

        # Episodic alignment boost (soft, optional)
        episodic_boost = 0.0
        for e in episodic:
            if (
                e["confidence_score"] >= 0.8
                and e["fact"].lower() in row["fact"].lower()
            ):
                episodic_boost = 1.5
                break

        row["_score"] = (
            (2.0 if topic_match else 0.0)
            + (1.0 - min(row["distance"], 1.0))
            + (row["importance"] / 10.0)
            + row["confidence_score"]
            + episodic_boost
        )

        ranked.append(row)

    ranked.sort(key=lambda r: r["_score"], reverse=True)

    # -------------------------------------------------
    # 4️⃣ Intent-aware capping (FACTUAL ONLY)
    # -------------------------------------------------
    limits = INTENT_LIMITS.get(intent, {})
    per_category = {}
    final_factual: List[Dict] = []

    for row in ranked:
        cat = row["category"]
        limit = limits.get(cat, 1)

        if per_category.get(cat, 0) >= limit:
            continue

        per_category[cat] = per_category.get(cat, 0) + 1
        row["relevance_score"] = round(row.pop("_score"), 6)
        final_factual.append(row)

    # -------------------------------------------------
    # 5️⃣ RETURN (SEPARATED, INTENTIONAL)
    # -------------------------------------------------
    return {
        "episodic": episodic,     # injected into STM later
        "factual": final_factual  # injected into reasoning
    }
