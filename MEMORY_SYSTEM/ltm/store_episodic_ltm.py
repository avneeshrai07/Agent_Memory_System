# MEMORY_SYSTEM/ltm/store_episodic_ltm.py

from typing import List, Dict
from datetime import datetime, timedelta
import traceback
import json

from MEMORY_SYSTEM.database.connect.connect import db_manager
from MEMORY_SYSTEM.embeddings.encoder import create_embedding


# =====================================================
# Episodic decay configuration
# =====================================================
EPISODIC_TTL = {
    "session": timedelta(hours=1),
    "multi_turn": timedelta(hours=6),
    "task": timedelta(days=2),
}


async def store_episodic_ltm(
    user_id: str,
    episodic_items: List[Dict],
    raw_context: str | None = None,
) -> None:
    """
    Store episodic long-term memory.

    Guarantees:
    - Episodic memories never collide with factual
    - Scope-based expiry is enforced
    - JSONB fields are explicitly serialized
    """

    if not episodic_items:
        return

    try:
        pool = await db_manager.get_pool()
    except Exception:
        traceback.print_exc()
        return

    async with pool.acquire() as conn:
        for item in episodic_items:
            try:
                scope = item.get("scope")
                ttl = EPISODIC_TTL.get(scope)
                if not ttl:
                    continue

                expires_at = datetime.utcnow() + ttl

                fact_text = f"{item.get('key')}: {item.get('value')}"

                # # Optional embedding (lightweight)
                # embedding = await create_embedding(fact_text)
                # if hasattr(embedding, "tolist"):
                #     embedding = embedding.tolist()

                await conn.execute(
                    """
                    INSERT INTO agentic_memory_schema.memories (
                        user_id,
                        memory_kind,
                        category,
                        topic,
                        fact,
                        confidence_score,
                        confidence_source,
                        importance,
                        metadata,
                        expires_at,
                        created_at,
                        last_updated
                    )
                    VALUES (
                        $1,
                        'episodic',
                        $2,
                        $3,
                        $4,
                        $5,
                        $6,
                        1.0,
                        $7,
                        $8,
                        NOW(),
                        NOW()
                    )
                    """,
                    user_id,
                    item.get("context_type"),          # category
                    item.get("key"),                   # topic
                    item.get("value"),                 # fact
                    item["confidence"]["score"],
                    item["confidence"]["source"],
                    json.dumps({                       # ✅ JSONB SAFE
                        "scope": scope,
                        "source": "episodic_extraction"
                    }),
                    expires_at,
                )
                print("\n🎉 [LTM EPISODIC] Storage completed successfully")
            except Exception:
                traceback.print_exc()
                continue
