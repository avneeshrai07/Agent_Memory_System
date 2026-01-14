from uuid import uuid4
from datetime import datetime
from typing import Dict, Optional
from MEMORY_SYSTEM.database.connect.connect import db_manager

async def commit_stm_intent(
    user_id: str,
    intent: Dict,
    applies_to: Optional[str] = None
) -> Dict:
    """
    Convert an approved LLM intent dict into an STM commit.

    Input:
      intent = {
        should_write: True,
        state_type: str,
        statement: str,
        rationale: str | None,
        confidence: float
      }

    Output:
      {
        ...intent,
        stm_id: UUID,
        created_at: datetime,
        is_active: True
      }
    """

    try:
        print("[STM_REPO] Committing STM intent")

        stm_id = uuid4()
        created_at = datetime.utcnow()

        pool = await db_manager.get_pool()

        async with pool.acquire() as conn:
            async with conn.transaction():

                # -------------------------------------------------
                # Optional: supersede previous active STM of same type
                # (conservative default: do NOT auto-supersede)
                # -------------------------------------------------
                # This is intentionally left explicit & optional

                # -------------------------------------------------
                # Insert STM entry
                # -------------------------------------------------
                await conn.execute(
                    """
                    INSERT INTO agentic_memory_schema.stm_entries (
                        stm_id,
                        user_id,
                        state_type,
                        statement,
                        rationale,
                        applies_to,
                        confidence,
                        is_active,
                        created_at
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,TRUE,$8)
                    """,
                    stm_id,
                    user_id,
                    intent["state_type"],
                    intent["statement"],
                    intent.get("rationale"),
                    applies_to,
                    intent.get("confidence", 1.0),
                    created_at
                )

            print("[STM_REPO] STM commit successful:", stm_id)

        # -------------------------------------------------
        # Augment intent dict with STM metadata
        # -------------------------------------------------
        committed_intent = {
            **intent,
            "stm_id": stm_id,
            "created_at": created_at,
            "is_active": True
        }

        return committed_intent

    except Exception as e:
        print("[STM_REPO][ERROR]", str(e))
        raise
