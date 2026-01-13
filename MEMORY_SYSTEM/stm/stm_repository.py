from uuid import uuid4
from datetime import datetime
from typing import Optional, List
from MEMORY_SYSTEM.stm.stm_schema import STMEntry


class STMRepository:

    def __init__(self, conn):
        self.conn = conn
        print("[STM_REPO] Initialized")

    async def add_entry(
        self,
        user_id,
        state_type,
        statement,
        rationale=None,
        applies_to=None,
        supersedes=None,
        confidence=1.0
    ) -> STMEntry:
        try:
            print("[STM_ADD] Starting STM entry creation")

            stm_id = uuid4()
            now = datetime.utcnow()

            entry = STMEntry(
                stm_id=stm_id,
                timestamp=now,
                state_type=state_type,
                statement=statement,
                rationale=rationale,
                applies_to=applies_to,
                supersedes=supersedes,
                confidence=confidence
            )

            async with self.conn.transaction():

                if supersedes:
                    print(f"[STM_ADD] Superseding STM {supersedes}")
                    await self.conn.execute(
                        """
                        UPDATE agentic_memory_schema.stm_entries
                        SET is_active = FALSE
                        WHERE stm_id = $1
                        """,
                        supersedes
                    )

                print("[STM_ADD] Inserting new STM entry")
                await self.conn.execute(
                    """
                    INSERT INTO agentic_memory_schema.stm_entries (
                        stm_id, user_id, state_type, statement,
                        rationale, applies_to, supersedes, confidence
                    )
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                    """,
                    stm_id, user_id, state_type, statement,
                    rationale, applies_to, supersedes, confidence
                )

            print("[STM_ADD] STM entry committed successfully")
            return entry

        except Exception as e:
            print("[STM_ADD][ERROR]", str(e))
            raise



    async def get_active_state(self, user_id) -> List[STMEntry]:
        try:
            print("[STM_READ] Fetching active STM entries")

            rows = await self.conn.fetch(
                """
                SELECT *
                FROM agentic_memory_schema.stm_entries
                WHERE user_id = $1 AND is_active = TRUE
                ORDER BY created_at DESC
                """,
                user_id
            )

            entries = [
                STMEntry(
                    stm_id=row["stm_id"],
                    timestamp=row["created_at"],
                    state_type=row["state_type"],
                    statement=row["statement"],
                    rationale=row["rationale"],
                    applies_to=row["applies_to"],
                    supersedes=row["supersedes"],
                    confidence=row["confidence"]
                )
                for row in rows
            ]

            print(f"[STM_READ] Retrieved {len(entries)} active STM entries")
            return entries

        except Exception as e:
            print("[STM_READ][ERROR]", str(e))
            raise

