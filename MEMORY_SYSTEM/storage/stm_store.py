# MEMORY_SYSTEM/storage/stm_store.py

from datetime import datetime
from typing import List
from MEMORY_SYSTEM.storage.stm_record import STMRecord
from MEMORY_SYSTEM.stm.stm_intent import STMIntent  # wherever you defined it


class STMStore:
    def __init__(self):
        self._records: List[STMRecord] = []

    async def apply_intent(
        self,
        *,
        user_id: str,
        intent: STMIntent,
    ) -> None:
        """
        Apply STMIntent to state memory.
        """
        try:
            if not intent.should_write:
                return

            if not intent.state_type or not intent.statement:
                return

            # Supersession: deactivate older records of same type
            for record in self._records:
                if (
                    record.user_id == user_id
                    and record.state_type == intent.state_type
                    and record.is_active
                ):
                    record.is_active = False

            # Write new state
            self._records.append(
                STMRecord(
                    user_id=user_id,
                    state_type=intent.state_type,
                    statement=intent.statement,
                    rationale=intent.rationale,
                    confidence=intent.confidence,
                    created_at=datetime.utcnow(),
                    is_active=True,
                )
            )

        except Exception as e:
            raise RuntimeError("Failed to apply STM intent") from e

    async def fetch_active_records(
        self,
        *,
        user_id: str,
        limit: int,
    ) -> List[STMRecord]:
        try:
            records = [
                r for r in self._records
                if r.user_id == user_id and r.is_active
            ]

            return sorted(records, key=lambda r: r.created_at)[-limit:]

        except Exception as e:
            raise RuntimeError("Failed to fetch STM records") from e
