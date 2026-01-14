# MEMORY_SYSTEM/direct_context/builder.py

from MEMORY_SYSTEM.direct_context.dc_types import DerivedCurrentContext


async def build_derived_current_context(
    *,
    session_id: str,
    user_id: str,
    message_store,
    stm_store,
    message_limit: int = 5,
    stm_limit: int = 7,
) -> DerivedCurrentContext:
    try:
        recent_messages = await message_store.fetch_last_messages(
            session_id=session_id,
            limit=message_limit,
        )

        stm_records = await stm_store.fetch_active_records(
            user_id=user_id,
            limit=stm_limit,
        )

        return {
            "recent_messages": [
                {
                    "role": m.role,
                    "content": m.content,
                }
                for m in recent_messages
            ],
            "active_state": [
                {
                    "type": r.state_type,
                    "content": r.statement,
                    "timestamp": r.created_at,
                }
                for r in stm_records
            ],
        }

    except Exception as e:
        raise RuntimeError("Failed to build Derived Current Context") from e
