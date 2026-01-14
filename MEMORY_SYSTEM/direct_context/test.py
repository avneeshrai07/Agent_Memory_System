import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR,  "..",".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from MEMORY_SYSTEM.storage.message_store import MessageStore
from MEMORY_SYSTEM.storage.stm_store import STMStore
from MEMORY_SYSTEM.direct_context.builder import build_derived_current_context
from MEMORY_SYSTEM.stm.stm_intent import STMIntent

import asyncio


async def main():
    try:
        message_store = MessageStore()
        stm_store = STMStore()

        # simulate interaction
        await message_store.add_message(
            session_id="s1",
            role="user",
            content="We should focus on enterprise customers",
        )

        # ✅ create STMIntent (this mirrors real LLM output)
        intent = STMIntent(
            should_write=True,
            state_type="decision",
            statement="Target enterprise customers",
            rationale="Enterprise has higher ACV",
            confidence=0.92,
        )

        # ✅ apply intent (correct API)
        await stm_store.apply_intent(
            user_id="u1",
            intent=intent,
        )

        dcc = await build_derived_current_context(
            session_id="s1",
            user_id="u1",
            message_store=message_store,
            stm_store=stm_store,
        )

        print("Derived Current Context:")
        print(dcc)

    except Exception as e:
        raise RuntimeError("Application failed") from e


if __name__ == "__main__":
    asyncio.run(main())
