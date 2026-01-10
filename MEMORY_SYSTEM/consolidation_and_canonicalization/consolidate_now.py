import sys
import os
import asyncio

# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from MEMORY_SYSTEM.consolidation_and_canonicalization.consolidate_pipeline import run_full_consolidation
from MEMORY_SYSTEM.database.connect.connect import db_manager

TEST_USER_ID = "test_user_011"


async def main():
    pool = await db_manager.get_pool()

    async with pool.acquire() as conn:
        print("🚀 Starting FULL consolidation (Level-1 + Level-2)")

        result = await run_full_consolidation(
            conn,
            user_id=TEST_USER_ID,
            similarity_threshold=0.85,
            candidate_limit=20,
        )

        print("✅ Consolidation result:")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
