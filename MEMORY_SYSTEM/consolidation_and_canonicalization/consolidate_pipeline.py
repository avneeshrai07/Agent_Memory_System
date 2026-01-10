from MEMORY_SYSTEM.consolidation_and_canonicalization.consolidate_memories import consolidate_memories
from MEMORY_SYSTEM.consolidation_and_canonicalization.topic_canonicalization import canonicalize_topics


async def run_full_consolidation(
    conn,
    user_id: str,
    similarity_threshold: float = 0.85,
    candidate_limit: int = 50,
):
    """
    Runs:
    1. Level-1 duplicate consolidation
    2. Level-2 topic canonicalization
    """

    level1 = await consolidate_memories(
        conn,
        user_id=user_id,
        similarity_threshold=similarity_threshold,
        candidate_limit=candidate_limit,
    )

    level2 = await canonicalize_topics(
        conn,
        user_id=user_id,
    )

    return {
        "level_1": level1,
        "level_2": level2,
    }
