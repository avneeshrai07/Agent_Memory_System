# MEMORY_SYSTEM/CONSOLIDATION/consolidate_memories.py

from typing import List
from pgvector.asyncpg import register_vector
import asyncio


async def consolidate_memories(conn, user_id: str, similarity_threshold: float = 0.85):
    """
    Merge highly similar memories for a user.
    - Finds clusters of similar embeddings
    - Picks a canonical memory
    - Updates others as merged (merged_into_id, status='merged', needs_consolidation=false)
    """
    print(f"🔄 [CONSOLIDATION] Starting for user {user_id} (threshold: {similarity_threshold})")
    
    await register_vector(conn)

    # 1) Get candidates that need consolidation
    print(f"📊 [CONSOLIDATION] Querying memories...")
    rows = await conn.fetch("""
        SELECT id, topic, fact, embedding, importance_score
        FROM agent_memory_system.memories
        WHERE user_id = $1
          AND status = 'active'
          AND needs_consolidation = true
        ORDER BY importance_score DESC;
    """, user_id)

    print(f"📊 [CONSOLIDATION] Found {len(rows)} memories needing consolidation")
    
    if len(rows) < 2:
        print(f"ℹ️ [CONSOLIDATION] Not enough memories (<2), skipping")
        return {"merged": 0}

    # 2) Naive O(n^2) clustering by cosine similarity (embedding <=> )
    merged_count = 0
    visited = set()
    consolidations = []

    print(f"🔍 [CONSOLIDATION] Processing {len(rows)} memories...")

    for i, base in enumerate(rows):
        if base['id'] in visited:
            continue

        print(f"   🔍 Checking memory {i+1}/{len(rows)}: '{base['topic'][:50]}...'")

        # Find similar memories to this base
        similar = await conn.fetch("""
            SELECT id, topic, fact, 1 - (embedding <=> $1::vector) as similarity
            FROM agent_memory_system.memories
            WHERE user_id = $2
              AND status = 'active'
              AND needs_consolidation = true
              AND id != $3
              AND 1 - (embedding <=> $1::vector) >= $4
            ORDER BY similarity DESC;
        """, base['embedding'], user_id, base['id'], similarity_threshold)

        print(f"      Found {len(similar)} similar memories (threshold: {similarity_threshold})")
        
        if similar:
            print(f"      Top matches:")
            for sim in similar[:3]:  # Show top 3
                print(f"        - {sim['topic'][:40]}... (similarity: {sim['similarity']:.1%})")

        similar_ids = [r['id'] for r in similar]
        if not similar_ids:
            # Just mark base as processed
            await conn.execute("""
                UPDATE agent_memory_system.memories
                SET needs_consolidation = false
                WHERE id = $1;
            """, base['id'])
            print(f"      ✅ Marked as processed (no matches)")
            continue

        # 3) Merge: use base as canonical (higher importance), mark others as merged
        new_freq = 1 + len(similar_ids)
        new_importance = max(base['importance_score'], *[r.get('importance_score', 1) for r in similar])
        
        # Update canonical memory (increase frequency/importance)
        await conn.execute("""
            UPDATE agent_memory_system.memories
            SET needs_consolidation = false,
                frequency = frequency + $2,
                importance_score = GREATEST(importance_score, $3)
            WHERE id = $1;
        """, base['id'], len(similar_ids), new_importance)
        
        print(f"      📈 Updated canonical '{base['topic'][:50]}...' -> freq={new_freq}, imp={new_importance}")

        # Mark merged memories
        await conn.execute("""
            UPDATE agent_memory_system.memories
            SET status = 'merged',
                merged_into_id = $1,
                needs_consolidation = false
            WHERE id = ANY($2);
        """, base['id'], similar_ids)

        merged_count += len(similar_ids)
        visited.add(base['id'])
        visited.update(similar_ids)
        
        # Log consolidation
        consolidations.append({
            "canonical_id": base['id'],
            "canonical_topic": base['topic'],
            "merged_count": len(similar_ids),
            "similarity_threshold": similarity_threshold,
            "merged_topics": [r['topic'] for r in similar]
        })
        
        print(f"      ✅ MERGED {len(similar_ids)} memories into '{base['topic'][:50]}...'")

    print(f"✅ [CONSOLIDATION] Complete: {merged_count} memories merged")
    
    # 4) Final stats
    stats = await conn.fetchrow("""
        SELECT 
        COUNT(*) FILTER (WHERE needs_consolidation = true) AS pending,
        COUNT(*) FILTER (WHERE status = 'merged') AS merged,
        COUNT(*) FILTER (WHERE frequency > 1) AS consolidated
        FROM agent_memory_system.memories
        WHERE user_id = $1;
    """, user_id)
    
    print(f"📈 [CONSOLIDATION] Final stats: pending={stats['pending']}, merged={stats['merged']}, consolidated={stats['consolidated']}")
    
    return {
        "merged": merged_count, 
        "consolidations": consolidations,
        "stats": dict(stats)
    }
