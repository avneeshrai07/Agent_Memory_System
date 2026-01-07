import sys
import os

# Get the absolute path of the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))  # '..' goes one level up

# Add root to sys.path
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


import asyncio
from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager
from MEMORY_SYSTEM.CONSOLIDATION.consolidate_memories import consolidate_memories





async def run_consolidation(user_id):
    """Run consolidation for your user."""
    
    pool = await db_manager.wait_for_connection_pool_pool()
    
    async with pool.acquire() as conn:
        print("="*80)
        print("📊 BEFORE CONSOLIDATION")
        print("="*80)
        
        # Before stats
        pending = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM agent_memory_system.memories
            WHERE user_id = $1 AND needs_consolidation = true;
        """, user_id)
        merged = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM agent_memory_system.memories
            WHERE user_id = $1 AND status = 'merged';
        """, user_id)
        print(f"Pending: {pending}, Merged: {merged}, Frequency>1: {await conn.fetchval('SELECT COUNT(*) FROM agent_memory_system.memories WHERE user_id = $1 AND frequency > 1', user_id)}")
        
        print("\n" + "="*80)
        print("🔄 RUNNING CONSOLIDATION...")
        print("="*80)
        
        # Run consolidation (lower threshold for more merges)
        result = await consolidate_memories(conn, user_id, similarity_threshold=0.75)
        
        print("\n" + "="*80)
        print("📊 AFTER CONSOLIDATION")
        print("="*80)
        
        # After stats
        pending_after = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM agent_memory_system.memories
            WHERE user_id = $1 AND needs_consolidation = true;
        """, user_id)
        merged_after = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM agent_memory_system.memories
            WHERE user_id = $1 AND status = 'merged';
        """, user_id)
        print(f"Pending: {pending_after}, Merged: {merged_after}")
        
        print(f"\n✅ SUCCESS: Merged {result['merged']} memories!")
        
        if result.get('consolidations', []):  # ← Safe access
            print("\n📋 Consolidation details:")
            for cons in result['consolidations']:
                print(f"  • '{cons['canonical_topic']}' ← {cons['merged_count']} merged")

if __name__ == "__main__":
    asyncio.run(run_consolidation(user_id = "570bfbe7-5474-4856-bf99-d5fac4b885a2"))
