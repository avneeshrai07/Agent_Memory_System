# layer3_pattern_detector.py
import asyncpg
from typing import List, Dict
from datetime import datetime, timedelta
from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager
from collections import Counter

class PatternDetector:
    def __init__(self):
        self.pool = None  
    
    async def _get_pool(self):
        if self.pool is None:
            self.pool = await db_manager.wait_for_connection_pool_pool()
        return self.pool
    
    async def detect_preferences(self, user_id: str, min_signals: int = 3):
        """Pattern 1: User preferences (e.g., 'code-only', 'email style')"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Count user_preference facts by topic
            rows = await conn.fetch("""
                SELECT topic, COUNT(*) as frequency, AVG(importance_score) as avg_importance
                FROM agent_memory_system.memories 
                WHERE user_id = $1 AND category = 'user_preference' AND status = 'active'
                GROUP BY topic 
                HAVING COUNT(*) >= $2
                ORDER BY frequency DESC
            """, user_id, min_signals)
            
            patterns = []
            for row in rows:
                confidence = min(0.95, row['frequency'] * row['avg_importance'] / 10)
                if confidence >= 0.75:
                    patterns.append({
                        'pattern_type': 'preference',
                        'topic': row['topic'],
                        'frequency': row['frequency'],
                        'confidence': confidence
                    })
            return patterns
        
    async def detect_domains(self, user_id: str, min_signals: int = 5) -> List[Dict]:
        """Pattern 2: Problem domains ('marketing', 'databases')"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT topic, COUNT(*) as frequency, AVG(importance_score) as avg_imp,
                       COUNT(*)::float / (SELECT COUNT(*) FROM agent_memory_system.memories 
                                          WHERE user_id = $1 AND status = 'active') as pct
                FROM agent_memory_system.memories 
                WHERE user_id = $1 AND category IN ('problem_domain', 'technical_context') 
                  AND status = 'active'
                GROUP BY topic HAVING COUNT(*) >= $2
                ORDER BY frequency DESC
            """, user_id, min_signals)
            
            return [{
                'pattern_type': 'domain',
                'topic': row['topic'],
                'frequency': row['frequency'],
                'percentage': row['pct'],
                'confidence': min(0.95, row['pct'])
            } for row in rows if row['pct'] >= 0.20]
    
    async def detect_expertise(self, user_id: str, min_signals: int = 4) -> List[Dict]:
        """Pattern 3: Expertise level (Python/DB expert)"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            # Complex topics + high importance = expertise
            rows = await conn.fetch("""
                SELECT topic, AVG(importance_score) as avg_imp, COUNT(*) as signals,
                       STRING_AGG(DISTINCT category, ', ') as categories
                FROM agent_memory_system.memories 
                WHERE user_id = $1 AND status = 'active'
                GROUP BY topic HAVING COUNT(*) >= $2 AND AVG(importance_score) >= 8
            """, user_id, min_signals)
            
            patterns = []
            for row in rows:
                expertise_score = (row['avg_imp'] / 10) * (row['signals'] / 10)
                if expertise_score >= 0.7:
                    patterns.append({
                        'pattern_type': 'expertise',
                        'topic': row['topic'],
                        'categories': row['categories'],
                        'confidence': expertise_score
                    })
            return patterns
    
    async def detect_constraints(self, user_id: str, min_signals: int = 2) -> List[Dict]:
        """Pattern 4: Constraints ('no schema changes')"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT fact, COUNT(*) as frequency
                FROM agent_memory_system.memories 
                WHERE user_id = $1 AND category = 'constraint' AND status = 'active'
                GROUP BY fact HAVING COUNT(*) >= $2
                ORDER BY frequency DESC
            """, user_id, min_signals)
            
            return [{
                'pattern_type': 'constraint',
                'fact': row['fact'],
                'frequency': row['frequency'],
                'confidence': min(1.0, row['frequency'] * 0.9)
            } for row in rows]
    
    async def detect_behaviors(self, user_id: str, min_signals: int = 3) -> List[Dict]:
        """Pattern 5: Learned behaviors ('ignores explanations')"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT topic, COUNT(*) as frequency
                FROM agent_memory_system.memories 
                WHERE user_id = $1 AND category = 'learned_pattern' AND status = 'active'
                GROUP BY topic HAVING COUNT(*) >= $2
                ORDER BY frequency DESC
            """, user_id, min_signals)
            
            return [{
                'pattern_type': 'behavior',
                'topic': row['topic'],
                'frequency': row['frequency'],
                'confidence': min(0.95, row['frequency'] / 5.0)
            } for row in rows]
        
    
