# create_layer_3_tables.py
from MEMORY_SYSTEM.DATABASE.CONNECT.connect import db_manager


async def ensure_layer_3_table_exists():
    """
    Creates agent_memory_system.user_patterns table with:
    - Pattern type support (preference, domain, constraint, expertise, validation, persona)
    - Confidence tracking and recalibration support
    - Validation metrics (success/failure tracking)
    - Persona metadata (tone, style, template, objectives, audience)
    - Proper indexes for fast pattern retrieval
    """
    try:
        pool = await db_manager.wait_for_connection_pool_pool()
        async with pool.acquire() as conn:
            
            # ================================================================
            # TABLE: user_patterns (Layer 3 - detected behavioral patterns)
            # ================================================================
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_memory_system.user_patterns (
                    -- Identity
                    id                  BIGSERIAL PRIMARY KEY,
                    user_id             TEXT NOT NULL,
                    
                    -- Pattern classification
                    pattern_type        TEXT NOT NULL 
                        CHECK (pattern_type IN (
                            'preference',    -- How user wants things (format, style, tone)
                            'expertise',     -- What user knows (skill level, domain knowledge)
                            'constraint',    -- What user cannot do (budget, timeline, technical)
                            'domain',        -- What problems user works on (recurring topics)
                            'validation',    -- What approaches work/don't work for this user
                            'persona'        -- Composite user communication profile
                        )),
                    pattern_name        TEXT NOT NULL,  -- e.g., 'formal_tone', 'marketing_email', 'cannot_change_schema'
                    description         TEXT,           -- Human-readable pattern description
                    
                    -- Confidence & frequency tracking
                    confidence          REAL NOT NULL 
                        CHECK (confidence >= 0 AND confidence <= 1),
                    frequency           INT NOT NULL DEFAULT 1,
                    signals_count       INT NOT NULL DEFAULT 0,  -- Total signals collected
                    
                    -- Validation tracking (for validation patterns)
                    validation_count    INT NOT NULL DEFAULT 0,  -- Times pattern was successfully applied
                    failure_count       INT NOT NULL DEFAULT 0,  -- Times pattern failed
                    
                    -- Temporal tracking
                    first_observed      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_observed       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_validated      TIMESTAMPTZ NULL,        -- Last time pattern was tested
                    
                    -- Pattern lifecycle
                    status              TEXT NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'historical', 'conflicting', 'deprecated')),
                    
                    -- Rich metadata (JSONB for flexibility)
                    -- For persona: {tone, style, template_type, objectives, target_audience}
                    -- For validation: {approach, success_rate, last_failure_reason}
                    -- For constraints: {severity, workaround_available}
                    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
                    
                    -- Timestamps
                    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    
                    -- Ensure unique patterns per user
                    CONSTRAINT uq_user_pattern UNIQUE (user_id, pattern_type, pattern_name)
                );
            """)

            # ================================================================
            # INDEXES: Fast pattern retrieval and filtering
            # ================================================================
            
            # Primary lookup: user + type + status
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_patterns_lookup
                ON agent_memory_system.user_patterns (user_id, pattern_type, status);
            """)

            # Confidence-based filtering (high-confidence patterns first)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_patterns_confidence
                ON agent_memory_system.user_patterns (user_id, confidence DESC)
                WHERE status = 'active';
            """)

            # Frequency-based queries (most common patterns)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_patterns_frequency
                ON agent_memory_system.user_patterns (user_id, frequency DESC)
                WHERE status = 'active';
            """)

            # Recency tracking (patterns needing recalibration)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_patterns_last_observed
                ON agent_memory_system.user_patterns (user_id, last_observed)
                WHERE status = 'active';
            """)

            # Persona patterns (one per user, fast access)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_patterns_persona
                ON agent_memory_system.user_patterns (user_id)
                WHERE pattern_type = 'persona' AND status = 'active';
            """)

            # JSONB metadata search (GIN index for flexible queries)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_patterns_metadata_gin
                ON agent_memory_system.user_patterns USING GIN (metadata);
            """)

            # ================================================================
            # TRIGGER: Auto-update updated_at on row modification
            # ================================================================
            await conn.execute("""
                CREATE OR REPLACE FUNCTION agent_memory_system.set_patterns_updated_at()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at := NOW();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)

            await conn.execute("""
                DROP TRIGGER IF EXISTS trg_user_patterns_set_updated_at 
                ON agent_memory_system.user_patterns;
                
                CREATE TRIGGER trg_user_patterns_set_updated_at
                BEFORE UPDATE ON agent_memory_system.user_patterns
                FOR EACH ROW
                EXECUTE FUNCTION agent_memory_system.set_patterns_updated_at();
            """)

            # ================================================================
            # TABLE: pattern_validation_log (track pattern application results)
            # ================================================================
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_memory_system.pattern_validation_log (
                    id                  BIGSERIAL PRIMARY KEY,
                    pattern_id          BIGINT NOT NULL
                        REFERENCES agent_memory_system.user_patterns(id) ON DELETE CASCADE,
                    user_id             TEXT NOT NULL,
                    
                    -- Validation details
                    applied_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    success             BOOLEAN NOT NULL,
                    feedback_type       TEXT CHECK (feedback_type IN ('explicit', 'implicit', 'automated')),
                    
                    -- Context
                    message_id          TEXT NULL,           -- Source message that triggered validation
                    response_quality    REAL NULL            -- 0.0-1.0 quality score (if available)
                        CHECK (response_quality IS NULL OR (response_quality >= 0 AND response_quality <= 1)),
                    failure_reason      TEXT NULL,           -- Why pattern failed (if applicable)
                    
                    -- Metadata
                    metadata            JSONB NOT NULL DEFAULT '{}'::jsonb
                );
            """)

            # Index for validation log queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_validation_log_pattern
                ON agent_memory_system.pattern_validation_log (pattern_id, applied_at DESC);
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_validation_log_user
                ON agent_memory_system.pattern_validation_log (user_id, applied_at DESC);
            """)

            # ================================================================
            # ANALYZE: Refresh query planner statistics
            # ================================================================
            await conn.execute("ANALYZE agent_memory_system.user_patterns;")
            await conn.execute("ANALYZE agent_memory_system.pattern_validation_log;")

            # ================================================================
            # SUCCESS
            # ================================================================
            print("✅ Layer 3 tables created successfully:")
            print("   📊 agent_memory_system.user_patterns")
            print("      - Pattern types: preference, expertise, constraint, domain, validation, persona")
            print("      - Confidence tracking: 0.0-1.0 with validation metrics")
            print("      - Status lifecycle: active, historical, conflicting, deprecated")
            print("      - Rich metadata: JSONB with GIN index")
            print("   📈 agent_memory_system.pattern_validation_log")
            print("      - Success/failure tracking for adaptive confidence")
            print("      - Feedback types: explicit, implicit, automated")
            print("   🚀 All indexes, constraints, and triggers ready")

    except Exception as e:
        print(f"❌ Layer 3 table creation failed: {e}")
        print("   - Check: Layer 2 tables exist (memories)?")
        print("   - Check: Database permissions?")
        raise



