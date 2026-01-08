-- MEMORY_SYSTEM/database/schema/ltm_memories.sql
CREATE SCHEMA IF NOT EXISTS agentic_memory_schema;

CREATE TABLE IF NOT EXISTS agentic_memory_schema.ltm_memories (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,

    topic TEXT NOT NULL,
    fact TEXT NOT NULL,
    category TEXT NOT NULL,  -- technical_context, preference, constraint, domain, etc.

    signal_type TEXT NOT NULL, -- explicit | implicit | derived
    signal_count INT NOT NULL DEFAULT 1,

    importance INT CHECK (importance BETWEEN 1 AND 10),

    embedding VECTOR(1024),

    status TEXT NOT NULL DEFAULT 'active', -- active | historical | conflicting
    frequency INT NOT NULL DEFAULT 1,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
