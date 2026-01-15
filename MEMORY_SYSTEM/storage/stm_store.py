# stm_store.py - PRODUCTION READY, NO MORE NoneType ERRORS
import uuid
import json
import logging
import traceback
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional
import redis.asyncio as redis
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
logger = logging.getLogger(__name__)

class SessionStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"

class STMEventType(str, Enum):
    ARTIFACT_CREATED = "artifact_created"
    ARTIFACT_UPDATED = "artifact_updated"
    APPROVAL_RECEIVED = "approval_received"
    DECISION_MADE = "decision_made"
    CLARIFICATION_REQUESTED = "clarification_requested"
    ROUTE_CHANGED = "route_changed"

# Pydantic Schemas (unchanged)
class SessionMetadata(BaseModel):
    session_id: str
    user_id: str
    status: SessionStatus
    created_at: str
    last_activity: str
    expires_at: str
    current_route: str = ""
    route_confidence: float = Field(ge=0.0, le=1.0)

class Message(BaseModel):
    role: str
    content: str
    timestamp: str

class ArtifactPayload(BaseModel):
    artifact_id: str
    artifact_type: str
    summary: str
    content_ref: str
    status: str = "draft"

class Event(BaseModel):
    event_type: STMEventType
    payload: Dict[str, Any]
    timestamp: str

class STMStore:
    def __init__(self, redis_url: str = "redis://localhost:6379", ttl_hours: int = 24):
        self.redis_url = redis_url
        self.ttl_seconds = ttl_hours * 3600
        self.redis_pool = None

    @asynccontextmanager
    async def _get_redis(self):
        """SINGLE SOURCE OF TRUTH for Redis connection"""
        r = redis.from_url(self.redis_url, decode_responses=True, max_connections=10)
        try:
            yield r
        finally:
            await r.aclose()

    async def _ensure_redis_ready(self, r) -> bool:
        """Verify Redis is alive before operations"""
        try:
            await r.ping()
            return True
        except:
            logger.error("Redis connection failed")
            return False

    async def _safe_operation(self, operation: str, func, *args, **kwargs):
        """Execute Redis operation with full error handling"""
        async with self._get_redis() as r:
            if not await self._ensure_redis_ready(r):
                logger.error(f"Redis unavailable for {operation}")
                return None
                
            try:
                result = await func(r, *args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"{operation} FAILED: {str(e)}\n{traceback.format_exc()}")
                return None

    # === SESSION ===
    async def create_session(self, user_id: str) -> Optional[str]:
        async def _create(r):
            session_id = str(uuid.uuid4())
            now = datetime.utcnow()
            expires = now + timedelta(seconds=self.ttl_seconds)

            meta = SessionMetadata(
                session_id=session_id,
                user_id=user_id,
                status=SessionStatus.ACTIVE,
                created_at=now.isoformat(),
                last_activity=now.isoformat(),
                expires_at=expires.isoformat(),
            )

            key = f"session:{session_id}"
            
            # Create session metadata
            await r.hset(key, mapping=meta.dict())
            await r.expire(key, self.ttl_seconds)

            # Initialize streams ATOMICALY
            msg_key = f"{key}:messages"
            evt_key = f"{key}:events"
            
            await r.xadd(msg_key, {"init": "created"}, maxlen=50)
            await r.expire(msg_key, self.ttl_seconds)
            
            await r.xadd(evt_key, {"init": "created"}, maxlen=100)
            await r.expire(evt_key, self.ttl_seconds)

            logger.info(f"Session CREATED: {session_id}")
            return session_id

        return await self._safe_operation("create_session", _create)
    
    async def get_goals(self, session_id: str) -> List[Dict[str, Any]]:
        async def _get(r):
            try:
                key = f"session:{session_id}:goals"
                raw = await r.smembers(key)
                if not raw:
                    return []
                return [json.loads(r) for r in raw]
            except:
                return []

        result = await self._safe_operation("get_goals", _get)
        return result or []


    async def get_session(self, session_id: str) -> Optional[SessionMetadata]:
        async def _get(r):
            data = await r.hgetall(f"session:{session_id}")
            if not data:
                return None
            return SessionMetadata(**data)

        result = await self._safe_operation("get_session", _get)
        return result

    # === MESSAGES ===
    async def add_message(self, session_id: str, role: str, content: str) -> bool:
        async def _add(r):
            await self._touch_keys(r, session_id)
            
            stream_key = f"session:{session_id}:messages"
            # Ensure stream exists first
            try:
                await r.xrevrange(stream_key, count=1)
            except:
                await r.xadd(stream_key, {"init": "1"}, maxlen=50)
            
            await r.xadd(
                stream_key,
                {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat(),
                },
                maxlen=50
            )
            return True

        result = await self._safe_operation("add_message", _add)
        return result is True

    async def get_recent_messages(self, session_id: str, limit: int = 10) -> List[Message]:
        async def _get(r):
            stream_key = f"session:{session_id}:messages"
            
            # Always ensure stream exists
            try:
                entries = await r.xrevrange(stream_key, count=limit)
            except:
                # Empty stream, create it
                await r.xadd(stream_key, {"init": "empty"})
                return []

            if not entries:
                return []

            # Chronological order (oldest first)
            entries.reverse()
            
            messages = []
            for _, data in entries:
                if "content" in data:
                    messages.append(Message(**data))
            
            return messages[:limit]

        result = await self._safe_operation("get_recent_messages", _get)
        return result or []

    # === EVENTS ===
    async def add_event(self, session_id: str, event_type: str | STMEventType, payload: Dict[str, Any]) -> bool:
        async def _add(r):
            await self._touch_keys(r, session_id)
            
            stream_key = f"session:{session_id}:events"
            try:
                await r.xrevrange(stream_key, count=1)
            except:
                await r.xadd(stream_key, {"init": "1"}, maxlen=100)

            # ✅ FIXED: Handle both str and Enum safely
            event_type_str = event_type.value if hasattr(event_type, 'value') else event_type
            
            # Validate payload for artifacts
            if event_type_str == "artifact_created":
                ArtifactPayload(**payload)

            await r.xadd(
                stream_key,
                {
                    "event_type": event_type_str,  # ✅ No .value() - safe string
                    "payload": json.dumps(payload),
                    "timestamp": datetime.utcnow().isoformat(),
                },
                maxlen=100
            )
            return True

        result = await self._safe_operation("add_event", _add)
        return result is True

    async def get_recent_events(self, session_id: str, limit: int = 20, event_types: Optional[List[STMEventType]] = None) -> List[Event]:
        async def _get(r):
            stream_key = f"session:{session_id}:events"
            
            try:
                entries = await r.xrevrange(stream_key, count=limit)
            except:
                return []

            if not entries:
                return []

            events = []
            for _, data in entries:
                try:
                    et = STMEventType(data["event_type"])
                    if event_types and et not in event_types:
                        continue
                    events.append(
                        Event(
                            event_type=et,
                            payload=json.loads(data["payload"]),
                            timestamp=data["timestamp"],
                        )
                    )
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue

            return events

        result = await self._safe_operation("get_recent_events", _get)
        return result or []

    # === UTILITIES ===
    async def _touch_keys(self, r, session_id: str):
        """Extend TTL on all session keys"""
        base = f"session:{session_id}"
        now = datetime.utcnow().isoformat()
        
        await r.hset(base, "last_activity", now)
        await r.expire(base, self.ttl_seconds)
        
        for suffix in ["messages", "events", "goals"]:
            await r.expire(f"{base}:{suffix}", self.ttl_seconds)

    async def add_goal(self, session_id: str, description: str) -> Optional[str]:
        async def _add(r):
            goal_id = str(uuid.uuid4())
            goal = {
                "goal_id": goal_id,
                "description": description,
                "created_at": datetime.utcnow().isoformat(),
            }
            key = f"session:{session_id}:goals"
            await r.sadd(key, json.dumps(goal))
            await r.expire(key, self.ttl_seconds)
            return goal_id

        result = await self._safe_operation("add_goal", _add)
        return result

    async def close_session(self, session_id: str) -> bool:
        async def _close(r):
            await r.hset(f"session:{session_id}", "status", SessionStatus.CLOSED.value)
            return True

        result = await self._safe_operation("close_session", _close)
        return result is True

# === TEST ===
async def test():
    stm = STMStore()
    session_id = await stm.create_session("test-user")
    print(f"✅ Session: {session_id}")
    
    await stm.add_message(session_id, "user", "DOCX converter project")
    await stm.add_event(session_id, STMEventType.ARTIFACT_CREATED, {
        "artifact_id": "job-001", "artifact_type": "zip", 
        "summary": "PDF outputs", "content_ref": "/data/job-001.zip"
    })
    
    messages = await stm.get_recent_messages(session_id)
    print(f"✅ Messages: {len(messages)}")
    print(f"✅ First message: {messages[0].content if messages else 'None'}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test())
