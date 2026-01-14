# storage/message_store.py

from typing import List
from datetime import datetime
from MEMORY_SYSTEM.storage.message_model import Message


class MessageStore:
    def __init__(self):
        self._messages: List[Message] = []

    async def add_message(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        try:
            self._messages.append(
                Message(
                    session_id=session_id,
                    role=role,
                    content=content,
                    created_at=datetime.utcnow(),
                )
            )
        except Exception as e:
            raise RuntimeError("Failed to add message") from e

    async def fetch_last_messages(
        self,
        *,
        session_id: str,
        limit: int,
    ) -> List[Message]:
        try:
            msgs = [
                m for m in self._messages
                if m.session_id == session_id
            ]
            return sorted(msgs, key=lambda m: m.created_at)[-limit:]
        except Exception as e:
            raise RuntimeError("Failed to fetch messages") from e
