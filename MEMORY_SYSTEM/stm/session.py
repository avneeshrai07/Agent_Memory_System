# MEMORY_SYSTEM/stm/session.py

from dataclasses import dataclass
from typing import List

@dataclass
class Message:
    role: str   # "user" | "assistant"
    content: str

@dataclass
class STMContext:
    session_id: str
    messages: List[Message]

    def add(self, role: str, content: str):
        self.messages.append(Message(role, content))

    def last_n(self, n: int) -> List[Message]:
        return self.messages[-n:]
