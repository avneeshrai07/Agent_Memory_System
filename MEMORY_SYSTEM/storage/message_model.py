

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Role = Literal["user", "assistant"]

@dataclass
class Message:
    session_id: str
    role: Role
    content: str
    created_at: datetime
