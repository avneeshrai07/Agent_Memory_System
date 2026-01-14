# dcc/types.py

from typing import TypedDict, List
from datetime import datetime

class RecentMessage(TypedDict):
    role: str
    content: str

class ActiveState(TypedDict):
    type: str
    content: str
    timestamp: datetime

class DerivedCurrentContext(TypedDict):
    recent_messages: List[RecentMessage]
    active_state: List[ActiveState]
