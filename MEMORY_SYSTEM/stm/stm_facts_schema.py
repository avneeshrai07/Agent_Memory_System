from typing import List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field


# -----------------------------
# Atomic STM message
# -----------------------------
class STMMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(
        ...,
        description="Who produced this message. Used only for short-term conversational coherence."
    )
    content: str = Field(
        ...,
        description="Raw message text. This should be pruned aggressively and never stored long-term."
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp when the message was produced."
    )


# -----------------------------
# STM cognitive state
# -----------------------------
class STMCognitiveState(BaseModel):
    current_goal: Optional[str] = Field(
        None,
        description="The user's immediate objective in this session (e.g. 'optimize database performance')."
    )

    stage: Optional[Literal["exploration", "diagnosis", "solution", "validation"]] = Field(
        None,
        description="Where the user currently is in the problem-solving lifecycle."
    )

    confirmed_constraints: List[str] = Field(
        default_factory=list,
        description="Explicit constraints confirmed during the session (temporary unless promoted to LTM)."
    )

    assumptions: List[str] = Field(
        default_factory=list,
        description="Working assumptions the agent is using for reasoning in this session."
    )

    decisions_made: List[str] = Field(
        default_factory=list,
        description="Decisions or conclusions already reached to avoid repeating reasoning."
    )

    open_questions: List[str] = Field(
        default_factory=list,
        description="Questions still unresolved in the current session."
    )


# -----------------------------
# STM execution / agent state
# -----------------------------
class STMAgentState(BaseModel):
    active_agent: Optional[str] = Field(
        None,
        description="Which internal agent is currently handling the session."
    )

    tools_used: List[str] = Field(
        default_factory=list,
        description="Tools or functions already invoked in this session."
    )

    waiting_for_user: bool = Field(
        default=False,
        description="Whether the agent is waiting for user input before proceeding."
    )


# -----------------------------
# Root STM object
# -----------------------------
class STMContext(BaseModel):
    session_id: str = Field(
        ...,
        description="Unique identifier for the active session."
    )

    user_id: str = Field(
        ...,
        description="Identifier of the user owning this session."
    )

    messages: List[STMMessage] = Field(
        default_factory=list,
        description="Rolling window of the most recent messages (hard-capped, e.g. last 5 turns)."
    )

    cognitive_state: STMCognitiveState = Field(
        default_factory=STMCognitiveState,
        description="Compressed reasoning state for continuity without token bloat."
    )

    agent_state: STMAgentState = Field(
        default_factory=STMAgentState,
        description="Internal agent execution state."
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Session creation timestamp."
    )

    expires_at: datetime = Field(
        ...,
        description="When this STM context must be discarded (TTL enforced)."
    )
