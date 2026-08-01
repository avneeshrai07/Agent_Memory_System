"""Settings objects for the library. No global/module-level state — a host
application constructs and passes these explicitly."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryConfig:
    """Top-level config a host application builds and passes to AgentMemory.

    Placeholder — fields will be added as each backend/provider is wired in
    (Postgres DSN, Redis URL, Bedrock region/model ids, retrieval budgets).
    """

    pass
