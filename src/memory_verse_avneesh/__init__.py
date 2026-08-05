"""memory_verse_avneesh: persistent, fast, accurate memory for conversational LLM agents.

The whole public surface for a host application:

    from memory_verse_avneesh import memory

    memory.setup(...)   # once, at startup
    memory.read(...)    # per request
    memory.write(...)   # after generation

See memory.py. Everything below it (connect, Memory, the stores, the
clients, the models) is the lower-level machinery that powers those three —
reach for it only when you need wiring `memory` doesn't cover.
"""

from memory_verse_avneesh import memory
from memory_verse_avneesh.connect import Memory, MemoryPool, connect, setup

__version__ = "0.6.0"

__all__ = ["memory", "Memory", "MemoryPool", "connect", "setup", "__version__"]
