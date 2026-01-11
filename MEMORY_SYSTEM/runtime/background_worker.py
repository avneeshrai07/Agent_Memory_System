# MEMORY_SYSTEM/runtime/background_worker.py

import asyncio
import traceback
from typing import Coroutine, Any

_task_queue: asyncio.Queue[Coroutine[Any, Any, None]] = asyncio.Queue()


async def background_worker():
    print("🟢 Background worker started", flush=True)

    while True:
        coro = await _task_queue.get()
        try:
            if not asyncio.iscoroutine(coro):
                raise TypeError(f"Invalid task enqueued: {coro}")

            print("⚙️ Running background task", flush=True)
            await coro

        except Exception as e:
            print("❌ Background task failed:", e, flush=True)
            traceback.print_exc()

        finally:
            _task_queue.task_done()


def submit_background_task(coro: Coroutine[Any, Any, None]) -> None:
    if not asyncio.iscoroutine(coro):
        raise TypeError(
            f"submit_background_task expects coroutine, got {type(coro)}"
        )
    _task_queue.put_nowait(coro)
