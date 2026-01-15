import asyncio
import traceback
from typing import Callable, Awaitable, Optional

_task_queue: Optional[asyncio.Queue] = None
_worker_started = False


async def background_worker():
    assert _task_queue is not None

    print(
        "🟢 Background worker started",
        "loop =", id(asyncio.get_running_loop()),
        "queue =", id(_task_queue),
        flush=True,
    )

    while True:
        task_factory = await _task_queue.get()

        try:
            coro = task_factory()
            if not asyncio.iscoroutine(coro):
                raise TypeError("Background task factory did not return coroutine")

            print("⚙️ Running background task", flush=True)
            await coro

        except Exception as e:
            print("❌ Background task failed:", e, flush=True)
            traceback.print_exc()

        finally:
            _task_queue.task_done()


async def start_background_worker():
    global _task_queue, _worker_started

    if _worker_started:
        return

    _task_queue = asyncio.Queue()
    asyncio.create_task(background_worker())
    _worker_started = True


def submit_background_task(task_factory: Callable[[], Awaitable[None]]) -> None:
    if _task_queue is None:
        raise RuntimeError("Background worker not initialized")

    print(
        "[BG SUBMIT]",
        "loop =", id(asyncio.get_running_loop()),
        "queue =", id(_task_queue),
        flush=True,
    )

    _task_queue.put_nowait(task_factory)
