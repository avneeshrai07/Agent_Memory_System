import asyncio

_task_queue: asyncio.Queue = asyncio.Queue()


async def background_worker():
    print("🟢 Background worker started")
    while True:
        coro = await _task_queue.get()
        try:
            print("⚙️ Running background task")
            await coro
        except Exception as e:
            print("❌ Background task failed:", e)


def submit_background_task(coro):
    _task_queue.put_nowait(coro)
