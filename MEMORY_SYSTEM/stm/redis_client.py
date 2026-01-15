# redis_client.py
import redis.asyncio as redis
import logging

logger = logging.getLogger(__name__)

class AsyncRedisClient:
    def __init__(self):
        self.redis_url = "rediss://default:AXjtAAIncDEzMDY5NWRkMjAxYzQ0ZGU5YmJlY2Q5Yzc3ZjM5YzE5M3AxMzA5NTc@polished-frog-30957.upstash.io:6379"
        self.client: redis.Redis | None = None

    async def connect(self) -> redis.Redis:
        self.client = redis.from_url(
            self.redis_url,
            decode_responses=True
        )

        info = await self.client.info()

        print(f"Connected to Redis")
        return self.client

    async def close(self):
        if self.client:
            await self.client.close()
            self.client = None
