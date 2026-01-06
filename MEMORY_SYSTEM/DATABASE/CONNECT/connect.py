import asyncio
import asyncpg
import os
import random
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
ENVIRONMENT = os.getenv("environment")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# from logger.logger_config import get_newsFetcher_#logger
#logger = get_newsFetcher_#logger()


class DatabaseManager:
    def __init__(self):
        self._db_pool: Optional[asyncpg.pool.Pool] = None
        self._pool_create_lock = asyncio.Lock()

    async def _ensure_env(self):
        if ENVIRONMENT == "local_environment":
            required_env_vars = ["DATABASE_URL"]
        else:
            required_env_vars = [
                "DB_HOST",
                "DB_USER",
                "DB_PASSWORD",
                "DB_NAME",
            ]

        for var in required_env_vars:
            if not os.getenv(var):
                #logger.error("%s is not set in env", var)
                raise RuntimeError(f"{var} is not set in env")

    async def _is_pool_alive(self, pool: Optional[asyncpg.pool.Pool]) -> bool:
        if pool is None:
            return False
        try:
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1;")
            return True
        except Exception as e:
            #logger.warning("Pool health check failed: %s", e)
            return False

    async def get_pool(self) -> asyncpg.pool.Pool:
        await self._ensure_env()

        if self._db_pool and await self._is_pool_alive(self._db_pool):
            #logger.info("Reusing existing DB pool")
            return self._db_pool

        async with self._pool_create_lock:
            if self._db_pool and await self._is_pool_alive(self._db_pool):
                #logger.info("Reusing existing DB pool (post-lock)")
                return self._db_pool

            try:
                #logger.info("Creating new DB pool…")

                pool_kwargs = {
                    "min_size": 2,
                    "max_size": 20,
                    "statement_cache_size": 1000,
                    "timeout": 10.0,
                }

                if ENVIRONMENT == "local_environment":
                    pool_kwargs["dsn"] = DATABASE_URL
                else:
                    pool_kwargs.update({
                        "host": DB_HOST,
                        "port": DB_PORT,
                        "user": DB_USER,
                        "password": DB_PASSWORD,
                        "database": DB_NAME,
                    })

                self._db_pool = await asyncpg.create_pool(**pool_kwargs)

                #logger.info("New DB pool established")
                return self._db_pool

            except Exception as e:
                #logger.error(
                #     "DB pool creation failed (%s): %r",
                #     type(e).__name__,
                #     e,
                #     exc_info=True
                # )
                self._db_pool = None
                raise

    async def wait_for_connection_pool_pool(
        self,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 30.0,
    ) -> asyncpg.pool.Pool:

        delay = initial_delay

        for attempt in range(1, max_retries + 1):
            #logger.info("Attempt %d/%d to create DB pool", attempt, max_retries)

            try:
                pool = await self.get_pool()
                if await self._is_pool_alive(pool):
                    #logger.info("Database connection POOL established.")
                    return pool
            except Exception:
                pass

            sleep_base = min(delay, max_delay)
            jitter = random.uniform(0, sleep_base * 0.1)
            total_sleep = sleep_base + jitter
            #logger.info("Retrying in %.2f seconds…", total_sleep)

            await asyncio.sleep(total_sleep)
            delay *= backoff_factor

        raise ConnectionError("DB pool creation failed after retries")

    async def close_pool(self) -> None:
        async with self._pool_create_lock:
            if self._db_pool:
                #logger.info("Closing DB pool...")
                await self._db_pool.close()
                self._db_pool = None
                #logger.info("DB pool closed successfully.")


db_manager = DatabaseManager()
