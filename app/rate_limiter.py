from fastapi import HTTPException, status
import redis.asyncio as aioredis
from app.config import REDIS_URL

TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local requested = tonumber(ARGV[1])
local budget = tonumber(ARGV[2])
local window = tonumber(ARGV[3])

local current = tonumber(redis.call("GET", key) or "0")
if current + requested > budget then
    return -1
end

local updated = redis.call("INCRBY", key, requested)
if updated == requested then
    redis.call("EXPIRE", key, window)
end
return budget - updated
"""

redis_client: aioredis.Redis = None
lua_sha: str = None


async def init_redis():
    global redis_client, lua_sha
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        lua_sha = await redis_client.script_load(TOKEN_BUCKET_LUA)
    except Exception:
        lua_sha = None


async def close_redis():
    global redis_client
    if redis_client:
        await redis_client.aclose()


async def enforce_token_rate_limit(org_id: str, tokens: int, budget: int) -> int:
    """Execute Redis Lua script atomically; opens circuit breaker if Redis is unreachable."""
    if not redis_client or not lua_sha:
        return budget

    key = f"rate_limit:tokens:{org_id}"
    try:
        remaining = await redis_client.evalsha(lua_sha, 1, key, tokens, budget, 60)
    except Exception:
        return budget

    if int(remaining) < 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Token budget ({budget} tokens/min) exceeded for org '{org_id}'.",
        )
    return int(remaining)


async def deduct_completion_tokens(org_id: str, tokens: int):
    """Increment token usage following upstream stream completion."""
    if redis_client:
        try:
            await redis_client.incrby(f"rate_limit:tokens:{org_id}", tokens)
        except Exception:
            pass