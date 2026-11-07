import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "llm_semantic_cache"
SIMILARITY_THRESHOLD = 0.90  # Tuned so natural paraphrases hit reliably (>0.90)
ORG_TOKEN_BUDGET_PER_MIN = 5000  # Milestone 4: Per-org token limit per 60s window

# Milestone 4: Atomic Redis Lua Script for Token-Aware Rate Limiting
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

# Global clients initialized at startup
embedder: SentenceTransformer = None
embed_lock: asyncio.Lock = None
qdrant: AsyncQdrantClient = None
redis_client: aioredis.Redis = None
lua_sha: str = None
llm_client: AsyncOpenAI = None


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, embed_lock, qdrant, redis_client, lua_sha, llm_client

    embed_lock = asyncio.Lock()

    # 1. Load embedding model into memory once (384-dim)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # 2. Connect to Qdrant & ensure collection exists with Cosine distance
    qdrant = AsyncQdrantClient(host="localhost", port=6333)
    collections = await qdrant.get_collections()
    if not any(c.name == COLLECTION_NAME for c in collections.collections):
        await qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    # 3. Connect to Redis & pre-load Lua rate limiter script
    redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
    try:
        lua_sha = await redis_client.script_load(TOKEN_BUCKET_LUA)
    except Exception:
        lua_sha = None

    # 4. Optional Groq / OpenAI client
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = "https://api.groq.com/openai/v1" if os.getenv("GROQ_API_KEY") else None
    if api_key:
        llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    yield
    await qdrant.close()
    if redis_client:
        await redis_client.aclose()


app = FastAPI(title="AI Gateway", lifespan=lifespan)


class EmbedRequest(BaseModel):
    text: str


class ChatRequest(BaseModel):
    org_id: str = "org_default"
    session_id: str = "s1"
    prompt: str
    model: str = "llama-3.1-8b-instant"
    token_budget: int = ORG_TOKEN_BUDGET_PER_MIN


async def encode_prompt(text: str) -> list[float]:
    # Serialize CPU PyTorch calls to avoid GIL/OpenMP thread thrashing under concurrency
    async with embed_lock:
        vec = await asyncio.to_thread(embedder.encode, text, normalize_embeddings=True)
        return vec.tolist()


async def enforce_token_rate_limit(org_id: str, tokens: int, budget: int) -> int:
    """Milestone 4: Execute Redis Lua script atomically; fallback gracefully if Redis is down."""
    if not redis_client or not lua_sha:
        return budget
    key = f"rate_limit:tokens:{org_id}"
    try:
        remaining = await redis_client.evalsha(lua_sha, 1, key, tokens, budget, 60)
    except Exception:
        return budget  # Circuit breaker open if Redis is unreachable

    if int(remaining) < 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Token budget ({budget} tokens/min) exceeded for org '{org_id}'.",
        )
    return int(remaining)


# --- MILESTONE 1: /embed Endpoint ---
@app.post("/embed")
async def create_embedding(req: EmbedRequest):
    t0 = time.perf_counter()
    vector = await encode_prompt(req.text)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "dimensions": len(vector),
        "latency_ms": latency_ms,
        "embedding_preview": vector[:5],
    }


# --- MILESTONE 2, 3 & 4: Semantic Cache + SSE Streaming + Circuit Breaker ---
async def stream_cached_sse(
    cached_text: str, score: float, lookup_ms: float
) -> AsyncGenerator[str, None]:
    yield f"data: [CACHE_HIT score={score:.4f} lookup_ms={lookup_ms}ms]\n\n"
    for token in cached_text.split(" "):
        yield f"data: {token} \n\n"
    yield "data: [DONE]\n\n"


async def stream_upstream_and_cache(
    org_id: str, prompt: str, vector: list[float], model: str, budget: int
) -> AsyncGenerator[str, None]:
    collected_chunks: list[str] = []

    if llm_client:
        try:
            stream = await asyncio.wait_for(
                llm_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                ),
                timeout=8.0,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    collected_chunks.append(delta)
                    yield f"data: {delta}\n\n"
        except Exception as exc:
            # Milestone 4: Circuit Breaker Fallback if Upstream LLM times out or 5xx fails
            fallback = f"[CIRCUIT_BREAKER_FALLBACK: Upstream unavailable ({type(exc).__name__}). Serving degraded response for '{prompt}'.]"
            collected_chunks.append(fallback)
            yield f"data: {fallback}\n\n"
    else:
        # Built-in fast stream when running without external API key
        simulated_tokens = (
            f"Semantic cache miss for '{prompt}'. Streaming tokens asynchronously over SSE "
            "and persisting the embedding vector to Qdrant."
        ).split(" ")
        for token in simulated_tokens:
            collected_chunks.append(token + " ")
            yield f"data: {token} \n\n"
            await asyncio.sleep(0.005)

    full_response = "".join(collected_chunks).strip()
    if full_response:
        # Deduct completion tokens in Redis and persist vector to Qdrant before closing stream
        completion_tokens = estimate_tokens(full_response)
        if redis_client:
            try:
                await redis_client.incrby(f"rate_limit:tokens:{org_id}", completion_tokens)
            except Exception:
                pass

        await qdrant.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"prompt": prompt, "response": full_response, "ts": time.time()},
                )
            ],
        )

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    t0 = time.perf_counter()

    # 1. Milestone 4: Check & deduct prompt tokens via Redis Lua script
    prompt_tokens = estimate_tokens(req.prompt)
    remaining_tokens = await enforce_token_rate_limit(
        req.org_id, prompt_tokens, req.token_budget
    )

    # 2. Compute normalized embedding (~8-12ms)
    vector = await encode_prompt(req.prompt)

    # 3. Query Qdrant for nearest neighbor
    search_result = await qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=1,
    )
    lookup_ms = round((time.perf_counter() - t0) * 1000, 2)
    top_score = search_result.points[0].score if search_result.points else 0.0

    # 4. Cache HIT
    if search_result.points and top_score >= SIMILARITY_THRESHOLD:
        top_hit = search_result.points[0]
        return StreamingResponse(
            stream_cached_sse(top_hit.payload["response"], top_score, lookup_ms),
            media_type="text/event-stream",
            headers={
                "X-Cache": "HIT",
                "X-Similarity-Score": f"{top_score:.4f}",
                "X-Lookup-Latency-Ms": str(lookup_ms),
                "X-Tokens-Remaining": str(remaining_tokens),
            },
        )

    # 5. Cache MISS -> Stream from upstream LLM & upsert to Qdrant
    return StreamingResponse(
        stream_upstream_and_cache(req.org_id, req.prompt, vector, req.model, req.token_budget),
        media_type="text/event-stream",
        headers={
            "X-Cache": "MISS",
            "X-Similarity-Score": f"{top_score:.4f}",
            "X-Lookup-Latency-Ms": str(lookup_ms),
            "X-Tokens-Remaining": str(remaining_tokens),
        },
    )