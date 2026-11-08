import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.config import SIMILARITY_THRESHOLD, ChatRequest, EmbedRequest, estimate_tokens
from app.rate_limiter import (
    close_redis,
    enforce_token_rate_limit,
    init_redis,
)
from app.cache import (
    close_cache,
    encode_prompt,
    init_cache,
    query_semantic_cache,
)
from app.streaming import (
    init_upstream_client,
    stream_cached_sse,
    stream_upstream_and_cache,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_cache()
    await init_redis()
    init_upstream_client()
    yield
    await close_cache()
    await close_redis()


app = FastAPI(
    title="High-Performance AI Gateway",
    version="1.0.0",
    lifespan=lifespan,
)


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


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    t0 = time.perf_counter()

    # Rate Limiting
    prompt_tokens = estimate_tokens(req.prompt)
    remaining_tokens = await enforce_token_rate_limit(
        req.org_id, prompt_tokens, req.token_budget
    )

    # Vector Generation & Cache Lookup
    vector = await encode_prompt(req.prompt)
    search_result = await query_semantic_cache(vector)
    lookup_ms = round((time.perf_counter() - t0) * 1000, 2)
    top_score = search_result.points[0].score if search_result.points else 0.0

    # Cache Hit
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

    # Cache Miss -> Stream & Upsert
    return StreamingResponse(
        stream_upstream_and_cache(req.org_id, req.prompt, vector, req.model),
        media_type="text/event-stream",
        headers={
            "X-Cache": "MISS",
            "X-Similarity-Score": f"{top_score:.4f}",
            "X-Lookup-Latency-Ms": str(lookup_ms),
            "X-Tokens-Remaining": str(remaining_tokens),
        },
    )