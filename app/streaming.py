import asyncio
from typing import AsyncGenerator
from openai import AsyncOpenAI
from app.config import UPSTREAM_API_KEY, UPSTREAM_BASE_URL, estimate_tokens
from app.cache import upsert_cache_record
from app.rate_limiter import deduct_completion_tokens

llm_client: AsyncOpenAI = None


def init_upstream_client():
    global llm_client
    if UPSTREAM_API_KEY:
        llm_client = AsyncOpenAI(api_key=UPSTREAM_API_KEY, base_url=UPSTREAM_BASE_URL)


async def stream_cached_sse(
    cached_text: str, score: float, lookup_ms: float
) -> AsyncGenerator[str, None]:
    yield f"data: [CACHE_HIT score={score:.4f} lookup_ms={lookup_ms}ms]\n\n"
    for token in cached_text.split(" "):
        yield f"data: {token} \n\n"
    yield "data: [DONE]\n\n"


async def stream_upstream_and_cache(
    org_id: str, prompt: str, vector: list[float], model: str
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
            fallback = f"[CIRCUIT_BREAKER_FALLBACK: Upstream unavailable ({type(exc).__name__}). Serving degraded response for '{prompt}'.]"
            collected_chunks.append(fallback)
            yield f"data: {fallback}\n\n"
    else:
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
        completion_tokens = estimate_tokens(full_response)
        await deduct_completion_tokens(org_id, completion_tokens)
        await upsert_cache_record(prompt, full_response, vector)

    yield "data: [DONE]\n\n"
