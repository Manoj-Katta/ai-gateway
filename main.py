import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "llm_semantic_cache"
SIMILARITY_THRESHOLD = 0.92  # Milestone 2 threshold

# Global clients initialized at startup
embedder: SentenceTransformer = None
qdrant: AsyncQdrantClient = None
llm_client: AsyncOpenAI = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, qdrant, llm_client
    # 1. Load embedding model into memory once (384-dim, fast inference)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 2. Connect to Qdrant & ensure collection exists with Cosine distance
    qdrant = AsyncQdrantClient(host="localhost", port=6333)
    collections = await qdrant.get_collections()
    if not any(c.name == COLLECTION_NAME for c in collections.collections):
        await qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    # 3. Optional Groq / OpenAI client (uses Groq's ultra-fast API if GROQ_API_KEY is set)
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = "https://api.groq.com/openai/v1" if os.getenv("GROQ_API_KEY") else None
    if api_key:
        llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    yield
    await qdrant.close()


app = FastAPI(title="AI Gateway", lifespan=lifespan)


class EmbedRequest(BaseModel):
    text: str


class ChatRequest(BaseModel):
    org_id: str = "org_default"
    prompt: str
    model: str = "llama-3.1-8b-instant"


# --- MILESTONE 1: /embed Endpoint ---
@app.post("/embed")
async def create_embedding(req: EmbedRequest):
    t0 = time.perf_counter()
    vector = await asyncio.to_thread(embedder.encode, req.text, normalize_embeddings=True)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "dimensions": len(vector),
        "latency_ms": latency_ms,
        "embedding_preview": vector[:5].tolist(),
    }


# --- MILESTONE 2 & 3: Semantic Cache + Async SSE Token Streaming ---
async def stream_cached_sse(cached_text: str, score: float, latency_ms: float) -> AsyncGenerator[str, None]:
    yield f"data: [CACHE_HIT score={score:.4f} lookup_ms={latency_ms}ms]\n\n"
    for token in cached_text.split(" "):
        yield f"data: {token} \n\n"
        await asyncio.sleep(0.005)
    yield "data: [DONE]\n\n"


async def stream_upstream_and_cache(
    prompt: str, vector: list[float], model: str
) -> AsyncGenerator[str, None]:
    collected_chunks: list[str] = []

    if llm_client:
        stream = await llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                collected_chunks.append(delta)
                yield f"data: {delta}\n\n"
    else:
        # Built-in fallback stream so you can test <150ms TTFB immediately without an API key
        simulated_tokens = (
            f"Semantic cache miss for '{prompt}'. Streaming tokens asynchronously over SSE "
            "and persisting the embedding vector to Qdrant in the background."
        ).split(" ")
        for token in simulated_tokens:
            collected_chunks.append(token + " ")
            yield f"data: {token} \n\n"
            await asyncio.sleep(0.02)

    yield "data: [DONE]\n\n"

    # Non-blocking write to Qdrant after stream completes
    full_response = "".join(collected_chunks).strip()
    if full_response:
        await qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"prompt": prompt, "response": full_response, "ts": time.time()},
                )
            ],
        )


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    t0 = time.perf_counter()

    # 1. Compute normalized embedding in worker thread (~10ms)
    vector_np = await asyncio.to_thread(embedder.encode, req.prompt, normalize_embeddings=True)
    vector = vector_np.tolist()

    # 2. Query Qdrant for nearest neighbor
    search_result = await qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=1,
    )
    lookup_ms = round((time.perf_counter() - t0) * 1000, 2)

    # 3. Cache HIT if cosine similarity > 0.92
    if search_result.points and search_result.points[0].score >= SIMILARITY_THRESHOLD:
        top_hit = search_result.points[0]
        return StreamingResponse(
            stream_cached_sse(top_hit.payload["response"], top_hit.score, lookup_ms),
            media_type="text/event-stream",
            headers={
                "X-Cache": "HIT",
                "X-Similarity-Score": f"{top_hit.score:.4f}",
                "X-Lookup-Latency-Ms": str(lookup_ms),
            },
        )

    # 4. Cache MISS -> Stream from upstream LLM & upsert to Qdrant
    return StreamingResponse(
        stream_upstream_and_cache(req.prompt, vector, req.model),
        media_type="text/event-stream",
        headers={
            "X-Cache": "MISS",
            "X-Lookup-Latency-Ms": str(lookup_ms),
        },
    )