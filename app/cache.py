import asyncio
import uuid
import time
from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from app.config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
)

embedder: SentenceTransformer = None
embed_lock: asyncio.Lock = None
qdrant: AsyncQdrantClient = None


async def init_cache():
    global embedder, embed_lock, qdrant
    embed_lock = asyncio.Lock()
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    qdrant = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    collections = await qdrant.get_collections()
    if not any(c.name == COLLECTION_NAME for c in collections.collections):
        await qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


async def close_cache():
    global qdrant
    if qdrant:
        await qdrant.close()


async def encode_prompt(text: str) -> list[float]:
    """Serialize PyTorch inference to prevent CPU thread contention."""
    async with embed_lock:
        vec = await asyncio.to_thread(embedder.encode, text, normalize_embeddings=True)
        return vec.tolist()


async def query_semantic_cache(vector: list[float]):
    return await qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=1,
    )


async def upsert_cache_record(prompt: str, response: str, vector: list[float]):
    await qdrant.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"prompt": prompt, "response": response, "ts": time.time()},
            )
        ],
    )