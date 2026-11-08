import os
from pydantic import BaseModel

COLLECTION_NAME = "llm_semantic_cache"
SIMILARITY_THRESHOLD = 0.90
ORG_TOKEN_BUDGET_PER_MIN = 5000
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

UPSTREAM_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
UPSTREAM_BASE_URL = (
    "https://api.groq.com/openai/v1" if os.getenv("GROQ_API_KEY") else None
)


class EmbedRequest(BaseModel):
    text: str


class ChatRequest(BaseModel):
    org_id: str = "org_default"
    session_id: str = "s1"
    prompt: str
    model: str = "llama-3.1-8b-instant"
    token_budget: int = ORG_TOKEN_BUDGET_PER_MIN


def estimate_tokens(text: str) -> int:
    """Heuristic token estimation (~4 characters per token)."""
    return max(1, len(text) // 4)
