# High-Performance AI Gateway (Semantic Cache + Token Rate Limiter)

An asynchronous LLM proxy built with **FastAPI**, **Qdrant**, **Redis (Lua)**, and **Sentence-Transformers**. Designed to minimize LLM API costs and latency via vector-based semantic caching, Server-Sent Events (SSE) token streaming, atomic token-budget rate limiting, and circuit-breaker fallbacks.

---

## System Architecture

```mermaid
flowchart TD
    Client([Client Request\nPOST /v1/chat/completions]) --> RL{Redis + Lua\nToken Rate Limiter}
    
    RL -- "Budget Exceeded" --> HTTP429([HTTP 429\nToo Many Requests - 5.2ms])
    RL -- "Tokens Reserved" --> Embed[Sentence-Transformers\nall-MiniLM-L6-v2\n384-dim Normalized Vector]
    
    Embed --> Qdrant[(Qdrant Vector DB\nCosine Similarity Search)]
    
    Qdrant -- "Similarity >= 0.90\n(CACHE HIT)" --> SSEHit[Async SSE Stream\nX-Cache: HIT]
    Qdrant -- "Similarity < 0.90\n(CACHE MISS)" --> Upstream[Upstream LLM Proxy\nGroq / OpenAI Async Stream]
    
    Upstream -- "Timeout / 5xx Error" --> CB[Circuit Breaker\nGraceful Degraded Stream]
    Upstream -- "Stream Complete" --> Upsert[Deduct Completion Tokens\n& Upsert Vector to Qdrant]
    
    SSEHit --> ClientOut([Real-Time SSE Client Stream])
    Upstream --> ClientOut
    CB --> ClientOut
```

---

## Key Engineering Features

1. **Vector Semantic Caching (`Qdrant` + `all-MiniLM-L6-v2`):** Converts incoming prompts into normalized 384-dimensional embeddings and queries Qdrant using Cosine Distance. Paraphrased prompts matching $\ge 0.90$ similarity are served directly from cache without calling upstream LLMs.
2. **Real-Time Async SSE Streaming:** Streams responses token-by-token over `text/event-stream` (`StreamingResponse`) with custom diagnostic headers (`X-Cache`, `X-Similarity-Score`, `X-Lookup-Latency-Ms`, `X-Tokens-Remaining`).
3. **Atomic Token-Aware Rate Limiting (`Redis` + `Lua`):** Executes a server-side Lua script via `EVALSHA` to atomically enforce per-organization (`org_id`) prompt and completion token budgets over a 60-second sliding window, eliminating race conditions under concurrent load.
4. **Circuit Breaker Resiliency:** Wraps upstream LLM completions (`AsyncOpenAI` / Groq) with strict async timeouts and automatic fallback streaming if the upstream provider experiences downtime or rate-limit errors.

---

## Benchmark Results

Measured locally using `benchmark.py` (`httpx.AsyncClient`, 50 requests, concurrency = 4, mixed exact and paraphrased queries):

| Metric | Result | Target |
| :--- | :--- | :--- |
| **Single-Request Cache Hit TTFT** | **52.8 ms** | `< 150 ms` |
| **Concurrent Cache Hit TTFT (p50)** | **119.1 ms** | `< 150 ms` |
| **Concurrent Cache Hit TTFT (p95)** | **166.2 ms** | `< 200 ms` |
| **Throughput (Concurrency = 4)** | **31.2 req/s** | `> 20 req/s` |
| **Paraphrase Cache Hit Rate** | **100.0%** | `> 90%` |
| **Redis + Lua Rate Limit Rejection** | **5.2 ms** | `< 10 ms` |

---

## Quick Start

### 1. Start Qdrant & Redis Containers
```bash
docker compose up -d
```

### 2. Install Dependencies & Run Gateway
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. Run the Benchmark Suite
```bash
python benchmark.py
```