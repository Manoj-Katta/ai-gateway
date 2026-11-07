import asyncio
import statistics
import time
import httpx

URL = "http://127.0.0.1:8000/v1/chat/completions"

SEED_PROMPTS = [
    "How do I scale Kubernetes pods horizontally?",
    "Explain the CAP theorem in distributed systems.",
    "How does Redis Lua guarantee atomicity?",
]

TEST_PROMPTS = [
    "How do I scale Kubernetes pods horizontally?",
    "How can I horizontally scale pods in Kubernetes?",      # Paraphrase hit (~0.916 similarity)
    "Explain the CAP theorem in distributed systems.",
    "Explain the CAP theorem in a distributed system.",      # Paraphrase hit (~0.965 similarity)
    "How does Redis Lua guarantee atomicity?",
]

sem = asyncio.Semaphore(4)


async def send_request(
    client: httpx.AsyncClient, prompt: str, org_id: str = "bench_org", budget: int = 50000
):
    async with sem:
        t0 = time.perf_counter()
        async with client.stream(
            "POST",
            URL,
            json={
                "org_id": org_id,
                "session_id": "s1",
                "prompt": prompt,
                "token_budget": budget,
            },
        ) as resp:
            if resp.status_code == 429:
                return ("RATE_LIMITED", (time.perf_counter() - t0) * 1000, 0.0, 0.0)

            first_byte_ms = None
            async for _ in resp.aiter_lines():
                if first_byte_ms is None:
                    first_byte_ms = (time.perf_counter() - t0) * 1000
            return (
                resp.headers.get("x-cache", "MISS"),
                first_byte_ms or 0.0,
                float(resp.headers.get("x-lookup-latency-ms", 0.0)),
                float(resp.headers.get("x-similarity-score", 0.0)),
            )


async def main():
    async with httpx.AsyncClient(timeout=15.0) as client:
        print("1. Warming up cache with 3 seed prompts...")
        for i, p in enumerate(SEED_PROMPTS, 1):
            status, ttft, lkp, score = await send_request(client, p)
            print(
                f"   Warmup {i}/3 -> {status} (TTFT: {ttft:.1f}ms, Lookup: {lkp:.1f}ms, Score: {score:.4f})"
            )

        print("\n2. Running 50 benchmark requests (concurrency=4)...")
        t_start = time.perf_counter()
        tasks = [
            send_request(client, TEST_PROMPTS[i % len(TEST_PROMPTS)])
            for i in range(50)
        ]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - t_start

        hits_ttft = [ttft for status, ttft, _, _ in results if status == "HIT"]
        hits_lookup = [lkp for status, _, lkp, _ in results if status == "HIT" and lkp > 0]
        hit_rate = (len(hits_ttft) / len(results)) * 100

        p50_ttft = statistics.median(hits_ttft) if hits_ttft else 0
        p95_ttft = sorted(hits_ttft)[int(len(hits_ttft) * 0.95)] if hits_ttft else 0
        p50_lkp = statistics.median(hits_lookup) if hits_lookup else p50_ttft

        print(
            f"   Throughput: {len(results)/total_time:.1f} req/s | "
            f"Hit Rate: {hit_rate:.1f}% | "
            f"Lookup p50: {p50_lkp:.1f}ms | "
            f"Hit TTFT p50: {p50_ttft:.1f}ms | "
            f"p95: {p95_ttft:.1f}ms"
        )

        print("\n3. Verifying Milestone 4 Redis + Lua Token Rate Limiter (budget=15 tokens)...")
        rl_org = f"strict_org_{int(time.time())}"
        for attempt in range(1, 4):
            status, ttft, _, _ = await send_request(
                client,
                "How do I scale Kubernetes pods horizontally?",
                org_id=rl_org,
                budget=15,
            )
            print(f"   Attempt {attempt} -> {status} ({ttft:.1f}ms)")


if __name__ == "__main__":
    asyncio.run(main())