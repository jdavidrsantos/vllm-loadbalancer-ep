#!/usr/bin/env python3
"""Benchmark Runpod Load Balancing worker from a machine with network access.

Usage:
  RUNPOD_API_KEY=... python3 scripts/benchmark_lb.py --endpoint 3cskuf3yc4n5vu --samples 5
  RUNPOD_API_KEY=... python3 scripts/benchmark_lb.py --endpoint 3cskuf3yc4n5vu --cold-wait

Measures separate health/ping, HTTP assignment, first token and total inference.
Does NOT claim to measure the cold-start of a worker it could not reach.
"""
import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request


def request(url, token, payload=None, timeout=150):
    headers = {"Authorization": "Bearer " + token}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    data = json.dumps(payload).encode() if payload is not None else None
    return urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=headers), timeout=timeout
    )


def sample(base, token, context_tokens=0, output_tokens=80):
    t0 = time.monotonic()
    health_code = None
    try:
        with request(base + "/ping", token, timeout=15) as res:
            health_code = res.status
    except urllib.error.HTTPError as exc:
        health_code = exc.code
    except (OSError, TimeoutError):
        pass
    health_ms = round((time.monotonic() - t0) * 1000)
    t0 = time.monotonic()
    # Approximate a real support prompt without sending production user data.
    # Fixed text makes comparisons across eager/CUDA Graph configurations repeatable.
    filler = ("Información sobre funciones del editor y publicación de sitios. " * 40)
    context = (filler * max(1, context_tokens // 450 + 1))[:context_tokens * 4] if context_tokens else ""
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "system", "content": context}, {"role": "user", "content": "Explica brevemente cómo crear y publicar un sitio web en CreaRápido."}],
        "max_tokens": output_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with request(base + "/v1/chat/completions", token, payload) as res:
        assigned_ms = round((time.monotonic() - t0) * 1000)
        worker_id = res.headers.get("X-Runpod-Worker-Id", "")
        first_token_ms = None
        last_content_at = None
        max_gap_ms = 0
        output_tokens_actual = None
        output = ""
        done = False
        for raw in res:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                done = True
                break
            obj = json.loads(data)
            if "error" in obj:
                raise ValueError("Model reported a stream error")
            content = (obj.get("choices") or [{}])[0].get("delta", {}).get("content", "")
            if isinstance(obj.get("usage"), dict):
                output_tokens_actual = obj["usage"].get("completion_tokens")
            if content:
                now_ms = round((time.monotonic() - t0) * 1000)
                if last_content_at is not None:
                    max_gap_ms = max(max_gap_ms, now_ms - last_content_at)
                last_content_at = now_ms
                if first_token_ms is None:
                    first_token_ms = now_ms
                output += content
        total_ms = round((time.monotonic() - t0) * 1000)
        if not done or not output.strip():
            raise ValueError("Incomplete model stream")
    return {
        "health_http": health_code,
        "health_ms": health_ms,
        "assignment_ms": assigned_ms,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
        "max_content_gap_ms": max_gap_ms,
        "output_tokens": output_tokens_actual,
        "decode_tokens_per_sec": round(output_tokens_actual / max(0.001, (total_ms - first_token_ms) / 1000), 2) if isinstance(output_tokens_actual, int) and first_token_ms is not None else None,
        "worker_id": worker_id,
        "sample": output[:80],
    }


def sample_queue(endpoint, token):
    """Optional queue baseline: submit exactly one job, then only poll status."""
    base = "https://api.runpod.ai/v2/" + endpoint
    t0 = time.monotonic()
    with request(base + "/run", token, {
        "input": {
            "messages": [{"role": "user", "content": "Di hola en una frase."}],
            "max_tokens": 80,
            "sampling_params": {"max_tokens": 80, "temperature": 0},
        }
    }) as res:
        job_id = json.load(res)["id"]
    submitted_ms = round((time.monotonic() - t0) * 1000)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        with request(base + "/status/" + job_id, token, timeout=15) as res:
            job = json.load(res)
        state = str(job.get("status", "")).upper()
        if state == "COMPLETED":
            return {
                "submitted_ms": submitted_ms,
                "total_ms": round((time.monotonic() - t0) * 1000),
                "delay_ms": job.get("delayTime"),
                "execution_ms": job.get("executionTime"),
                "worker_id": job.get("workerId", ""),
                "state": state,
            }
        if state in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError("Queue worker ended with " + state)
        time.sleep(2)
    raise TimeoutError("Queue worker not completed within 180 seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--context-tokens", type=int, default=1900, help="Approximate support-context length; synthetic text, no user content")
    parser.add_argument("--output-tokens", type=int, default=120)
    parser.add_argument("--queue-endpoint", default="",
                        help="Optional old queue endpoint, for explicit comparison only")
    parser.add_argument("--cold-wait", action="store_true",
                        help="Label first sample as cold candidate; manually scale to zero first")
    args = parser.parse_args()
    token = os.environ.get("RUNPOD_API_KEY", "")
    if not token:
        parser.error("Set RUNPOD_API_KEY in the environment")
    base = "https://" + args.endpoint + ".api.runpod.ai"
    results = []
    for n in range(max(1, min(args.samples, 30))):
        try:
            data = sample(base, token, max(0, min(args.context_tokens, 3500)), max(1, min(args.output_tokens, 512)))
            data["iteration"] = n + 1
            data["cold_candidate"] = bool(args.cold_wait and n == 0)
            results.append(data)
            print(json.dumps(data, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"iteration": n + 1, "error": type(exc).__name__, "detail": str(exc)[:180]}))
    for field in ("health_ms", "assignment_ms", "first_token_ms", "total_ms", "max_content_gap_ms", "decode_tokens_per_sec"):
        values = [r[field] for r in results if isinstance(r.get(field), int)]
        if values:
            print(field, "mean=", round(statistics.mean(values)),
                  "p90=", sorted(values)[min(len(values)-1, int(len(values)*.9))])
    if args.queue_endpoint:
        print("Queue baseline (deliberate extra billable test request):")
        try:
            queue = sample_queue(args.queue_endpoint, token)
            print(json.dumps(queue, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"queue_error": type(exc).__name__, "detail": str(exc)[:180]}))
    print("Note: Compare multiple identical prompts; model, GPU and traffic may differ.")
    print("Note: /ping and inference may reach different workers. Compare worker IDs and Runpod logs.")


if __name__ == "__main__":
    main()
