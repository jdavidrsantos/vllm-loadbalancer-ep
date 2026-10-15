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


def sample(base, token):
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
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Di hola en una frase."}],
        "max_tokens": 80,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with request(base + "/v1/chat/completions", token, payload) as res:
        assigned_ms = round((time.monotonic() - t0) * 1000)
        worker_id = res.headers.get("X-Runpod-Worker-Id", "")
        first_token_ms = None
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
            if content:
                first_token_ms = first_token_ms or round((time.monotonic() - t0) * 1000)
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
        "worker_id": worker_id,
        "sample": output[:80],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--samples", type=int, default=5)
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
            data = sample(base, token)
            data["iteration"] = n + 1
            data["cold_candidate"] = bool(args.cold_wait and n == 0)
            results.append(data)
            print(json.dumps(data, ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"iteration": n + 1, "error": type(exc).__name__, "detail": str(exc)[:180]}))
    for field in ("health_ms", "assignment_ms", "first_token_ms", "total_ms"):
        values = [r[field] for r in results if isinstance(r.get(field), int)]
        if values:
            print(field, "mean=", round(statistics.mean(values)),
                  "p90=", sorted(values)[min(len(values)-1, int(len(values)*.9))])
    print("Note: /ping and inference may reach different workers. Compare worker IDs and Runpod logs.")


if __name__ == "__main__":
    main()
