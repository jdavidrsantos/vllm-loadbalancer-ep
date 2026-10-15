"""CPU-only tests: no real vLLM/GPU, model download, or Runpod key required."""
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import handler as gateway


@pytest.fixture(autouse=True)
def reset_runtime():
    gateway.runtime.phase = "initializing"
    gateway.runtime.process = None
    gateway.runtime.active = 0
    gateway.runtime.completed = 0
    gateway.runtime.failed = 0
    gateway.runtime.ready_at = None
    yield


@pytest.mark.asyncio
async def test_health_204_until_engine_ready():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://test") as client:
        response = await client.get("/ping")
        assert response.status_code == 204
        assert response.content == b""
        gateway.runtime.phase = "failed"
        assert (await client.get("/ping")).status_code == 503
        gateway.runtime.phase = "ready"
        gateway.runtime.process = SimpleNamespace(returncode=None)
        assert (await client.get("/ping")).status_code == 200


@pytest.mark.asyncio
async def test_chat_non_stream_injects_model_and_returns_usage():
    def reply(request):
        data = __import__("json").loads(request.content)
        assert data["model"] == gateway.MODEL_ID
        assert request.headers.get("authorization") is None
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hola"}}], "usage": {"total_tokens": 4}})

    gateway.runtime.phase = "ready"
    gateway.runtime.process = SimpleNamespace(returncode=None)
    gateway.runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://test") as client:
            response = await client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "Hola"}]},
                                         headers={"Authorization": "Bearer sensitive"})
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == "Hola"
            assert response.headers["x-cr-worker-processing-ms"].isdigit()
            assert response.headers["x-cr-request-id"]
            assert gateway.runtime.active == 0
            assert gateway.runtime.completed == 1
    finally:
        await gateway.runtime.client.aclose()


@pytest.mark.asyncio
async def test_native_openai_sse_remains_native_by_default():
    original = (b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"Hola"}}]}\n\n'
                b'data: [DONE]\n\n')
    gateway.runtime.phase = "ready"
    gateway.runtime.process = SimpleNamespace(returncode=None)
    gateway.runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(
        200, content=original, headers={"content-type": "text/event-stream"})))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://test") as client:
            response = await client.post("/v1/chat/completions", json={"messages": [], "stream": True})
            assert response.status_code == 200
            assert response.content == original
            assert gateway.runtime.completed == 1
            assert gateway.runtime.active == 0
    finally:
        await gateway.runtime.client.aclose()


@pytest.mark.asyncio
async def test_status_events_only_when_opted_in():
    chunks = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
    gateway.runtime.phase = "ready"
    gateway.runtime.process = SimpleNamespace(returncode=None)
    gateway.runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=chunks)))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://test") as client:
            response = await client.post("/v1/completions", json={"prompt": "test", "stream": True},
                                         headers={"X-CR-Status-Events": "1"})
            assert response.status_code == 200
            assert b"event: cr.status" in response.content
            assert b'data: [DONE]\n\n' in response.content
    finally:
        await gateway.runtime.client.aclose()


@pytest.mark.asyncio
async def test_unavailable_and_validation_fail_without_leaking_secrets():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://test") as client:
        assert (await client.post("/v1/chat/completions", json={"messages": []})).status_code == 503
        gateway.runtime.phase = "ready"
        gateway.runtime.process = SimpleNamespace(returncode=None)
        assert (await client.post("/v1/chat/completions", json={"messages": "bad"})).status_code == 422
        assert (await client.post("/v1/chat/completions", json={"messages": [], "model": "fake"})).status_code == 400


def test_stream_probe_distinguishes_role_from_content_token():
    probe = gateway.StreamProbe(__import__("time").monotonic())
    assert not probe.observe(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
    assert probe.first_token_ms is None
    assert not probe.observe(b'data: {"choices":[{"delta":{"content":"Bon')
    assert probe.observe(b'jour"}}]}\n\n')
    assert probe.first_token_ms is not None
