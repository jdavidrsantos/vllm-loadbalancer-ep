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


@pytest.mark.asyncio
async def test_status_requires_model_loaded_and_available_slots():
    gateway.runtime.phase = "ready"
    gateway.runtime.process = SimpleNamespace(returncode=None)
    gateway.runtime.instance_id = "boot-a"
    gateway.runtime.warmup_completed = True
    gateway.runtime.active = gateway.MAX_IN_FLIGHT
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://test") as client:
        result = (await client.get("/status")).json()
        assert result["model_loaded"] is True
        assert result["ready_for_request"] is False
        assert result["available_slots"] == 0
        gateway.runtime.active = 0
        result = (await client.get("/status")).json()
        assert result["ready_for_request"] is True
        assert result["instance_id"] == "boot-a"


@pytest.mark.asyncio
async def test_instance_mismatch_is_rejected_before_any_inference():
    gateway.runtime.phase = "ready"
    gateway.runtime.process = SimpleNamespace(returncode=None)
    gateway.runtime.instance_id = "new-worker-instance"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://test") as client:
        result = await client.post(
            "/v1/chat/completions",
            headers={"X-CR-Expected-Instance-Id": "previous-worker-instance"},
            json={"messages": [{"role": "user", "content": "Hola"}]},
        )
        assert result.status_code == 409
        assert result.headers["x-cr-safe-rejection"] == "1"
        assert gateway.runtime.active == 0


@pytest.mark.asyncio
async def test_completed_warmup_gates_instance_ready_for_request():
    gateway.runtime.phase = "ready"
    gateway.runtime.process = SimpleNamespace(returncode=None)
    gateway.runtime.warmup_completed = False
    gateway.runtime.active = 0
    gateway.runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 1900, "completion_tokens": 24}})
    ))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://test") as client:
            before = (await client.get("/status")).json()
            assert before["model_loaded"] is True
            assert before["warmup_completed"] is False
            assert before["ready_for_request"] is False
            response = await client.post(
                "/v1/chat/completions",
                headers={"X-CR-Warmup": "1"},
                json={"messages": [{"role": "user", "content": "hi"}],
                      "max_tokens": 1, "stream": False},
            )
            assert response.status_code == 200
            after = (await client.get("/status")).json()
            assert after["warmup_completed"] is True
            assert after["ready_for_request"] is True
    finally:
        await gateway.runtime.client.aclose()


@pytest.mark.asyncio
async def test_parallel_warmups_generate_only_once():
    import asyncio
    calls = 0
    async def handler(request):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.03)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 1900, "completion_tokens": 24}})
    gateway.runtime.phase = "ready"
    gateway.runtime.process = SimpleNamespace(returncode=None)
    gateway.runtime.warmup_completed = False
    gateway.runtime.warmup_lock = asyncio.Lock()
    gateway.runtime.warmup_attempts = 0
    gateway.runtime.warmup_reused = 0
    gateway.runtime.active = 0
    gateway.runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "stream": False}
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://test") as client:
            a, b = await asyncio.gather(
                client.post("/v1/chat/completions", json=payload, headers={"X-CR-Warmup": "1"}),
                client.post("/v1/chat/completions", json=payload, headers={"X-CR-Warmup": "1"}),
            )
            assert [a.status_code, b.status_code] == [200, 200]
            assert calls == 1
            assert gateway.runtime.warmup_attempts == 1
            assert gateway.runtime.warmup_reused == 1
            assert a.headers.get("x-cr-warmup-reused") == "1" or b.headers.get("x-cr-warmup-reused") == "1"
            assert gateway.runtime.active == 0
    finally:
        await gateway.runtime.client.aclose()


@pytest.mark.asyncio
async def test_failed_warmup_does_not_poison_future_attempt():
    import asyncio
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(503 if calls == 1 else 200, json={"usage": {"prompt_tokens": 1900, "completion_tokens": 24}})
    gateway.runtime.phase = "ready"
    gateway.runtime.process = SimpleNamespace(returncode=None)
    gateway.runtime.warmup_completed = False
    gateway.runtime.warmup_lock = asyncio.Lock()
    gateway.runtime.active = 0
    gateway.runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    payload = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "stream": False}
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url="http://test") as client:
            first = await client.post("/v1/chat/completions", json=payload, headers={"X-CR-Warmup": "1"})
            second = await client.post("/v1/chat/completions", json=payload, headers={"X-CR-Warmup": "1"})
            assert (first.status_code, second.status_code) == (503, 200)
            assert calls == 2
            assert gateway.runtime.warmup_completed
    finally:
        await gateway.runtime.client.aclose()


def test_stream_probe_tracks_content_gap_and_usage():
    import time
    probe = gateway.StreamProbe(time.monotonic())
    assert probe.observe(b'data: {"choices":[{"delta":{"content":"Hola"}}]}\\n\\n'.replace(b'\\n', b'\n'))
    assert not probe.observe(b'data: {"choices":[{"delta":{"content":" mundo"}}]}\\n\\n'.replace(b'\\n', b'\n'))
    assert not probe.observe(b'data: {"choices":[],"usage":{"completion_tokens":2}}\\n\\n'.replace(b'\\n', b'\n'))
    assert probe.content_events == 2
    assert probe.output_tokens == 2
    assert probe.max_content_gap_ms >= 0


def test_representative_warmup_profile_has_fixed_synthetic_context(monkeypatch):
    from warmup_profile import synthetic_support_warmup, warmup_context_chars
    monkeypatch.setenv("WARMUP_OUTPUT_TOKENS", "24")
    payload = synthetic_support_warmup(gateway.MODEL_ID)
    assert payload["max_tokens"] == 24
    assert payload["min_tokens"] == 16
    assert payload["stream"] is False
    assert len(payload["messages"]) == 12
    assert warmup_context_chars() >= 5500
    assert all("secret" not in m["content"].lower() for m in payload["messages"])
    monkeypatch.setenv("WARMUP_OUTPUT_TOKENS", "40")
    with pytest.raises(ValueError):
        synthetic_support_warmup(gateway.MODEL_ID)


@pytest.mark.asyncio
async def test_legacy_warmup_is_replaced_with_realistic_synthetic_request():
    import asyncio
    import json
    sent = []
    def fake_vllm(request):
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "respuesta sintética"}}],
            "usage": {"prompt_tokens": 1935, "completion_tokens": 24},
        })
    gateway.runtime.phase = "ready"
    gateway.runtime.process = SimpleNamespace(returncode=None)
    gateway.runtime.instance_id = "warm-boot"
    gateway.runtime.warmup_completed = False
    gateway.runtime.warmup_lock = asyncio.Lock()
    gateway.runtime.active = 0
    gateway.runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(fake_vllm))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app),
                                    base_url="http://test") as client:
            result = await client.post("/v1/chat/completions",
                json={"messages": [{"role":"user","content":"private-message-not-to-forward"}],
                      "max_tokens":1,"stream":False},
                headers={"X-CR-Warmup":"1"})
            assert result.status_code == 200
            assert len(sent) == 1
            assert sent[0]["max_tokens"] == 24
            assert sent[0]["min_tokens"] == 16
            assert "private-message-not-to-forward" not in json.dumps(sent[0])
            assert gateway.runtime.warmup_completed
            status = (await client.get("/status")).json()
            assert status["warmup_profile"] == "support-chat-v2"
            assert status["warmup_prompt_tokens"] == 1935
            assert status["warmup_output_tokens"] == 24
    finally:
        await gateway.runtime.client.aclose()


@pytest.mark.asyncio
async def test_warmup_without_actual_decode_does_not_mark_worker_ready():
    import asyncio
    gateway.runtime.phase = "ready"
    gateway.runtime.process = SimpleNamespace(returncode=None)
    gateway.runtime.warmup_completed = False
    gateway.runtime.warmup_lock = asyncio.Lock()
    gateway.runtime.active = 0
    gateway.runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={
            "usage": {"prompt_tokens": 1900, "completion_tokens": 2},
        })
    ))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app),
                                    base_url="http://test") as client:
            result = await client.post("/v1/chat/completions",
                json={"messages":[{"role":"user","content":"hi"}],"max_tokens":1},
                headers={"X-CR-Warmup":"1"})
            assert result.status_code == 502
            assert gateway.runtime.warmup_completed is False
            assert gateway.runtime.active == 0
    finally:
        await gateway.runtime.client.aclose()
