"""CreaRapido's OpenAI-compatible, observable Runpod load-balancing worker.

Runpod checks /ping on PORT; 204 means initializing and 200 means ready.
The real vLLM server runs on loopback only, after this health server starts.
"""

from __future__ import annotations

import asyncio
import codecs
from contextlib import asynccontextmanager, suppress
import json
import logging
import os
import time
from uuid import uuid4

import httpx
from warmup_profile import WARMUP_PROFILE, synthetic_support_warmup
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_PATH = os.getenv("MODEL_PATH", "/models/Qwen2.5-7B-Instruct")
INTERNAL_PORT = int(os.getenv("INTERNAL_VLLM_PORT", "8001"))
MODEL_URL = f"http://127.0.0.1:{INTERNAL_PORT}"
MAX_IN_FLIGHT = int(os.getenv("MAX_IN_FLIGHT", os.getenv("MAX_NUM_SEQS", "16")))
STARTUP_TIMEOUT = int(os.getenv("STARTUP_TIMEOUT_SECONDS", "900"))
LOG = logging.getLogger("crearapido.worker")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")


def log(event: str, **fields: object) -> None:
    """Log structured events, never prompts, tokens or credentials."""
    LOG.info(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str))


def elapsed_ms(start: float) -> int:
    return round((time.monotonic() - start) * 1000)


class Runtime:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.instance_id = str(uuid4())
        self.last_health_ok_at: float | None = None
        self.phase = "initializing"
        self.ready_at: float | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.monitor: asyncio.Task | None = None
        self.client: httpx.AsyncClient | None = None
        self.active = 0
        self.completed = 0
        self.failed = 0
        self.warmup_completed = False
        self.warmup_finished_ms: int | None = None
        self.warmup_processing_ms: int | None = None
        self.warmup_attempts = 0
        self.warmup_reused = 0
        self.warmup_profile: str | None = None
        self.warmup_prompt_tokens: int | None = None
        self.warmup_output_tokens: int | None = None
        self.warmup_lock = asyncio.Lock()
        self.lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self.phase == "ready" and self.process is not None and self.process.returncode is None


runtime = Runtime()


def command() -> list[str]:
    cmd = [
        "vllm", "serve", MODEL_PATH,
        "--served-model-name", MODEL_ID,
        "--host", "127.0.0.1", "--port", str(INTERNAL_PORT),
        "--dtype", os.getenv("DTYPE", "auto"),
        "--max-model-len", os.getenv("MAX_MODEL_LEN", "8192"),
        "--max-num-seqs", os.getenv("MAX_NUM_SEQS", "16"),
        "--gpu-memory-utilization", os.getenv("GPU_MEMORY_UTILIZATION", "0.90"),
        "--tensor-parallel-size", os.getenv("TENSOR_PARALLEL_SIZE", "1"),
        "--generation-config", "vllm",
        "--disable-log-requests",
    ]
    if os.getenv("ENFORCE_EAGER", "true").lower() in ("1", "true", "yes"):
        cmd.append("--enforce-eager")
    return cmd


async def monitor_model() -> None:
    """Keep checking actual vLLM health for the entire worker lifetime.

    Runpod's RUNNING/starting is NOT a readiness signal. Once vLLM passes
    /health the worker is routable, but a later engine failure must revoke it.
    """
    deadline = time.monotonic() + STARTUP_TIMEOUT
    last_report = 0.0
    while True:
        now = time.monotonic()
        if runtime.process and runtime.process.returncode is not None:
            runtime.phase = "failed"
            log("model_process_exited", code=runtime.process.returncode, boot_ms=elapsed_ms(runtime.started))
            return

        healthy = False
        try:
            response = await runtime.client.get(f"{MODEL_URL}/health", timeout=2.0)
            healthy = response.status_code == 200
        except (httpx.HTTPError, RuntimeError):
            pass

        if healthy:
            if runtime.phase != "ready":
                runtime.phase = "ready"
                if runtime.ready_at is None:
                    runtime.ready_at = time.monotonic()
                log("worker_ready", model=MODEL_ID, boot_ms=elapsed_ms(runtime.started),
                    instance_id=runtime.instance_id)
            runtime.last_health_ok_at = time.monotonic()
        elif runtime.ready_at is None and now >= deadline:
            runtime.phase = "failed"
            log("model_startup_timeout", boot_ms=elapsed_ms(runtime.started))
            return
        elif runtime.phase == "ready" and (
            runtime.last_health_ok_at is None or now - runtime.last_health_ok_at > 4
        ):
            runtime.phase = "unhealthy"
            log("model_health_lost", instance_id=runtime.instance_id)
        elif runtime.phase == "initializing" and now - last_report >= 10:
            log("model_initializing", elapsed_ms=elapsed_ms(runtime.started),
                instance_id=runtime.instance_id)
            last_report = now
        await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(_: FastAPI):
    runtime.started = time.monotonic()
    runtime.instance_id = str(uuid4())
    runtime.last_health_ok_at = None
    runtime.phase = "initializing"
    runtime.ready_at = None
    runtime.warmup_completed = False
    runtime.warmup_finished_ms = None
    runtime.warmup_processing_ms = None
    runtime.warmup_attempts = 0
    runtime.warmup_reused = 0
    runtime.warmup_profile = None
    runtime.warmup_prompt_tokens = None
    runtime.warmup_output_tokens = None
    runtime.warmup_lock = asyncio.Lock()
    runtime.client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=315, write=30, pool=10))
    try:
        log("worker_boot", model=MODEL_ID, baked_model=MODEL_PATH,
            instance_id=runtime.instance_id)
        runtime.process = await asyncio.create_subprocess_exec(*command())
        runtime.monitor = asyncio.create_task(monitor_model())
    except (OSError, ValueError) as exc:
        runtime.phase = "failed"
        log("model_start_failed", error_type=type(exc).__name__)
    yield  # Open the health server immediately, even when model is initializing.
    if runtime.monitor:
        runtime.monitor.cancel()
        with suppress(asyncio.CancelledError):
            await runtime.monitor
    if runtime.process and runtime.process.returncode is None:
        runtime.process.terminate()
        try:
            await asyncio.wait_for(runtime.process.wait(), timeout=15)
        except asyncio.TimeoutError:
            runtime.process.kill()
            await runtime.process.wait()
    await runtime.client.aclose()
    log("worker_stopped")


app = FastAPI(title="CreaRapido vLLM Worker", version="2.0.0", lifespan=lifespan)


@app.get("/ping")
async def ping():
    if runtime.phase == "failed" or (runtime.process and runtime.process.returncode is not None):
        return Response(status_code=503)
    if not runtime.ready:
        return Response(status_code=204)  # No response body is allowed for HTTP 204.
    return {"status": "healthy"}


@app.get("/status")
async def status():
    """Worker-local readiness evidence; never aggregate this into a fleet count."""
    model_loaded = runtime.ready
    available_slots = max(0, MAX_IN_FLIGHT - runtime.active) if model_loaded else 0
    return JSONResponse(
        {
            "phase": "ready" if model_loaded else runtime.phase,
            "model": MODEL_ID,
            "max_model_len": int(os.getenv("MAX_MODEL_LEN", "8192")),
            "model_loaded": model_loaded,
            "warmup_completed": runtime.warmup_completed,
            "warmup_finished_ms": runtime.warmup_finished_ms,
            "warmup_processing_ms": runtime.warmup_processing_ms,
            "warmup_in_progress": runtime.warmup_lock.locked(),
            "warmup_attempts": runtime.warmup_attempts,
            "warmup_reused": runtime.warmup_reused,
            "warmup_profile": runtime.warmup_profile,
            "warmup_prompt_tokens": runtime.warmup_prompt_tokens,
            "warmup_output_tokens": runtime.warmup_output_tokens,
            "instance_id": runtime.instance_id,
            "ready_for_request": model_loaded and runtime.warmup_completed and available_slots > 0,
            "active_requests": runtime.active,
            "max_in_flight": MAX_IN_FLIGHT,
            "available_slots": available_slots,
            "worker_uptime_ms": elapsed_ms(runtime.started),
            "model_boot_ms": round((runtime.ready_at - runtime.started) * 1000) if runtime.ready_at else None,
            "last_health_age_ms": elapsed_ms(runtime.last_health_ok_at) if runtime.last_health_ok_at else None,
            "completed_requests": runtime.completed,
            "failed_requests": runtime.failed,
        },
        headers={"Cache-Control": "no-store", "X-CR-Worker-Instance-Id": runtime.instance_id},
    )

def worker_headers(trace_id: str, processing_ms: int | None = None) -> dict[str, str]:
    headers = {"X-CR-Request-Id": trace_id, "X-CR-Worker-State": "ready", "Cache-Control": "no-store"}
    if processing_ms is not None:
        headers["X-CR-Worker-Processing-Ms"] = str(processing_ms)
        headers["Server-Timing"] = f"worker;dur={processing_ms}"
    return headers


async def reserve(trace_id: str) -> None:
    async with runtime.lock:
        if runtime.active >= MAX_IN_FLIGHT:
            log("request_rejected", reason="overloaded", request_id=trace_id, active=runtime.active)
            raise HTTPException(status_code=429, detail="Worker at capacity",
                                headers={"Retry-After": "2", "X-CR-Safe-Rejection": "1"})
        runtime.active += 1


async def release(failed: bool) -> None:
    async with runtime.lock:
        runtime.active -= 1
        if failed:
            runtime.failed += 1
        else:
            runtime.completed += 1


class StreamProbe:
    """Read event boundaries only to timestamp the first content token."""

    def __init__(self, start: float) -> None:
        self.started = start
        self.first_byte_ms: int | None = None
        self.first_token_ms: int | None = None
        self.buffer = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.content_events = 0
        self.last_content_ms: int | None = None
        self.max_content_gap_ms = 0
        self.output_tokens: int | None = None

    def observe(self, chunk: bytes) -> bool:
        if self.first_byte_ms is None:
            self.first_byte_ms = elapsed_ms(self.started)
        self.buffer += self.decoder.decode(chunk)
        # SSE events can span multiple TCP chunks; ignore unbounded invalid input.
        if len(self.buffer) > 262144:
            self.buffer = ""
            return False
        self.buffer = self.buffer.replace("\r\n", "\n")
        first_content = False
        while "\n\n" in self.buffer:
            frame, self.buffer = self.buffer.split("\n\n", 1)
            for line in frame.splitlines():
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                    if not isinstance(data, dict):
                        continue
                    usage = data.get("usage")
                    if isinstance(usage, dict):
                        tokens = usage.get("completion_tokens")
                        if type(tokens) is int and tokens >= 0:
                            self.output_tokens = tokens
                    choices = data.get("choices") or []
                    if not any(c.get("delta", {}).get("content") or c.get("text") for c in choices):
                        continue
                    now_ms = elapsed_ms(self.started)
                    if self.last_content_ms is not None:
                        self.max_content_gap_ms = max(
                            self.max_content_gap_ms, now_ms - self.last_content_ms
                        )
                    self.last_content_ms = now_ms
                    self.content_events += 1
                    if self.first_token_ms is None:
                        self.first_token_ms = now_ms
                        first_content = True
                except (ValueError, AttributeError, TypeError):
                    pass
        return first_content


def stage_event(phase: str, request_id: str) -> bytes:
    return f"event: cr.status\ndata: {json.dumps({'phase': phase, 'request_id': request_id})}\n\n".encode()


async def forward_nonstream(
    trace_id: str,
    started: float,
    route: str,
    payload: dict,
    is_warmup: bool,
) -> Response:
    """One admission/release per real inference, even when an upstream call fails."""
    await reserve(trace_id)
    log("request_accepted", request_id=trace_id, route=route, stream=False,
        message_count=len(payload.get("messages", [])) if isinstance(payload.get("messages"), list) else 0,
        active=runtime.active)
    failed = True
    try:
        if is_warmup:
            runtime.warmup_attempts += 1
        upstream = await runtime.client.post(
            MODEL_URL + route, json=payload, headers={"X-Request-Id": trace_id}
        )
        processing_ms = elapsed_ms(started)
        failed = upstream.status_code >= 400
        if is_warmup and upstream.status_code == 200:
            try:
                result = upstream.json()
                usage = result.get("usage") if isinstance(result, dict) else None
                completion = usage.get("completion_tokens") if isinstance(usage, dict) else None
                prompt = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                if not isinstance(completion, int) or completion < 16:
                    log("warmup_incomplete", instance_id=runtime.instance_id,
                        request_id=trace_id, output_tokens=completion)
                    return Response(content=upstream.content, status_code=502,
                                    media_type="application/json",
                                    headers=worker_headers(trace_id, processing_ms))
                runtime.warmup_profile = WARMUP_PROFILE
                runtime.warmup_output_tokens = completion
                runtime.warmup_prompt_tokens = prompt if isinstance(prompt, int) else None
            except (ValueError, TypeError, AttributeError):
                log("warmup_invalid_response", request_id=trace_id)
                return Response(status_code=502, headers=worker_headers(trace_id, processing_ms))
            runtime.warmup_completed = True
            runtime.warmup_finished_ms = elapsed_ms(runtime.started)
            runtime.warmup_processing_ms = processing_ms
            log("warmup_completed", instance_id=runtime.instance_id, request_id=trace_id,
                warmup_finished_ms=runtime.warmup_finished_ms,
                warmup_processing_ms=processing_ms,
                warmup_profile=runtime.warmup_profile,
                prompt_tokens=runtime.warmup_prompt_tokens,
                output_tokens=runtime.warmup_output_tokens)
        log("request_finished", request_id=trace_id, status=upstream.status_code,
            processing_ms=processing_ms, stream=False)
        return Response(content=upstream.content, status_code=upstream.status_code,
                        media_type="application/json", headers=worker_headers(trace_id, processing_ms))
    except httpx.HTTPError as exc:
        log("upstream_error", request_id=trace_id, error_type=type(exc).__name__,
            ms=elapsed_ms(started))
        raise HTTPException(status_code=502, detail="Inference engine temporarily unavailable") from None
    finally:
        await release(failed)


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
async def inference(request: Request):
    trace_id = str(uuid4())  # Never trust arbitrary client-supplied IDs in logs.
    started = time.monotonic()
    if not runtime.ready:
        log("request_rejected", reason=runtime.phase, request_id=trace_id)
        raise HTTPException(status_code=503, detail="Model not ready",
                            headers={"Retry-After": "3", "X-CR-Safe-Rejection": "1"})

    expected_instance = request.headers.get("x-cr-expected-instance-id")
    if expected_instance and expected_instance != runtime.instance_id:
        # The load balancer chose a different worker after the verified probe.
        # Reject BEFORE reserving a slot or submitting anything to vLLM.
        log("request_rejected", reason="worker_changed", request_id=trace_id,
            instance_id=runtime.instance_id)
        raise HTTPException(status_code=409, detail="Verified worker changed",
                            headers={"X-CR-Safe-Rejection": "1"})

    try:
        payload = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Expected a JSON object") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected a JSON object")
    payload.setdefault("model", MODEL_ID)
    if payload["model"] != MODEL_ID:
        raise HTTPException(status_code=400, detail=f"Only {MODEL_ID} is served")
    if request.url.path.endswith("chat/completions") and not isinstance(payload.get("messages"), list):
        raise HTTPException(status_code=422, detail="messages must be a list")

    streaming = payload.get("stream") is True
    is_warmup = request.headers.get("x-cr-warmup") == "1"
    if is_warmup and streaming:
        raise HTTPException(status_code=400, detail="Warmup must be nonstreaming")
    if is_warmup:
        # Replace any legacy one-token request with the fixed, synthetic workload.
        # Never send real chat content to the warmup engine.
        payload = synthetic_support_warmup(MODEL_ID)
    opt_in_events = streaming and request.headers.get("x-cr-status-events") == "1"
    if is_warmup:
        # Runpod can route several backend warmup calls to the same worker.
        # Serialize them per boot, then reuse the first successful result.
        # Keep the normal inference path unlocked so it is not held up.
        async with runtime.warmup_lock:
            if not runtime.ready:
                raise HTTPException(status_code=503, detail="Model not ready",
                                    headers={"Retry-After": "3", "X-CR-Safe-Rejection": "1"})
            if runtime.warmup_completed:
                runtime.warmup_reused += 1
                log("warmup_reused", request_id=trace_id, instance_id=runtime.instance_id,
                    reused_count=runtime.warmup_reused)
                return JSONResponse(
                    {"status": "already_warm", "instance_id": runtime.instance_id,
                     "warmup_profile": runtime.warmup_profile},
                    headers={**worker_headers(trace_id, elapsed_ms(started)),
                             "X-CR-Warmup-Reused": "1"},
                )
            return await forward_nonstream(
                trace_id, started, request.url.path, payload, is_warmup=True
            )
    if not streaming:
        return await forward_nonstream(
            trace_id, started, request.url.path, payload, is_warmup=False
        )

    await reserve(trace_id)
    log("request_accepted", request_id=trace_id, route=request.url.path, stream=True,
        message_count=len(payload.get("messages", [])) if isinstance(payload.get("messages"), list) else 0,
        active=runtime.active)
    headers = {"X-Request-Id": trace_id}  # Do not forward the Runpod bearer token.
    try:
        context = runtime.client.stream("POST", MODEL_URL + request.url.path, json=payload, headers=headers)
        upstream = await context.__aenter__()
        if upstream.status_code != 200:
            content = await upstream.aread()
            await context.__aexit__(None, None, None)
            await release(True)
            log("request_finished", request_id=trace_id, status=upstream.status_code,
                processing_ms=elapsed_ms(started), stream=True)
            return Response(content=content, status_code=upstream.status_code, media_type="application/json",
                            headers=worker_headers(trace_id, elapsed_ms(started)))
    except httpx.HTTPError as exc:
        await release(True)
        log("upstream_error", request_id=trace_id, error_type=type(exc).__name__, ms=elapsed_ms(started))
        raise HTTPException(status_code=502, detail="Inference engine temporarily unavailable") from None

    async def forward():
        probe = StreamProbe(started)
        failed = False
        try:
            if opt_in_events:
                yield stage_event("worker_ready", trace_id)
            async for chunk in upstream.aiter_bytes():
                first_token = probe.observe(chunk)
                if first_token:
                    log("first_token", request_id=trace_id, first_token_ms=probe.first_token_ms)
                    if opt_in_events:
                        yield stage_event("streaming", trace_id)
                yield chunk  # OpenAI-compatible SSE is preserved byte-for-byte by default.
        except (httpx.HTTPError, asyncio.TimeoutError) as exc:
            failed = True
            log("stream_error", request_id=trace_id, error_type=type(exc).__name__)
            # Never mark a broken generation as successfully completed.
            # Clients must reject an interrupted stream even if tokens were emitted.
            yield b'data: {"error":{"message":"Upstream stream interrupted"}}\n\n'
        except asyncio.CancelledError:
            failed = True
            log("client_disconnected", request_id=trace_id, ms=elapsed_ms(started))
            raise
        finally:
            await context.__aexit__(None, None, None)
            await release(failed)
            processing_ms = elapsed_ms(started)
            decoding_ms = (
                max(1, processing_ms - probe.first_token_ms)
                if probe.first_token_ms is not None else None
            )
            tokens_per_second = (
                round(probe.output_tokens / (decoding_ms / 1000), 2)
                if probe.output_tokens is not None and decoding_ms else None
            )
            log("request_finished", request_id=trace_id, stream=True, failed=failed,
                processing_ms=processing_ms, first_byte_ms=probe.first_byte_ms,
                first_token_ms=probe.first_token_ms, content_events=probe.content_events,
                max_content_gap_ms=probe.max_content_gap_ms,
                output_tokens=probe.output_tokens, tokens_per_second=tokens_per_second)

    return StreamingResponse(forward(), media_type="text/event-stream",
                             headers={**worker_headers(trace_id), "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), access_log=False)
