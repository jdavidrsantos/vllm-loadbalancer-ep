# CreaRapido vLLM load-balancing worker

Runpod **Load Balancing** worker for **`Qwen/Qwen2.5-7B-Instruct`**. The public model weights and tokenizer are downloaded **during the Docker build** and live at `/models/Qwen2.5-7B-Instruct` in the resulting image. No Hugging Face token is needed for this public model, and **no token should be checked into Git or passed in a Docker build argument**.

The worker exposes a lightweight FastAPI gateway on `PORT` immediately, then starts vLLM on `127.0.0.1:8001` (not publicly exposed). Runpod's `/ping` reports **204 initializing → 200 ready** only after vLLM actually answers its internal `/health` probe. An unrecoverable startup failure returns 503.

> IMPORTANT: This is a change to **Load Balancing**, not queue-based Serverless. Create a **Load Balancer endpoint**. Queue `/run`, `/runsync`, and job webhooks do not apply to this worker. The image is large and baking weights eliminates download during startup but **cannot eliminate platform provisioning, image pulls, loading VRAM, or cold starts**.

## Build and deploy

```bash
docker build --platform linux/amd64 -t YOUR_DOCKER_USER/crearapido-qwen25-7b:v1 .
docker push YOUR_DOCKER_USER/crearapido-qwen25-7b:v1
```

In the Runpod console, **Serverless → New Endpoint → Import from Docker Registry**, select **Load Balancer** and enter the pushed image. Configure:

| Setting | Value |
|---|---|
| HTTP port exposed | `8000` |
| `PORT` | `8000` |
| `PORT_HEALTH` | `8000` |
| Health path | `/ping` |
| GPU | Recommend **24 GB VRAM or more** for BF16/FP16 weights plus KV cache; test exact GPU/model length |
| Active/min workers | `0` for minimal idle GPU billing; expect cold starts |
| Max workers | Start at `2`, increase based on observed concurrency and budgets |

Do **not** configure a Hugging Face token for these public weights. The model is baked locally; production runs with `HF_HUB_OFFLINE=1`. A 24 GB GPU is a safer starting point than 16 GB for unquantized 7B. Models, CUDA and GPU compatibility must be tested on the chosen GPU before production.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | FastAPI ingress and `/ping` port |
| `PORT_HEALTH` | `8000` | Set in Runpod to same exposed port |
| `MODEL_PATH` | `/models/Qwen2.5-7B-Instruct` | Baked local model directory |
| `INTERNAL_VLLM_PORT` | `8001` | Localhost-only vLLM server |
| `MAX_MODEL_LEN` | `4096` | Context budget; raise after measuring VRAM |
| `MAX_NUM_SEQS` | `16` | vLLM concurrency cap |
| `MAX_IN_FLIGHT` | `32` | FastAPI backpressure, HTTP 429 on overload |
| `DTYPE` | `auto` | Weight dtype (BF16 on supported GPUs) |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction available to engine |
| `TENSOR_PARALLEL_SIZE` | `1` | Set to actual GPU count for tensor parallelism |
| `ENFORCE_EAGER` | `true` | Faster cold start; possibly slower steady-state inference |
| `STARTUP_TIMEOUT_SECONDS` | `900` | Fail health check if internal vLLM never becomes ready |

To prioritize throughput over startup latency, try `ENFORCE_EAGER=false` and experiment with a *persistent* `VLLM_CACHE_ROOT` if attaching a Network Volume. Do not configure cache paths on a nonexistent volume. For repeatable builds, pass `--build-arg MODEL_REVISION=<verified Hugging Face commit SHA>` rather than `main`.

## API

All requests go through the authenticated **Runpod Load Balancing** URL, never direct to the internal vLLM port:

```bash
curl -N "https://ENDPOINT_ID.api.runpod.ai/v1/chat/completions" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Hola, ¿qué puedes hacer?"}],"max_tokens":150,"stream":true}'
```

- `POST /v1/chat/completions` — OpenAI-compatible chat and streaming.
- `POST /v1/completions` — text completions (for existing clients).
- `GET /ping` — Runpod readiness probe: HTTP **204** until vLLM is ready; **200** afterward; 503 if the process fails.
- `GET /status` — worker-local status and counters (only routable from the LB after a worker becomes healthy).

`model` may be omitted by older clients and will be filled in automatically. The gateway preserves vLLM's OpenAI streaming `data: ...\n\n` protocol, including `[DONE]`, and **does not repeatedly emit cumulative text**. The backend should process each `choices[].delta.content` fragment only once.

### Optional CreaRapido progress events

Add `X-CR-Status-Events: 1` **only** if the backend understands named SSE events. In addition to standard OpenAI `data:` frames, the gateway emits:

```text
event: cr.status
data: {"phase":"worker_ready","request_id":"..."}

event: cr.status
data: {"phase":"streaming","request_id":"..."}
```

The second event indicates the *first content token reached this gateway*, not when the worker began inference. **Do not enable this option in generic OpenAI SDK clients.** See [CreaRapido integration](docs/crearapido-integration.md).

### Observability and limitations

Worker stdout contains JSON events: `worker_boot`, `model_initializing`, `worker_ready`, `request_accepted`, `first_token`, `request_finished`, `client_disconnected`, `stream_error`, and failure events. Request IDs can correlate log lines. Logs deliberately do not include prompts, generated text, bearer tokens or personal data.

Response headers: `X-CR-Request-Id`, `X-CR-Worker-State`, and for non-streaming `X-CR-Worker-Processing-Ms` and `Server-Timing`. Streaming logs include `first_byte_ms`, `first_token_ms`, and `processing_ms`. The public Runpod LB may not deliver requests to *any* worker during an initial cold start, so no worker-side log can observe the **pre-routing wait**. Measure that from CreaRapido's backend (send → headers), and consult the endpoint's cold-start/delay metrics. Do **not** label all delays as cold starts.

## Test without a GPU

```bash
python -m pip install -r builder/requirements.txt pytest pytest-asyncio
pytest -q tests/
```

These tests mock the internal vLLM HTTP service. **They do not test CUDA compatibility, GPU memory use, cold-start duration, Runpod routing, or full Docker build**; test these on a staging Runpod endpoint before production.

## Security

- Never expose `RUNPOD_API_KEY` or a Hugging Face token in browser JS, client-side events, GitHub or image layers. Only CreaRapido's backend holds the Runpod API key.
- Runpod authenticates gateway calls; the internal vLLM server binds to **127.0.0.1** and the image publishes only port `8000`.
- The gateway never forwards the external `Authorization` header to the internal engine.
- Authentication, per-user quotas, rate limiting and message validation for CreaRapido users remain the responsibility of **CreaRapido's backend**.

## Blackwell MIG 24 GB compatibility and slow inference diagnostics

The pinned upstream `vllm/vllm-openai:v0.11.2` release is intended for CUDA 12.8+
Blackwell support. Its exact installed dependencies must still be verified
rather than assuming a tag proves the final image is compatible. The Docker
build now executes `pip check` and `scripts/check_gpu_stack.py --mode build`
and **fails** if the compiled CUDA is older than 12.8 or vLLM differs from
0.11.2. The script prints the built PyTorch, CUDA, FastAPI, Hugging Face Hub,
and supported architecture versions. The existing vLLM/PyTorch dependencies
are preserved instead of running a broad `pip install` with open version
ranges from `builder/requirements.txt`. The base must supply `httpx`,
`fastapi`, `uvicorn` and `huggingface_hub`; a missing component will be
reported during the build, not silently installed from an unpinned release.

**GPU-specific validation requires an actual running MIG worker.** After
deploying the test image, in its container run:

```bash
python3 /app/check_gpu_stack.py --mode runtime
nvidia-smi
```

Verify `compute_capability` (Blackwell `12.0`), available memory
(approximately 24 GiB before allocations), and host driver compatibility.
The optional `sm120_precompiled` field refers to the PyTorch wheel's
compiled targets; `false` alone does not prove every library is incompatible
because some CUDA kernels may compile at runtime. The MIG slice provides a
fraction of the full 96GB RTX PRO 6000 GPU's resources and cannot be assumed
to perform like a whole card.

**Do not change `ENFORCE_EAGER` blindly.** For latency measurements,
compare the same 2000-token context and output length on the same MIG
profile with `ENFORCE_EAGER=true` vs `false`. Track time to readiness,
first content token, 50th/95th percentile output tokens/second, and VRAM.
CUDA Graphs can improve steady-state speed but add initialization time and
memory consumption. Do not compare different worker instances or
different GPU partitions and attribute the difference to dependencies.

The base-image tag is retained; pin to a registry digest **only after**
verifying the actual manifest for the chosen platform.
