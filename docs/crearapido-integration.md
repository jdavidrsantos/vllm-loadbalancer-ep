# Integration with CreaRapido chat

**Recommendation: backend SSE bridge, not a webhook.** A Load Balancing endpoint does not use the queue-based `/run` handler/job webhook contract. OpenAI-compatible vLLM streaming is a much better match for interactive chat.

```
Browser chat  ← SSE event stream ←  CreaRapido backend  ← OpenAI SSE ← Runpod LB  ← vLLM worker
       |                                 |                     |
       +---- authenticated user --------+---- Runpod key -----+
```

The **backend** must create and authenticate the user session, enforce quotas, limit prompt size/output tokens, sanitize/log metadata only, and call `https://ENDPOINT_ID.api.runpod.ai/v1/chat/completions` with a **server-only** bearer key. Do not connect the browser directly to Runpod.

## State machine and honest messages

| Event measured by CreaRapido backend | User-visible text | Meaning |
|---|---|---|
| Request accepted by your backend | `Conectando con el asistente…` | Backend has begun request, not proof of GPU cold start |
| No upstream headers after ~3 seconds | `Buscando capacidad disponible…` | May be provisioning, LB wait, network, or capacity; exact cause unknown |
| No upstream headers after ~15 seconds | `El modelo puede estar iniciando. Esto puede tardar un poco.` | *Possible* cold start, not a confirmed diagnosis |
| Runpod upstream response received | `Preparando la respuesta…` | Gateway/engine returned headers; may still be preparing tokens |
| First nonempty `choices[].delta.content` | Hide status, render tokens immediately | Actual content is streaming |
| `[DONE]` | Mark completed, persist final answer | Generation has ended |
| 429 / 502 / 503 before content | Friendly recoverable error + bounded retry | Retry **only before** any visible token; do not blindly double-send replies |

Show a single evolving status, not both `Waking up` and `Levantando el asistente`. After the first token, stop displaying cold-start messages.

## Required backend telemetry

Use one backend-generated request ID and log these timestamps (server monotonic clock):

1. `request_sent_at` — immediately before HTTP call.
2. `headers_received_at` — after upstream response headers.
3. `first_content_token_at` — after first non-empty content delta.
4. `completed_at` — upon `[DONE]` or stream end.

Record `http_status`, `X-Runpod-Worker-Id` (if passed through), `X-CR-Request-Id`, whether stream had content, and error category. Record **durations, not prompts or API credentials**.

- `headers_received - request_sent` = **pre-response wait** (network, provisioning, LB, or engine wait): **not proof of a cold start**.
- `first_token - headers_received` = time to first content token after headers.
- `completed - first_token` = streaming duration.
- The Runpod Metrics tab separately reports cold-start counts/times and can confirm whether cold starts occurred.

Set a generous but bounded HTTP connection/response timeout because a zero-worker endpoint can take time to become ready. Runpod Load Balancing is not a durable queue; if overload matters, add backend-side admissions control, jittered retries **only before streaming**, and user-friendly failure handling. Do not retry after content has been sent without an explicit user action.

### Backend pseudocode (NestJS/Node; adapt to your existing route)

```ts
res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
res.setHeader('Cache-Control', 'no-cache, no-transform');
res.setHeader('X-Accel-Buffering', 'no');
const emit = (name: string, data: object) =>
  res.write(`event: ${name}\ndata: ${JSON.stringify(data)}\n\n`);

emit('status', { phase: 'connecting' });
const controller = new AbortController();
res.on('close', () => controller.abort());
const startedAt = performance.now();
const softWait = setTimeout(() => emit('status', { phase: 'waiting_for_capacity' }), 3000);
try {
  const upstream = await fetch(`${process.env.RUNPOD_LB_BASE_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${process.env.RUNPOD_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ model: 'Qwen/Qwen2.5-7B-Instruct', messages, stream: true,
      max_tokens: 384, temperature: 0.6 }),
    signal: controller.signal,
  });
  clearTimeout(softWait);
  if (!upstream.ok || !upstream.body) throw new Error(`upstream_${upstream.status}`);
  emit('status', { phase: 'processing' });
  // IMPORTANT: parse SSE frames incrementally from upstream.body, including
  // frames split across TCP chunks. For every choices[0].delta.content, emit
  // your existing browser token event; [DONE] closes the stream.
  // Do not forward a raw byte chunk as though it were a complete SSE frame.
  // Log timing to first content token and total duration for this request.
} catch (error) {
  emit('error', { message: 'El asistente no está disponible temporalmente.' });
} finally {
  clearTimeout(softWait);
  res.end();
}
```

The sample is an **integration sketch**, not a replacement for existing CreaRapido controller/service code. Preserve your existing authentication, chat histories, quotas, and UI. This repository intentionally does not modify those uninspected CreaRapido files.
