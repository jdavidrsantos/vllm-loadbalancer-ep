# vLLM image includes a CUDA-compatible torch/vLLM build. Pin for reproducibility.
# The upstream Docker Hub index digest has been verified for v0.11.2.
# Pinning it keeps the CUDA/PyTorch/vLLM stack reproducible across rebuilds.
FROM vllm/vllm-openai:v0.11.2@sha256:2c908d5a84ed251b6a17d179f42d06df1aff353007779ac5eecd8a0ea3fe9331

ARG MODEL_ID=Qwen/Qwen2.5-7B-Instruct
ARG MODEL_REVISION=main
ENV MODEL_PATH=/models/Qwen2.5-7B-Instruct \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PORT=8000 \
    INTERNAL_VLLM_PORT=8001 \
    MAX_MODEL_LEN=8192 \
    MAX_NUM_SEQS=16 \
    ENFORCE_EAGER=true \
    PYTHONUNBUFFERED=1

WORKDIR /app
# FastAPI, Uvicorn, httpx and huggingface-hub are already included in the
# official vLLM image. Re-installing loosely ranged libraries here could
# silently upgrade/downgrade a dependency and break the matched CUDA wheels.
# Keep the bundled, tested vLLM/PyTorch/FlashAttention dependencies intact.
COPY scripts/check_gpu_stack.py /app/check_gpu_stack.py
COPY scripts/check_base_dependencies.py /app/check_base_dependencies.py
RUN python3 /app/check_base_dependencies.py && \
    python3 /app/check_gpu_stack.py --mode build --output /app/cuda-stack.json

# Public Apache-2.0 model. Download at build-time (NOT on each GPU cold start).
# Never pass a Hugging Face token via build args or bake secrets into an image.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 MODEL_ID="$MODEL_ID" MODEL_REVISION="$MODEL_REVISION" \
    python3 -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["MODEL_ID"], revision=os.environ["MODEL_REVISION"], local_dir="/models/Qwen2.5-7B-Instruct", ignore_patterns=["*.bin", "*.pt", "*.msgpack", "*.h5"])'

COPY src/handler.py /app/handler.py
COPY src/warmup_profile.py /app/warmup_profile.py
# Fail during image build rather than crashing every Runpod worker at boot.
RUN python3 -m py_compile /app/handler.py /app/warmup_profile.py && \
    cd /app && python3 -c 'import warmup_profile; from warmup_profile import synthetic_support_warmup; assert synthetic_support_warmup("Qwen/Qwen2.5-7B-Instruct")["max_tokens"] >= 16'
EXPOSE 8000
# Override vLLM's base-image entrypoint so health responds while the engine boots.
ENTRYPOINT ["python3", "/app/handler.py"]
