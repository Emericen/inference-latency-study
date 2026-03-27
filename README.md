# Inference Latency Study

Controlled benchmark for measuring inference latency under agent-like payloads.

This repo answers one question: **What drives inference latency for computer-use agents?**

The goal is to identify which latency bottlenecks constrain the design of fast and usable agent products.

## Scope

This benchmark measures:

- time to first token
- total completion latency
- decode throughput

This benchmark does not measure:

- output quality
- tool execution time
- task completion success

The current main study focuses on:

- `Qwen3-VL`
- stateless OpenAI-compatible `vLLM`
- realistic screenshot payloads from `ScreenSpot`
- remote request size buckets instead of raw image resolution

## Repo Layout

- `data/`: prepares payloads and token accounting
- `server/`: talks to `vLLM` and records per-request latency
- `scripts/`: human entrypoints for prepare, run, aggregate, and server lifecycle
- `configs/`: experiment YAML files

## Main Experiment

The core experiment varies serialized screenshot payload size while holding the rest of the stack fixed.

Current bucket targets:

- `128 KB`
- `256 KB`
- `512 KB`
- `768 KB`
- `1 MB`

Each request logs:

- `ttft_s`
- `total_latency_s`
- `decode_tps`
- `prompt_tokens`
- `vision_tokens_total`
- `image_bytes`
- `request_bytes_total`

Reported summaries use:

- `p50`
- `p95`
- mean

## Setup

Create an environment and install the repo:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

On GPU images that already ship with CUDA and PyTorch, prefer:

```bash
uv venv --system-site-packages
source .venv/bin/activate
uv pip install -e .
```

If this machine will host `vllm serve`, also install:

```bash
uv pip install vllm
```

## Prepare Data

Pre-download the ScreenSpot shard used by the main benchmark:

```bash
python data/prepare.py
```

This creates local prepared JPEG buckets and a manifest under `data/prepared/`.
Those generated assets are used by the benchmark run, and they are intentionally not committed to version control.

## Run Locally On The Inference Node

Start `vllm serve` in one terminal:

```bash
make -C scripts vllm-up MODEL=Qwen/Qwen3-VL-8B-Instruct
make -C scripts vllm-wait
```

Run the main payload-bucket benchmark in another terminal:

```bash
python scripts/run.py \
  --config configs/screenspot_payload_buckets.yaml \
  --base-url http://127.0.0.1:8000/v1 \
  --client-mode local \
  --region us-ca \
  --gpu-type "NVIDIA H100 80GB HBM3" \
  --gpu-count 1 \
  --tp-size 1 \
  --output results/raw/screenspot_payload_buckets_local.jsonl
```

Stop the server when done:

```bash
make -C scripts vllm-down
```

## Run Remotely From Another Machine

Point the same benchmark at the pod proxy URL:

```bash
python scripts/run.py \
  --config configs/screenspot_payload_buckets.yaml \
  --base-url https://<POD_ID>-8000.proxy.runpod.net/v1 \
  --client-mode remote \
  --region us-ca \
  --gpu-type "NVIDIA H100 80GB HBM3" \
  --gpu-count 1 \
  --tp-size 1 \
  --output results/raw/screenspot_payload_buckets_remote.jsonl
```

Local versus remote should differ only by `--base-url` and client metadata.

## Aggregate Results

```bash
python scripts/aggregate.py \
  --input results/raw/screenspot_payload_buckets_remote.jsonl \
  --output results/summaries/screenspot_payload_buckets_remote.csv
```

## RunPod

RunPod provisioning and SSH workflow live in [AGENTS.md](AGENTS.md).

This repo assumes direct `vllm serve` on port `8000`. RunPod does not support Docker-in-Docker here, so server lifecycle is managed by shell scripts and Make targets instead of nested containers.
