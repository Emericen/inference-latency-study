# Inference Latency Study

Controlled benchmark for measuring inference latency under agent-like payloads.

This repo answers one question: **What drives inference latency for computer-use agents?**

The goal is to identify which latency bottlenecks constrain the design of fast and usable agent products.

## What it measures

- Time to first token (TTFT)
- Total completion latency
- Decode throughput

The core comparison is **image+text vs text-only** at matched token counts, so latency differences come from modality rather than payload size.

## Repo layout

```
Makefile                   # Server-side: start/stop vLLM
requirements.txt           # Client-side Python dependencies
data/
  capture.py               # Capture screenshots (left-click to save, right-click to stop)
  dataset.py               # Build paired dataset from screenshots
  screenshots/             # Raw captures (gitignored)
  prepared/                # Built dataset (gitignored)
scripts/
  run.py                   # Run benchmark against a vLLM server
  aggregate.py             # JSONL → CSV summary
```

## Setup

### Client side (your laptop)

```bash
pip install -r requirements.txt
```

### Server side (GPU node)

Clone the repo and install vLLM:

```bash
pip install vllm
```

## Step 1: Capture screenshots

On your local machine, run the capture script and use your computer normally. Left-click saves a screenshot, right-click stops.

```bash
python data/capture.py
```

Aim for ~100 screenshots. They save to `data/screenshots/` with timestamp filenames.

## Step 2: Build the dataset

This counts vision tokens per screenshot using Qwen3-VL's image processor, then generates a random text chunk with the exact same token count for each image.

```bash
python data/dataset.py
```

To download the tokenizer on first run (if not cached locally):

```bash
python data/dataset.py --no-local-files-only
```

The dataset saves to `data/prepared/benchmark_dataset/`.

## Step 3: Start the server

SSH into your GPU node and start vLLM:

```bash
make vllm-up
```

Watch the logs until the server is ready:

```bash
make vllm-logs
```

You'll see a line like `Uvicorn running on http://0.0.0.0:8000` when it's ready.

To use a different model:

```bash
make vllm-up MODEL=Qwen/Qwen3-VL-4B-Instruct
```

## Step 4: Run the benchmark

Back on the client machine, run once per modality:

```bash
# Image + text
python scripts/run.py \
  --base-url http://<SERVER_IP>:8000/v1 \
  --modality image \
  --output results/raw/image_remote.jsonl

# Text only (matched token count)
python scripts/run.py \
  --base-url http://<SERVER_IP>:8000/v1 \
  --modality text \
  --output results/raw/text_remote.jsonl
```

For a local comparison (run this on the GPU node itself):

```bash
python scripts/run.py \
  --base-url http://localhost:8000/v1 \
  --modality image \
  --output results/raw/image_local.jsonl

python scripts/run.py \
  --base-url http://localhost:8000/v1 \
  --modality text \
  --output results/raw/text_local.jsonl
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3-VL-8B-Instruct` | Model name served by vLLM |
| `--warmups` | `2` | Warmup requests (excluded from results) |
| `--runs` | `10` | Measured requests |
| `--max-tokens` | `10` | Max completion tokens per request |
| `--dataset` | `data/prepared/benchmark_dataset` | Path to built dataset |

## Step 5: Aggregate results

```bash
python scripts/aggregate.py \
  --input results/raw/image_remote.jsonl \
  --output results/summaries/image_remote.csv
```

## Step 6: Stop the server

```bash
make vllm-down
```

## RunPod

Create an H100 pod:

```bash
runpodctl create pod \
  --gpuType "NVIDIA H100 80GB HBM3" \
  --name "inference-latency-study" \
  --dataCenterId "US-CA-2" \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --containerDiskSize 50 \
  --volumeSize 50 \
  --ports "8000/http" \
  --startSSH
```

The vLLM API is exposed at:

```
https://<POD_ID>-8000.proxy.runpod.net/v1
```

SSH in, clone the repo, `make vllm-up`, then run the benchmark from your laptop pointing at the proxy URL.

```bash
runpodctl get pod
runpodctl ssh connect <POD_ID>
runpodctl stop pod <POD_ID>
runpodctl remove pod <POD_ID>
```
