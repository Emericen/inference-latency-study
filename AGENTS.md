# AGENTS.md

## Purpose

This repo studies inference latency for computer-use agents under controlled conditions.

Keep the scope narrow:

- measure inference latency
- vary payload shape, model choice, and deployment path
- stay inside one primary model family when making causal claims

Do not broaden the benchmark to include:

- output quality
- task success
- full end-to-end agent UX
- screenshot capture time

## Current Main Study

The mainline experiment is:

- `Qwen3-VL`
- stateless OpenAI-compatible `vLLM`
- `ScreenSpot` screenshots
- remote payload size buckets up to `1 MB`

Primary logged fields:

- `ttft_s`
- `total_latency_s`
- `decode_tps`
- `prompt_tokens`
- `vision_tokens_total`
- `image_bytes`
- `request_bytes_total`

Primary summaries:

- `p50`
- `p95`
- mean

## Repo Layout

- `data/`: payload preparation and token accounting
- `server/`: vLLM client, readiness check, scenario runner, result schema
- `scripts/`: prepare, run, aggregate, and server lifecycle entrypoints
- `configs/`: scenario YAMLs

Keep each area small. Prefer plain functions and `dataclass` records over deep abstractions.

## Run Flow

The benchmark does not manage server lifecycle programmatically.

Use shell entrypoints for server operations:

- `make -C scripts vllm-up`
- `make -C scripts vllm-wait`
- `make -C scripts vllm-down`
- `make -C scripts vllm-logs`

Then run the benchmark separately:

```bash
python scripts/run.py \
  --config configs/screenspot_payload_buckets.yaml \
  --base-url http://127.0.0.1:8000/v1 \
  --output results/raw/screenspot_payload_buckets.jsonl
```

Local versus remote should differ only by `--base-url` and metadata such as `--client-mode` and `--region`.

Run `python data/prepare.py` before measurement runs that use prepared screenshot buckets.
The prepared JPEGs and manifest live under `data/prepared/` locally and are not tracked in git.
Prepared-image runs must not reuse an image within the same server lifetime unless that reuse is intentional. The prep step should generate at least `warmups + runs` unique images per bucket, and reruns on the same server should use a different `--prepared-index-offset` or a fresh server process.

## Environment

Preferred setup:

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

If the machine hosts `vllm serve`, also install:

```bash
uv pip install vllm
```

## RunPod

RunPod is the default remote environment.

Prerequisites:

```bash
brew install runpod/runpodctl/runpodctl
runpodctl config --apiKey $RUNPOD_API_KEY
```

Create a near-region H100 pod:

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

Create a farther-region comparison pod:

```bash
runpodctl create pod \
  --gpuType "NVIDIA H100 80GB HBM3" \
  --name "inference-latency-study-east" \
  --dataCenterId "US-MO-1" \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --containerDiskSize 50 \
  --volumeSize 50 \
  --ports "8000/http" \
  --startSSH
```

Inspect or stop pods:

```bash
runpodctl get pod
runpodctl get pod <POD_ID>
runpodctl stop pod <POD_ID>
runpodctl start pod <POD_ID>
runpodctl remove pod <POD_ID>
```

SSH with the RunPod key:

```bash
runpodctl ssh connect <POD_ID>
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519_runpod
```

RunPod exposes the `vllm` API at:

```text
https://<POD_ID>-8000.proxy.runpod.net/v1
```

RunPod does not support Docker-in-Docker for this workflow, so use direct `vllm serve` instead of nested containers.
