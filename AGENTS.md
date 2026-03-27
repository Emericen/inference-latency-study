# AGENTS.md

## Purpose

This repo exists to benchmark **inference latency** for agent-like model requests under controlled conditions.

Do not broaden the scope casually.

The benchmark is about:

- model-serving latency
- payload shape
- serving mode
- deployment topology
- controlled comparisons inside one model family

The benchmark is not about:

- quality
- task success
- full product UX
- human approval timing
- screenshot capture performance

## Hard Constraints

- Use `Qwen3-VL` as the core model family.
- Keep the core benchmark synthetic and privacy-safe.
- Match text-only and image+text prompt budgets as closely as possible.
- Separate local-vs-remote effects from stateful-vs-stateless effects.
- Separate frontier-provider comparisons from the core causal benchmark.
- Prefer a small number of interpretable plots over a large number of weak ones.

## Benchmark Shape

Core controlled studies:

1. Model scale: `2B`, `4B`, `8B`, `32B`
2. Dense vs MoE: `32B` vs `30B-A3B`
3. Precision: BF16 vs FP8
4. Modality: matched text-only vs matched image+text
5. Serving mode: stateless vs stateful
6. Client placement: local vs remote
7. Topology: 1 GPU vs tensor parallel

Recommended summary stats:

- `p50`
- `p95`
- mean

Recommended per-request metrics:

- `ttft_s`
- `total_latency_s`
- `decode_tps`
- `prompt_tokens`
- `completion_tokens`

## Data Rules

Use synthetic payloads for the main benchmark.

Text:

- generate or trim to a fixed token budget
- use deterministic seeds
- use unique payloads for uncached runs
- use repeated payloads for cache-sensitive runs

Images:

- use fixed resolution for controlled studies
- prefer deterministic synthetic generation
- keep image-token footprint fixed when comparing modality

Real screenshots may be used only for local code-path testing or a later appendix, not for the main benchmark claims.

## Current Token Budget Direction

As of the initial repo scaffold:

- the 16 screenshots in `desktop-autocomplete/experiments/s6_s3_bucket/data` are all `1920x1080`
- under `Qwen3-VL` preprocessing, each expands to the same prompt size
- that prompt size is `961` total tokens for image+question
- text-only prompts should be trimmed to the same total budget for matched modality tests

## Code Organization

Prefer one reusable harness:

- `bench/payloads/`
- `bench/token_budget.py`
- `bench/backends/`
- `bench/infra/`
- `bench/runner.py`
- `bench/schema.py`
- `analysis/aggregate.py`
- `analysis/plots.py`

Avoid copy-pasting near-identical benchmark scripts for each variation.

Scenarios should be data-driven. The runner should own measurement, logging, and repetition.

## Run Flow

Local and remote benchmark runs should differ at the benchmark layer by **base URL only**:

- local: `http://localhost:8000/v1`
- remote: `https://<POD_ID>-8000.proxy.runpod.net/v1`

Operationally, there are two supported modes:

- managed local server: `ils run --launch-server ...`
- attach to existing server: `ils run --base-url ...`

Do not require users to manually start and stop `vllm serve` for the normal local workflow.

## Environment Setup

Preferred local setup:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

If this machine will launch vLLM locally, also install:

```bash
uv pip install vllm
```

`openai` is a normal project dependency and should be installed with the repo itself, not imported behind a fallback `try/except` in runtime code.

## RunPod Operations

RunPod is the preferred deployment layer for controlled remote experiments.

### Setup

#### Prerequisites

```bash
brew install runpod/runpodctl/runpodctl
runpodctl config --apiKey $RUNPOD_API_KEY
```

#### SSH Key

RunPod SSH uses the key at `~/.ssh/id_ed25519_runpod`.

The SSH proxy format is:

```bash
ssh <POD_ID>-<HASH>@ssh.runpod.io -i ~/.ssh/id_ed25519_runpod
```

### Pod Lifecycle

#### Create

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

Example far-region pod for latency comparison:

```bash
runpodctl create pod \
  --gpuType "NVIDIA H100 80GB HBM3" \
  --name "inference-latency-study-eu" \
  --dataCenterId "EU-NL-1" \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --containerDiskSize 50 \
  --volumeSize 50 \
  --ports "8000/http" \
  --startSSH
```

GPU type must match the exact name from `runpodctl get cloud`. Common options:

| GPU | VRAM | On-Demand $/hr | Notes |
|-----|------|----------------|-------|
| `NVIDIA H100 80GB HBM3` | 80 GB | varies | Preferred for this benchmark. |
| `NVIDIA H100 PCIe` | 80 GB | varies | Acceptable fallback. |
| `NVIDIA H200` | 141 GB | varies | Higher headroom, higher cost. |
| `NVIDIA A100 80GB PCIe` | 80 GB | varies | Budget fallback. |

Use two GPUs only when a topology or larger-model run actually requires it.

#### List / Inspect

```bash
runpodctl get pod
runpodctl get pod <POD_ID>
runpodctl get cloud
```

#### Stop

```bash
runpodctl stop pod <POD_ID>
```

- `/workspace` persists
- the GPU is released
- restarting may land on a different physical machine

#### Start

```bash
runpodctl start pod <POD_ID>
```

- pod ID and proxy URL are preserved
- vLLM still needs several minutes to load the model on every start

#### Terminate

```bash
runpodctl remove pod <POD_ID>
```

This destroys the volume and all data.

### Connecting

#### HTTP

RunPod proxies HTTP ports through:

```text
https://<POD_ID>-<INTERNAL_PORT>.proxy.runpod.net
```

For direct vLLM:

```bash
curl https://<POD_ID>-8000.proxy.runpod.net/v1/models
```

#### SSH

```bash
ssh <POD_ID>-<HASH>@ssh.runpod.io -i ~/.ssh/id_ed25519_runpod
```

To run commands non-interactively through the proxy:

```bash
echo "<COMMAND> && exit" | ssh -tt -i ~/.ssh/id_ed25519_runpod <POD_ID>-<HASH>@ssh.runpod.io 2>&1
```

### Pod Setup For This Repo

Inside the pod:

```bash
cd /workspace
git clone <REPO_URL> inference-latency-study
cd inference-latency-study

uv venv --system-site-packages
source .venv/bin/activate
uv pip install -e .
uv pip install vllm
```

If the repo is already cloned in `/root`, the same setup works there as well.

### Benchmark Execution

#### Local on the pod

This mode lets the benchmark launch and stop `vllm serve` itself:

```bash
ils run \
  --scenario bench/scenarios/modality.yaml \
  --launch-server \
  --server-port 8000 \
  --client-mode local \
  --region us \
  --gpu-type "NVIDIA H100 80GB HBM3" \
  --gpu-count 1 \
  --tp-size 1 \
  --output results/raw/modality_local.jsonl
```

#### Remote from the laptop

This mode attaches to an already-running server:

```bash
ils run \
  --scenario bench/scenarios/modality.yaml \
  --base-url https://<POD_ID>-8000.proxy.runpod.net/v1 \
  --client-mode remote \
  --region us \
  --gpu-type "NVIDIA H100 80GB HBM3" \
  --gpu-count 1 \
  --tp-size 1 \
  --output results/raw/modality_remote.jsonl
```

The benchmark-level difference between local and remote runs should be `--base-url` only:

- local on the pod: `http://localhost:8000/v1`
- remote from the laptop: `https://<POD_ID>-8000.proxy.runpod.net/v1`

### Troubleshooting

#### Check if the model is loaded

```bash
nvidia-smi
curl -s http://localhost:8000/v1/models
```

#### Check processes

```bash
ps aux | grep -E "vllm"
```

#### Startup timeline

1. Container boots
2. `vllm serve` starts
3. model weights load into VRAM
4. `/v1/models` returns successfully

Expect several minutes of startup time for larger models.

#### Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Port 8000 not ready | Model still loading | Wait, then check `nvidia-smi` and `curl localhost:8000/v1/models` |
| `vllm` not found | Missing package in env | `uv pip install vllm` |
| SSH permission denied | Wrong key | Use `~/.ssh/id_ed25519_runpod` |
| SSH command hangs | Missing PTY | Use `ssh -tt` |
| Remote proxy errors | RunPod proxy flake | Retry the request; verify `localhost:8000` from inside the pod |

### Cost Management

- Running pod: GPU hourly cost depends on type
- Stopped pod: persistent storage cost remains
- Terminated pod: all storage is deleted

Daily workflow:

- stop the pod when done
- start it when needed
- keep the same pod when you want stable URLs and persistent `/workspace`

## Measurement Guidance

Do not rely on three-run averages as the main summary.

Instead:

- run warmups and discard them
- collect a distribution of measured requests per scenario
- report `p50`, `p95`, and mean

For streamed responses:

- `ttft_s` is measured at the first streamed token
- `total_latency_s` is measured at final token completion
- `decode_tps` is computed after the first token

## Editing Guidance

When extending this repo:

- preserve the narrow benchmark objective
- keep claims causal only where variables are actually controlled
- keep appendix comparisons clearly separated from the core study
- document assumptions in the README before expanding the matrix
