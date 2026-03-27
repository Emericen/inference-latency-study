# Inference Latency Study

Controlled benchmark for measuring inference latency under agent-like payloads.

This repo answers one question: **What drives inference latency for computer-use agents?**

The goal is to understand which latency bottlenecks constrain the design of fast and usable agent products.

## Scope

This benchmark measures:

- time to first token
- total completion latency
- decode throughput

This benchmark studies latency as a function of:

- model scale
- modality and payload shape
- dense vs MoE architecture
- precision choice such as BF16 vs FP8
- stateful vs stateless serving
- local vs remote client placement
- tensor parallelism and deployment topology

This benchmark does **not** measure:

- output quality
- tool execution time
- task completion success

## Model Family

The core benchmark uses the `Qwen3-VL` family only so that comparisons stay within one model generation.

Planned core variants:

- `Qwen/Qwen3-VL-2B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3-VL-32B-Instruct`
- `Qwen/Qwen3-VL-30B-A3B-Instruct`

Planned precision variants where available:

- BF16 / default weights
- FP8

## Data Strategy

The main benchmark uses synthetic data, not real user screenshots or curated text corpora.

Why:

- exact control over token budgets
- less hidden variation across samples
- explicit cache-on vs cache-off testing

Planned payloads:

- text-only payloads with fixed prompt-token budgets
- image+text payloads with fixed image resolution and fixed image-token budgets
- unique seeded payloads for uncached runs
- repeated seeded payloads for cache-sensitive runs

## Current Prompt Budget Direction

Using `Qwen3-VL` tokenization rules:

- the 16 `1920x1080` screenshots in `desktop-autocomplete/experiments/s6_s3_bucket/data` all expand to the same image prompt size
- that image+question prompt is `961` tokens total
- text-only payloads should therefore be trimmed to the same total prompt-token budget for clean modality comparisons

## Experiment Plan

The benchmark will be organized as a small number of controlled studies instead of one giant Cartesian product.

### 1. Model Scale

Fixed payload, fixed precision, fixed server mode:

- `2B`
- `4B`
- `8B`
- `32B`

### 2. Dense vs MoE

Fixed payload, fixed precision, fixed server mode:

- `32B dense`
- `30B-A3B MoE`

### 3. Precision

Same model, same payload:

- BF16
- FP8

### 4. Modality

Matched prompt-token budgets:

- text-only
- image+text

### 5. Serving Mode

Same model, same payload:

- stateless
- stateful

### 6. Client Placement

Same server, same model, same payload:

- local client on the inference node
- remote client off-node

### 7. Topology

Same model, same payload:

- single GPU
- tensor parallel multi-GPU

## Metrics

Each measured request should log:

- `ttft_s`
- `total_latency_s`
- `decode_tps`
- `prompt_tokens`
- `completion_tokens`
- `model_id`
- `arch_type`
- `precision`
- `server_mode`
- `client_mode`
- `payload_kind`
- `gpu_type`
- `gpu_count`
- `tp_size`
- `region`
- `run_id`
- `request_id`

Reported summary stats should use:

- `p50`
- `p95`
- mean

Three-run averages are not the target summary method for this repo.

## Implementation Direction

The code should be rebuilt around one reusable harness instead of many near-duplicate scripts.

Planned components:

- payload builder
- token-budget calculator
- backend adapters for OpenAI-compatible and stateful serving
- local vLLM server management with readiness checks
- scenario runner
- normalized result schema
- aggregation and plotting scripts

## How To Run

Create an environment and install the benchmark:

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

If this machine will launch `vllm serve` locally, also install vLLM:

```bash
uv pip install vllm
```

Run one scenario with a managed local server:

```bash
ils run \
  --scenario bench/scenarios/modality.yaml \
  --launch-server \
  --server-port 8000 \
  --client-mode local \
  --output results/raw/modality_local.jsonl
```

Run the same benchmark remotely against an already-running pod:

```bash
ils run \
  --scenario bench/scenarios/modality.yaml \
  --base-url https://<POD_ID>-8000.proxy.runpod.net/v1 \
  --client-mode remote \
  --output results/raw/modality_remote.jsonl
```

Aggregate raw results:

```bash
ils aggregate \
  --input results/raw/modality_local.jsonl \
  --output results/summaries/modality_local.csv
```

The current harness provides:

- synthetic text-only payloads with fixed prompt-token budgets
- synthetic image+text payloads with fixed image size
- scenario-driven execution
- per-request JSONL logging
- summary CSV aggregation

For benchmark purposes, local vs remote should differ mainly by `--base-url`:

- local: `http://localhost:8000/v1`
- remote: `https://<POD_ID>-8000.proxy.runpod.net/v1`

The initial implementation prioritizes reproducibility and measurement discipline over broad provider coverage.

## RunPod Workflow

RunPod provisioning, SSH, troubleshooting, and command examples now live in [AGENTS.md](AGENTS.md).

The intended flow is:

1. Create or start a pod from the RunPod CLI.
2. SSH into the pod and install the repo with `uv`.
3. Run `ils run --launch-server ...` on the pod for local measurements.
4. Run `ils run --base-url https://<POD_ID>-8000.proxy.runpod.net/v1 ...` from your laptop for remote measurements.

This repo should talk directly to `vllm serve` on port `8000`, not a separate wrapper service.

Example RunPod creation for a near-region H100:

```bash
runpodctl create pod \
  --name "ils-us-ca" \
  --gpuType "NVIDIA H100 80GB HBM3" \
  --dataCenterId "US-CA-2" \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --containerDiskSize 50 \
  --volumeSize 50 \
  --ports "8000/http" \
  --startSSH
```

Example far-region pod:

```bash
runpodctl create pod \
  --name "ils-eu-nl" \
  --gpuType "NVIDIA H100 80GB HBM3" \
  --dataCenterId "EU-NL-1" \
  --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" \
  --containerDiskSize 50 \
  --volumeSize 50 \
  --ports "8000/http" \
  --startSSH
```

Example remote-from-laptop run:

```bash
ils run \
  --scenario bench/scenarios/modality.yaml \
  --base-url https://<POD_ID>-8000.proxy.runpod.net/v1 \
  --client-mode remote \
  --region us \
  --gpu-type "NVIDIA H200" \
  --gpu-count 1 \
  --tp-size 1 \
  --output results/raw/modality_remote.jsonl
```

Current status:

- OpenAI-compatible stateless benchmarking is implemented
- local vs remote comparisons are supported
- stateful-server benchmarking is not implemented yet

## Non-Core Appendix

Provider comparisons such as frontier APIs vs self-hosted open-source models may still be included later, but only as a product-facing appendix. They should not be mixed into the core causal claims because provider internals, model sizes, and deployment stacks are not controlled.
