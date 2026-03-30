# Inference Latency Study

Controlled benchmark for measuring inference latency in screenshot-based computer-use VLM agents.

This repo answers one question:

**What are the main latency bottlenecks in computer-use multimodal inference, and how do model size, payload growth, and deployment distance change that latency?**

Specifically, this repo asks:

1. How does inference latency scale with model size within the same dense `Qwen3-VL` family?
   Compare `2B`, `4B`, `8B`, and `32B` dense models under the same hardware, payload, and server settings.
2. How does a MoE model compare to a dense model at roughly the same scale?
   Compare `Qwen3-VL-30B-A3B` against `Qwen3-VL-32B` under the same conditions.
3. How does latency grow as context size grows?
   Measure latency as the number of text+image pairs in a single `user` message increases from `1` to `100`.
4. How much of that growth comes from resending past images versus preserving only textual state?
   Compare full-history requests against requests where prior images are replaced with a text placeholder such as `"image omitted"`.
5. How much does network distance matter when only the current image is sent?
   Compare a nearby remote server against a farther remote server when prior images are omitted and only one image is included per request.
6. How much does network distance matter when multimodal history is fully resent each turn?
   Compare a nearby remote server against a farther remote server as image history grows from `1` to `100` images.
7. How do hosted providers compare in practice?
   Run the same product-facing payloads against selected providers on OpenRouter as a practical appendix, separate from the controlled `Qwen3-VL` study.

This repo studies application-layer decisions for screenshot-based VLM agents: how latency changes as context grows, how much of that cost comes from resending images, how deployment distance affects the result, and how those tradeoffs move across model choices and providers.

## What It Measures

- Time to first token (`TTFT`)
- Total completion latency

`decode_tps` may still be logged as a secondary diagnostic, but the primary metrics in this study are `TTFT` and total latency.

## Repo Layout

```text
Makefile                  Server-side: start/stop vLLM
requirements.txt          Python dependencies
data/
  screenshots/            Fixed screenshot corpus used by the study
study/
  capture.py              Optional screenshot capture utility
  run.py                  Run one benchmark case from a YAML config
  aggregate.py            Re-aggregate a JSONL file into CSV
  configs/                Experiment configs
```

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

On the GPU node, start vLLM:

```bash
make vllm-up
```

This bootstraps a repo-local `.venv/` if needed and uses that environment's `vllm` binary.

To watch startup logs:

```bash
make vllm-logs
```

To use a different backend model:

```bash
make vllm-up MODEL=Qwen/Qwen3-VL-4B-Instruct
```

The client config should use the same real model name that the server is serving.

## Running One Case

Each YAML file in `study/configs/` describes one benchmark case. `study/run.py` reads the config, runs the requests, writes raw JSONL, and writes a CSV summary automatically at the end.

One context unit means:

- one text content block
- one screenshot content block
- both placed inside the same single OpenAI `user` message

So `context_max_size: 4` means the study will warm the server first, then measure context sizes `1`, `2`, `3`, and `4` on one live server, where each measured request is one `user` message containing that many text+image pairs unless the context mode omits past screenshots.

The repo currently includes:

- `4` dense-scale sweeps
- `2` dense-vs-MoE sweeps
- `5` context-growth sweeps
- provider appendix sweeps under `study/configs/providers/`

Example local run:

```bash
python study/run.py --config study/configs/dense_scale/qwen3_vl_8b_local.yaml
```

Example remote run with a config override:

```bash
python study/run.py \
  --config study/configs/history_sweeps/full_remote_near.yaml \
  --base-url https://<POD_ID>-8000.proxy.runpod.net/v1
```

Example remote run with both model and base URL override:

```bash
python study/run.py \
  --config study/configs/history_sweeps/omit_past_remote_far.yaml \
  --base-url https://<POD_ID>-8000.proxy.runpod.net/v1 \
  --model Qwen/Qwen3-VL-8B-Instruct
```

Useful config fields:

- `experiment`
- `model`
- `base_url`
- `region`
- `context_mode`
- `context_max_size`
- `warmup_size`
- `max_tokens`
- `output_path`

Current context modes:

- `full_history`: every context unit keeps both its text block and its screenshot block
- `omit_past_history`: only the latest context unit keeps its screenshot block; earlier screenshot blocks are replaced with the text placeholder `"[image omitted]"`

## Methodology

Every experiment in this repo does the same thing:

1. start a live server
2. warm it up with `warmup_size` requests
3. measure a context sweep from size `1` through `context_max_size`
4. write raw JSONL and a CSV summary

The sweep is the experiment. We compare that same sweep across:

- dense model scales
- dense versus MoE
- local versus remote-near versus remote-far
- full-history versus omitted-history context modes
- different providers

Within a sweep, do **not** restart the server between increments. The study intentionally assumes prefix-cache reuse during context growth, because that matches real-world screenshot-agent usage.

Between independent sweeps, restart the server manually:

```bash
make vllm-down
make vllm-up
```

## Optional: Capture More Screenshots

The repo already includes a fixed screenshot corpus under `data/screenshots/`, so capture is not required for normal use.

If you want to collect more screenshots:

```bash
python study/capture.py
```

Left-click saves a screenshot. Right-click stops.

## Optional: Re-Aggregate A Run

`study/run.py` already writes a summary CSV automatically. `study/aggregate.py` only exists if you want to re-aggregate an old JSONL file.

```bash
python study/aggregate.py \
  --input results/raw/history_sweep_full_remote_near.jsonl
```

## RunPod

Create a pod:

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

The vLLM API will be exposed at:

```text
https://<POD_ID>-8000.proxy.runpod.net/v1
```

Useful commands:

```bash
runpodctl get pod
runpodctl ssh connect <POD_ID>
runpodctl stop pod <POD_ID>
runpodctl remove pod <POD_ID>
```

## What Still Needs To Change

The config matrix is now in place. The remaining work is to run it cleanly and turn the outputs into findings:

- run the dense scale sweeps: `2B`, `4B`, `8B`, `32B`
- run the dense vs MoE sweeps: `32B` vs `30B-A3B`
- run local, remote-near, and remote-far sweeps with manual cold restarts between them
- run the full-history and omitted-history sweeps
- run the OpenRouter provider appendix sweeps
- summarize the resulting sweep curves into tables and plots for the README
