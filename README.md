# Inference Latency Study

Small practical benchmark for screenshot-based computer-use agents.

This repo asks a simple question:

**Among the latency levers we can actually pull in a product, which one matters most?**

The clearest answer from the runs in this repo is:

- keeping screenshot history small matters the most
- smaller models help
- closer servers help

In this benchmark, omitting past screenshots and keeping only the latest screenshot in the request kept latency much flatter than resending full screenshot history every turn.

This repo measures latency, not quality. The model response is discarded. It does not answer how many screenshots should be kept for best agent performance.

## Quick Start

On the GPU node:

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Start the server manually in the foreground. Example:

```bash
.venv/bin/vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

Then run a case:

```bash
python study/run.py --config study/configs/dense_scale/qwen3_vl_8b_local.yaml
```

Or run from another machine against the RunPod proxy URL:

```bash
python study/run.py \
  --config study/configs/screenshot_history/full_us_ca_2.yaml \
  --base-url https://<POD_ID>-8000.proxy.runpod.net/v1
```

## What The Benchmark Measures

Primary metrics:

- time to first token (`TTFT`)
- total completion latency

One request is a single OpenAI `user` message containing repeated content blocks.

One context unit means:

- one fixed text block
- one `1080p` screenshot block

So a context size of `100` means one request with `100` text blocks and `100` screenshots inside the same `user` message.

Context always grows cumulatively by `1` in this repo.

Current context modes:

- `full_history`: 
  Every past screenshot stays in the request.
- `omit_past_history`: 
  Only the latest screenshot stays in the request. Older screenshots are replaced with the text placeholder `"[image omitted]"`.

Region names used in this repo:

- `local`
  The client script runs on the same machine as the `vllm` server and talks to `localhost`.
- `us-ca-2`
  The client script runs from Mountain View and talks to a RunPod server in `US-CA-2`.
- `us-mo-1`
  The client script runs from Mountain View and talks to a RunPod server in `US-MO-1`, which is in Missouri.

## What I Ran

Completed runs:

- local dense scale on `1x H100`: `Qwen/Qwen3-VL-2B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen3-VL-8B-Instruct`
- local dense vs MoE on `2x H100`: `Qwen/Qwen3-VL-32B-Instruct` and `Qwen/Qwen3-VL-30B-A3B-Instruct`
- local screenshot history on `Qwen/Qwen3-VL-8B-Instruct`
- `US-CA-2` screenshot history on `Qwen/Qwen3-VL-8B-Instruct`
- `US-MO-1` screenshot history on `Qwen/Qwen3-VL-8B-Instruct`

We did not run the provider comparison appendix.

## Findings

Reducing the number of screenshots included per request is the biggest lever affecting latency growth.

Second order effects are:

- model size
- distance to the server

### Local Dense Scale

![Dense scale local TTFT](assets/plots/dense_scale_local_ttft.svg)

This was run on the GPU node itself, requesting `localhost`.

- On the same `1x H100`, local TTFT at context size `100` is about `0.863s` for `Qwen/Qwen3-VL-2B-Instruct`, `1.107s` for `Qwen/Qwen3-VL-4B-Instruct`, and `1.169s` for `Qwen/Qwen3-VL-8B-Instruct`.
- Model scale matters, but within `2B -> 8B` it matters less than resending screenshot history.

### Local 32B Dense vs 30B-A3B MoE

![Dense vs MoE local TTFT](assets/plots/dense_vs_moe_local_ttft.svg)

This was run on `2x H100` with tensor parallel size `2`.

- `Qwen/Qwen3-VL-30B-A3B-Instruct` is faster than `Qwen/Qwen3-VL-32B-Instruct`, but only modestly.
- At context size `100`, local TTFT is about `7.419s` for `30B-A3B` and `7.933s` for `32B dense`.

### Full Screenshot History by Region

![Full screenshot history by region](assets/plots/full_screenshot_history_by_region_ttft.svg)

- Resending every past screenshot is the steepest latency driver in the repo.
- At context size `100`, TTFT is about `1.170s` for `local`, `2.089s` for `us-ca-2`, and `2.340s` for `us-mo-1`.
- The remote full-history runs also showed a few large spikes.

### Omit Past Screenshot History by Region

![Omit past screenshot history by region](assets/plots/omit_past_screenshot_history_by_region_ttft.svg)

- Replacing old screenshots with text placeholders keeps the latency curve much flatter.
- At context size `100`, TTFT is about `0.262s` for `local`, `0.405s` for `us-ca-2`, and `0.723s` for `us-mo-1`.
- Relative to full screenshot history at context size `100`, omitted history reduces TTFT to about `22%` for `local`, `19%` for `us-ca-2`, and `31%` for `us-mo-1`.

## Data And Outputs

Input screenshots:

- [data/screenshots](/Users/eddyliang/Desktop/workfile/inference-latency-study/data/screenshots)

Raw outputs:

- [results/raw](/Users/eddyliang/Desktop/workfile/inference-latency-study/results/raw)

Summary CSVs:

- [results/summaries](/Users/eddyliang/Desktop/workfile/inference-latency-study/results/summaries)

Generated plot assets:

- [assets/plots](/Users/eddyliang/Desktop/workfile/inference-latency-study/assets/plots)

The main generated files for the screenshot-history runs are:

- [screenshot_history_full_local.jsonl](/Users/eddyliang/Desktop/workfile/inference-latency-study/results/raw/screenshot_history_full_local.jsonl)
- [screenshot_history_omit_past_local.jsonl](/Users/eddyliang/Desktop/workfile/inference-latency-study/results/raw/screenshot_history_omit_past_local.jsonl)
- [screenshot_history_full_us_ca_2.jsonl](/Users/eddyliang/Desktop/workfile/inference-latency-study/results/raw/screenshot_history_full_us_ca_2.jsonl)
- [screenshot_history_omit_past_us_ca_2.jsonl](/Users/eddyliang/Desktop/workfile/inference-latency-study/results/raw/screenshot_history_omit_past_us_ca_2.jsonl)
- [screenshot_history_full_us_mo_1.jsonl](/Users/eddyliang/Desktop/workfile/inference-latency-study/results/raw/screenshot_history_full_us_mo_1.jsonl)
- [screenshot_history_omit_past_us_mo_1.jsonl](/Users/eddyliang/Desktop/workfile/inference-latency-study/results/raw/screenshot_history_omit_past_us_mo_1.jsonl)

Their summary CSVs are:

- [screenshot_history_full_local.csv](/Users/eddyliang/Desktop/workfile/inference-latency-study/results/summaries/screenshot_history_full_local.csv)
- [screenshot_history_omit_past_local.csv](/Users/eddyliang/Desktop/workfile/inference-latency-study/results/summaries/screenshot_history_omit_past_local.csv)
- [screenshot_history_full_us_ca_2.csv](/Users/eddyliang/Desktop/workfile/inference-latency-study/results/summaries/screenshot_history_full_us_ca_2.csv)
- [screenshot_history_omit_past_us_ca_2.csv](/Users/eddyliang/Desktop/workfile/inference-latency-study/results/summaries/screenshot_history_omit_past_us_ca_2.csv)
- [screenshot_history_full_us_mo_1.csv](/Users/eddyliang/Desktop/workfile/inference-latency-study/results/summaries/screenshot_history_full_us_mo_1.csv)
- [screenshot_history_omit_past_us_mo_1.csv](/Users/eddyliang/Desktop/workfile/inference-latency-study/results/summaries/screenshot_history_omit_past_us_mo_1.csv)

## Exact Server Commands Used

`1x H100`, `Qwen/Qwen3-VL-2B-Instruct`

```bash
.venv/bin/vllm serve Qwen/Qwen3-VL-2B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen/Qwen3-VL-2B-Instruct \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

`1x H100`, `Qwen/Qwen3-VL-4B-Instruct`

```bash
.venv/bin/vllm serve Qwen/Qwen3-VL-4B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen/Qwen3-VL-4B-Instruct \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

`1x H100`, `Qwen/Qwen3-VL-8B-Instruct`

```bash
.venv/bin/vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

`2x H100`, `Qwen/Qwen3-VL-32B-Instruct`

```bash
.venv/bin/vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen/Qwen3-VL-32B-Instruct \
  --enable-prefix-caching \
  --tensor-parallel-size 2 \
  --trust-remote-code
```

`2x H100`, `Qwen/Qwen3-VL-30B-A3B-Instruct`

```bash
.venv/bin/vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen/Qwen3-VL-30B-A3B-Instruct \
  --enable-prefix-caching \
  --tensor-parallel-size 2 \
  --trust-remote-code
```

## Plot Generation

Regenerate the README plots from the summary CSVs with:

```bash
python study/plot.py
```

## RunPod Notes

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

## Caveats

- This repo measures latency, not quality.
- The response content is thrown away.
- The repo does not answer how many screenshots should be kept for best agent performance.
- Each context size currently has a single measured request, so the curves should be read as observed traces, not strong percentile claims.
