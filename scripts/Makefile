MODEL ?= Qwen/Qwen3-VL-8B-Instruct
HOST ?= 0.0.0.0
PORT ?= 8000
BASE_URL ?= http://127.0.0.1:$(PORT)/v1
ROOT ?= ..
PYTHON ?= $(ROOT)/.venv/bin/python
VLLM ?= $(ROOT)/.venv/bin/vllm
LOG_DIR ?= $(ROOT)/results/logs
LOG_FILE ?= $(LOG_DIR)/vllm.log
PID_FILE ?= $(LOG_DIR)/vllm.pid
VLLM_EXTRA_ARGS ?=

.PHONY: prepare vllm-up vllm-wait vllm-down vllm-logs run-buckets aggregate-buckets

prepare:
	$(PYTHON) $(ROOT)/data/prepare.py

vllm-up:
	mkdir -p $(LOG_DIR)
	@if [ -f $(PID_FILE) ] && kill -0 "$$(cat $(PID_FILE))" 2>/dev/null; then \
		echo "vLLM already running: $$(cat $(PID_FILE))"; \
	else \
		nohup $(VLLM) serve "$(MODEL)" --host "$(HOST)" --port "$(PORT)" --served-model-name "$(MODEL)" --trust-remote-code $(VLLM_EXTRA_ARGS) > "$(LOG_FILE)" 2>&1 < /dev/null & echo $$! > "$(PID_FILE)"; \
		echo "started vLLM pid $$(cat $(PID_FILE))"; \
	fi

vllm-wait:
	$(PYTHON) $(ROOT)/scripts/wait_for_vllm.py --base-url "$(BASE_URL)" --timeout-s 900

vllm-down:
	@if [ -f $(PID_FILE) ]; then \
		kill "$$(cat $(PID_FILE))" 2>/dev/null || true; \
		rm -f "$(PID_FILE)"; \
		echo "stopped vLLM"; \
	else \
		echo "no vLLM pid file"; \
	fi

vllm-logs:
	tail -f "$(LOG_FILE)"

run-buckets:
	$(PYTHON) $(ROOT)/scripts/run.py \
	  --config $(ROOT)/configs/screenspot_payload_buckets.yaml \
	  --base-url "$(BASE_URL)" \
	  --output $(ROOT)/results/raw/screenspot_payload_buckets.jsonl

aggregate-buckets:
	$(PYTHON) $(ROOT)/scripts/aggregate.py \
	  --input $(ROOT)/results/raw/screenspot_payload_buckets.jsonl \
	  --output $(ROOT)/results/summaries/screenspot_payload_buckets.csv
