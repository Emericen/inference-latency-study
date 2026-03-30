MODEL ?= Qwen/Qwen3-VL-8B-Instruct
PORT ?= 8000
TENSOR_PARALLEL ?= 1
LOG_DIR ?= results/logs
LOG_FILE ?= $(LOG_DIR)/vllm.log
PID_FILE ?= $(LOG_DIR)/vllm.pid
PYTHON ?= python3
UV ?= uv
VENV_PYTHON ?= .venv/bin/python
VLLM ?= .venv/bin/vllm

.PHONY: ensure-venv vllm-up vllm-down vllm-logs

ensure-venv:
	@command -v "$(UV)" >/dev/null 2>&1 || $(PYTHON) -m pip install uv
	@test -x "$(VENV_PYTHON)" || "$(UV)" venv
	@"$(UV)" pip install --python "$(VENV_PYTHON)" vllm

vllm-up: ensure-venv
	mkdir -p $(LOG_DIR)
	@if [ -f $(PID_FILE) ] && kill -0 "$$(cat $(PID_FILE))" 2>/dev/null; then \
		echo "vLLM already running: $$(cat $(PID_FILE))"; \
	else \
		nohup "$(VLLM)" serve "$(MODEL)" \
			--host 0.0.0.0 \
			--port "$(PORT)" \
			--served-model-name "$(MODEL)" \
			--enable-prefix-caching \
			--tensor-parallel-size $(TENSOR_PARALLEL) \
			--trust-remote-code \
			> "$(LOG_FILE)" 2>&1 < /dev/null & echo $$! > "$(PID_FILE)"; \
		echo "started vLLM pid $$(cat $(PID_FILE))"; \
	fi

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
