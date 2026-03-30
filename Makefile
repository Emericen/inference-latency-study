MODEL ?= Qwen/Qwen3-VL-8B-Instruct
HOST ?= 0.0.0.0
PORT ?= 8000
LOG_DIR ?= results/logs
LOG_FILE ?= $(LOG_DIR)/vllm.log
PID_FILE ?= $(LOG_DIR)/vllm.pid
VLLM_EXTRA_ARGS ?=

.PHONY: vllm-up vllm-down vllm-logs

vllm-up:
	mkdir -p $(LOG_DIR)
	@if [ -f $(PID_FILE) ] && kill -0 "$$(cat $(PID_FILE))" 2>/dev/null; then \
		echo "vLLM already running: $$(cat $(PID_FILE))"; \
	else \
		nohup vllm serve "$(MODEL)" --host "$(HOST)" --port "$(PORT)" --served-model-name "$(MODEL)" --trust-remote-code $(VLLM_EXTRA_ARGS) > "$(LOG_FILE)" 2>&1 < /dev/null & echo $$! > "$(PID_FILE)"; \
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
