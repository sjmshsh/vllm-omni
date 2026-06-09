#!/usr/bin/env bash
# Qwen3-Omni random-mm benchmark: audio input → text+audio output
# 4 cases: short/long input × short/long output
set -euo pipefail

HOST="${BENCH_HOST:-localhost}"
PORT="${BENCH_PORT:-8091}"
MODEL="${BENCH_MODEL:-/home/admin/model/}"
NUM_PROMPTS="${BENCH_NUM_PROMPTS:-128}"
MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-16}"
RESULT_DIR="${BENCH_RESULT_DIR:-./results/qwen3_omni_random_mm}"

mkdir -p "$RESULT_DIR"

COMMON_ARGS=(
  --omni
  --host "$HOST"
  --port "$PORT"
  --model "$MODEL"
  --endpoint /v1/chat/completions
  --backend openai-chat-omni
  --dataset-name random-mm
  --num-prompts "$NUM_PROMPTS"
  --max-concurrency "$MAX_CONCURRENCY"
  --random-mm-base-items-per-request 1
  --random-mm-limit-mm-per-prompt '{"image":0,"video":0,"audio":1}'
  --ignore-eos
  --extra-body '{"modalities":["text","audio"]}'
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf
  --save-result
  --result-dir "$RESULT_DIR"
)

echo "============================================================"
echo "Case 1/4: 短输入 + 短输出 (audio 3s + 200 tokens → 150 tokens)"
echo "============================================================"
vllm bench serve "${COMMON_ARGS[@]}" \
  --random-input-len 200 \
  --random-output-len 150 \
  --random-mm-bucket-config '{"(0, 3, 1)": 1.0}' \
  --result-filename short_in_short_out.json

echo ""
echo "============================================================"
echo "Case 2/4: 短输入 + 长输出 (audio 3s + 200 tokens → 1000 tokens)"
echo "============================================================"
vllm bench serve "${COMMON_ARGS[@]}" \
  --random-input-len 200 \
  --random-output-len 1000 \
  --random-mm-bucket-config '{"(0, 3, 1)": 1.0}' \
  --result-filename short_in_long_out.json

echo ""
echo "============================================================"
echo "Case 3/4: 长输入 + 短输出 (audio 15s + 2000 tokens → 150 tokens)"
echo "============================================================"
vllm bench serve "${COMMON_ARGS[@]}" \
  --random-input-len 2000 \
  --random-output-len 150 \
  --random-mm-bucket-config '{"(0, 15, 1)": 1.0}' \
  --result-filename long_in_short_out.json

echo ""
echo "============================================================"
echo "Case 4/4: 长输入 + 长输出 (audio 15s + 2000 tokens → 1000 tokens)"
echo "============================================================"
vllm bench serve "${COMMON_ARGS[@]}" \
  --random-input-len 2000 \
  --random-output-len 1000 \
  --random-mm-bucket-config '{"(0, 15, 1)": 1.0}' \
  --result-filename long_in_long_out.json

echo ""
echo "============================================================"
echo "ALL DONE. Results saved to: $RESULT_DIR/"
echo "============================================================"
ls -lh "$RESULT_DIR"/*.json 2>/dev/null || true
