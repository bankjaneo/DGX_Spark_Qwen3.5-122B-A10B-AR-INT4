#!/bin/bash
# v2-tq: v2 + TurboQuant KV Cache Compression
# Result: TBD tok/s, but 4x more KV cache (1.4M tokens)
# Context: 200K on dual DGX Spark
# Tensor Parallel: 2 (dual DGX Spark)

docker run -d --name vllm-qwen35-tq \
  --gpus all --net=host --ipc=host \
  -v /path/to/models:/models \
  vllm-qwen35-v019-v2-tq \
  serve /models/qwen35-397b-hybrid-int4fp8 \
  --served-model-name qwen \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --kv-cache-dtype turboquant35 --enable-turboquant \
  --turboquant-metadata-path /models/qwen35-397b-hybrid-int4fp8/turboquant_kv.json \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
