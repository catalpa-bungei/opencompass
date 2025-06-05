#!/bin/bash
conda activate opencompass-vllm
# qwen2.5-vl-7B
vllm serve /fs-computility/ai-shen/shared/hf-hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/5b5eecc7efc2c3e86839993f2689bbbdf06bd8d4 --dtype half --port 8007 --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --limit_mm_per_prompt image=10 --max_model_len 20000 --served-model-name Qwen2.5-VL-7B-Instruct

# qwen2.5-vl-32B
export CUDA_VISIBLE_DEVICES=2,3
vllm serve /fs-computility/ai-shen/shared/hf-hub/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/7cfb30d71a1f4f49a57592323337a4a4727301da --dtype half --port 8032 --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --limit_mm_per_prompt image=10 --max_model_len 20000 --served-model-name Qwen2.5-VL-32B-Instruct

# qwen2.5-vl-72B
export CUDA_VISIBLE_DEVICES=4,5,6,7
vllm serve /fs-computility/ai-shen/shared/hf-hub/models--Qwen--Qwen2.5-VL-72B-Instruct/snapshots/5d8e171e5ee60e8ca4c6daa380bd29f78fe19021 --dtype half --port 8072 --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --limit_mm_per_prompt image=10 --max_model_len 20000 --served-model-name Qwen2.5-VL-72B-Instruct

# qwen3-32B
export CUDA_VISIBLE_DEVICES=2,3
vllm serve /fs-computility/ai-shen/shared/hf-hub/Qwen3-32B/ --dtype half --port 8332 --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max_model_len 20000 --served-model-name Qwen3-32B
# qwen3-32B-non-thinking
vllm serve /fs-computility/ai-shen/shared/hf-hub/Qwen3-32B/ --dtype half --port 8432 --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max_model_len 20000 --served-model-name Qwen3-32B-non-thinking