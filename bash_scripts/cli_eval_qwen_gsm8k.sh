#!/bin/bash

#SBATCH --job-name=eval_qwen_8B_gsm8k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:a100l:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yash.more@mila.quebec

source .venv/bin/activate
uv run inference --model.name "Qwen/Qwen3-8B" &
# wait for 30s
sleep 120
vf-eval gsm8k -m "Qwen/Qwen3-8B" --api-key-var "" --api-base-url http://localhost:8000/v1 -n 1319 --rollouts-per-example 1  --max-concurrent 32 -s --save-to-hf-hub --hf-hub-dataset-name sert121/qwen3-8b-gsm8k-rl