#!/bin/bash

#SBATCH --job-name=eval_gemma_12b_gsm8k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:a100l:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yash.more@mila.quebec

source .venv/bin/activate
uv run inference --model.name "google/gemma-3-12b-it" &
# wait for 30s
sleep 320
vf-eval gsm8k -m "google/gemma-3-12b-it" --api-key-var "" --api-base-url http://localhost:8000/v1 -n 10 --rollouts-per-example 1319 --max-concurrent 32 -s --save-to-hf-hub --hf-hub-dataset-name sert121/gemma-12b-gsm8k-rl