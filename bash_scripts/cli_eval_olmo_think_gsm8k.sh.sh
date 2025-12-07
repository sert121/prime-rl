#!/bin/bash

#SBATCH --job-name=eval_allen_7B_think_gsm8k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:a100l:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yash.more@mila.quebec

source .venv/bin/activate
uv run inference --model.name "allenai/Olmo-3-7B-Think" &
# wait for 30s
sleep 320
vf-eval gsm8k -m "allenai/Olmo-3-7B-Think" --api-key-var "" --api-base-url http://localhost:8000/v1 -n 10 --rollouts-per-example 1319 --max-concurrent 32 -s --save-to-hf-hub --hf-hub-dataset-name sert121/olmo3-think-gsm8k-rl