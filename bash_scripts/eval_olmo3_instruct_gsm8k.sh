#!/bin/bash

#SBATCH --job-name=eval_qwen_0.6B_gsm8k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=3:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:a100l:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yash.more@mila.quebec

source .venv/bin/activate
uv run inference --model.name "allenai/Olmo-3-7B-Instruct" &
# wait for 30s
sleep 600
python3 /home/mila/y/yash.more/scratch/sokoban_experiments/evals/gsm8k/gsm8k_qwen.py --model "allenai/Olmo-3-7B-Instruct"