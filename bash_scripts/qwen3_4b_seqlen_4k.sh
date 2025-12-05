#!/bin/bash

#SBATCH --job-name=sokoban-rl-run-qwen3_4b_seqlen_4k
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=20:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:a100l:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yash.more@mila.quebec

source .venv/bin/activate
uv run rl --trainer @ configs/sokoban/train.toml --orchestrator @ configs/sokoban/orch.toml --inference @ configs/sokoban/infer.toml --wandb.project sokoban-rl-run --wandb.name sokoban-rl-run-$(date +%Y%m%d%H%M%S) --trainer.model.optimization-dtype bfloat16 --trainer.model.reduce-dtype bfloat16 --trainer.model.name "Qwen/Qwen3-4B" --orchestrator.seq-len 4096 --orchestrator.sampling.max-tokens 2048