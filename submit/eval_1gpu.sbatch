#!/usr/bin/env bash
#SBATCH --job-name=pp_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/admin/home/ryan.kim/nanopath/base-models/nanopath-jepa/.claude/worktrees/wf_f58f62b7-d44-2/slurm/%j.out
#SBATCH --error=/admin/home/ryan.kim/nanopath/base-models/nanopath-jepa/.claude/worktrees/wf_f58f62b7-d44-2/slurm/%j.err
#SBATCH --partition=n
#SBATCH --qos=normal
#SBATCH --time=8:00:00

set -euo pipefail

WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKTREE"

# Resolve venv (same logic as train_1gpu.sbatch)
VENV_DIR="$WORKTREE/.venv"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  VENV_DIR="$HOME/nanopath/.venv"
fi
source "$VENV_DIR/bin/activate"

export WANDB_MODE="disabled"
echo "=== launch_full_preprobe: eval-only probe ==="
echo "PROBE_REQUEST=/tmp/preprobe-panel.json/probe_request.json"
echo "OUTPUT_DIR=/tmp/preprobe-panel.json"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"

python probe.py "/tmp/preprobe-panel.json/probe_request.json"

echo "=== probe done ==="
