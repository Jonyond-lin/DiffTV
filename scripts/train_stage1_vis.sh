#!/usr/bin/env bash
# Stage 1a - VQ-VAE on VISIBLE faces. Becomes the frozen first stage of the LDM.
set -e
cd "$(dirname "$0")/.."
source scripts/env.sh
echo "[stage1-vis] DATA_ROOT=$DATA_ROOT GPUS=$GPUS (${NGPU} gpu)"
python main.py --base configs/stage1_vqvae_vis.yaml --train True --no-test True -n stage1_vqvae_vis \
  lightning.trainer.gpus="\"$GPUS\"" $(strategy_override) $(recon_overrides VIS) "$@"
