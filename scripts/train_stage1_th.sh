#!/usr/bin/env bash
# Stage 1b - VQ-VAE on THERMAL faces. Its ENCODER becomes the frozen conditional encoder.
set -e
cd "$(dirname "$0")/.."
source scripts/env.sh
echo "[stage1-th] DATA_ROOT=$DATA_ROOT GPUS=$GPUS (${NGPU} gpu)"
python main.py --base configs/stage1_vqvae_th.yaml --train True --no-test True -n stage1_vqvae_th \
  lightning.trainer.gpus="\"$GPUS\"" $(strategy_override) $(recon_overrides TH) "$@"
