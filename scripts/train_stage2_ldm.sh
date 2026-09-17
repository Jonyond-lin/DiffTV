#!/usr/bin/env bash
# Stage 2 - thermal-conditioned latent diffusion. Both stage-1 models stay frozen.
set -e
cd "$(dirname "$0")/.."
source scripts/env.sh
VIS_CKPT="${VIS_CKPT:-$(latest_ckpt stage1_vqvae_vis)}"
TH_CKPT="${TH_CKPT:-$(latest_ckpt stage1_vqvae_th)}"
[ -f "$VIS_CKPT" ] || { echo "run scripts/train_stage1_vis.sh first (or set VIS_CKPT)"; exit 1; }
[ -f "$TH_CKPT"  ] || { echo "run scripts/train_stage1_th.sh first (or set TH_CKPT)";  exit 1; }
mkdir -p configs/_resolved
sed -e "s|__STAGE1A_VIS_CKPT__|$VIS_CKPT|" -e "s|__STAGE1B_TH_CKPT__|$TH_CKPT|" \
    configs/stage2_ldm.yaml > configs/_resolved/stage2_ldm.yaml
echo "[stage2] first_stage=$VIS_CKPT"
echo "[stage2] cond_stage =$TH_CKPT"
python main.py --base configs/_resolved/stage2_ldm.yaml --train True --no-test True -n stage2_ldm \
  lightning.trainer.gpus="\"$GPUS\"" $(strategy_override) $(data_overrides) "$@"
