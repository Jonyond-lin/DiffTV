#!/usr/bin/env bash
# Stage 3 - ControlNet refinement. Base UNet frozen; only the control branch trains.
set -e
cd "$(dirname "$0")/.."
source scripts/env.sh
VIS_CKPT="${VIS_CKPT:-$(latest_ckpt stage1_vqvae_vis)}"
TH_CKPT="${TH_CKPT:-$(latest_ckpt stage1_vqvae_th)}"
LDM_CKPT="${LDM_CKPT:-$(latest_ckpt stage2_ldm)}"
[ -f "$LDM_CKPT" ] || { echo "run scripts/train_stage2_ldm.sh first (or set LDM_CKPT)"; exit 1; }
mkdir -p configs/_resolved outputs/stage3_controlnet
INIT=outputs/stage3_controlnet/cldm_init.ckpt
sed -e "s|__STAGE1A_VIS_CKPT__|$VIS_CKPT|" -e "s|__STAGE1B_TH_CKPT__|$TH_CKPT|" \
    -e "s|__CLDM_INIT_CKPT__|$INIT|" configs/stage3_controlnet.yaml > configs/_resolved/stage3_controlnet.yaml
# A training job rewrites last.ckpt periodically; copy before reading so we never get a torn file.
SNAP=outputs/stage3_controlnet/_ldm_snapshot.ckpt
cp "$LDM_CKPT" "$SNAP"
echo "[stage3] initialising ControlNet from $LDM_CKPT"
python scripts/tool_add_control.py --config configs/_resolved/stage3_controlnet.yaml \
       --input "$SNAP" --output "$INIT" --overwrite
python main.py --base configs/_resolved/stage3_controlnet.yaml --train True --no-test True -n stage3_controlnet \
  lightning.trainer.gpus="\"$GPUS\"" $(strategy_override) $(data_overrides) "$@"
