#!/usr/bin/env bash
# Shared settings for every DiffTV stage. Override anything from your shell:
#   DATA_ROOT=/path/to/your/data GPUS="0,1" bash scripts/train_stage1_vis.sh
#
# DATA_ROOT must contain: train/TH train/VIS test/TH test/VIS  (128x128 PNGs, TH and VIS
# paired by IDENTICAL filename). Build it from the raw SpeakingFaces split with
#   python scripts/prepare_data.py --src /path/to/raw --dst data/thermal2visible_speakingfaces
# or point DATA_ROOT at examples/thermal2visible_mini to smoke-test the pipeline.
DATA_ROOT="${DATA_ROOT:-data/thermal2visible_speakingfaces}"
GPUS="${GPUS:-0}"                 # comma-separated GPU ids, e.g. "0,1,2"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOGDIR="${LOGDIR:-logs}"

NGPU=$(awk -F',' '{print NF}' <<< "$GPUS")

# Some clusters preload a vendor NCCL plugin built against a different NCCL version than
# the one bundled with torch; the first collective then aborts with no Python traceback.
# Single-node training does not need it.
if [ "${DIFFTV_DISABLE_NCCL_PLUGIN:-1}" = "1" ]; then
  for v in $(env | grep -o '^NCCL_FASTRAK[A-Z_]*' || true); do unset "$v"; done
  unset NCCL_NET_PLUGIN NCCL_TUNER_PLUGIN NCCL_TUNER_CONFIG_PATH 2>/dev/null || true
  export NCCL_NET_PLUGIN=none
fi
export PYTHONUNBUFFERED=1

data_overrides () {   # $1 = "train"|"both"
  echo "data.params.num_workers=${NUM_WORKERS}" \
       "data.params.train.params.th_dir=${DATA_ROOT}/train/TH" \
       "data.params.train.params.vis_dir=${DATA_ROOT}/train/VIS" \
       "data.params.validation.params.th_dir=${DATA_ROOT}/test/TH" \
       "data.params.validation.params.vis_dir=${DATA_ROOT}/test/VIS" \
       "data.params.test.params.th_dir=${DATA_ROOT}/test/TH" \
       "data.params.test.params.vis_dir=${DATA_ROOT}/test/VIS"
}
recon_overrides () {  # $1 = VIS|TH   (stage-1 autoencoders see ONE modality)
  echo "data.params.num_workers=${NUM_WORKERS}" \
       "data.params.train.params.data_dir=${DATA_ROOT}/train/$1" \
       "data.params.validation.params.data_dir=${DATA_ROOT}/test/$1"
}
# Single-GPU runs should not need NCCL at all. Only request DDP when there is
# actually more than one device -- otherwise a machine with a broken NCCL/NVML
# setup fails at the first collective with an opaque "unhandled system error".
strategy_override () {
  [ "$NGPU" -gt 1 ] && echo "lightning.trainer.strategy=ddp"
}

latest_ckpt () {      # newest run matching $1 that ACTUALLY has a checkpoint
  # test.py names its log dir after the CONFIG file, so running inference creates a
  # second logs/<ts>_stage2_ldm with no checkpoints/. Taking the newest directory
  # blindly would pick that empty one and report "train stage 2 first". Skip dirs
  # that contain no last.ckpt.
  local d
  for d in $(ls -1d "${LOGDIR}"/*"$1" 2>/dev/null | sort -r); do
    if [ -f "$d/checkpoints/last.ckpt" ]; then echo "$d/checkpoints/last.ckpt"; return 0; fi
  done
  return 1
}
