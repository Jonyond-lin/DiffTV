#!/usr/bin/env bash
# End-to-end: stage 1a -> 1b -> 2 -> 3, then inference + metrics.
set -e
cd "$(dirname "$0")/.."
bash scripts/train_stage1_vis.sh
bash scripts/train_stage1_th.sh
bash scripts/train_stage2_ldm.sh
STAGE=2 bash scripts/test.sh
bash scripts/train_stage3_controlnet.sh
STAGE=3 bash scripts/test.sh
