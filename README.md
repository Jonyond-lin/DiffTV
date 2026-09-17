# DiffTV — Thermal-to-Visible Face Translation with Latent Diffusion

Implementation of **DiffTV: Identity-Preserved Thermal-to-Visible Face Translation via Feature Alignment and Dual-Stage Conditions**, ACM Multimedia 2024.

Jingyu Lin, Guiqin Zhao, Jing Xu, Guoli Wang, Zejin Wang, Antitza Dantcheva, Lan Du, Cunjian Chen

---

## Pipeline

| Stage | What it trains | Output |
|-------|----------------|--------|
| **1a** | VQ-VAE on **visible** faces | first stage (encoder + decoder) of the LDM, then frozen |
| **1b** | VQ-VAE on **thermal** faces | its **encoder** becomes the frozen conditional thermal encoder |
| **2**  | Latent-diffusion UNet, thermal-conditioned | the translation model |
| **3**  | ControlNet refinement on top of stage 2 | control branch only; base UNet stays frozen |

Images are `128 x 128`. The autoencoder downsamples by **f = 8**, so the diffusion UNet operates on a `4 x 16 x 16` latent. Conditioning is **concatenation**: the thermal latent is concatenated to the noisy latent, giving the UNet `in_channels = 8` and `out_channels = 4`. There is no cross-attention and no text encoder anywhere in this model.

---

## Install

```bash
conda create -n difftv python=3.10 -y
conda activate difftv

# install torch matching your CUDA first
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Tested on PyTorch 2.4.1 / CUDA 12.1 / PyTorch-Lightning 1.9.5, Python 3.10, on NVIDIA H100.

Stage 1 downloads the LPIPS perceptual-loss head (~7 KB) on first run, so it needs one-time internet access.

---

## Data

Expected layout — thermal and visible images **paired by identical filename**:

```
<DATA_ROOT>/
├── train/TH/  xxx.png      # thermal,  128x128
├── train/VIS/ xxx.png      # visible,  128x128   (same filename as its TH pair)
├── test/TH/   yyy.png
└── test/VIS/  yyy.png
```

This repo ships a tiny sample at `examples/thermal2visible_mini` (64 train + 10 test pairs)
purely so you can verify the pipeline runs end to end.

To build the real dataset from the raw SpeakingFaces thermal/visible split
(`trainA`/`trainB`/`testA`/`testB`, where `A` = thermal and `B` = visible):

```bash
python scripts/prepare_data.py \
  --src path/to/thermal2visible_speakingfaces \
  --dst data/thermal2visible_speakingfaces \
  --size 128
```

The raw crops are **not** uniformly sized, so this rescaling step is required, not optional. `prepare_data.py` also strips the modality suffix so each pair shares one basename, verifies every TH file has a VIS partner, and reports how many sources were upscaled.

---

## Training

Every stage is one command. Point `DATA_ROOT` at **your own** data directory.

```bash
export DATA_ROOT=path/to/your/data       # or: examples/thermal2visible_mini
export GPUS=0                            # "0,1,2" for multi-GPU (DDP is used only when >1)

bash scripts/train_stage1_vis.sh         # stage 1a  visible VQ-VAE
bash scripts/train_stage1_th.sh          # stage 1b  thermal VQ-VAE
bash scripts/train_stage2_ldm.sh         # stage 2   latent diffusion
bash scripts/train_stage3_controlnet.sh  # stage 3   ControlNet refinement
```

or the whole pipeline plus evaluation:

```bash
DATA_ROOT=path/to/your/data bash scripts/run_all.sh
```

Each stage auto-discovers the previous stage's checkpoint from `logs/`. Override explicitly with `VIS_CKPT=`, `TH_CKPT=`, `LDM_CKPT=` if you want a specific one. Any extra argument is passed straight through as an OmegaConf override, e.g.:

```bash
bash scripts/train_stage2_ldm.sh lightning.trainer.max_epochs=50 data.params.batch_size=21
```

---

## Inference and metrics

```bash
STAGE=2 bash scripts/test.sh     # stage-2 LDM
STAGE=3 bash scripts/test.sh     # stage-3 ControlNet
```

> **`BATCH` must divide the test-set size exactly.** `test_step` caches the first batch's size and reuses it to allocate the noise tensor, so a short final batch raises a shape error. The default `BATCH=42` divides 2268 (the SpeakingFaces test split) exactly; for the bundled 10-pair example use `BATCH=10`.

---

## Repository layout

```
configs/      one YAML per stage
scripts/      one-click training / inference shell scripts + data prep + metrics + tooling
ldm/          latent-diffusion model code (autoencoder, UNet, DDPM/DDIM, data loaders)
cldm_difftv/  ControlNet for DiffTV: ControlledUnetModel, ControlNet, ControlLDM
examples/     tiny paired sample so the pipeline can be run immediately
main.py       training entry point (all stages)
test.py       inference entry point
```

---

## Citation

```bibtex
@inproceedings{lin2024difftv,
  title     = {DiffTV: Identity-Preserved Thermal-to-Visible Face Translation via
               Feature Alignment and Dual-Stage Conditions},
  author    = {Lin, Jingyu and Zhao, Guiqin and Xu, Jing and Wang, Guoli and Wang, Zejin and
               Dantcheva, Antitza and Du, Lan and Chen, Cunjian},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia (MM '24)},
  year      = {2024},
  doi       = {10.1145/3664647.3680635}
}
```

---

## Acknowledgements and licence

The model code is derived from [latent-diffusion / Stable Diffusion](https://github.com/CompVis/stable-diffusion) by Robin Rombach, Patrick Esser and contributors, and is distributed under the **CreativeML Open RAIL-M** licence included in [`LICENSE`](LICENSE). The ControlNet design follows [ControlNet](https://github.com/lllyasviel/ControlNet) (Zhang et al.).

The sample images under `examples/` come from the [**SpeakingFaces**](https://github.com/IS2AI/SpeakingFaces) dataset (Abdrakhmanova et al., *Sensors* 2021). They are included only as a functional smoke test. Please obtain the full dataset from the official source and follow its licence and terms of use; do not treat this sample as a redistribution of the dataset.
