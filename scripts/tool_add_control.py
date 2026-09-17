#!/usr/bin/env python
# ---------------------------------------------------------------------------
# Initialise a DiffTV stage-3 ControlLDM checkpoint from a trained stage-2
# LatentDiffusion checkpoint.  Implements _setup/design/controlnet_design.md
# section 5.4.
#
# THE MAPPING RULE, for every key `k` of the freshly-built ControlLDM:
#     k.startswith('control_model.')  ->  source = 'model.diffusion_model.' + k[len('control_model.'):]
#     otherwise                       ->  source = k
# and if the source is absent from the stage-2 checkpoint, keep the scratch
# (freshly-initialised) tensor.
#
# Consequences:
#   * model.diffusion_model.*  is copied UNCHANGED (ControlledUnetModel has the
#     same module tree as UNetModel, so the keys line up 1:1).
#   * control_model.{time_embed,input_blocks,middle_block}.*  are seeded from
#     the UNet ENCODER weights -- this is what makes ControlNet converge fast.
#   * control_model.{zero_convs,middle_block_out}.*  stay at their ZERO init.
#   * control_model.input_hint_block.*  stays at its random init (except its
#     terminal conv, which is zero-initialised).
#   * first_stage_model.*, cond_stage_model.*, logvar and the diffusion
#     schedule buffers are copied unchanged.
#
# Upstream tool_add_control.py expresses the same rule obliquely (strip
# 'control_', prepend 'model.diffusion_'), which misfires on any other key
# starting with 'control_'.  This version is explicit and FAILS LOUDLY on any
# shape mismatch instead of silently keeping scratch weights.
#
# Usage:
#   python tools/tool_add_control_difftv.py \
#       --config configs/difftv/cldm_difftv.yaml \
#       --input  logs/<stage2>/checkpoints/last.ckpt \
#       --output models/difftv_control_ini.ckpt
# ---------------------------------------------------------------------------
import argparse
import os
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldm.util import instantiate_from_config  # noqa: E402

CONTROL_PREFIX = 'control_model.'
BASE_UNET_PREFIX = 'model.diffusion_model.'

# Only these control_model sub-trees are ALLOWED to be freshly initialised.
# Anything else appearing as "new" means control_stage_config disagrees with
# unet_config -- that assertion is the whole value of this tool.
ALLOWED_FRESH_PREFIXES = (
    CONTROL_PREFIX + 'zero_convs.',
    CONTROL_PREFIX + 'middle_block_out.',
    CONTROL_PREFIX + 'input_hint_block.',
)


def source_key_for(target_key):
    """Map a ControlLDM state_dict key to its stage-2 LatentDiffusion source key."""
    if target_key.startswith(CONTROL_PREFIX):
        return BASE_UNET_PREFIX + target_key[len(CONTROL_PREFIX):]
    return target_key


def build_ema_redirect(model):
    """Map 'model.<name>' -> 'model_ema.<name with dots stripped>' (LitEma naming)."""
    redirect = {}
    for name, _ in model.model.named_parameters():
        redirect['model.' + name] = 'model_ema.' + name.replace('.', '')
    return redirect


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True,
                    help='stage-3 config, e.g. configs/difftv/cldm_difftv.yaml')
    ap.add_argument('--input', required=True,
                    help='trained stage-2 LatentDiffusion checkpoint')
    ap.add_argument('--output', required=True,
                    help='ControlLDM checkpoint to write')
    ap.add_argument('--use_ema', action='store_true',
                    help='seed both branches from model_ema.* instead of model.*')
    ap.add_argument('--overwrite', action='store_true',
                    help='allow overwriting --output')
    a = ap.parse_args()

    if not os.path.exists(a.input):
        raise SystemExit(f'[FATAL] --input does not exist: {a.input}')
    if os.path.exists(a.output) and not a.overwrite:
        raise SystemExit(f'[FATAL] --output already exists (pass --overwrite): {a.output}')

    cfg = OmegaConf.load(a.config)
    # Build clean: the stage-3 ckpt_path is what we are about to CREATE.
    if 'ckpt_path' in cfg.model.params:
        cfg.model.params.pop('ckpt_path')
    print(f'[INFO] instantiating {cfg.model.target} on CPU ...')
    model = instantiate_from_config(cfg.model).cpu()

    ck = torch.load(a.input, map_location='cpu', weights_only=False)
    pre = ck['state_dict'] if 'state_dict' in ck else ck
    print(f'[INFO] loaded stage-2 checkpoint: {len(pre)} tensors from {a.input}')

    ema = build_ema_redirect(model) if a.use_ema else {}
    if a.use_ema:
        print(f'[INFO] --use_ema: redirecting {len(ema)} model.* keys to model_ema.*')

    scratch = model.state_dict()
    target = {}
    copied, fresh, mismatched = [], [], []

    for k, v in scratch.items():
        src = source_key_for(k)
        src = ema.get(src, src)
        if src in pre:
            if tuple(pre[src].shape) == tuple(v.shape):
                target[k] = pre[src].clone()
                copied.append((k, src))
                continue
            mismatched.append((k, tuple(pre[src].shape), tuple(v.shape)))
        target[k] = v.clone()
        fresh.append(k)

    if mismatched:
        print('\n[FATAL] shape mismatches between the stage-2 checkpoint and the stage-3 model:')
        for k, s_shape, t_shape in mismatched:
            print(f'    {k}\n        ckpt  {s_shape}\n        model {t_shape}')
        raise SystemExit(
            '[FATAL] Refusing to write a checkpoint with silently-dropped weights. '
            'Reconcile the stage-3 config with the stage-2 config and re-run.')

    model.load_state_dict(target, strict=True)

    # --- report ---------------------------------------------------------
    copied_control = [k for k, _ in copied if k.startswith(CONTROL_PREFIX)]
    copied_unet = [k for k, _ in copied if k.startswith(BASE_UNET_PREFIX)]
    copied_other = [k for k, _ in copied
                    if not k.startswith(CONTROL_PREFIX) and not k.startswith(BASE_UNET_PREFIX)]

    print('\n================ COPIED ================')
    print(f'  {len(copied_unet):5d}  model.diffusion_model.*   (frozen base UNet, verbatim)')
    print(f'  {len(copied_control):5d}  control_model.*           (seeded from the UNet encoder)')
    print(f'  {len(copied_other):5d}  other                     (first/cond stage, buffers, logvar)')
    print(f'  {len(copied):5d}  TOTAL copied')
    print('\n  control_model.* keys seeded from the base UNet:')
    for k in copied_control:
        print(f'    {k}  <-  {source_key_for(k)}')

    print('\n============ RANDOM / ZERO INIT ============')
    print(f'  {len(fresh):5d}  TOTAL newly initialised')
    for k in fresh:
        print(f'    new: {k}')

    bad = [k for k in fresh if not k.startswith(ALLOWED_FRESH_PREFIXES)]
    if bad:
        print('\n[FATAL] these keys were NOT found in the stage-2 checkpoint but should have been:')
        for k in bad:
            print(f'    {k}   (looked for {source_key_for(k)})')
        raise SystemExit(
            '[FATAL] control_stage_config disagrees with unet_config (or the stage-2 '
            'checkpoint is not the model this config describes). Stop and fix the config.')

    # --- invariants ------------------------------------------------------
    n_zero_params = 0
    for n, p in model.control_model.named_parameters():
        if n.startswith('zero_convs.') or n.startswith('middle_block_out.'):
            assert p.abs().max().item() == 0.0, f'[FATAL] {n} is not zero after init!'
            n_zero_params += 1
    n_res = model.control_model.num_residuals
    assert n_res == len(model.control_scales), \
        f'[FATAL] control_scales has {len(model.control_scales)} entries, model has {n_res} residuals'
    print(f'\n[OK] {n_zero_params} zero-conv tensors verified exactly zero')
    print(f'[OK] {len(model.control_model.zero_convs)} zero_convs + 1 middle_block_out '
          f'= {n_res} control residuals')

    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or '.', exist_ok=True)
    # Saved under a 'state_dict' key so DDPM.init_from_ckpt can read it.
    torch.save({'state_dict': model.state_dict()}, a.output)
    print(f'\n[OK] wrote {a.output}')


if __name__ == '__main__':
    main()
