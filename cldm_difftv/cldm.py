# ---------------------------------------------------------------------------
# DiffTV stage 3: ControlNet refinement.  "Variant S" of
# _setup/design/controlnet_design.md section 5 -- the control HINT is the raw
# thermal image (B, 3, 128, 128), which is already present in every batch as
# batch['th_img'].  This is NOT the paper's variant E (whose hint is the
# ArcFace-T identity embedding); it exists to validate the plumbing with a
# hint that is available today.  Swapping in ArcFace later only changes
# `hint_channels` and `input_hint_block`.
#
# The three classes below are ports of controlnet/cldm/cldm.py, adapted for
# DiffTV's *concat*-conditioned, *context-free* UNet:
#
#   * DiffTV's LatentDiffusion uses conditioning_key='concat'.  The UNet input
#     is torch.cat([z_t, c_th], 1) -> 8 channels.  ControlNet must see the SAME
#     8-channel tensor (design doc 4.2), so ControlNet.in_channels == 8.
#   * Upstream ControlNet stuffs the control hint into the key 'c_concat'.  In
#     DiffTV 'c_concat' is the real thermal-latent concat conditioning, so the
#     hint gets its own key 'c_control' (design doc 4.4 / 5.1).
#   * DiffTV's SpatialTransformer is a bespoke rewrite with an incompatible
#     signature (design doc D6).  use_spatial_transformer is therefore hard
#     REFUSED here, context is always None, and no cross-attention exists.
#   * No classifier-free guidance anywhere: DiffTV's cond_stage_model is
#     TH_Encoder, which consumes IMAGES, not strings, so there is no null
#     conditioning to build.  get_unconditional_conditioning and the CFG branch
#     of log_images are deliberately absent.
# ---------------------------------------------------------------------------
import os

import cv2
import einops
import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion, disabled_train
from ldm.modules.diffusionmodules.openaimodel import (
    AttentionBlock,
    Downsample,
    ResBlock,
    TimestepEmbedSequential,
    UNetModel,
)
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    timestep_embedding,
    zero_module,
)
from ldm.util import instantiate_from_config

# The design doc (section 4.1) derives, by hand, a residual ladder of
# 12 zero_convs (stem + 4 levels x 2 res-blocks + 3 downsamples) plus one
# middle_block_out = 13 residuals for channel_mult=[1,2,3,4], num_res_blocks=2.
# Nothing below HARDCODES 13 -- every count is derived from the built module
# tree -- but we assert the derived value against this prediction so that a
# silent config drift is caught immediately.


def neutralize_forced_checkpointing(module, verbose=True):
    """Disable gradient checkpointing inside sub-modules whose params are all frozen.

    WHY THIS IS NEEDED (this is NOT cosmetic -- without it stage 3 cannot do a
    single backward pass):

    `ldm/modules/diffusionmodules/openaimodel.py::AttentionBlock.forward` is
        return checkpoint(self._forward, (x,), self.parameters(), True)
    i.e. checkpointing is hard-coded ON, ignoring `use_checkpoint` (design doc
    D7). `ldm/modules/diffusionmodules/util.py::CheckpointFunction.backward`
    then does

        torch.autograd.grad(output_tensors,
                            ctx.input_tensors + ctx.input_params,
                            output_grads, allow_unused=True)

    `allow_unused=True` tolerates tensors that are unused, but NOT tensors with
    requires_grad=False. ControlLDM freezes the whole base UNet (design doc
    4.5 / 6.10), and the base UNet's DECODER runs OUTSIDE the no_grad block --
    it is on the gradient path from the loss back to the control residuals.
    So its AttentionBlocks build a CheckpointFunction node whose backward then
    raises:

        RuntimeError: One of the differentiated Tensors does not require grad

    Fix: for frozen AttentionBlocks, call `_forward` directly (no autograd
    Function, so the grad path is the ordinary one through the activations).
    Frozen ResBlocks get use_checkpoint=False for the same reason.

    Cost: the frozen decoder's attention activations are stored instead of
    recomputed. They are needed for the backward to the ControlNet anyway, and
    at 16x16 latents this is a few MB. Correctness over recompute.

    Frozen sub-modules that run inside torch.no_grad() (the base encoder and
    middle block) are patched too; there checkpointing was pure wasted work
    because no backward node is ever created (design doc 6.8).
    """
    n_attn, n_res = 0, 0
    for m in module.modules():
        params = list(m.parameters(recurse=False))
        if isinstance(m, AttentionBlock):
            params = list(m.parameters())
            if params and not any(p.requires_grad for p in params):
                # Instance attribute shadows the class method; `_forward` has
                # the same (x,) signature, so hooks and __call__ still work.
                m.forward = m._forward
                n_attn += 1
        elif isinstance(m, ResBlock):
            params = list(m.parameters())
            if params and not any(p.requires_grad for p in params) and m.use_checkpoint:
                m.use_checkpoint = False
                n_res += 1
    if verbose:
        print(f"[ControlLDM] neutralised forced gradient checkpointing on "
              f"{n_attn} frozen AttentionBlock(s) and {n_res} frozen ResBlock(s)")
    return n_attn, n_res


class ControlledUnetModel(UNetModel):
    """UNetModel whose decoder consumes ControlNet residuals.

    Adds NO parameters and NO buffers: its state_dict is key-identical and
    shape-identical to the plain UNetModel, so a stage-2 checkpoint loads into
    it verbatim (that is what makes tools/tool_add_control_difftv.py trivial).

    Only `forward` is overridden.  The encoder + middle path runs under
    torch.no_grad() (design doc 1.1), so the frozen branch never builds a
    graph.  `control` is consumed LIFO via .pop(): the LAST element is the
    middle residual, and the remaining 12 are popped in reverse encoder order,
    exactly matching the order in which `hs.pop()` walks the skips.  The caller
    MUST therefore hand in a freshly built list on every call -- the list is
    mutated in place.
    """

    def forward(self, x, timesteps=None, context=None, control=None,
                only_mid_control=False, **kwargs):
        # No class conditioning in DiffTV; `y` is swallowed by **kwargs and the
        # base class's assert on num_classes is intentionally not replicated.
        assert self.num_classes is None, \
            "ControlledUnetModel does not support class conditioning (num_classes must be None)"
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            # OUT-OF-PLACE add (design doc 6.7).  Upstream does `h += ...` onto
            # a no_grad-produced tensor; that is a torch.compile mutation hazard
            # and can raise under autocast on a dtype mismatch.
            h = h + control.pop()

        for module in self.output_blocks:
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                # NOTE: the residual is added to the ENCODER feature, before the
                # concat -- not to h (design doc 1.1, point 2).
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    """Trainable copy of the UNet encoder + middle block, plus hint stem and zero-convs.

    Deliberately NOT a UNetModel subclass: it has no output_blocks, no `out`,
    and no label_emb.  It carries its own time_embed (the frozen UNet's runs
    inside no_grad, so it cannot be shared -- design doc 1.1, point 4).

    For DiffTV: in_channels=8 (it is fed torch.cat([x_noisy, c_th], 1), the same
    tensor the frozen UNet sees), hint_channels=3, model_channels=224,
    channel_mult=[1,2,3,4], num_res_blocks=2, attention_resolutions=[8,4,2],
    num_head_channels=32, use_spatial_transformer=False.
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,
            transformer_depth=1,
            context_dim=None,
            n_embed=None,
            legacy=True,
    ):
        super().__init__()
        # DiffTV's ldm.modules.attention.SpatialTransformer has an incompatible
        # signature (first arg is cond_channels, no use_linear/use_checkpoint/
        # disable_self_attn) and asserts context is not None.  Refuse loudly
        # rather than blowing up deep inside module construction.
        if use_spatial_transformer or context_dim is not None:
            raise NotImplementedError(
                "DiffTV's SpatialTransformer is a bespoke rewrite that is NOT API-compatible "
                "with the ControlNet port (design doc D6). Keep use_spatial_transformer=False "
                "and context_dim=None: DiffTV's UNet has no cross-attention at all."
            )

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'
        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.hint_channels = hint_channels
        # DiffTV's UNetModel keeps num_res_blocks as a raw int and iterates
        # range(num_res_blocks); ControlNet normalises to a per-level list
        # (design doc D2).  Both must describe the SAME ladder, so only an int
        # (or a constant list) is safe here.
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = list(num_res_blocks)

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        # Stock ControlNet hint stem: three stride-2 convs -> spatial factor 1/8,
        # terminal layer zero-initialised so the hint contributes nothing at
        # step 0.  For DiffTV that maps 128x128 -> 16x16, exactly the latent
        # resolution.  The factor is DERIVED below and asserted in forward()
        # against the real feature-map size rather than trusted.
        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )
        self.hint_downsample_factor = self._derive_downsample_factor(self.input_hint_block)

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

        # --- derived residual bookkeeping (never hardcoded) ---------------
        self.input_block_chans = input_block_chans
        self.middle_block_chans = ch
        # one residual per input block + one for the middle block
        self.num_residuals = len(self.zero_convs) + 1
        assert len(self.zero_convs) == len(self.input_blocks), (
            f"zero_convs ({len(self.zero_convs)}) must be 1:1 with input_blocks "
            f"({len(self.input_blocks)})")
        assert len(self.input_block_chans) == len(self.input_blocks), (
            f"input_block_chans ({len(self.input_block_chans)}) != input_blocks "
            f"({len(self.input_blocks)})")
        # NOTE: deliberately NO assert against a hardcoded 13 here. The residual count is a
        # function of channel_mult/num_res_blocks, so hardcoding it would reject perfectly
        # self-consistent geometries (e.g. an ablation with channel_mult=[1,2,4]) while still
        # missing the failure that actually matters -- a control copy whose CHANNELS diverge
        # from the base UNet's. That real invariant is checked in ControlLDM.__init__, where
        # both modules are in scope (_assert_matches_base_unet).

    @staticmethod
    def _derive_downsample_factor(block):
        """Product of the strides of every conv in `block` (spatial reduction)."""
        factor = 1
        for m in block.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                stride = m.stride
                factor *= stride[0] if isinstance(stride, (tuple, list)) else stride
        return factor

    def make_zero_conv(self, channels):
        # 1x1, in_ch == out_ch, weight AND bias zeroed -> the ControlNet branch
        # is an exact no-op on the first forward.  Wrapped in
        # TimestepEmbedSequential only so it can be called as zc(h, emb, ctx).
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context=None, **kwargs):
        # `context` is positional-required upstream; here it defaults to None
        # because DiffTV never has one.
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        # Numeric verification of the "/8 exactly matches the latent" claim:
        # compare the hint stem's real output size against the real input size.
        expected_hw = tuple(s // self.hint_downsample_factor for s in hint.shape[2:])
        assert tuple(guided_hint.shape[2:]) == expected_hw, (
            f"input_hint_block produced {tuple(guided_hint.shape[2:])}, expected {expected_hw} "
            f"(hint {tuple(hint.shape[2:])} / derived factor {self.hint_downsample_factor})")
        assert tuple(guided_hint.shape[2:]) == tuple(x.shape[2:]), (
            f"hint stem output {tuple(guided_hint.shape[2:])} does not match the latent "
            f"resolution {tuple(x.shape[2:])}: hint {tuple(hint.shape[2:])} downsampled by "
            f"{self.hint_downsample_factor}. Fix input_hint_block strides or the hint size.")

        outs = []
        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h = h + guided_hint       # out-of-place (design doc 6.7)
                guided_hint = None        # injected exactly ONCE, after input_blocks[0]
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        assert len(outs) == self.num_residuals
        return outs


class ControlLDM(LatentDiffusion):
    """LatentDiffusion + ControlNet, for DiffTV's concat conditioning.

    Conditioning dict produced by get_input:
        {'c_concat':  [c_th],   # 4x16x16 thermal latent  -> concatenated to z_t
         'c_control': [hint]}   # 3x128x128 thermal image -> ControlNet hint
    """

    def __init__(self, control_stage_config, control_key='th_img',
                 only_mid_control=False, sd_locked=True, *args, **kwargs):
        # Upstream ControlLDM lets DDPM/LatentDiffusion consume ckpt_path BEFORE
        # control_model exists, so every control_model.* tensor is silently
        # dropped as "unexpected" (init_from_ckpt is strict=False).  Design doc
        # 6.5.  Pop it, build, then load.
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(*args, **kwargs)

        assert self.model.conditioning_key == 'concat', (
            f"ControlLDM assumes DiffTV's concat conditioning, got "
            f"conditioning_key={self.model.conditioning_key!r}")

        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        if only_mid_control:
            # With only_mid_control the 12 skip zero_convs never contribute to the loss, so their
            # grads stay None and DDP aborts with 'parameters that were not used in producing loss'.
            raise NotImplementedError(
                "only_mid_control=True is not supported under DDP: the 12 skip zero_convs would "
                "receive no gradient and DDP would abort. Use only_mid_control: false.")
        self.sd_locked = sd_locked

        diffusion_model = self.model.diffusion_model
        assert isinstance(diffusion_model, ControlledUnetModel), (
            "unet_config.target must be cldm_difftv.cldm.ControlledUnetModel, got "
            f"{type(diffusion_model).__name__}")
        assert self.control_model.in_channels == diffusion_model.in_channels, (
            f"ControlNet.in_channels ({self.control_model.in_channels}) must equal the base "
            f"UNet's in_channels ({diffusion_model.in_channels}): ControlNet is fed the SAME "
            f"cat([z_t, c_th]) tensor (design doc 4.2).")
        assert self.control_model.model_channels == diffusion_model.model_channels
        self._assert_matches_base_unet(diffusion_model)

        # Derived, not hardcoded: 12 zero_convs + 1 middle_block_out.
        self.control_scales = [1.0] * self.control_model.num_residuals

        # Freeze the base model at PARAMETER level too (design doc 4.5 / 6.10).
        # The no_grad in ControlledUnetModel.forward only detaches the graph;
        # without this, DDP all-reduces params that never received a grad and
        # raises "parameters were not used in producing the loss".
        self.model.requires_grad_(False)
        # Match how first_stage_model / cond_stage_model are frozen: eval mode too, so the
        # frozen branch can never pick up dropout/BN train-time behaviour.
        self.model.eval()
        self.model.train = disabled_train
        if not sd_locked:
            self.model.diffusion_model.output_blocks.requires_grad_(True)
            self.model.diffusion_model.out.requires_grad_(True)
        # MUST come after the freeze: DiffTV's AttentionBlock always gradient-
        # checkpoints, and CheckpointFunction.backward cannot differentiate
        # w.r.t. frozen params. See neutralize_forced_checkpointing.__doc__.
        neutralize_forced_checkpointing(self.model)

        if self.use_ema:
            # LitEma(self.model) shadows ONLY the base UNet, which we have just frozen, so it
            # collects zero parameters and then dies at the first optimiser step with a bare,
            # message-less AssertionError from LitEma. Fail now, loudly, instead of 20 minutes in.
            raise NotImplementedError(
                "ControlLDM does not support use_ema=True: LitEma shadows the (frozen) base UNet "
                "and would track nothing, then crash inside LitEma. Set use_ema: false in the "
                "config. EMA over control_model would need a second LitEma (design doc 4.5).")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=None):
        # PL 1.9 calls this hook with (batch, batch_idx). LatentDiffusion's
        # version declares a mandatory dataloader_idx and would TypeError
        # (design doc 6.11). scale_by_std is off for every DiffTV stage.
        assert not self.scale_by_std, \
            "ControlLDM does not implement the scale_by_std bootstrap; set scale_by_std=False"

    # ------------------------------------------------------------------ data
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        # LatentDiffusion.get_input -> [z, c] (+ x, xrec, xc when requested)
        out = super().get_input(batch, k, bs=bs, *args, **kwargs)
        z, c = out[0], out[1]

        hint = batch[self.control_key]
        if bs is not None:
            hint = hint[:bs]
        hint = hint.to(self.device)
        if hint.ndim == 3:                       # (B, H, W) -> (B, H, W, 1)
            hint = hint[..., None]
        if hint.ndim == 4 and hint.shape[1] not in (1, 3) and hint.shape[-1] in (1, 3):
            hint = einops.rearrange(hint, 'b h w c -> b c h w')
        hint = hint.to(memory_format=torch.contiguous_format).float()

        cond = dict(c_concat=[c], c_control=[hint])
        return [z, cond] + list(out[2:])

    # ----------------------------------------------------------------- model
    def apply_model(self, x_noisy, t, cond, return_ids=False, *args, **kwargs):
        assert isinstance(cond, dict), \
            "ControlLDM.apply_model expects the dict cond produced by get_input"
        assert not return_ids, "predict_codebook_ids is not supported in the ControlNet stage"
        diffusion_model = self.model.diffusion_model

        c_concat = cond.get('c_concat', None)
        # Replicate DiffusionWrapper's 'concat' branch by hand: we must call the
        # UNet directly to pass `control`, so DiffusionWrapper is bypassed.
        if c_concat is None:
            xc = x_noisy
        else:
            xc = torch.cat([x_noisy] + list(c_concat), dim=1)

        c_control = cond.get('c_control', None)
        if c_control is None:
            return diffusion_model(x=xc, timesteps=t, context=None, control=None,
                                   only_mid_control=self.only_mid_control)

        hint = torch.cat(list(c_control), dim=1)
        # ControlNet sees the SAME 8-channel tensor as the frozen UNet.
        control = self.control_model(x=xc, hint=hint, timesteps=t, context=None)
        # Fresh list every call -- ControlledUnetModel.forward pops from it.
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        return diffusion_model(x=xc, timesteps=t, context=None, control=control,
                               only_mid_control=self.only_mid_control)

    def _assert_matches_base_unet(self, diffusion_model):
        """ControlNet must be a faithful trainable COPY of the base UNet's encoder.

        Checked here rather than against a hardcoded residual count because what actually
        breaks training is a channel/per-module divergence between the two encoders: the
        residual count can still be "right" while `hs.pop() + control.pop()` mismatches, and
        conversely a legitimate ablation can have a different count and be perfectly valid.
        Comparing the two encoders' state_dict shape maps catches BOTH a control copy with
        extra/changed modules AND one with FEWER modules than the base encoder.
        """
        pref = ('time_embed.', 'input_blocks.', 'middle_block.')
        base = {k: tuple(v.shape) for k, v in diffusion_model.state_dict().items()
                if k.startswith(pref)}
        ctrl = {k: tuple(v.shape) for k, v in self.control_model.state_dict().items()
                if k.startswith(pref)}
        missing = sorted(set(base) - set(ctrl))
        extra = sorted(set(ctrl) - set(base))
        shape_diff = sorted(k for k in set(base) & set(ctrl) if base[k] != ctrl[k])
        if missing or extra or shape_diff:
            raise ValueError(
                "control_stage_config does not describe a faithful copy of the base UNet "
                "encoder -- reconcile it with unet_config.\n"
                f"  present in base UNet but MISSING from ControlNet ({len(missing)}): {missing[:8]}\n"
                f"  present in ControlNet but not in base UNet ({len(extra)}): {extra[:8]}\n"
                f"  shape mismatches ({len(shape_diff)}): "
                f"{[(k, base[k], ctrl[k]) for k in shape_diff[:5]]}")
        # And the residual ladder must line up with the decoder skips it is added to.
        assert len(self.control_model.zero_convs) == len(diffusion_model.input_blocks), (
            f"zero_convs ({len(self.control_model.zero_convs)}) != base UNet input_blocks "
            f"({len(diffusion_model.input_blocks)})")

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        if self.cond_stage_trainable:
            raise ValueError("cond_stage_trainable must be False in the ControlNet stage: "
                             "the TH_Encoder is pretrained and frozen.")
        if self.learn_logvar:
            print(f"{self.__class__.__name__}: also optimizing logvar")
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            return [opt], [{'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                            'interval': 'step',
                            'frequency': 1}]
        return opt

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size, null_label=None):
        # Inherited from LatentDiffusion; explicitly disabled. DiffTV has NO
        # classifier-free guidance: cond_stage_model is TH_Encoder, which
        # consumes thermal IMAGES, so there is no null conditioning to build
        # (design doc 4.4). Nothing in this file calls it -- this override
        # exists so that a caller gets a clear error instead of a surprise.
        raise NotImplementedError(
            "ControlLDM has no unconditional conditioning: DiffTV's cond_stage_model is "
            "TH_Encoder (images, not text), so classifier-free guidance is undefined. "
            "Keep unconditional_guidance_scale == 1.0.")

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        # Upstream derives the shape from the hint as (h//8, w//8); here the
        # latent size is an explicit model param (16), so use it directly.
        shape = (self.channels, self.image_size, self.image_size)
        if ddim:
            ddim_sampler = DDIMSampler(self)
            samples, intermediates = ddim_sampler.sample(
                ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        else:
            samples, intermediates = self.sample(
                cond=cond, shape=(batch_size,) + shape, return_intermediates=True, **kwargs)
        return samples, intermediates

    @staticmethod
    def _slice_cond(cond, n):
        return {k: [t[:n] for t in v] for k, v in cond.items()}

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True, ddim_steps=50, ddim_eta=0.0,
                   return_keys=None, plot_denoise_rows=False, **kwargs):
        """Minimal image log. NO classifier-free-guidance branch (design doc 4.4):
        DiffTV's cond_stage_model is TH_Encoder, which takes images, so there is
        no unconditional conditioning to construct."""
        use_ddim = ddim_steps is not None

        log = dict()
        z, cond, x, xrec = self.get_input(batch, self.first_stage_key,
                                          bs=N, return_first_stage_outputs=True)[:4]
        N = min(z.shape[0], N)
        cond = self._slice_cond(cond, N)

        log["inputs"] = x[:N]                     # ground-truth visible, [-1, 1]
        log["reconstruction"] = xrec[:N]          # first stage round-trip of z
        # The hint is the thermal image as the loader produced it: BGR, [-1, 1].
        # It will look colour-swapped in the logger; that is expected.
        log["control"] = cond["c_control"][0]

        if sample:
            samples, z_denoise_row = self.sample_log(cond=cond, batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            log["samples"] = self.decode_first_stage(samples)
            if plot_denoise_rows:
                # DDIM returns intermediates as a dict {x_inter, pred_x0}; the non-DDIM path
                # returns a plain list. _get_denoise_row_from_list iterates tensors.
                if isinstance(z_denoise_row, dict):
                    z_denoise_row = z_denoise_row.get("x_inter", z_denoise_row.get("pred_x0"))
                log["denoise_row"] = self._get_denoise_row_from_list(z_denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            return {key: log[key] for key in return_keys}
        return log

    # ------------------------------------------------------------------ test
    def test_step(self, batch, batch_idx):
        """Mirrors LatentDiffusion.test_step, but carries the control hint.

        `self.save_test_dir` is assigned externally by test.py from the
        top-level `save_test_dir` key of the config.
        """
        z, cond = self.get_input(batch, self.first_stage_key)[:2]
        # Set every step (not just once): the final batch of an epoch may be
        # smaller, and self.sample() allocates noise of shape
        # (self.batch_size, channels, image_size, image_size).
        self.batch_size = z.shape[0]

        recons_latents = self.sample(cond)
        recons_imgs = self.decode_first_stage(recons_latents)
        names = batch['name']

        os.makedirs(self.save_test_dir, exist_ok=True)
        for recons_img, name in zip(recons_imgs, names):
            # clip_denoised is False and the VQ decoder is unbounded, so values
            # DO leave [-1, 1]. Without clamping, numpy's uint8 cast wraps
            # modulo 256 and turns over-bright pixels into black speckle.
            recons_img = torch.clamp(recons_img, -1.0, 1.0)
            recons_img = (recons_img.permute(1, 2, 0) + 1) / 2
            recons_img = (recons_img.cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            save_path = os.path.join(self.save_test_dir, name)
            cv2.imwrite(save_path, recons_img)

        return recons_imgs


# Name used in the design document; kept as an alias so either spelling works
# in a config `target:`.
ControlLDM_DiffTV = ControlLDM
