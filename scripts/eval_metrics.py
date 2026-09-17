#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_metrics.py -- standalone evaluation of DiffTV (thermal -> visible face translation).

Computes PSNR / SSIM / LPIPS / FID between a directory of generated images (--pred)
and a directory of ground-truth images (--gt).

--------------------------------------------------------------------------------
CHANNEL ORDER -- READ THIS BEFORE CHANGING ANYTHING
--------------------------------------------------------------------------------
The repo is 100% OpenCV-based and therefore BGR-ordered *in memory*:

  * ldm/data/ldm_t2v.py / ldm/data/reconstruct.py read with ``cv2.imread(path)``,
    which returns an **HWC BGR** uint8/float array. So the tensor that the model
    sees has channel order (B, G, R).
  * ldm/models/diffusion/ddpm.py::test_step writes with ``cv2.imwrite(save_path,
    recons_img)``, and ``cv2.imwrite`` *interprets* its input as BGR.

The round trip is therefore self-consistent: BGR in -> BGR out -> PNG on disk has
correct, natural colours. Both --pred and --gt PNGs on disk are ordinary,
correctly-coloured images. What matters for us is only how we LOAD them:

  ``cv2.imread`` gives BGR, and LPIPS + Inception (FID) both expect **RGB**.

So the correct default is: ``cv2.imread`` then ``cv2.cvtColor(..., COLOR_BGR2RGB)``.
That is what ``--assume-bgr`` (the default) does. ``--no-assume-bgr`` skips the
conversion and is only correct if you produced the files with a non-OpenCV writer
that already baked in a channel swap (i.e. the PNGs are colour-swapped on disk).
The convention actually in use is printed at the top of every run.

PSNR/SSIM on *grayscale* are nearly invariant to an R<->B swap only if the luma
weights were symmetric -- they are not (BT.601: 0.299 R, 0.587 G, 0.114 B), so
getting this wrong measurably changes the numbers. FID and LPIPS are strongly
affected. Do not "simplify" this away.

--------------------------------------------------------------------------------
FILENAME PAIRING
--------------------------------------------------------------------------------
Pairing is by filename (stem), NOT by index. Note that DatasetT2VLDM returns
``'name': th_path.split('/')[-1]`` -- i.e. predictions inherit the *thermal*
filename, which in the SpeakingFaces layout ends in ``_1.png``, while the
ground-truth visible files end in ``_3.png``. Exact stem matching would then find
zero pairs, so if exact matching yields nothing we automatically retry with a
trailing ``_<digits>`` group stripped from both sides (loudly reported).
Use --pred-sub / --gt-sub for full manual control.

--------------------------------------------------------------------------------
METRIC DEFINITIONS
--------------------------------------------------------------------------------
PSNR  : the paper reports "PSNR of the underlying grayscale image". We report
        psnr_gray (primary, BT.601 luma, data_range=255, uint8 domain) and also
        psnr_rgb so both numbers are available. Averaged over pairs.
SSIM  : skimage structural_similarity on grayscale (primary) and on RGB
        (channel_axis=-1), data_range=255.
LPIPS : lpips package, net='alex' by default. Inputs normalised to [-1, 1] float,
        NCHW, RGB order.
FID   : Inception-V3 pool3, 2048 dims. FID is a *distribution* metric, so it is
        computed over ALL readable images in each directory (not only matched
        pairs) -- that is the standard protocol and it is also how the two
        directories having different counts is handled. The per-directory sample
        counts are always printed and stored in the JSON.

  Backend notes: pytorch-fid and torchmetrics agree closely (they share the
  pool3 Inception weights); clean-fid deliberately uses a different (higher
  quality) resizing pipeline, so its values are offset and must not be mixed
  with the other two. clean-fid also wraps its extractor in DataParallel over
  every visible GPU, which stalls in NCCL on multi-GPU nodes -- run it with
  ``CUDA_VISIBLE_DEVICES=0`` if it hangs.

  !! FID CAVEAT !! FID is a biased estimator: its expected value decreases
  monotonically with the number of samples. With ~2268 images (SpeakingFaces
  test split) the estimate carries a substantial positive bias and a
  non-negligible variance. FID values are only comparable against other FID
  values computed with the SAME implementation and the SAME sample count. This
  is why n_pred_fid / n_gt_fid are printed next to the number.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
from collections import OrderedDict, defaultdict

import numpy as np

# cv2 is imported eagerly: it is the *only* supported image reader here, on
# purpose, so that the BGR story above is unambiguous.
import cv2

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm")


# ----------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------
# module name -> distribution name, for packages that expose no __version__
_DIST_NAME = {
    "cv2": "opencv-python-headless",
    "skimage": "scikit-image",
    "lpips": "lpips",
    "pytorch_fid": "pytorch-fid",
    "cleanfid": "clean-fid",
}


def _version(modname, attr="__version__"):
    """Version of an importable module, or None if it is not installed.

    Falls back to importlib.metadata because several of these packages
    (lpips, clean-fid) ship no __version__ attribute at all.
    """
    try:
        mod = __import__(modname)
        for part in modname.split(".")[1:]:
            mod = getattr(mod, part)
    except Exception:
        return None
    v = getattr(mod, attr, None)
    if v:
        return str(v)
    try:
        from importlib.metadata import version as _dist_version
        return str(_dist_version(_DIST_NAME.get(modname, modname)))
    except Exception:
        return "unknown"


def collect_versions():
    v = OrderedDict()
    v["python"] = sys.version.split()[0]
    for name, key in [
        ("numpy", "numpy"),
        ("cv2", "opencv"),
        ("skimage", "scikit-image"),
        ("scipy", "scipy"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("lpips", "lpips"),
        ("pytorch_fid", "pytorch-fid"),
        ("cleanfid", "clean-fid"),
        ("torchmetrics", "torchmetrics"),
    ]:
        v[key] = _version(name)
    return v


def list_images(d):
    """Sorted list of image file paths directly inside directory `d`."""
    if not os.path.isdir(d):
        raise SystemExit("FATAL: not a directory: %s" % d)
    out = [
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.lower().endswith(IMG_EXTS) and os.path.isfile(os.path.join(d, f))
    ]
    out.sort()
    return out


def parse_sub(spec):
    """Parse a '--pred-sub REGEX::REPL' spec into (compiled_regex, repl)."""
    if spec is None:
        return None
    if "::" not in spec:
        raise SystemExit(
            "FATAL: --pred-sub/--gt-sub must look like 'REGEX::REPLACEMENT' "
            "(got %r)" % spec
        )
    pat, repl = spec.split("::", 1)
    return (re.compile(pat), repl)


def make_key(path, sub, strip_digits):
    """Derive the matching key for a file path."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if sub is not None:
        stem = sub[0].sub(sub[1], stem)
    if strip_digits:
        stem = re.sub(r"_\d+$", "", stem)
    return stem


def build_index(paths, sub, strip_digits):
    """Map key -> path, warning about collisions."""
    idx = {}
    dupes = []
    for p in paths:
        k = make_key(p, sub, strip_digits)
        if k in idx:
            dupes.append((k, idx[k], p))
        else:
            idx[k] = p
    return idx, dupes


# ----------------------------------------------------------------------------
# image loading
# ----------------------------------------------------------------------------
def load_rgb(path, assume_bgr):
    """Load `path` as an HWC uint8 RGB array.

    cv2.imread returns BGR (see module docstring). With assume_bgr=True (the
    default, and the correct setting for anything this repo wrote) we convert
    BGR->RGB. With assume_bgr=False we take the bytes as-is, i.e. we claim the
    file is already stored channel-swapped.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # always 3-channel BGR uint8
    if img is None:
        return None
    if assume_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(img)


def resize_to(img, hw):
    """Resize HWC image to (h, w). INTER_AREA when shrinking (the requested
    behaviour, and the right filter for downscaling), INTER_CUBIC when growing."""
    h, w = hw
    if img.shape[0] == h and img.shape[1] == w:
        return img
    shrinking = (h * w) < (img.shape[0] * img.shape[1])
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def to_gray(rgb):
    """BT.601 luma from an RGB uint8 array -> uint8 HW."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


# ----------------------------------------------------------------------------
# LPIPS
# ----------------------------------------------------------------------------
def compute_lpips(pred_list, gt_list, net, device, batch_size):
    import torch
    import lpips as lpips_pkg

    loss_fn = lpips_pkg.LPIPS(net=net, verbose=False).to(device).eval()
    for p in loss_fn.parameters():
        p.requires_grad_(False)

    # Group by shape so we can batch even when --size <= 0 leaves sizes ragged.
    groups = defaultdict(list)
    for i, a in enumerate(pred_list):
        groups[a.shape].append(i)

    vals = np.zeros(len(pred_list), dtype=np.float64)
    with torch.no_grad():
        for shape, idxs in groups.items():
            for s in range(0, len(idxs), batch_size):
                chunk = idxs[s : s + batch_size]
                pb = np.stack([pred_list[i] for i in chunk]).astype(np.float32)
                gb = np.stack([gt_list[i] for i in chunk]).astype(np.float32)
                # uint8 RGB HWC -> float NCHW in [-1, 1], as LPIPS requires.
                pt = torch.from_numpy(pb).permute(0, 3, 1, 2).div_(127.5).sub_(1.0)
                gt = torch.from_numpy(gb).permute(0, 3, 1, 2).div_(127.5).sub_(1.0)
                d = loss_fn(pt.to(device), gt.to(device))
                vals[chunk] = d.detach().flatten().float().cpu().numpy()
    return vals


# ----------------------------------------------------------------------------
# FID backends
# ----------------------------------------------------------------------------
def _fid_stack(arrays, fid_size):
    """Uniform-size uint8 NHWC RGB stack for in-memory FID backends."""
    return np.stack([resize_to(a, (fid_size, fid_size)) for a in arrays])


def fid_pytorch_fid(pred_arrays, gt_arrays, device, batch_size, fid_size, dims=2048):
    """pytorch-fid's own InceptionV3 (pool3, 2048-d) + Frechet distance.

    Computed in-memory rather than by pointing at the directories, so that the
    images going into Inception are byte-identical to the ones used for
    LPIPS/PSNR (same BGR->RGB conversion, same resize). Model weights, feature
    layer and the distance formula are pytorch-fid's.
    """
    import torch
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device).eval()

    def activations(arrays):
        stack = _fid_stack(arrays, fid_size)
        feats = []
        with torch.no_grad():
            for s in range(0, len(stack), batch_size):
                b = stack[s : s + batch_size].astype(np.float32)
                # Inception wants float in [0, 1], NCHW, RGB. resize_input=True
                # (the InceptionV3 default) bilinearly resizes to 299x299.
                t = torch.from_numpy(b).permute(0, 3, 1, 2).div_(255.0).to(device)
                p = model(t)[0]
                if p.size(2) != 1 or p.size(3) != 1:
                    p = torch.nn.functional.adaptive_avg_pool2d(p, output_size=(1, 1))
                feats.append(p.squeeze(3).squeeze(2).cpu().numpy())
        return np.concatenate(feats, axis=0)

    a_pred = activations(pred_arrays)
    a_gt = activations(gt_arrays)
    mu1, s1 = a_pred.mean(axis=0), np.cov(a_pred, rowvar=False)
    mu2, s2 = a_gt.mean(axis=0), np.cov(a_gt, rowvar=False)
    return float(calculate_frechet_distance(mu1, s1, mu2, s2))


def fid_torchmetrics(pred_arrays, gt_arrays, device, batch_size, fid_size):
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance

    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    for arrays, real in ((gt_arrays, True), (pred_arrays, False)):
        stack = _fid_stack(arrays, fid_size)
        for s in range(0, len(stack), batch_size):
            b = stack[s : s + batch_size]
            # torchmetrics with normalize=False wants uint8 NCHW in [0, 255], RGB.
            t = torch.from_numpy(b).permute(0, 3, 1, 2).contiguous().to(device)
            metric.update(t, real=real)
    return float(metric.compute().item())


def fid_clean_fid(pred_arrays, gt_arrays, pred_names, gt_names, device, batch_size,
                  fid_size, workdir):
    """clean-fid only has a directory API, so materialise the *preprocessed*
    images (post BGR->RGB, post resize) into temp dirs and point it at those.

    NOTE: clean-fid builds its feature extractor with nn.DataParallel over all
    visible GPUs; on a multi-GPU node that can stall inside NCCL. Export
    CUDA_VISIBLE_DEVICES=0 before running if this backend hangs.
    """
    import torch
    from cleanfid import fid as cleanfid_fid

    dp = os.path.join(workdir, "cf_pred")
    dg = os.path.join(workdir, "cf_gt")
    for d in (dp, dg):
        os.makedirs(d, exist_ok=True)
    for d, arrays, names in ((dp, pred_arrays, pred_names), (dg, gt_arrays, gt_names)):
        for a, n in zip(arrays, names):
            a = resize_to(a, (fid_size, fid_size))
            # a is RGB; cv2.imwrite expects BGR, so swap back on the way out
            # to produce a correctly-coloured PNG for clean-fid's PIL reader.
            cv2.imwrite(os.path.join(d, os.path.splitext(n)[0] + ".png"),
                        cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    return float(
        cleanfid_fid.compute_fid(dp, dg, mode="clean", batch_size=batch_size,
                                 device=torch.device(device), num_workers=0)
    )


def compute_fid(impl, pred_arrays, gt_arrays, pred_names, gt_names, device,
                batch_size, fid_size, workdir):
    """Returns (fid_value, impl_used, error_or_None)."""
    order = (["pytorch-fid", "torchmetrics", "clean-fid"] if impl == "auto" else [impl])
    last_err = None
    for name in order:
        try:
            if name == "pytorch-fid":
                return fid_pytorch_fid(pred_arrays, gt_arrays, device, batch_size,
                                       fid_size), name, None
            if name == "torchmetrics":
                return fid_torchmetrics(pred_arrays, gt_arrays, device, batch_size,
                                        fid_size), name, None
            if name == "clean-fid":
                return fid_clean_fid(pred_arrays, gt_arrays, pred_names, gt_names,
                                     device, batch_size, fid_size, workdir), name, None
            raise SystemExit("FATAL: unknown --fid-impl %r" % name)
        except Exception as e:  # noqa: BLE001
            last_err = "%s: %s" % (type(e).__name__, e)
            if impl != "auto":
                raise
            print("  [fid] backend %-12s unavailable -> %s" % (name, last_err))
    return None, None, last_err


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="eval_metrics.py",
        description="PSNR / SSIM / LPIPS / FID between a prediction dir and a GT dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred", required=True, help="directory of generated images")
    p.add_argument("--gt", required=True, help="directory of ground-truth images")
    p.add_argument("--size", type=int, default=128,
                   help="evaluate at this square resolution; <=0 keeps the GT's "
                        "native size and only resizes pred to match")
    p.add_argument("--device", default="cuda", help="cuda | cuda:N | cpu")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"])
    p.add_argument("--out", default="results.json", help="JSON output path")
    p.add_argument("--fid-impl", default="auto",
                   choices=["auto", "pytorch-fid", "clean-fid", "torchmetrics"])
    p.add_argument("--fid-size", type=int, default=0,
                   help="square size fed to Inception (0 = use --size, or 299 "
                        "when --size <= 0)")
    p.add_argument("--no-fid", action="store_true", help="skip FID entirely")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--assume-bgr", dest="assume_bgr", action="store_true",
                   default=True,
                   help="files are read with cv2.imread (BGR) and converted to "
                        "RGB. CORRECT for this repo; this is the default.")
    g.add_argument("--no-assume-bgr", dest="assume_bgr", action="store_false",
                   help="skip the BGR->RGB conversion (only for files already "
                        "stored channel-swapped)")

    p.add_argument("--pred-sub", default=None, metavar="REGEX::REPL",
                   help="regex substitution applied to the pred filename stem to "
                        "form the matching key, e.g. '_1$::'")
    p.add_argument("--gt-sub", default=None, metavar="REGEX::REPL",
                   help="same, for GT stems, e.g. '_3$::'")
    p.add_argument("--strip-suffix-digits", default="auto",
                   choices=["auto", "yes", "no"],
                   help="strip a trailing '_<digits>' from both stems before "
                        "matching. 'auto' only does it if exact matching found "
                        "zero pairs (the DiffTV _1/_3 case).")
    return p


# ----------------------------------------------------------------------------
def _json_safe(obj):
    """Recursively replace non-finite floats so the dump is strict RFC-8259 JSON.

    json.dump would happily write bare Infinity/NaN, which json.loads accepts but
    JavaScript's JSON.parse, pandas.read_json and most other readers reject. An
    infinite PSNR is a real, meaningful result (pixel-identical images), so we
    keep it as the string "Infinity" rather than dropping it to null.
    """
    if isinstance(obj, dict):
        return OrderedDict((k, _json_safe(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if np.isinf(f):
            return "Infinity" if f > 0 else "-Infinity"
        if np.isnan(f):
            return "NaN"
        return f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main(argv=None):
    args = build_parser().parse_args(argv)
    versions = collect_versions()

    print("=" * 78)
    print("eval_metrics.py  --  PSNR / SSIM / LPIPS / FID")
    print("=" * 78)
    print("pred dir : %s" % os.path.abspath(args.pred))
    print("gt   dir : %s" % os.path.abspath(args.gt))
    if args.assume_bgr:
        print("channels : --assume-bgr  -> cv2.imread (BGR) then BGR->RGB. "
              "Metrics see RGB. [correct for cv2.imwrite-produced files]")
    else:
        print("channels : --no-assume-bgr -> raw cv2.imread bytes used as if RGB. "
              "*** only correct for channel-swapped files ***")

    # ---- device ------------------------------------------------------------
    import torch  # imported here so --help works without torch

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but torch.cuda.is_available() is False; "
              "falling back to CPU.")
        device = "cpu"
    print("device   : %s" % device)

    # ---- pairing -----------------------------------------------------------
    pred_paths = list_images(args.pred)
    gt_paths = list_images(args.gt)
    if not pred_paths:
        raise SystemExit("FATAL: no images found in --pred %s" % args.pred)
    if not gt_paths:
        raise SystemExit("FATAL: no images found in --gt %s" % args.gt)

    pred_sub = parse_sub(args.pred_sub)
    gt_sub = parse_sub(args.gt_sub)

    strip = args.strip_suffix_digits == "yes"
    p_idx, p_dupes = build_index(pred_paths, pred_sub, strip)
    g_idx, g_dupes = build_index(gt_paths, gt_sub, strip)
    keys = sorted(set(p_idx) & set(g_idx))

    if not keys and args.strip_suffix_digits == "auto":
        print("NOTE: exact filename matching produced 0 pairs; retrying with a "
              "trailing '_<digits>' stripped from both sides (the DiffTV case "
              "where predictions inherit the thermal '_1' name and GT is '_3').")
        strip = True
        p_idx, p_dupes = build_index(pred_paths, pred_sub, strip)
        g_idx, g_dupes = build_index(gt_paths, gt_sub, strip)
        keys = sorted(set(p_idx) & set(g_idx))

    for label, dupes in (("pred", p_dupes), ("gt", g_dupes)):
        for k, a, b in dupes[:10]:
            print("WARNING: duplicate %s key %r -> keeping %s, ignoring %s"
                  % (label, k, os.path.basename(a), os.path.basename(b)))
        if len(dupes) > 10:
            print("WARNING: ... and %d more duplicate %s keys"
                  % (len(dupes) - 10, label))

    only_pred = sorted(set(p_idx) - set(g_idx))
    only_gt = sorted(set(g_idx) - set(p_idx))
    print("-" * 78)
    print("files in pred      : %d" % len(pred_paths))
    print("files in gt        : %d" % len(gt_paths))
    print("matched pairs      : %d   (key = filename stem%s)"
          % (len(keys), ", trailing '_<digits>' stripped" if strip else ""))
    print("unmatched in pred  : %d   (skipped)%s"
          % (len(only_pred),
             "  e.g. " + ", ".join(only_pred[:3]) if only_pred else ""))
    print("unmatched in gt    : %d   (skipped)%s"
          % (len(only_gt),
             "  e.g. " + ", ".join(only_gt[:3]) if only_gt else ""))

    if len(keys) < 1:
        raise SystemExit(
            "FATAL: 0 matched pairs between --pred and --gt.\n"
            "  pred example stems: %s\n"
            "  gt   example stems: %s\n"
            "  Use --pred-sub / --gt-sub (REGEX::REPL) or --strip-suffix-digits "
            "yes to align the naming." % (sorted(p_idx)[:3], sorted(g_idx)[:3])
        )

    # ---- load pairs --------------------------------------------------------
    print("-" * 78)
    print("loading %d pairs ..." % len(keys))
    pred_imgs, gt_imgs, used_keys = [], [], []
    n_resized, n_unreadable, n_size_mismatch = 0, 0, 0
    resize_examples = []
    mismatch_examples = []
    for k in keys:
        pi = load_rgb(p_idx[k], args.assume_bgr)
        gi = load_rgb(g_idx[k], args.assume_bgr)
        if pi is None or gi is None:
            n_unreadable += 1
            continue
        if pi.shape[:2] != gi.shape[:2]:
            n_size_mismatch += 1
            if len(mismatch_examples) < 3:
                mismatch_examples.append(
                    "%s pred%s vs gt%s" % (k, pi.shape[:2], gi.shape[:2]))
        if args.size > 0:
            target = (args.size, args.size)
        else:
            target = (gi.shape[0], gi.shape[1])
        if pi.shape[:2] != target:
            if args.size <= 0:
                n_resized += 1
                if len(resize_examples) < 3:
                    resize_examples.append(
                        "%s %s -> %s" % (k, pi.shape[:2], target))
            pi = resize_to(pi, target)
        if gi.shape[:2] != target:
            gi = resize_to(gi, target)
        pred_imgs.append(pi)
        gt_imgs.append(gi)
        used_keys.append(k)

    if n_unreadable:
        print("WARNING: %d pair(s) skipped, image unreadable by cv2.imread"
              % n_unreadable)
    if n_size_mismatch:
        print("WARNING: %d pair(s) had pred and gt at DIFFERENT native sizes; "
              "pred was resampled to the GT/target size with INTER_AREA "
              "(downscale) / INTER_CUBIC (upscale). e.g. %s"
              % (n_size_mismatch, "; ".join(mismatch_examples)))
    if args.size > 0:
        print("all images resized to %dx%d (INTER_AREA when shrinking, "
              "INTER_CUBIC when growing)" % (args.size, args.size))
    elif n_resized:
        print("WARNING: %d pred image(s) differed in size from their GT and were "
              "resized with INTER_AREA. e.g. %s" % (n_resized, "; ".join(resize_examples)))
    n_pairs = len(pred_imgs)
    if n_pairs < 1:
        raise SystemExit("FATAL: 0 usable pairs after loading.")
    print("usable pairs       : %d" % n_pairs)

    # ---- PSNR / SSIM -------------------------------------------------------
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim

    psnr_gray, psnr_rgb, ssim_gray, ssim_rgb = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # identical images -> divide-by-zero -> inf
        with np.errstate(divide="ignore", invalid="ignore"):
            for pi, gi in zip(pred_imgs, gt_imgs):
                pg = to_gray(pi).astype(np.float64)
                gg = to_gray(gi).astype(np.float64)
                pf = pi.astype(np.float64)
                gf = gi.astype(np.float64)
                psnr_gray.append(sk_psnr(gg, pg, data_range=255))
                psnr_rgb.append(sk_psnr(gf, pf, data_range=255))
                ssim_gray.append(sk_ssim(gg, pg, data_range=255))
                ssim_rgb.append(sk_ssim(gf, pf, data_range=255, channel_axis=-1))

    psnr_gray = np.asarray(psnr_gray, dtype=np.float64)
    psnr_rgb = np.asarray(psnr_rgb, dtype=np.float64)
    n_inf_gray = int(np.isinf(psnr_gray).sum())
    n_inf_rgb = int(np.isinf(psnr_rgb).sum())

    def _mean(a):
        return float(np.mean(a))

    def _finite_mean(a):
        f = a[np.isfinite(a)]
        return float(np.mean(f)) if f.size else float("nan")

    # ---- LPIPS -------------------------------------------------------------
    print("-" * 78)
    print("computing LPIPS (net=%s) ..." % args.lpips_net)
    lp = compute_lpips(pred_imgs, gt_imgs, args.lpips_net, device, args.batch_size)

    # ---- FID ---------------------------------------------------------------
    fid_val, fid_used, fid_err = None, None, None
    n_pred_fid = n_gt_fid = 0
    fid_size = args.fid_size if args.fid_size > 0 else (args.size if args.size > 0 else 299)
    workdir = None
    if not args.no_fid:
        print("computing FID (impl=%s, inception input %dpx) ..." % (args.fid_impl, fid_size))
        # FID is a distribution metric: use every readable image in each dir.
        fid_pred, fid_pred_names = [], []
        for p in pred_paths:
            a = load_rgb(p, args.assume_bgr)
            if a is not None:
                fid_pred.append(a)
                fid_pred_names.append(os.path.basename(p))
        fid_gt, fid_gt_names = [], []
        for p in gt_paths:
            a = load_rgb(p, args.assume_bgr)
            if a is not None:
                fid_gt.append(a)
                fid_gt_names.append(os.path.basename(p))
        n_pred_fid, n_gt_fid = len(fid_pred), len(fid_gt)
        if n_pred_fid != n_gt_fid:
            print("NOTE: pred and gt directories hold different image counts "
                  "(%d vs %d). FID does not require a 1:1 correspondence, but "
                  "the estimate's bias depends on the sample size, so the two "
                  "counts are reported below." % (n_pred_fid, n_gt_fid))
        if min(n_pred_fid, n_gt_fid) < 2:
            fid_err = "need >=2 images per directory for a covariance estimate"
            print("WARNING: skipping FID -- %s" % fid_err)
        else:
            workdir = tempfile.mkdtemp(prefix="eval_metrics_fid_")
            try:
                fid_val, fid_used, fid_err = compute_fid(
                    args.fid_impl, fid_pred, fid_gt, fid_pred_names, fid_gt_names,
                    device, args.batch_size, fid_size, workdir)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
            if fid_val is None:
                print("WARNING: FID could not be computed (%s)" % fid_err)
        del fid_pred, fid_gt

    # ---- results -----------------------------------------------------------
    results = OrderedDict()
    results["psnr_gray"] = _mean(psnr_gray)
    results["psnr_gray_finite_mean"] = _finite_mean(psnr_gray)
    results["psnr_gray_n_inf"] = n_inf_gray
    results["psnr_rgb"] = _mean(psnr_rgb)
    results["psnr_rgb_finite_mean"] = _finite_mean(psnr_rgb)
    results["psnr_rgb_n_inf"] = n_inf_rgb
    results["ssim_gray"] = float(np.mean(ssim_gray))
    results["ssim_rgb"] = float(np.mean(ssim_rgb))
    results["lpips"] = float(np.mean(lp))
    results["lpips_std"] = float(np.std(lp))
    results["fid"] = fid_val
    results["fid_impl_used"] = fid_used
    results["fid_error"] = fid_err
    results["fid_inception_input_px"] = fid_size if not args.no_fid else None

    counts = OrderedDict([
        ("n_files_pred", len(pred_paths)),
        ("n_files_gt", len(gt_paths)),
        ("n_pairs", n_pairs),
        ("n_unmatched_pred", len(only_pred)),
        ("n_unmatched_gt", len(only_gt)),
        ("n_unreadable_pairs", n_unreadable),
        ("n_pairs_native_size_mismatch", n_size_mismatch),
        ("n_pred_fid", n_pred_fid),
        ("n_gt_fid", n_gt_fid),
        ("key_strip_suffix_digits", strip),
    ])

    def fmt(x, nd=4):
        if x is None:
            return "n/a"
        if isinstance(x, float) and np.isinf(x):
            return "inf"
        if isinstance(x, float) and np.isnan(x):
            return "nan"
        if isinstance(x, float):
            return ("%%.%df" % nd) % x
        return str(x)

    print()
    print("=" * 78)
    print("RESULTS   (%d pairs; pred=%s  gt=%s)"
          % (n_pairs, os.path.basename(os.path.normpath(args.pred)),
             os.path.basename(os.path.normpath(args.gt))))
    print("=" * 78)
    rows = [
        ("PSNR  (grayscale)", fmt(results["psnr_gray"]), "dB  <- paper's PSNR"),
        ("PSNR  (RGB)", fmt(results["psnr_rgb"]), "dB"),
        ("SSIM  (grayscale)", fmt(results["ssim_gray"]), "    <- paper's SSIM"),
        ("SSIM  (RGB)", fmt(results["ssim_rgb"]), ""),
        ("LPIPS (%s)" % args.lpips_net, fmt(results["lpips"]),
         "+/- %s (std over pairs)" % fmt(results["lpips_std"])),
        ("FID   (%s)" % (fid_used or args.fid_impl), fmt(results["fid"], 4),
         "n_pred=%d  n_gt=%d" % (n_pred_fid, n_gt_fid)),
    ]
    for name, val, note in rows:
        print("  %-20s %14s   %s" % (name, val, note))
    if n_inf_gray or n_inf_rgb:
        print()
        print("  NOTE: %d/%d grayscale and %d/%d RGB pairs are pixel-identical "
              "(PSNR = inf), so the plain mean is inf." % (n_inf_gray, n_pairs,
                                                           n_inf_rgb, n_pairs))
        print("        finite-only means: gray %s dB, rgb %s dB"
              % (fmt(results["psnr_gray_finite_mean"]),
                 fmt(results["psnr_rgb_finite_mean"])))
    if results["fid"] is not None:
        print()
        print("  FID caveat: FID is a biased estimator whose expectation falls "
              "with sample size.")
        print("              n_pred=%d / n_gt=%d is small by FID standards "
              "(50k is the usual" % (n_pred_fid, n_gt_fid))
        print("              reference), so this value is only comparable "
              "against other FIDs from")
        print("              the same implementation at the same sample count.")
    print("=" * 78)

    payload = OrderedDict()
    payload["pred_dir"] = os.path.abspath(args.pred)
    payload["gt_dir"] = os.path.abspath(args.gt)
    payload["metrics"] = results
    payload["counts"] = counts
    payload["flags"] = OrderedDict(sorted(vars(args).items()))
    payload["flags"]["device_used"] = device
    payload["flags"]["channel_convention"] = (
        "cv2.imread -> BGR, converted BGR->RGB before LPIPS/FID"
        if args.assume_bgr else
        "cv2.imread bytes used directly as RGB (no conversion)")
    payload["versions"] = versions

    out = args.out
    if out:
        d = os.path.dirname(os.path.abspath(out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(out, "w") as f:
            json.dump(_json_safe(payload), f, indent=2, default=str,
                      allow_nan=False)
        print("JSON written to: %s" % os.path.abspath(out))

    return payload


if __name__ == "__main__":
    main()
