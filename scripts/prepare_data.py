#!/usr/bin/env python
"""Prepare the SpeakingFaces thermal->visible dataset for the DiffTV configs.

Source layout (CycleGAN style, variable-size face crops)::

    <src>/trainA/<prefix>_1.png   thermal  (TH)
    <src>/trainB/<prefix>_3.png   visible  (VIS)
    <src>/testA/<prefix>_1.png
    <src>/testB/<prefix>_3.png

Destination layout (what ``ldm/data/reconstruct.py`` and ``ldm/data/ldm_t2v.py``
expect)::

    <dst>/train/TH   <dst>/train/VIS
    <dst>/test/TH    <dst>/test/VIS

``DatasetT2VLDM`` globs ``*.png`` in each of TH/VIS, ``sort()``s both lists and
pairs them **by index**.  There is no name-based matching whatsoever, so the
sorted TH list must correspond element-wise to the sorted VIS list.  To make
that trivially true (and obvious to a human reading the directories) this
script strips the trailing ``_1`` / ``_3`` modality suffix and writes both
modalities under the *same* shared base name, e.g.::

    100_1_2_1_1134_36_1.png  ->  train/TH/100_1_2_1_1134_36.png
    100_1_2_1_1134_36_3.png  ->  train/VIS/100_1_2_1_1134_36.png

Every image is resized to ``--size`` x ``--size`` (INTER_AREA when shrinking,
INTER_CUBIC when enlarging) and written as lossless PNG.

Dependencies: stdlib + numpy + opencv-python-headless + tqdm.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_SRC = "path/to/thermal2visible_speakingfaces"   # raw split: trainA/trainB/testA/testB
DEFAULT_DST = "data/thermal2visible_speakingfaces"       # output: {train,test}/{TH,VIS}

# split name -> (source A dir = thermal, source B dir = visible)
SPLITS: Dict[str, Tuple[str, str]] = {
    "train": ("trainA", "trainB"),
    "test": ("testA", "testB"),
}

TH_SUFFIX = "_1"   # A == thermal
VIS_SUFFIX = "_3"  # B == visible


# --------------------------------------------------------------------------- #
# scanning / validation
# --------------------------------------------------------------------------- #
def scan_dir(path: str, suffix: str) -> Tuple[Dict[str, str], List[str]]:
    """Return ({base_name: filename}, [files with an unexpected suffix])."""
    if not os.path.isdir(path):
        raise SystemExit(f"[fatal] source directory does not exist: {path}")

    mapping: Dict[str, str] = {}
    odd: List[str] = []
    for fname in sorted(os.listdir(path)):
        if not fname.lower().endswith(".png"):
            continue
        stem = fname[: -len(".png")]
        if not stem.endswith(suffix):
            odd.append(fname)
            continue
        base = stem[: -len(suffix)]
        if base in mapping:
            odd.append(fname)  # duplicate base -> treat as unusable
            continue
        mapping[base] = fname
    return mapping, odd


def read_png_size(path: str) -> Optional[Tuple[int, int]]:
    """(width, height) straight out of the PNG IHDR chunk -- no decode."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(26)
        if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        return width, height
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# conversion worker (must be top-level for ProcessPoolExecutor)
# --------------------------------------------------------------------------- #
def convert_one(job: Tuple[str, str, int]) -> Tuple[bool, str, str]:
    """(src, dst, size) -> (ok, src, error-message)."""
    src, dst, size = job
    try:
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            return False, src, "cv2.imread returned None"
        h, w = img.shape[:2]
        if (w, h) != (size, size):
            # INTER_AREA is the right kernel for decimation, INTER_CUBIC for
            # magnification.  Mixed cases (shrink one axis, grow the other) are
            # treated as upscaling so we never alias-free-but-blur a grow.
            interp = cv2.INTER_AREA if (w >= size and h >= size) else cv2.INTER_CUBIC
            img = cv2.resize(img, (size, size), interpolation=interp)
        tmp = dst + ".tmp.png"
        ok = cv2.imwrite(tmp, img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if not ok:
            if os.path.exists(tmp):
                os.remove(tmp)
            return False, src, "cv2.imwrite failed"
        os.replace(tmp, dst)  # atomic -> a killed run never leaves half a PNG
        return True, src, ""
    except Exception as exc:  # noqa: BLE001 - worker must never die silently
        return False, src, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def human(triple: Tuple[int, int, int]) -> str:
    return "x".join(str(v) for v in triple) if triple else "-"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert the SpeakingFaces trainA/trainB/testA/testB dataset "
                    "into the train|test / TH|VIS layout the DiffTV configs expect.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", default=DEFAULT_SRC, help="source dataset root")
    ap.add_argument("--dst", default=DEFAULT_DST, help="destination dataset root")
    ap.add_argument("--size", type=int, default=128, help="output square side, in pixels")
    ap.add_argument("--workers", type=int, default=32, help="process pool size")
    ap.add_argument("--force", action="store_true",
                    help="re-convert files that already exist in the destination")
    args = ap.parse_args()

    if args.size <= 0:
        raise SystemExit("[fatal] --size must be positive")
    workers = max(1, min(args.workers, (os.cpu_count() or 1) * 4))

    print(f"src     : {args.src}")
    print(f"dst     : {args.dst}")
    print(f"size    : {args.size}x{args.size}")
    print(f"workers : {workers}{'' if workers == args.workers else f' (clamped from {args.workers})'}")
    print(f"force   : {args.force}")
    print()

    jobs: List[Tuple[str, str, int]] = []
    summary: Dict[str, dict] = {}
    manifest_splits: Dict[str, dict] = {}
    any_problem = False

    for split, (a_dir, b_dir) in SPLITS.items():
        a_path = os.path.join(args.src, a_dir)
        b_path = os.path.join(args.src, b_dir)
        th_out = os.path.join(args.dst, split, "TH")
        vis_out = os.path.join(args.dst, split, "VIS")
        os.makedirs(th_out, exist_ok=True)
        os.makedirs(vis_out, exist_ok=True)

        a_map, a_odd = scan_dir(a_path, TH_SUFFIX)
        b_map, b_odd = scan_dir(b_path, VIS_SUFFIX)

        a_only = sorted(set(a_map) - set(b_map))
        b_only = sorted(set(b_map) - set(a_map))
        paired = sorted(set(a_map) & set(b_map))

        print(f"[{split}] {a_dir}: {len(a_map)} usable  |  {b_dir}: {len(b_map)} usable  "
              f"-> {len(paired)} pairs")
        for label, items, where in (
            (f"unexpected-suffix/duplicate files in {a_dir}", a_odd, a_dir),
            (f"unexpected-suffix/duplicate files in {b_dir}", b_odd, b_dir),
            (f"orphan thermal (no {b_dir} counterpart)", a_only, a_dir),
            (f"orphan visible (no {a_dir} counterpart)", b_only, b_dir),
        ):
            if items:
                any_problem = True
                print(f"  !! {len(items)} {label} -- SKIPPED:")
                for name in items[:20]:
                    print(f"       {where}/{name}")
                if len(items) > 20:
                    print(f"       ... and {len(items) - 20} more")

        # ---- source-dimension statistics + A/B dimension agreement ----------
        widths: List[int] = []
        heights: List[int] = []
        areas: List[Tuple[int, int, int]] = []  # (area, w, h) for median-by-area
        dim_mismatch: List[dict] = []
        upscaled = 0
        unreadable: List[str] = []

        for base in tqdm(paired, desc=f"[{split}] probing source sizes", unit="pair",
                         leave=False, file=sys.stdout):
            a_sz = read_png_size(os.path.join(a_path, a_map[base]))
            b_sz = read_png_size(os.path.join(b_path, b_map[base]))
            if a_sz is None or b_sz is None:
                unreadable.append(base)
                continue
            if a_sz != b_sz:
                dim_mismatch.append({"base": base,
                                     "thermal": list(a_sz), "visible": list(b_sz)})
            w, h = a_sz
            widths.append(w)
            heights.append(h)
            areas.append((w * h, w, h))
            # each modality is resized independently, so a pair counts as
            # upscaled if either frame is under --size on either axis
            if min(w, h) < args.size or min(b_sz) < args.size:
                upscaled += 1

        if unreadable:
            any_problem = True
            print(f"  !! {len(unreadable)} pairs with an unreadable PNG header "
                  f"(kept, cv2 will be the judge): {unreadable[:5]}")
        if dim_mismatch:
            any_problem = True
            print(f"  !! {len(dim_mismatch)} pairs where thermal and visible source "
                  f"dimensions DISAGREE (expected 0):")
            for m in dim_mismatch[:20]:
                print(f"       {m['base']}: TH {m['thermal'][0]}x{m['thermal'][1]} "
                      f"vs VIS {m['visible'][0]}x{m['visible'][1]}")
            if len(dim_mismatch) > 20:
                print(f"       ... and {len(dim_mismatch) - 20} more")
        else:
            print(f"  ok {len(paired)} pairs, 0 source-dimension mismatches between TH and VIS")

        # ---- queue the conversions -----------------------------------------
        queued = 0
        skipped = 0
        for base in paired:
            out_name = base + ".png"
            for src_dir, src_map, out_dir in ((a_path, a_map, th_out),
                                              (b_path, b_map, vis_out)):
                dst_file = os.path.join(out_dir, out_name)
                if not args.force and os.path.isfile(dst_file) and os.path.getsize(dst_file) > 0:
                    skipped += 1
                    continue
                jobs.append((os.path.join(src_dir, src_map[base]), dst_file, args.size))
                queued += 1
        print(f"  queued {queued} images, {skipped} already present"
              f"{' (use --force to redo)' if skipped else ''}")
        print()

        def stat3(values: List[int]) -> Optional[Tuple[int, int, int]]:
            if not values:
                return None
            return (min(values), int(statistics.median(values)), max(values))

        med_wh: Optional[Tuple[int, int]] = None
        if areas:
            areas.sort()
            med_wh = (areas[len(areas) // 2][1], areas[len(areas) // 2][2])

        summary[split] = {
            "pairs": len(paired),
            "width": stat3(widths),
            "height": stat3(heights),
            "median_wh": med_wh,
            "upscaled": upscaled,
            "mismatch": len(dim_mismatch),
            "orphans": len(a_only) + len(b_only),
        }
        manifest_splits[split] = {
            "source_thermal_dir": a_path,
            "source_visible_dir": b_path,
            "out_thermal_dir": th_out,
            "out_visible_dir": vis_out,
            "source_thermal_files": len(a_map),
            "source_visible_files": len(b_map),
            "pairs": len(paired),
            "orphan_thermal_bases": a_only,
            "orphan_visible_bases": b_only,
            "unexpected_thermal_files": a_odd,
            "unexpected_visible_files": b_odd,
            "dimension_mismatches": dim_mismatch,
            "unreadable_header_bases": unreadable,
            "source_width_min_median_max": stat3(widths),
            "source_height_min_median_max": stat3(heights),
            "upscaled_sources": upscaled,
        }

    # ---- run the conversions ------------------------------------------------
    failures: List[Tuple[str, str]] = []
    if jobs:
        print(f"converting {len(jobs)} images with {workers} workers ...")
        chunk = max(1, min(64, len(jobs) // (workers * 4) or 1))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for ok, src, err in tqdm(pool.map(convert_one, jobs, chunksize=chunk),
                                     total=len(jobs), unit="img", file=sys.stdout):
                if not ok:
                    failures.append((src, err))
        print()
    else:
        print("nothing to convert (everything already present; use --force to redo)\n")

    if failures:
        any_problem = True
        print(f"!! {len(failures)} conversions FAILED:")
        for src, err in failures[:20]:
            print(f"   {src}: {err}")
        if len(failures) > 20:
            print(f"   ... and {len(failures) - 20} more")
        print()

    # ---- verify the on-disk result -----------------------------------------
    print("verifying destination directories ...")
    verify_ok = True
    for split in SPLITS:
        th_out = os.path.join(args.dst, split, "TH")
        vis_out = os.path.join(args.dst, split, "VIS")
        th_files = sorted(f for f in os.listdir(th_out) if f.endswith(".png"))
        vis_files = sorted(f for f in os.listdir(vis_out) if f.endswith(".png"))
        # This is the invariant DatasetT2VLDM silently relies on: sorted(TH)
        # and sorted(VIS) must line up element-wise.
        if th_files != vis_files:
            verify_ok = False
            any_problem = True
            only_th = sorted(set(th_files) - set(vis_files))[:10]
            only_vis = sorted(set(vis_files) - set(th_files))[:10]
            print(f"  !! [{split}] TH and VIS file lists DIFFER "
                  f"({len(th_files)} vs {len(vis_files)}); TH-only {only_th}, VIS-only {only_vis}")
        else:
            print(f"  ok [{split}] {len(th_files)} TH == {len(vis_files)} VIS, "
                  f"identical sorted basenames")
        summary[split]["written"] = len(th_files)
        manifest_splits[split]["written_thermal"] = len(th_files)
        manifest_splits[split]["written_visible"] = len(vis_files)
        manifest_splits[split]["filelists_identical"] = (th_files == vis_files)

    assert verify_ok, ("TH and VIS destination file lists are not identical -- "
                       "DatasetT2VLDM pairs by sorted index and would mis-pair.")
    print()

    # ---- summary table ------------------------------------------------------
    header = f"{'split':<8}{'pairs':>8}{'written':>9}{'src W min/med/max':>22}{'src H min/med/max':>22}{'upscaled':>10}{'mismatch':>10}{'orphans':>9}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    total_upscaled = 0
    for split in SPLITS:
        s = summary[split]
        total_upscaled += s["upscaled"]
        print(f"{split:<8}{s['pairs']:>8}{s['written']:>9}"
              f"{human(s['width']):>22}{human(s['height']):>22}"
              f"{s['upscaled']:>10}{s['mismatch']:>10}{s['orphans']:>9}")
    print("=" * len(header))
    for split in SPLITS:
        s = summary[split]
        if s["median_wh"]:
            print(f"  [{split}] median source frame (by area): "
                  f"{s['median_wh'][0]}x{s['median_wh'][1]}")
    if total_upscaled:
        print(f"\n  WARNING: {total_upscaled} source images are smaller than "
              f"{args.size}px in at least one dimension and were UPSCALED "
              f"(INTER_CUBIC). Their effective resolution is lower than {args.size}x{args.size}.")
    else:
        print(f"\n  all source images are >= {args.size}px in both dimensions "
              f"(pure downscale, INTER_AREA); nothing was upscaled.")

    # ---- manifest -----------------------------------------------------------
    manifest = {
        "src": args.src,
        "dst": args.dst,
        "size": args.size,
        "interpolation": "INTER_AREA when downscaling, INTER_CUBIC when upscaling",
        "naming": ("modality suffix stripped: <prefix>_1.png (A/thermal) -> TH/<prefix>.png, "
                   "<prefix>_3.png (B/visible) -> VIS/<prefix>.png, so sorted(TH) and "
                   "sorted(VIS) pair element-wise as DatasetT2VLDM requires"),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "force": args.force,
        "splits": manifest_splits,
        "conversion_failures": [{"src": s, "error": e} for s, e in failures],
        "total_written": sum(summary[s]["written"] for s in SPLITS),
        "clean": (not any_problem),
    }
    os.makedirs(args.dst, exist_ok=True)
    manifest_path = os.path.join(args.dst, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nmanifest written to {manifest_path}")

    if any_problem:
        print("\nFINISHED WITH WARNINGS (see above).")
        return 1
    print("\nOK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
