"""Build OCR fine-tune pairs from a synthetic TestSerialSet-style PDF + JSON.

The synthesizer drops two files in a directory::

    <set>/<name>.pdf       — N pages, each a rendered ATF form
    <set>/<name>.json      — per-page manifest with form_year + firearms

Page rasterization + serial-block crop reuses the same path as
``pipeline.ocr_finetune_data`` (annotated ``serial_block`` from
``FormTemplates/<year>/form_config.json``, equal-height row split).

Output layout::

    <out>/train/<id>.png
    <out>/train/labels.tsv
    <out>/val/<id>.png
    <out>/val/labels.tsv

Usage::

    python -m pipeline.ocr_synth_data \\
        --src TestSerialSet/serial_only_500 \\
        --out ocr_synth_data \\
        --val-frac 0.10
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
from pdf2image import convert_from_path

from pipeline import localize
from pipeline.recognize import normalize_serial

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "FormTemplates"
_CROP_DPI = 200


def _row_crops(
    page_img: Image.Image, cfg: dict[str, Any], expected_rows: int,
) -> list[Image.Image]:
    """Same equal-height geometric split used elsewhere in the pipeline."""
    page_size = cfg.get("page_size") or [612, 792]
    page_w_pt, page_h_pt = float(page_size[0]), float(page_size[1])
    iw, ih = page_img.size
    box_pts = localize._serial_block_pts(cfg)
    L, T, R, B = localize._pts_to_pixels(box_pts, iw, ih, page_w_pt, page_h_pt)
    block = page_img.crop((L, T, R, B))
    bw, bh = block.size
    crops: list[Image.Image] = []
    for i in range(expected_rows):
        top = int(i * bh / expected_rows)
        bot = int((i + 1) * bh / expected_rows) if i + 1 < expected_rows else bh
        crops.append(block.crop((0, top, bw, bot)))
    return crops


def _gather_sources(src_arg: Path) -> list[tuple[Path, Path]]:
    """Resolve --src into a list of (pdf, json) pairs.

    Two modes:
      - Path stem (no extension) → single (stem.pdf, stem.json) pair.
      - Directory → all (*.pdf, matching *.json) inside it, sorted.
    """
    pairs: list[tuple[Path, Path]] = []
    if src_arg.is_dir():
        for pdf in sorted(src_arg.glob("*.pdf")):
            jpath = pdf.with_suffix(".json")
            if jpath.is_file():
                pairs.append((pdf, jpath))
            else:
                print(f"  skipping {pdf.name}: no matching .json")
    else:
        pdf = src_arg.with_suffix(".pdf")
        jp = src_arg.with_suffix(".json")
        if pdf.is_file() and jp.is_file():
            pairs.append((pdf, jp))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True,
                    help="Either a path stem (e.g. TestSerialSet/serial_only_500) "
                         "or a directory containing matched *.pdf/*.json pairs (e.g. SerialSets)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory (created/cleared)")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit-pages", type=int, default=None,
                    help="Only first N PDF pages per source PDF (debug)")
    ap.add_argument("--include-corrupt", action="store_true",
                    help="Also include rows whose firearm has corruption — only "
                         "use for verifier training, not OCR fine-tune.")
    args = ap.parse_args()

    sources = _gather_sources(args.src)
    if not sources:
        print(f"No usable (pdf, json) pairs found at {args.src}")
        return 1
    print(f"Found {len(sources)} source PDF(s)")

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "train").mkdir(parents=True)
    (args.out / "val").mkdir(parents=True)

    cfg_cache: dict[str, dict[str, Any] | None] = {}

    def get_cfg(year: str) -> dict[str, Any] | None:
        if year not in cfg_cache:
            cfg_cache[year] = localize._load_form_config(TEMPLATES, year)
        return cfg_cache[year]

    rng = random.Random(args.seed)
    pairs_train: list[tuple[str, str]] = []
    pairs_val: list[tuple[str, str]] = []
    skip_no_template = skip_overflow = skip_corrupt = skip_empty = 0

    for src_pdf, src_json in sources:
        d = json.load(open(src_json))
        pages = [pg for pg in d["pages"] if pg.get("firearms")]
        if args.limit_pages:
            pages = pages[: args.limit_pages]
        print(f"\nSource: {src_pdf.name}  ({len(pages)} form-bearing pages)")
        # ``src_id`` namespaces the output filenames so two source PDFs with
        # overlapping page numbers don't collide.
        src_id = src_pdf.stem

        for pg_i, pg in enumerate(pages):
            year = pg["form_year"]
            cfg = get_cfg(year)
            if not cfg:
                skip_no_template += 1
                continue
            row_y_count = len(cfg["firearm_rows"]["row_y"])
            if len(pg["firearms"]) > row_y_count:
                skip_overflow += 1
                continue

            pdf_pageno = int(pg["page"])
            try:
                rendered = convert_from_path(
                    str(src_pdf), dpi=_CROP_DPI,
                    first_page=pdf_pageno, last_page=pdf_pageno,
                )
            except Exception as e:
                print(f"  raster fail p{pdf_pageno}: {e}")
                continue
            if not rendered:
                continue
            page_img = rendered[0].convert("RGB")
            crops = _row_crops(page_img, cfg, row_y_count)

            is_val = rng.random() < args.val_frac
            split_dir = "val" if is_val else "train"
            target_pairs = pairs_val if is_val else pairs_train

            for i, fa in enumerate(pg["firearms"]):
                if not args.include_corrupt and fa.get("corruption"):
                    skip_corrupt += 1
                    continue
                truth = normalize_serial(fa.get("serial", ""))
                if not truth:
                    skip_empty += 1
                    continue
                stem_id = f"{src_id}_p{pdf_pageno:04d}_r{i}"
                fn = f"{stem_id}.png"
                crops[i].save(args.out / split_dir / fn)
                target_pairs.append((fn, truth))

            if (pg_i + 1) % 100 == 0:
                print(f"  [{pg_i + 1}/{len(pages)}] train={len(pairs_train)} val={len(pairs_val)}", flush=True)

    for split, pairs in (("train", pairs_train), ("val", pairs_val)):
        with open(args.out / split / "labels.tsv", "w") as f:
            for fn, label in pairs:
                f.write(f"{fn}\t{label}\n")

    print()
    print(f"Train pairs : {len(pairs_train)}")
    print(f"Val pairs   : {len(pairs_val)}")
    print(f"Skipped (no template / overflow / corrupt / empty): "
          f"{skip_no_template} / {skip_overflow} / {skip_corrupt} / {skip_empty}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
