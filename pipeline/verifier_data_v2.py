"""Generate stage-4 verifier training data from v2 multi-form PDFs.

Inputs (under ``Version 2 Datasets/``):
    v2dataset_no_errors_{1,2}.pdf  — clean serials only
    v2dataset_errors_{1,2}.pdf     — same forms but ~80% of firearms have
                                     deliberate corruption recorded in JSON

Optional additional corruption-rich datasets (e.g., ``Serial Error Pages/``)
can be added via ``--extra-source DIR``. Their pages contribute to the
``train`` split as additional negatives.

Train/val split is by PDF, not by row, to prevent leakage:
    train: v2dataset_no_errors_1.pdf + v2dataset_errors_1.pdf  (+ extras)
    val:   v2dataset_no_errors_2.pdf + v2dataset_errors_2.pdf

Per-row labeling (matches ``pipeline.evaluate_v2._classify_firearm``):
    positive: clean firearm, OR corruption type 'serial_overflow'
    negative: corruption types 'pii_in_serial', 'name_in_serial', 'field_swap',
              or 'overflow_into_serial' with serial_also_written=False
    skip:     'overflow_into_serial' with serial_also_written=True (ambiguous)
    negative: empty rows beyond the firearm count (capped at 1 per form)

Output layout::

    verifier_data_v2/
      train/positive/<id>.png
      train/negative/<id>.png
      val/positive/<id>.png
      val/negative/<id>.png
      manifest.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
from pdf2image import convert_from_path

from pipeline import localize

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "FormTemplates"
V2_DIR = ROOT / "Version 2 Datasets"
MT_DIR = ROOT / "MasterTraining"
OUT_DEFAULT = ROOT / "verifier_data_v2"

_CROP_DPI = 200

# Training data sources (synthetic v2 + MasterTraining real-distribution).
# MasterTraining adds: case_report (CR*), complete_train, cont_train across
# all 11 years, plus matching v2dataset_*_1 copies and serial_only +
# serial_only_error PDFs. ~8000 form pages with ~1200 corruption labels.
TRAIN_PDFS = [
    (V2_DIR, "v2dataset_no_errors_1"),
    (V2_DIR, "v2dataset_errors_1"),
    # MasterTraining — full set
    (MT_DIR, "1985_complete_train"),
    (MT_DIR, "1998_complete_train"),
    (MT_DIR, "2001_complete_train"),
    (MT_DIR, "2005_complete_train"),
    (MT_DIR, "2007_complete_train"),
    (MT_DIR, "2008_complete_train"),
    (MT_DIR, "2012_complete_train"),
    (MT_DIR, "2016_complete_train"),
    (MT_DIR, "2020_complete_train"),
    (MT_DIR, "2020_cont_train"),
    (MT_DIR, "2022_complete_train"),
    (MT_DIR, "2022_cont_train"),
    (MT_DIR, "2023_complete_train"),
    (MT_DIR, "2023_cont_train"),
    (MT_DIR, "CR3"), (MT_DIR, "CR4"), (MT_DIR, "CR5"),
    (MT_DIR, "CR8"), (MT_DIR, "CR10"), (MT_DIR, "CR12"),
    (MT_DIR, "CR13"), (MT_DIR, "CR15"), (MT_DIR, "CR16"),
    (MT_DIR, "CR19"), (MT_DIR, "CR20"), (MT_DIR, "CR22"), (MT_DIR, "CR23"),
    # MasterTraining serial_only PDFs are split 1..6 (use all)
    (MT_DIR, "serial_only_1"),
    (MT_DIR, "serial_only_2"),
    (MT_DIR, "serial_only_3"),
    (MT_DIR, "serial_only_4"),
    (MT_DIR, "serial_only_5"),
    (MT_DIR, "serial_only_6"),
    (MT_DIR, "serial_only_error_1"),
    # NOTE: MasterTraining/v2dataset_no_errors_2 and v2dataset_errors_2 are
    # byte-identical duplicates of Version 2 Datasets/v2dataset_*_2 which is
    # the verifier validation set + part of @compliance eval. Excluded from
    # train to avoid leakage.
]
# Validation = held-out v2 _2 ONLY. MasterTesting MUST NOT enter the val set
# because the master_report uses MasterTesting as the held-out evaluation.
# Putting MasterTesting in val would let threshold selection leak into the
# eval and inflate the reported metrics.
VAL_PDFS = [(V2_DIR, "v2dataset_no_errors_2"),
            (V2_DIR, "v2dataset_errors_2")]

_CORRUPTION_AS_POSITIVE = {"serial_overflow"}
_CORRUPTION_AS_NEGATIVE = {"pii_in_serial", "name_in_serial", "field_swap"}
_EMPTY_ROWS_PER_FORM = 1


def _row_crops(
    page_img: Image.Image, cfg: dict[str, Any], expected_rows: int,
) -> list[Image.Image]:
    """Equal-height geometric split inside the annotated serial_block."""
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


def _classify_firearm(firearm: dict[str, Any]) -> str:
    corr = firearm.get("corruption")
    if not corr:
        return "positive"
    ctype = corr.get("type")
    if ctype in _CORRUPTION_AS_POSITIVE:
        return "positive"
    if ctype in _CORRUPTION_AS_NEGATIVE:
        return "negative"
    if ctype == "overflow_into_serial":
        if corr.get("serial_also_written") is False:
            return "negative"
        return "skip"
    return "skip"


def _gather_extra_sources(extra_dirs: list[Path]) -> list[tuple[Path, str]]:
    """For each extra dir, find (dir, stem) pairs (one per matched pdf+json)."""
    out: list[tuple[Path, str]] = []
    for d in extra_dirs:
        if not d.is_dir():
            print(f"  WARNING: extra source {d} is not a directory")
            continue
        for pdf in sorted(d.glob("*.pdf")):
            jp = pdf.with_suffix(".json")
            if not jp.is_file():
                print(f"  WARNING: {pdf.name} has no matching .json — skipping")
                continue
            out.append((d, pdf.stem))
    return out


def _process_one(
    src_dir: Path, stem: str, split: str, args, cfg_cache, manifest, args_dst,
):
    pdf_path = src_dir / f"{stem}.pdf"
    json_path = src_dir / f"{stem}.json"
    if not pdf_path.is_file() or not json_path.is_file():
        print(f"  MISSING: {pdf_path.name}")
        return
    d = json.load(open(json_path))

    form_pages = [pg for pg in d["pages"] if pg.get("firearms")]
    if args.limit_pages:
        form_pages = form_pages[: args.limit_pages]
    print(f"\n[{split}] {stem}: {len(form_pages)} form-bearing pages", flush=True)

    stem_stats = {"pos": 0, "neg": 0, "pages_processed": 0, "pages_skipped": 0}

    for pg_i, pg in enumerate(form_pages):
        year = pg["form_year"]
        if year not in cfg_cache:
            cfg_cache[year] = localize._load_form_config(TEMPLATES, year)
        cfg = cfg_cache[year]
        if not cfg:
            manifest["skipped_no_template"] += 1
            stem_stats["pages_skipped"] += 1
            continue
        row_y_count = len(cfg["firearm_rows"]["row_y"])
        if len(pg["firearms"]) > row_y_count:
            manifest["skipped_overflow_pages"] += 1
            stem_stats["pages_skipped"] += 1
            continue

        pdf_pageno = int(pg["page"])
        try:
            pages = convert_from_path(
                str(pdf_path), dpi=_CROP_DPI,
                first_page=pdf_pageno, last_page=pdf_pageno,
            )
        except Exception as e:
            print(f"  raster fail p{pdf_pageno}: {e}")
            stem_stats["pages_skipped"] += 1
            continue
        if not pages:
            stem_stats["pages_skipped"] += 1
            continue
        page_img = pages[0].convert("RGB")
        rows = _row_crops(page_img, cfg, row_y_count)

        # Source-prefix the IDs so two source PDFs with overlapping page
        # numbers don't collide in the output dir.
        src_id = f"{src_dir.name.replace(' ', '_')}_{stem}"
        stem_id = f"{src_id}_{pdf_pageno:04d}"

        for i, fa in enumerate(pg["firearms"]):
            label = _classify_firearm(fa)
            if label == "skip":
                manifest["skipped_ambiguous_firearms"] += 1
                continue
            crop = rows[i]
            out_dir = args_dst / split / label
            fname = f"{stem_id}_r{i}_{label[:3]}.png"
            crop.save(out_dir / fname)
            if label == "positive":
                stem_stats["pos"] += 1
            else:
                stem_stats["neg"] += 1

        # Empty rows beyond firearm count → negative (capped)
        empty_count = 0
        for i in range(len(pg["firearms"]), len(rows)):
            if empty_count >= _EMPTY_ROWS_PER_FORM:
                break
            rows[i].save(args_dst / split / "negative" / f"{stem_id}_r{i}_empty.png")
            stem_stats["neg"] += 1
            empty_count += 1

        stem_stats["pages_processed"] += 1
        if (pg_i + 1) % 50 == 0:
            print(f"    [{pg_i + 1}/{len(form_pages)}] "
                  f"pos={stem_stats['pos']} neg={stem_stats['neg']}", flush=True)

    print(f"  done: pages_processed={stem_stats['pages_processed']} "
          f"skipped={stem_stats['pages_skipped']} "
          f"pos={stem_stats['pos']} neg={stem_stats['neg']}", flush=True)
    manifest["splits"].setdefault(split, {})[f"{src_dir.name}/{stem}"] = stem_stats
    manifest["totals"][f"{split}_pos"] += stem_stats["pos"]
    manifest["totals"][f"{split}_neg"] += stem_stats["neg"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--limit-pages", type=int, default=None,
                    help="Only first N form-bearing pages per PDF (debug)")
    ap.add_argument("--extra-source", type=Path, action="append", default=[],
                    help="Extra dir with pdf/json pairs to ADD to the train split as more negatives. "
                         "May be passed multiple times.")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    for split in ("train", "val"):
        for cls in ("positive", "negative"):
            (args.out / split / cls).mkdir(parents=True, exist_ok=True)

    cfg_cache: dict[str, dict[str, Any] | None] = {}

    manifest: dict[str, Any] = {
        "splits": {},
        "totals": {"train_pos": 0, "train_neg": 0, "val_pos": 0, "val_neg": 0},
        "skipped_overflow_pages": 0,
        "skipped_no_template": 0,
        "skipped_ambiguous_firearms": 0,
        "extra_sources": [str(p) for p in args.extra_source],
    }

    # Standard v2 train/val
    for src_dir, stem in TRAIN_PDFS:
        _process_one(src_dir, stem, "train", args, cfg_cache, manifest, args.out)
    for src_dir, stem in VAL_PDFS:
        _process_one(src_dir, stem, "val", args, cfg_cache, manifest, args.out)

    # Extra negative sources go to TRAIN
    for src_dir, stem in _gather_extra_sources(args.extra_source):
        _process_one(src_dir, stem, "train", args, cfg_cache, manifest, args.out)

    with open(args.out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    t = manifest["totals"]
    print()
    print(f"Total positives: train={t['train_pos']}  val={t['val_pos']}")
    print(f"Total negatives: train={t['train_neg']}  val={t['val_neg']}")
    print(f"Skipped overflow pages: {manifest['skipped_overflow_pages']}")
    print(f"Skipped ambiguous firearms: {manifest['skipped_ambiguous_firearms']}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
