"""Visual sanity check for stages 2+3.

Samples N labeled PDFs, classifies each, crops the full serial column block
using the predicted year's ``form_config.json``, and writes debug images to
``crops_debug/``:

- ``<stem>_page.png`` — full serial-bearing page with the ROI outlined
- ``<stem>_block.png`` — the cropped serial block (what stage 5 OCRs)

Run::

    python -m pipeline.visualize_crops --n 5
    python -m pipeline.visualize_crops --n 10 --seed 7 --out /tmp/crops
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import ImageDraw

from page_sampling_pipeline import _discover_labeled_pdfs
from pipeline import classify, localize

ROOT = Path(__file__).resolve().parent.parent
FORM_TEMPLATES = ROOT / "FormTemplates"


def _box_pts_to_pixels(box_pts, img_size, page_size) -> tuple[int, int, int, int]:
    iw, ih = img_size
    page_w_pt, page_h_pt = page_size
    left_pt, top_pt, w_pt, h_pt = box_pts
    sx = iw / page_w_pt
    sy = ih / page_h_pt
    return (
        int(left_pt * sx),
        int(top_pt * sy),
        int((left_pt + w_pt) * sx),
        int((top_pt + h_pt) * sy),
    )


def visualize(pdf_path: Path, true_year: str, out_dir: Path) -> dict:
    """Classify + crop every row. Returns a summary dict."""
    cr = classify.classify_pdf(pdf_path, FORM_TEMPLATES)
    pred_year = cr.get("label")
    score = float(cr.get("score") or 0.0)

    page_img, cfg, page_0based = localize.rasterize_serial_page(
        pdf_path, pred_year, FORM_TEMPLATES
    )
    if page_img is None or cfg is None:
        return {
            "pdf": pdf_path.name,
            "true_year": true_year,
            "pred_year": pred_year,
            "score": score,
            "error": "rasterize_failed",
        }

    page_size = cfg.get("page_size") or [612, 792]
    stem = pdf_path.stem

    roi, box_pts = localize.crop_serial_block(page_img, cfg)
    L, T, R, B = _box_pts_to_pixels(box_pts, page_img.size, page_size)

    annotated = page_img.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle([L, T, R, B], outline="red", width=3)
    draw.text((L + 4, T + 2), "serial block", fill="red")

    block_path = out_dir / f"{stem}_block.png"
    roi.save(block_path)
    page_path = out_dir / f"{stem}_page.png"
    annotated.save(page_path)

    return {
        "pdf": pdf_path.name,
        "true_year": true_year,
        "pred_year": pred_year,
        "year_correct": pred_year == true_year,
        "score": round(score, 4),
        "page_0based": page_0based,
        "block_px": (R - L, B - T),
        "page_image": page_path.name,
        "block_image": block_path.name,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=5, help="Number of PDFs to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=ROOT / "crops_debug")
    p.add_argument(
        "--year",
        type=str,
        default=None,
        help="Only sample PDFs with this ground-truth year (e.g. 2016)",
    )
    args = p.parse_args()

    labeled = _discover_labeled_pdfs()
    if args.year:
        labeled = [(pdf, yr) for pdf, yr in labeled if yr == args.year]
    if not labeled:
        print("No labeled PDFs found.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    sample = rng.sample(labeled, min(args.n, len(labeled)))

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(sample)} sample(s) to {args.out}")
    print("-" * 72)

    for pdf, true_year in sample:
        info = visualize(pdf, true_year, args.out)
        mark = "✓" if info.get("year_correct") else "✗"
        block_sz = info.get("block_px")
        print(
            f"{mark} {info['pdf']:40s}  true={info['true_year']}  "
            f"pred={info.get('pred_year')}  score={info.get('score')}  "
            f"block_px={block_sz}"
        )

    print("-" * 72)
    print(f"Open {args.out} to inspect.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
