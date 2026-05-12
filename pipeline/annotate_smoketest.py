"""Smoke-test serial_block annotations.

For each year (or one year), render the actual crop the pipeline will
extract from a sample real PDF using the current ``serial_block`` annotation,
plus a side-by-side visualization showing the page with the box drawn on top.
You eyeball the result to confirm the annotation is right.

Saves PNG files to ``annotation_smoketest/`` for review:
    annotation_smoketest/
      <year>_template_with_box.png   ← the form template, box drawn on top
      <year>_template_crop.png        ← what the pipeline crops from the template
      <year>_real_<source>.png        ← crops from real PDFs (CR/v2/master)

Usage::

    python -m pipeline.annotate_smoketest 1998
    python -m pipeline.annotate_smoketest --all
    python -m pipeline.annotate_smoketest 1998 --include-real

Then open the saved PNGs and check:
  1. Does the crop start at the TOP of the topmost data cell? (no header bleed)
  2. Does it end at the BOTTOM of the bottommost data cell? (no footer)
  3. Does it cover the FULL width of every serial? (no left/right truncation)
  4. Does the same crop look right on REAL PDFs (CR/master), not just templates?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from pdf2image import convert_from_path

from pipeline import localize

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "FormTemplates"
OUT_DIR = ROOT / "annotation_smoketest"

# Sample real PDFs per year for the "does this annotation work on real data?"
# check. These are MasterTesting PDFs (held-out), so verifying here doesn't
# leak training data.
_SAMPLES_PER_YEAR: dict[str, list[tuple[str, str, int]]] = {
    # year -> [(label, pdf_path_relative_to_repo, pdf_page_1based)]
    "1985": [("complete_test_p1", "MasterTesting/1985_complete_test.pdf", 2)],
    "1998": [("complete_test_p1", "MasterTesting/1998_complete_test.pdf", 2)],
    "2001": [("complete_test_p1", "MasterTesting/2001_complete_test.pdf", 2)],
    "2005": [("complete_test_p1", "MasterTesting/2005_complete_test.pdf", 2)],
    "2007": [("complete_test_p1", "MasterTesting/2007_complete_test.pdf", 2)],
    "2008": [("complete_test_p1", "MasterTesting/2008_complete_test.pdf", 3)],
    "2012": [("complete_test_p1", "MasterTesting/2012_complete_test.pdf", 3)],
    "2016": [("complete_test_p1", "MasterTesting/2016_complete_test.pdf", 3)],
    "2020": [("complete_test_p1", "MasterTesting/2020_complete_test.pdf", 1),
             ("cont_test", "MasterTesting/2020_cont_test.pdf", 1)],
    "2022": [("complete_test_p1", "MasterTesting/2022_complete_test.pdf", 1),
             ("cont_test", "MasterTesting/2022_cont_test.pdf", 1)],
    "2023": [("complete_test_p1", "MasterTesting/2023_complete_test.pdf", 1),
             ("cont_test", "MasterTesting/2023_cont_test.pdf", 1)],
}


def _draw_box_overlay(img: Image.Image, box_pt: tuple[float, float, float, float],
                       page_w_pt: float, page_h_pt: float) -> Image.Image:
    """Render the page with the serial_block drawn as a red rectangle."""
    iw, ih = img.size
    sx = iw / page_w_pt
    sy = ih / page_h_pt
    x, y, w, h = box_pt
    L, T = int(x * sx), int(y * sy)
    R, B = int((x + w) * sx), int((y + h) * sy)

    out = img.convert("RGB").copy()
    drw = ImageDraw.Draw(out)
    # Outline + faint fill
    drw.rectangle([L, T, R, B], outline=(255, 0, 0), width=4)
    # Crosshairs at corners (helps spot off-by-N alignment)
    for cx, cy in [(L, T), (R, T), (L, B), (R, B)]:
        drw.line([(cx - 12, cy), (cx + 12, cy)], fill=(255, 0, 0), width=2)
        drw.line([(cx, cy - 12), (cx, cy + 12)], fill=(255, 0, 0), width=2)
    return out


def _crop_serial_block_from_image(img: Image.Image, cfg: dict) -> Image.Image:
    """Same crop pipeline.localize.crop_serial_block uses, but on an
    already-rasterized image."""
    block, _ = localize.crop_serial_block(img, cfg)
    return block


def smoketest_one(year: str, include_real: bool, log: bool = True) -> None:
    cfg_path = TEMPLATES / year / "form_config.json"
    if not cfg_path.is_file():
        print(f"  [{year}] no form_config.json — skipping")
        return
    cfg = json.load(open(cfg_path))
    sb = cfg.get("serial_block")
    if not sb:
        print(f"  [{year}] no serial_block — annotate first")
        return

    OUT_DIR.mkdir(exist_ok=True)
    page_w_pt, page_h_pt = (cfg.get("page_size") or [612, 792])[:2]
    page_0based = int((cfg.get("firearm_rows") or {}).get("page", 0))
    box_pt = (sb["x"], sb["y"], sb["width"], sb["height"])

    if log:
        print(f"\n[{year}] serial_block (pt): "
              f"x={sb['x']:.1f} y={sb['y']:.1f} w={sb['width']:.1f} h={sb['height']:.1f}")

    # 1) Template with box overlay + template crop
    template_pdf = TEMPLATES / year / "Form.pdf"
    if template_pdf.is_file():
        pages = convert_from_path(str(template_pdf), dpi=200,
                                   first_page=page_0based + 1, last_page=page_0based + 1)
        if pages:
            page_img = pages[0].convert("RGB")
            overlay = _draw_box_overlay(page_img, box_pt, page_w_pt, page_h_pt)
            overlay.save(OUT_DIR / f"{year}_template_with_box.png")
            crop = _crop_serial_block_from_image(page_img, cfg)
            crop.save(OUT_DIR / f"{year}_template_crop.png")
            if log:
                print(f"  wrote {year}_template_with_box.png + {year}_template_crop.png")

    # 2) Real-distribution PDFs (held-out MasterTesting samples)
    if include_real and year in _SAMPLES_PER_YEAR:
        for label, pdf_rel, page_1based in _SAMPLES_PER_YEAR[year]:
            pdf_path = ROOT / pdf_rel
            if not pdf_path.is_file():
                print(f"  [{year}] sample missing: {pdf_rel}")
                continue
            try:
                pages = convert_from_path(str(pdf_path), dpi=200,
                                           first_page=page_1based, last_page=page_1based)
            except Exception as e:
                print(f"  [{year}] {label}: rasterize failed ({e})")
                continue
            if not pages:
                continue
            page_img = pages[0].convert("RGB")
            overlay = _draw_box_overlay(page_img, box_pt, page_w_pt, page_h_pt)
            overlay.save(OUT_DIR / f"{year}_real_{label}_with_box.png")
            crop = _crop_serial_block_from_image(page_img, cfg)
            crop.save(OUT_DIR / f"{year}_real_{label}_crop.png")
            if log:
                print(f"  wrote {year}_real_{label}_*.png  (from {pdf_rel} p{page_1based})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("year", nargs="?", default=None,
                    help="Year to test. Omit + use --all to test every year.")
    ap.add_argument("--all", action="store_true",
                    help="Test every year under FormTemplates/")
    ap.add_argument("--include-real", action="store_true",
                    help="Also crop a sample real PDF per year (held-out MasterTesting). "
                         "STRONGLY RECOMMENDED — annotation looks right on the template "
                         "but wrong on real PDFs is the most common failure mode.")
    args = ap.parse_args()

    years: list[str]
    if args.all:
        years = sorted(p.name for p in TEMPLATES.iterdir()
                       if p.is_dir() and (p / "form_config.json").is_file())
    elif args.year:
        years = [args.year]
    else:
        ap.error("provide a year or --all")

    print(f"Smoke-testing serial_block annotations for {len(years)} year(s)...")
    print(f"Output dir: {OUT_DIR}")
    print()
    print("WHAT TO LOOK FOR in each saved PNG:")
    print("  *_template_with_box.png  — RED box should tightly enclose just the")
    print("                              serial-column DATA cells (not the header)")
    print("  *_template_crop.png       — should show ONLY the serial-column cells,")
    print("                              top to bottom of the table")
    print("  *_real_*_*.png            — same checks on a real PDF; if these look")
    print("                              wrong but template looks right, the real")
    print("                              PDFs use a slightly different layout")
    print()
    for year in years:
        smoketest_one(year, include_real=args.include_real)
    print(f"\nOpen {OUT_DIR}/ to review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
