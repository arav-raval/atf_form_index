"""Stage 3 — Localize ROI and crop.

One crop per page: the **full serial column block** covering every firearm row.
Downstream OCR (stage 5) reads the block in one pass and splits on newlines.

Coords come from ``FormTemplates/<year>/form_config.json``:
- ``firearm_rows.columns.serial`` — left edge (pt)
- next column to the right of serial (or page width) — right edge
- ``firearm_rows.row_y`` — each entry is the **center** of a data row

Later: learned layout model replaces the static coords.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pdf2image import convert_from_path

_CROP_DPI = 200

# Extra margin (pt) added to every side of the serial block crop — both for
# annotated boxes and the fallback computation. Bump this if OCR is losing
# edge characters; drop it if captions/neighbors are bleeding in.
_BLOCK_PAD = 8.0

# Fallback-only padding (used when no serial_block annotation exists).
_PAD_X = 4.0
_PAD_TOP = 15.0
_PAD_BOTTOM = 15.0
_FALLBACK_ROW_HEIGHT = 20.0
_FALLBACK_COL_WIDTH = 130.0


def _expand_box(
    box_pts: tuple[float, float, float, float], pad: float
) -> tuple[float, float, float, float]:
    x, y, w, h = box_pts
    return (x - pad, y - pad, w + 2 * pad, h + 2 * pad)


def _load_form_config(templates_dir: Path, year: str) -> dict[str, Any] | None:
    p = templates_dir / year / "form_config.json"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _serial_block_pts(
    cfg: dict[str, Any],
) -> tuple[float, float, float, float]:
    """Return ``(left, top, width, height)`` in points for the serial column block.

    Prefers an annotated ``serial_block`` key (produced by ``pipeline.annotate``).
    Falls back to computing from ``firearm_rows.columns`` + ``row_y`` for years
    that haven't been annotated yet.
    """
    sb = cfg.get("serial_block")
    if sb and "x" in sb and "y" in sb and "width" in sb and "height" in sb:
        raw = (float(sb["x"]), float(sb["y"]), float(sb["width"]), float(sb["height"]))
        return _expand_box(raw, _BLOCK_PAD)

    fr = cfg.get("firearm_rows") or {}
    cols = fr.get("columns") or {}
    row_y = sorted(float(y) for y in (fr.get("row_y") or [400.0]))

    # Left/right: serial column, bounded on the right by the nearest column
    # that sits to the right of it (not whatever is named "type" — some older
    # forms order columns differently).
    x_serial = float(cols.get("serial", 0.0))
    page_w = float((cfg.get("page_size") or [612, 792])[0])
    x_right_candidates = [float(x) for x in cols.values() if float(x) > x_serial]
    x_right = min(x_right_candidates) if x_right_candidates else x_serial + _FALLBACK_COL_WIDTH
    if x_right <= x_serial:
        x_right = x_serial + _FALLBACK_COL_WIDTH
    x_right = min(x_right, page_w)

    # Top/bottom: row_y are centers; estimate row height from median spacing
    # (or a fallback) and expand to cover the first and last rows.
    if len(row_y) >= 2:
        gaps = [b - a for a, b in zip(row_y, row_y[1:])]
        row_h = sorted(gaps)[len(gaps) // 2]
    else:
        row_h = _FALLBACK_ROW_HEIGHT

    top = row_y[0] - row_h / 2.0 - _PAD_TOP
    bottom = row_y[-1] + row_h / 2.0 + _PAD_BOTTOM

    left = x_serial - _PAD_X
    width = (x_right - x_serial) + 2 * _PAD_X
    height = bottom - top
    return (left, top, width, height)


def _pts_to_pixels(
    box_pts: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    page_w_pt: float,
    page_h_pt: float,
) -> tuple[int, int, int, int]:
    left_pt, top_pt, w_pt, h_pt = box_pts
    sx = img_w / page_w_pt
    sy = img_h / page_h_pt
    left = max(0, int(left_pt * sx))
    top = max(0, int(top_pt * sy))
    right = min(img_w, int((left_pt + w_pt) * sx))
    bottom = min(img_h, int((top_pt + h_pt) * sy))
    if right <= left or bottom <= top:
        return 0, 0, min(1, img_w), min(1, img_h)
    return left, top, right, bottom


def rasterize_serial_page(
    pdf_path: Path, year: str, templates_dir: Path
) -> tuple[Image.Image | None, dict[str, Any] | None, int]:
    cfg = _load_form_config(templates_dir, year)
    if not cfg:
        return None, None, 0
    fr = cfg.get("firearm_rows") or {}
    page_0based = int(fr.get("page", 0))
    p1 = page_0based + 1
    pages = convert_from_path(str(pdf_path), dpi=_CROP_DPI, first_page=p1, last_page=p1)
    if not pages:
        return None, cfg, page_0based
    return pages[0].convert("RGB"), cfg, page_0based


_template_block_cache: dict[str, Image.Image] = {}


def get_template_block(
    year: str, templates_dir: Path
) -> Image.Image | None:
    """Load the blank-template's serial-block crop (cached). Returns None if the
    template file isn't found."""
    if year in _template_block_cache:
        return _template_block_cache[year]
    cfg = _load_form_config(templates_dir, year)
    if not cfg:
        return None
    template_pdf = templates_dir / year / "Form.pdf"
    if not template_pdf.is_file():
        return None
    fr = cfg.get("firearm_rows") or {}
    page_0based = int(fr.get("page", 0))
    pages = convert_from_path(
        str(template_pdf), dpi=_CROP_DPI,
        first_page=page_0based + 1, last_page=page_0based + 1,
    )
    if not pages:
        return None
    page_img = pages[0].convert("RGB")
    block, _ = crop_serial_block(page_img, cfg)
    _template_block_cache[year] = block
    return block


def crop_serial_block(
    page_image: Image.Image,
    cfg: dict[str, Any],
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Crop the full serial column block. Returns (crop, box_in_points)."""
    page_size = cfg.get("page_size") or [612, 792]
    page_w_pt, page_h_pt = float(page_size[0]), float(page_size[1])
    iw, ih = page_image.size
    box_pts = _serial_block_pts(cfg)
    L, T, R, B = _pts_to_pixels(box_pts, iw, ih, page_w_pt, page_h_pt)
    return page_image.crop((L, T, R, B)), box_pts


def remove_table_lines(crop: Image.Image, line_threshold: float = 0.5) -> Image.Image:
    """Whiten image rows/columns dominated by a continuous black line — i.e.,
    table borders.

    Leaves user ink alone (rows/columns of digits or letters have lots of
    whitespace between strokes) and only erases bands that are dominated by
    a continuous black line. Cheaper and more accurate than subtracting the
    template for OCR purposes.
    """
    arr = np.asarray(crop.convert("L"), dtype=np.float32)
    h, w = arr.shape
    if h == 0 or w == 0:
        return crop
    row_dark = 1.0 - arr.mean(axis=1) / 255.0
    col_dark = 1.0 - arr.mean(axis=0) / 255.0
    out = arr.copy()
    out[row_dark > line_threshold, :] = 255.0
    out[:, col_dark > line_threshold] = 255.0
    return Image.fromarray(out.astype(np.uint8), mode="L")


def split_block_into_rows(
    block_image: Image.Image,
    expected_rows: int,
    *,
    min_row_height_px: int = 12,
) -> list[tuple[Image.Image, tuple[int, int, int, int]]]:
    """Split a serial-column block into per-row crops.

    Uses horizontal projection of dark ink: rows of pixels with high ink density
    are "row content", low-density runs are gutters. We then split at the gutter
    midpoints. Falls back to even geometric division if the projection is too
    flat (e.g., empty block).

    Returns list of ``(row_crop, (L,T,R,B) within block)``.
    """
    arr = np.asarray(block_image.convert("L"), dtype=np.float32)
    h, w = arr.shape
    if h == 0 or w == 0:
        return []

    # Ink density per row: 1.0 = totally black, 0.0 = totally white
    ink = 1.0 - arr.mean(axis=1) / 255.0

    # Smooth with a small box filter so single-pixel noise doesn't create
    # spurious gutters.
    k = max(3, h // 60)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    smoothed = np.convolve(np.pad(ink, pad, mode="edge"), np.ones(k) / k, mode="valid")

    # A row is a "content" stretch if it's above this threshold; gutters are
    # contiguous below-threshold stretches between them.
    thresh = max(0.02, smoothed.mean() * 0.6)
    is_content = smoothed > thresh

    # Find runs of content
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, c in enumerate(is_content):
        if c and not in_run:
            start = i
            in_run = True
        elif not c and in_run:
            if i - start >= min_row_height_px:
                runs.append((start, i))
            in_run = False
    if in_run and h - start >= min_row_height_px:
        runs.append((start, h))

    # If projection didn't yield enough rows (e.g., faint ink), fall back to
    # even geometric split. This is the common case for completely empty blocks.
    if len(runs) < 1:
        runs = []
        step = h / max(1, expected_rows)
        for i in range(expected_rows):
            top = int(i * step)
            bot = int((i + 1) * step) if i + 1 < expected_rows else h
            runs.append((top, bot))

    # Pad each run vertically so Tesseract has whitespace above and below the
    # characters. ~12 px at 200dpi ≈ 4pt — about one line-leading.
    # Trim the right margin so adjacent-column noise (model name, type field
    # next to the serial column) doesn't leak into OCR. The annotated block
    # already sits over the serial column, but the rightmost ~8% commonly
    # contains the start of the next column's text.
    pad_y = 12
    right_trim = int(w * 0.08)
    R_eff = max(1, w - right_trim)
    out: list[tuple[Image.Image, tuple[int, int, int, int]]] = []
    for top, bot in runs:
        T = max(0, top - pad_y)
        B = min(h, bot + pad_y)
        L, R = 0, R_eff
        out.append((block_image.crop((L, T, R, B)), (L, T, R, B)))
    return out
