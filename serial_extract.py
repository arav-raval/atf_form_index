"""
Crop the firearm serial field from a PDF page using FormTemplates/<year>/form_config.json
coordinates, then run OCR (pytesseract) when available.

Coordinates are treated as **top-left** origin in the same units as ``page_size``
(points). If OCR is unavailable or fails, ``text`` may be empty — the pipeline
still records ``method`` for evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image
from pdf2image import convert_from_path

log = logging.getLogger(__name__)

# Rasterization DPI for field crops (higher helps OCR)
_CROP_DPI = 200


def _load_form_config(templates_dir: Path, year: str) -> dict[str, Any] | None:
    p = templates_dir / year / "form_config.json"
    if not p.is_file():
        log.warning("No form_config for year %s at %s", year, p)
        return None
    import json

    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _serial_region_pts(cfg: dict[str, Any], row_index: int = 0) -> tuple[float, float, float, float]:
    """
    Return crop box in **points** (left, top, width, height) relative to page top-left.
    """
    fr = cfg.get("firearm_rows") or {}
    cols = fr.get("columns") or {}
    row_y = fr.get("row_y") or [400]
    x0 = float(cols.get("serial", 0))
    # Serial field spans until the "type" column (or fixed width)
    x1 = float(cols.get("type", x0 + 140))
    if x1 <= x0:
        x1 = x0 + 140
    y0 = float(row_y[row_index]) if row_index < len(row_y) else float(row_y[0])
    if row_index + 1 < len(row_y):
        y1 = float(row_y[row_index + 1])
    else:
        y1 = y0 + 18.0
    pad = 2.0
    return (x0 - pad, y0 - pad, (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad)


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


def extract_serial_from_pdf(
    pdf_path: str | Path,
    year: str,
    templates_dir: str | Path,
    *,
    row_index: int = 0,
) -> dict[str, Any]:
    """
    Rasterize the firearm_rows page, crop the serial column for ``row_index``, OCR.

    Returns:
        ``text`` (raw), ``normalized`` (alphanumeric upper), ``method``, ``page`` (0-based).
    """
    pdf_path = Path(pdf_path)
    templates_dir = Path(templates_dir)
    cfg = _load_form_config(templates_dir, year)
    if not cfg:
        return {
            "text": "",
            "normalized": "",
            "method": "no_config",
            "page": 0,
        }

    page_size = cfg.get("page_size") or [612, 792]
    page_w_pt, page_h_pt = float(page_size[0]), float(page_size[1])
    fr = cfg.get("firearm_rows") or {}
    page_0based = int(fr.get("page", 0))

    p1 = page_0based + 1
    pages = convert_from_path(
        str(pdf_path),
        dpi=_CROP_DPI,
        first_page=p1,
        last_page=p1,
    )
    if not pages:
        return {"text": "", "normalized": "", "method": "no_page", "page": page_0based}

    img = pages[0].convert("RGB")
    iw, ih = img.size
    box_pts = _serial_region_pts(cfg, row_index)
    L, T, R, B = _pts_to_pixels(box_pts, iw, ih, page_w_pt, page_h_pt)
    crop = img.crop((L, T, R, B))

    text = ""
    method = "none"
    try:
        import pytesseract

        gray = crop.convert("L")
        text = pytesseract.image_to_string(gray, config="--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-").strip()
        method = "tesseract"
    except ImportError:
        log.debug("pytesseract not installed; serial OCR skipped")
        method = "no_tesseract"
    except Exception as e:
        log.warning("OCR failed: %s", e)
        method = "ocr_error"

    normalized = normalize_serial(text)
    return {
        "text": text,
        "normalized": normalized,
        "method": method,
        "page": page_0based,
    }


def normalize_serial(text: str) -> str:
    """Alphanumeric only, upper — for index lookup."""
    return "".join(c for c in text.upper() if c.isalnum())


def loose_serial_match(query: str, candidate: str) -> bool:
    """Exact after normalization, or prefix if query short."""
    q = normalize_serial(query)
    c = normalize_serial(candidate)
    if not q or not c:
        return False
    return q == c or (len(q) >= 4 and c.startswith(q))
