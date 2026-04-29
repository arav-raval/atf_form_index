"""Stage 5 — OCR a verified ROI.

Stage 4 hands us per-row crops that have already been screened for "looks
like a serial number." We OCR each row independently and apply a pattern
filter that rejects strings that don't look serial-shaped — defense in depth
against verifier false positives.

Backends, in order of preference:
  1. TrOCR (HuggingFace ``microsoft/trocr-base-*``) — when ``transformers`` is
     installed. Trained on isolated text lines; substantially better on serial
     numbers than Tesseract.
  2. Tesseract via pytesseract — fallback when TrOCR isn't available, e.g. in
     environments without ``transformers``/``torch``.

Pick the backend with ``OCR_BACKEND`` env var (``trocr`` | ``tesseract`` |
``auto``, default ``auto``).
"""
from __future__ import annotations

import logging
import os
import re

from PIL import Image

log = logging.getLogger(__name__)

_WHITELIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-"
_PSM = 7  # treat the input as a single text line

_BACKEND = os.environ.get("OCR_BACKEND", "auto").lower()

# A real ATF serial is alphanumeric, uppercase, dashes allowed. Length 3–18 covers
# the realistic range. Reject lines that are pure dashes or that contain no digits.
_SERIAL_RE = re.compile(r"^[A-Z0-9-]{3,18}$")


def normalize_serial(text: str) -> str:
    return "".join(c for c in text.upper() if c.isalnum())


def looks_like_serial(text: str) -> bool:
    """Pattern tripwire: rejects lines that aren't serial-shaped."""
    s = text.strip().upper()
    if not _SERIAL_RE.match(s):
        return False
    if not any(c.isdigit() for c in s):
        return False
    if not any(c.isalnum() for c in s):
        return False
    return True


_SERIAL_TOKEN_RE = re.compile(r"[A-Z0-9-]{3,18}")


def best_serial_token(text: str) -> str:
    """Return the longest serial-shaped token in ``text``, or ''.

    OCR commonly attaches single-character noise from adjacent columns (e.g.,
    "15GS055 F"). Splitting on whitespace and picking the longest token that
    passes :func:`looks_like_serial` recovers the serial from such cases.
    """
    s = text.strip().upper()
    candidates: list[str] = []
    for tok in _SERIAL_TOKEN_RE.findall(s):
        if looks_like_serial(tok):
            candidates.append(tok)
    if not candidates:
        return ""
    return max(candidates, key=len)


# Tesseract performs noticeably better with larger character heights. We
# upscale row crops aggressively before OCR; 140 px maps to ~80 px x-height,
# inside Tesseract's documented sweet spot.
_OCR_TARGET_HEIGHT = 140


def _upscale_for_ocr(crop: Image.Image) -> Image.Image:
    w, h = crop.size
    if h <= 0 or h >= _OCR_TARGET_HEIGHT:
        return crop
    scale = _OCR_TARGET_HEIGHT / h
    return crop.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _recognize_tesseract(crop: Image.Image) -> tuple[str, str]:
    try:
        import pytesseract
    except ImportError:
        return "", "no_tesseract"
    try:
        from pipeline import localize as _loc
        cleaned = _loc.remove_table_lines(crop)
        gray = _upscale_for_ocr(cleaned.convert("L"))
        text = pytesseract.image_to_string(
            gray, config=f"--psm {_PSM} -c tessedit_char_whitelist={_WHITELIST}"
        ).strip()
        return text, "tesseract"
    except Exception as e:
        log.warning("Tesseract failed: %s", e)
        return "", "ocr_error"


def _recognize_trocr(crop: Image.Image) -> tuple[str, str]:
    from pipeline import ocr_trocr

    if not ocr_trocr.is_available():
        return "", "trocr_unavailable"
    rec = ocr_trocr.recognize_trocr(crop)
    return rec.get("text", ""), rec.get("method", "trocr_error")


def recognize(roi_crop: Image.Image) -> dict:
    """OCR a single-line row crop. Returns ``{text, normalized, looks_serial, method}``."""
    text, method = "", "none"
    if _BACKEND in ("trocr", "auto"):
        text, method = _recognize_trocr(roi_crop)
    if (not text) and _BACKEND in ("tesseract", "auto"):
        # Tesseract fallback when TrOCR is unavailable or returned nothing.
        t_text, t_method = _recognize_tesseract(roi_crop)
        if t_text or method == "trocr_unavailable":
            text, method = t_text, t_method

    # Prefer the best serial-shaped token if the raw output has noise glued on
    # from adjacent columns (e.g., "15GS055 F" → "15GS055").
    best = best_serial_token(text)
    final = best if best else text
    return {
        "text": text,
        "normalized": normalize_serial(final),
        "looks_serial": bool(best) or looks_like_serial(text),
        "method": method,
    }
