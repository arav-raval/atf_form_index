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


_DEFAULT_NUM_HYPOTHESES = int(os.environ.get("OCR_NUM_HYPOTHESES", "5"))


def _hypotheses_from_trocr(crop: Image.Image, num: int) -> tuple[list[dict], str]:
    """Return (list of {text, normalized, score}, method)."""
    from pipeline import ocr_trocr

    if not ocr_trocr.is_available():
        return [], "trocr_unavailable"

    # Printed-first beam search; fall back to handwritten if no hypothesis
    # passes the serial-shape regex (handwriting is the harder distribution).
    rec_pr = ocr_trocr.recognize_trocr(crop, variant="printed", num_hypotheses=num)
    raw_pr = rec_pr.get("hypotheses", []) or [{"text": rec_pr.get("text", ""), "score": 0.0}]
    method_pr = rec_pr.get("method", "trocr_error")
    pr_hyps = _filter_and_dedup_hypotheses(raw_pr)
    if pr_hyps:
        return pr_hyps, method_pr

    rec_hw = ocr_trocr.recognize_trocr(crop, variant="handwritten", num_hypotheses=num)
    raw_hw = rec_hw.get("hypotheses", []) or [{"text": rec_hw.get("text", ""), "score": 0.0}]
    method_hw = rec_hw.get("method", "trocr_error")
    hw_hyps = _filter_and_dedup_hypotheses(raw_hw)
    if hw_hyps:
        return hw_hyps, method_hw

    # Neither produced a serial-shape hit — return printed top-1 anyway so
    # downstream sees something for diagnostics; index won't admit it (no
    # token passes ``looks_like_serial``).
    if raw_pr:
        return [{"text": raw_pr[0]["text"],
                 "normalized": normalize_serial(raw_pr[0]["text"]),
                 "score": float(raw_pr[0].get("score", 0.0))}], method_pr
    return [], method_pr


def _hypotheses_from_trocr_batch(crops: list[Image.Image], num: int) -> list[tuple[list[dict], str]]:
    """Batched dual-pass TrOCR: ALL crops through printed first, then any
    that didn't yield a serial-shaped token go through handwritten as a
    second batch. Returns one (hyps, method) tuple per input crop, in order.

    Substantially faster than calling :func:`_hypotheses_from_trocr` in a
    loop because both passes vectorize across crops.
    """
    from pipeline import ocr_trocr

    if not crops:
        return []
    if not ocr_trocr.is_available():
        return [([], "trocr_unavailable")] * len(crops)

    # Pass 1: printed for everyone
    rec_pr = ocr_trocr.recognize_trocr_batch(crops, variant="printed", num_hypotheses=num)
    out: list[tuple[list[dict], str] | None] = [None] * len(crops)
    fallback_idxs: list[int] = []

    for i, rec in enumerate(rec_pr):
        raw = rec.get("hypotheses", []) or [{"text": rec.get("text", ""), "score": 0.0}]
        method = rec.get("method", "trocr_error")
        hyps = _filter_and_dedup_hypotheses(raw)
        if hyps:
            out[i] = (hyps, method)
        else:
            fallback_idxs.append(i)
            # remember the printed top-1 in case handwritten also fails
            out[i] = (raw, method)  # placeholder; will be overwritten

    # Pass 2: handwritten for the leftovers
    if fallback_idxs:
        leftover = [crops[i] for i in fallback_idxs]
        rec_hw = ocr_trocr.recognize_trocr_batch(leftover, variant="handwritten", num_hypotheses=num)
        for j, i in enumerate(fallback_idxs):
            rec = rec_hw[j]
            raw = rec.get("hypotheses", []) or [{"text": rec.get("text", ""), "score": 0.0}]
            method = rec.get("method", "trocr_error")
            hyps = _filter_and_dedup_hypotheses(raw)
            if hyps:
                out[i] = (hyps, method)
            else:
                # Both passes failed regex. Fall back to printed top-1 as
                # diagnostic surface (won't admit downstream because the
                # token doesn't pass the regex).
                printed_raw = rec_pr[i].get("hypotheses", []) or []
                if printed_raw:
                    out[i] = (
                        [{"text": printed_raw[0]["text"],
                          "normalized": normalize_serial(printed_raw[0]["text"]),
                          "score": float(printed_raw[0].get("score", 0.0))}],
                        rec_pr[i].get("method", "trocr_error"),
                    )
                else:
                    out[i] = ([], rec_pr[i].get("method", "trocr_error"))
    return out  # type: ignore[return-value]


def recognize_batch(
    crops: list[Image.Image], num_hypotheses: int | None = None,
) -> list[dict]:
    """Batched OCR. Returns one dict per input crop, same shape as
    :func:`recognize`. Falls back to single-crop calls if Tesseract is the
    only available backend (Tesseract has no batch API)."""
    if not crops:
        return []
    n = max(1, num_hypotheses if num_hypotheses is not None else _DEFAULT_NUM_HYPOTHESES)

    if _BACKEND in ("trocr", "auto"):
        from pipeline import ocr_trocr
        if ocr_trocr.is_available():
            results = _hypotheses_from_trocr_batch(crops, n)
            out: list[dict] = []
            for hyps, method in results:
                if hyps:
                    top = hyps[0]
                    out.append({
                        "text": top["text"],
                        "normalized": top.get("normalized") or normalize_serial(top["text"]),
                        "looks_serial": bool(top.get("normalized")),
                        "method": method,
                        "hypotheses": hyps,
                    })
                else:
                    out.append({
                        "text": "", "normalized": "", "looks_serial": False,
                        "method": method, "hypotheses": [],
                    })
            return out

    # Tesseract fallback — no batch, call recognize() per crop
    return [recognize(c, num_hypotheses=n) for c in crops]


def _filter_and_dedup_hypotheses(raw: list[dict]) -> list[dict]:
    """Keep only hypotheses with a serial-shape token, dedupe by normalized
    form (best score wins). Returns list of {text, normalized, score}."""
    out: dict[str, dict] = {}  # normalized -> best entry
    for h in raw:
        raw_text = (h.get("text") or "").strip()
        tok = best_serial_token(raw_text)
        if not tok:
            continue
        norm = normalize_serial(tok)
        if not norm:
            continue
        score = float(h.get("score", 0.0))
        if norm not in out or score > out[norm]["score"]:
            out[norm] = {"text": raw_text, "normalized": norm, "score": score}
    return sorted(out.values(), key=lambda d: -d["score"])


def recognize(roi_crop: Image.Image, num_hypotheses: int | None = None) -> dict:
    """OCR a row crop. Returns
    ``{text, normalized, looks_serial, method, hypotheses}``.

    ``hypotheses`` is a list of ``{text, normalized, score}`` of length up to
    ``num_hypotheses`` (default :data:`_DEFAULT_NUM_HYPOTHESES`), each having
    already passed the serial-shape regex. ``text`` / ``normalized`` are the
    top-1 hypothesis; if no hypothesis passes the regex, ``hypotheses`` is
    empty and ``looks_serial`` is False.

    Tesseract is used only when TrOCR is unavailable; it returns a single
    hypothesis.
    """
    n = max(1, num_hypotheses if num_hypotheses is not None else _DEFAULT_NUM_HYPOTHESES)

    hyps: list[dict] = []
    method = "none"
    if _BACKEND in ("trocr", "auto"):
        hyps, method = _hypotheses_from_trocr(roi_crop, n)

    if not hyps and _BACKEND in ("tesseract", "auto"):
        t_text, t_method = _recognize_tesseract(roi_crop)
        if t_text:
            tok = best_serial_token(t_text)
            if tok:
                hyps = [{"text": t_text,
                         "normalized": normalize_serial(tok),
                         "score": 0.0}]
            method = t_method
        elif method == "trocr_unavailable":
            method = t_method

    if hyps:
        top = hyps[0]
        return {
            "text": top["text"],
            "normalized": top["normalized"],
            "looks_serial": True,
            "method": method,
            "hypotheses": hyps,
        }
    return {
        "text": "",
        "normalized": "",
        "looks_serial": False,
        "method": method,
        "hypotheses": [],
    }
