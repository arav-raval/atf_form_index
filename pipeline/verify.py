"""Stage 4 — Verify a row crop visually before any OCR runs.

This is the compliance gate. It decides, using image features only, whether a
row crop "looks like" a serial number. OCR (stage 5) runs only on rows this
function approves; rejected rows are never read as text.

Two backends, picked at call time:
  1. ML — TrOCR ViT encoder (frozen) + MLP head trained on v2 row crops with
     corruption-aware labels. Loaded lazily from ``verifier_model_vit.pt``.
     Returns ``reason="ml_vit"``. Best PII compliance.
  2. Heuristic — density + projection + template subtraction rules. Used when
     the ML checkpoint is missing or torch isn't installed.

Heuristic verdict reasons:
    ok            — passed all checks
    empty         — too little ink (no firearm entered in that row)
    bleed_top     — ink concentrated at top edge (descender from row above)
    bleed_bottom  — ink concentrated at bottom edge (ascender from row below)
    multi_band    — multiple horizontal ink bands (more than one line of text)
    too_dense     — way too much ink (likely a name field or table border)
    template_only — row is nearly identical to the blank template (pre-printed
                    form chrome — column header, watermark, captions)
ML backend always returns ``reason="ml_vit"``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH_VIT = _REPO_ROOT / "verifier_model_vit.pt"
_MODEL_CACHE: dict | None = None  # populated on first verify() call

# Optional override of the saved decision threshold. Useful for sweeping the
# recall/leak tradeoff at eval time without retraining.
_THRESHOLD_OVERRIDE: float | None = None
try:
    _env_thresh = os.environ.get("VERIFIER_THRESHOLD")
    if _env_thresh:
        _THRESHOLD_OVERRIDE = float(_env_thresh)
except ValueError:
    pass


def _try_load_model() -> dict | None:
    """Load the trained ML verifier on first call. Cached.
    Returns None if torch isn't installed or the checkpoint isn't present.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not _MODEL_PATH_VIT.is_file():
        return None
    try:
        import torch
    except ImportError:
        return None
    try:
        ckpt = torch.load(_MODEL_PATH_VIT, map_location="cpu", weights_only=False)
        from pipeline.verifier_train_vit import _Head as VitHead
        head = VitHead(in_dim=int(ckpt["head_in_dim"]))
        head.load_state_dict(ckpt["head_state_dict"])
        head.eval()
        _MODEL_CACHE = {
            "head": head,
            "scaler_mean": ckpt["scaler_mean"],
            "scaler_scale": ckpt["scaler_scale"],
            "decision_threshold": float(ckpt.get("decision_threshold", 0.9)),
            "encoder_hf_id": ckpt.get("encoder_hf_id", "microsoft/trocr-base-printed"),
            "_encoder_bundle": None,  # lazy
        }
        return _MODEL_CACHE
    except Exception:
        return None


def _vit_encoder_bundle(cache: dict) -> tuple:
    """Lazily load the TrOCR encoder shared with the verifier."""
    if cache.get("_encoder_bundle") is not None:
        return cache["_encoder_bundle"]
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    hf_id = cache["encoder_hf_id"]
    processor = TrOCRProcessor.from_pretrained(hf_id)
    full = VisionEncoderDecoderModel.from_pretrained(hf_id)
    encoder = full.encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    bundle = (processor, encoder, device)
    cache["_encoder_bundle"] = bundle
    return bundle


def _ml_verify(crop: Image.Image, cache: dict) -> dict:
    """Run the ViT verifier. Returns ``{is_serial, confidence, reason}``."""
    import torch

    processor, encoder, device = _vit_encoder_bundle(cache)
    rgb = crop.convert("RGB")
    pixel_values = processor(images=rgb, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        out = encoder(pixel_values=pixel_values)
        feats = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
    x = (feats - cache["scaler_mean"]) / cache["scaler_scale"]
    x = torch.from_numpy(x).unsqueeze(0)
    head = cache["head"]
    with torch.no_grad():
        logits = head(x)
        probs = torch.softmax(logits, dim=1)[0]
    p_pos = float(probs[1])
    threshold = _THRESHOLD_OVERRIDE if _THRESHOLD_OVERRIDE is not None else cache.get("decision_threshold", 0.9)
    is_serial = p_pos >= threshold
    return {
        "is_serial": is_serial,
        "confidence": round(p_pos if is_serial else 1 - p_pos, 3),
        "p_pos": round(p_pos, 4),  # raw P(positive) — for callers that
                                   # want to apply a different threshold
        "reason": "ml_vit",
    }

# Ink density (fraction of dark pixels) bounds that a serial row should fall in.
_MIN_INK = 0.012   # below this, row is empty
_MAX_INK = 0.32    # above this, row is too crowded to be a single short serial

# Edge-band fraction: if this share of total ink lies in the top/bottom strip,
# we treat it as bleed from a neighbor.
_EDGE_FRAC = 0.18
_EDGE_BAND_RATIO = 0.55  # share of ink that must be in the edge strip to flag

# Multi-band: if there are 3+ distinct dark bands, it's not one line of text.
_BAND_COUNT_LIMIT = 2


def _ink_mask(crop: Image.Image) -> np.ndarray:
    """Binary mask: True where pixel is darker than ~halfway."""
    arr = np.asarray(crop.convert("L"), dtype=np.float32)
    return arr < 160.0  # 160/255: tolerant of mid-gray scan artifacts


def _user_ink_mask(
    crop: Image.Image, template_crop: Image.Image | None
) -> np.ndarray:
    """Binary mask of user-added ink only.

    When ``template_crop`` is provided, we subtract the template's grayscale
    intensity from the data crop before thresholding. Pre-printed form chrome
    (table borders, column headers, watermarks) cancels; only ink the user
    added to this PDF survives.

    Falls back to a plain ink mask when no template is available.
    """
    if template_crop is None:
        return _ink_mask(crop)
    a = np.asarray(crop.convert("L").resize(template_crop.size), dtype=np.float32)
    b = np.asarray(template_crop.convert("L"), dtype=np.float32)
    # User added darker pixels where template was lighter; we want a positive
    # signal there. Clip so chrome (where data >= template) zeros out.
    residual = np.clip(b - a, 0, 255)
    return residual > 40.0  # tolerant threshold; small enough to catch faint ink


def _band_count(ink_per_row: np.ndarray, threshold: float) -> int:
    above = ink_per_row > threshold
    bands = 0
    in_band = False
    for v in above:
        if v and not in_band:
            bands += 1
            in_band = True
        elif not v:
            in_band = False
    return bands


def verify(
    roi_crop: Image.Image,
    template_crop: Image.Image | None = None,
) -> dict:
    """Return ``{is_serial, confidence, reason}``.

    Prefers the trained ML verifier (``verifier_model_vit.pt``) when available;
    falls back to the heuristic backend below. The heuristic backend operates
    on the **user-added ink mask** (template subtracted when ``template_crop``
    is supplied).
    """
    cache = _try_load_model()
    if cache is not None:
        return _ml_verify(roi_crop, cache)

    mask = _user_ink_mask(roi_crop, template_crop)
    h, w = mask.shape
    if h == 0 or w == 0:
        return {"is_serial": False, "confidence": 1.0, "reason": "empty"}

    total = float(mask.sum())
    area = float(h * w)
    ink_frac = total / area

    if template_crop is not None and ink_frac < _MIN_INK:
        # Distinguish "blank user content" from "no content at all": when we
        # have a template and there's no user ink, the row is form chrome.
        return {"is_serial": False, "confidence": 0.95, "reason": "template_only"}
    if ink_frac < _MIN_INK:
        return {"is_serial": False, "confidence": 0.95, "reason": "empty"}
    if ink_frac > _MAX_INK:
        return {"is_serial": False, "confidence": 0.85, "reason": "too_dense"}

    # Edge-band check: is the ink mostly clinging to the top or bottom?
    band_h = max(2, int(h * _EDGE_FRAC))
    top_ink = float(mask[:band_h, :].sum())
    bot_ink = float(mask[-band_h:, :].sum())
    if total > 0:
        if top_ink / total > _EDGE_BAND_RATIO:
            return {"is_serial": False, "confidence": 0.80, "reason": "bleed_top"}
        if bot_ink / total > _EDGE_BAND_RATIO:
            return {"is_serial": False, "confidence": 0.80, "reason": "bleed_bottom"}

    # Multi-band check: count distinct horizontal ink bands.
    ink_per_row = mask.sum(axis=1).astype(np.float32) / w
    band_thresh = max(0.02, ink_per_row.mean() * 0.5)
    bands = _band_count(ink_per_row, band_thresh)
    if bands > _BAND_COUNT_LIMIT:
        return {"is_serial": False, "confidence": 0.75, "reason": "multi_band"}

    # Confidence rises with how "central" the ink mass is — a typed line in the
    # middle of a row is the prototype. We compute it as 1 minus the absolute
    # vertical offset of the ink centroid from the row center, normalized.
    rows_idx = np.arange(h, dtype=np.float32)
    row_weights = ink_per_row + 1e-6
    centroid = float((rows_idx * row_weights).sum() / row_weights.sum())
    offset = abs(centroid - h / 2.0) / (h / 2.0)
    confidence = max(0.5, 1.0 - offset)

    return {"is_serial": True, "confidence": round(confidence, 3), "reason": "ok"}
