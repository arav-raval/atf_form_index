"""Stage 1 — Preprocess.

Currently does **deskew only**: detects small page rotations (typical of
scanned PDFs, <2°) and rotates the page back to 0° before localization.

We measured ~0.25° to 1.5° skew on most MasterTesting + v2 PDFs. Even 1°
of rotation produces ~14 pixels of horizontal drift at the bottom of an
800-px page, which is enough to clip the leading character off a serial
when the annotated ``serial_block`` is positioned in PDF-point space.

ConvNeXt classifier and TrOCR consume RGB; we keep RGB throughout (no
binarization). Sauvola was useful for the legacy SSIM classifier but
hurts modern vision-transformer inputs.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# Skew search is bounded — most document scans are <3° but extremes occur.
# Coarse step finds the basin; fine step refines within it.
_DESKEW_COARSE_RANGE = 5.0   # degrees
_DESKEW_COARSE_STEP = 0.5
_DESKEW_FINE_RANGE = 0.5     # degrees around the coarse maximum
_DESKEW_FINE_STEP = 0.1
# Rotations smaller than this are treated as no-op (avoids degrading
# already-aligned pages with sub-pixel rotation artifacts).
_DESKEW_MIN_DEG = 0.15


def detect_skew_angle(img: Image.Image) -> float:
    """Return the rotation angle (in degrees, positive = counterclockwise to
    correct) that maximizes horizontal-projection variance.

    Intuition: a well-aligned text page has dense rows of ink with bright
    inter-line gutters. Projecting ink intensity to the y-axis gives a
    high-variance signal. A skewed page smears each row across multiple y
    positions, flattening the projection. Rotating to maximize variance
    finds the alignment angle.

    Uses a downsampled grayscale view to keep cost per page <50 ms.
    """
    # Downsample to ~600 px wide for speed; still preserves the row
    # structure we need for projection.
    w, h = img.size
    if max(w, h) > 800:
        scale = 600 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    arr = 255 - np.asarray(img.convert("L"), dtype=np.float32)  # ink-positive

    def _proj_var(angle: float) -> float:
        if abs(angle) < 1e-3:
            rotated = arr
        else:
            from scipy.ndimage import rotate as nd_rotate
            rotated = nd_rotate(
                arr, angle, reshape=False, order=1, mode="constant", cval=0.0,
            )
        return float(rotated.mean(axis=1).var())

    # Coarse pass
    coarse = np.arange(-_DESKEW_COARSE_RANGE,
                        _DESKEW_COARSE_RANGE + 1e-6,
                        _DESKEW_COARSE_STEP)
    coarse_scores = [_proj_var(a) for a in coarse]
    best_coarse = float(coarse[int(np.argmax(coarse_scores))])

    # Fine pass around the coarse winner
    lo = best_coarse - _DESKEW_FINE_RANGE
    hi = best_coarse + _DESKEW_FINE_RANGE
    fine = np.arange(lo, hi + 1e-6, _DESKEW_FINE_STEP)
    fine_scores = [_proj_var(a) for a in fine]
    return float(fine[int(np.argmax(fine_scores))])


def deskew(img: Image.Image, angle: float | None = None) -> Image.Image:
    """Rotate ``img`` by the negative of its detected skew angle (so text
    rows become horizontal). Pass an explicit ``angle`` to skip detection.

    Returns the input unchanged if the detected angle is below
    :data:`_DESKEW_MIN_DEG` — sub-pixel rotations cost more than they
    fix because of bilinear-resample blur.
    """
    if angle is None:
        angle = detect_skew_angle(img)
    if abs(angle) < _DESKEW_MIN_DEG:
        return img
    # ``angle`` is the rotation (in scipy/PIL convention: + = counterclockwise)
    # that maximizes horizontal-projection variance, i.e. the rotation that
    # *aligns* the page. Apply it directly. White fill on the corner triangles.
    return img.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))


def preprocess(page_image: Image.Image, *, do_deskew: bool = True) -> Image.Image:
    """Stage-1 preprocess. Currently deskew-only.

    Returns an RGB image of the same dimensions, rotated to 0° if
    ``do_deskew`` is True. No binarization — downstream consumers
    (ConvNeXt classifier, ViT verifier, TrOCR) all want RGB.
    """
    if not do_deskew:
        return page_image
    return deskew(page_image)
