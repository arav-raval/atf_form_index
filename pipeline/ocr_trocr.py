"""TrOCR backend — Vision-Transformer encoder + Transformer decoder OCR.

Two pre-trained checkpoints from Microsoft:
  - ``microsoft/trocr-base-printed``       — typed/printed text
  - ``microsoft/trocr-base-handwritten``   — handwritten text

The handwritten model also reads printed text reasonably well, so for a single
backend choice it's the safer default. We expose both and let the caller pick.

The model is loaded lazily on first call and cached. First call downloads ~330MB
to the HuggingFace cache; subsequent calls are local.
"""
from __future__ import annotations

import logging
from typing import Any

from PIL import Image

log = logging.getLogger(__name__)

_MODELS: dict[str, dict[str, Any]] = {
    "printed": {"hf_id": "microsoft/trocr-base-printed", "loaded": None},
    "handwritten": {"hf_id": "microsoft/trocr-base-handwritten", "loaded": None},
}

_DEFAULT_VARIANT = "printed"


def _load(variant: str) -> tuple[Any, Any, Any] | None:
    if variant not in _MODELS:
        raise ValueError(f"Unknown TrOCR variant {variant!r}")
    cached = _MODELS[variant]["loaded"]
    if cached is not None:
        return cached
    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError:
        return None

    hf_id = _MODELS[variant]["hf_id"]
    log.info("Loading TrOCR %s (first call may download ~330MB)…", hf_id)
    processor = TrOCRProcessor.from_pretrained(hf_id)
    model = VisionEncoderDecoderModel.from_pretrained(hf_id)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    bundle = (processor, model, device)
    _MODELS[variant]["loaded"] = bundle
    return bundle


def is_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def recognize_trocr(
    crop: Image.Image,
    variant: str = _DEFAULT_VARIANT,
) -> dict:
    """Run TrOCR on a single-line crop. Returns ``{text, method}``.

    ``method`` is ``"trocr_<variant>"`` on success, or ``"trocr_unavailable"``
    / ``"trocr_error"`` on failure. The caller is responsible for normalization
    and pattern filtering — this function only does inference.
    """
    bundle = _load(variant)
    if bundle is None:
        return {"text": "", "method": "trocr_unavailable"}
    try:
        import torch

        processor, model, device = bundle
        rgb = crop.convert("RGB")
        pixel_values = processor(images=rgb, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=24)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"text": text.strip(), "method": f"trocr_{variant}"}
    except Exception as e:
        log.warning("TrOCR failed: %s", e)
        return {"text": "", "method": "trocr_error"}
