"""TrOCR backend — Vision-Transformer encoder + Transformer decoder OCR.

Two pre-trained checkpoints from Microsoft:
  - ``microsoft/trocr-base-printed``       — typed/printed text
  - ``microsoft/trocr-base-handwritten``   — handwritten text

The ``printed`` variant transparently prefers a locally fine-tuned checkpoint
at ``trocr_finetuned_10k/`` when it exists (a HuggingFace ``save_pretrained``
directory). This lets ``recognize.recognize()`` benefit from fine-tuning
without changing any caller. The base printed model is used otherwise.

The model is loaded lazily on first call and cached. First call may download
~330MB if the base model isn't in the HF cache; subsequent calls are local.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image

log = logging.getLogger(__name__)

_FINETUNED_DIR = Path(__file__).resolve().parent.parent / "trocr_finetuned_10k"

_MODELS: dict[str, dict[str, Any]] = {
    "printed": {"hf_id": "microsoft/trocr-base-printed", "loaded": None},
    "printed_base": {"hf_id": "microsoft/trocr-base-printed", "loaded": None},
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

    # `printed` resolves to the fine-tuned local dir when it exists. Anyone
    # who explicitly wants the unmodified base model can ask for `printed_base`.
    if variant == "printed" and _FINETUNED_DIR.is_dir() and (_FINETUNED_DIR / "config.json").is_file():
        source: str | Path = _FINETUNED_DIR
        log.info("Loading TrOCR printed (fine-tuned) from %s", _FINETUNED_DIR)
    else:
        source = _MODELS[variant]["hf_id"]
        log.info("Loading TrOCR %s (%s)", variant, source)
    processor = TrOCRProcessor.from_pretrained(source)
    model = VisionEncoderDecoderModel.from_pretrained(source)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
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


def recognize_trocr_batch(
    crops: list[Image.Image],
    variant: str = _DEFAULT_VARIANT,
    *,
    num_hypotheses: int = 1,
) -> list[dict]:
    """Run TrOCR on a batch of crops in a single forward pass.

    Returns a list of dicts (one per crop) with the same shape as
    :func:`recognize_trocr`. Substantially faster than calling the single-crop
    function in a loop because the GPU/MPS parallelism is wasted on N=1 calls.

    Empty input returns an empty list.
    """
    if not crops:
        return []
    bundle = _load(variant)
    if bundle is None:
        return [{"text": "", "method": "trocr_unavailable", "hypotheses": []} for _ in crops]
    try:
        import torch

        processor, model, device = bundle
        rgbs = [c.convert("RGB") for c in crops]
        # Processor handles list[Image] natively → (N, 3, 384, 384) tensor
        pixel_values = processor(images=rgbs, return_tensors="pt").pixel_values.to(device)
        N = pixel_values.shape[0]

        if num_hypotheses <= 1:
            with torch.no_grad():
                gen = model.generate(pixel_values, max_new_tokens=24)
            texts = processor.batch_decode(gen, skip_special_tokens=True)
            method = f"trocr_{variant}"
            return [
                {"text": t.strip(), "method": method,
                 "hypotheses": [{"text": t.strip(), "score": 0.0}]}
                for t in texts
            ]

        # Beam search returning multiple sequences per input.
        n = max(2, num_hypotheses)
        with torch.no_grad():
            out = model.generate(
                pixel_values,
                max_new_tokens=24,
                num_beams=n,
                num_return_sequences=n,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        seqs = out.sequences  # (N*n, T)
        scores = (
            out.sequences_scores.tolist()
            if getattr(out, "sequences_scores", None) is not None
            else [0.0] * seqs.shape[0]
        )
        decoded = processor.batch_decode(seqs, skip_special_tokens=True)
        method = f"trocr_{variant}"

        results: list[dict] = []
        for i in range(N):
            sub_decoded = decoded[i * n : (i + 1) * n]
            sub_scores = scores[i * n : (i + 1) * n]
            best_for: dict[str, float] = {}
            for s, t in zip(sub_scores, sub_decoded):
                t = t.strip()
                if not t:
                    continue
                if t not in best_for or s > best_for[t]:
                    best_for[t] = s
            ordered = sorted(best_for.items(), key=lambda kv: -kv[1])[:num_hypotheses]
            hyps = [{"text": t, "score": float(s)} for t, s in ordered]
            results.append({
                "text": hyps[0]["text"] if hyps else "",
                "method": method,
                "hypotheses": hyps,
            })
        return results
    except Exception as e:
        log.warning("TrOCR batch failed: %s", e)
        return [{"text": "", "method": "trocr_error", "hypotheses": []} for _ in crops]


def recognize_trocr(
    crop: Image.Image,
    variant: str = _DEFAULT_VARIANT,
    *,
    num_hypotheses: int = 1,
) -> dict:
    """Run TrOCR on a single-line crop. Returns
    ``{text, method, hypotheses}``.

    Single-call wrapper around :func:`recognize_trocr_batch`. Prefer the batch
    API when OCRing many rows — N=1 calls waste the GPU/MPS parallelism.
    """
    out = recognize_trocr_batch([crop], variant=variant, num_hypotheses=num_hypotheses)
    return out[0] if out else {"text": "", "method": "trocr_error", "hypotheses": []}
