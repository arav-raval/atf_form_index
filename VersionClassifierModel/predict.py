#!/usr/bin/env python3
"""
Classify a document page as either "discard" (not a serial-number page) or the
specific (form_year, page_within_form) identity of a serial-number page.

Pixel-only — no text is extracted or stored.

Usage:
    python predict.py page.png
    python predict.py document.pdf --page 3
    python predict.py page.jpg --top 5
"""

from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import convnext_tiny
from PIL import Image

IMAGE_SIZE = 224
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def _load_model(model_path: Path, num_classes: int) -> nn.Module:
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.0),  # disabled at inference
        nn.Linear(in_features, num_classes),
    )
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_image(
    img: Image.Image,
    model_path: Path | str = "best_model.pt",
    label_map_path: Path | str = "label_map.pkl",
    top_k: int = 3,
) -> list[dict]:
    """
    Classify a single page image.

    Args:
        img: PIL Image of the document page. Pixel values only — no text
             is extracted or retained in memory at any point.
        model_path: path to the trained checkpoint (best_model.pt).
        label_map_path: path to the label map pickle (label_map.pkl).
        top_k: number of top predictions to return.

    Returns:
        List of dicts sorted by confidence (highest first). Each dict has:
            "label"       – raw class string, e.g. "2016_serial", "discard"
            "is_serial"   – True for *_serial and *_continuation classes
            "form_year"   – e.g. "2016"  (None for discard)
            "page_type"   – "serial", "continuation", or None for discard
            "confidence"  – float 0-1
    """
    with open(label_map_path, "rb") as f:
        meta = pickle.load(f)
    idx_to_label: dict[int, str] = meta["idx_to_label"]
    num_classes = len(idx_to_label)

    model = _load_model(Path(model_path), num_classes)

    tensor = _transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]

    top_probs, top_idxs = probs.topk(min(top_k, num_classes))
    results = []
    for p, i in zip(top_probs, top_idxs):
        label = idx_to_label[i.item()]
        if label == "discard":
            results.append({
                "label":     "discard",
                "is_serial": False,
                "form_year": None,
                "page_type": None,
                "confidence": round(p.item(), 6),
            })
        else:
            # label is e.g. "2016_serial" or "2020_continuation"
            year, page_type = label.rsplit("_", 1)
            results.append({
                "label":     label,
                "is_serial": True,
                "form_year": year,
                "page_type": page_type,   # "serial" or "continuation"
                "confidence": round(p.item(), 6),
            })
    return results


def render_pdf_page(pdf_path: Path | str, page_1based: int) -> Image.Image:
    """
    Render one page of a PDF to a PIL Image.
    Returns pixel data only — page text is never decoded or stored.
    """
    import fitz
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_1based - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), colorspace=fitz.csRGB)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify a document page: serial-number page or discard."
    )
    parser.add_argument("input", help="Image file (.png, .jpg) or PDF")
    parser.add_argument(
        "--page", type=int, default=1, metavar="N",
        help="Page number (1-based) when input is a PDF  [default: 1]",
    )
    parser.add_argument("--model",  default="best_model.pt", help="Model checkpoint")
    parser.add_argument("--labels", default="label_map.pkl", help="Label map pickle")
    parser.add_argument("--top",    type=int, default=3,     help="Top-k predictions")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.suffix.lower() == ".pdf":
        img = render_pdf_page(path, args.page)
        src = f"{path.name}  (page {args.page})"
    else:
        img = Image.open(path)
        src = path.name

    results = predict_image(img, args.model, args.labels, top_k=args.top)

    print(f"\nPredictions for: {src}")
    for rank, r in enumerate(results, 1):
        if r["is_serial"]:
            desc = f"Form year {r['form_year']} — {r['page_type']} page"
        else:
            desc = "Discard (not a serial-number page)"
        print(f"  {rank}.  {desc}  —  {r['confidence']:.1%}")
