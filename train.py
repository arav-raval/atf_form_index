#!/usr/bin/env python3
"""
ConvNeXt-Tiny document classifier.
Outputs either "discard" (not a serial number page) or the specific
(form_year, page_within_form) identity of a serial-number page.
Pixel-only inputs — no OCR, no text extraction.
"""

from __future__ import annotations
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
TRAINING_DIR    = Path("Training")
TESTING_DIR     = Path("Testing")
IMAGE_SIZE      = 224
BATCH_SIZE      = 16
EPOCHS          = 50
LR_HEAD         = 3e-4   # higher LR for the new classifier head
LR_BACKBONE     = 5e-5   # lower LR to preserve pretrained visual features
WEIGHT_DECAY    = 0.05
LABEL_SMOOTHING = 0.1
DROPOUT         = 0.3
MIXUP_ALPHA     = 0.2
VAL_SPLIT       = 0.15   # fraction of forms held out for validation
SEED            = 42
NUM_WORKERS     = 4
PATIENCE        = 10
MODEL_PATH      = Path("best_model.pt")
LABEL_MAP_PATH  = Path("label_map.pkl")

DISCARD_KEY = "discard"   # label key for non-serial pages

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ── Label map ─────────────────────────────────────────────────────────────────

def build_label_map(
    json_paths: list[Path],
) -> tuple[dict, set[tuple[str, int]]]:
    """
    Scan all JSON files to find which (year, page_within_form) positions
    always carry firearms (serial number fields).

    Returns:
        label_map: {"discard": 0, (year, pos): 1, ...}
        serial_pages: set of (year, pos) keys that are serial number pages
    """
    has_firearms:    set[tuple[str, int]] = set()
    has_no_firearms: set[tuple[str, int]] = set()

    for jpath in json_paths:
        with open(jpath) as f:
            data = json.load(f)
        forms: dict[str, list] = defaultdict(list)
        for p in data["pages"]:
            forms[p["transaction_number"]].append(p)
        for form_pages in forms.values():
            sorted_pgs = sorted(form_pages, key=lambda x: x["page"])
            year = sorted_pgs[0]["form_year"]
            for pos, pg in enumerate(sorted_pgs):
                key = (year, pos + 1)
                if pg["firearms"]:
                    has_firearms.add(key)
                else:
                    has_no_firearms.add(key)

    # A page type is a "serial number page" only if it ALWAYS has firearms
    # across every form instance (no page was ever blank for that type).
    serial_pages = has_firearms - has_no_firearms

    label_map: dict = {DISCARD_KEY: 0}
    for i, lbl in enumerate(sorted(serial_pages), start=1):
        label_map[lbl] = i

    return label_map, serial_pages

# ── Dataset ───────────────────────────────────────────────────────────────────

_pdf_handles: dict[str, fitz.Document] = {}


def _open_pdf(path: str) -> fitz.Document:
    if path not in _pdf_handles:
        _pdf_handles[path] = fitz.open(path)
    return _pdf_handles[path]


class DocumentDataset(Dataset):
    """
    Yields (image_tensor, class_index) for each document page.
    Pages are rendered at 72 DPI — sufficient to distinguish layout structure
    while keeping text unreadable, meeting the pixel-only requirement.
    """
    def __init__(self, samples: list[tuple[str, int, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pdf_path, page_idx, label = self.samples[idx]
        doc = _open_pdf(pdf_path)
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if self.transform:
            img = self.transform(img)
        return img, label


def _page_label(year: str, pos_1based: int,
                label_map: dict,
                serial_pages: set[tuple[str, int]]) -> int:
    key = (year, pos_1based)
    return label_map[key] if key in serial_pages else label_map[DISCARD_KEY]


def _make_samples(pdf_path: Path, json_path: Path,
                  label_map: dict,
                  serial_pages: set[tuple[str, int]]) -> list[tuple[str, int, int]]:
    with open(json_path) as f:
        data = json.load(f)
    forms: dict[str, list] = defaultdict(list)
    for p in data["pages"]:
        forms[p["transaction_number"]].append(p)
    samples = []
    for form_pages in forms.values():
        sorted_pgs = sorted(form_pages, key=lambda x: x["page"])
        year = sorted_pgs[0]["form_year"]
        for pos, pg in enumerate(sorted_pgs):
            lbl = _page_label(year, pos + 1, label_map, serial_pages)
            samples.append((str(pdf_path), pg["page"] - 1, lbl))
    return samples


def _form_split(pdf_path: Path, json_path: Path,
                label_map: dict, serial_pages: set[tuple[str, int]],
                val_frac: float, seed: int) -> tuple[list, list]:
    """Split by form so all pages of one transaction stay in the same split."""
    with open(json_path) as f:
        data = json.load(f)
    forms: dict[str, list] = defaultdict(list)
    for p in data["pages"]:
        forms[p["transaction_number"]].append(p)

    txns = sorted(forms.keys())
    rng = random.Random(seed)
    rng.shuffle(txns)
    val_txns = set(txns[:max(1, int(len(txns) * val_frac))])

    train_s, val_s = [], []
    for txn, form_pages in forms.items():
        sorted_pgs = sorted(form_pages, key=lambda x: x["page"])
        year = sorted_pgs[0]["form_year"]
        for pos, pg in enumerate(sorted_pgs):
            lbl = _page_label(year, pos + 1, label_map, serial_pages)
            entry = (str(pdf_path), pg["page"] - 1, lbl)
            (val_s if txn in val_txns else train_s).append(entry)
    return train_s, val_s

# ── Transforms ────────────────────────────────────────────────────────────────

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
    transforms.RandomRotation(degrees=2, fill=255),  # white fill = blank paper background
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), value=1.0),  # simulate stamps/redactions
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# ── Mixup ─────────────────────────────────────────────────────────────────────

def mixup(x: torch.Tensor, y: torch.Tensor,
          alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    # ConvNeXt classifier stack: LayerNorm2d → Flatten → Linear
    # Insert Dropout before the final Linear for regularisation.
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, num_classes),
    )
    return model

# ── Training / evaluation ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_correct = total_n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        imgs, ya, yb, lam = mixup(imgs, labels, MIXUP_ALPHA)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
        loss.backward()
        optimizer.step()
        preds = logits.argmax(1)
        total_loss    += loss.item() * imgs.size(0)
        total_correct += (lam * (preds == ya).float() + (1 - lam) * (preds == yb).float()).sum().item()
        total_n       += imgs.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total_n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss    += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n       += imgs.size(0)
    return total_loss / total_n, total_correct / total_n

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # Derive the label map from training data only. This ensures the serial-page
    # set reflects the authoritative form structure. Test pages outside these
    # known classes (e.g. data-generation artefacts) will be labelled "discard".
    train_jsons = [
        TRAINING_DIR / "v2dataset_errors_2 copy.json",
        TRAINING_DIR / "v2dataset_no_errors_2 copy.json",
    ]
    label_map, serial_pages = build_label_map(train_jsons)
    idx_to_label: dict[int, object] = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    print(f"Classes: {num_classes}  "
          f"(1 discard + {num_classes - 1} serial-number page types)")
    print("Serial-number page types:")
    for key in sorted(serial_pages):
        print(f"  year={key[0]}  page_within_form={key[1]}  → class {label_map[key]}")

    with open(LABEL_MAP_PATH, "wb") as f:
        pickle.dump({
            "label_map": label_map,
            "idx_to_label": idx_to_label,
            "serial_pages": serial_pages,
        }, f)
    print(f"Saved label map → {LABEL_MAP_PATH}")

    # Build train/val from the Training folder only
    train_samples: list[tuple[str, int, int]] = []
    val_samples:   list[tuple[str, int, int]] = []
    for pdf_name, json_name in [
        ("v2dataset_errors_2 copy.pdf",    "v2dataset_errors_2 copy.json"),
        ("v2dataset_no_errors_2 copy.pdf", "v2dataset_no_errors_2 copy.json"),
    ]:
        tr, va = _form_split(
            TRAINING_DIR / pdf_name,
            TRAINING_DIR / json_name,
            label_map, serial_pages, VAL_SPLIT, SEED,
        )
        train_samples.extend(tr)
        val_samples.extend(va)

    n_serial_train = sum(1 for _, _, l in train_samples if l != label_map[DISCARD_KEY])
    print(f"\nTrain pages: {len(train_samples)}  "
          f"({n_serial_train} serial, {len(train_samples) - n_serial_train} discard)")
    print(f"Val pages:   {len(val_samples)}")

    # Weighted sampler so every class (including rare serial page types) is
    # seen equally often each epoch despite the large discard imbalance.
    label_counts: dict[int, int] = defaultdict(int)
    for _, _, lbl in train_samples:
        label_counts[lbl] += 1
    sample_weights = [1.0 / label_counts[lbl] for _, _, lbl in train_samples]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    pin = device.type == "cuda"
    pw  = NUM_WORKERS > 0

    train_ds = DocumentDataset(train_samples, transform=train_transform)
    val_ds   = DocumentDataset(val_samples,   transform=eval_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=pw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=pw)

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(),   "lr": LR_BACKBONE},
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_val_acc = 0.0
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        saved = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_count = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": va_acc,
                "num_classes": num_classes,
            }, MODEL_PATH)
            saved = "  *saved*"
        else:
            patience_count += 1

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train {tr_loss:.4f}/{tr_acc:.3f}  "
              f"val {va_loss:.4f}/{va_acc:.3f}{saved}")

        if patience_count >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Evaluate best checkpoint on held-out Testing folder
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print("Evaluating best model on test set…")

    test_samples: list[tuple[str, int, int]] = []
    for pdf_name, json_name in [
        ("v2dataset_errors_1 copy.pdf",    "v2dataset_errors_1 copy.json"),
        ("v2dataset_no_errors_1 copy.pdf", "v2dataset_no_errors_1 copy.json"),
    ]:
        test_samples.extend(
            _make_samples(
                TESTING_DIR / pdf_name,
                TESTING_DIR / json_name,
                label_map, serial_pages,
            )
        )

    test_ds = DocumentDataset(test_samples, transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test  loss={test_loss:.4f}  acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
