#!/usr/bin/env python3
"""
ConvNeXt-Tiny document classifier.
Classifies scanned pages as "discard", "{year}_serial", or "{year}_continuation".
Pixel-only inputs — no OCR, no text extraction.
"""

from __future__ import annotations
import json
import pickle
import re
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
TRAINING_DIR    = Path("MasterTraining")
TESTING_DIR     = Path("MasterTesting")
IMAGE_SIZE      = 224
BATCH_SIZE      = 16
EPOCHS          = 50
LR_HEAD         = 3e-4
LR_BACKBONE     = 5e-5
WEIGHT_DECAY    = 0.05
LABEL_SMOOTHING = 0.1
DROPOUT         = 0.3
MIXUP_ALPHA     = 0.2
VAL_SPLIT       = 0.20   # fraction of PDFs held out for validation
SEED            = 42
NUM_WORKERS     = 4
PATIENCE        = 10
MODEL_PATH      = Path("best_model.pt")
LABEL_MAP_PATH  = Path("label_map.pkl")

# Permissible confusion groups — errors between years in the same group do not count
# against permissive accuracy, for both _serial and _continuation page types.
PERMISSIBLE_GROUPS: list[set[str]] = [
    {"2005", "2007"},
    {"2008", "2012"},
    {"2020", "2022", "2023"},
]

# Continuation pages whose visual layout is identical to another year's continuation.
# The aliased year's pages are assigned the target year's continuation label instead.
CONTINUATION_ALIASES: dict[str, str] = {
    "2022": "2020",
}

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ── PDF → JSON matching ───────────────────────────────────────────────────────

def find_json(pdf_path: Path) -> Path | None:
    """
    Return the JSON for a PDF.
    x.pdf       → x.json  (direct match)
    x_1.pdf     → x.json  (strip trailing _N)
    Case-insensitive suffix check covers both .json and .JSON.
    """
    direct = pdf_path.with_suffix(".json")
    if direct.exists():
        return direct
    # Strip trailing _<digits>
    stem = re.sub(r"_\d+$", "", pdf_path.stem)
    for ext in (".json", ".JSON"):
        candidate = pdf_path.parent / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def collect_pairs(directory: Path) -> list[tuple[Path, Path]]:
    """Return (pdf, json) for every PDF in directory that has a matching JSON."""
    pairs = []
    missing = []
    for pdf in sorted(directory.glob("*.pdf")):
        j = find_json(pdf)
        if j:
            pairs.append((pdf, j))
        else:
            missing.append(pdf.name)
    if missing:
        print(f"WARNING: no JSON found for {len(missing)} PDF(s): {missing}")
    return pairs

# ── Label map ─────────────────────────────────────────────────────────────────

def build_label_map(
    json_paths: list[Path],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Scan training JSONs to determine, for each form_year:
      - which within-form page is the primary serial number page (lowest page number)
      - which within-form pages are continuation pages (all others with firearms)

    Labels:  "discard" | "{year}_serial" | "{year}_continuation"

    Returns:
        label_map:       class-name string → integer index
        year_serial_pos: year → within-form page number of the primary serial page
    """
    # year → set of within-form page positions that carry serial numbers
    year_fw_pages: dict[str, set[int]] = defaultdict(set)

    for jpath in sorted(set(json_paths)):   # deduplicate: _N PDFs share a JSON
        with open(jpath) as f:
            data = json.load(f)
        for page_entry in data["pages"]:
            firearms = page_entry.get("firearms", [])
            if not firearms:
                continue
            year = page_entry["form_year"]
            for fw in firearms:
                year_fw_pages[year].add(fw["page"])   # within-form page number

    # Primary serial page = lowest within-form page that has firearms
    year_serial_pos: dict[str, int] = {
        yr: min(pages) for yr, pages in year_fw_pages.items()
    }

    # Build sorted label map: discard=0, then serial/continuation alphabetically.
    # Aliased continuation years are omitted — their pages map to the target year's label.
    label_map: dict[str, int] = {"discard": 0}
    idx = 1
    for year in sorted(year_fw_pages):
        label_map[f"{year}_serial"] = idx
        idx += 1
        if len(year_fw_pages[year]) > 1 and year not in CONTINUATION_ALIASES:
            label_map[f"{year}_continuation"] = idx
            idx += 1

    return label_map, year_serial_pos


def page_to_label(
    page_entry: dict,
    label_map: dict[str, int],
    year_serial_pos: dict[str, int],
) -> int:
    """Assign a label integer to a single JSON page entry."""
    firearms = page_entry.get("firearms", [])
    if not firearms:
        return label_map["discard"]

    year = page_entry["form_year"]
    serial_pos = year_serial_pos.get(year)
    if serial_pos is None:
        return label_map["discard"]   # year unseen in training

    within_form_page = firearms[0]["page"]
    if within_form_page == serial_pos:
        return label_map.get(f"{year}_serial", label_map["discard"])
    else:
        cont_year = CONTINUATION_ALIASES.get(year, year)
        return label_map.get(f"{cont_year}_continuation", label_map["discard"])

# ── Sample building ───────────────────────────────────────────────────────────

def make_samples(
    pairs: list[tuple[Path, Path]],
    label_map: dict[str, int],
    year_serial_pos: dict[str, int],
) -> list[tuple[str, int, int]]:
    """Build a flat list of (pdf_path_str, page_idx_0based, label) from PDF/JSON pairs."""
    samples = []
    for pdf_path, json_path in pairs:
        with open(json_path) as f:
            data = json.load(f)
        for page_entry in data["pages"]:
            lbl      = page_to_label(page_entry, label_map, year_serial_pos)
            page_idx = page_entry["page"] - 1   # 0-based
            samples.append((str(pdf_path), page_idx, lbl))
    return samples


def split_by_page(
    pairs: list[tuple[Path, Path]],
    label_map: dict[str, int],
    year_serial_pos: dict[str, int],
    val_frac: float,
    seed: int,
) -> tuple[list, list]:
    """
    Split into train/val at the page level.
    Pages are evaluated independently, so there is no leakage risk.
    For each JSON the same page-index split is applied across all PDF
    copies that share it, so equivalent pages across copies land in the
    same partition.
    """
    by_json: dict[Path, list[Path]] = defaultdict(list)
    for pdf, json_path in sorted(pairs):
        by_json[json_path].append(pdf)

    train_s: list[tuple[str, int, int]] = []
    val_s:   list[tuple[str, int, int]] = []
    rng = random.Random(seed)

    for json_path in sorted(by_json):
        with open(json_path) as f:
            data = json.load(f)
        pages = data["pages"]
        n = len(pages)

        indices = list(range(n))
        rng.shuffle(indices)
        n_val  = max(0, int(n * val_frac))
        val_idx = set(indices[:n_val])

        pdfs = by_json[json_path]
        for pi, page_entry in enumerate(pages):
            lbl    = page_to_label(page_entry, label_map, year_serial_pos)
            target = val_s if pi in val_idx else train_s
            for pdf_path in pdfs:
                target.append((str(pdf_path), page_entry["page"] - 1, lbl))

    return train_s, val_s

# ── Dataset ───────────────────────────────────────────────────────────────────

_pdf_handles: dict[str, fitz.Document] = {}


def _open_pdf(path: str) -> fitz.Document:
    if path not in _pdf_handles:
        _pdf_handles[path] = fitz.open(path)
    return _pdf_handles[path]


class DocumentDataset(Dataset):
    """
    Yields (image_tensor, class_index) for each document page.
    Pages rendered at 72 DPI — enough for layout recognition, text unreadable.
    """
    def __init__(self, samples: list[tuple[str, int, int]], transform=None):
        self.samples  = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        pdf_path, page_idx, label = self.samples[idx]
        doc  = _open_pdf(pdf_path)
        page = doc[page_idx]
        pix  = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), colorspace=fitz.csRGB)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        if self.transform:
            img = self.transform(img)
        return img, label

# ── Transforms ────────────────────────────────────────────────────────────────

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
    transforms.RandomRotation(degrees=2, fill=255),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), value=1.0),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# ── Mixup ─────────────────────────────────────────────────────────────────────

def mixup(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
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
        loss   = criterion(logits, labels)
        total_loss    += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n       += imgs.size(0)
    return total_loss / total_n, total_correct / total_n

# ── Permissive accuracy ───────────────────────────────────────────────────────

def is_permissible(true_lbl: int, pred_lbl: int,
                   idx_to_label: dict[int, str]) -> bool:
    """
    Returns True if a confusion between true_lbl and pred_lbl is permissible.
    Applies to _serial or _continuation classes within the same PERMISSIBLE_GROUPS entry,
    provided both labels share the same page type suffix.
    """
    true_name = idx_to_label[true_lbl]
    pred_name = idx_to_label[pred_lbl]
    for suffix in ("_serial", "_continuation"):
        if true_name.endswith(suffix) and pred_name.endswith(suffix):
            true_year = true_name.removesuffix(suffix)
            pred_year = pred_name.removesuffix(suffix)
            return any(true_year in g and pred_year in g for g in PERMISSIBLE_GROUPS)
    return False


def is_safe_error(true_lbl: int, pred_lbl: int,
                  idx_to_label: dict[int, str]) -> bool:
    """
    Returns True when a serial/continuation page is misclassified as discard.
    Safer than the reverse (non-serial read as serial), so excluded from safe metrics.
    """
    return idx_to_label[true_lbl] != "discard" and idx_to_label[pred_lbl] == "discard"


@torch.no_grad()
def evaluate_test(model, loader, criterion, device,
                  idx_to_label: dict[int, str]) -> tuple[float, float, float, float, float]:
    """
    Returns (loss, standard_acc, permissive_acc, standard_safe_acc, perm_safe_acc).
    Safe metrics exclude errors where a serial/continuation page is classified as discard.
    """
    model.eval()
    total_loss = total_n = 0
    standard_correct = permissive_correct = standard_safe_correct = perm_safe_correct = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        for t, p in zip(labels.tolist(), preds.tolist()):
            total_n += 1
            if t == p:
                standard_correct      += 1
                permissive_correct    += 1
                standard_safe_correct += 1
                perm_safe_correct     += 1
            elif is_permissible(t, p, idx_to_label):
                permissive_correct += 1
                perm_safe_correct  += 1
            elif is_safe_error(t, p, idx_to_label):
                standard_safe_correct += 1
                perm_safe_correct     += 1

    return (total_loss / total_n,
            standard_correct      / total_n,
            permissive_correct    / total_n,
            standard_safe_correct / total_n,
            perm_safe_correct     / total_n)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    # ── Discover all training PDFs and their JSONs ─────────────────────────
    all_train_pairs = collect_pairs(TRAINING_DIR)
    print(f"Found {len(all_train_pairs)} training PDFs")

    # Build label map from unique training JSONs only
    train_jsons = list({j for _, j in all_train_pairs})
    label_map, year_serial_pos = build_label_map(train_jsons)
    idx_to_label: dict[int, str] = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)

    print(f"\nClasses ({num_classes}):")
    for name, idx in sorted(label_map.items(), key=lambda x: x[1]):
        marker = ""
        if name.endswith("_serial"):
            yr = name.removesuffix("_serial")
            marker = f"  [primary serial page: within-form pg {year_serial_pos[yr]}]"
        print(f"  {idx:2d}  {name}{marker}")

    with open(LABEL_MAP_PATH, "wb") as f:
        pickle.dump({
            "label_map":      label_map,
            "idx_to_label":   idx_to_label,
            "year_serial_pos": year_serial_pos,
        }, f)
    print(f"\nSaved label map → {LABEL_MAP_PATH}")

    # ── Train / val split at the page level ───────────────────────────────
    # Pages are independent, so a page-level split has no leakage risk.
    # The same page-index split is applied to all PDF copies of each JSON.
    train_samples, val_samples = split_by_page(
        all_train_pairs, label_map, year_serial_pos, VAL_SPLIT, SEED
    )

    n_serial = sum(1 for _, _, l in train_samples if l != label_map["discard"])
    print(f"Train pages: {len(train_samples)}  ({n_serial} serial/continuation, "
          f"{len(train_samples) - n_serial} discard)")
    print(f"Val pages:   {len(val_samples)}")

    # ── Weighted sampler for class balance ────────────────────────────────
    label_counts: dict[int, int] = defaultdict(int)
    for _, _, lbl in train_samples:
        label_counts[lbl] += 1
    sample_weights = [1.0 / label_counts[lbl] for _, _, lbl in train_samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    pin = device.type == "cuda"
    pw  = NUM_WORKERS > 0

    train_ds = DocumentDataset(train_samples, transform=train_transform)
    val_ds   = DocumentDataset(val_samples,   transform=eval_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=pw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin, persistent_workers=pw)

    # ── Model, loss, optimiser ────────────────────────────────────────────
    model     = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(),   "lr": LR_BACKBONE},
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc   = 0.0
    patience_count = 0
    print()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        saved = ""
        if va_acc > best_val_acc:
            best_val_acc   = va_acc
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_acc":     va_acc,
                "num_classes": num_classes,
            }, MODEL_PATH)
            saved = "  *saved*"
        else:
            patience_count += 1

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train {tr_loss:.4f}/{tr_acc:.3f}  "
              f"val {va_loss:.4f}/{va_acc:.3f}{saved}")

        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # ── Test evaluation ───────────────────────────────────────────────────
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print("Evaluating on test set…")

    all_test_pairs = collect_pairs(TESTING_DIR)
    print(f"Found {len(all_test_pairs)} test PDFs")
    test_samples = make_samples(all_test_pairs, label_map, year_serial_pos)
    print(f"Test pages: {len(test_samples)}")

    test_ds     = DocumentDataset(test_samples, transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    t_loss, t_acc, t_perm, t_safe, t_perm_safe = evaluate_test(
        model, test_loader, criterion, device, idx_to_label
    )
    print(f"\nTest loss:                   {t_loss:.4f}")
    print(f"Standard accuracy:           {t_acc:.4f}  ({t_acc:.1%})")
    print(f"Permissive accuracy:         {t_perm:.4f}  ({t_perm:.1%})")
    print(f"Standard safe accuracy:      {t_safe:.4f}  ({t_safe:.1%})")
    print(f"Safe + permissive accuracy:  {t_perm_safe:.4f}  ({t_perm_safe:.1%})")
    print(f"\nPermissible confusion groups (_serial and _continuation):")
    for g in PERMISSIBLE_GROUPS:
        print(f"  {sorted(g)}")


if __name__ == "__main__":
    main()
