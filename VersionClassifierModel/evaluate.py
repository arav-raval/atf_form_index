#!/usr/bin/env python3
"""
Evaluate a trained checkpoint against the held-out test set.
Prints per-error details, standard and permissive accuracy, and saves a
confusion matrix image.

Usage:
    python evaluate.py
    python evaluate.py --model best_model.pt --labels label_map.pkl --out confusion.png
"""

from __future__ import annotations
import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny
from PIL import Image
import torch.nn as nn

from train import DocumentDataset

TESTING_DIR  = Path("MasterTesting")
BATCH_SIZE   = 32
NUM_WORKERS  = 4

PERMISSIBLE_GROUPS: list[set[str]] = [
    {"2005", "2007"},
    {"2008", "2012"},
    {"2020", "2022", "2023"},
]

CONTINUATION_ALIASES: dict[str, str] = {
    "2022": "2020",
}

IMAGE_SIZE = 224
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_json(pdf_path: Path) -> Path | None:
    direct = pdf_path.with_suffix(".json")
    if direct.exists():
        return direct
    stem = re.sub(r"_\d+$", "", pdf_path.stem)
    for ext in (".json", ".JSON"):
        candidate = pdf_path.parent / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def collect_pairs(directory: Path) -> list[tuple[Path, Path]]:
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


def page_to_label(
    page_entry: dict,
    label_map: dict[str, int],
    year_serial_pos: dict[str, int],
) -> int:
    firearms = page_entry.get("firearms", [])
    if not firearms:
        return label_map["discard"]
    year = page_entry["form_year"]
    serial_pos = year_serial_pos.get(year)
    if serial_pos is None:
        return label_map["discard"]
    within_form_page = firearms[0]["page"]
    if within_form_page == serial_pos:
        return label_map.get(f"{year}_serial", label_map["discard"])
    else:
        cont_year = CONTINUATION_ALIASES.get(year, year)
        return label_map.get(f"{cont_year}_continuation", label_map["discard"])


def is_safe_error(true_lbl: int, pred_lbl: int,
                  idx_to_label: dict[int, str]) -> bool:
    """True when a serial/continuation page is misclassified as discard."""
    return idx_to_label[true_lbl] != "discard" and idx_to_label[pred_lbl] == "discard"


def is_permissible(true_lbl: int, pred_lbl: int,
                   idx_to_label: dict[int, str]) -> bool:
    true_name = idx_to_label[true_lbl]
    pred_name = idx_to_label[pred_lbl]
    for suffix in ("_serial", "_continuation"):
        if true_name.endswith(suffix) and pred_name.endswith(suffix):
            true_year = true_name.removesuffix(suffix)
            pred_year = pred_name.removesuffix(suffix)
            return any(true_year in g and pred_year in g for g in PERMISSIBLE_GROUPS)
    return False


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(model_path: Path, num_classes: int,
                device: torch.device) -> nn.Module:
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.0),
        nn.Linear(in_features, num_classes),
    )
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.to(device)
    saved_epoch = ckpt.get("epoch", "?")
    saved_val   = ckpt.get("val_acc", float("nan"))
    print(f"Loaded checkpoint: epoch {saved_epoch},  val_acc={saved_val:.4f}")
    return model

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint on MasterTesting set (or CR* training files)."
    )
    parser.add_argument("--model",  default="best_model.pt")
    parser.add_argument("--labels", default="label_map.pkl")
    parser.add_argument("--out",    default="confusion.png", help="Confusion matrix output path")
    parser.add_argument("--cr",     action="store_true",
                        help="Run on CR* PDFs in MasterTraining instead of MasterTesting")
    args = parser.parse_args()

    with open(args.labels, "rb") as f:
        meta = pickle.load(f)
    label_map:       dict[str, int] = meta["label_map"]
    idx_to_label:    dict[int, str] = meta["idx_to_label"]
    year_serial_pos: dict[str, int] = meta["year_serial_pos"]
    num_classes = len(label_map)

    device = _get_device()
    print(f"Device: {device}")
    model = _load_model(Path(args.model), num_classes, device)

    # ── Build test sample list ─────────────────────────────────────────────
    if args.cr:
        source_dir = Path("MasterTraining")
        all_pairs  = collect_pairs(source_dir)
        pairs      = [(p, j) for p, j in all_pairs if p.name.upper().startswith("CR")]
        print(f"\nCR mode: found {len(pairs)} CR* PDFs in {source_dir}")
    else:
        pairs = collect_pairs(TESTING_DIR)
        print(f"\nFound {len(pairs)} test PDFs in {TESTING_DIR}")

    test_samples: list[tuple[str, int, int]] = []
    for pdf_path, json_path in pairs:
        with open(json_path) as f:
            data = json.load(f)
        for page_entry in data["pages"]:
            lbl = page_to_label(page_entry, label_map, year_serial_pos)
            test_samples.append((str(pdf_path), page_entry["page"] - 1, lbl))

    print(f"Test pages: {len(test_samples)}")

    # ── Run inference ──────────────────────────────────────────────────────
    pin = device.type == "cuda"
    test_ds     = DocumentDataset(test_samples, transform=_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin,
                             persistent_workers=NUM_WORKERS > 0)

    all_preds: list[int]   = []
    all_confs: list[float] = []

    print(f"\nRunning inference on {len(test_samples)} pages…")
    done = 0
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs  = imgs.to(device)
            probs = F.softmax(model(imgs), dim=1)
            confs, preds = probs.max(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_confs.extend(confs.cpu().tolist())
            done += imgs.size(0)
            if done % 500 == 0 or done == len(test_samples):
                print(f"  {done}/{len(test_samples)}")

    y_true = [lbl for _, _, lbl in test_samples]
    y_pred = all_preds
    y_conf = all_confs
    errors: list[dict] = []
    for (pdf_path, page_idx, true_lbl), pred_lbl, confidence in zip(
            test_samples, y_pred, y_conf):
        if pred_lbl != true_lbl:
            errors.append({
                "pdf":         Path(pdf_path).name,
                "pdf_page":    page_idx + 1,
                "true":        idx_to_label[true_lbl],
                "predicted":   idx_to_label[pred_lbl],
                "permissible": is_permissible(true_lbl, pred_lbl, idx_to_label),
                "safe":        is_safe_error(true_lbl, pred_lbl, idx_to_label),
                "confidence":  confidence,
            })

    # ── Summary ────────────────────────────────────────────────────────────
    total         = len(y_true)
    correct       = sum(t == p for t, p in zip(y_true, y_pred))
    n_perm        = sum(1 for e in errors if e["permissible"])
    n_safe        = sum(1 for e in errors if e["safe"])
    n_perm_safe   = sum(1 for e in errors if e["permissible"] or e["safe"])
    perm_ok       = correct + n_perm
    safe_ok       = correct + n_safe
    perm_safe_ok  = correct + n_perm_safe

    print(f"\n{'─'*60}")
    print(f"Standard accuracy:           {correct}/{total}  ({correct/total:.1%})")
    print(f"Permissive accuracy:         {perm_ok}/{total}  ({perm_ok/total:.1%})")
    print(f"Standard safe accuracy:      {safe_ok}/{total}  ({safe_ok/total:.1%})")
    print(f"Safe + permissive accuracy:  {perm_safe_ok}/{total}  ({perm_safe_ok/total:.1%})")
    print(f"Errors:                      {len(errors)}  "
          f"({n_perm} permissible, {n_safe} safe, "
          f"{len(errors) - n_perm_safe} hard)")

    print(f"\nPermissible confusion groups (_serial and _continuation):")
    for g in PERMISSIBLE_GROUPS:
        print(f"  {sorted(g)}")

    # ── Confidence threshold analysis (≥50%) ──────────────────────────────
    THRESHOLD = 0.50
    kept   = [(t, p, c) for t, p, c in zip(y_true, y_pred, y_conf) if c >= THRESHOLD]
    skipped = [(t, p, c) for t, p, c in zip(y_true, y_pred, y_conf) if c <  THRESHOLD]

    k_total   = len(kept)
    k_correct = sum(t == p for t, p, c in kept)
    k_errors  = [e for e in errors if e["confidence"] >= THRESHOLD]
    k_perm    = sum(1 for e in k_errors if e["permissible"])
    k_safe    = sum(1 for e in k_errors if e["safe"])
    k_ps      = sum(1 for e in k_errors if e["permissible"] or e["safe"])

    skipped_correct = sum(1 for t, p, c in skipped if t == p)
    skipped_errors  = len(skipped) - skipped_correct

    print(f"\n{'─'*60}")
    print(f"Confidence threshold ≥{THRESHOLD:.0%}  "
          f"({k_total} kept, {len(skipped)} skipped)")
    print(f"  Standard accuracy:           "
          f"{k_correct}/{k_total}  ({k_correct/k_total:.1%})  "
          f"[{len(k_errors) - k_ps} hard errors]")
    print(f"  Permissive accuracy:         "
          f"{k_correct+k_perm}/{k_total}  ({(k_correct+k_perm)/k_total:.1%})")
    print(f"  Standard safe accuracy:      "
          f"{k_correct+k_safe}/{k_total}  ({(k_correct+k_safe)/k_total:.1%})")
    print(f"  Safe + permissive accuracy:  "
          f"{k_correct+k_ps}/{k_total}  ({(k_correct+k_ps)/k_total:.1%})")
    print(f"\n  Skipped by threshold: {len(skipped)} pages  "
          f"({skipped_correct} correct predictions lost, "
          f"{skipped_errors} errors also removed)")

    # Per-class accuracy
    class_correct: dict[int, int] = defaultdict(int)
    class_total:   dict[int, int] = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1

    print(f"\n{'─'*60}")
    print("Per-class accuracy:")
    for cls in sorted(class_total):
        n    = class_total[cls]
        ok   = class_correct[cls]
        name = idx_to_label[cls]
        bar  = "█" * int(20 * ok / n) + "░" * (20 - int(20 * ok / n))
        print(f"  {name:>25}  {bar}  {ok:3d}/{n:3d}  ({ok/n:.0%})")

    # ── Error list ─────────────────────────────────────────────────────────
    if errors:
        print(f"\n{'─'*60}")
        print(f"Errors ({len(errors)} total):")
        print(f"  {'PDF file':<42} {'pg':>4}  {'true':>25}  {'predicted':>25}  {'conf':>6}  flag")
        print(f"  {'─'*42} {'─'*4}  {'─'*25}  {'─'*25}  {'─'*6}  {'─'*4}")
        for e in errors:
            flag = "  perm" if e["permissible"] else ("  safe" if e["safe"] else "")
            print(f"  {e['pdf']:<42} {e['pdf_page']:>4}  "
                  f"{e['true']:>25}  {e['predicted']:>25}  {e['confidence']:>6.1%}{flag}")
    else:
        print("\nNo errors — perfect test accuracy!")

    # ── Confusion matrix ───────────────────────────────────────────────────
    class_names = [idx_to_label[i] for i in range(num_classes)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    active = sorted(set(y_true) | set(y_pred))
    cm_active = cm[np.ix_(active, active)]
    names_active = [class_names[i] for i in active]

    fig, ax = plt.subplots(figsize=(max(8, len(active)), max(6, len(active))))
    im = ax.imshow(cm_active, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(names_active)))
    ax.set_yticks(range(len(names_active)))
    ax.set_xticklabels(names_active, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names_active, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(
        f"Confusion Matrix — std {correct/total:.1%}  perm {perm_ok/total:.1%}  ({correct}/{total})",
        fontsize=12,
    )

    thresh = cm_active.max() / 2.0
    for i in range(len(names_active)):
        for j in range(len(names_active)):
            val = cm_active[i, j]
            if val > 0:
                ax.text(j, i, str(val),
                        ha="center", va="center", fontsize=8,
                        color="white" if val > thresh else "black")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nConfusion matrix saved → {args.out}")


if __name__ == "__main__":
    main()
