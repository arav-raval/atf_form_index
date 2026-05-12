"""Train the ViT verifier head: TrOCR encoder (frozen) + MLP head.

Pipeline:
    crop -> TrOCRProcessor -> ViT encoder (frozen) -> 768-dim mean-pool feature
         -> StandardScaler -> MLP[768->128->2] -> p_pos
                              ^^^^^^^^^^^^^^^^ trained, ~100k params

Saves ``verifier_model_vit.pt`` with the head state_dict + scaler stats +
chosen decision threshold. ``pipeline.verify`` loads this transparently.

Best-checkpoint policy: maximize specificity subject to ``recall >= --target-recall``.
With a ``--target-recall`` of e.g. 0.5 you can push the operating point hard
toward "reject anything ambiguous" — desirable when PII leak must be near 0.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "verifier_data_v2"
MODEL_PATH = ROOT / "verifier_model_vit.pt"
CACHE_PATH = ROOT / "verifier_vit_features.npz"

ENCODER_HF_ID = "microsoft/trocr-base-printed"


class _Head(nn.Module):
    """MLP head: 768 -> 128 -> 2."""

    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_encoder() -> tuple:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    print(f"Loading TrOCR encoder from {ENCODER_HF_ID}...", flush=True)
    processor = TrOCRProcessor.from_pretrained(ENCODER_HF_ID)
    full = VisionEncoderDecoderModel.from_pretrained(ENCODER_HF_ID)
    encoder = full.encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    encoder.to(device)
    print(f"  encoder on {device}, {sum(p.numel() for p in encoder.parameters()):,} (frozen) params", flush=True)
    return processor, encoder, device


def _embed(crop: Image.Image, processor, encoder, device) -> np.ndarray:
    rgb = crop.convert("RGB")
    pv = processor(images=rgb, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        out = encoder(pixel_values=pv)
        feats = out.last_hidden_state.mean(dim=1).squeeze(0)
    return feats.cpu().numpy().astype(np.float32)


def _load_split_features(split: str, processor, encoder, device, data_root: Path):
    pos_dir = data_root / split / "positive"
    neg_dir = data_root / split / "negative"
    items: list[tuple[Path, int, str]] = []
    for p in sorted(pos_dir.glob("*.png")):
        items.append((p, 1, "positive"))
    for p in sorted(neg_dir.glob("*.png")):
        kind = "empty" if "_empty" in p.name else "corrupted"
        items.append((p, 0, kind))

    print(f"Embedding {len(items)} {split} crops...", flush=True)
    feats: list[np.ndarray] = []
    labels: list[int] = []
    kinds: list[str] = []
    t0 = time.time()
    for i, (p, lbl, k) in enumerate(items):
        img = Image.open(p)
        feats.append(_embed(img, processor, encoder, device))
        labels.append(lbl)
        kinds.append(k)
        if (i + 1) % 100 == 0:
            print(f"  [{split}] {i+1}/{len(items)} ({(i+1)/(time.time()-t0):.1f}/s)", flush=True)
    X = np.stack(feats)
    y = np.array(labels, dtype=np.int64)
    print(f"  [{split}] done in {time.time()-t0:.1f}s, X shape={X.shape}", flush=True)
    return X, y, kinds


def _confusion(y_true: np.ndarray, p_pos: np.ndarray, threshold: float) -> dict:
    pred = p_pos >= threshold
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    n = tp + fn + tn + fp
    return {
        "thresh": threshold,
        "tp": tp, "fn": fn, "tn": tn, "fp": fp, "n": n,
        "recall": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "precision": tp / max(1, tp + fp),
        "accuracy": (tp + tn) / max(1, n),
        "pii_leak_rate": fp / max(1, fp + tn),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DATA_ROOT)
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--cache", type=Path, default=CACHE_PATH,
                    help="Cache embeddings here so reruns skip the slow encoder pass. "
                         "Delete this file if --data has changed.")
    ap.add_argument("--target-recall", type=float, default=0.5,
                    help="Pick checkpoint with highest specificity at recall>=target")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if args.cache.is_file():
        print(f"Loading cached embeddings from {args.cache}", flush=True)
        d = np.load(args.cache, allow_pickle=True)
        X_tr, y_tr, kinds_tr = d["X_tr"], d["y_tr"], list(d["kinds_tr"])
        X_va, y_va, kinds_va = d["X_va"], d["y_va"], list(d["kinds_va"])
    else:
        processor, encoder, device = _load_encoder()
        X_tr, y_tr, kinds_tr = _load_split_features("train", processor, encoder, device, args.data)
        X_va, y_va, kinds_va = _load_split_features("val", processor, encoder, device, args.data)
        np.savez(args.cache, X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va,
                 kinds_tr=np.array(kinds_tr), kinds_va=np.array(kinds_va))
        print(f"Cached embeddings to {args.cache}", flush=True)

    print(f"\nTrain: {len(y_tr)}  (pos={int((y_tr==1).sum())}, neg={int((y_tr==0).sum())})", flush=True)
    print(f"Val:   {len(y_va)}  (pos={int((y_va==1).sum())}, neg={int((y_va==0).sum())})", flush=True)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)

    head_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    head = _Head(in_dim=X_tr_s.shape[1]).to(head_device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"\nMLP head: {n_params:,} trainable params on {head_device}", flush=True)

    Xt_tr = torch.from_numpy(X_tr_s).to(head_device)
    yt_tr = torch.from_numpy(y_tr).to(head_device)
    Xt_va = torch.from_numpy(X_va_s).to(head_device)
    yt_va = torch.from_numpy(y_va).to(head_device)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_spec = -1.0
    best_metrics: dict = {}
    best_state: dict | None = None
    n_train = len(yt_tr)

    candidate_thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                             0.80, 0.85, 0.90, 0.95, 0.97, 0.99]

    for epoch in range(1, args.epochs + 1):
        head.train()
        perm = torch.randperm(n_train, device=head_device)
        running = 0.0
        for i in range(0, n_train, args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = Xt_tr[idx], yt_tr[idx]
            opt.zero_grad()
            logits = head(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        sched.step()
        train_loss = running / max(1, n_train)

        head.eval()
        with torch.no_grad():
            p_va = torch.softmax(head(Xt_va), dim=1)[:, 1].cpu().numpy()

        # Find this epoch's best operating point under the recall constraint.
        epoch_best_spec = -1.0
        epoch_best: dict = {}
        for t in candidate_thresholds:
            c = _confusion(y_va, p_va, t)
            if c["recall"] >= args.target_recall and c["specificity"] > epoch_best_spec:
                epoch_best_spec = c["specificity"]
                epoch_best = c

        c50 = _confusion(y_va, p_va, 0.5)
        line = (f"Epoch {epoch:2d}  loss={train_loss:.4f}  "
                f"@0.5 rec={c50['recall']:.3f} spec={c50['specificity']:.3f}")
        if epoch_best:
            line += (f"  best@{epoch_best['thresh']:.2f} "
                     f"rec={epoch_best['recall']:.3f} spec={epoch_best['specificity']:.3f}")
        print(line, flush=True)

        if epoch_best and epoch_best["specificity"] > best_spec:
            best_spec = epoch_best["specificity"]
            best_metrics = epoch_best
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    if not best_metrics:
        head.eval()
        with torch.no_grad():
            p_va = torch.softmax(head(Xt_va), dim=1)[:, 1].cpu().numpy()
        sweep = [_confusion(y_va, p_va, t) for t in candidate_thresholds]
        best_metrics = max(sweep, key=lambda c: 0.5 * (c["recall"] + c["specificity"]))
        best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        print("WARNING: no epoch hit target recall, using best-balanced-acc fallback", flush=True)

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        p_va = torch.softmax(head(Xt_va), dim=1)[:, 1].cpu().numpy()
    pred = p_va >= best_metrics["thresh"]
    print("\nPer-kind val performance @ chosen threshold:", flush=True)
    for kind in ["positive", "corrupted", "empty"]:
        mask = np.array([k == kind for k in kinds_va])
        if not mask.any(): continue
        adm = int(pred[mask].sum())
        print(f"  {kind:12s}  n={mask.sum():4d}  admitted={adm:4d}  ({100*adm/mask.sum():.1f}%)", flush=True)

    print("\nFull threshold sweep (so you can pick a different operating point):", flush=True)
    print(f"  {'thresh':>7s}  {'recall':>7s}  {'specif':>7s}  {'precis':>7s}  {'pii_leak':>9s}  tp/fn  tn/fp", flush=True)
    for t in candidate_thresholds:
        c = _confusion(y_va, p_va, t)
        marker = "  ← saved" if abs(t - best_metrics["thresh"]) < 1e-9 else ""
        print(f"  {t:>7.2f}  {c['recall']:>7.4f}  {c['specificity']:>7.4f}  "
              f"{c['precision']:>7.4f}  {c['pii_leak_rate']:>9.4f}  "
              f"{c['tp']}/{c['fn']}  {c['tn']}/{c['fp']}{marker}", flush=True)

    bundle = {
        "head_state_dict": best_state,
        "head_in_dim": X_tr_s.shape[1],
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "decision_threshold": best_metrics["thresh"],
        "metrics": best_metrics,
        "encoder_hf_id": ENCODER_HF_ID,
        "arch": "VitTrocrEncoder",
    }
    torch.save(bundle, args.out)
    print(f"\nSaved {args.out}", flush=True)
    print(f"Best metrics: {json.dumps(best_metrics, indent=2)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
