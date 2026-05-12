"""Train ViT verifier head with class rebalancing.

Same architecture as ``verifier_train_vit`` but subsamples the cached
embeddings before fitting so that the three groups are balanced:

    positive    : capped to N_POS per epoch
    corrupted   : all kept (these are the hard, valuable negatives)
    empty       : capped to ~match corrupted count

Without rebalancing, the dataset is ~10x skewed toward "easy" classes
(positives + empties), and the model learns to be lenient because most
of training is "say yes" or "say no on blank rows" — neither of which
exercises the actual PII discrimination.

Reads the existing ``verifier_vit_features.npz`` cache so we skip the
slow encoder pass entirely. Saves to ``verifier_model_vit.pt`` (overwrites).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from pipeline.verifier_train_vit import _Head, _confusion

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "verifier_vit_features.npz"
MODEL_PATH = ROOT / "verifier_model_vit.pt"
ENCODER_HF_ID = "microsoft/trocr-base-printed"


def _subsample_balanced(
    X: np.ndarray, y: np.ndarray, kinds: list[str], *,
    pos_per_epoch: int, empty_per_epoch: int, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Cap positives to ``pos_per_epoch`` and empties to ``empty_per_epoch``.
    Keeps ALL corrupted negatives (they're the rare, valuable hard cases)."""
    rng = np.random.default_rng(seed)
    pos_idx = np.array([i for i, (yy, k) in enumerate(zip(y, kinds))
                        if yy == 1 and k == "positive"])
    corr_idx = np.array([i for i, (yy, k) in enumerate(zip(y, kinds))
                         if yy == 0 and k == "corrupted"])
    emp_idx = np.array([i for i, (yy, k) in enumerate(zip(y, kinds))
                        if yy == 0 and k == "empty"])

    if len(pos_idx) > pos_per_epoch:
        pos_idx = rng.choice(pos_idx, pos_per_epoch, replace=False)
    if len(emp_idx) > empty_per_epoch:
        emp_idx = rng.choice(emp_idx, empty_per_epoch, replace=False)

    keep = np.concatenate([pos_idx, corr_idx, emp_idx])
    rng.shuffle(keep)
    return X[keep], y[keep], [kinds[i] for i in keep]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=CACHE_PATH)
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--target-recall", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=80,
                    help="More epochs OK because subsampling shrinks the per-epoch dataset")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-per-epoch", type=int, default=2000,
                    help="Cap positives per epoch (default 2000 = ~2.4x corrupted count)")
    ap.add_argument("--empty-per-epoch", type=int, default=1000,
                    help="Cap empty negatives per epoch (default 1000 = ~1.2x corrupted count)")
    ap.add_argument("--reseed-each-epoch", action="store_true",
                    help="Resample positives/empties every epoch (more diverse, slower)")
    args = ap.parse_args()

    if not args.cache.is_file():
        raise SystemExit(f"Cache not found: {args.cache}. Run verifier_train_vit first to extract embeddings.")

    print(f"Loading cached embeddings from {args.cache}", flush=True)
    d = np.load(args.cache, allow_pickle=True)
    X_tr, y_tr, kinds_tr = d["X_tr"], d["y_tr"], list(d["kinds_tr"])
    X_va, y_va, kinds_va = d["X_va"], d["y_va"], list(d["kinds_va"])
    print(f"Full train: {len(y_tr)}  (pos={int((y_tr==1).sum())}, neg={int((y_tr==0).sum())})", flush=True)
    print(f"Val:        {len(y_va)}  (pos={int((y_va==1).sum())}, neg={int((y_va==0).sum())})", flush=True)

    n_corr = sum(1 for k in kinds_tr if k == "corrupted")
    print(f"\nRebalancing target per epoch:")
    print(f"  positives:           {args.pos_per_epoch}  (full set has {sum(1 for k in kinds_tr if k=='positive')})")
    print(f"  corrupted negatives: {n_corr}  (kept all)")
    print(f"  empty negatives:     {args.empty_per_epoch}  (full set has {sum(1 for k in kinds_tr if k=='empty')})")
    print(f"  effective per-epoch n: {args.pos_per_epoch + n_corr + args.empty_per_epoch}", flush=True)

    # Initial subsample (used as the static training set unless --reseed-each-epoch)
    X_tr_b, y_tr_b, kinds_tr_b = _subsample_balanced(
        X_tr, y_tr, kinds_tr,
        pos_per_epoch=args.pos_per_epoch, empty_per_epoch=args.empty_per_epoch,
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_b).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)

    head_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    head = _Head(in_dim=X_tr_s.shape[1]).to(head_device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"\nMLP head: {n_params:,} trainable params on {head_device}", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_spec = -1.0
    best_metrics: dict = {}
    best_state: dict | None = None
    candidate_thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                            0.80, 0.85, 0.90, 0.95, 0.97, 0.99]

    Xt_va = torch.from_numpy(X_va_s).to(head_device)

    for epoch in range(1, args.epochs + 1):
        if args.reseed_each_epoch and epoch > 1:
            X_tr_b, y_tr_b, kinds_tr_b = _subsample_balanced(
                X_tr, y_tr, kinds_tr,
                pos_per_epoch=args.pos_per_epoch, empty_per_epoch=args.empty_per_epoch,
                seed=epoch,
            )
            X_tr_s = scaler.transform(X_tr_b).astype(np.float32)
        Xt_tr = torch.from_numpy(X_tr_s).to(head_device)
        yt_tr = torch.from_numpy(y_tr_b).to(head_device)
        n_train = len(yt_tr)

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
                     f"rec={epoch_best['recall']:.3f} spec={epoch_best['specificity']:.3f} "
                     f"leak={epoch_best['pii_leak_rate']:.3f}")
        print(line, flush=True)

        if epoch_best and epoch_best["specificity"] > best_spec:
            best_spec = epoch_best["specificity"]
            best_metrics = epoch_best
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    if best_state is None:
        print("WARNING: no epoch hit target recall. Using last epoch state.")
        best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        best_metrics = c50

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        p_va = torch.softmax(head(Xt_va), dim=1)[:, 1].cpu().numpy()
    print(f"\nFinal threshold sweep on val:")
    print(f"   {'thresh':>7s}  {'recall':>7s}  {'specif':>7s}  {'precis':>7s}  {'pii_leak':>9s}  tp/fn  tn/fp")
    for t in candidate_thresholds:
        c = _confusion(y_va, p_va, t)
        marker = "  ← saved" if abs(t - best_metrics["thresh"]) < 1e-6 else ""
        print(f"   {t:>7.2f}  {c['recall']:>7.4f}  {c['specificity']:>7.4f}  {c['precision']:>7.4f}  "
              f"{c['pii_leak_rate']:>9.4f}  {c['tp']}/{c['fn']}  {c['tn']}/{c['fp']}{marker}")

    pred = p_va >= best_metrics["thresh"]
    print(f"\nPer-kind val performance @ chosen threshold ({best_metrics['thresh']}):")
    for kind in ["positive", "corrupted", "empty"]:
        mask = np.array([k == kind for k in kinds_va])
        if not mask.any(): continue
        adm = int(pred[mask].sum())
        print(f"  {kind:12s}  n={mask.sum():4d}  admitted={adm:4d}  ({100*adm/mask.sum():.1f}%)")

    bundle = {
        "head_state_dict": best_state,
        "head_in_dim": X_tr_s.shape[1],
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "decision_threshold": best_metrics["thresh"],
        "metrics": best_metrics,
        "encoder_hf_id": ENCODER_HF_ID,
        "arch": "VitTrocrEncoder",
        "training": "balanced_subsample",
        "config": {
            "pos_per_epoch": args.pos_per_epoch,
            "empty_per_epoch": args.empty_per_epoch,
            "epochs": args.epochs,
            "target_recall": args.target_recall,
        },
    }
    torch.save(bundle, args.out)
    print(f"\nSaved {args.out}")
    print(f"Best metrics: {json.dumps(best_metrics, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
