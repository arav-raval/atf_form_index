"""Fine-tune TrOCR (microsoft/trocr-base-printed) on row-crop pairs.

Strategy:
  - Freeze the vision encoder (it already produces good features and we have
    only ~hundreds-to-low-thousands of training pairs).
  - Train the decoder + cross-attention only.
  - Low LR, gradient clipping, OneCycle schedule.
  - Best-checkpoint by exact-match on the val split.

Reads data from ``--data`` (default ``ocr_finetune_data``) with layout::

    <data>/train/<id>.png
    <data>/train/labels.tsv          # ``<id>\\t<NORMALIZED_SERIAL>`` per line
    <data>/val/<id>.png
    <data>/val/labels.tsv

Saves the best-by-val-exact-match checkpoint to ``--out`` as a HuggingFace
``save_pretrained`` directory (config.json, generation_config.json,
model.safetensors). The processor is also saved so load is one-call.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "ocr_finetune_data"
DEFAULT_OUT = ROOT / "trocr_finetuned"
BASE_HF_ID = "microsoft/trocr-base-printed"


class _PairDataset(Dataset):
    def __init__(self, root: Path, split: str, processor, tokenizer):
        self.root = root / split
        self.processor = processor
        self.tokenizer = tokenizer
        self.items: list[tuple[Path, str]] = []
        labels_tsv = self.root / "labels.tsv"
        with open(labels_tsv) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                fn, label = line.split("\t", 1)
                self.items.append((self.root / fn, label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values[0]
        token_ids = self.tokenizer(
            label, padding="max_length", max_length=24, truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        token_ids = torch.where(
            token_ids == self.tokenizer.pad_token_id,
            torch.full_like(token_ids, -100),
            token_ids,
        )
        return {"pixel_values": pixel_values, "labels": token_ids, "_label_str": label}


def _collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "_label_strs": [b["_label_str"] for b in batch],
    }


def _normalize(s: str) -> str:
    return "".join(c for c in s.upper() if c.isalnum())


def _eval_exact_match(model, processor, loader, device):
    model.eval()
    correct = total = 0
    samples: list[tuple[str, str]] = []
    with torch.no_grad():
        for batch in loader:
            pv = batch["pixel_values"].to(device)
            ids = model.generate(pv, max_new_tokens=24)
            preds = processor.batch_decode(ids, skip_special_tokens=True)
            for pred, truth in zip(preds, batch["_label_strs"]):
                if _normalize(pred) == _normalize(truth):
                    correct += 1
                total += 1
                if len(samples) < 8:
                    samples.append((truth, pred))
    return (correct / max(1, total)), samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help="Dataset directory with train/ and val/ subdirs (each with labels.tsv)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="Save best checkpoint here")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cpu", "mps", "cuda"],
                    help="Force device. MPS may hang on TrOCR fine-tune ops "
                         "after laptop sleep — CPU is the safe fallback.")
    ap.add_argument("--unfreeze-encoder", action="store_true",
                    help="Train the vision encoder too (with --encoder-lr-scale "
                         "× --lr). Only viable with thousands of training pairs.")
    ap.add_argument("--encoder-lr-scale", type=float, default=0.1,
                    help="LR multiplier for the encoder when --unfreeze-encoder "
                         "is set (default 0.1 = 10x lower than decoder).")
    args = ap.parse_args()

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    processor = TrOCRProcessor.from_pretrained(BASE_HF_ID)
    model = VisionEncoderDecoderModel.from_pretrained(BASE_HF_ID)
    tokenizer = processor.tokenizer
    model.config.decoder_start_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    if not args.unfreeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {n_trainable:,} / total {n_total:,} (encoder frozen)", flush=True)
    else:
        # All params trainable; the optimizer will use a lower LR for the encoder.
        n_trainable = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {n_trainable:,} (encoder UNFROZEN at LR={args.lr * args.encoder_lr_scale:.1e})", flush=True)

    model.to(device)

    train_ds = _PairDataset(args.data, "train", processor, tokenizer)
    val_ds = _PairDataset(args.data, "val", processor, tokenizer)
    print(f"Train pairs: {len(train_ds)}  Val pairs: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=_collate)

    n_steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * n_steps_per_epoch
    if args.unfreeze_encoder:
        # Per-group LR: encoder lower, decoder full.
        encoder_params = list(model.encoder.parameters())
        decoder_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
        param_groups = [
            {"params": encoder_params, "lr": args.lr * args.encoder_lr_scale},
            {"params": decoder_params, "lr": args.lr},
        ]
        opt = torch.optim.AdamW(param_groups, weight_decay=0.01)
        max_lrs = [args.lr * args.encoder_lr_scale, args.lr]
    else:
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.01,
        )
        max_lrs = args.lr
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lrs, total_steps=total_steps,
        pct_start=args.warmup / max(args.warmup, total_steps),
        anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0,
    )

    base_acc, base_samples = _eval_exact_match(model, processor, val_loader, device)
    print(f"\nBaseline (no fine-tune) val exact-match: {base_acc:.4f}", flush=True)
    print("  baseline samples (truth -> pred):", flush=True)
    for t, p in base_samples:
        print(f"    {t!r:25s} -> {p!r}", flush=True)

    best_acc = base_acc
    best_epoch = 0
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"  baseline saved to {args.out} (will be overwritten on improvement)", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        t0 = time.time()
        n_steps = len(train_loader)
        for step, batch in enumerate(train_loader, 1):
            pv = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            opt.zero_grad()
            out = model(pixel_values=pv, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * pv.size(0)
            n_seen += pv.size(0)
            if step % 25 == 0 or step == n_steps:
                rate = n_seen / (time.time() - t0)
                print(f"  ep{epoch} step {step}/{n_steps}  "
                      f"loss(avg)={running/max(1,n_seen):.4f}  "
                      f"({rate:.1f} samp/s)", flush=True)
        train_loss = running / max(1, n_seen)

        val_acc, samples = _eval_exact_match(model, processor, val_loader, device)
        line = (f"Epoch {epoch:2d}  loss={train_loss:.4f}  val_exact={val_acc:.4f}  "
                f"({time.time()-t0:.1f}s)")
        print(line, flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            model.save_pretrained(args.out)
            processor.save_pretrained(args.out)
            print(f"  → saved {args.out}  (val_exact={best_acc:.4f})", flush=True)
            print("  example predictions:", flush=True)
            for t, p in samples[:5]:
                marker = " ✓" if _normalize(t) == _normalize(p) else ""
                print(f"    {t!r:25s} -> {p!r}{marker}", flush=True)

    print()
    print(f"Best val exact-match: {best_acc:.4f}  (epoch {best_epoch})", flush=True)
    print(f"Saved fine-tuned model to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
