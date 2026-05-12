"""Compliance-focused report on a v2 evaluation JSON.

Reads the per-row JSON dumped by ``evaluate_v2 --json`` and prints a
compliance-first view: PII leak rate is the headline number, with per-
corruption-type breakdown so we can see which kinds of PII slip through.

Usage::

    python -m pipeline.report_compliance /tmp/eval_v2.json
    python -m pipeline.report_compliance /tmp/eval_v2.json --threshold-sweep
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


# Corruption type taxonomy. Two categories of "negatives":
#   PII corruptions: privacy-violating if admitted (name/PII written in the
#                    serial cell). Compliance-critical.
#   Data-quality corruptions: not PII but still wrong data (field swap,
#                    overflow from neighboring cell). Index pollution issue.
# ``serial_overflow`` is a positive in our scheme — the serial IS in the box.
PII_CORRUPTIONS: set[str] = {"pii_in_serial", "name_in_serial"}
DATA_QUALITY_CORRUPTIONS: set[str] = {"field_swap", "overflow_into_serial"}


def _verifier_confusion(
    rows: list[dict], threshold: float | None = None,
    *, neg_filter: set[str] | None = None,
) -> dict:
    """Compute confusion matrix. If ``neg_filter`` is provided, only count
    negative rows whose ``corruption_type`` is in the filter set (e.g. only
    PII corruptions, or only data-quality corruptions).
    """
    tp = fn = tn = fp = 0
    for r in rows:
        if r["truth_label"] not in ("positive", "negative"):
            continue
        # Filter negatives by corruption type if requested
        if r["truth_label"] == "negative" and neg_filter is not None:
            ct = r.get("corruption_type", "")
            if ct not in neg_filter:
                continue
        if threshold is None:
            pred = r["verify_is_serial"]
        else:
            p_pos = r.get("verify_p_pos")
            if p_pos is None:
                p_pos = r["verify_confidence"] if r["verify_is_serial"] else (1 - r["verify_confidence"])
            pred = p_pos >= threshold
        truth_pos = r["truth_label"] == "positive"
        if truth_pos and pred: tp += 1
        elif truth_pos: fn += 1
        elif pred: fp += 1
        else: tn += 1
    n = tp + fn + tn + fp
    return {
        "tp": tp, "fn": fn, "tn": tn, "fp": fp, "n": n,
        "recall": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "precision": tp / max(1, tp + fp),
        "accuracy": (tp + tn) / max(1, n),
        "pii_leak_rate": fp / max(1, fp + tn),
    }


def _print_confusion_matrix(c: dict) -> None:
    print()
    print(f"   {'':28s}  {'verifier: ADMIT':>16s}  {'verifier: REJECT':>17s}")
    print(f"   {'truth: Real serial':28s}  {c['tp']:>16d}  {c['fn']:>17d}")
    print(f"   {'truth: PII / corrupted':28s}  {c['fp']:>16d}  {c['tn']:>17d}")
    print()
    print(f"   Recall (real serials kept)        : {c['recall']:.4f}  ({c['tp']}/{c['tp']+c['fn']})")
    print(f"   Specificity (PII rejected)        : {c['specificity']:.4f}  ({c['tn']}/{c['tn']+c['fp']})")
    print(f"   Precision (admitted are real)     : {c['precision']:.4f}  ({c['tp']}/{c['tp']+c['fp']})")
    print(f"   PII LEAK RATE (lower is better)   : {c['pii_leak_rate']:.4f}  ({c['fp']}/{c['fp']+c['tn']} non-serials wrongly admitted)")


def _backfill_corruption_type(rows: list[dict]) -> int:
    """Older eval JSONs don't store corruption_type. Look it up from the
    source dataset JSONs across every dir we know about."""
    repo = Path(__file__).resolve().parent.parent
    candidate_dirs = [
        repo / "Version 2 Datasets",
        repo / "Serial Error Pages",
        repo / "SerialSets",
        repo / "TestSerialSet",
        repo / "Datasets",
    ]
    cache: dict[str, dict[tuple[int, int], str]] = {}

    def index(stem: str) -> dict[tuple[int, int], str]:
        if stem in cache:
            return cache[stem]
        for d_dir in candidate_dirs:
            p = d_dir / f"{stem}.json"
            if p.is_file():
                d = json.load(open(p))
                idx: dict[tuple[int, int], str] = {}
                for pg in d["pages"]:
                    for i, fa in enumerate(pg["firearms"]):
                        ct = (fa.get("corruption") or {}).get("type", "")
                        idx[(int(pg["page"]), i)] = ct
                cache[stem] = idx
                return idx
        cache[stem] = {}
        return cache[stem]

    updated = 0
    for r in rows:
        if r.get("corruption_type"):
            continue
        ct = index(r["pdf"]).get((int(r["page"]), int(r["row_idx"])), "")
        if ct:
            r["corruption_type"] = ct
            updated += 1
        elif "corruption_type" not in r:
            r["corruption_type"] = ""
    return updated


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("json_path", type=Path)
    ap.add_argument("--threshold-sweep", action="store_true",
                    help="Also print verifier confusion at multiple thresholds")
    args = ap.parse_args()

    data = json.load(open(args.json_path))
    rows = data.get("rows", [])
    if not rows:
        print("No rows in JSON.")
        return 1

    n_backfilled = _backfill_corruption_type(rows)
    if n_backfilled:
        print(f"(backfilled corruption_type from v2 JSONs for {n_backfilled} rows)")

    # A row is "clean" if its source PDF advertises no errors, OR if it's
    # from a dataset with no corruption labels at all (e.g. SerialSets which
    # is plain serials, no PII). A row is "err" if its source PDF advertises
    # errors, even using "error" (singular) in the name.
    def _is_clean(pdf: str) -> bool:
        return "no_errors" in pdf or pdf.startswith("serial_only_") and "error" not in pdf
    def _is_err(pdf: str) -> bool:
        return ("no_errors" not in pdf) and ("errors" in pdf or "error" in pdf)
    clean = [r for r in rows if _is_clean(r["pdf"])]
    err = [r for r in rows if _is_err(r["pdf"])]

    print("=" * 80)
    print(f" COMPLIANCE REPORT — {args.json_path.name}")
    print(f" Total rows evaluated: {len(rows)}  (clean PDF: {len(clean)}, corrupted PDF: {len(err)})")
    print("=" * 80)

    print()
    print(" VERIFIER COMPLIANCE GATE")
    print(" " + "-" * 78)
    print(" Combined confusion (both PDFs, ALL corruption types):")
    _print_confusion_matrix(_verifier_confusion(rows))
    print()
    print(" Corrupted-PDF only confusion (the adversarial test):")
    _print_confusion_matrix(_verifier_confusion(err))

    # PII-only confusion: filter negatives to just the privacy-violating
    # corruptions (name_in_serial, pii_in_serial). This is the metric an
    # actual compliance audit cares about. field_swap and overflow_into_serial
    # are wrong data but not personal info — different concern.
    print()
    print(" PII-ONLY confusion (negatives = name_in_serial + pii_in_serial):")
    print(" " + "-" * 78)
    print(" These are the privacy-violating corruption types. Field swaps and")
    print(" overflow-from-neighbor are wrong data but NOT personal info, so")
    print(" they're excluded from this slice.")
    _print_confusion_matrix(_verifier_confusion(err, neg_filter=PII_CORRUPTIONS))

    print()
    print(" DATA-QUALITY-ONLY confusion (negatives = field_swap + overflow_into_serial):")
    print(" " + "-" * 78)
    _print_confusion_matrix(_verifier_confusion(err, neg_filter=DATA_QUALITY_CORRUPTIONS))

    print()
    print(" Per-corruption-type leak (errors PDF):")
    print(" " + "-" * 78)
    by_corr: dict[str, list[dict]] = defaultdict(list)
    for r in err:
        ct = r.get("corruption_type") or "(clean)"
        by_corr[ct].append(r)
    print(f"   {'corruption type':30s}  {'category':12s}  {'n':>5s}  "
          f"{'admitted':>9s}  {'rejected':>9s}  {'leak%':>6s}")
    print(f"   {'-' * 30}  {'-' * 12}  {'-' * 5}  {'-' * 9}  {'-' * 9}  {'-' * 6}")
    for ct, sub in sorted(by_corr.items(), key=lambda kv: -len(kv[1])):
        admitted = sum(1 for r in sub if r["verify_is_serial"])
        rejected = len(sub) - admitted
        if ct == "(clean)":
            cat = "positive"
            suffix = "*"
        elif ct in PII_CORRUPTIONS:
            cat = "PII"
            suffix = ""
        elif ct in DATA_QUALITY_CORRUPTIONS:
            cat = "data-quality"
            suffix = ""
        elif ct == "serial_overflow":
            cat = "positive"
            suffix = "†"
        else:
            cat = "?"
            suffix = ""
        print(f"   {ct:30s}  {cat:12s}  {len(sub):>5d}  {admitted:>9d}  {rejected:>9d}  "
              f"{100*admitted/max(1,len(sub)):>5.1f}%{suffix}")
    print(f"   * For (clean) the percentage shown is the admit rate, not a leak.")
    print(f"   † serial_overflow IS a positive in our scheme — high admit% is correct.")

    if args.threshold_sweep:
        print()
        print(" Threshold sweep — leak rates split by corruption category:")
        print(" " + "-" * 78)
        print(f"   {'thresh':>7s}  {'recall':>7s}  "
              f"{'pii_leak':>9s}  {'dq_leak':>9s}  {'all_leak':>9s}  "
              f"{'tp':>4s}/{'fn':<4s}")
        for thresh in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                       0.80, 0.85, 0.90, 0.95, 0.97, 0.985, 0.99]:
            c_all = _verifier_confusion(rows, threshold=thresh)
            c_pii = _verifier_confusion(rows, threshold=thresh, neg_filter=PII_CORRUPTIONS)
            c_dq = _verifier_confusion(rows, threshold=thresh, neg_filter=DATA_QUALITY_CORRUPTIONS)
            print(f"   {thresh:>7.3f}  {c_all['recall']:>7.4f}  "
                  f"{c_pii['pii_leak_rate']:>9.4f}  {c_dq['pii_leak_rate']:>9.4f}  "
                  f"{c_all['pii_leak_rate']:>9.4f}  "
                  f"{c_all['tp']:>4d}/{c_all['fn']:<4d}")

    print()
    print(" SERIAL EXTRACTION (clean PDF)")
    print(" " + "-" * 78)
    from pipeline.fuzzy_search import canonical_form

    def _substring_match(pred: str, truth: str) -> bool:
        if not pred or not truth or len(truth) < 4:
            return False
        return truth in pred or pred in truth

    positives = [r for r in clean if r["truth_label"] == "positive" and r["truth_serial"]]
    admitted = [r for r in positives if r["verify_is_serial"] and r["ocr_normalized"]]
    exact = sum(1 for r in admitted if r["ocr_normalized"] == r["truth_serial"])
    fuzzy = sum(1 for r in admitted
                if canonical_form(r["ocr_normalized"]) == canonical_form(r["truth_serial"]))
    substr = sum(1 for r in admitted
                 if _substring_match(r["ocr_normalized"], r["truth_serial"])
                 or canonical_form(r["ocr_normalized"]) == canonical_form(r["truth_serial"]))
    cers = [r["cer"] for r in admitted if r["cer"] is not None]
    n_pos = max(1, len(positives))
    n_adm = max(1, len(admitted))
    print(f"   Real firearms (positive+serial)              : {len(positives)}")
    print(f"   Verifier admitted                            : {len(admitted)}  ({100*len(admitted)/n_pos:.1f}%)")
    print(f"   Exact match    (of admitted / of positives)  : {exact:>3d}  ({100*exact/n_adm:.1f}% / {100*exact/n_pos:.1f}%)")
    print(f"   Fuzzy match    (of admitted / of positives)  : {fuzzy:>3d}  ({100*fuzzy/n_adm:.1f}% / {100*fuzzy/n_pos:.1f}%)")
    print(f"   Substring+fuzzy(of admitted / of positives)  : {substr:>3d}  ({100*substr/n_adm:.1f}% / {100*substr/n_pos:.1f}%)")
    if cers:
        print(f"   CER (admitted) mean / median                 : {statistics.mean(cers):.4f} / {statistics.median(cers):.4f}")

    print()
    print(" PER-YEAR (clean PDF)")
    print(" " + "-" * 78)
    by_year: dict[str, list[dict]] = defaultdict(list)
    for r in clean:
        if r["truth_label"] == "positive" and r["truth_serial"]:
            by_year[r["truth_year"]].append(r)
    print(f"   {'year':5s}  {'n':>4s}  {'admit%':>7s}  {'exact%':>7s}  {'cer_med':>8s}")
    for y, sub in sorted(by_year.items()):
        adm = [r for r in sub if r["verify_is_serial"] and r["ocr_normalized"]]
        ex = sum(1 for r in adm if r["ocr_normalized"] == r["truth_serial"])
        cers_y = [r["cer"] for r in adm if r["cer"] is not None]
        cer_str = f"{statistics.median(cers_y):.3f}" if cers_y else "  -  "
        print(f"   {y:5s}  {len(sub):>4d}  {100*len(adm)/max(1,len(sub)):>6.1f}%  "
              f"{100*ex/max(1,len(sub)):>6.1f}%  {cer_str:>8s}")

    print()
    print(" CLASSIFIER (per page)")
    print(" " + "-" * 78)
    seen_pages: dict[tuple[str, int], tuple[str, str | None, str]] = {}
    for r in rows:
        seen_pages[(r["pdf"], r["page"])] = (r["truth_year"], r["pred_year"], r["classifier_status"])
    n_pages = len(seen_pages)
    correct = sum(1 for ty, py, _ in seen_pages.values() if py == ty)
    unsure = sum(1 for *_, st in seen_pages.values() if st == "UNSURE")
    print(f"   Pages : {n_pages}")
    print(f"   Top-1 year accuracy : {correct/max(1,n_pages):.4f}  (UNSURE: {unsure})")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
