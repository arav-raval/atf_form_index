"""Master report on a v2 evaluation JSON.

Produces a comprehensive evaluation suitable for the MasterTesting set:

  1. Stage results table (PII Leak, Classification Acc, Admit, Exact, CER)
  2. Verifier confusion matrix (combined and split)
  3. By-document-type breakdown (case_report vs complete_test vs cont vs ...)
  4. By-error-type breakdown (Safe = no corruption vs Unsafe = corrupted)
  5. Per-corruption-type leak
  6. Top-K retrieval (option iii — confusion-aware Levenshtein-distance match
     of the user-typed truth serial against every admitted serial in the
     extraction index). Reports P(true serial in top-K) for K = 1, 5, 10, 25.
  7. CER mean / median / p90 (admitted rows on clean PDFs only)
  8. ASCII top-K chart

Usage::

    python -m pipeline.master_report /tmp/eval_master.json
    python -m pipeline.master_report /tmp/eval_master.json --threshold-sweep
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline.evaluate_v2 import MASTERTESTING_DOC_TYPE, PDF_STEMS
from pipeline.fuzzy_search import canonical_form

# Backfill helper from the existing compliance report
from pipeline.report_compliance import _backfill_corruption_type, _verifier_confusion, _print_confusion_matrix


# ---------- pdf-stem -> doc-type lookup ----------------------------------

def _doc_type_for_pdf(pdf_field: str) -> str:
    """The eval JSON's 'pdf' field is the raw file stem (e.g.
    'v2dataset_no_errors_2', '1985_complete_test'). We need to map it to
    one of the categorical doc types. Walk PDF_STEMS to find a matching
    short name, then look it up in MASTERTESTING_DOC_TYPE.
    """
    for short_name, (_dir, stem) in PDF_STEMS.items():
        if stem == pdf_field and short_name in MASTERTESTING_DOC_TYPE:
            return MASTERTESTING_DOC_TYPE[short_name]
    # Default heuristic for non-MasterTesting PDFs
    if "no_errors" in pdf_field:
        return "v2_clean"
    if pdf_field.startswith("v2dataset"):
        return "v2_corrupted"
    if pdf_field.startswith("serial_only_error"):
        return "serial_error"
    if pdf_field.startswith("serial_only"):
        return "serial_only"
    if pdf_field.startswith("dataset_"):
        return "ds"
    return "other"


# ---------- Levenshtein (small, fast) ------------------------------------

def _levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


# ---------- Top-K retrieval ----------------------------------------------

def _build_index(
    rows: list[dict], admit_threshold: float = 0.50,
) -> list[tuple[str, str, str]]:
    """Return list of (doc_id, ocr_normalized, ocr_canon) for every admitted
    hypothesis from every admitted row. ``doc_id`` = pdf#p<page>.

    A row is "admitted" to the index if its verifier ``p_pos >= admit_threshold``
    (default 0.50, looser than the strict compliance threshold). Each row
    contributes ALL its OCR hypotheses, not just the top-1 — this is the
    multi-hypothesis index from Fix A.
    """
    out: list[tuple[str, str, str]] = []
    for r in rows:
        p_pos = float(r.get("verify_p_pos") or 0.0)
        # Backwards compat: if verify_p_pos missing, fold from confidence.
        if "verify_p_pos" not in r:
            conf = float(r.get("verify_confidence") or 0.0)
            p_pos = conf if r.get("verify_is_serial") else (1.0 - conf)
        if p_pos < admit_threshold:
            continue
        doc_id = f"{r['pdf']}#p{r['page']}"
        hyps = r.get("ocr_hypotheses") or []
        if hyps:
            for h in hyps:
                norm = (h.get("normalized") or "").upper()
                if not norm:
                    continue
                out.append((doc_id, norm, canonical_form(norm)))
        else:
            # Fall back to top-1 fields for old JSONs
            ocr = (r.get("ocr_normalized") or "").upper()
            if ocr:
                out.append((doc_id, ocr, canonical_form(ocr)))
    return out


def _topk_retrieval(
    rows: list[dict], k_values: list[int], admit_threshold: float = 0.50,
) -> dict:
    """For every (PDF, page, firearm) row whose truth has a real serial,
    simulate the user typing that serial and rank all admitted hypotheses
    by confusion-aware Levenshtein. Return P(true doc in top-K)."""
    index = _build_index(rows, admit_threshold)
    if not index:
        return {"queries": 0, "topk": {k: 0.0 for k in k_values},
                "topk_counts": {k: 0 for k in k_values},
                "index_size": 0}

    queries: list[tuple[str, str]] = []
    for r in rows:
        if r.get("truth_label") != "positive":
            continue
        if not r.get("truth_serial"):
            continue
        true_doc_id = f"{r['pdf']}#p{r['page']}"
        queries.append((true_doc_id, r["truth_serial"]))

    if not queries:
        return {"queries": 0, "topk": {k: 0.0 for k in k_values},
                "topk_counts": {k: 0 for k in k_values},
                "index_size": len(index)}

    hits = {k: 0 for k in k_values}
    for true_doc_id, q in queries:
        q_canon = canonical_form(q)
        # Rank by best (min) Levenshtein over each doc's hypotheses.
        # First compute per-(doc, hypothesis) distance, then aggregate by doc
        # taking the minimum.
        per_doc_min: dict[str, int] = {}
        for d, _norm, oc in index:
            dist = _levenshtein(q_canon, oc)
            if d not in per_doc_min or dist < per_doc_min[d]:
                per_doc_min[d] = dist
        scored = sorted(per_doc_min.items(), key=lambda x: x[1])
        # P@K — true_doc is in top K positions
        for k in k_values:
            top_docs = {d for d, _ in scored[:k]}
            if true_doc_id in top_docs:
                hits[k] += 1
    return {
        "queries": len(queries),
        "topk": {k: hits[k] / len(queries) for k in k_values},
        "topk_counts": {k: hits[k] for k in k_values},
        "index_size": len(index),
        "n_unique_docs": len({d for d, _, _ in index}),
    }


# ---------- Per-doc-type metrics -----------------------------------------

def _doc_type_metrics(rows: list[dict]) -> dict[str, dict]:
    """Group rows by doc type; for each, report: n_rows, classifier acc on
    pages, admit rate on positives, exact-match rate on positives, PII leak
    rate (FP / total negatives), CER median on admitted positives."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[_doc_type_for_pdf(r["pdf"])].append(r)

    out: dict[str, dict] = {}
    for dtype, sub in by_type.items():
        positives = [r for r in sub if r["truth_label"] == "positive" and r["truth_serial"]]
        negatives = [r for r in sub if r["truth_label"] == "negative"]
        admitted_pos = [r for r in positives if r["verify_is_serial"] and r["ocr_normalized"]]
        admitted_neg = [r for r in negatives if r["verify_is_serial"]]
        exact = sum(1 for r in admitted_pos if r["ocr_normalized"] == r["truth_serial"])
        cers = [r["cer"] for r in admitted_pos if r.get("cer") is not None]

        # Page-level classifier metrics
        seen_pages: dict[tuple[str, int], tuple[str, str | None]] = {}
        for r in sub:
            seen_pages[(r["pdf"], r["page"])] = (r["truth_year"], r["pred_year"])
        cls_correct = sum(1 for ty, py in seen_pages.values() if py == ty)

        out[dtype] = {
            "n_rows": len(sub),
            "n_pages": len(seen_pages),
            "n_pos": len(positives),
            "n_neg": len(negatives),
            "classifier_acc": cls_correct / max(1, len(seen_pages)),
            "admit_rate_pos": len(admitted_pos) / max(1, len(positives)),
            "exact_match_rate": exact / max(1, len(positives)),
            "pii_leak_rate": len(admitted_neg) / max(1, len(negatives)) if negatives else None,
            "cer_median": statistics.median(cers) if cers else None,
            "cer_mean": statistics.mean(cers) if cers else None,
        }
    return out


# ---------- ASCII top-K chart --------------------------------------------

def _ascii_topk_chart(topk: dict[int, float], width: int = 50) -> str:
    lines: list[str] = []
    lines.append(f"  {'K':>4s}  {'P(top-K)':>9s}  {'-'*width}")
    for k, p in sorted(topk.items()):
        bar_len = int(round(p * width))
        bar = "█" * bar_len + "·" * (width - bar_len)
        lines.append(f"  {k:>4d}  {p:>9.4f}  {bar}")
    return "\n".join(lines)


# ---------- Main ----------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("json_path", type=Path)
    ap.add_argument("--threshold-sweep", action="store_true")
    ap.add_argument("--topk", nargs="+", type=int,
                    default=[1, 3, 5, 10, 25, 50, 100],
                    help="K values for the top-K retrieval analysis")
    ap.add_argument("--admit-threshold", type=float, default=0.50,
                    help="Index-admit threshold on verifier p_pos. Looser than "
                         "the strict compliance gate; the regex tripwire on each "
                         "OCR hypothesis filters PII at the output level.")
    ap.add_argument("--topk-sweep", action="store_true",
                    help="Print top-K curves at multiple admit thresholds")
    args = ap.parse_args()

    data = json.load(open(args.json_path))
    rows = data.get("rows", [])
    if not rows:
        print("No rows in JSON.")
        return 1
    n_back = _backfill_corruption_type(rows)
    if n_back:
        print(f"(backfilled corruption_type for {n_back} rows)")

    # -------- 1. Stage results table -------------------------------------
    seen_pages: dict[tuple[str, int], tuple[str, str | None, str]] = {}
    for r in rows:
        seen_pages[(r["pdf"], r["page"])] = (r["truth_year"], r["pred_year"], r["classifier_status"])
    cls_correct = sum(1 for ty, py, _ in seen_pages.values() if py == ty)
    cls_unsure = sum(1 for *_, st in seen_pages.values() if st == "UNSURE")

    positives = [r for r in rows if r["truth_label"] == "positive" and r["truth_serial"]]
    negatives = [r for r in rows if r["truth_label"] == "negative"]
    admitted_pos = [r for r in positives if r["verify_is_serial"] and r["ocr_normalized"]]
    admitted_neg = [r for r in negatives if r["verify_is_serial"]]
    exact = sum(1 for r in admitted_pos if r["ocr_normalized"] == r["truth_serial"])
    fuzzy = sum(1 for r in admitted_pos if canonical_form(r["ocr_normalized"]) == canonical_form(r["truth_serial"]))
    cers = [r["cer"] for r in admitted_pos if r.get("cer") is not None]
    pii_leak = len(admitted_neg) / max(1, len(negatives)) if negatives else 0.0

    # Best-of-N hypothesis stats: how often is the truth among the top-N
    # OCR hypotheses (Fix A)? Uses cer_best_hyp if present in the row.
    pos_with_hyps = [r for r in positives if r.get("ocr_hypotheses")]
    exact_in_hyps = sum(
        1 for r in pos_with_hyps
        if any(h.get("normalized") == r["truth_serial"]
               for h in r.get("ocr_hypotheses", []))
    )
    cers_best = [r["cer_best_hyp"] for r in pos_with_hyps
                 if r.get("cer_best_hyp") is not None]

    print()
    print("=" * 88)
    print(f" MASTER REPORT — {args.json_path.name}")
    print(f" {len(rows)} rows  |  {len(seen_pages)} pages  |  {len(positives)} positives  |  {len(negatives)} negatives")
    print("=" * 88)

    print()
    print(" STAGE RESULTS")
    print(" " + "-" * 86)
    print(f"   {'Stage':28s}  {'Metric':35s}  {'Value':>10s}")
    print(f"   {'-' * 28}  {'-' * 35}  {'-' * 10}")
    print(f"   {'1. Classify (page)':28s}  {'top-1 year accuracy':35s}  "
          f"{cls_correct / max(1, len(seen_pages)):>10.4f}")
    print(f"   {'1. Classify (page)':28s}  {'UNSURE count':35s}  {cls_unsure:>10d}")
    print(f"   {'2. Verify (compliance gate)':28s}  {'recall (real serials kept)':35s}  "
          f"{len(admitted_pos)/max(1,len(positives)):>10.4f}")
    if negatives:
        print(f"   {'2. Verify (compliance gate)':28s}  {'PII LEAK RATE (lower is better)':35s}  "
              f"{pii_leak:>10.4f}")
    print(f"   {'3. OCR (admitted only)':28s}  {'exact match / admitted':35s}  "
          f"{exact/max(1,len(admitted_pos)):>10.4f}")
    print(f"   {'3. OCR (admitted only)':28s}  {'fuzzy match / admitted':35s}  "
          f"{fuzzy/max(1,len(admitted_pos)):>10.4f}")
    if cers:
        print(f"   {'3. OCR (admitted only)':28s}  {'CER mean / median / p90':35s}  "
              f"{statistics.mean(cers):.3f}/{statistics.median(cers):.3f}/{sorted(cers)[int(0.9*len(cers))]:.3f}")
    print(f"   {'End-to-end':28s}  {'exact match / all positives':35s}  "
          f"{exact/max(1,len(positives)):>10.4f}")
    print(f"   {'End-to-end':28s}  {'fuzzy match / all positives':35s}  "
          f"{fuzzy/max(1,len(positives)):>10.4f}")
    if pos_with_hyps:
        print(f"   {'3. OCR (top-N hypotheses)':28s}  {'truth in any hyp / positives':35s}  "
              f"{exact_in_hyps/max(1,len(positives)):>10.4f}")
        print(f"   {'3. OCR (top-N hypotheses)':28s}  {'truth in any hyp / OCRd':35s}  "
              f"{exact_in_hyps/max(1,len(pos_with_hyps)):>10.4f}")
        if cers_best:
            print(f"   {'3. OCR (top-N hypotheses)':28s}  {'best-hyp CER mean / median':35s}  "
                  f"{statistics.mean(cers_best):.3f}/{statistics.median(cers_best):.3f}")

    # -------- 2. Confusion matrix ----------------------------------------
    print()
    print(" VERIFIER CONFUSION (Real serial vs PII / corrupted)")
    print(" " + "-" * 86)
    _print_confusion_matrix(_verifier_confusion(rows))

    # -------- 3. By-document-type ----------------------------------------
    print()
    print(" BY DOCUMENT TYPE")
    print(" " + "-" * 86)
    dt_metrics = _doc_type_metrics(rows)
    print(f"   {'doc_type':25s}  {'n_pos':>5s}  {'n_neg':>5s}  {'cls_acc':>7s}  "
          f"{'admit%':>7s}  {'exact%':>7s}  {'leak%':>6s}  {'cer_med':>8s}")
    for dtype, m in sorted(dt_metrics.items(), key=lambda kv: -kv[1]["n_rows"]):
        leak_str = f"{100*m['pii_leak_rate']:.1f}" if m['pii_leak_rate'] is not None else "  -  "
        cer_str = f"{m['cer_median']:.3f}" if m['cer_median'] is not None else "  -  "
        print(f"   {dtype:25s}  {m['n_pos']:>5d}  {m['n_neg']:>5d}  "
              f"{m['classifier_acc']:>7.4f}  {100*m['admit_rate_pos']:>6.1f}%  "
              f"{100*m['exact_match_rate']:>6.1f}%  {leak_str:>5s}%  {cer_str:>8s}")

    # -------- 4. By error type (Safe vs Unsafe) --------------------------
    print()
    print(" BY ERROR TYPE")
    print(" " + "-" * 86)
    safe = [r for r in rows if r["truth_label"] == "positive"]   # real serial = SAFE to admit
    unsafe = [r for r in rows if r["truth_label"] == "negative"] # PII/corruption = UNSAFE to admit
    safe_admit = sum(1 for r in safe if r["verify_is_serial"])
    unsafe_admit = sum(1 for r in unsafe if r["verify_is_serial"])
    print(f"   {'category':30s}  {'n':>5s}  {'admitted':>9s}  {'rejected':>9s}  {'rate':>7s}")
    print(f"   {'SAFE (real serial)':30s}  {len(safe):>5d}  {safe_admit:>9d}  "
          f"{len(safe)-safe_admit:>9d}  {100*safe_admit/max(1,len(safe)):>6.1f}%  ← admit is correct here")
    print(f"   {'UNSAFE (PII / corrupted)':30s}  {len(unsafe):>5d}  {unsafe_admit:>9d}  "
          f"{len(unsafe)-unsafe_admit:>9d}  {100*unsafe_admit/max(1,len(unsafe)):>6.1f}%  ← reject is correct here")

    # -------- 5. Per-corruption-type leak --------------------------------
    print()
    print(" PER-CORRUPTION-TYPE LEAK (corrupted firearms only)")
    print(" " + "-" * 86)
    by_corr: dict[str, list[dict]] = defaultdict(list)
    for r in unsafe:
        by_corr[r.get("corruption_type") or "(unknown)"].append(r)
    print(f"   {'corruption type':30s}  {'n':>5s}  {'admitted':>9s}  {'leak%':>6s}")
    for ct, sub in sorted(by_corr.items(), key=lambda kv: -len(kv[1])):
        adm = sum(1 for r in sub if r["verify_is_serial"])
        print(f"   {ct:30s}  {len(sub):>5d}  {adm:>9d}  {100*adm/max(1,len(sub)):>5.1f}%")

    # -------- 6. Top-K retrieval -----------------------------------------
    print()
    print(" TOP-K RETRIEVAL (option iii: user types the true serial → rank by")
    print(" confusion-aware Levenshtein against every admitted hypothesis;")
    print(" doc score = min hypothesis distance; P(true doc in top-K))")
    print(" " + "-" * 86)
    tk = _topk_retrieval(rows, args.topk, admit_threshold=args.admit_threshold)
    print(f"   admit threshold  : {args.admit_threshold:.2f}  (looser than compliance gate)")
    print(f"   queries simulated: {tk['queries']}  (positives with a known serial)")
    print(f"   index size       : {tk.get('index_size', 0)} hypotheses across "
          f"{tk.get('n_unique_docs', 0)} unique docs")
    print()
    if tk["queries"]:
        print(_ascii_topk_chart(tk["topk"]))

    if args.topk_sweep:
        print()
        print(" TOP-K AT DIFFERENT ADMIT THRESHOLDS")
        print(" " + "-" * 86)
        header_ks = args.topk
        head = f"   {'thresh':>7s}  " + "  ".join(f"{f'top-{k}':>7s}" for k in header_ks) + f"  {'idx':>5s}"
        print(head)
        for t in [0.30, 0.50, 0.65, 0.80, 0.90]:
            sw = _topk_retrieval(rows, header_ks, admit_threshold=t)
            row = f"   {t:>7.2f}  " + "  ".join(f"{sw['topk'][k]:>7.4f}" for k in header_ks)
            row += f"  {sw.get('index_size', 0):>5d}"
            print(row)

    # -------- 7. Threshold sweep (optional) ------------------------------
    if args.threshold_sweep:
        print()
        print(" VERIFIER THRESHOLD SWEEP")
        print(" " + "-" * 86)
        print(f"   {'thresh':>7s}  {'recall':>7s}  {'specif':>7s}  {'precis':>7s}  {'pii leak':>9s}  "
              f"{'tp':>4s}/{'fn':<4s}  {'tn':>4s}/{'fp':<4s}")
        for t in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            c = _verifier_confusion(rows, threshold=t)
            print(f"   {t:>7.2f}  {c['recall']:>7.4f}  {c['specificity']:>7.4f}  "
                  f"{c['precision']:>7.4f}  {c['pii_leak_rate']:>9.4f}  "
                  f"{c['tp']:>4d}/{c['fn']:<4d}  {c['tn']:>4d}/{c['fp']:<4d}")

    print()
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
