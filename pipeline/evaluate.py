"""Evaluation harness — runs the pipeline over labeled test data and reports
classifier accuracy, serial-match rate, character error rate, and a breakdown
of verifier verdicts.

Usage::

    python -m pipeline.evaluate                       # both TestData1 + TestData2
    python -m pipeline.evaluate --dirs TestData1      # one cohort
    python -m pipeline.evaluate --limit 20            # quick subset
    python -m pipeline.evaluate --json out.json       # also dump full results

This is a pure-read evaluation: it runs the pipeline (no sidecar writes, no
search-index writes) so it never mutates state. Run it before and after a
change to see what moved.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from page_sampling_pipeline import _discover_labeled_pdfs
from pipeline import orchestrator
from pipeline.fuzzy_search import canonical_form
from pipeline.recognize import normalize_serial

ROOT = Path(__file__).resolve().parent.parent
FORM_TEMPLATES = ROOT / "FormTemplates"


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _cer(pred: str, truth: str) -> float:
    if not truth:
        return 0.0 if not pred else 1.0
    return _levenshtein(pred, truth) / len(truth)


@dataclass
class CaseResult:
    pdf: str
    cohort: str
    true_year: str
    pred_year: str | None
    classifier_score: float
    classifier_status: str
    pipeline_status: str
    true_serial: str
    true_serial_norm: str
    admitted_serials: list[str]
    best_match: str            # closest admitted serial by Levenshtein
    cer: float
    exact_match: bool
    fuzzy_match: bool          # True if any admitted serial is OCR-equivalent to truth
    elapsed_s: float
    verify_counts: dict[str, int] = field(default_factory=dict)


def evaluate_one(pdf: Path, cohort: str, true_year: str, true_serial: str) -> CaseResult:
    t0 = time.perf_counter()
    pr = orchestrator.run(pdf, templates_dir=FORM_TEMPLATES)
    elapsed = time.perf_counter() - t0

    admitted = pr.admitted_serials
    truth_norm = normalize_serial(true_serial)

    # "Best match": admitted serial with minimum CER vs truth, for char-level
    # error reporting even when nothing matches exactly.
    if admitted:
        best = min(admitted, key=lambda s: _levenshtein(s, truth_norm))
    else:
        best = ""

    verify_counts = Counter(r.verify_status for r in pr.rows)

    truth_canon = canonical_form(truth_norm)
    fuzzy = any(canonical_form(s) == truth_canon for s in admitted)

    return CaseResult(
        pdf=pdf.name,
        cohort=cohort,
        true_year=true_year,
        pred_year=pr.predicted_year,
        classifier_score=round(pr.classifier_score, 4),
        classifier_status=pr.classifier_status,
        pipeline_status=pr.status,
        true_serial=true_serial,
        true_serial_norm=truth_norm,
        admitted_serials=admitted,
        best_match=best,
        cer=round(_cer(best, truth_norm), 4),
        exact_match=truth_norm in admitted,
        fuzzy_match=fuzzy,
        elapsed_s=round(elapsed, 3),
        verify_counts=dict(verify_counts),
    )


def _aggregate(rs: list[CaseResult]) -> dict[str, Any]:
    if not rs:
        return {}

    total = len(rs)
    yr_correct = sum(1 for r in rs if r.pred_year == r.true_year)
    classifier_unsure = sum(1 for r in rs if r.classifier_status == "UNSURE")
    classifier_error = sum(1 for r in rs if r.classifier_status == "ERROR")

    exact = sum(1 for r in rs if r.exact_match)
    fuzzy = sum(1 for r in rs if r.fuzzy_match)
    has_admitted = sum(1 for r in rs if r.admitted_serials)
    cers = [r.cer for r in rs if r.true_serial_norm]
    elapsed = [r.elapsed_s for r in rs]

    verify_total: Counter = Counter()
    for r in rs:
        verify_total.update(r.verify_counts)

    # Compliance proxy: how many rows that the verifier rejected would have been
    # OCR'd by the previous pipeline. (Today we just count rejection reasons.)
    rejections = {k: v for k, v in verify_total.items() if k != "ok"}

    # Per-year breakdown
    by_year: dict[str, dict[str, Any]] = {}
    years = sorted({r.true_year for r in rs})
    for y in years:
        sub = [r for r in rs if r.true_year == y]
        by_year[y] = {
            "n": len(sub),
            "year_acc": sum(1 for r in sub if r.pred_year == r.true_year) / len(sub),
            "exact_match_rate": sum(1 for r in sub if r.exact_match) / len(sub),
            "fuzzy_match_rate": sum(1 for r in sub if r.fuzzy_match) / len(sub),
            "median_cer": round(statistics.median(r.cer for r in sub), 4),
            "mean_cer": round(statistics.mean(r.cer for r in sub), 4),
        }

    # Per-cohort breakdown
    by_cohort: dict[str, dict[str, Any]] = {}
    for c in sorted({r.cohort for r in rs}):
        sub = [r for r in rs if r.cohort == c]
        by_cohort[c] = {
            "n": len(sub),
            "year_acc": sum(1 for r in sub if r.pred_year == r.true_year) / len(sub),
            "exact_match_rate": sum(1 for r in sub if r.exact_match) / len(sub),
            "fuzzy_match_rate": sum(1 for r in sub if r.fuzzy_match) / len(sub),
            "median_cer": round(statistics.median(r.cer for r in sub), 4),
            "mean_cer": round(statistics.mean(r.cer for r in sub), 4),
        }

    return {
        "n": total,
        "classifier": {
            "year_top1_acc": round(yr_correct / total, 4),
            "unsure": classifier_unsure,
            "error": classifier_error,
        },
        "serial": {
            "exact_match_rate": round(exact / total, 4),
            "fuzzy_match_rate": round(fuzzy / total, 4),
            "any_admitted_rate": round(has_admitted / total, 4),
            "mean_cer": round(statistics.mean(cers), 4) if cers else None,
            "median_cer": round(statistics.median(cers), 4) if cers else None,
        },
        "verify_row_counts": dict(verify_total),
        "verify_rejection_counts": rejections,
        "throughput": {
            "mean_s_per_pdf": round(statistics.mean(elapsed), 3),
            "median_s_per_pdf": round(statistics.median(elapsed), 3),
        },
        "by_year": by_year,
        "by_cohort": by_cohort,
    }


def _print_report(agg: dict[str, Any]) -> None:
    if not agg:
        print("No cases evaluated.")
        return
    print("=" * 72)
    print(f" Evaluation report — n={agg['n']}")
    print("=" * 72)

    cl = agg["classifier"]
    print(f" Classifier top-1 year accuracy : {cl['year_top1_acc']:.4f}  "
          f"(UNSURE: {cl['unsure']}, ERROR: {cl['error']})")

    s = agg["serial"]
    print(f" Serial exact-match rate        : {s['exact_match_rate']:.4f}")
    print(f" Serial fuzzy-match rate        : {s['fuzzy_match_rate']:.4f}  "
          f"(matches survive OCR confusions like 0/O, 1/I, Z/2)")
    print(f" Any admitted serial            : {s['any_admitted_rate']:.4f}")
    if s["mean_cer"] is not None:
        print(f" CER (best admitted vs truth)   : "
              f"mean={s['mean_cer']:.4f}  median={s['median_cer']:.4f}")

    print(f" Throughput                     : "
          f"{agg['throughput']['mean_s_per_pdf']:.3f}s mean, "
          f"{agg['throughput']['median_s_per_pdf']:.3f}s median")

    print()
    print(" Verifier row verdicts:")
    for k, v in sorted(agg["verify_row_counts"].items(), key=lambda kv: -kv[1]):
        print(f"   {k:14s} {v}")

    print()
    print(" Per-cohort:")
    print(f"   {'cohort':12s}  {'n':>4s}  {'year_acc':>8s}  {'exact':>6s}  {'fuzzy':>6s}  {'meanCER':>7s}")
    for c, m in agg["by_cohort"].items():
        print(f"   {c:12s}  {m['n']:>4d}  {m['year_acc']:>8.4f}  "
              f"{m['exact_match_rate']:>6.4f}  {m['fuzzy_match_rate']:>6.4f}  {m['mean_cer']:>7.4f}")

    print()
    print(" Per-year:")
    print(f"   {'year':6s}  {'n':>4s}  {'year_acc':>8s}  {'exact':>6s}  {'fuzzy':>6s}  {'meanCER':>7s}")
    for y, m in agg["by_year"].items():
        print(f"   {y:6s}  {m['n']:>4d}  {m['year_acc']:>8.4f}  "
              f"{m['exact_match_rate']:>6.4f}  {m['fuzzy_match_rate']:>6.4f}  {m['mean_cer']:>7.4f}")
    print("=" * 72)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dirs", nargs="*", default=None,
                   help="Cohort directories under repo root (default: TestData1 TestData2)")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N PDFs (for quick smoke-test runs)")
    p.add_argument("--year", type=str, default=None,
                   help="Only evaluate PDFs with this ground-truth year")
    p.add_argument("--exclude-year", action="append", default=[],
                   help="Skip PDFs with this ground-truth year (repeatable)")
    p.add_argument("--json", type=Path, default=None,
                   help="If set, dump full per-case results + aggregate to this path")
    p.add_argument("--quiet-classifier", action="store_true",
                   help="Drop classifier log noise (recommended)")
    args = p.parse_args()

    if args.quiet_classifier:
        import logging
        from pipeline import classify
        classify.log.setLevel(logging.WARNING)

    cases = _discover_labeled_pdfs()  # list[(Path, year)]
    if args.dirs:
        wanted = {ROOT / d for d in args.dirs}
        cases = [c for c in cases if any(p in c[0].parents for p in wanted)]
    if args.year:
        cases = [c for c in cases if c[1] == args.year]
    if args.exclude_year:
        excluded = set(args.exclude_year)
        cases = [c for c in cases if c[1] not in excluded]
    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        print("No cases found.")
        return 1

    print(f"Evaluating {len(cases)} PDF(s)...")

    results: list[CaseResult] = []
    for i, (pdf, true_year) in enumerate(cases, 1):
        # Read truth serial from the sibling JSON
        jp = pdf.with_suffix(".json")
        try:
            doc = json.load(open(jp))
        except Exception:
            true_serial = ""
        else:
            true_serial = str((doc.get("form") or {}).get("serial") or "")

        cohort = pdf.parent.name
        try:
            r = evaluate_one(pdf, cohort, true_year, true_serial)
        except Exception as e:
            print(f"  [{i}/{len(cases)}] {pdf.name}: ERROR {e}")
            continue
        results.append(r)
        if i % 25 == 0 or i == len(cases):
            print(f"  [{i}/{len(cases)}] processed")

    agg = _aggregate(results)
    _print_report(agg)

    if args.json:
        payload = {
            "aggregate": agg,
            "cases": [r.__dict__ for r in results],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
