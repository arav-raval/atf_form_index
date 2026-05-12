"""Retrieval evaluation: simulate searching for ground-truth serials in the
index of admitted OCR'd serials, score by ``recall@K``.

This is the production-relevant metric. Given a per-row eval JSON from
``evaluate_v2``:

  1. Build an INDEX from every admitted row's OCR output:
        index[ocr_normalized] -> [(pdf, page, row, truth_serial), ...]
     (multiple rows may collide on the same OCR string; that's fine.)

  2. For every ground-truth serial in the clean PDF that the verifier
     ADMITTED, treat it as a USER QUERY. Run three rankers:

        a) "exact"        : index lookup by normalized query
        b) "fuzzy"        : exact + confusion-equivalent (existing search())
        c) "edit"         : top-K by ``fuzzy_search.score`` (new ranker)

  3. For each ranker, count: did the row whose truth == the query show up
     in the top-K candidates?

The most informative result is "edit" recall@5 — the realistic metric for
"if a user types this serial, will the right document appear in the first 5
results?"

Caveat: rows the verifier REJECTED can never be retrieved. The reported
recall is conditional on admission; we also print the absolute "found / all
positive truths" number so the verifier's recall ceiling is visible.

Usage::

    python -m pipeline.eval_retrieval /tmp/eval_v2_v6_10k.json
    python -m pipeline.eval_retrieval /tmp/eval_v2_v6_10k.json --examples 10
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline.fuzzy_search import canonical_form, rank, score


def _build_index(rows: list[dict]) -> dict[str, list[dict]]:
    """Group admitted-row OCR outputs into a {ocr_str -> [refs]} index."""
    idx: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r["verify_is_serial"]:
            continue
        ocr = r["ocr_normalized"]
        if not ocr:
            continue
        idx[ocr].append({
            "pdf": r["pdf"],
            "page": r["page"],
            "row_idx": r["row_idx"],
            "truth_serial": r["truth_serial"],
            "truth_year": r["truth_year"],
        })
    return dict(idx)


def _exact_top_k(query: str, idx: dict, k: int) -> list[str]:
    return [query] if query in idx else []


def _fuzzy_top_k(query: str, idx: dict, k: int) -> list[str]:
    """Existing fuzzy_search.search() — exact + canonical-equivalent only."""
    from pipeline.fuzzy_search import search
    return [m.indexed_serial for m in search(query, idx)][:k]


def _edit_top_k(query: str, idx: dict, k: int) -> list[str]:
    """New rank() — sorts all candidates by edit-distance score."""
    return [rc.candidate for rc in rank(query, idx, top_k=k)]


def _row_is_target(ref: dict, query_row: dict) -> bool:
    """A retrieved index entry is THE answer iff it points at the row whose
    truth_serial generated the query AND it's the same physical row."""
    return (ref["pdf"] == query_row["pdf"]
            and ref["page"] == query_row["page"]
            and ref["row_idx"] == query_row["row_idx"])


def _evaluate_recall(
    rows: list[dict], idx: dict, ks: list[int],
) -> dict[str, dict[int, float]]:
    """For each ranker × K, what fraction of admitted positives have their
    target row in the top-K results when queried by their truth serial?"""
    rankers = {
        "exact": _exact_top_k,
        "fuzzy": _fuzzy_top_k,
        "edit":  _edit_top_k,
    }
    out: dict[str, dict[int, float]] = {n: {} for n in rankers}
    counts_by_k: dict[str, dict[int, int]] = {n: {k: 0 for k in ks} for n in rankers}
    n_queries = 0

    # Only query against admitted positives — that's the universe of rows
    # whose truth serial we believe represents a real serial in the data.
    # Querying the corrupted-PDF positives is also fair (they're real
    # serials too), but their visual data was rejected by the verifier in
    # ~30% of cases; we accept that as a verifier-recall ceiling.
    queries = [
        r for r in rows
        if r["truth_label"] == "positive"
        and r["truth_serial"]
        and r["verify_is_serial"]
        and r["ocr_normalized"]
    ]

    for q in queries:
        n_queries += 1
        truth = q["truth_serial"]
        max_k = max(ks)
        for ranker_name, fn in rankers.items():
            top = fn(truth, idx, max_k)
            # Walk top in order; record the rank where target appears.
            target_rank = None
            for rank_idx, candidate in enumerate(top, 1):
                refs = idx.get(candidate, [])
                if any(_row_is_target(ref, q) for ref in refs):
                    target_rank = rank_idx
                    break
            for k in ks:
                if target_rank is not None and target_rank <= k:
                    counts_by_k[ranker_name][k] += 1

    for name in rankers:
        out[name] = {
            k: counts_by_k[name][k] / max(1, n_queries) for k in ks
        }
    out["_n_queries"] = n_queries
    return out


def _per_year_recall(
    rows: list[dict], idx: dict, k: int,
) -> dict[str, dict[str, float]]:
    """Edit-recall@k broken down by truth_year."""
    by_year: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if (r["truth_label"] == "positive" and r["truth_serial"]
                and r["verify_is_serial"] and r["ocr_normalized"]):
            by_year[r["truth_year"]].append(r)

    out: dict[str, dict[str, float]] = {}
    for year, qs in sorted(by_year.items()):
        n_year = len(qs)
        hits = 0
        for q in qs:
            top = _edit_top_k(q["truth_serial"], idx, k)
            target_rank = None
            for rank_idx, candidate in enumerate(top, 1):
                refs = idx.get(candidate, [])
                if any(_row_is_target(ref, q) for ref in refs):
                    target_rank = rank_idx
                    break
            if target_rank is not None and target_rank <= k:
                hits += 1
        out[year] = {"n": n_year, "recall": hits / max(1, n_year)}
    return out


def _show_examples(
    rows: list[dict], idx: dict, k: int, n_examples: int,
) -> None:
    """Print a few sample queries with their top-K candidates."""
    queries = [
        r for r in rows
        if r["truth_label"] == "positive"
        and r["truth_serial"]
        and r["verify_is_serial"]
        and r["ocr_normalized"]
    ]
    # Pick a mix: some where edit-rank found it past position 1, some misses.
    interesting: list[dict] = []
    for q in queries:
        top = _edit_top_k(q["truth_serial"], idx, k)
        target_rank = None
        for rank_idx, c in enumerate(top, 1):
            if any(_row_is_target(ref, q) for ref in idx.get(c, [])):
                target_rank = rank_idx
                break
        if target_rank is not None and target_rank > 1:
            interesting.append({"q": q, "top": top, "rank": target_rank})
        elif target_rank is None:
            interesting.append({"q": q, "top": top, "rank": None})
        if len(interesting) >= n_examples:
            break
    print()
    print(f" Example queries (showing edit-rank top-{k}):")
    print(" " + "-" * 78)
    for ex in interesting:
        q = ex["q"]
        marker = "MISS" if ex["rank"] is None else f"@{ex['rank']}"
        print(f"   query={q['truth_serial']:18s}  ocr_was={q['ocr_normalized']:20s}  target {marker}")
        for j, c in enumerate(ex["top"], 1):
            d = score(q["truth_serial"], c)
            is_target = any(_row_is_target(ref, q) for ref in idx.get(c, []))
            mark = "  ←" if is_target else ""
            print(f"     {j}.  {c:20s}  d={d:5.2f}{mark}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("json_path", type=Path)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--examples", type=int, default=0,
                    help="Print N example queries with their top-K candidates")
    args = ap.parse_args()

    data = json.load(open(args.json_path))
    rows = data["rows"]

    # Restrict to clean PDF for retrieval queries — the corrupted PDF has a
    # mix of real serials and PII; we want the production-style scenario
    # of "user searches for a real serial".
    clean = [r for r in rows if "no_errors" in r["pdf"]]
    err = [r for r in rows if "no_errors" not in r["pdf"] and "errors" in r["pdf"]]

    # Build the index from ALL admitted rows in the clean PDF (so we have
    # something to search against). The corrupted PDF rows are not part of
    # the searchable universe in this experiment.
    idx = _build_index(clean)
    n_index = sum(len(v) for v in idx.values())
    print("=" * 80)
    print(f" RETRIEVAL EVALUATION — {args.json_path.name}")
    print(f" Index : {len(idx)} unique OCR strings, {n_index} indexed rows  (built from clean PDF)")
    print(f" Pool of corrupted-PDF rows (excluded from index): {len(err)}")
    print("=" * 80)

    pos_in_clean = [r for r in clean if r["truth_label"] == "positive" and r["truth_serial"]]
    admitted_in_clean = [r for r in pos_in_clean if r["verify_is_serial"] and r["ocr_normalized"]]
    print()
    print(f" Universe : {len(pos_in_clean)} positive serials in clean PDF")
    print(f"            of which {len(admitted_in_clean)} were verifier-admitted ({100*len(admitted_in_clean)/max(1,len(pos_in_clean)):.1f}%)")
    print(f"            (rows the verifier rejected can NEVER be retrieved — verifier ceiling)")

    # Recall@K, conditional on admission
    rec = _evaluate_recall(clean, idx, args.ks)
    n_q = rec["_n_queries"]

    print()
    print(f" RECALL@K (conditional on admission, n={n_q})")
    print(" " + "-" * 78)
    print(f"   {'ranker':10s}  " + "  ".join(f"@{k:>2d}" for k in args.ks))
    for name in ["exact", "fuzzy", "edit"]:
        cells = "  ".join(f"{rec[name][k]:>5.3f}" for k in args.ks)
        print(f"   {name:10s}  {cells}")

    # Absolute recall (over ALL positives, including rejected)
    print()
    print(f" RECALL@K (absolute, over all {len(pos_in_clean)} positive serials)")
    print(" " + "-" * 78)
    print(f"   {'ranker':10s}  " + "  ".join(f"@{k:>2d}" for k in args.ks))
    for name in ["exact", "fuzzy", "edit"]:
        cells = "  ".join(
            f"{rec[name][k] * len(admitted_in_clean) / max(1, len(pos_in_clean)):>5.3f}"
            for k in args.ks
        )
        print(f"   {name:10s}  {cells}")

    # Per-year breakdown for edit-rank @5
    print()
    print(" PER-YEAR (edit-ranker recall@5, conditional on admission)")
    print(" " + "-" * 78)
    pyr = _per_year_recall(clean, idx, k=5)
    print(f"   {'year':5s}  {'n':>4s}  {'recall@5':>8s}")
    for year, m in pyr.items():
        print(f"   {year:5s}  {m['n']:>4d}  {m['recall']:>7.3f}")

    if args.examples > 0:
        _show_examples(clean, idx, k=5, n_examples=args.examples)

    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
