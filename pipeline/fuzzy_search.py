"""Confusion-aware near-match search over ``search_index.json``.

When OCR produces ``JZN8O1OTRL`` and the user later searches for the true serial
``JZN8010TRL``, an exact lookup misses. This module performs a confusion-aware
search that treats common OCR-confusable characters as equivalent classes.

The index itself stays exact — no pollution from speculative variants. Fanout
happens only at query time, bounded by the size of the confusion classes.

Match quality is reported back so the caller can show the user what was matched
and how confident the match is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pipeline.recognize import normalize_serial

# Equivalence classes for character-level OCR confusions. All characters in a
# class collapse to the same canonical key for fuzzy matching. These were chosen
# from the residual-error analysis of the eval set plus well-known Tesseract /
# TrOCR confusions on alphanumeric tokens.
_CONFUSION_CLASSES: list[str] = [
    "0OQDU",   # zero-like
    "1IL",     # one-like
    "2Z",      # two-like
    "5S",      # five/S
    "6G",      # six/G
    "8B",      # eight/B
    "7T",      # seven/T (occasional)
]


def _build_canon_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for cls in _CONFUSION_CLASSES:
        canon = cls[0]
        for c in cls:
            m[c] = canon
    return m


_CANON = _build_canon_map()


def canonical_form(serial: str) -> str:
    """Project a normalized serial into its OCR-equivalence canonical form.

    ``"JZN8O1OTRL"`` and ``"JZN8010TRL"`` both project to the same canonical
    string. ``serial`` is assumed already-normalized (uppercase alphanumeric).
    """
    return "".join(_CANON.get(c, c) for c in serial)


@dataclass
class FuzzyMatch:
    indexed_serial: str        # the serial as stored in the index
    refs: list[dict]           # the index entries for that serial
    distance: int              # 0 for exact, 1 for canonical-equivalent only
    reason: str                # "exact" | "confusion_equiv"


def search(
    query: str,
    by_serial: dict[str, list[dict]],
) -> list[FuzzyMatch]:
    """Return matches for ``query`` against an index ``by_serial`` mapping.

    Order: exact matches first, then confusion-equivalent matches. Each
    indexed serial appears at most once in the result.
    """
    nq = normalize_serial(query)
    if not nq:
        return []

    matches: list[FuzzyMatch] = []
    seen: set[str] = set()

    # 1. Exact (the normal index lookup)
    refs = by_serial.get(nq)
    if refs:
        matches.append(FuzzyMatch(indexed_serial=nq, refs=refs, distance=0, reason="exact"))
        seen.add(nq)

    # 2. Confusion-equivalent: same canonical projection, same length
    canon_q = canonical_form(nq)
    for ind_serial, ind_refs in by_serial.items():
        if ind_serial in seen:
            continue
        if len(ind_serial) != len(nq):
            continue
        if canonical_form(ind_serial) == canon_q:
            matches.append(
                FuzzyMatch(
                    indexed_serial=ind_serial,
                    refs=ind_refs,
                    distance=1,
                    reason="confusion_equiv",
                )
            )
            seen.add(ind_serial)

    return matches


def all_matches_flat(
    query: str,
    by_serial: dict[str, list[dict]],
) -> Iterable[tuple[FuzzyMatch, dict]]:
    """Convenience: yield ``(match, ref)`` pairs across all matched serials."""
    for m in search(query, by_serial):
        for ref in m.refs:
            yield m, ref


# ---------------------------------------------------------------------------
# Edit-distance scoring for top-K retrieval.
#
# The strict ``search`` above only returns exact + confusion-class equivalents.
# That misses near-misses that involve insertions, deletions, or substitutions
# between non-equivalent characters (e.g. ``H7649647`` vs ``117649647`` is a
# 1-char insertion; ``45430384`` vs ``49930384`` is two ``4↔9`` substitutions
# that we don't fold into a confusion class because doing so would collide
# many legitimate distinct serials).
#
# ``score(query, candidate)`` returns a numeric distance in canonical space,
# with two soft bonuses: an OCR-confusion-aware substitution cost (so swaps
# inside a confusion class cost less), and a substring containment bonus
# (so hallucinated prefixes/suffixes don't dominate the distance).
# ---------------------------------------------------------------------------

# Empirically-observed substitution pairs from v6 eval errors (see
# ``/tmp/analyze_ocr_errors.py``). Substitutions in this set cost LESS at
# scoring time but DO NOT enter ``_CONFUSION_CLASSES`` — that would conflate
# legitimately distinct indexed serials at lookup time. This is a soft cost
# bias for ranking only.
_SOFT_CONFUSIONS: frozenset[tuple[str, str]] = frozenset(
    tuple(sorted(p)) for p in [
        ("4", "9"), ("1", "7"), ("5", "6"), ("3", "8"), ("6", "8"),
        ("3", "6"), ("1", "8"), ("0", "8"), ("4", "6"), ("1", "K"),
        ("0", "G"), ("M", "W"), ("U", "V"), ("7", "J"), ("C", "G"),
    ]
)
_SOFT_SUB_COST = 0.5  # vs 1.0 for arbitrary substitutions


def _sub_cost(a: str, b: str) -> float:
    """Cost of substituting ``a`` for ``b`` in the canonical form."""
    if a == b:
        return 0.0
    if tuple(sorted((a, b))) in _SOFT_CONFUSIONS:
        return _SOFT_SUB_COST
    return 1.0


def _weighted_levenshtein(a: str, b: str) -> float:
    """Levenshtein distance with soft costs for OCR-confusable substitutions.

    Inserts and deletes still cost 1.0 each; substitutions of equal characters
    cost 0; substitutions in ``_SOFT_CONFUSIONS`` cost ``_SOFT_SUB_COST``;
    everything else costs 1.0.
    """
    n, m = len(a), len(b)
    if n == 0: return float(m)
    if m == 0: return float(n)
    prev = [float(j) for j in range(m + 1)]
    cur = [0.0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = float(i)
        for j in range(1, m + 1):
            cur[j] = min(
                prev[j] + 1.0,                  # delete
                cur[j - 1] + 1.0,               # insert
                prev[j - 1] + _sub_cost(a[i - 1], b[j - 1]),  # substitute
            )
        prev, cur = cur, prev
    return prev[m]


def score(query: str, candidate: str) -> float:
    """Return a non-negative distance from ``query`` to ``candidate``.

    Both strings are projected to ``canonical_form`` first (folding hard
    confusion classes like ``0OQDU``). Then we compute a soft-weighted
    Levenshtein distance on the canonical projections, with a substring
    containment bonus to handle hallucinated prefixes / suffixes.

    Lower is better. ``score(s, s)`` is 0.0; arbitrary unrelated pairs
    typically score >= max(len(query), len(candidate)).
    """
    if not query or not candidate:
        return float(max(len(query), len(candidate)))
    cq = canonical_form(query)
    cc = canonical_form(candidate)
    base = _weighted_levenshtein(cq, cc)
    # Substring bonus: if the shorter is a substring of the longer (in
    # canonical form), distance shouldn't exceed the absolute length diff.
    # Catches "OADL89612570" containing the truth "89612570".
    short, long_ = (cq, cc) if len(cq) <= len(cc) else (cc, cq)
    if len(short) >= 4 and short in long_:
        # Bonus: at most the length difference, with a small fixed bonus.
        base = min(base, float(len(long_) - len(short)) + 0.25)
    return base


@dataclass
class RankedCandidate:
    """A scored candidate from ``rank()``."""
    candidate: str          # the indexed string
    distance: float         # score(query, candidate); lower is better
    refs: list[dict]        # index entries for this candidate
    is_exact: bool          # True iff the indexed string == normalized query
    is_canonical_equiv: bool  # True iff canonical_form matches and not exact


def rank(
    query: str,
    by_serial: dict[str, list[dict]],
    top_k: int = 5,
    max_distance: float | None = None,
) -> list[RankedCandidate]:
    """Return the top-K candidates for ``query`` ranked by ``score``.

    The index format mirrors :func:`search` — a dict mapping
    indexed-serial-string to a list of reference entries. We score every
    indexed serial and return the K with the lowest distance.

    For large indexes this is O(|index| * max_serial_len^2). Fine for
    tens of thousands of serials; revisit if we get to millions.

    ``max_distance``: if set, only return candidates with distance below
    this threshold. Useful to prune nonsense long-tail matches.
    """
    nq = normalize_serial(query)
    if not nq:
        return []
    canon_q = canonical_form(nq)

    scored: list[RankedCandidate] = []
    for ind_serial, refs in by_serial.items():
        d = score(nq, ind_serial)
        if max_distance is not None and d > max_distance:
            continue
        is_exact = (ind_serial == nq)
        is_canon_equiv = (not is_exact) and (canonical_form(ind_serial) == canon_q)
        scored.append(RankedCandidate(
            candidate=ind_serial,
            distance=d,
            refs=refs,
            is_exact=is_exact,
            is_canonical_equiv=is_canon_equiv,
        ))
    scored.sort(key=lambda rc: (rc.distance, not rc.is_exact, not rc.is_canonical_equiv, rc.candidate))
    return scored[:top_k]
