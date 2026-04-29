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
