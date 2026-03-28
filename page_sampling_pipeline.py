"""
Pipeline step 1: page-level random sampling + single-page classification.

Collects labeled test PDFs, expands to (file, page) pairs, shuffles, runs
:classifier.classify_single_page on each, optionally filters to “relevant” hits,
and reports **page-level** metrics so you can track recall on key pages vs
false positives on other pages.

**Evaluation idea (ground truth from your existing layout)**

For each test PDF you know ``form.year`` = ``Y``. The “canonical” form page for
that revision is ``p = firearm_rows.page`` in ``FormTemplates/Y/form_config.json``
(0-based), same index used to build templates.

- **Key page** = ``(pdf, p)`` for that document.
- **Off page** = any ``(pdf, q)`` with ``q != p``.

Metrics:

1. **key_page_recall** — Among key-page samples, fraction with ``pred == Y`` and
   ``status == OK``. Measures whether you *detect the right year on the right
   page* when that page is shown.

2. **off_page_fp_same_year** — Among off-page samples, fraction with
   ``status == OK`` and ``pred == Y``. These are the worst false positives: the
   model is confident the page matches the *correct* year even though it is not
   the designated form page (should be rare or you will duplicate/downstream noise).

3. **off_page_fp_any_ok** — Among off-page samples, fraction with ``status == OK``
   (any label). Any confident hit on a non-key page is a potential pipeline false
   start.

Tune ``CONFIDENCE_THRESHOLD`` in ``classifier.py`` (or pass ``threshold=``) to
trade off UNSURE vs OK and move these rates.

Later you can feed filtered ``PageMatch`` rows into rendering (path + page index
+ predicted label).

Run::

    python page_sampling_pipeline.py --seed 42 --max-samples 500
    python page_sampling_pipeline.py --help

Logs go to ``page_sampling_pipeline.log`` next to this file and to stderr (same
handler pattern as ``classifier.py`` — dedicated logger, ``propagate=False``).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parent
FORM_TEMPLATES = ROOT / "FormTemplates"
TEST_DIRS = [ROOT / "TestData1", ROOT / "TestData2"]
LOG_FILE = str(ROOT / "page_sampling_pipeline.log")

_LOG_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _configure_pipeline_logger() -> logging.Logger:
    """Separate from ``classifier`` — own file + stream, no propagation to root."""
    lg = logging.getLogger("page_sampling_pipeline")
    if lg.handlers:
        return lg
    lg.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(_LOG_FMT)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(_LOG_FMT)
    lg.addHandler(fh)
    lg.addHandler(sh)
    lg.propagate = False
    return lg


log = _configure_pipeline_logger()

FilterMode = Literal["none", "ok", "ok_correct"]


@dataclass(frozen=True)
class PageSample:
    """One page position in a test PDF with labels for evaluation."""

    pdf_path: Path
    page_0based: int
    true_year: str
    key_page_0based: int

    @property
    def is_key_page(self) -> bool:
        return self.page_0based == self.key_page_0based


@dataclass
class PageMatch:
    """One classification result; suitable for logging / future rendering."""

    pdf_path: Path
    page_0based: int
    true_year: str
    predicted_label: str | None
    score: float
    status: str
    is_key_page: bool
    raw: dict[str, Any]


def _discover_labeled_pdfs() -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    for td in TEST_DIRS:
        if not td.is_dir():
            continue
        for jp in sorted(td.glob("*.json")):
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
            form = data.get("form") or {}
            year = form.get("year")
            if not year:
                continue
            pdf_name = form.get("pdf_file") or f"{jp.stem}.pdf"
            pdf_path = jp.parent / pdf_name
            if not pdf_path.is_file():
                continue
            if pdf_path.suffix.lower() != ".pdf":
                continue
            rows.append((pdf_path.resolve(), str(year)))
    return rows


def pdf_page_count(pdf_path: Path) -> int:
    from pdf2image import pdfinfo_from_path

    info = pdfinfo_from_path(str(pdf_path))
    return int(info["Pages"])


def build_page_inventory() -> list[PageSample]:
    """All (pdf, page) pairs for the labeled test PDF set."""
    import classifier

    samples: list[PageSample] = []
    for pdf_path, year in _discover_labeled_pdfs():
        key_p = classifier.template_firearm_page(str(FORM_TEMPLATES), year)
        try:
            n = pdf_page_count(pdf_path)
        except Exception:
            continue
        if key_p >= n:
            log.warning(
                "firearm_rows.page (%s) >= page count (%s) for %s; "
                "no page in this PDF is labeled as the key page.",
                key_p,
                n,
                pdf_path.name,
            )
        for p in range(n):
            samples.append(
                PageSample(
                    pdf_path=pdf_path,
                    page_0based=p,
                    true_year=year,
                    key_page_0based=key_p,
                )
            )
    return samples


def shuffle_and_truncate(
    samples: list[PageSample],
    rng: random.Random,
    max_samples: int | None,
) -> list[PageSample]:
    out = samples.copy()
    rng.shuffle(out)
    if max_samples is not None and max_samples < len(out):
        out = out[:max_samples]
    return out


def run_page_sampling(
    library: dict,
    samples: list[PageSample],
    *,
    threshold: float,
    silent: bool = True,
) -> list[tuple[PageSample, dict[str, Any]]]:
    """Run ``classify_single_page`` for each sample in order."""
    import classifier

    out: list[tuple[PageSample, dict[str, Any]]] = []
    for s in samples:
        r = classifier.classify_single_page(
            str(s.pdf_path),
            s.page_0based,
            library,
            threshold=threshold,
            silent=silent,
        )
        out.append((s, r))
    return out


def filter_matches(
    pairs: list[tuple[PageSample, dict[str, Any]]],
    mode: FilterMode,
) -> list[PageMatch]:
    """Keep rows according to ``mode`` (for downstream use / export)."""
    matches: list[PageMatch] = []
    for s, r in pairs:
        pred = r.get("label")
        score = float(r.get("score") or 0.0)
        st = str(r.get("status") or "ERROR")
        if mode == "none":
            pass
        elif mode == "ok" and st != "OK":
            continue
        elif mode == "ok_correct" and (st != "OK" or pred != s.true_year):
            continue
        matches.append(
            PageMatch(
                pdf_path=s.pdf_path,
                page_0based=s.page_0based,
                true_year=s.true_year,
                predicted_label=pred,
                score=score,
                status=st,
                is_key_page=s.is_key_page,
                raw=r,
            )
        )
    return matches


def evaluate_page_samples(
    pairs: list[tuple[PageSample, dict[str, Any]]],
) -> dict[str, Any]:
    """
    Aggregate metrics over *this run’s* sampled pairs (not necessarily every
    page in the corpus unless you sampled all).
    """
    key_rows = [(s, r) for s, r in pairs if s.is_key_page]
    off_rows = [(s, r) for s, r in pairs if not s.is_key_page]

    def ok_year(sr: tuple[PageSample, dict[str, Any]]) -> bool:
        s, r = sr
        return r.get("status") == "OK" and r.get("label") == s.true_year

    key_ok = sum(1 for sr in key_rows if ok_year(sr))
    key_recall = key_ok / len(key_rows) if key_rows else 0.0

    off_fp_same = sum(
        1
        for s, r in off_rows
        if r.get("status") == "OK" and r.get("label") == s.true_year
    )
    off_fp_any = sum(1 for s, r in off_rows if r.get("status") == "OK")
    off_rate_same = off_fp_same / len(off_rows) if off_rows else 0.0
    off_rate_any = off_fp_any / len(off_rows) if off_rows else 0.0

    return {
        "sampled_pairs": len(pairs),
        "key_page_samples": len(key_rows),
        "off_page_samples": len(off_rows),
        "key_page_hits_ok_correct": key_ok,
        "key_page_recall": round(key_recall, 4),
        "off_page_fp_same_year_ok": off_fp_same,
        "off_page_fp_same_year_rate": round(off_rate_same, 4),
        "off_page_fp_any_ok": off_fp_any,
        "off_page_fp_any_ok_rate": round(off_rate_any, 4),
    }


def format_metrics(m: dict[str, Any]) -> str:
    lines = [
        "",
        "=== Page-sample evaluation (see module docstring for definitions) ===",
        f"Sampled (page, file) pairs:     {m['sampled_pairs']}",
        f"  of which key-page samples:     {m['key_page_samples']}",
        f"  of which off-page samples:     {m['off_page_samples']}",
        f"Key-page OK + correct year:      {m['key_page_hits_ok_correct']}  "
        f"(recall {m['key_page_recall']})",
        f"Off-page FP (OK + pred=true yr): {m['off_page_fp_same_year_ok']}  "
        f"(rate {m['off_page_fp_same_year_rate']})",
        f"Off-page FP (any OK):            {m['off_page_fp_any_ok']}  "
        f"(rate {m['off_page_fp_any_ok_rate']})",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    import classifier

    p = argparse.ArgumentParser(description="Random page sampling + single-page classify")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for shuffle")
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap number of (pdf, page) pairs after shuffle (default: all)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Confidence threshold (default: {classifier.CONFIDENCE_THRESHOLD})",
    )
    p.add_argument(
        "--filter",
        choices=("none", "ok", "ok_correct"),
        default="ok_correct",
        help="Which PageMatch rows to log as 'relevant' (default: ok_correct)",
    )
    p.add_argument(
        "--verbose-classifier",
        action="store_true",
        help="Log each classify_single_page to classifier.log at INFO",
    )
    args = p.parse_args()

    th = args.threshold if args.threshold is not None else classifier.CONFIDENCE_THRESHOLD

    log.info("Starting page sampling pipeline (log file: %s)", LOG_FILE)

    inv = build_page_inventory()
    if not inv:
        log.error("No labeled test PDFs found.")
        return 1

    log.info("Page inventory: %s (pdf, page) pairs before shuffle/truncate", len(inv))

    rng = random.Random(args.seed)
    samples = shuffle_and_truncate(inv, rng, args.max_samples)
    log.info(
        "After shuffle (seed=%s) and max_samples=%s: %s pairs to classify",
        args.seed,
        args.max_samples,
        len(samples),
    )

    library = classifier.build_template_library(str(FORM_TEMPLATES))
    pairs = run_page_sampling(
        library, samples, threshold=th, silent=not args.verbose_classifier
    )

    metrics = evaluate_page_samples(pairs)
    log.info("%s", format_metrics(metrics).rstrip())

    mode: FilterMode = args.filter  # type: ignore[assignment]
    kept = filter_matches(pairs, mode)
    log.info("Filtered matches (%s): %s", mode, len(kept))
    for m in kept[:50]:
        log.info(
            "  %s  page=%s  true=%s  pred=%s  score=%.4f  key_page=%s",
            m.pdf_path.name,
            m.page_0based,
            m.true_year,
            m.predicted_label,
            m.score,
            m.is_key_page,
        )
    if len(kept) > 50:
        log.info("  ... and %s more", len(kept) - 50)

    log.info("Pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
