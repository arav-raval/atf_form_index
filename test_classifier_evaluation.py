"""
End-to-end evaluation of template matching (classifier.py) against TestData*.

- Builds the template library from FormTemplates/<year>/Form.pdf (and any other images).
- Ground truth: form.year in each sibling JSON file.
- Metrics: accuracy, per-year counts, confusion pairs, UNSURE/ERROR rates.

Requires the same dependencies as classifier.py (numpy, Pillow, pdf2image, scikit-image)
and Poppler on PATH for PDF rendering (pdfinfo/pdftoppm).

``classifier.log`` lives next to ``classifier.py`` (not necessarily the process
CWD). Logging uses the module logger so it still works when pytest configures
the root logger first.

Pytest captures stdout/stderr by default: use ``pytest -s`` (or
``--capture=no``) to see classifier lines on the terminal; the log file is
written either way.

Run:
  pytest test_classifier_evaluation.py -v -s
  python test_classifier_evaluation.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Project root (directory containing this file)
ROOT = Path(__file__).resolve().parent
FORM_TEMPLATES = ROOT / "FormTemplates"
TEST_DIRS = [ROOT / "TestData1", ROOT / "TestData2"]


def _pdf_conversion_available() -> bool:
    """True if pdf2image can rasterize a template PDF (needs Poppler)."""
    sample = FORM_TEMPLATES / "1985" / "Form.pdf"
    if not sample.is_file():
        return False
    try:
        from pdf2image import convert_from_path

        convert_from_path(str(sample), dpi=72, first_page=1, last_page=1)
        return True
    except Exception:
        return False


PDF_OK = _pdf_conversion_available()


@dataclass
class EvalCase:
    json_path: Path
    pdf_path: Path
    expected_year: str


@dataclass
class EvalMetrics:
    total: int = 0
    correct: int = 0
    errors: int = 0
    unsure: int = 0
    per_year_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_year_correct: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    confusion: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.total - self.errors <= 0:
            return 0.0
        return self.correct / (self.total - self.errors)

    def summary_lines(self) -> list[str]:
        lines = [
            "",
            "=== Evaluation summary (ground truth: form.year) ===",
            f"Cases (with PDF):     {self.total}",
            f"Load/render errors:   {self.errors}",
            f"Correct (top-1 year): {self.correct}",
            f"Wrong:                {self.total - self.errors - self.correct}",
            f"UNSURE (any):         {self.unsure}",
            f"Accuracy (excl. ERR): {self.accuracy:.4f}",
            "",
            "Per expected year:",
        ]
        years = sorted(set(self.per_year_total) | set(self.per_year_correct))
        for y in years:
            t = self.per_year_total[y]
            c = self.per_year_correct[y]
            lines.append(f"  {y}: {c}/{t}")
        wrong_pairs = [(k, v) for k, v in self.confusion.items() if k[0] != k[1] and v > 0]
        if wrong_pairs:
            lines.append("")
            lines.append("Confusion (expected_year -> predicted), count:")
            for (exp, pred), n in sorted(wrong_pairs, key=lambda x: (-x[1], x[0])):
                lines.append(f"  {exp} -> {pred}: {n}")
        lines.append("")
        return lines


def discover_cases() -> list[EvalCase]:
    cases: list[EvalCase] = []
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
            cases.append(EvalCase(json_path=jp, pdf_path=pdf_path, expected_year=str(year)))
    return cases


def run_evaluation(
    templates_dir: Path | None = None,
    threshold: float | None = None,
    *,
    quiet_classifier: bool = False,
) -> tuple[Any, EvalMetrics]:
    """Build template library and classify all test PDFs.

    By default the classifier logger stays at INFO so ``classifier.log`` matches a
    normal CLI run (see ``classifier.py``).
    Pass ``quiet_classifier=True`` to set the classifier logger to WARNING only
    (e.g. to cut noise under pytest); that also silences writes to ``classifier.log``.
    """
    import classifier

    if quiet_classifier:
        classifier.log.setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    templates_dir = templates_dir or FORM_TEMPLATES
    th = threshold if threshold is not None else classifier.CONFIDENCE_THRESHOLD

    library = classifier.build_template_library(str(templates_dir))
    cases = discover_cases()
    metrics = EvalMetrics()

    for case in cases:
        metrics.total += 1
        metrics.per_year_total[case.expected_year] += 1

        result = classifier.classify(str(case.pdf_path), library, threshold=th)
        pred = result.get("label")
        status = result.get("status")
        score = result.get("score", 0.0)

        if status == "ERROR":
            metrics.errors += 1
            metrics.confusion[(case.expected_year, "ERROR")] += 1
            metrics.details.append(
                {
                    "file": str(case.pdf_path),
                    "expected": case.expected_year,
                    "predicted": None,
                    "status": status,
                    "score": score,
                }
            )
            continue

        if status == "UNSURE":
            metrics.unsure += 1

        ok = pred == case.expected_year
        if ok:
            metrics.correct += 1
            metrics.per_year_correct[case.expected_year] += 1
        metrics.confusion[(case.expected_year, pred or "None")] += 1

        metrics.details.append(
            {
                "file": str(case.pdf_path),
                "expected": case.expected_year,
                "predicted": pred,
                "status": status,
                "score": score,
                "match": ok,
            }
        )

    _log_evaluation_summary(classifier, metrics)
    return library, metrics


def _log_evaluation_summary(classifier_mod: Any, metrics: EvalMetrics) -> None:
    """Append evaluation aggregate stats to classifier.log (and stream)."""
    log = classifier_mod.log
    log.info("=" * 60)
    for line in metrics.summary_lines():
        if line.strip():
            log.info(line)
    log.info("=" * 60)


try:
    import pytest
except ImportError:
    pytest = None  # type: ignore[misc, assignment]


if pytest is not None:

    @pytest.mark.skipif(not PDF_OK, reason="Poppler/pdf2image not available (install poppler)")
    def test_evaluation_completes_with_templates():
        if not FORM_TEMPLATES.is_dir():
            pytest.skip("FormTemplates missing")
        cases = discover_cases()
        assert cases, "No test cases (JSON + PDF) found"
        _, metrics = run_evaluation()
        assert metrics.total == len(cases)
        assert metrics.total > 0


def main() -> int:
    if not PDF_OK:
        print(
            "Skipping evaluation: Poppler is not available (pdfinfo not in PATH). "
            "Install Poppler and ensure pdf2image can rasterize PDFs.",
            file=sys.stderr,
        )
        return 1
    if not FORM_TEMPLATES.is_dir():
        print("FormTemplates directory not found.", file=sys.stderr)
        return 1
    cases = discover_cases()
    if not cases:
        print("No test cases found (need TestData*/<id>.json with form.year and matching PDF).", file=sys.stderr)
        return 1

    _, metrics = run_evaluation()
    for line in metrics.summary_lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
