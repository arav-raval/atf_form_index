"""
Offline processing: classify PDFs, extract serial (OCR), write ``processing`` into
sidecar JSON, and maintain ``search_index.json`` for the web app.

**Document-level processing** (recommended): full PDF → :func:`classify` → serial
crop/OCR on the predicted year’s layout.

**Page stream** (incremental): queue of ``(pdf_path, page_0based)`` →
:func:`classify_single_page` + serial OCR on that page only (useful when
pages arrive one at a time).

Environment / layout:
  - ``FormTemplates/<year>/form_config.json`` — field geometry for serial OCR
  - Sidecar JSON next to each PDF (updated in place)
  - :mod:`serial_extract` for OCR (optional ``pytesseract``)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parent
FORM_TEMPLATES = ROOT / "FormTemplates"
DEFAULT_INDEX = ROOT / "search_index.json"

log = logging.getLogger("document_pipeline")
if not log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def load_search_index(path: Path = DEFAULT_INDEX) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 2, "by_serial": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_search_index(idx: dict[str, Any], path: Path = DEFAULT_INDEX) -> None:
    save_json(path, idx)


def _ground_truth_serial(doc: dict[str, Any]) -> str | None:
    form = doc.get("form") or {}
    s = form.get("serial")
    if s is None:
        return None
    return str(s)


def _merge_processing(
    doc: dict[str, Any],
    processing: dict[str, Any],
) -> dict[str, Any]:
    out = dict(doc)
    out["processing"] = {**(doc.get("processing") or {}), **processing}
    return out


def _index_upsert(
    idx: dict[str, Any],
    serial_norm: str,
    pdf_path: str,
    json_path: str,
    predicted_year: str,
    ground_truth: str | None,
) -> None:
    """Append (or replace) a document reference under ``by_serial[serial_norm]``.

    The index is list-valued: multiple PDFs can share a serial, and one PDF can
    contribute multiple serials. Replacement is keyed on ``(pdf_path, serial)``
    so re-ingesting a PDF updates existing entries instead of duplicating them.
    """
    if not serial_norm:
        return
    bys = idx.setdefault("by_serial", {})
    entries = bys.setdefault(serial_norm, [])
    ref = {
        "pdf_path": pdf_path,
        "json_path": json_path,
        "predicted_year": predicted_year,
        "ground_truth_serial": ground_truth,
    }
    for i, e in enumerate(entries):
        if e.get("pdf_path") == pdf_path:
            entries[i] = ref
            return
    entries.append(ref)


def _index_remove_pdf(idx: dict[str, Any], pdf_path: str) -> None:
    """Drop every reference to ``pdf_path`` across all serials."""
    bys = idx.get("by_serial") or {}
    for serial, entries in list(bys.items()):
        kept = [e for e in entries if e.get("pdf_path") != pdf_path]
        if kept:
            bys[serial] = kept
        else:
            del bys[serial]


@dataclass
class DocumentProcessResult:
    pdf_path: Path
    json_path: Path | None
    predicted_year: str | None
    predicted_serials: list[str]
    classifier_score: float
    classifier_status: str
    pipeline_status: str
    ground_truth_serial: str | None
    error: str | None = None


def process_pdf_document(
    pdf_path: str | Path,
    *,
    json_path: str | Path | None = None,
    templates_dir: Path = FORM_TEMPLATES,
    index_path: Path = DEFAULT_INDEX,
    threshold: float | None = None,
    update_index: bool = True,
) -> DocumentProcessResult:
    """
    Full-document pipeline: delegates to :mod:`pipeline.orchestrator`, then
    persists the result to a sidecar JSON and updates the search index.

    If ``json_path`` is omitted, uses ``<pdf_stem>.json`` beside the PDF.
    """
    from pipeline import orchestrator
    from pipeline.recognize import normalize_serial

    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.is_file():
        return DocumentProcessResult(
            pdf_path=pdf_path,
            json_path=None,
            predicted_year=None,
            predicted_serials=[],
            classifier_score=0.0,
            classifier_status="ERROR",
            pipeline_status="ERROR",
            ground_truth_serial=None,
            error="PDF not found",
        )

    jp = Path(json_path).resolve() if json_path else pdf_path.with_suffix(".json")

    gt: str | None = None
    doc: dict[str, Any] = {}
    if jp.is_file():
        try:
            doc = load_json(jp)
            gt = _ground_truth_serial(doc)
        except Exception as e:
            log.warning("Could not read JSON %s: %s", jp, e)

    pr = orchestrator.run(pdf_path, templates_dir=templates_dir, threshold=threshold)

    predicted_norms = list(pr.admitted_serials)

    rows_payload = [
        {
            "row_index": r.row_index,
            "verify_status": r.verify_status,
            "verify_confidence": r.verify_confidence,
            "ocr_method": r.ocr_method,
            "serial_raw": r.serial_raw,
            "serial": r.serial_normalized,
            "looks_serial": r.looks_serial,
            "admitted": r.admitted,
        }
        for r in pr.rows
    ]

    proc = {
        "predicted_year": pr.predicted_year,
        "predicted_serials": predicted_norms,
        "rows": rows_payload,
        "classifier_score": round(pr.classifier_score, 4),
        "classifier_status": pr.classifier_status,
        "pipeline_status": pr.status,
        "processed_at": _utc_now_iso(),
    }
    if gt is not None:
        gt_norm = normalize_serial(gt)
        proc["ground_truth_serial"] = gt
        proc["serial_match"] = gt_norm in predicted_norms if predicted_norms else None

    if not doc:
        doc = {"form": {"pdf_file": pdf_path.name}}
    merged = _merge_processing(doc, proc)
    if not jp.parent.is_dir():
        jp.parent.mkdir(parents=True, exist_ok=True)
    save_json(jp, merged)
    log.info(
        "Processed %s → year=%s serials=%s (status=%s)",
        pdf_path.name,
        pr.predicted_year,
        predicted_norms or "(none)",
        pr.status,
    )

    if update_index:
        idx = load_search_index(index_path)
        # Re-ingest should not leave stale entries from a prior run.
        _index_remove_pdf(idx, str(pdf_path))
        for serial_norm in predicted_norms:
            _index_upsert(
                idx,
                serial_norm,
                str(pdf_path),
                str(jp),
                str(pr.predicted_year or ""),
                gt,
            )
        save_search_index(idx, index_path)

    return DocumentProcessResult(
        pdf_path=pdf_path,
        json_path=jp,
        predicted_year=pr.predicted_year,
        predicted_serials=predicted_norms,
        classifier_score=pr.classifier_score,
        classifier_status=pr.classifier_status,
        pipeline_status=pr.status,
        ground_truth_serial=gt,
        error=pr.error,
    )


@dataclass
class PageStreamItem:
    pdf_path: Path
    page_0based: int
    doc_id: str | None = None


class PageStreamProcessor:
    """
    Queue of pages; call :meth:`process_one` repeatedly (or :meth:`drain`).

    Each item runs single-page classification + serial OCR on **that** page using
    the **predicted** year’s template (serial field may not align if the page is
    wrong — use full :func:`process_pdf_document` when the file is available).
    """

    def __init__(
        self,
        templates_dir: Path = FORM_TEMPLATES,
        index_path: Path = DEFAULT_INDEX,
        threshold: float | None = None,
        on_result: Callable[[PageStreamItem, dict[str, Any]], None] | None = None,
    ) -> None:
        self.templates_dir = templates_dir
        self.index_path = index_path
        self.threshold = threshold
        self.on_result = on_result
        self._q: deque[PageStreamItem] = deque()
        from pipeline import classify as _classify

        self._classifier = _classify
        self._library: dict | None = None

    def _library_lazy(self) -> dict:
        if self._library is None:
            self._library = self._classifier.build_template_library(str(self.templates_dir))
        return self._library

    def submit(self, pdf_path: str | Path, page_0based: int, doc_id: str | None = None) -> None:
        self._q.append(PageStreamItem(Path(pdf_path).resolve(), page_0based, doc_id))

    def submit_pdf_all_pages(self, pdf_path: str | Path, doc_id: str | None = None) -> None:
        """Enqueue every page of a PDF (uses pdfinfo)."""
        from pdf2image import pdfinfo_from_path

        p = Path(pdf_path).resolve()
        n = int(pdfinfo_from_path(str(p))["Pages"])
        for i in range(n):
            self.submit(p, i, doc_id)

    def __len__(self) -> int:
        return len(self._q)

    def process_one(self) -> tuple[PageStreamItem, dict[str, Any]] | None:
        import serial_extract

        if not self._q:
            return None
        item = self._q.popleft()
        lib = self._library_lazy()
        th = self.threshold if self.threshold is not None else self._classifier.CONFIDENCE_THRESHOLD
        r = self._classifier.classify_single_page(
            str(item.pdf_path),
            item.page_0based,
            lib,
            threshold=th,
            silent=True,
        )
        pred_year = r.get("label")
        status = str(r.get("status") or "ERROR")
        serial_norm = ""
        serial_raw = ""
        serial_method = "skipped"
        if pred_year and status != "ERROR":
            ex = serial_extract.extract_serial_from_pdf(
                item.pdf_path, pred_year, self.templates_dir, row_index=0
            )
            serial_raw = str(ex.get("text") or "")
            serial_norm = str(ex.get("normalized") or "")
            serial_method = str(ex.get("method") or "")

        out = {
            "classify": r,
            "predicted_year": pred_year,
            "predicted_serial": serial_norm,
            "predicted_serial_raw": serial_raw,
            "serial_extraction_method": serial_method,
            "page_0based": item.page_0based,
            "doc_id": item.doc_id,
            "processed_at": _utc_now_iso(),
        }
        if self.on_result:
            self.on_result(item, out)
        return item, out

    def drain(self) -> list[tuple[PageStreamItem, dict[str, Any]]]:
        done = []
        while self._q:
            x = self.process_one()
            if x:
                done.append(x)
        return done


def iter_pdf_files(directories: list[Path]) -> Iterator[Path]:
    for d in directories:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.pdf")):
            yield p.resolve()


def ingest_directories(
    directories: list[Path],
    *,
    templates_dir: Path = FORM_TEMPLATES,
    index_path: Path = DEFAULT_INDEX,
    threshold: float | None = None,
) -> list[DocumentProcessResult]:
    """Batch-process every ``*.pdf`` under the given directories."""
    results: list[DocumentProcessResult] = []
    for pdf in iter_pdf_files(directories):
        results.append(
            process_pdf_document(
                pdf,
                templates_dir=templates_dir,
                index_path=index_path,
                threshold=threshold,
            )
        )
    return results


def rebuild_index_from_json(
    directories: list[Path],
    index_path: Path = DEFAULT_INDEX,
) -> int:
    """Scan sidecar JSONs and rebuild ``search_index`` from ``processing.predicted_serials``."""
    idx: dict[str, Any] = {"version": 2, "by_serial": {}}
    n = 0
    for d in directories:
        if not d.is_dir():
            continue
        for jp in d.glob("*.json"):
            try:
                doc = load_json(jp)
            except Exception:
                continue
            proc = doc.get("processing") or {}
            form = doc.get("form") or {}
            pdf_name = form.get("pdf_file") or f"{jp.stem}.pdf"
            pdf_path = (jp.parent / pdf_name).resolve()
            predicted_year = str(proc.get("predicted_year") or "")
            gt = form.get("serial")

            for sn in proc.get("predicted_serials") or []:
                if not sn:
                    continue
                _index_upsert(
                    idx,
                    str(sn),
                    str(pdf_path),
                    str(jp.resolve()),
                    predicted_year,
                    gt,
                )
                n += 1
    save_search_index(idx, index_path)
    log.info("Rebuilt search index: %s entries → %s", n, index_path)
    return n


def main() -> int:
    p = argparse.ArgumentParser(description="Offline document processing pipeline")
    p.add_argument(
        "command",
        choices=("ingest", "rebuild-index", "stream-demo"),
        help="ingest: process PDFs; rebuild-index: scan JSON; stream-demo: queue pages",
    )
    p.add_argument(
        "--dirs",
        nargs="*",
        default=["TestData1", "TestData2"],
        help="Directories to scan (default: TestData1 TestData2)",
    )
    args = p.parse_args()

    dirs = [ROOT / d for d in args.dirs]

    if args.command == "ingest":
        ingest_directories(dirs)
        return 0
    if args.command == "rebuild-index":
        rebuild_index_from_json(dirs)
        return 0
    if args.command == "stream-demo":
        ps = PageStreamProcessor()
        for pdf in list(iter_pdf_files(dirs))[:3]:
            ps.submit_pdf_all_pages(pdf, doc_id=pdf.stem)
        log.info("Stream queue: %s pages", len(ps))
        while ps:
            one = ps.process_one()
            if not one:
                break
            item, out = one
            log.info(
                "  %s page=%s year=%s serial=%s",
                item.pdf_path.name,
                item.page_0based,
                out.get("predicted_year"),
                out.get("predicted_serial") or "-",
            )
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
