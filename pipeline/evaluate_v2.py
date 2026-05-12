"""End-to-end evaluation against the Version 2 Datasets.

Two PDFs evaluated by default::

    Version 2 Datasets/v2dataset_no_errors_2.pdf  — clean serials
    Version 2 Datasets/v2dataset_errors_2.pdf     — ~80% corruptions

For each form-bearing page the harness runs the single-page pipeline:

    rasterize page  →  classify (single page)
                   →  localize (annotated serial_block)
                   →  split into rows (equal-height geometric)
                   →  verify (ML or heuristic)
                   →  OCR (TrOCR) on admitted rows

Per-row outcomes are compared against JSON ground truth. Output: prints a
short summary and (optionally) dumps a per-row JSON for ``report_compliance``.

Usage::

    python -m pipeline.evaluate_v2
    python -m pipeline.evaluate_v2 --limit-pages 25
    python -m pipeline.evaluate_v2 --json /tmp/eval_v2.json
    python -m pipeline.evaluate_v2 --pdfs no_errors_2  # one PDF only
    python -m pipeline.evaluate_v2 --pdfs no_errors_1 errors_1   # train PDFs
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from pdf2image import convert_from_path

from pipeline import classify, localize, preprocess, recognize, verify
from pipeline.recognize import normalize_serial

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "FormTemplates"
V2_DIR = ROOT / "Version 2 Datasets"

# Each entry: short_name -> (directory, file_stem). Stems are looked up as
# ``<dir>/<stem>.pdf`` and ``<dir>/<stem>.json``.
PDF_STEMS: dict[str, tuple[Path, str]] = {
    # v2 _1 = training data; _2 = held-out test
    "v2_no_errors_1": (V2_DIR, "v2dataset_no_errors_1"),
    "v2_errors_1":    (V2_DIR, "v2dataset_errors_1"),
    "v2_no_errors_2": (V2_DIR, "v2dataset_no_errors_2"),
    "v2_errors_2":    (V2_DIR, "v2dataset_errors_2"),

    # Additional held-out datasets (never used in training):
    # SerialSets — 10 PDFs, ~1k form pages each, clean serials only
    **{f"serial_only_{i}": (ROOT / "SerialSets", f"serial_only_{i}")
       for i in range(1, 11)},
    # Serial Error Pages — 2 PDFs with corruption labels
    "serial_error_1": (ROOT / "Serial Error Pages", "serial_only_error_1"),
    "serial_error_2": (ROOT / "Serial Error Pages", "serial_only_error_2"),
    # TestSerialSet — 1 PDF, 500 clean form pages
    "serial_test_500": (ROOT / "TestSerialSet", "serial_only_500"),
    # Datasets/ — small additional examples
    "ds_no_errors_1": (ROOT / "Datasets", "dataset_no_errors_1"),
    "ds_errors_1":    (ROOT / "Datasets", "dataset_errors_1"),

    # MasterTesting/ — comprehensive test set spanning real & synthetic, all years
    "mt_1985_complete": (ROOT / "MasterTesting", "1985_complete_test"),
    "mt_1998_complete": (ROOT / "MasterTesting", "1998_complete_test"),
    "mt_2001_complete": (ROOT / "MasterTesting", "2001_complete_test"),
    "mt_2005_complete": (ROOT / "MasterTesting", "2005_complete_test"),
    "mt_2007_complete": (ROOT / "MasterTesting", "2007_complete_test"),
    "mt_2008_complete": (ROOT / "MasterTesting", "2008_complete_test"),
    "mt_2012_complete": (ROOT / "MasterTesting", "2012_complete_test"),
    "mt_2016_complete": (ROOT / "MasterTesting", "2016_complete_test"),
    "mt_2020_complete": (ROOT / "MasterTesting", "2020_complete_test"),
    "mt_2020_cont":     (ROOT / "MasterTesting", "2020_cont_test"),
    "mt_2022_complete": (ROOT / "MasterTesting", "2022_complete_test"),
    "mt_2022_cont":     (ROOT / "MasterTesting", "2022_cont_test"),
    "mt_2023_complete": (ROOT / "MasterTesting", "2023_complete_test"),
    "mt_2023_cont":     (ROOT / "MasterTesting", "2023_cont_test"),
    "mt_CR1":  (ROOT / "MasterTesting", "CR1"),
    "mt_CR2":  (ROOT / "MasterTesting", "CR2"),
    "mt_CR7":  (ROOT / "MasterTesting", "CR7"),
    "mt_CR9":  (ROOT / "MasterTesting", "CR9"),
    "mt_CR11": (ROOT / "MasterTesting", "CR11"),
    "mt_CR14": (ROOT / "MasterTesting", "CR14"),
    "mt_CR17": (ROOT / "MasterTesting", "CR17"),
    "mt_CR18": (ROOT / "MasterTesting", "CR18"),
    "mt_CR21": (ROOT / "MasterTesting", "CR21"),
    "mt_CR24": (ROOT / "MasterTesting", "CR24"),
    "mt_serial_only_7": (ROOT / "MasterTesting", "serial_only_7"),
    "mt_serial_only_8": (ROOT / "MasterTesting", "serial_only_8"),
    "mt_serial_only_9": (ROOT / "MasterTesting", "serial_only_9"),
    "mt_v2_errors_1":    (ROOT / "MasterTesting", "v2dataset_errors_1"),
    "mt_v2_no_errors_1": (ROOT / "MasterTesting", "v2dataset_no_errors_1"),
}

# MasterTesting preset: every PDF in MasterTesting/. Categorized by document
# type for the master_report's "By Document Type" view.
MASTERTESTING_PDFS = [k for k in PDF_STEMS if k.startswith("mt_")]

# Held-out subset: MasterTesting MINUS the v2_1 PDFs (which are byte-identical
# to Version 2 Datasets/v2dataset_*_1 used in verifier training). Use this
# preset for fair evaluation.
MASTERTESTING_HELDOUT_PDFS = [
    k for k in MASTERTESTING_PDFS
    if k not in ("mt_v2_errors_1", "mt_v2_no_errors_1")
]
MASTERTESTING_DOC_TYPE = {
    # short_name -> document type label
    **{f"mt_{y}_complete": "complete_test" for y in
       ["1985","1998","2001","2005","2007","2008","2012","2016","2020","2022","2023"]},
    **{f"mt_{y}_cont": "cont_test" for y in ["2020","2022","2023"]},
    **{f"mt_CR{n}": "case_report" for n in [1,2,7,9,11,14,17,18,21,24]},
    **{f"mt_serial_only_{n}": "serial_only" for n in [7,8,9]},
    "mt_v2_errors_1": "v2dataset_corrupted",
    "mt_v2_no_errors_1": "v2dataset_clean",
}

# Default eval set: held-out v2 _2 (the original compliance-focused split).
DEFAULT_PDFS = ["v2_no_errors_2", "v2_errors_2"]

# Compliance preset: every PDF with corruption labels (PII leak rate
# is meaningful only on these). Held-out only — excludes the v2 _1 train set.
COMPLIANCE_PDFS = ["v2_no_errors_2", "v2_errors_2",
                   "serial_error_1", "serial_error_2", "ds_errors_1"]

# Full preset: every PDF in PDF_STEMS except training data.
FULL_HELDOUT_PDFS = COMPLIANCE_PDFS + [
    f"serial_only_{i}" for i in range(1, 11)
] + ["serial_test_500", "ds_no_errors_1"]

# Same labeling rule as ``verifier_data_v2``: serial_overflow is positive
# (the serial IS in the box), other corruption types are negative.
_CORRUPTION_AS_POSITIVE = {"serial_overflow"}
_CORRUPTION_AS_NEGATIVE = {"pii_in_serial", "name_in_serial", "field_swap"}


def _classify_firearm(firearm: dict[str, Any]) -> str:
    corr = firearm.get("corruption")
    if not corr:
        return "positive"
    ctype = corr.get("type")
    if ctype in _CORRUPTION_AS_POSITIVE:
        return "positive"
    if ctype in _CORRUPTION_AS_NEGATIVE:
        return "negative"
    if ctype == "overflow_into_serial":
        if corr.get("serial_also_written") is False:
            return "negative"
        return "skip"
    return "skip"


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


def _row_crops(
    page_img: Image.Image, cfg: dict[str, Any], expected_rows: int
) -> list[Image.Image]:
    page_size = cfg.get("page_size") or [612, 792]
    page_w_pt, page_h_pt = float(page_size[0]), float(page_size[1])
    iw, ih = page_img.size
    box_pts = localize._serial_block_pts(cfg)
    L, T, R, B = localize._pts_to_pixels(box_pts, iw, ih, page_w_pt, page_h_pt)
    block = page_img.crop((L, T, R, B))
    bw, bh = block.size
    crops: list[Image.Image] = []
    for i in range(expected_rows):
        top = int(i * bh / expected_rows)
        bot = int((i + 1) * bh / expected_rows) if i + 1 < expected_rows else bh
        crops.append(block.crop((0, top, bw, bot)))
    return crops


@dataclass
class RowEval:
    pdf: str
    page: int
    row_idx: int
    truth_year: str
    pred_year: str | None
    classifier_status: str
    classifier_score: float
    truth_label: str
    truth_serial: str
    corruption_type: str
    verify_is_serial: bool         # strict-threshold decision (compliance gate)
    verify_confidence: float       # legacy: max(p_pos, 1-p_pos)
    verify_p_pos: float            # raw P(positive); enables threshold sweeps
    ocr_text: str                  # top-1 hypothesis raw text
    ocr_normalized: str            # top-1 normalized
    ocr_method: str
    cer: float | None              # CER of top-1 vs truth (admitted positives only)
    # Multi-hypothesis output: each {text, normalized, score}. Empty if not OCR'd.
    ocr_hypotheses: list = None    # type: ignore[assignment]
    # CER of the BEST hypothesis vs truth (min over all hypotheses)
    cer_best_hyp: float | None = None


def _evaluate_pdf(
    short_name: str, library: dict, limit_pages: int | None, log: logging.Logger,
    *, loose_threshold: float = 0.50, num_hypotheses: int = 5,
    ocr_batch: int = 16, do_deskew: bool = True,
) -> list[RowEval]:
    if short_name not in PDF_STEMS:
        log.warning(f"  UNKNOWN PDF: {short_name}")
        return []
    pdf_dir, stem = PDF_STEMS[short_name]
    pdf_path = pdf_dir / f"{stem}.pdf"
    json_path = pdf_dir / f"{stem}.json"
    if not pdf_path.is_file() or not json_path.is_file():
        log.warning(f"  MISSING: {short_name} ({pdf_path})")
        return []

    d = json.load(open(json_path))
    form_pages = [pg for pg in d["pages"] if pg.get("firearms")]
    if limit_pages:
        form_pages = form_pages[:limit_pages]
    log.info(f"\n[{stem}] {len(form_pages)} form-bearing pages")

    cfg_cache: dict[str, dict[str, Any] | None] = {}

    def get_cfg(year: str) -> dict[str, Any] | None:
        if year not in cfg_cache:
            cfg_cache[year] = localize._load_form_config(TEMPLATES, year)
        return cfg_cache[year]

    # Two-phase processing for batched OCR:
    #   Phase 1: classify + rasterize + verify EVERY firearm row, collect
    #            (row metadata, crop, p_pos) tuples. Verifier is fast (~5ms)
    #            so per-row is fine here.
    #   Phase 2: gather all crops with p_pos >= loose_threshold, OCR them
    #            in chunks of ``ocr_batch`` via recognize.recognize_batch.
    #
    # Batching the OCR alone gives ~3-5x throughput on MPS because TrOCR's
    # per-call overhead dominates single-row inference.
    pending: list[dict] = []  # rows pending OCR (p_pos >= loose) — has crop
    rows_skip_ocr: list[dict] = []  # rows below loose threshold — no OCR
    t0 = time.time()
    for pg_i, pg in enumerate(form_pages):
        truth_year = pg["form_year"]
        cfg = get_cfg(truth_year)
        if not cfg:
            continue
        row_y_count = len(cfg["firearm_rows"]["row_y"])
        if len(pg["firearms"]) > row_y_count:
            continue

        pdf_pageno = int(pg["page"])
        try:
            cls_result = classify.classify_single_page(
                str(pdf_path), pdf_pageno - 1, library, silent=True
            )
        except Exception as e:
            log.warning(f"  classify fail p{pdf_pageno}: {e}")
            continue
        pred_year = cls_result["label"]
        cls_status = cls_result["status"]
        cls_score = float(cls_result["score"])

        try:
            pages = convert_from_path(
                str(pdf_path), dpi=200,
                first_page=pdf_pageno, last_page=pdf_pageno,
            )
        except Exception as e:
            log.warning(f"  raster fail p{pdf_pageno}: {e}")
            continue
        if not pages:
            continue
        page_img = pages[0].convert("RGB")
        if do_deskew:
            page_img = preprocess.preprocess(page_img, do_deskew=True)
        crops = _row_crops(page_img, cfg, row_y_count)

        for i, fa in enumerate(pg["firearms"]):
            label = _classify_firearm(fa)
            truth_serial = normalize_serial(fa.get("serial", ""))
            corruption_type = (fa.get("corruption") or {}).get("type", "")
            crop = crops[i]
            v = verify.verify(crop)
            if "p_pos" in v:
                p_pos = float(v["p_pos"])
            else:
                conf = float(v["confidence"])
                p_pos = conf if v["is_serial"] else (1.0 - conf)

            base = dict(
                pdf=stem, page=pdf_pageno, row_idx=i,
                truth_year=truth_year, pred_year=pred_year,
                classifier_status=cls_status, classifier_score=cls_score,
                truth_label=label, truth_serial=truth_serial,
                corruption_type=corruption_type,
                verify_is_serial=bool(v["is_serial"]),
                verify_confidence=float(v["confidence"]),
                verify_p_pos=p_pos,
            )
            if p_pos >= loose_threshold:
                base["_crop"] = crop
                pending.append(base)
            else:
                rows_skip_ocr.append(base)

        if (pg_i + 1) % 50 == 0:
            rate = (pg_i + 1) / (time.time() - t0)
            eta = (len(form_pages) - pg_i - 1) / max(0.01, rate)
            log.info(f"  [{pg_i+1}/{len(form_pages)}] verify done, OCR pending={len(pending)} "
                     f"({rate:.2f} pg/s, eta {eta:.0f}s)")

    log.info(f"  [{stem}] verify+collect done. {len(pending)} rows to OCR, "
             f"{len(rows_skip_ocr)} below loose threshold.")

    # Phase 2: batched OCR
    rows: list[RowEval] = []
    if pending:
        t1 = time.time()
        ocr_results: list[dict] = []
        for s in range(0, len(pending), ocr_batch):
            chunk = pending[s : s + ocr_batch]
            crops_chunk = [r["_crop"] for r in chunk]
            recs = recognize.recognize_batch(crops_chunk, num_hypotheses=num_hypotheses)
            ocr_results.extend(recs)
            if (s // ocr_batch) % 10 == 0 and s > 0:
                rate = (s + len(chunk)) / (time.time() - t1)
                eta = (len(pending) - s - len(chunk)) / max(0.01, rate)
                log.info(f"  [{stem}] OCR {s+len(chunk)}/{len(pending)} "
                         f"({rate:.2f} crops/s, eta {eta:.0f}s)")
        log.info(f"  [{stem}] OCR done in {time.time()-t1:.0f}s")

        for base, rec in zip(pending, ocr_results):
            ocr_text = rec.get("text", "")
            ocr_normalized = rec.get("normalized", "")
            ocr_method = rec.get("method", "")
            ocr_hypotheses = rec.get("hypotheses", [])
            label = base["truth_label"]
            truth_serial = base["truth_serial"]
            cer: float | None = None
            cer_best: float | None = None
            if label == "positive" and truth_serial:
                if ocr_normalized:
                    cer = _levenshtein(ocr_normalized, truth_serial) / max(1, len(truth_serial))
                if ocr_hypotheses:
                    best_d = min(
                        _levenshtein(h["normalized"], truth_serial)
                        for h in ocr_hypotheses if h.get("normalized")
                    )
                    cer_best = best_d / max(1, len(truth_serial))
            del base["_crop"]
            rows.append(RowEval(
                **base,
                ocr_text=ocr_text, ocr_normalized=ocr_normalized,
                ocr_method=ocr_method, cer=cer,
                ocr_hypotheses=ocr_hypotheses,
                cer_best_hyp=cer_best,
            ))

    for base in rows_skip_ocr:
        rows.append(RowEval(
            **base,
            ocr_text="", ocr_normalized="", ocr_method="", cer=None,
            ocr_hypotheses=[], cer_best_hyp=None,
        ))

    log.info(f"  done {stem}: {len(rows)} rows in {time.time() - t0:.0f}s")
    return rows


def _short_summary(rows: list[RowEval]) -> None:
    if not rows:
        print("No rows evaluated.")
        return
    n = len(rows)
    seen_pages = {(r.pdf, r.page) for r in rows}
    pred_correct = sum(1 for r in rows if r.pred_year == r.truth_year)
    pos_rows = [r for r in rows if r.truth_label == "positive" and r.truth_serial]
    adm = [r for r in pos_rows if r.verify_is_serial and r.ocr_normalized]
    exact = sum(1 for r in adm if r.ocr_normalized == r.truth_serial)
    print()
    print(f"v2 evaluation — {n} rows from {len(seen_pages)} pages")
    print(f"  classifier accuracy : {pred_correct/n:.4f}")
    print(f"  positives admitted  : {len(adm)}/{len(pos_rows)} ({100*len(adm)/max(1,len(pos_rows)):.1f}%)")
    print(f"  serial exact match  : {exact}/{len(pos_rows)} ({100*exact/max(1,len(pos_rows)):.1f}%)")
    print(f"  → run report_compliance for the full breakdown")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdfs", nargs="+", default=DEFAULT_PDFS,
                    choices=list(PDF_STEMS.keys()) + ["@compliance", "@full", "@master"],
                    help="PDF short-names from PDF_STEMS, or @compliance / @full / @master presets")
    ap.add_argument("--limit-pages", type=int, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--loose-threshold", type=float, default=0.50,
                    help="OCR is run on rows with verifier p_pos >= this. "
                         "Strict (compliance) decision is independent — "
                         "stored in verify_is_serial. Reports apply their own "
                         "admit threshold downstream.")
    ap.add_argument("--num-hypotheses", type=int, default=5,
                    help="TrOCR beam search returns this many hypotheses per row.")
    ap.add_argument("--ocr-batch", type=int, default=16,
                    help="Number of crops per batched TrOCR call. Larger = "
                         "faster but more memory. 16 is a good MPS default.")
    ap.add_argument("--no-deskew", dest="do_deskew", action="store_false",
                    help="Disable page deskew. Default is to deskew (most "
                         "real PDFs have small skew that drifts the crop "
                         "off the data cells).")
    args = ap.parse_args()

    # Expand presets
    pdfs: list[str] = []
    for p in args.pdfs:
        if p == "@compliance":
            pdfs.extend(COMPLIANCE_PDFS)
        elif p == "@full":
            pdfs.extend(FULL_HELDOUT_PDFS)
        elif p == "@master":
            # Held-out only — excludes the v2_1 PDFs that overlap with training
            pdfs.extend(MASTERTESTING_HELDOUT_PDFS)
        else:
            pdfs.append(p)
    # De-duplicate while preserving order
    seen: set[str] = set()
    pdfs = [p for p in pdfs if not (p in seen or seen.add(p))]

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    log = logging.getLogger(__name__)
    logging.getLogger("pipeline.classify").setLevel(logging.WARNING)

    log.info(f"PDFs to evaluate ({len(pdfs)}): {', '.join(pdfs)}")
    log.info("Building classifier template library...")
    library = classify.build_template_library(str(TEMPLATES))

    all_rows: list[RowEval] = []
    for short in pdfs:
        all_rows.extend(_evaluate_pdf(
            short, library, args.limit_pages, log,
            loose_threshold=args.loose_threshold,
            num_hypotheses=args.num_hypotheses,
            ocr_batch=args.ocr_batch,
            do_deskew=args.do_deskew,
        ))

    log.info(f"\nAggregating {len(all_rows)} row evaluations...")
    _short_summary(all_rows)

    if args.json:
        out = {
            "rows": [
                {
                    "pdf": r.pdf, "page": r.page, "row_idx": r.row_idx,
                    "truth_year": r.truth_year, "pred_year": r.pred_year,
                    "classifier_status": r.classifier_status,
                    "classifier_score": r.classifier_score,
                    "truth_label": r.truth_label,
                    "truth_serial": r.truth_serial,
                    "corruption_type": r.corruption_type,
                    "verify_is_serial": r.verify_is_serial,
                    "verify_confidence": r.verify_confidence,
                    "verify_p_pos": r.verify_p_pos,
                    "ocr_text": r.ocr_text,
                    "ocr_normalized": r.ocr_normalized,
                    "ocr_method": r.ocr_method,
                    "cer": r.cer,
                    "ocr_hypotheses": r.ocr_hypotheses or [],
                    "cer_best_hyp": r.cer_best_hyp,
                }
                for r in all_rows
            ],
            "config": {
                "loose_threshold": args.loose_threshold,
                "num_hypotheses": args.num_hypotheses,
                "ocr_batch": args.ocr_batch,
                "do_deskew": args.do_deskew,
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        log.info(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
