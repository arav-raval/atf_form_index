"""Glues stages 1→5. Returns a single :class:`PipelineResult`.

Per-row flow:
    classify → rasterize → preprocess → crop block → split into rows
    for each row: verify (no OCR yet) → OCR only if admitted

Persistence (sidecar JSON, search index) lives in ``document_pipeline.py`` —
this module is pure: PDF in, result out.
"""
from __future__ import annotations

from pathlib import Path

from pipeline import classify, localize, preprocess, recognize, verify
from pipeline.types import PipelineResult, RowResult


def run(
    pdf_path: str | Path,
    templates_dir: Path,
    threshold: float | None = None,
) -> PipelineResult:
    pdf_path = Path(pdf_path).resolve()
    result = PipelineResult(pdf_path=pdf_path)

    if not pdf_path.is_file():
        result.status = "ERROR"
        result.error = "PDF not found"
        return result

    # Stage 2 — classify
    cr = classify.classify_pdf(pdf_path, templates_dir, threshold=threshold)
    result.predicted_year = cr.get("label")
    result.classifier_score = float(cr.get("score") or 0.0)
    result.classifier_status = str(cr.get("status") or "ERROR")

    if not result.predicted_year or result.classifier_status == "ERROR":
        result.status = "CLASSIFY_FAILED"
        return result

    # Stage 3a — rasterize the serial-bearing page
    page_img, cfg, page_0based = localize.rasterize_serial_page(
        pdf_path, result.predicted_year, templates_dir
    )
    result.page_0based = page_0based
    if page_img is None or cfg is None:
        result.status = "RASTERIZE_FAILED"
        return result

    # Stage 1 — preprocess
    page_img = preprocess.preprocess(page_img)
    result.page_image = page_img

    # Stage 3b — crop the full serial column block
    block, box_pts = localize.crop_serial_block(page_img, cfg)
    result.block_crop = block
    result.block_box_pts = box_pts

    # Stage 3c — split block into per-row crops
    expected_rows = int((cfg.get("firearm_rows") or {}).get("max", 3))
    row_crops = localize.split_block_into_rows(block, expected_rows)

    # Load the blank-template block once; the verifier uses it to detect rows
    # that are just pre-printed form chrome. Resized to match the data block so
    # we can crop the same row coordinates from both.
    template_block = localize.get_template_block(result.predicted_year, templates_dir)
    if template_block is not None and template_block.size != block.size:
        template_block = template_block.resize(block.size)

    # Stages 4 + 5 — verify-then-OCR per row
    for i, (row_crop, box_in_block) in enumerate(row_crops):
        row = RowResult(row_index=i, box_in_block=box_in_block)

        template_row = None
        if template_block is not None:
            L, T, R, B = box_in_block
            template_row = template_block.crop((L, T, R, B))

        v = verify.verify(row_crop, template_crop=template_row)
        row.verify_status = str(v["reason"])
        row.verify_confidence = float(v["confidence"])

        # Compliance gate: only OCR if verifier says yes. Recognize() picks
        # the OCR backend (TrOCR if available, else Tesseract) and applies any
        # backend-specific preprocessing.
        if v["is_serial"]:
            rec = recognize.recognize(row_crop)
            row.serial_raw = rec["text"]
            row.serial_normalized = rec["normalized"]
            row.looks_serial = bool(rec["looks_serial"])
            row.ocr_method = rec["method"]

        result.rows.append(row)

    result.status = "OK"
    return result
