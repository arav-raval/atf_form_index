import json
import os
import logging
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from skimage.metrics import structural_similarity as ssim

# ── Config ───────────────────────────────────────────────────────────────────
TEMPLATES_DIR    = "templates"
IMG_SIZE         = (800, 1000)
CONFIDENCE_THRESHOLD = 0.75
_LOG_DIR         = os.path.dirname(os.path.abspath(__file__))
LOG_FILE         = os.path.join(_LOG_DIR, "classifier.log")
# Only rasterize these as templates (skip form_config.json, README, etc.)
_TEMPLATE_EXTS   = {".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# ─────────────────────────────────────────────────────────────────────────────

# ── Logging setup ─────────────────────────────────────────────────────────────
# Do not rely on logging.basicConfig(): pytest (and other hosts) configure the
# root logger first, so basicConfig becomes a no-op and no FileHandler is added.
# Handlers on this module's logger work regardless, and propagate=False avoids
# duplicate lines when the root logger also has handlers.
_LOG_FMT = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _configure_logger() -> logging.Logger:
    lg = logging.getLogger(__name__)
    if lg.handlers:
        return lg
    lg.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(_LOG_FMT)
    sh = logging.StreamHandler()
    sh.setFormatter(_LOG_FMT)
    lg.addHandler(fh)
    lg.addHandler(sh)
    lg.propagate = False
    return lg


log = _configure_logger()
# ─────────────────────────────────────────────────────────────────────────────


def _raster_to_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L").resize(IMG_SIZE))


def _firearm_page_from_config(form_dir: str) -> int:
    """0-based page index from form_config.json firearm_rows.page (default 0)."""
    cfg_path = os.path.join(form_dir, "form_config.json")
    if not os.path.isfile(cfg_path):
        return 0
    try:
        with open(cfg_path, encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("firearm_rows", {}).get("page", 0))
    except Exception as e:
        log.warning(f"  Could not read firearm_rows.page from {cfg_path}: {e}")
        return 0


def template_firearm_page(templates_dir: str, year_label: str) -> int:
    """0-based ``firearm_rows.page`` for ``templates_dir/<year_label>/form_config.json``."""
    return _firearm_page_from_config(os.path.join(templates_dir, year_label))


def load_template_page(path: str, page_0based: int) -> np.ndarray:
    """
    Load a single template page: for PDFs, page_0based is 0-based index.
    For single-page images, page_0based must be 0.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        p1 = page_0based + 1
        pages = convert_from_path(path, dpi=150, first_page=p1, last_page=p1)
        if not pages:
            raise ValueError(f"No page {p1} in {path}")
        return _raster_to_array(pages[0])
    if ext in (".tif", ".tiff"):
        img = Image.open(path)
        n = getattr(img, "n_frames", 1)
        idx = min(page_0based, max(0, n - 1))
        img.seek(idx)
        return _raster_to_array(img.copy())
    if page_0based != 0:
        log.warning(f"  Ignoring page index {page_0based} for non-PDF template {path}")
    img = Image.open(path)
    return _raster_to_array(img)


def load_document_page(path: str, page_0based: int) -> np.ndarray:
    """Load one page of a document PDF or multi-page TIFF (same rules as template pages)."""
    return load_template_page(path, page_0based)


def load_document_pages(path: str) -> list[np.ndarray]:
    """Load every page of a PDF or multi-page TIFF as grayscale IMG_SIZE arrays."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        pages = convert_from_path(path, dpi=150)
        return [_raster_to_array(p) for p in pages]
    if ext in (".tif", ".tiff"):
        out = []
        img = Image.open(path)
        n = getattr(img, "n_frames", 1)
        for i in range(n):
            img.seek(i)
            out.append(_raster_to_array(img.copy()))
        return out if out else [_raster_to_array(Image.open(path))]
    img = Image.open(path)
    try:
        img.seek(0)
    except EOFError:
        pass
    return [_raster_to_array(img.copy())]


def _best_ssim_across_pages(ref: np.ndarray, doc_pages: list[np.ndarray]) -> tuple[float, int]:
    """Max SSIM(ref, page) over document pages; return (score, 0-based page index)."""
    best = -1.0
    best_i = 0
    for i, pg in enumerate(doc_pages):
        try:
            sc = ssim(ref, pg, data_range=255)
        except Exception:
            sc = 0.0
        if sc > best:
            best = sc
            best_i = i
    return best, best_i


def _score_document_against_library(
    doc_pages: list[np.ndarray],
    library: dict,
) -> tuple[dict[str, float], dict[str, int]]:
    """SSIM scores per label; label_best_page is 0-based index into doc_pages."""
    all_scores: dict[str, float] = {}
    label_best_page: dict[str, int] = {}
    for label, ref_images in library.items():
        best_label_score = -1.0
        label_page = 0
        for ref_img in ref_images:
            score, pg = _best_ssim_across_pages(ref_img, doc_pages)
            if score > best_label_score:
                best_label_score = score
                label_page = pg
        all_scores[label] = round(best_label_score, 4)
        label_best_page[label] = label_page
    return all_scores, label_best_page


def build_template_library(templates_dir: str) -> dict:
    """
    Load reference pages from templates_dir.

    Each subfolder is a label (e.g. year). ``form_config.json`` supplies
    ``firearm_rows.page`` (0-based). That page is loaded in full from each
    template file (PDF, image, …) in the folder.
    """
    library: dict[str, list[np.ndarray]] = {}
    for form_name in sorted(os.listdir(templates_dir)):
        form_dir = os.path.join(templates_dir, form_name)
        if not os.path.isdir(form_dir):
            continue
        page_idx = _firearm_page_from_config(form_dir)
        refs: list[np.ndarray] = []
        for fname in sorted(os.listdir(form_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _TEMPLATE_EXTS:
                continue
            fpath = os.path.join(form_dir, fname)
            try:
                refs.append(load_template_page(fpath, page_idx))
            except Exception as e:
                log.warning(f"  Could not load template {form_name}/{fname} (page {page_idx}): {e}")
        if refs:
            library[form_name] = refs
            log.info(
                f"  Template {form_name}: firearm_rows.page={page_idx}, "
                f"{len(refs)} reference file(s)"
            )

    log.info(f"Template library built — {len(library)} form type(s): {list(library.keys())}")
    return library


def classify(doc_path: str, library: dict, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """
    Compare the document against all templates.

    For each template year, each reference page (full page from firearm_rows.page)
    is compared to **every** page of the test document; the best SSIM wins for that
    reference, then the best across references wins for that label.

    Returns:
        label, score, status, all_scores, plus best_doc_page (0-based, same as
        firearm_rows.page) for the winning label’s strongest match.
    """
    log.info(f"Classifying: {doc_path}")

    try:
        doc_pages = load_document_pages(doc_path)
    except Exception as e:
        log.error(f"  Failed to load document: {e}")
        return {
            "file": doc_path,
            "label": None,
            "score": 0.0,
            "status": "ERROR",
            "all_scores": {},
            "best_doc_page": None,
            "num_doc_pages": 0,
        }

    if not doc_pages:
        log.error("  Document has no pages")
        return {
            "file": doc_path,
            "label": None,
            "score": 0.0,
            "status": "ERROR",
            "all_scores": {},
            "best_doc_page": None,
            "num_doc_pages": 0,
        }

    log.info(f"  Test document: {len(doc_pages)} page(s); matching full template page vs each page")

    all_scores, label_best_page = _score_document_against_library(doc_pages, library)

    best_label = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_label]
    best_page = label_best_page.get(best_label, 0)
    status = "OK" if best_score >= threshold else "UNSURE"

    ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    log.info("  Scores (ranked):")
    for rank, (lbl, sc) in enumerate(ranked, 1):
        flag = " ← best" if lbl == best_label else ""
        pg = label_best_page.get(lbl, 0)
        log.info(f"    {rank}. {lbl:20s}  {sc:.4f}  (best test page index: {pg}){flag}")

    if status == "UNSURE":
        log.warning(
            f"  LOW CONFIDENCE — best score {best_score:.4f} is below "
            f"threshold {threshold}. Flagged as UNSURE."
        )
    else:
        log.info(
            f"  Result: {best_label}  (score: {best_score:.4f})  [{status}]  "
            f"best match on test page index {best_page} (0-based)"
        )

    return {
        "file": doc_path,
        "label": best_label,
        "score": best_score,
        "status": status,
        "all_scores": all_scores,
        "best_doc_page": best_page,
        "num_doc_pages": len(doc_pages),
    }


def classify_single_page(
    doc_path: str,
    page_0based: int,
    library: dict,
    threshold: float = CONFIDENCE_THRESHOLD,
    *,
    silent: bool = False,
) -> dict:
    """
    Classify using only one page of ``doc_path`` (0-based index).

    Use this for page-by-page streaming or random page sampling. For full
    documents, prefer :func:`classify`.

    When ``silent`` is True, skips per-call INFO logs (still logs ERROR).
    """
    if not silent:
        log.info(f"Classifying single page: {doc_path}  [page index {page_0based}]")

    try:
        one = load_document_page(doc_path, page_0based)
    except Exception as e:
        log.error(f"  Failed to load page: {e}")
        return {
            "file": doc_path,
            "label": None,
            "score": 0.0,
            "status": "ERROR",
            "all_scores": {},
            "best_doc_page": page_0based,
            "num_doc_pages": 1,
            "page_0based": page_0based,
        }

    doc_pages = [one]
    all_scores, label_best_page = _score_document_against_library(doc_pages, library)

    best_label = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_label]
    best_page = label_best_page.get(best_label, 0)
    status = "OK" if best_score >= threshold else "UNSURE"

    if not silent:
        ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        log.info("  Scores (ranked):")
        for rank, (lbl, sc) in enumerate(ranked, 1):
            flag = " ← best" if lbl == best_label else ""
            log.info(f"    {rank}. {lbl:20s}  {sc:.4f}{flag}")
        if status == "UNSURE":
            log.warning(
                f"  LOW CONFIDENCE — best score {best_score:.4f} is below "
                f"threshold {threshold}."
            )
        else:
            log.info(
                f"  Result: {best_label}  (score: {best_score:.4f})  [{status}]"
            )

    return {
        "file": doc_path,
        "label": best_label,
        "score": best_score,
        "status": status,
        "all_scores": all_scores,
        "best_doc_page": best_page,
        "num_doc_pages": 1,
        "page_0based": page_0based,
    }


def classify_batch(doc_paths: list, library: dict, threshold: float = CONFIDENCE_THRESHOLD) -> list:
    """Classify a list of documents and print a summary table at the end."""
    results = [classify(p, library, threshold) for p in doc_paths]

    ok = [r for r in results if r["status"] == "OK"]
    unsure = [r for r in results if r["status"] == "UNSURE"]
    errors = [r for r in results if r["status"] == "ERROR"]

    log.info("=" * 60)
    log.info("CLASSIFICATION SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total:   {len(results)}")
    log.info(f"  OK:      {len(ok)}")
    log.info(f"  UNSURE:  {len(unsure)}")
    log.info(f"  ERRORS:  {len(errors)}")
    log.info("-" * 60)

    for r in results:
        label_str = r["label"] or "N/A"
        pg = r.get("best_doc_page")
        pg_s = f"  doc page idx {pg}" if pg is not None else ""
        log.info(
            f"  [{r['status']:6s}]  {r['file']:40s}  →  {label_str:20s}  "
            f"score: {r['score']:.4f}{pg_s}"
        )

    if unsure:
        log.warning("Files requiring manual review:")
        for r in unsure:
            log.warning(f"  {r['file']}")

    log.info("=" * 60)
    return results


if __name__ == "__main__":
    import sys

    library = build_template_library(TEMPLATES_DIR)

    docs = sys.argv[1:]
    if not docs:
        print("Usage: python classifier.py doc1.pdf doc2.tiff ...")
        sys.exit(1)

    classify_batch(docs, library)
