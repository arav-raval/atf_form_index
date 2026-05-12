"""Stage 2 — Classify form version.

Two backends, picked at call time:
  1. ConvNeXt-Tiny page classifier (``best_model.pt`` + ``label_map.pkl`` at
     repo root). Per-page (year, page_within_form) prediction with a
     ``discard`` class for non-form pages. Trained from scratch, not a
     template-matching heuristic. Preferred when the checkpoint exists.
  2. SSIM template matching against ``FormTemplates/<year>/``. Used when no
     ConvNeXt checkpoint is present.

Both backends operate on the **raw rasterized page** — no Sauvola
preprocessing. Preprocessing is applied later (stage 3) before
localization / verification.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image
from pdf2image import convert_from_path
from skimage.metrics import structural_similarity as ssim

TEMPLATES_DIR = "templates"
IMG_SIZE = (800, 1000)
CONFIDENCE_THRESHOLD = 0.75
_LOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(_LOG_DIR, "classifier.log")
_TEMPLATE_EXTS = {".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp"}

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


def _raster_to_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L").resize(IMG_SIZE))


def _firearm_page_from_config(form_dir: str) -> int:
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
    return _firearm_page_from_config(os.path.join(templates_dir, year_label))


def load_template_page(path: str, page_0based: int) -> np.ndarray:
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
    return load_template_page(path, page_0based)


def load_document_pages(path: str) -> list[np.ndarray]:
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


def _best_ssim_across_pages(
    ref: np.ndarray, doc_pages: list[np.ndarray]
) -> tuple[float, int]:
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
                log.warning(
                    f"  Could not load template {form_name}/{fname} (page {page_idx}): {e}"
                )
        if refs:
            library[form_name] = refs
            log.info(
                f"  Template {form_name}: firearm_rows.page={page_idx}, "
                f"{len(refs)} reference file(s)"
            )

    log.info(f"Template library built — {len(library)} form type(s): {list(library.keys())}")
    return library


def classify(doc_path: str, library: dict, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
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

    log.info(
        f"  Test document: {len(doc_pages)} page(s); "
        f"matching full template page vs each page"
    )

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
    threshold: float | None = None,
    *,
    silent: bool = False,
) -> dict:
    if not silent:
        log.info(f"Classifying single page: {doc_path}  [page index {page_0based}]")

    # Prefer ConvNeXt when available
    if _convnext_available():
        th = threshold if threshold is not None else CONVNEXT_THRESHOLD
        result = _classify_single_page_convnext(doc_path, page_0based, th)
        if result is not None:
            return result

    # SSIM fallback
    if threshold is None:
        threshold = CONFIDENCE_THRESHOLD

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
            log.info(f"  Result: {best_label}  (score: {best_score:.4f})  [{status}]")

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


def classify_batch(
    doc_paths: list, library: dict, threshold: float = CONFIDENCE_THRESHOLD
) -> list:
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


_library_cache: dict | None = None

# ---------------------------------------------------------------------------
# ConvNeXt backend
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
# ConvNeXt checkpoint lives in ``Classifier Package/`` (the original
# training output dir). Falls back to repo root for either file if
# someone copies them up.
_CONVNEXT_PKG_DIR = _REPO_ROOT / "Classifier Package"
_CONVNEXT_MODEL_PATH = _CONVNEXT_PKG_DIR / "best_model.pt"
if not _CONVNEXT_MODEL_PATH.is_file():
    _CONVNEXT_MODEL_PATH = _REPO_ROOT / "best_model.pt"
_CONVNEXT_LABELS_PATH = _CONVNEXT_PKG_DIR / "label_map.pkl"
if not _CONVNEXT_LABELS_PATH.is_file():
    _CONVNEXT_LABELS_PATH = _REPO_ROOT / "label_map.pkl"
_CONVNEXT_CACHE: dict | None = None
# ConvNeXt's intrinsic decision threshold on the top-class softmax probability.
# Below this we mark UNSURE.
CONVNEXT_THRESHOLD = 0.50


def _convnext_available() -> bool:
    return _CONVNEXT_MODEL_PATH.is_file() and _CONVNEXT_LABELS_PATH.is_file()


def _try_load_convnext() -> dict | None:
    """Lazily load ConvNeXt + label map. Cached. Returns None if torch isn't
    installed or files are missing."""
    global _CONVNEXT_CACHE
    if _CONVNEXT_CACHE is not None:
        return _CONVNEXT_CACHE
    if not _convnext_available():
        return None
    try:
        import pickle
        import torch
        import torch.nn as nn
        from torchvision import transforms
        from torchvision.models import convnext_tiny
    except ImportError:
        return None
    try:
        with open(_CONVNEXT_LABELS_PATH, "rb") as f:
            meta = pickle.load(f)
        idx_to_label = meta["idx_to_label"]
        num_classes = len(idx_to_label)

        model = convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.0),
            nn.Linear(in_features, num_classes),
        )
        ckpt = torch.load(_CONVNEXT_MODEL_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available()
                              else "cpu")
        model.to(device)

        IMAGE_SIZE = 224
        tx = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        _CONVNEXT_CACHE = {
            "model": model,
            "tx": tx,
            "idx_to_label": idx_to_label,
            "device": device,
            "num_classes": num_classes,
        }
        log.info(
            f"  ConvNeXt classifier loaded ({num_classes} classes) on {device}"
        )
        return _CONVNEXT_CACHE
    except Exception as e:
        log.warning(f"  ConvNeXt load failed: {e}")
        return None


def _convnext_predict(img: Image.Image) -> dict:
    """Predict on a single PIL image. Returns the same dict shape as the SSIM
    classifier, with extra ``page_within_form`` field."""
    import torch
    import torch.nn.functional as F

    cache = _try_load_convnext()
    if cache is None:
        return None  # caller falls back to SSIM

    rgb = img.convert("RGB")
    tensor = cache["tx"](rgb).unsqueeze(0).to(cache["device"])
    with torch.no_grad():
        probs = F.softmax(cache["model"](tensor), dim=1)[0]
    top = int(probs.argmax().item())
    top_score = float(probs[top].item())
    label_obj = cache["idx_to_label"][top]

    def _parse(lbl):
        """Returns (year, page_type) where page_type is 'serial' /
        'continuation' / None. Handles both string labels
        ('2016_serial', 'discard') and legacy tuple labels (('2016', 3))."""
        if lbl == "discard":
            return None, None
        if isinstance(lbl, str):
            if "_" in lbl:
                year, page_type = lbl.rsplit("_", 1)
                return year, page_type
            return lbl, None
        if isinstance(lbl, tuple) and len(lbl) >= 2:
            return str(lbl[0]), lbl[1]
        return None, None

    year, page_type = _parse(label_obj)
    is_discard = label_obj == "discard"

    # Per-year max-prob (ignoring page_type), for SSIM-shape compatibility.
    per_year: dict[str, float] = {}
    for idx, lbl in cache["idx_to_label"].items():
        y, _pt = _parse(lbl)
        if y is None:
            continue
        s = float(probs[idx].item())
        if y not in per_year or s > per_year[y]:
            per_year[y] = round(s, 4)

    return {
        "label": year,
        "score": round(top_score, 4),
        "page_within_form": page_type,
        "is_discard": is_discard,
        "all_scores": per_year,
    }


def _classify_pdf_convnext(pdf_path: str, threshold: float) -> dict | None:
    """ConvNeXt-backed equivalent of ``classify``. Predicts every page,
    picks the (page, year) with the highest non-discard confidence as the
    document's classification."""
    cache = _try_load_convnext()
    if cache is None:
        return None
    try:
        if pdf_path.lower().endswith(".pdf"):
            pages = convert_from_path(pdf_path, dpi=150)
        else:
            pages = [Image.open(pdf_path)]
    except Exception as e:
        log.error(f"  Failed to load document for ConvNeXt: {e}")
        return {
            "file": pdf_path,
            "label": None, "score": 0.0, "status": "ERROR",
            "all_scores": {}, "best_doc_page": None, "num_doc_pages": 0,
        }
    if not pages:
        return {
            "file": pdf_path,
            "label": None, "score": 0.0, "status": "ERROR",
            "all_scores": {}, "best_doc_page": None, "num_doc_pages": 0,
        }

    best_page_idx = 0
    best_year: str | None = None
    best_score = 0.0
    page_within_form: int | None = None
    all_scores_agg: dict[str, float] = {}
    for i, pg in enumerate(pages):
        pred = _convnext_predict(pg)
        if pred is None:
            continue
        # Aggregate per-year scores across pages by max
        for y, s in pred["all_scores"].items():
            if y not in all_scores_agg or s > all_scores_agg[y]:
                all_scores_agg[y] = s
        # Pick the page+label with the highest non-discard score
        if not pred["is_discard"] and pred["score"] > best_score:
            best_score = pred["score"]
            best_year = pred["label"]
            best_page_idx = i
            page_within_form = pred["page_within_form"]

    status = "OK" if best_score >= threshold else "UNSURE"
    return {
        "file": pdf_path,
        "label": best_year,
        "score": best_score,
        "status": status,
        "all_scores": all_scores_agg,
        "best_doc_page": best_page_idx,
        "num_doc_pages": len(pages),
        "page_within_form": page_within_form,
        "backend": "convnext",
    }


def _classify_single_page_convnext(
    pdf_path: str, page_0based: int, threshold: float,
) -> dict | None:
    cache = _try_load_convnext()
    if cache is None:
        return None
    try:
        if pdf_path.lower().endswith(".pdf"):
            p1 = page_0based + 1
            pages = convert_from_path(pdf_path, dpi=150,
                                      first_page=p1, last_page=p1)
        else:
            pages = [Image.open(pdf_path)]
    except Exception as e:
        log.error(f"  ConvNeXt page load failed: {e}")
        return {
            "file": pdf_path,
            "label": None, "score": 0.0, "status": "ERROR",
            "all_scores": {}, "best_doc_page": page_0based,
            "num_doc_pages": 1, "page_0based": page_0based,
        }
    if not pages:
        return {
            "file": pdf_path,
            "label": None, "score": 0.0, "status": "ERROR",
            "all_scores": {}, "best_doc_page": page_0based,
            "num_doc_pages": 1, "page_0based": page_0based,
        }
    pred = _convnext_predict(pages[0])
    if pred is None:
        return None

    label = pred["label"] if not pred["is_discard"] else None
    score = pred["score"]
    status = "OK" if (label and score >= threshold) else "UNSURE"
    return {
        "file": pdf_path,
        "label": label,
        "score": score,
        "status": status,
        "all_scores": pred["all_scores"],
        "best_doc_page": page_0based,
        "num_doc_pages": 1,
        "page_0based": page_0based,
        "page_within_form": pred["page_within_form"],
        "is_discard": pred["is_discard"],
        "backend": "convnext",
    }


def classify_pdf(
    pdf_path,
    templates_dir,
    threshold: float | None = None,
) -> dict:
    """Cached convenience wrapper used by :mod:`pipeline.orchestrator`.

    Prefers the ConvNeXt backend when available; falls back to SSIM template
    matching. ``threshold`` is interpreted by the active backend (ConvNeXt
    uses softmax probability; SSIM uses ``CONFIDENCE_THRESHOLD``).
    """
    if _convnext_available():
        th = threshold if threshold is not None else CONVNEXT_THRESHOLD
        result = _classify_pdf_convnext(str(pdf_path), th)
        if result is not None:
            return result
    # SSIM fallback
    global _library_cache
    if _library_cache is None:
        _library_cache = build_template_library(str(templates_dir))
    th = threshold if threshold is not None else CONFIDENCE_THRESHOLD
    return classify(str(pdf_path), _library_cache, threshold=th)


if __name__ == "__main__":
    import sys

    library = build_template_library(TEMPLATES_DIR)

    docs = sys.argv[1:]
    if not docs:
        print("Usage: python -m pipeline.classify doc1.pdf doc2.tiff ...")
        sys.exit(1)

    classify_batch(docs, library)
