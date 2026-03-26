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
LOG_FILE         = "classifier.log"
# ─────────────────────────────────────────────────────────────────────────────

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()          # also print to console
    ]
)
log = logging.getLogger(__name__)
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Load the first page of a PDF or image file including multi-page TIFF as grayscale."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        pages = convert_from_path(path, dpi=150, first_page=1, last_page=1)
        img = pages[0]
    elif ext in (".tif", ".tiff"):
        img = Image.open(path)
        try:
            img.seek(0)
        except EOFError:
            pass
        img = img.copy()
    else:
        img = Image.open(path)
    return np.array(img.convert("L").resize(IMG_SIZE))


def build_template_library(templates_dir: str) -> dict:
    """
    Load all reference forms from templates_dir.
    Folder structure:
        templates/
            ATF-4473/  reference.pdf
            ATF-4467/  reference.pdf
    """
    library = {}
    for form_name in sorted(os.listdir(templates_dir)):
        form_dir = os.path.join(templates_dir, form_name)
        if not os.path.isdir(form_dir):
            continue
        refs = []
        for fname in os.listdir(form_dir):
            fpath = os.path.join(form_dir, fname)
            try:
                refs.append(load_image(fpath))
                log.debug(f"  Loaded template: {form_name}/{fname}")
            except Exception as e:
                log.warning(f"  Could not load template {form_name}/{fname}: {e}")
        if refs:
            library[form_name] = refs

    log.info(f"Template library built — {len(library)} form type(s): {list(library.keys())}")
    return library


def crop_header(img: np.ndarray, pct: float = 0.15) -> np.ndarray:
    """Crop to the top percentage of the image to focus on form headers/numbers."""
    h = int(img.shape[0] * pct)
    return img[:h, :]


def classify(doc_path: str, library: dict, threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """
    Compare doc against all templates and return a result dict:
        {
            "file":       str,
            "label":      str,
            "score":      float,
            "status":     "OK" | "UNSURE",
            "all_scores": { form_name: best_score, ... }
        }
    """
    log.info(f"Classifying: {doc_path}")

    try:
        doc_img = load_image(doc_path)
    except Exception as e:
        log.error(f"  Failed to load document: {e}")
        return {"file": doc_path, "label": None, "score": 0.0,
                "status": "ERROR", "all_scores": {}}

    doc_header = crop_header(doc_img)
    all_scores = {}

    for label, ref_images in library.items():
        # Score against each reference image; take the best
        best_ref_score = -1.0
        for ref_img in ref_images:
            ref_header = crop_header(ref_img)
            try:
                score = ssim(doc_header, ref_header, data_range=255)
            except Exception as e:
                log.warning(f"  SSIM failed for {label}: {e}")
                score = 0.0
            best_ref_score = max(best_ref_score, score)

        all_scores[label] = round(best_ref_score, 4)
        log.debug(f"  {label:20s}  score: {best_ref_score:.4f}")

    # Pick winner
    best_label = max(all_scores, key=all_scores.get)
    best_score = all_scores[best_label]
    status     = "OK" if best_score >= threshold else "UNSURE"

    # Sort scores descending for clean logging
    ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    log.info(f"  Scores (ranked):")
    for rank, (lbl, sc) in enumerate(ranked, 1):
        flag = " ← best" if lbl == best_label else ""
        log.info(f"    {rank}. {lbl:20s}  {sc:.4f}{flag}")

    if status == "UNSURE":
        log.warning(f"  LOW CONFIDENCE — best score {best_score:.4f} is below "
                    f"threshold {threshold}. Flagged as UNSURE.")
    else:
        log.info(f"  Result: {best_label}  (score: {best_score:.4f})  [{status}]")

    return {
        "file":       doc_path,
        "label":      best_label,
        "score":      best_score,
        "status":     status,
        "all_scores": all_scores
    }


def classify_batch(doc_paths: list, library: dict, threshold: float = CONFIDENCE_THRESHOLD) -> list:
    """Classify a list of documents and print a summary table at the end."""
    results = [classify(p, library, threshold) for p in doc_paths]

    # ── Summary report ────────────────────────────────────────────────────────
    ok     = [r for r in results if r["status"] == "OK"]
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
        log.info(f"  [{r['status']:6s}]  {r['file']:40s}  →  {label_str:20s}  score: {r['score']:.4f}")

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