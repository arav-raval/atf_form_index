"""
Minimal serial search UI + PDF viewer.

Uses ``search_index.json`` produced by :mod:`document_pipeline`. Run::

    pip install flask
    python webapp.py

Open http://127.0.0.1:5000/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import base64

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# Allow override via env for deployment
SEARCH_INDEX = Path(os.environ.get("SEARCH_INDEX", str(ROOT / "search_index.json")))


def _safe_pdf_path(stored: str) -> Path | None:
    """Only serve files under the project tree."""
    p = Path(stored).resolve()
    root = ROOT.resolve()
    if p == root or root in p.parents:
        if p.is_file() and p.suffix.lower() == ".pdf":
            return p
    return None


def _path_to_token(p: Path) -> str:
    raw = str(p.resolve()).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _token_to_path(token: str) -> Path | None:
    pad = "=" * (-len(token) % 4)
    try:
        raw = base64.urlsafe_b64decode(token + pad).decode()
    except Exception:
        return None
    return _safe_pdf_path(raw)


def create_app():
    from flask import Flask, abort, redirect, render_template_string, request, send_file, url_for

    app = Flask(__name__)

    HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Serial search</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.25rem; font-weight: 600; }
    input { font-size: 1rem; padding: 0.45rem 0.6rem; width: min(100%, 320px); }
    button { font-size: 1rem; padding: 0.45rem 1rem; cursor: pointer; }
    .err { color: #b00020; margin-top: 0.75rem; }
    .meta { color: #444; font-size: 0.9rem; margin: 0.75rem 0; }
    .pdf-frame { width: 100%; height: 78vh; border: 1px solid #ccc; border-radius: 4px; }
    a.dl { font-size: 0.9rem; margin-left: 0.5rem; }
  </style>
</head>
<body>
  <h1>Document search (predicted serial)</h1>
  <form method="get" action="{{ url_for('search') }}">
    <input type="search" name="q" placeholder="Serial number" value="{{ q|e }}" autocomplete="off" />
    <button type="submit">Search</button>
  </form>
  {% if error %}
  <p class="err">{{ error }}</p>
  {% endif %}
  {% if entry %}
  <p class="meta">
    Predicted year: <strong>{{ entry.predicted_year|e }}</strong>
    · Ground truth (if any): <strong>{{ entry.ground_truth|e }}</strong>
  </p>
  <p class="meta">
    <a class="dl" href="{{ entry.pdf_url|e }}">Open PDF</a>
  </p>
  <iframe class="pdf-frame" title="PDF" src="{{ entry.viewer_url|e }}"></iframe>
  {% endif %}
  <p class="meta" style="margin-top:2rem;font-size:0.8rem;color:#888;">
    Index: {{ index_path|e }} — ingest PDFs with <code>python document_pipeline.py ingest</code> (from repo root)
  </p>
</body>
</html>
"""

    @app.get("/")
    def home():
        return redirect(url_for("search"))

    @app.get("/search")
    def search():
        q = (request.args.get("q") or "").strip()
        error = None
        entry = None
        if q:
            from document_pipeline import load_search_index
            from serial_extract import normalize_serial

            nq = normalize_serial(q)
            idx = load_search_index(SEARCH_INDEX)
            raw = (idx.get("by_serial") or {}).get(nq)
            if not raw:
                error = f"No indexed document with normalized serial “{nq}”. Run offline ingest first."
            else:
                pdf_path = raw.get("pdf_path") or ""
                sp = _safe_pdf_path(pdf_path)
                if not sp:
                    error = "PDF path in index is not allowed or missing."
                else:
                    tok = _path_to_token(sp)
                    entry = {
                        "predicted_year": raw.get("predicted_year", ""),
                        "ground_truth": raw.get("ground_truth_serial") or "—",
                        "viewer_url": url_for("view_pdf", token=tok),
                        "pdf_url": url_for("download_pdf", token=tok),
                    }
        return render_template_string(
            HTML,
            q=q,
            error=error,
            entry=entry,
            index_path=str(SEARCH_INDEX),
        )

    @app.get("/pdf/view/<token>")
    def view_pdf(token: str):
        """Inline PDF for iframe (same file as download)."""
        p = _token_to_path(token)
        if not p:
            abort(404)
        return send_file(p, mimetype="application/pdf")

    @app.get("/pdf/file/<token>")
    def download_pdf(token: str):
        p = _token_to_path(token)
        if not p:
            abort(404)
        return send_file(
            p,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=p.name,
        )

    return app


def main() -> int:
    try:
        from flask import Flask  # noqa: F401
    except ImportError:
        print("Install Flask: pip install flask", file=sys.stderr)
        return 1
    app = create_app()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
