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
  {% if matches %}
    {% if matches|length > 1 %}
    <p class="meta">{{ matches|length }} matching documents.</p>
    {% endif %}
    {% for entry in matches %}
    <p class="meta">
      Matched indexed serial: <strong>{{ entry.matched_serial|e }}</strong>
      {% if entry.match_type == 'exact' %}
        <span style="color:#0a0;">(exact)</span>
      {% else %}
        <span style="color:#a80;">(near-match: OCR confusion)</span>
      {% endif %}
      <br>
      Predicted year: <strong>{{ entry.predicted_year|e }}</strong>
      · Ground truth (if any): <strong>{{ entry.ground_truth|e }}</strong>
      · <a class="dl" href="{{ entry.pdf_url|e }}">Open PDF</a>
    </p>
    <iframe class="pdf-frame" title="PDF" src="{{ entry.viewer_url|e }}"></iframe>
    {% endfor %}
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
        matches: list[dict] = []
        if q:
            from document_pipeline import load_search_index
            from pipeline import fuzzy_search
            from pipeline.recognize import normalize_serial

            nq = normalize_serial(q)
            idx = load_search_index(SEARCH_INDEX)
            by_serial = idx.get("by_serial") or {}
            fuzzy_matches = fuzzy_search.search(q, by_serial)

            if not fuzzy_matches:
                error = (
                    f"No indexed document matches “{nq}” (exact or near-match). "
                    "Run offline ingest first."
                )
            else:
                for fm in fuzzy_matches:
                    for raw in fm.refs:
                        sp = _safe_pdf_path(raw.get("pdf_path") or "")
                        if not sp:
                            continue
                        tok = _path_to_token(sp)
                        matches.append({
                            "predicted_year": raw.get("predicted_year", ""),
                            "ground_truth": raw.get("ground_truth_serial") or "—",
                            "matched_serial": fm.indexed_serial,
                            "match_type": fm.reason,
                            "viewer_url": url_for("view_pdf", token=tok),
                            "pdf_url": url_for("download_pdf", token=tok),
                        })
                if not matches:
                    error = "PDF paths in the index are not allowed or missing."
        return render_template_string(
            HTML,
            q=q,
            error=error,
            matches=matches,
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
