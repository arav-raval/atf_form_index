"""Build a full set of publication-grade figures + tables from the v7
evaluation JSONs.

Generates a single ``report_figures/`` directory with:
    01_top_level_results.png         — headline metrics bar chart
    01_top_level_results.md          — same as a table
    02_per_corruption_breakdown.png  — leak by corruption type
    02_per_corruption_breakdown.md
    03_master_stage_results.png      — stage-by-stage funnel
    03_master_stage_results.md
    04_topk_curve.png                — P(top-K) retrieval curve
    04_topk_curve.md
    05a_compliance_confusion.png     — combined / PII-only / DQ-only matrices
    05b_compliance_overall.md        — combined compliance table
    06_per_year_exact_match.png      — bar chart per year
    06_per_year_exact_match.md
    07_threshold_tradeoffs.png       — recall vs leak (PII vs DQ vs all)
    07_threshold_tradeoffs.md
    08_by_doc_type.png               — admit% / exact% by doc type
    08_by_doc_type.md
    09_safe_vs_unsafe.png            — error-type categorical
    10_cer_distribution.png          — CER histogram (admitted positives)
    11_roc_curve.png                 — verifier ROC (overall + PII-only)
    11_roc_curve.md
    12_pr_curve.png                  — verifier precision-recall curve
    12_pr_curve.md
    13_classifier_confusion.png      — per-year classifier confusion matrix
    13_classifier_confusion.md
    14_verifier_score_distribution.png — p_pos histogram by truth class
    14_verifier_score_distribution.md
    15_cer_by_year.png               — OCR CER quartiles by year
    15_cer_by_year.md
    16_error_decomposition.png       — where end-to-end exact-match fails
    16_error_decomposition.md
    17_combined_stage_table.md       — end-to-end stage table (master+compliance)

Inputs:
    /tmp/eval_v2_compliance_v7.json   (raw, all p_pos preserved)
    /tmp/eval_master_v7.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Match-style for a policy report: clean, neutral colors
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

PII_CORRUPTIONS = {"pii_in_serial", "name_in_serial"}
DATA_QUALITY_CORRUPTIONS = {"field_swap", "overflow_into_serial"}

# Operating threshold (matches saved verifier_model_vit.pt)
DEFAULT_THRESHOLD = 0.95

# A consistent palette
C_GOOD = "#2E7D32"        # green
C_BAD = "#C62828"         # red
C_NEUTRAL = "#1976D2"     # blue
C_ACCENT = "#F9A825"      # amber
C_GREY = "#616161"


def _admit(r: dict, threshold: float) -> bool:
    p = r.get("verify_p_pos")
    if p is None:
        c = r.get("verify_confidence", 0.0)
        p = c if r.get("verify_is_serial") else (1.0 - c)
    return p >= threshold


def _confusion(rows: list[dict], threshold: float, neg_filter: set | None = None) -> dict:
    tp = fn = tn = fp = 0
    for r in rows:
        lbl = r.get("truth_label")
        if lbl not in ("positive", "negative"):
            continue
        if lbl == "negative" and neg_filter is not None:
            if r.get("corruption_type", "") not in neg_filter:
                continue
        admit = _admit(r, threshold)
        if lbl == "positive" and admit: tp += 1
        elif lbl == "positive": fn += 1
        elif admit: fp += 1
        else: tn += 1
    n = tp + fn + tn + fp
    return {
        "tp": tp, "fn": fn, "tn": tn, "fp": fp, "n": n,
        "recall": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "precision": tp / max(1, tp + fp),
        "leak": fp / max(1, fp + tn),
    }


def _levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    p = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        c = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            c[j] = min(c[j-1]+1, p[j]+1, p[j-1]+cost)
        p = c
    return p[-1]


# ─────────────────────────────────────────────────────────────────────────
# 1. Top-level results
# ─────────────────────────────────────────────────────────────────────────
def fig01_top_level(comp_rows, master_rows, out_dir, thresh):
    c_pii = _confusion(comp_rows, thresh, PII_CORRUPTIONS)
    c_dq = _confusion(comp_rows, thresh, DATA_QUALITY_CORRUPTIONS)
    c_all = _confusion(comp_rows, thresh)

    # End-to-end on compliance clean PDF
    clean = [r for r in comp_rows if "no_errors" in r["pdf"]]
    pos_clean = [r for r in clean if r.get("truth_label") == "positive" and r.get("truth_serial")]
    adm_clean = [r for r in pos_clean if _admit(r, thresh) and r.get("ocr_normalized")]
    exact_clean = sum(1 for r in adm_clean if r["ocr_normalized"] == r["truth_serial"])

    # Master end-to-end
    pos_master = [r for r in master_rows if r.get("truth_label") == "positive" and r.get("truth_serial")]
    adm_master = [r for r in pos_master if _admit(r, thresh) and r.get("ocr_normalized")]
    exact_master = sum(1 for r in adm_master if r["ocr_normalized"] == r["truth_serial"])

    # Classifier accuracy across both datasets
    seen_pages: dict = {}
    for r in (comp_rows + master_rows):
        seen_pages[(r["pdf"], r["page"])] = (r["truth_year"], r["pred_year"])
    cls_acc = sum(1 for ty, py in seen_pages.values() if py == ty) / max(1, len(seen_pages))

    metrics = [
        ("Classifier\naccuracy", cls_acc, C_GOOD),
        ("Verifier\nrecall", c_all["recall"], C_NEUTRAL),
        ("PII leak rate\n(lower better)", c_pii["leak"], C_BAD),
        ("DQ leak rate\n(lower better)", c_dq["leak"], C_ACCENT),
        ("Top-1\nretrieval", _topk(master_rows, [1], admit_threshold=thresh, hypotheses_per_row=1)[1], C_NEUTRAL),
        ("End-to-end\nexact (master)", exact_master / max(1, len(pos_master)), C_GOOD),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [m[0] for m in metrics]
    vals = [m[1] for m in metrics]
    colors = [m[2] for m in metrics]
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.1%}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title(f"Top-Level Pipeline Results — verifier threshold {thresh:.2f}")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(out_dir / "01_top_level_results.png")
    plt.close(fig)

    # Markdown table
    md = ["# Top-Level Pipeline Results",
          "",
          f"Verifier operating threshold: **{thresh:.2f}**",
          "",
          "| Metric | Value |",
          "|---|---:|",
          f"| Classifier page accuracy (combined held-out) | {cls_acc:.4f} ({100*cls_acc:.1f}%) |",
          f"| Verifier recall (compliance set) | {c_all['recall']:.4f} ({100*c_all['recall']:.1f}%) |",
          f"| **PII leak rate** (privacy-violating only) | **{c_pii['leak']:.4f}** ({100*c_pii['leak']:.2f}%) |",
          f"| Data-quality leak rate | {c_dq['leak']:.4f} ({100*c_dq['leak']:.2f}%) |",
          f"| Combined leak rate (all corruption types) | {c_all['leak']:.4f} ({100*c_all['leak']:.2f}%) |",
          f"| Top-1 retrieval (master, production gate) | {_topk(master_rows, [1], admit_threshold=thresh, hypotheses_per_row=1)[1]:.4f} |",
          f"| Top-100 retrieval (master, production gate) | {_topk(master_rows, [100], admit_threshold=thresh, hypotheses_per_row=1)[100]:.4f} |",
          f"| End-to-end exact match — clean compliance PDF | {exact_clean / max(1, len(pos_clean)):.4f} |",
          f"| End-to-end exact match — master held-out | {exact_master / max(1, len(pos_master)):.4f} |",
          ]
    (out_dir / "01_top_level_results.md").write_text("\n".join(md) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 2. Per-corruption-type breakdown
# ─────────────────────────────────────────────────────────────────────────
def fig02_per_corruption(comp_rows, out_dir, thresh):
    err = [r for r in comp_rows if "no_errors" not in r["pdf"]
           and ("errors" in r["pdf"] or "error" in r["pdf"])]
    by = defaultdict(list)
    for r in err:
        ct = r.get("corruption_type") or "(clean)"
        by[ct].append(r)

    rows = []
    for ct, sub in by.items():
        if ct == "(clean)": continue
        admitted = sum(1 for r in sub if _admit(r, thresh))
        leak = admitted / max(1, len(sub))
        if ct in PII_CORRUPTIONS:
            cat = "PII"; color = C_BAD
        elif ct in DATA_QUALITY_CORRUPTIONS:
            cat = "Data-Quality"; color = C_ACCENT
        elif ct == "serial_overflow":
            cat = "Positive (correct admit)"; color = C_GOOD
        else:
            cat = "Other"; color = C_GREY
        rows.append((ct, cat, len(sub), admitted, leak, color))
    rows.sort(key=lambda r: (r[1] != "PII", -r[4]))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [r[0] for r in rows]
    leaks = [r[4] for r in rows]
    colors = [r[5] for r in rows]
    bars = ax.barh(labels, leaks, color=colors, edgecolor="black", linewidth=0.5)
    for bar, (_, _, n, adm, leak, _) in zip(bars, rows):
        ax.text(leak + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{leak:.1%}  ({adm}/{n})",
                va="center", fontsize=9)
    ax.set_xlim(0, max(leaks) * 1.25 + 0.05)
    ax.set_xlabel("Admit rate (lower is better for negatives)")
    ax.set_title(f"Per-Corruption-Type Admit Rate — threshold {thresh:.2f}")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=C_BAD, label="PII (must reject)"),
        Patch(color=C_ACCENT, label="Data-quality (should reject)"),
        Patch(color=C_GOOD, label="Positive (should admit)"),
    ], loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_dir / "02_per_corruption_breakdown.png")
    plt.close(fig)

    md = ["# Per-Corruption-Type Breakdown", "",
          f"Verifier threshold: **{thresh:.2f}**", "",
          "| Corruption type | Category | n | Admitted | Leak% |",
          "|---|---|---:|---:|---:|"]
    for ct, cat, n, adm, leak, _ in rows:
        md.append(f"| `{ct}` | {cat} | {n} | {adm} | {100*leak:.2f}% |")
    md += ["",
           "**Categories:**",
           "- **PII**: privacy-violating — must be rejected for compliance.",
           "- **Data-quality**: wrong data (field swap or overflow from neighbor cell). Not personal info, but should not enter the index.",
           "- **Positive**: `serial_overflow` is labeled positive in our scheme because the serial IS in the box, just messy. High admit rate is correct."]
    (out_dir / "02_per_corruption_breakdown.md").write_text("\n".join(md) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 3. Master stage results (funnel)
# ─────────────────────────────────────────────────────────────────────────
def fig03_stage_results(master_rows, out_dir, thresh):
    pos = [r for r in master_rows if r.get("truth_label") == "positive" and r.get("truth_serial")]
    n_pos = len(pos)

    # Stages
    n_pages = len({(r["pdf"], r["page"]) for r in master_rows})
    n_cls_correct = sum(1 for ty, py in
                        {(r["pdf"], r["page"]): (r["truth_year"], r["pred_year"]) for r in master_rows}.values()
                        if py == ty)

    n_admitted = sum(1 for r in pos if _admit(r, thresh))
    n_admitted_with_ocr = sum(1 for r in pos if _admit(r, thresh) and r.get("ocr_normalized"))
    n_exact = sum(1 for r in pos if _admit(r, thresh) and r.get("ocr_normalized") == r["truth_serial"])

    # As percentage of total positives, for the funnel
    stages = [
        ("Total positives", n_pos, 100.0),
        ("Verifier admits", n_admitted, 100 * n_admitted / max(1, n_pos)),
        ("OCR produced output", n_admitted_with_ocr, 100 * n_admitted_with_ocr / max(1, n_pos)),
        ("Exact match", n_exact, 100 * n_exact / max(1, n_pos)),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [s[0] for s in stages]
    counts = [s[1] for s in stages]
    pcts = [s[2] for s in stages]

    colors = [C_NEUTRAL, "#42A5F5", "#90CAF9", C_GOOD]
    bars = ax.bar(names, counts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, n, p in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, n + max(counts) * 0.015,
                f"{n}\n({p:.1f}%)", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.set_ylabel("Number of positive firearms")
    ax.set_title(f"Master Eval — Pipeline Stage Funnel (threshold {thresh:.2f})\n"
                 f"Held-out MasterTesting: {n_pos} positives across {n_pages} pages")
    fig.tight_layout()
    fig.savefig(out_dir / "03_master_stage_results.png")
    plt.close(fig)

    md = ["# Master Eval — Stage Results", "",
          f"Held-out MasterTesting set: **{n_pos} positives**, {n_pages} pages, threshold {thresh:.2f}",
          "",
          "| Stage | Metric | Count | % of positives |",
          "|---|---|---:|---:|",
          f"| 1. Classify | Page top-1 accuracy | {n_cls_correct}/{n_pages} | {100*n_cls_correct/max(1, n_pages):.2f}% |",
          f"| 2. Verify | Admitted | {n_admitted}/{n_pos} | {100*n_admitted/max(1, n_pos):.1f}% |",
          f"| 3. OCR | Produced any output | {n_admitted_with_ocr}/{n_pos} | {100*n_admitted_with_ocr/max(1, n_pos):.1f}% |",
          f"| End-to-end | Exact match | {n_exact}/{n_pos} | {100*n_exact/max(1, n_pos):.1f}% |",
          ]
    (out_dir / "03_master_stage_results.md").write_text("\n".join(md) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 4. Top-K curve — production-equivalent
# ─────────────────────────────────────────────────────────────────────────
def _topk(
    rows: list[dict],
    k_values: list[int],
    admit_threshold: float = 0.95,
    hypotheses_per_row: int = 1,
) -> dict:
    """Production-equivalent top-K retrieval.

    The index is built the way ``document_pipeline.py`` builds
    ``search_index.json``: only rows whose verifier ``p_pos >= admit_threshold``
    contribute, and only the top-``hypotheses_per_row`` OCR hypotheses
    per admitted row (production stores top-1 only).

    Queries are every positive truth serial. A rejected row counts as a miss:
    if no row from a form is admitted, that form has nothing in the index and
    can only be retrieved via fuzzy-match coincidence with a sibling row.

    Note on "rescue": when a sibling row on the same form was admitted, its
    OCR string can rank the form in top-K even when the queried row itself was
    rejected. In master eval this happens for ~1.2% of queries at threshold
    0.95 — disclosed in the audit. The doc_key is (pdf, page), matching how a
    real query routes the user to a physical form.
    """
    from pipeline.fuzzy_search import canonical_form

    index: list[tuple[str, str]] = []  # (form_key, canonical_ocr)
    for r in rows:
        if not _admit(r, admit_threshold):
            continue
        form_key = f"{r['pdf']}#p{r['page']}"
        hyps = (r.get("ocr_hypotheses") or [])[:hypotheses_per_row]
        for h in hyps:
            n = (h.get("normalized") or "").upper()
            if n:
                index.append((form_key, canonical_form(n)))

    queries = [(f"{r['pdf']}#p{r['page']}", r["truth_serial"]) for r in rows
               if r.get("truth_label") == "positive" and r.get("truth_serial")]

    if not queries:
        return {k: 0.0 for k in k_values}
    if not index:
        return {k: 0.0 for k in k_values}

    hits = {k: 0 for k in k_values}
    for true_form, q in queries:
        q_canon = canonical_form(q)
        per_form: dict[str, int] = {}
        for f, oc in index:
            dist = _levenshtein(q_canon, oc)
            if f not in per_form or dist < per_form[f]:
                per_form[f] = dist
        scored = sorted(per_form.items(), key=lambda x: x[1])
        for k in k_values:
            top_forms = {f for f, _ in scored[:k]}
            if true_form in top_forms:
                hits[k] += 1
    return {k: hits[k] / len(queries) for k in k_values}


def fig04_topk(master_rows, out_dir, thresh):
    ks = [1, 3, 5, 10, 25, 50, 100]
    p = _topk(master_rows, ks, admit_threshold=thresh, hypotheses_per_row=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    vals = [p[k] for k in ks]
    ax.plot(ks, vals, marker="o", linewidth=2.2, markersize=8, color=C_NEUTRAL)
    for k, v in zip(ks, vals):
        ax.annotate(f"{v:.1%}", (k, v), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels(ks)
    ax.set_xlabel("K (number of candidate documents returned)")
    ax.set_ylabel("P(true document in top-K)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Top-K Retrieval — User-types-truth-serial Query\n"
                 "Master held-out, confusion-aware Levenshtein on OCR hypotheses")
    fig.tight_layout()
    fig.savefig(out_dir / "04_topk_curve.png")
    plt.close(fig)

    md = ["# Top-K Retrieval", "",
          "Simulated query: a user types the *true* serial; the system ranks all",
          "extracted-serial hypotheses by confusion-aware edit distance and returns",
          "the top-K candidate documents. P(top-K) = fraction of queries where the",
          "true source document appears within the top-K.",
          "",
          "| K | P(top-K) |",
          "|---:|---:|"]
    for k in ks:
        md.append(f"| {k} | {p[k]:.4f} ({100*p[k]:.2f}%) |")
    (out_dir / "04_topk_curve.md").write_text("\n".join(md) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 5. Compliance — confusion matrices
# ─────────────────────────────────────────────────────────────────────────
def fig05_confusion(comp_rows, out_dir, thresh):
    err = [r for r in comp_rows if "no_errors" not in r["pdf"]
           and ("errors" in r["pdf"] or "error" in r["pdf"])]
    cases = [
        ("All corruptions", _confusion(comp_rows, thresh)),
        ("PII only", _confusion(err, thresh, PII_CORRUPTIONS)),
        ("Data-quality only", _confusion(err, thresh, DATA_QUALITY_CORRUPTIONS)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (name, c) in zip(axes, cases):
        m = np.array([[c["tp"], c["fn"]], [c["fp"], c["tn"]]])
        im = ax.imshow(m, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["ADMIT", "REJECT"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Real serial", "Negative"])
        for i in range(2):
            for j in range(2):
                v = m[i, j]
                # contrast text
                color = "white" if v > m.max() / 2 else "black"
                ax.text(j, i, f"{v}", ha="center", va="center",
                        fontsize=14, fontweight="bold", color=color)
        ax.set_title(f"{name}\nrecall={c['recall']:.3f}  leak={c['leak']:.3f}",
                     fontsize=10)
        ax.set_xlabel("Verifier prediction")
        if ax is axes[0]:
            ax.set_ylabel("Truth")

    fig.suptitle(f"Compliance Confusion Matrices — threshold {thresh:.2f}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "05_compliance_confusion.png")
    plt.close(fig)

    md = ["# Compliance Confusion Matrices", "",
          f"Verifier threshold: **{thresh:.2f}**  |  ",
          "Held-out v2 _2 + Serial Error Pages",
          "",
          "## All corruption types combined",
          "",
          ]
    for name, c in cases:
        md += [f"### {name}",
               "",
               f"|  | Verifier ADMIT | Verifier REJECT |",
               f"|---|---:|---:|",
               f"| Real serial | {c['tp']} (TP) | {c['fn']} (FN) |",
               f"| Negative | {c['fp']} (FP) | {c['tn']} (TN) |",
               "",
               f"- Recall: **{c['recall']:.4f}**",
               f"- Specificity: **{c['specificity']:.4f}**",
               f"- Precision: **{c['precision']:.4f}**",
               f"- **Leak rate: {c['leak']:.4f} ({100*c['leak']:.2f}%)**",
               ""]
    (out_dir / "05_compliance_confusion.md").write_text("\n".join(md) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 6. Per-year exact match (clean compliance PDF)
# ─────────────────────────────────────────────────────────────────────────
def fig06_per_year(comp_rows, out_dir, thresh):
    clean = [r for r in comp_rows if "no_errors" in r["pdf"]
             and r.get("truth_label") == "positive" and r.get("truth_serial")]
    by = defaultdict(list)
    for r in clean:
        by[r["truth_year"]].append(r)

    years = sorted(by)
    n_each = [len(by[y]) for y in years]
    admit_pct = [sum(_admit(r, thresh) for r in by[y]) / max(1, len(by[y])) for y in years]
    exact_pct = [sum(1 for r in by[y]
                     if _admit(r, thresh) and r.get("ocr_normalized") == r["truth_serial"])
                 / max(1, len(by[y])) for y in years]

    x = np.arange(len(years))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - w/2, admit_pct, w, color=C_NEUTRAL, label="Admitted", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, exact_pct, w, color=C_GOOD, label="Exact match", edgecolor="black", linewidth=0.5)

    for bar, pct, n in zip(b1, admit_pct, n_each):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 0.015, f"{pct:.0%}",
                ha="center", fontsize=8)
    for bar, pct in zip(b2, exact_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 0.015, f"{pct:.0%}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{y}\n(n={n})" for y, n in zip(years, n_each)])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Rate")
    ax.set_title(f"Per-Year Admit & Exact-Match — clean compliance PDF (threshold {thresh:.2f})")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "06_per_year_exact_match.png")
    plt.close(fig)

    md = ["# Per-Year Exact Match (clean compliance PDF)", "",
          f"Threshold: **{thresh:.2f}**", "",
          "| Year | n | Admit% | Exact% (of all positives) |",
          "|---|---:|---:|---:|"]
    for y, n, ap, ep in zip(years, n_each, admit_pct, exact_pct):
        md.append(f"| {y} | {n} | {100*ap:.1f}% | {100*ep:.1f}% |")
    (out_dir / "06_per_year_exact_match.md").write_text("\n".join(md) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 7. Threshold tradeoffs
# ─────────────────────────────────────────────────────────────────────────
def fig07_threshold_tradeoffs(comp_rows, out_dir, current_thresh):
    err = [r for r in comp_rows if "no_errors" not in r["pdf"]
           and ("errors" in r["pdf"] or "error" in r["pdf"])]

    thresholds = np.arange(0.30, 1.00, 0.025)
    recall = []; pii_leak = []; dq_leak = []; all_leak = []
    for t in thresholds:
        c_all = _confusion(comp_rows, float(t))
        c_pii = _confusion(err, float(t), PII_CORRUPTIONS)
        c_dq = _confusion(err, float(t), DATA_QUALITY_CORRUPTIONS)
        recall.append(c_all["recall"])
        pii_leak.append(c_pii["leak"])
        dq_leak.append(c_dq["leak"])
        all_leak.append(c_all["leak"])

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(thresholds, recall, color=C_NEUTRAL, linewidth=2.4, label="Recall (real serials kept)")
    ax1.plot(thresholds, pii_leak, color=C_BAD, linewidth=2.4, label="PII leak rate")
    ax1.plot(thresholds, dq_leak, color=C_ACCENT, linewidth=2.0, linestyle="--", label="Data-quality leak rate")
    ax1.plot(thresholds, all_leak, color=C_GREY, linewidth=1.6, linestyle=":", label="Combined leak rate")

    # Operating-point markers
    ax1.axvline(current_thresh, color="black", linestyle=":", alpha=0.5)
    ax1.text(current_thresh, 1.02, f"saved\n{current_thresh:.2f}",
             ha="center", fontsize=8, color="black")

    ax1.set_xlabel("Verifier threshold")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax1.set_title("Threshold Tradeoffs — Recall vs Leak Rates")
    ax1.legend(loc="center left")
    fig.tight_layout()
    fig.savefig(out_dir / "07_threshold_tradeoffs.png")
    plt.close(fig)

    md = ["# Threshold Tradeoffs", "",
          f"Current saved threshold: **{current_thresh:.2f}**", "",
          "| Threshold | Recall | PII leak | Data-quality leak | Combined leak |",
          "|---:|---:|---:|---:|---:|"]
    for t in [0.30, 0.50, 0.65, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.985, 0.99]:
        c_all = _confusion(comp_rows, t)
        c_pii = _confusion(err, t, PII_CORRUPTIONS)
        c_dq = _confusion(err, t, DATA_QUALITY_CORRUPTIONS)
        marker = " ← **current**" if abs(t - current_thresh) < 1e-3 else ""
        md.append(f"| {t:.3f} | {100*c_all['recall']:.1f}% | "
                  f"{100*c_pii['leak']:.2f}% | {100*c_dq['leak']:.2f}% | "
                  f"{100*c_all['leak']:.2f}% |{marker}")
    (out_dir / "07_threshold_tradeoffs.md").write_text("\n".join(md) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 8. By document type
# ─────────────────────────────────────────────────────────────────────────
def _doc_type(pdf: str) -> str:
    if "complete_test" in pdf: return "complete_test"
    if "cont_test" in pdf: return "cont_test"
    if pdf.startswith("CR"): return "case_report"
    if "v2dataset" in pdf and "no_errors" not in pdf: return "v2_corrupted"
    if "v2dataset" in pdf: return "v2_clean"
    if "serial_only_error" in pdf: return "serial_only_error"
    if "serial_only" in pdf: return "serial_only"
    if "dataset_no_errors" in pdf: return "ds_clean"
    if "dataset_errors" in pdf: return "ds_corrupted"
    return "other"


def fig08_by_doc_type(master_rows, out_dir, thresh):
    by = defaultdict(list)
    for r in master_rows:
        by[_doc_type(r["pdf"])].append(r)

    types = sorted(by)
    n_each = [sum(1 for r in by[t] if r.get("truth_label") == "positive" and r.get("truth_serial")) for t in types]
    admit_pct = []; exact_pct = []
    for t in types:
        pos = [r for r in by[t] if r.get("truth_label") == "positive" and r.get("truth_serial")]
        adm = [r for r in pos if _admit(r, thresh)]
        admit_pct.append(len(adm) / max(1, len(pos)))
        ex = sum(1 for r in adm if r.get("ocr_normalized") == r["truth_serial"])
        exact_pct.append(ex / max(1, len(pos)))

    x = np.arange(len(types)); w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w/2, admit_pct, w, color=C_NEUTRAL, label="Admit %", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, exact_pct, w, color=C_GOOD, label="Exact match %", edgecolor="black", linewidth=0.5)
    for bar, p, n in zip(b1, admit_pct, n_each):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.015, f"{p:.0%}",
                ha="center", fontsize=8)
    for bar, p in zip(b2, exact_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.015, f"{p:.0%}",
                ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\nn={n}" for t, n in zip(types, n_each)], rotation=0, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Rate")
    ax.set_title(f"By Document Type — Master Eval (threshold {thresh:.2f})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "08_by_doc_type.png")
    plt.close(fig)

    md = ["# By Document Type — Master Eval", "",
          f"Threshold: **{thresh:.2f}**", "",
          "| Document type | n positives | Admit% | Exact% |",
          "|---|---:|---:|---:|"]
    for t, n, ap, ep in zip(types, n_each, admit_pct, exact_pct):
        md.append(f"| `{t}` | {n} | {100*ap:.1f}% | {100*ep:.1f}% |")
    (out_dir / "08_by_doc_type.md").write_text("\n".join(md) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 9. Safe vs Unsafe (error-type categorical)
# ─────────────────────────────────────────────────────────────────────────
def fig09_safe_vs_unsafe(comp_rows, out_dir, thresh):
    safe = [r for r in comp_rows if r.get("truth_label") == "positive"]
    unsafe = [r for r in comp_rows if r.get("truth_label") == "negative"]

    safe_admit = sum(1 for r in safe if _admit(r, thresh))
    unsafe_admit = sum(1 for r in unsafe if _admit(r, thresh))

    cats = ["SAFE\n(real serial)", "UNSAFE\n(corruption)"]
    correct_decisions = [safe_admit, len(unsafe) - unsafe_admit]
    wrong_decisions = [len(safe) - safe_admit, unsafe_admit]

    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x, correct_decisions, color=C_GOOD, label="Correct decision",
                edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x, wrong_decisions, bottom=correct_decisions, color=C_BAD,
                label="Incorrect decision", edgecolor="black", linewidth=0.5)
    totals = [c + w for c, w in zip(correct_decisions, wrong_decisions)]
    for i, (c, w_, tot) in enumerate(zip(correct_decisions, wrong_decisions, totals)):
        ax.text(i, c / 2, f"{c}\n{100*c/max(1,tot):.1f}%", ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")
        ax.text(i, c + w_/2, f"{w_}\n{100*w_/max(1,tot):.1f}%", ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("Rows")
    ax.set_title(f"Safe vs Unsafe Decisions — threshold {thresh:.2f}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "09_safe_vs_unsafe.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# 10. CER distribution
# ─────────────────────────────────────────────────────────────────────────
def fig10_cer_distribution(master_rows, out_dir, thresh):
    pos = [r for r in master_rows if r.get("truth_label") == "positive" and r.get("truth_serial")]
    cers = [r["cer"] for r in pos
            if _admit(r, thresh) and r.get("cer") is not None]
    if not cers:
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(cers, bins=np.arange(0, 1.05, 0.05), color=C_NEUTRAL,
            edgecolor="black", linewidth=0.5)
    median = statistics.median(cers); mean = statistics.mean(cers)
    ax.axvline(median, color=C_GOOD, linewidth=2, label=f"median = {median:.3f}")
    ax.axvline(mean, color=C_BAD, linestyle="--", linewidth=2, label=f"mean = {mean:.3f}")
    ax.set_xlabel("Character Error Rate (CER)")
    ax.set_ylabel("Count of admitted positives")
    ax.set_title(f"OCR CER Distribution — admitted positives (master, threshold {thresh:.2f})\n"
                 f"n = {len(cers)} crops with truth + OCR output")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "10_cer_distribution.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# 11. ROC curve (verifier)
# ─────────────────────────────────────────────────────────────────────────
def _roc_points(rows: list[dict], neg_filter: set | None = None) -> tuple[list[float], list[float], list[float]]:
    """Return (fpr, tpr, thresholds) by sweeping all unique p_pos values."""
    scored = []
    for r in rows:
        lbl = r.get("truth_label")
        if lbl not in ("positive", "negative"):
            continue
        if lbl == "negative" and neg_filter is not None:
            if r.get("corruption_type", "") not in neg_filter:
                continue
        p = r.get("verify_p_pos")
        if p is None:
            c = r.get("verify_confidence", 0.0)
            p = c if r.get("verify_is_serial") else (1.0 - c)
        scored.append((float(p), 1 if lbl == "positive" else 0))
    if not scored:
        return [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]
    scored.sort(key=lambda x: -x[0])
    P = sum(1 for _, y in scored if y == 1)
    N = sum(1 for _, y in scored if y == 0)
    tpr_pts, fpr_pts, thr_pts = [0.0], [0.0], [1.01]
    tp = fp = 0
    prev_score = None
    for score, y in scored:
        if prev_score is not None and score != prev_score:
            tpr_pts.append(tp / max(1, P))
            fpr_pts.append(fp / max(1, N))
            thr_pts.append(prev_score)
        if y == 1: tp += 1
        else: fp += 1
        prev_score = score
    tpr_pts.append(tp / max(1, P))
    fpr_pts.append(fp / max(1, N))
    thr_pts.append(prev_score if prev_score is not None else 0.0)
    return fpr_pts, tpr_pts, thr_pts


def _auc(xs: list[float], ys: list[float]) -> float:
    pairs = sorted(zip(xs, ys))
    auc = 0.0
    for i in range(1, len(pairs)):
        x0, y0 = pairs[i - 1]; x1, y1 = pairs[i]
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return auc


def fig11_roc_curve(comp_rows, out_dir, thresh):
    err = [r for r in comp_rows if "no_errors" not in r["pdf"]
           and ("errors" in r["pdf"] or "error" in r["pdf"])]
    cases = [
        ("All corruptions", comp_rows, None, C_NEUTRAL),
        ("PII only", err, PII_CORRUPTIONS, C_BAD),
        ("Data-quality only", err, DATA_QUALITY_CORRUPTIONS, C_ACCENT),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    md_lines = ["# ROC — Verifier",
                "",
                f"Operating threshold: **{thresh:.2f}**",
                "",
                "| Subset | AUC | TPR @ saved threshold | FPR @ saved threshold |",
                "|---|---:|---:|---:|"]
    for name, rows, neg_filter, color in cases:
        fpr, tpr, thr = _roc_points(rows, neg_filter)
        auc = _auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2.0, label=f"{name} (AUC={auc:.3f})")
        c = _confusion(rows, thresh, neg_filter)
        ax.plot([c["leak"]], [c["recall"]], marker="o", color=color,
                markersize=8, markeredgecolor="black", markeredgewidth=0.7)
        md_lines.append(f"| {name} | {auc:.4f} | {c['recall']:.4f} | {c['leak']:.4f} |")
    ax.plot([0, 1], [0, 1], color=C_GREY, linestyle=":", linewidth=1, label="Random")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
    ax.set_xlabel("False positive rate (leak)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(f"Verifier ROC — operating point ●  (threshold {thresh:.2f})")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_dir / "11_roc_curve.png")
    plt.close(fig)

    md_lines += ["",
                 "Operating-point dots show the verifier's behavior at the saved",
                 "threshold; the AUC summarizes performance across all thresholds."]
    (out_dir / "11_roc_curve.md").write_text("\n".join(md_lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 12. Precision-Recall curve
# ─────────────────────────────────────────────────────────────────────────
def fig12_pr_curve(comp_rows, out_dir, thresh):
    scored = []
    for r in comp_rows:
        lbl = r.get("truth_label")
        if lbl not in ("positive", "negative"):
            continue
        p = r.get("verify_p_pos")
        if p is None:
            cc = r.get("verify_confidence", 0.0)
            p = cc if r.get("verify_is_serial") else (1.0 - cc)
        scored.append((float(p), 1 if lbl == "positive" else 0))
    if not scored:
        return
    scored.sort(key=lambda x: -x[0])
    P = sum(1 for _, y in scored if y == 1)
    tp = fp = 0
    precisions, recalls, thr_pts = [], [], []
    prev_score = None
    for s, y in scored:
        if y == 1: tp += 1
        else: fp += 1
        if prev_score is None or s != prev_score:
            precisions.append(tp / max(1, tp + fp))
            recalls.append(tp / max(1, P))
            thr_pts.append(s)
        prev_score = s
    # AP via trapezoid on (recall, precision)
    ap_pairs = sorted(zip(recalls, precisions))
    ap = 0.0
    for i in range(1, len(ap_pairs)):
        ap += (ap_pairs[i][0] - ap_pairs[i - 1][0]) * (ap_pairs[i][1] + ap_pairs[i - 1][1]) / 2.0

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(recalls, precisions, color=C_NEUTRAL, linewidth=2.2,
            label=f"Verifier  (AP={ap:.3f})")
    c_all = _confusion(comp_rows, thresh)
    ax.plot([c_all["recall"]], [c_all["precision"]], marker="o", color="black",
            markersize=9, label=f"saved threshold {thresh:.2f}")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Verifier Precision-Recall — compliance set")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "12_pr_curve.png")
    plt.close(fig)

    md = ["# Verifier Precision-Recall",
          "",
          f"- **Average precision (AP)**: {ap:.4f}",
          f"- At saved threshold **{thresh:.2f}** — precision **{c_all['precision']:.4f}**, "
          f"recall **{c_all['recall']:.4f}**, leak **{c_all['leak']:.4f}**."]
    (out_dir / "12_pr_curve.md").write_text("\n".join(md) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 13. Classifier per-year confusion matrix
# ─────────────────────────────────────────────────────────────────────────
def fig13_classifier_confusion(master_rows, out_dir):
    pages: dict[tuple, tuple] = {}
    for r in master_rows:
        pages[(r["pdf"], r["page"])] = (r.get("truth_year"), r.get("pred_year"))
    truths = sorted({ty for ty, _ in pages.values() if ty})
    preds = sorted({py for _, py in pages.values() if py})
    labels = sorted(set(truths) | set(preds))
    idx = {y: i for i, y in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for ty, py in pages.values():
        if ty and py:
            M[idx[ty], idx[py]] += 1

    row_sums = M.sum(axis=1, keepdims=True)
    M_norm = np.where(row_sums > 0, M / np.maximum(row_sums, 1), 0)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(M_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = M[i, j]
            if v == 0: continue
            color = "white" if M_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)
    ax.set_xlabel("Predicted year"); ax.set_ylabel("True year")
    correct = int(np.trace(M)); total = int(M.sum())
    ax.set_title(f"Classifier Confusion — Master Eval\n"
                 f"Page accuracy = {correct}/{total} = {100*correct/max(1, total):.2f}%")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03,
                 label="Row-normalized rate")
    fig.tight_layout()
    fig.savefig(out_dir / "13_classifier_confusion.png")
    plt.close(fig)

    md = ["# Classifier — Per-Year Confusion (Master Eval)", "",
          f"Page-level accuracy: **{correct}/{total} = {100*correct/max(1, total):.2f}%**",
          "",
          "| True year | n | Top-1 correct | Per-year accuracy |",
          "|---|---:|---:|---:|"]
    for y in labels:
        n = int(M[idx[y], :].sum())
        ok = int(M[idx[y], idx[y]])
        md.append(f"| {y} | {n} | {ok} | {100*ok/max(1,n):.2f}% |")
    (out_dir / "13_classifier_confusion.md").write_text("\n".join(md) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 14. Verifier score distribution
# ─────────────────────────────────────────────────────────────────────────
def fig14_score_distribution(comp_rows, out_dir, thresh):
    pos_scores, neg_pii, neg_dq, neg_other = [], [], [], []
    for r in comp_rows:
        lbl = r.get("truth_label")
        p = r.get("verify_p_pos")
        if p is None:
            c = r.get("verify_confidence", 0.0)
            p = c if r.get("verify_is_serial") else (1.0 - c)
        p = float(p)
        if lbl == "positive":
            pos_scores.append(p)
        elif lbl == "negative":
            ct = r.get("corruption_type", "")
            if ct in PII_CORRUPTIONS: neg_pii.append(p)
            elif ct in DATA_QUALITY_CORRUPTIONS: neg_dq.append(p)
            else: neg_other.append(p)

    bins = np.linspace(0, 1, 41)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(pos_scores, bins=bins, alpha=0.7, color=C_GOOD, label=f"Positive (n={len(pos_scores)})", edgecolor="black", linewidth=0.3)
    ax.hist(neg_pii, bins=bins, alpha=0.7, color=C_BAD, label=f"PII negative (n={len(neg_pii)})", edgecolor="black", linewidth=0.3)
    ax.hist(neg_dq, bins=bins, alpha=0.7, color=C_ACCENT, label=f"Data-quality negative (n={len(neg_dq)})", edgecolor="black", linewidth=0.3)
    if neg_other:
        ax.hist(neg_other, bins=bins, alpha=0.5, color=C_GREY, label=f"Other negative (n={len(neg_other)})", edgecolor="black", linewidth=0.3)
    ax.axvline(thresh, color="black", linestyle="--", linewidth=1.5, label=f"saved threshold {thresh:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("Verifier P(positive)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Verifier Score Distribution — by Truth Class")
    ax.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(out_dir / "14_verifier_score_distribution.png")
    plt.close(fig)

    def stats(xs: list[float]) -> str:
        if not xs: return "n/a"
        return f"median {statistics.median(xs):.3f}, mean {statistics.mean(xs):.3f}"

    md = ["# Verifier Score Distribution",
          "",
          "Distribution of verifier `p_pos` on the compliance set, broken out by truth class.",
          "",
          "| Class | n | Stats | Admitted at saved threshold |",
          "|---|---:|---|---:|"]
    for name, xs in [("Positive", pos_scores), ("PII negative", neg_pii),
                     ("DQ negative", neg_dq), ("Other negative", neg_other)]:
        adm = sum(1 for v in xs if v >= thresh)
        md.append(f"| {name} | {len(xs)} | {stats(xs)} | {adm}/{len(xs)} = {100*adm/max(1, len(xs)):.2f}% |")
    (out_dir / "14_verifier_score_distribution.md").write_text("\n".join(md) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 15. CER by year
# ─────────────────────────────────────────────────────────────────────────
def fig15_cer_by_year(master_rows, out_dir, thresh):
    by_year = defaultdict(list)
    for r in master_rows:
        if r.get("truth_label") != "positive" or not r.get("truth_serial"):
            continue
        if not _admit(r, thresh):
            continue
        cer = r.get("cer")
        if cer is None: continue
        by_year[r["truth_year"]].append(float(cer))
    years = sorted(by_year)
    if not years:
        return
    data = [by_year[y] for y in years]
    means = [statistics.mean(d) if d else 0.0 for d in data]
    medians = [statistics.median(d) if d else 0.0 for d in data]
    exact_pct = [sum(1 for v in d if v == 0.0) / max(1, len(d)) for d in data]
    counts = [len(d) for d in data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    bp = ax1.boxplot(data, positions=np.arange(len(years)),
                     widths=0.55, patch_artist=True, showmeans=False, showfliers=True,
                     medianprops=dict(color=C_BAD, linewidth=2),
                     boxprops=dict(facecolor=C_NEUTRAL, alpha=0.55, edgecolor="black"),
                     whiskerprops=dict(color="black"), capprops=dict(color="black"),
                     flierprops=dict(marker="o", markerfacecolor=C_GREY, markersize=3, linestyle="none"))
    for i, m in enumerate(means):
        ax1.plot(i, m, marker="D", color=C_GOOD, markersize=7, markeredgecolor="black")
    ax1.set_ylabel("CER (per crop)")
    ax1.set_title(f"OCR CER by Year — admitted positives (master, threshold {thresh:.2f})\n"
                  f"box = quartiles · red bar = median · green ◆ = mean")
    ax1.set_ylim(-0.02, max(1.05, max((max(d) if d else 0) for d in data) + 0.05))

    bars = ax2.bar(np.arange(len(years)), exact_pct, color=C_GOOD,
                   edgecolor="black", linewidth=0.5)
    for bar, p, n in zip(bars, exact_pct, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, p + 0.02, f"{p:.0%}",
                 ha="center", fontsize=8)
    ax2.set_ylim(0, 1.1); ax2.set_ylabel("CER = 0 (exact OCR)")
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax2.set_xticks(np.arange(len(years)))
    ax2.set_xticklabels([f"{y}\n(n={n})" for y, n in zip(years, counts)])

    fig.tight_layout()
    fig.savefig(out_dir / "15_cer_by_year.png")
    plt.close(fig)

    md = ["# OCR CER by Year (Master Eval, admitted positives)", "",
          f"Threshold: **{thresh:.2f}**",
          "",
          "| Year | n | Mean CER | Median CER | Exact-OCR rate |",
          "|---|---:|---:|---:|---:|"]
    for y, n, m, med, ep in zip(years, counts, means, medians, exact_pct):
        md.append(f"| {y} | {n} | {m:.3f} | {med:.3f} | {100*ep:.1f}% |")
    (out_dir / "15_cer_by_year.md").write_text("\n".join(md) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 16. Error decomposition: where exact-match fails
# ─────────────────────────────────────────────────────────────────────────
def fig16_error_decomposition(master_rows, out_dir, thresh):
    pos = [r for r in master_rows if r.get("truth_label") == "positive" and r.get("truth_serial")]
    n_pos = len(pos)
    cls_correct = sum(1 for r in pos if r.get("pred_year") == r.get("truth_year"))
    classifier_failures = n_pos - cls_correct
    admitted = [r for r in pos if r.get("pred_year") == r.get("truth_year") and _admit(r, thresh)]
    verifier_rejects = (n_pos - classifier_failures) - len(admitted)
    no_ocr = sum(1 for r in admitted if not r.get("ocr_normalized"))
    with_ocr = [r for r in admitted if r.get("ocr_normalized")]
    exact = sum(1 for r in with_ocr if r.get("ocr_normalized") == r.get("truth_serial"))
    near_miss = sum(1 for r in with_ocr
                    if r.get("ocr_normalized") != r.get("truth_serial")
                    and (r.get("cer_best_hyp") is not None and r["cer_best_hyp"] <= 0.20))
    far_miss = (len(with_ocr) - exact) - near_miss

    cats = [
        ("Exact match", exact, C_GOOD),
        ("OCR near-miss\n(best hyp CER ≤ 0.20)", near_miss, "#9CCC65"),
        ("OCR far-miss\n(CER > 0.20)", far_miss, C_ACCENT),
        ("Verifier admitted,\nno OCR output", no_ocr, "#FFB74D"),
        ("Verifier rejected", verifier_rejects, C_BAD),
        ("Classifier wrong year", classifier_failures, C_GREY),
    ]
    counts = [c[1] for c in cats]
    labels = [c[0] for c in cats]
    colors = [c[2] for c in cats]
    pcts = [v / max(1, n_pos) for v in counts]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(labels, counts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, n, p in zip(bars, counts, pcts):
        ax.text(n + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{n}  ({100*p:.1f}%)", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.20)
    ax.set_xlabel(f"Count (of {n_pos} positive firearms)")
    ax.set_title(f"End-to-End Error Decomposition — Master Eval (threshold {thresh:.2f})")
    fig.tight_layout()
    fig.savefig(out_dir / "16_error_decomposition.png")
    plt.close(fig)

    md = ["# End-to-End Error Decomposition", "",
          f"Master held-out, **{n_pos} positive firearms**, threshold **{thresh:.2f}**.",
          "",
          "Where each positive ends up in the pipeline (mutually exclusive):",
          "",
          "| Outcome | Count | % of positives |",
          "|---|---:|---:|"]
    for name, n, _ in cats:
        md.append(f"| {name.replace(chr(10), ' ')} | {n} | {100*n/max(1, n_pos):.2f}% |")
    md.append(f"| **Total** | **{sum(counts)}** | **{100*sum(counts)/max(1, n_pos):.1f}%** |")
    (out_dir / "16_error_decomposition.md").write_text("\n".join(md) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# 17. Combined stage table (master + compliance)
# ─────────────────────────────────────────────────────────────────────────
def fig17_combined_stage_table(comp_rows, master_rows, out_dir, thresh):
    def stage_block(rows, name):
        pos = [r for r in rows if r.get("truth_label") == "positive" and r.get("truth_serial")]
        neg = [r for r in rows if r.get("truth_label") == "negative"]
        pages = {(r["pdf"], r["page"]): (r.get("truth_year"), r.get("pred_year")) for r in rows}
        cls_ok = sum(1 for ty, py in pages.values() if ty and py and ty == py)
        n_admit = sum(1 for r in pos if _admit(r, thresh))
        n_ocr = sum(1 for r in pos if _admit(r, thresh) and r.get("ocr_normalized"))
        n_exact = sum(1 for r in pos if _admit(r, thresh) and r.get("ocr_normalized") == r.get("truth_serial"))
        leak = sum(1 for r in neg if _admit(r, thresh)) / max(1, len(neg)) if neg else float("nan")
        return {
            "name": name,
            "n_pages": len(pages),
            "cls_acc": cls_ok / max(1, len(pages)),
            "n_pos": len(pos),
            "n_neg": len(neg),
            "admit_pct": n_admit / max(1, len(pos)) if pos else 0.0,
            "ocr_pct": n_ocr / max(1, len(pos)) if pos else 0.0,
            "exact_pct": n_exact / max(1, len(pos)) if pos else 0.0,
            "leak": leak,
        }

    blocks = [
        stage_block(master_rows, "Master held-out"),
        stage_block(comp_rows, "Compliance (all)"),
        stage_block([r for r in comp_rows if "no_errors" in r["pdf"]], "Compliance — clean only"),
        stage_block([r for r in comp_rows if "no_errors" not in r["pdf"]
                     and ("errors" in r["pdf"] or "error" in r["pdf"])],
                    "Compliance — corrupted only"),
    ]

    md = ["# End-to-End Stage Results — Combined", "",
          f"Verifier operating threshold: **{thresh:.2f}**", "",
          "| Subset | Pages | Classifier acc | Positives | Admit% | OCR% | Exact% | Negatives | Leak% |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for b in blocks:
        leak_s = f"{100*b['leak']:.2f}%" if b["leak"] == b["leak"] else "—"
        md.append(f"| {b['name']} | {b['n_pages']} | {100*b['cls_acc']:.2f}% | "
                  f"{b['n_pos']} | {100*b['admit_pct']:.1f}% | {100*b['ocr_pct']:.1f}% | "
                  f"{100*b['exact_pct']:.1f}% | {b['n_neg']} | {leak_s} |")
    md += ["",
           "**Definitions** — Admit% = verifier admits a true positive; OCR% = "
           "admitted *and* OCR produced output; Exact% = OCR matched truth exactly; "
           "Leak% = negatives admitted (lower better, especially on compliance corrupted)."]
    (out_dir / "17_combined_stage_table.md").write_text("\n".join(md) + "\n")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compliance-json", default="/tmp/eval_v2_compliance_v7.json")
    ap.add_argument("--master-json", default="/tmp/eval_master_v7.json")
    ap.add_argument("--out-dir", default="report_figures", type=Path)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    print(f"Reading {args.compliance_json}, {args.master_json}")
    comp = json.load(open(args.compliance_json))
    master = json.load(open(args.master_json))
    comp_rows = comp["rows"]
    master_rows = master["rows"]
    print(f"Compliance: {len(comp_rows)} rows. Master: {len(master_rows)} rows.")

    # Backfill corruption_type if needed
    from pipeline.report_compliance import _backfill_corruption_type
    n1 = _backfill_corruption_type(comp_rows)
    n2 = _backfill_corruption_type(master_rows)
    if n1 + n2:
        print(f"Backfilled corruption_type for {n1+n2} rows")

    print(f"Generating figures at threshold {args.threshold:.2f}...")
    fig01_top_level(comp_rows, master_rows, out_dir, args.threshold)
    fig02_per_corruption(comp_rows, out_dir, args.threshold)
    fig03_stage_results(master_rows, out_dir, args.threshold)
    fig04_topk(master_rows, out_dir, args.threshold)
    fig05_confusion(comp_rows, out_dir, args.threshold)
    fig06_per_year(comp_rows, out_dir, args.threshold)
    fig07_threshold_tradeoffs(comp_rows, out_dir, args.threshold)
    fig08_by_doc_type(master_rows, out_dir, args.threshold)
    fig09_safe_vs_unsafe(comp_rows, out_dir, args.threshold)
    fig10_cer_distribution(master_rows, out_dir, args.threshold)
    fig11_roc_curve(comp_rows, out_dir, args.threshold)
    fig12_pr_curve(comp_rows, out_dir, args.threshold)
    fig13_classifier_confusion(master_rows, out_dir)
    fig14_score_distribution(comp_rows, out_dir, args.threshold)
    fig15_cer_by_year(master_rows, out_dir, args.threshold)
    fig16_error_decomposition(master_rows, out_dir, args.threshold)
    fig17_combined_stage_table(comp_rows, master_rows, out_dir, args.threshold)

    print(f"\nWrote figures + tables to {out_dir}/")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
