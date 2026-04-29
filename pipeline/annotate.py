"""Interactive serial-block annotator.

Opens a template PDF (``FormTemplates/<year>/Form.pdf``), renders the
serial-bearing page, and lets you click-drag a rectangle over the serial
column block. Writes the box to ``form_config.json["serial_block"]`` in
**points** (same units as ``page_size``).

Controls (per window):
    left click + drag   — draw a new box
    s                   — save and advance to next year
    n                   — skip (don't save, advance to next year)
    r                   — reset (clear current box)
    q                   — quit the whole session

Usage::

    python -m pipeline.annotate 2022                 # one year
    python -m pipeline.annotate --all                # every year under FormTemplates/
    python -m pipeline.annotate --all --skip-done    # only years without serial_block
    python -m pipeline.annotate 1985 --template-file Continuation.pdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

# Release keys we bind ourselves before anything reads rcParams.
# Matplotlib's defaults grab s (save fig), q (quit), r (home), f (fullscreen)…
for _rc in ("keymap.save", "keymap.quit", "keymap.home", "keymap.fullscreen",
            "keymap.back", "keymap.forward", "keymap.pan", "keymap.zoom",
            "keymap.grid", "keymap.yscale", "keymap.xscale"):
    matplotlib.rcParams[_rc] = []

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.widgets import RectangleSelector  # noqa: E402
from pdf2image import convert_from_path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FORM_TEMPLATES = ROOT / "FormTemplates"

_RENDER_DPI = 150


def _load_cfg(year: str) -> tuple[Path, dict]:
    cfg_path = FORM_TEMPLATES / year / "form_config.json"
    if not cfg_path.is_file():
        print(f"Missing {cfg_path}", file=sys.stderr)
        sys.exit(1)
    with open(cfg_path, encoding="utf-8") as f:
        return cfg_path, json.load(f)


def _save_cfg(cfg_path: Path, cfg: dict) -> None:
    tmp = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    tmp.replace(cfg_path)


def annotate(
    year: str,
    template_file: str = "Form.pdf",
    *,
    position: str = "",
) -> str:
    """Annotate one year. Returns ``"saved"``, ``"skipped"``, or ``"quit"``."""
    cfg_path, cfg = _load_cfg(year)
    pdf_path = FORM_TEMPLATES / year / template_file
    if not pdf_path.is_file():
        print(f"Missing template PDF {pdf_path}", file=sys.stderr)
        return "skipped"

    page_w_pt, page_h_pt = (cfg.get("page_size") or [612, 792])[:2]
    page_0based = int((cfg.get("firearm_rows") or {}).get("page", 0))
    existing = cfg.get("serial_block")

    pages = convert_from_path(
        str(pdf_path),
        dpi=_RENDER_DPI,
        first_page=page_0based + 1,
        last_page=page_0based + 1,
    )
    if not pages:
        print(f"Could not render page {page_0based} of {pdf_path}", file=sys.stderr)
        return "skipped"
    img = pages[0]
    iw, ih = img.size
    sx = iw / page_w_pt
    sy = ih / page_h_pt

    fig, ax = plt.subplots(figsize=(10, 13))
    fig.canvas.manager.set_window_title(
        f"Annotate serial block — {year} ({template_file}, page {page_0based})"
    )
    ax.imshow(img)
    pos_suffix = f"  {position}" if position else ""
    ax.set_title(
        f"{year}{pos_suffix}: drag a box over the serial column block.  "
        f"[s]ave  [n]ext  [r]eset  [q]uit"
    )
    ax.set_axis_off()

    # Show the existing box (if any) in a different color
    if existing:
        L = existing["x"] * sx
        T = existing["y"] * sy
        W = existing["width"] * sx
        H = existing["height"] * sy
        ax.add_patch(
            Rectangle(
                (L, T), W, H,
                linewidth=1.5, edgecolor="orange", facecolor="none",
                linestyle="--", label="existing",
            )
        )

    # Mutable container so the selector callback can update it.
    # "outcome" is what we return to the caller loop.
    state: dict = {"box_px": None, "patch": None, "outcome": "skipped"}

    def on_select(eclick, erelease):
        x0, y0 = sorted([eclick.xdata, erelease.xdata])
        y_a, y_b = sorted([eclick.ydata, erelease.ydata])
        L, T = x0, y_a
        W, H = y0 - x0, y_b - y_a
        # Remove previous patch
        if state["patch"] is not None:
            state["patch"].remove()
        patch = Rectangle(
            (L, T), W, H,
            linewidth=2, edgecolor="red", facecolor="none",
        )
        ax.add_patch(patch)
        state["patch"] = patch
        state["box_px"] = (L, T, W, H)

        x_pt = L / sx
        y_pt = T / sy
        w_pt = W / sx
        h_pt = H / sy
        ax.set_title(
            f"{year}: box = x={x_pt:.1f} y={y_pt:.1f} "
            f"w={w_pt:.1f} h={h_pt:.1f} pt  (press s to save)"
        )
        fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="data",
        interactive=True,
    )

    def on_key(event):
        if event.key == "q":
            state["outcome"] = "quit"
            plt.close(fig)
        elif event.key == "n":
            state["outcome"] = "skipped"
            plt.close(fig)
        elif event.key == "r":
            if state["patch"] is not None:
                state["patch"].remove()
                state["patch"] = None
            state["box_px"] = None
            ax.set_title(f"{year}: reset — draw a new box")
            fig.canvas.draw_idle()
        elif event.key == "s":
            if state["box_px"] is None:
                print("No box drawn yet. Drag a rectangle first.")
                return
            L, T, W, H = state["box_px"]
            cfg["serial_block"] = {
                "page": page_0based,
                "x": round(L / sx, 2),
                "y": round(T / sy, 2),
                "width": round(W / sx, 2),
                "height": round(H / sy, 2),
            }
            _save_cfg(cfg_path, cfg)
            print(f"Saved serial_block to {cfg_path}:")
            print(f"  {cfg['serial_block']}")
            state["outcome"] = "saved"
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()

    # Keep selector alive until window closes
    _ = selector
    return state["outcome"]


def _list_years() -> list[str]:
    return sorted(
        d.name for d in FORM_TEMPLATES.iterdir()
        if d.is_dir() and (d / "form_config.json").is_file()
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "year",
        nargs="?",
        help="Year folder under FormTemplates/ (e.g. 2022). Omit with --all.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Iterate every year under FormTemplates/ in order.",
    )
    p.add_argument(
        "--skip-done",
        action="store_true",
        help="With --all, skip years whose form_config.json already has serial_block.",
    )
    p.add_argument(
        "--template-file",
        default="Form.pdf",
        help="Which template PDF to annotate (default: Form.pdf)",
    )
    args = p.parse_args()

    if args.all:
        years = _list_years()
        if args.skip_done:
            remaining = []
            for y in years:
                _, cfg = _load_cfg(y)
                if "serial_block" not in cfg:
                    remaining.append(y)
            years = remaining
        if not years:
            print("Nothing to annotate.")
            return 0

        print(f"Annotating {len(years)} year(s): {', '.join(years)}")
        saved = skipped = 0
        for i, y in enumerate(years, 1):
            outcome = annotate(
                y,
                args.template_file,
                position=f"[{i}/{len(years)}]",
            )
            if outcome == "saved":
                saved += 1
            elif outcome == "skipped":
                skipped += 1
            elif outcome == "quit":
                print(f"Quit after {i - 1} year(s).")
                break
        print(f"Done. saved={saved}  skipped={skipped}")
        return 0

    if not args.year:
        p.error("Provide a year or --all.")
    outcome = annotate(args.year, args.template_file)
    return 0 if outcome in ("saved", "skipped") else 0


if __name__ == "__main__":
    raise SystemExit(main())
