from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class RowResult:
    row_index: int
    box_in_block: tuple[int, int, int, int] | None = None  # px, relative to block

    verify_status: str = "pending"   # ok | empty | bleed_top | bleed_bottom | multi_band | too_dense
    verify_confidence: float = 0.0

    serial_raw: str = ""
    serial_normalized: str = ""
    looks_serial: bool = False
    ocr_method: str = ""

    @property
    def admitted(self) -> bool:
        """True when this row contributes to the search index."""
        return (
            self.verify_status == "ok"
            and bool(self.serial_normalized)
            and self.looks_serial
        )


@dataclass
class PipelineResult:
    pdf_path: Path

    page_image: Image.Image | None = None
    page_0based: int = 0

    predicted_year: str | None = None
    classifier_score: float = 0.0
    classifier_status: str = "PENDING"

    block_crop: Image.Image | None = None
    block_box_pts: tuple[float, float, float, float] | None = None

    rows: list[RowResult] = field(default_factory=list)

    status: str = "PENDING"
    error: str | None = None

    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def admitted_serials(self) -> list[str]:
        return [r.serial_normalized for r in self.rows if r.admitted]
