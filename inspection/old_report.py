"""Generate the OLD reports: combined train report + per-camera reports.

The combined report is old_code's own `reporting/combined_train_report.py`, called
UNCHANGED. Its layout, KPI grid, 10-column wagon table, anomaly row tinting, video
URL table and JSON view-model are therefore preserved exactly -- the only thing
that changed is the identity in the GW_n column, which is now the current global
wagon id instead of a camera-local number.

The per-camera reports are new *plumbing* around the same view model: old_code
shipped the combined report only, so rather than invent a different style, each
camera report reuses the identical reportlab idiom, KPI framing, table styling and
status vocabulary, filtered to what one camera actually observed. That keeps a
reviewer's mental model constant across the two documents.

Nothing here runs a model. Both reports read the persisted `UnifiedWagonState`
view, so a report can never disagree with the processors or with the JSON.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Sequence

from core import constants as C
from core.unified_wagon_state import UnifiedWagonState, summarize_wagons

__all__ = ["build_combined_report", "build_camera_reports", "build_all_reports"]


def build_combined_report(
    state: Any,
    unified: Dict[str, UnifiedWagonState],
    output_dir: str,
    batch_key: str,
    source_video_urls: Optional[Dict[str, str]] = None,
    processed_video_urls: Optional[Dict[str, str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Optional[str]]:
    """Call old_code's combined report verbatim.

    Returns {"json_path", "pdf_path"}; pdf_path is None if reportlab failed, which
    old_code already handles by writing the JSON regardless.
    """
    from features.reporting import combined_train_report as old_report

    return old_report.build(
        state=state, unified=unified, output_dir=output_dir,
        batch_key=batch_key,
        source_video_urls=source_video_urls,
        processed_video_urls=processed_video_urls,
        extra_metadata=extra_metadata, verbose=verbose)


# ---------------------------------------------------------------------------
# per-camera reports
# ---------------------------------------------------------------------------

_CAMERA_FEATURES: Dict[str, List[str]] = {
    C.CAMERA_RIGHT_UP: ["right_door"],
    C.CAMERA_LEFT_UP: ["left_door"],
    C.CAMERA_RIGHT_UP_TOP: ["load", "top_damage"],
    C.CAMERA_LEFT_UP_TOP: ["load", "top_damage"],
}
"""Which features each camera is responsible for, preserving old_code's camera
authority: RIGHT_UP -> right door, LEFT_UP -> left door, top cameras -> load and
top damage (RIGHT_UP_TOP authoritative)."""


def _camera_rows(cam: str, wagons: Sequence[UnifiedWagonState]) -> List[List[str]]:
    """One row per global wagon -- every wagon appears, findings or not."""
    rows: List[List[str]] = []
    for i, u in enumerate(wagons, start=1):
        status_map = (u.camera_status.get(cam) or {})
        observed = next(iter(status_map.values()), "UNRESOLVED")
        cells = [str(i), u.global_id, u.classification]
        for feat in _CAMERA_FEATURES.get(cam, []):
            if feat == "right_door":
                cells += [u.right_door, f"{u.right_door_confidence:.2f}"]
            elif feat == "left_door":
                cells += [u.left_door, f"{u.left_door_confidence:.2f}"]
            elif feat == "load":
                cells += [u.load_status, f"{u.load_confidence:.2f}"]
            elif feat == "top_damage":
                cells += [u.top_damage, f"{u.top_damage_confidence:.2f}"]
        cells.append(observed)
        rows.append(cells)
    return rows


def _camera_headers(cam: str) -> List[str]:
    head = ["SR", "GW_n", "CLASS"]
    for feat in _CAMERA_FEATURES.get(cam, []):
        head += {"right_door": ["R_DOOR", "CONF"],
                 "left_door": ["L_DOOR", "CONF"],
                 "load": ["LOAD", "CONF"],
                 "top_damage": ["TOP_DMG", "CONF"]}[feat]
    head.append("OBSERVED")
    return head


def build_camera_reports(
    state: Any,
    unified: Dict[str, UnifiedWagonState],
    output_dir: str,
    batch_key: str,
    camera_status: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Dict[str, Optional[str]]:
    """One PDF per camera, in the same visual idiom as the combined report."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    os.makedirs(output_dir, exist_ok=True)
    camera_status = camera_status or {}
    wagons = [unified[w.global_id] for w in state.wagons
              if w.global_id in unified]

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=15,
                        textColor=colors.HexColor("#0d2c54"))
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8)
    note = ParagraphStyle("Note", parent=styles["Normal"], fontSize=7,
                          textColor=colors.grey)

    out: Dict[str, Optional[str]] = {}
    for cam in C.ALL_CAMERAS:
        path = os.path.join(output_dir, f"camera_report_{cam}.pdf")
        try:
            doc = SimpleDocTemplate(path, pagesize=landscape(A4),
                                    leftMargin=10 * mm, rightMargin=10 * mm,
                                    topMargin=10 * mm, bottomMargin=10 * mm)
            el: List[Any] = []
            el.append(Paragraph(
                f"WagonEye Camera Report &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>{cam}</b> &nbsp;&nbsp;|&nbsp;&nbsp; Batch <b>{batch_key}</b>",
                h1))
            el.append(Paragraph(time.strftime("%Y-%m-%d %H:%M:%S"), small))
            el.append(Spacer(1, 3 * mm))

            responsible = ", ".join(_CAMERA_FEATURES.get(cam, [])) or "none"
            el.append(Paragraph(
                f"Responsible for: <b>{responsible}</b> &nbsp;·&nbsp; "
                f"synchronization: <b>{camera_status.get(cam, 'UNKNOWN')}</b> "
                f"&nbsp;·&nbsp; global wagons: <b>{len(wagons)}</b>", small))
            el.append(Spacer(1, 4 * mm))

            headers = _camera_headers(cam)
            rows = [headers] + _camera_rows(cam, wagons)
            row_styles: List[tuple] = []
            for idx, u in enumerate(wagons, start=1):
                if u.has_open_door or u.has_damage:
                    row_styles.append(("BACKGROUND", (0, idx), (-1, idx),
                                       colors.HexColor("#fde2e2")))
                elif u.classification == C.CLASS_ENGINE:
                    row_styles.append(("BACKGROUND", (0, idx), (-1, idx),
                                       colors.HexColor("#fff4e0")))
                elif u.classification == C.CLASS_BRAKE_VAN:
                    row_styles.append(("BACKGROUND", (0, idx), (-1, idx),
                                       colors.HexColor("#e8f0fe")))
            table = Table(rows, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d2c54")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ] + row_styles))
            el.append(table)
            el.append(Spacer(1, 4 * mm))
            el.append(Paragraph(
                "OBSERVED: INSPECTED = this camera contributed evidence · "
                "NO_DETECTION = inspected, nothing found · "
                "NOT_VISIBLE = wagon outside this camera's footage · "
                "UNRESOLVED = camera clock offset never resolved, so nothing here "
                "is attributed to a wagon. NOT_VISIBLE and UNRESOLVED are NOT "
                "evidence of a clean wagon.", note))
            doc.build(el)
            out[cam] = path
            if verbose:
                print(f"  [REPORT/{cam}] wrote {path}")
        except Exception as exc:
            out[cam] = None
            if verbose:
                print(f"  [REPORT/{cam}] FAILED: {type(exc).__name__}: {exc}")
    return out


def build_all_reports(
    state: Any,
    unified: Dict[str, UnifiedWagonState],
    output_root: str,
    batch_key: str,
    camera_status: Optional[Dict[str, str]] = None,
    source_video_urls: Optional[Dict[str, str]] = None,
    processed_video_urls: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Combined report + four camera reports, all from the persisted view."""
    report_dir = os.path.join(output_root, "reports")
    os.makedirs(report_dir, exist_ok=True)
    combined = build_combined_report(
        state, unified, report_dir, batch_key,
        source_video_urls=source_video_urls,
        processed_video_urls=processed_video_urls,
        extra_metadata={"inspection_source": "old_code door/load/damage",
                        "ocr": "REMOVED_FROM_SCOPE"},
        verbose=verbose)
    cameras = build_camera_reports(state, unified, report_dir, batch_key,
                                  camera_status=camera_status, verbose=verbose)
    return {"combined": combined, "cameras": cameras,
            "report_dir": report_dir,
            "summary": summarize_wagons(list(unified.values()))}
