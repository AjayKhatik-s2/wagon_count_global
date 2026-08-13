"""Stage 5 -- emit `combined_train_report.json` + `combined_train_report.pdf`.

Implemented FROM SCRATCH with reportlab; no dependency on the legacy
`RIGHT_UP/combined_report_generator.py` or `TrainSession`.

PDF structure:
    Page 1
        ─ Title banner + batch metadata
        ─ Train summary table (counts + anomaly counts + processing time)
        ─ Source / Processed video URL table (4 rows, RAW + PROCESSED columns)
    Page 2..N
        ─ Wagon inspection table.  One row per GW_n.
          Columns: SR | GW | CLASS | OCR | L_DOOR | R_DOOR | LOAD
                   | TOP_DMG | SIDE_DMG | CONF
          Rows highlighted RED when has_open_door OR has_damage.
    Final page
        ─ Anomaly list (one line per flagged wagon).

The JSON schema is the canonical machine-readable mirror of the same
content; downstream consumers should prefer it.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from core import constants as C
from core.global_state_loader import GlobalTrainState
from core.unified_wagon_state import UnifiedWagonState, summarize_wagons


_REPORT_SCHEMA = "wagon_eye.combined_report.v2"


# -----------------------------------------------------------------------------
# Time / formatting helpers
# -----------------------------------------------------------------------------

def _now_ist_iso() -> str:
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).isoformat(timespec="seconds")


def _now_ist_human() -> str:
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).strftime("%d-%m-%Y %H:%M IST")


def _short(value: str, n: int = 14) -> str:
    s = str(value or "")
    return s if len(s) <= n else s[: n - 1] + "…"


# -----------------------------------------------------------------------------
# JSON
# -----------------------------------------------------------------------------

def _build_json(
    *,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    batch_key: str,
    source_video_urls: Optional[Dict[str, str]] = None,
    processed_video_urls: Optional[Dict[str, str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    wagons_in_order = [unified[w.global_id]
                       for w in state.wagons
                       if w.global_id in unified]
    summary = summarize_wagons(wagons_in_order)
    doc: Dict[str, Any] = {
        "schema": _REPORT_SCHEMA,
        "batch_key": batch_key,
        "generated_at": _now_ist_iso(),
        "train_metadata": {
            "master_camera":       state.master_camera,
            "master_fps":          state.master_fps,
            "master_total_frames": state.master_total_frames,
            "source_video_urls":   dict(source_video_urls or {}),
            "processed_video_urls":dict(processed_video_urls or {}),
        },
        "summary": summary,
        "stage0_fallback_used":    state.fallback_used,
        "stage0_fallback_reason":  state.fallback_reason,
        "stage0_corrections_applied": list(state.corrections_applied),
        "per_camera_local_counts": dict(state.per_camera_local_counts),
        "wagons": [u.to_dict() for u in wagons_in_order],
    }
    if extra_metadata:
        doc["train_metadata"].update(extra_metadata)
    return doc


# -----------------------------------------------------------------------------
# PDF (reportlab)
# -----------------------------------------------------------------------------

def _build_pdf(
    *,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    batch_key: str,
    output_pdf: str,
    source_video_urls: Optional[Dict[str, str]] = None,
    processed_video_urls: Optional[Dict[str, str]] = None,
) -> str:
    """Render the PDF. Raises if reportlab is missing or rendering fails."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)) or ".", exist_ok=True)

    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=landscape(A4),
        leftMargin=10 * mm, rightMargin=10 * mm,
        topMargin=10 * mm,  bottomMargin=10 * mm,
        title=f"WagonEye Combined Report -- {batch_key}",
        author="WagonEye",
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"],
                        fontSize=18, textColor=colors.HexColor("#0d2c54"),
                        spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                        fontSize=12, textColor=colors.HexColor("#0d2c54"),
                        spaceBefore=8, spaceAfter=4)
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8)
    note  = ParagraphStyle("Note", parent=styles["Normal"], fontSize=7,
                            textColor=colors.grey)

    elements: List[Any] = []
    wagons_in_order = [unified[w.global_id]
                       for w in state.wagons
                       if w.global_id in unified]
    summary = summarize_wagons(wagons_in_order)

    # ---- Header banner ----
    elements.append(Paragraph(
        f"WagonEye Combined Train Report &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Batch <b>{batch_key}</b>", h1))
    elements.append(Paragraph(_now_ist_human(), small))
    elements.append(Spacer(1, 4 * mm))

    # ---- Train summary table ----
    summary_rows = [
        ["TOTAL WAGONS", summary["total_wagons"],
         "ENGINES", summary["engine_count"],
         "BRAKE VANS", summary["brake_van_count"]],
        ["REGULAR WAGONS", summary["wagon_count"],
         "OCR CAPTURED", summary["ocr_captured"],
         "MASTER CAMERA", state.master_camera],
        ["L DOORS OPEN", summary["left_doors_open"],
         "R DOORS OPEN", summary["right_doors_open"],
         "ANOMALIES", sum(len(u.anomalies) for u in wagons_in_order)],
        ["LOADED", summary["loaded"],
         "EMPTY", summary["empty"],
         "TOP DAMAGED", summary["top_damaged"]],
    ]
    summary_table = Table(summary_rows, hAlign="LEFT",
                          colWidths=[35 * mm, 25 * mm,
                                     35 * mm, 25 * mm,
                                     35 * mm, 35 * mm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f6fb")),
        ("BOX",        (0, 0), (-1, -1), 0.5, colors.HexColor("#0d2c54")),
        ("INNERGRID",  (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("FONTNAME",   (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",(0, 0), (-1, -1), 6),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",   (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTNAME",   (4, 0), (4, -1), "Helvetica-Bold"),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 6 * mm))

    # ---- Video URL table (if any URLs provided) ----
    src  = source_video_urls or {}
    proc = processed_video_urls or {}
    if any(src.values()) or any(proc.values()):
        elements.append(Paragraph("Video Evidence", h2))
        url_rows = [["CAMERA", "RAW VIDEO", "PROCESSED VIDEO"]]
        for cam in C.ALL_CAMERAS:
            url_rows.append([cam, _short(src.get(cam, ""), 60),
                                  _short(proc.get(cam, ""), 60)])
        url_table = Table(url_rows, hAlign="LEFT",
                          colWidths=[35 * mm, 100 * mm, 100 * mm])
        url_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d2c54")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("INNERGRID",  (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("BOX",        (0, 0), (-1, -1), 0.5, colors.HexColor("#0d2c54")),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ]))
        elements.append(url_table)
        elements.append(Spacer(1, 4 * mm))

    elements.append(PageBreak())

    # ---- Wagon table ----
    elements.append(Paragraph("Wagon Inspection", h1))
    elements.append(Spacer(1, 3 * mm))

    header = ["SR", "GW_n", "CLASS", "OCR",
              "L_DOOR", "R_DOOR", "LOAD",
              "TOP_DMG", "SIDE_DMG", "CONF"]
    rows: List[List[str]] = [header]
    row_styles: List[tuple] = []
    for i, u in enumerate(wagons_in_order, start=1):
        row_idx = i  # index inside `rows` (header is row 0)
        rows.append([
            str(i),
            u.global_id,
            u.classification,
            _short(u.wagon_identifier, 16),
            u.left_door,
            u.right_door,
            u.load_status,
            u.top_damage,
            u.side_damage,
            f"{u.confidence:.2f}",
        ])
        if u.has_open_door or u.has_damage:
            row_styles.append(("BACKGROUND",
                               (0, row_idx), (-1, row_idx),
                               colors.HexColor("#fde2e2")))
        elif u.classification == C.CLASS_ENGINE:
            row_styles.append(("BACKGROUND",
                               (0, row_idx), (-1, row_idx),
                               colors.HexColor("#fff4e0")))
        elif u.classification == C.CLASS_BRAKE_VAN:
            row_styles.append(("BACKGROUND",
                               (0, row_idx), (-1, row_idx),
                               colors.HexColor("#e8f0fe")))

    wagon_table = Table(
        rows, repeatRows=1, hAlign="LEFT",
        colWidths=[10 * mm, 18 * mm, 22 * mm, 36 * mm,
                   20 * mm, 20 * mm, 20 * mm,
                   22 * mm, 22 * mm, 15 * mm],
    )
    base_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d2c54")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("FONTSIZE",   (0, 1), (-1, -1), 8),
        ("INNERGRID",  (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("BOX",        (0, 0), (-1, -1), 0.5, colors.HexColor("#0d2c54")),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN",      (0, 0), (0, -1), "RIGHT"),
        ("ALIGN",      (-1, 0), (-1, -1), "RIGHT"),
    ]
    wagon_table.setStyle(TableStyle(base_style + row_styles))
    elements.append(wagon_table)

    # ---- Anomaly list ----
    anomalies = [(u.global_id, u.anomalies) for u in wagons_in_order if u.anomalies]
    if anomalies:
        elements.append(PageBreak())
        elements.append(Paragraph("Flagged Wagons", h1))
        for gw_id, anom in anomalies:
            elements.append(Paragraph(
                f"<b>{gw_id}</b> — " + ", ".join(anom),
                small,
            ))
            elements.append(Spacer(1, 1 * mm))

    # ---- Footer note ----
    elements.append(Spacer(1, 4 * mm))
    elements.append(Paragraph(
        f"Generated by WagonEye v4. Schema {_REPORT_SCHEMA}. "
        f"Stage-0 fallback: {state.fallback_used}.",
        note,
    ))

    doc.build(elements)
    return output_pdf


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def build(
    *,
    state: GlobalTrainState,
    unified: Dict[str, UnifiedWagonState],
    output_dir: str,
    batch_key: str,
    source_video_urls: Optional[Dict[str, str]] = None,
    processed_video_urls: Optional[Dict[str, str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Optional[str]]:
    """Stage 5 public entry.  Writes JSON always; PDF if reportlab is OK.

    Returns:
        {"json_path": "...", "pdf_path": "..." | None}
    """
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    json_doc = _build_json(
        state=state, unified=unified, batch_key=batch_key,
        source_video_urls=source_video_urls,
        processed_video_urls=processed_video_urls,
        extra_metadata=extra_metadata,
    )
    json_path = os.path.join(output_dir, "combined_train_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_doc, f, indent=2, default=str)
    if verbose:
        print(f"[STAGE5] wrote {json_path}")

    pdf_path: Optional[str] = os.path.join(output_dir, "combined_train_report.pdf")
    try:
        _build_pdf(
            state=state, unified=unified, batch_key=batch_key,
            output_pdf=pdf_path,
            source_video_urls=source_video_urls,
            processed_video_urls=processed_video_urls,
        )
        if verbose:
            print(f"[STAGE5] wrote {pdf_path}")
    except Exception as e:
        import traceback
        print(f"[STAGE5] PDF generation FAILED: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        pdf_path = None

    if verbose:
        print(f"[STAGE5] done in {time.time() - t0:.1f}s")
    return {"json_path": json_path, "pdf_path": pdf_path}
