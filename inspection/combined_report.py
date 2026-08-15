"""The COMBINED WAGON EYE REPORT -- the production layout, rebuilt exactly.

    page 1      banner, VIDEO EVIDENCE, DETAILED REPORTS, INSPECTION SUMMARY
    pages 2..k  WAGON INSPECTION DETAILS -- one row per GLOBAL wagon
    pages k..n  Damaged Wagon Report -- per-wagon evidence panels

WHERE THE NUMBERS COME FROM
---------------------------
Nothing here detects anything. Every cell is read from what the inspection
stage already persisted:

    <camera>/inspection_data.internal.json   verdicts + global wagon identity
    <camera>/damage_results.csv              per-class BANDS (see below)
    <camera>/problem_frames.csv              local evidence image paths

That is what keeps the PDF, the dashboard JSON and the processed videos in
agreement: none of them is allowed to form its own opinion. A model import in
this module would break that, and a test asserts there isn't one.

"DOOR 1 CLOSED / DOOR 2 PARTIAL CLOSED"
---------------------------------------
The legacy JSON carries one fused ``door_status`` per wagon per camera, which
cannot express two doors in different states. The per-door detail in this report
comes from the BANDS instead: ``analyze_detection_bands`` groups a class's
detections into temporally separate runs, so a wagon whose left side shows a
closed door and then a partly-closed one produces two bands. Sorted by
``start_frame`` and numbered, those bands ARE "DOOR 1" and "DOOR 2" -- the same
evidence the fused status was voted from, presented per instance rather than
collapsed.

A wagon with no door band at all reads "NO DOOR DETECTED", which is deliberately
not "CLOSED": the legacy report distinguishes "nothing was found" from "a closed
door was found", and so does this one.

THE ROW SET IS THE GLOBAL ROSTER
--------------------------------
SR.NO runs 1..N over ``state.wagons``, so the table has exactly as many rows as
the global wagon count -- whatever any camera saw. A camera missing a wagon
prints NO DOOR DETECTED / NOT VISIBLE in its column; it can never shorten the
report.

OCR IS DISABLED, and the columns that displayed it are preserved: WAGON NUMBER
and LOCO NUMBER render "-" for every row, exactly as the legacy report does for
a wagon whose number was not read.
"""

from __future__ import annotations

import ast
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C

from . import global_bridge as gb

__all__ = ["build_combined_report_pdf", "wagon_report_rows", "summary_row",
           "damaged_wagon_entries", "door_cell_text", "CAMERA_ORDER",
           "CAMERA_DISPLAY"]


# ---------------------------------------------------------------------------
# palette + vocabulary (matched to the production report)
# ---------------------------------------------------------------------------

NAVY = "#1a2f52"
TEAL = "#1a8f7a"
SUBHEAD = "#e9ecef"
ROW_ALT = "#f7f8fa"
PINK = "#fdeaea"
BLUE_TXT = "#1a5fb4"
BLUE_BG = "#e8f0fe"
ORANGE = "#e8710a"
RED = "#d93025"
GREY = "#9aa0a6"

CAMERA_ORDER: Tuple[str, ...] = (
    C.CAMERA_LEFT_UP, C.CAMERA_RIGHT_UP, C.CAMERA_RIGHT_UP_TOP,
    C.CAMERA_LEFT_UP_TOP,
)
"""Column order on page 1. Matches the production report exactly."""

CAMERA_DISPLAY = {
    C.CAMERA_LEFT_UP: "Left",
    C.CAMERA_RIGHT_UP: "Right",
    C.CAMERA_LEFT_UP_TOP: "Left-Top",
    C.CAMERA_RIGHT_UP_TOP: "Right-Top",
}

DETAIL_LABEL = {
    C.CAMERA_LEFT_UP: "LEFT Detail Report",
    C.CAMERA_RIGHT_UP: "RIGHT Detail Report",
    C.CAMERA_RIGHT_UP_TOP: "R-TOP Detail Report",
    C.CAMERA_LEFT_UP_TOP: "L-TOP Detail Report",
}

DOOR_CLASS_LABEL = {
    "open_door": "OPEN",
    "closed_door": "CLOSED",
    "partially_closed": "PARTIAL CLOSED",
}
DOOR_BAND_COLUMNS = {
    "open_door": "open_door_band_info",
    "closed_door": "closed_door_band_info",
    "partially_closed": "partially_closed_band_info",
}

NO_DOOR = "NO DOOR DETECTED"
NOT_VISIBLE_TEXT = "NOT VISIBLE"

PROBLEM_CAPTION = {
    "open_door": "Open Door",
    "partially_closed": "Partial Closed",
    "closed_door": "Closed Door",
    "damage": "Damage",
    "floor_dmg": "Damage",
    "inner_wall_dmg": "Damage",
    "floor_dmg_probable": "Probable Damage",
}


# ---------------------------------------------------------------------------
# persisted-state readers
# ---------------------------------------------------------------------------

def _camera_dir(output_root: str, camera_id: str) -> str:
    return os.path.join(output_root, gb.camera_profile(camera_id).legacy_name)


def _read_csv(path: str):
    import pandas as pd
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:                                  # noqa: BLE001
        return pd.DataFrame()


def _as_list(value: Any) -> List[dict]:
    """Revive a band list that a CSV round-trip turned into its repr."""
    if isinstance(value, list):
        return [b for b in value if isinstance(b, dict)]
    if value is None:
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    return [b for b in parsed if isinstance(b, dict)] if isinstance(parsed, list) else []


def _door_bands_by_wagon(output_root: str, camera_id: str) -> Dict[int, List[dict]]:
    """``{segment_id: [{class, start_frame}, ...]}`` -- one entry per door seen.

    Bands are the door instances (see the module docstring). Sorted by
    ``start_frame`` so DOOR 1 is the first door that came into view, which is the
    order the production report numbers them in.
    """
    df = _read_csv(os.path.join(_camera_dir(output_root, camera_id),
                                "damage_results.csv"))
    out: Dict[int, List[dict]] = {}
    if df is None or len(df) == 0 or "wagon_id" not in df.columns:
        return out
    for _, row in df.iterrows():
        doors: List[dict] = []
        for cls, column in DOOR_BAND_COLUMNS.items():
            if column not in df.columns:
                continue
            for band in _as_list(row.get(column)):
                doors.append({
                    "class": cls,
                    "start_frame": int(band.get("start_frame", 0) or 0),
                    "confidence": float(band.get("avg_confidence", 0.0) or 0.0),
                })
        doors.sort(key=lambda d: d["start_frame"])
        out[int(row["wagon_id"])] = doors
    return out


def _problem_rows_by_wagon(output_root: str, camera_id: str
                           ) -> Dict[int, List[dict]]:
    """``{segment_id: [{problem_type, image_path, frame_number}, ...]}``.

    Prefers the annotated image (the one carrying the drawn box) and falls back
    to the raw frame -- the legacy rule, never both.
    """
    df = _read_csv(os.path.join(_camera_dir(output_root, camera_id),
                                "problem_frames.csv"))
    out: Dict[int, List[dict]] = {}
    if df is None or len(df) == 0 or "wagon_id" not in df.columns:
        return out
    for _, row in df.iterrows():
        annotated = row.get("annotated_image_path")
        raw = row.get("frame_path")
        path = None
        for candidate in (annotated, raw):
            if isinstance(candidate, str) and candidate and os.path.exists(candidate):
                path = candidate
                break
        out.setdefault(int(row["wagon_id"]), []).append({
            "problem_type": str(row.get("problem_type") or ""),
            "image_path": path,
            "frame_number": int(row.get("frame_number") or 0),
        })
    return out


# ---------------------------------------------------------------------------
# cell builders
# ---------------------------------------------------------------------------

def door_cell_text(doors: Sequence[dict], visible: bool = True) -> str:
    """``DOOR 1 CLOSED / DOOR 2 PARTIAL CLOSED`` from a wagon's door bands."""
    if not visible:
        return NOT_VISIBLE_TEXT
    if not doors:
        return NO_DOOR
    return " / ".join(
        f"DOOR {i} {DOOR_CLASS_LABEL.get(d['class'], d['class'].upper())}"
        for i, d in enumerate(doors, start=1))


def _segments_by_index(payload: Optional[dict]) -> Dict[int, dict]:
    if not payload:
        return {}
    out: Dict[int, dict] = {}
    for seg in (payload.get("inspection_data", {}).get("wagon_segments") or []):
        count = seg.get("wagon_count")
        if count is not None:
            out[int(count)] = seg
    return out


def wagon_report_rows(state: Any, payloads: Dict[str, dict], output_root: str,
                      load_status_by_wagon: Optional[Dict[str, str]] = None
                      ) -> List[Dict[str, Any]]:
    """One row per GLOBAL wagon, in roster order. The report's row set.

    Built from the roster rather than from the payloads, so the table has
    exactly N rows for N global wagons however many any camera observed.
    """
    load_status_by_wagon = load_status_by_wagon or {}
    segs = {cam: _segments_by_index(payloads.get(cam)) for cam in CAMERA_ORDER}
    doors = {cam: _door_bands_by_wagon(output_root, cam)
             for cam in (C.CAMERA_LEFT_UP, C.CAMERA_RIGHT_UP)}

    rows: List[Dict[str, Any]] = []
    for wagon in gb._iter_roster(state):
        gid, index, classification = gb._wagon_fields(wagon)
        if gid is None or index is None:
            continue

        def visible(cam: str) -> bool:
            seg = segs[cam].get(int(index))
            return bool(seg) and seg.get("inspection_status") not in (
                "NOT_VISIBLE", "UNRESOLVED")

        def damaged(cam: str) -> bool:
            seg = segs[cam].get(int(index)) or {}
            return bool(seg.get("damage_detected"))

        left_doors = doors[C.CAMERA_LEFT_UP].get(int(index), [])
        right_doors = doors[C.CAMERA_RIGHT_UP].get(int(index), [])
        load = load_status_by_wagon.get(str(gid), C.NO_DATA)

        rows.append({
            "sr_no": int(index),
            "global_wagon_id": str(gid),
            "classification": classification,
            "wagon_number": "-",          # OCR disabled; column preserved
            "left_doors": left_doors,
            "right_doors": right_doors,
            "left_text": door_cell_text(left_doors, visible(C.CAMERA_LEFT_UP)),
            "right_text": door_cell_text(right_doors, visible(C.CAMERA_RIGHT_UP)),
            "r_top_damage": damaged(C.CAMERA_RIGHT_UP_TOP),
            "l_top_damage": damaged(C.CAMERA_LEFT_UP_TOP),
            "load_status": load,
            "wagon_type": ("LOADED" if load == C.LOAD_LOADED
                           else "EMPTY" if load == C.LOAD_EMPTY else "-"),
            "has_open": any(d["class"] == "open_door"
                            for d in left_doors + right_doors),
            "has_partial": {
                "left": any(d["class"] == "partially_closed" for d in left_doors),
                "right": any(d["class"] == "partially_closed" for d in right_doors),
            },
            "open_by_side": {
                "left": any(d["class"] == "open_door" for d in left_doors),
                "right": any(d["class"] == "open_door" for d in right_doors),
            },
        })
    return rows


def _is_anomalous(row: Dict[str, Any]) -> bool:
    return bool(row["has_open"] or row["r_top_damage"] or row["l_top_damage"])


def summary_row(rows: Sequence[Dict[str, Any]],
                when: datetime) -> Dict[str, Any]:
    """The INSPECTION SUMMARY line.

    TOTAL WAGONS is ``len(rows)`` -- the global roster size, not any camera's
    tally. LOCO NUMBER is "-" because OCR is disabled; the column is preserved.
    """
    left_open = sum(1 for r in rows if r["open_by_side"]["left"])
    right_open = sum(1 for r in rows if r["open_by_side"]["right"])
    left_partial = sum(1 for r in rows if r["has_partial"]["left"])
    right_partial = sum(1 for r in rows if r["has_partial"]["right"])
    r_top = sum(1 for r in rows if r["r_top_damage"])
    l_top = sum(1 for r in rows if r["l_top_damage"])
    loaded = sum(1 for r in rows if r["wagon_type"] == "LOADED")
    empty = sum(1 for r in rows if r["wagon_type"] == "EMPTY")
    issues = left_open + right_open + r_top + l_top
    return {
        "date_time": when.strftime("%d-%m-%Y - %H:%M:%S"),
        "loco_number": "-",               # OCR disabled; column preserved
        "total_wagons": len(rows),
        "left_open_doors": left_open,
        "right_open_doors": right_open,
        "r_top_damages": r_top,
        "l_top_damages": l_top,
        "partial_closed": f"L {left_partial} / R {right_partial}",
        "rake_type": "LOADED RAKE" if loaded >= empty else "EMPTY RAKE",
        "status": "NOT OK" if issues else "OK",
    }


def damaged_wagon_entries(rows: Sequence[Dict[str, Any]], output_root: str,
                          when: datetime) -> List[Dict[str, Any]]:
    """One entry per wagon with a finding, with its evidence images.

    "Damaged" here means what the production report means: any anomaly worth a
    picture -- an open door or a damage detection -- not only the damage class.
    """
    problems = {cam: _problem_rows_by_wagon(output_root, cam)
                for cam in CAMERA_ORDER}
    entries: List[Dict[str, Any]] = []
    for row in rows:
        if not _is_anomalous(row):
            continue
        images: List[Dict[str, str]] = []
        angles: List[str] = []
        for cam in (C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP,
                    C.CAMERA_LEFT_UP_TOP, C.CAMERA_RIGHT_UP_TOP):
            for entry in problems[cam].get(row["sr_no"], []):
                if not entry["image_path"]:
                    continue
                caption = PROBLEM_CAPTION.get(entry["problem_type"], "Issue")
                images.append({
                    "caption": f"{CAMERA_DISPLAY[cam]} Camera – {caption}",
                    "path": entry["image_path"],
                })
                if CAMERA_DISPLAY[cam] not in angles:
                    angles.append(CAMERA_DISPLAY[cam])
        if not angles:
            # A finding with no surviving evidence frame still belongs in the
            # list -- omitting it would under-report the rake.
            for cam, flag in ((C.CAMERA_RIGHT_UP_TOP, row["r_top_damage"]),
                              (C.CAMERA_LEFT_UP_TOP, row["l_top_damage"])):
                if flag:
                    angles.append(CAMERA_DISPLAY[cam])
            if row["open_by_side"]["left"]:
                angles.append(CAMERA_DISPLAY[C.CAMERA_LEFT_UP])
            if row["open_by_side"]["right"]:
                angles.append(CAMERA_DISPLAY[C.CAMERA_RIGHT_UP])
        entries.append({
            "wagon_id": row["sr_no"],
            "global_wagon_id": row["global_wagon_id"],
            "wagon_number": row["wagon_number"],
            # Angles read alphabetically (Left-Top, Right, Right-Top) while the
            # images below stay in capture order (side first, then top), which
            # is how the production report presents them.
            "angles": ", ".join(sorted(angles)) or "-",
            "issues": max(len(images), len(angles)),
            "date_time": when.strftime("%d-%m-%Y %H:%M:%S IST"),
            "images": images,
        })
    return entries


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def build_combined_report_pdf(
    state: Any,
    output_root: str,
    payloads: Dict[str, dict],
    output_path: str,
    load_status_by_wagon: Optional[Dict[str, str]] = None,
    source_video_urls: Optional[Dict[str, str]] = None,
    processed_video_urls: Optional[Dict[str, str]] = None,
    camera_report_urls: Optional[Dict[str, str]] = None,
    logo_path: Optional[str] = None,
    when: Optional[datetime] = None,
    rows_per_page: int = 14,
) -> str:
    """Render the combined report. Returns ``output_path``."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
        Table, TableStyle,
    )

    when = when or datetime.now()
    rows = wagon_report_rows(state, payloads, output_root, load_status_by_wagon)
    summary = summary_row(rows, when)
    damaged = damaged_wagon_entries(rows, output_root, when)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".",
                exist_ok=True)
    doc = SimpleDocTemplate(
        output_path, pagesize=landscape(A4),
        leftMargin=12 * mm, rightMargin=12 * mm,
        topMargin=10 * mm, bottomMargin=10 * mm,
        title="Combined Wagon Eye Report", author="WagonEye")

    base = getSampleStyleSheet()
    st_banner = ParagraphStyle("Banner", parent=base["Normal"], fontSize=19,
                               leading=23, alignment=TA_CENTER,
                               textColor=colors.white,
                               fontName="Helvetica-Bold")
    st_bansub = ParagraphStyle("BannerSub", parent=base["Normal"], fontSize=9,
                               alignment=TA_CENTER,
                               textColor=colors.HexColor("#c7d0de"))
    st_sect = ParagraphStyle("Section", parent=base["Normal"], fontSize=9,
                             alignment=TA_CENTER, textColor=colors.white,
                             fontName="Helvetica-Bold")
    st_head = ParagraphStyle("Head", parent=base["Normal"], fontSize=7,
                             leading=9, alignment=TA_CENTER,
                             fontName="Helvetica-Bold",
                             textColor=colors.HexColor("#3c4043"))
    st_cell = ParagraphStyle("Cell", parent=base["Normal"], fontSize=7,
                             leading=9, alignment=TA_CENTER)
    st_link = ParagraphStyle("Link", parent=st_cell, fontSize=8,
                             textColor=colors.HexColor("#1a5fb4"),
                             fontName="Helvetica-Bold")
    st_cap = ParagraphStyle("Cap", parent=base["Normal"], fontSize=7.5,
                            alignment=TA_CENTER, fontName="Helvetica-Bold")
    st_total = ParagraphStyle("Total", parent=base["Normal"], fontSize=11,
                              alignment=TA_CENTER, fontName="Helvetica-Bold",
                              textColor=colors.HexColor(GREY))

    width = doc.width
    el: List[Any] = []

    def section_bar(text: str, color: str) -> Table:
        bar = Table([[Paragraph(text, st_sect)]], colWidths=[width])
        bar.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(color)),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        return bar

    def link(url: Optional[str], text: str) -> Paragraph:
        if not url:
            return Paragraph("<font color='#9aa0a6'>Not available</font>", st_cell)
        safe = str(url).replace("&", "&amp;")
        return Paragraph(f'<link href="{safe}"><u>{text}</u></link>', st_link)

    # ---- logo -----------------------------------------------------------
    if logo_path and os.path.exists(logo_path):
        try:
            el.append(Image(logo_path, width=22 * mm, height=9 * mm,
                            hAlign="LEFT"))
            el.append(Spacer(1, 4 * mm))
        except Exception:                              # noqa: BLE001
            pass

    # ---- banner ---------------------------------------------------------
    banner = Table([[Paragraph("COMBINED WAGON EYE REPORT", st_banner)],
                    [Paragraph(when.strftime("%d-%m-%Y | %H:%M IST"), st_bansub)]],
                   colWidths=[width])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(NAVY)),
        ("TOPPADDING", (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 12),
    ]))
    el.append(banner)
    el.append(Spacer(1, 6 * mm))

    # ---- VIDEO EVIDENCE -------------------------------------------------
    el.append(section_bar("VIDEO EVIDENCE", NAVY))
    src = source_video_urls or {}
    proc = processed_video_urls or {}
    col = (width - 34 * mm) / 4.0
    video_rows = [
        [Paragraph("", st_cell)] + [Paragraph(c, st_head) for c in CAMERA_ORDER],
        [Paragraph("Raw Video", st_head)]
        + [link(src.get(c), "Click to View") for c in CAMERA_ORDER],
        [Paragraph("Processed Video", st_head)]
        + [link(proc.get(c), "Click to View") for c in CAMERA_ORDER],
    ]
    video_tbl = Table(video_rows, colWidths=[34 * mm] + [col] * 4)
    video_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(SUBHEAD)),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor(SUBHEAD)),
        ("BACKGROUND", (1, 2), (-1, 2), colors.HexColor(ROW_ALT)),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d5dd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    el.append(video_tbl)
    el.append(Spacer(1, 5 * mm))

    # ---- DETAILED REPORTS ----------------------------------------------
    el.append(section_bar("DETAILED REPORTS", TEAL))
    reports = camera_report_urls or {}
    det_rows = [
        [Paragraph(c, st_head) for c in CAMERA_ORDER],
        [link(reports.get(c), DETAIL_LABEL[c]) for c in CAMERA_ORDER],
    ]
    det_tbl = Table(det_rows, colWidths=[width / 4.0] * 4)
    det_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(SUBHEAD)),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d5dd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    el.append(det_tbl)
    el.append(Spacer(1, 5 * mm))

    # ---- INSPECTION SUMMARY --------------------------------------------
    el.append(section_bar("INSPECTION SUMMARY", NAVY))
    head = ["DATE-TIME", "LOCO NUMBER", "TOTAL\nWAGONS", "LEFT OPEN\nDOORS",
            "RIGHT\nOPEN\nDOORS", "R-TOP\nDAMAGES", "L-TOP\nDAMAGES",
            "PARTIAL\nCLOSED", "RAKE\nTYPE", "STATUS"]
    sum_rows = [
        [Paragraph(h.replace("\n", "<br/>"), st_head) for h in head],
        [Paragraph(summary["date_time"].replace(" - ", " -<br/>"), st_cell),
         Paragraph(f"<b>{summary['loco_number']}</b>", st_cell),
         Paragraph(f"<b>{summary['total_wagons']}</b>", st_cell),
         Paragraph(f"<b>{summary['left_open_doors']}</b>", st_cell),
         Paragraph(f"<b>{summary['right_open_doors']}</b>", st_cell),
         Paragraph(f"<b>{summary['r_top_damages']}</b>", st_cell),
         Paragraph(f"<b>{summary['l_top_damages']}</b>", st_cell),
         Paragraph(summary["partial_closed"], st_cell),
         Paragraph(f"<b><font color='{BLUE_TXT}'>"
                   f"{summary['rake_type'].replace(' ', '<br/>')}</font></b>", st_cell),
         Paragraph(f"<b><font color='"
                   f"{RED if summary['status'] == 'NOT OK' else '#1a8f4a'}'>"
                   f"{summary['status']}</font></b>", st_cell)],
    ]
    sum_tbl = Table(sum_rows, colWidths=[width / 10.0] * 10)
    sum_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d5dd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("BACKGROUND", (8, 1), (8, 1), colors.HexColor(BLUE_BG)),
        ("BACKGROUND", (9, 1), (9, 1),
         colors.HexColor(PINK if summary["status"] == "NOT OK" else "#e6f4ea")),
    ]))
    # The navy header needs white text; Paragraph carries its own colour, so it
    # is restated here rather than relying on the table style alone.
    sum_rows[0] = [Paragraph(h.replace("\n", "<br/>"),
                             ParagraphStyle("HW", parent=st_head,
                                            textColor=colors.white))
                   for h in head]
    sum_tbl = Table(sum_rows, colWidths=[width / 10.0] * 10)
    sum_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d5dd")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("BACKGROUND", (8, 1), (8, 1), colors.HexColor(BLUE_BG)),
        ("BACKGROUND", (9, 1), (9, 1),
         colors.HexColor(PINK if summary["status"] == "NOT OK" else "#e6f4ea")),
    ]))
    el.append(sum_tbl)

    # ---- WAGON INSPECTION DETAILS --------------------------------------
    detail_head = ["SR.NO", "WAGON NUMBER", "LEFT CAMERA\nDOORS",
                   "RIGHT CAMERA\nDOORS", "R-TOP\nDAMAGES", "L-TOP\nDAMAGES",
                   "WAGON\nTYPE"]
    detail_widths = [14 * mm, 34 * mm, 0, 0, 22 * mm, 22 * mm, 24 * mm]
    door_w = (width - sum(w for w in detail_widths if w)) / 2.0
    detail_widths[2] = detail_widths[3] = door_w

    st_head_w = ParagraphStyle("HeadW", parent=st_head, textColor=colors.white)

    for page_index in range(max(1, (len(rows) + rows_per_page - 1) // rows_per_page)):
        chunk = rows[page_index * rows_per_page:(page_index + 1) * rows_per_page]
        if not chunk and page_index:
            break
        el.append(PageBreak())
        table_rows: List[List[Any]] = [
            [Paragraph("WAGON INSPECTION DETAILS", st_sect)] + [""] * 6,
            [Paragraph(h.replace("\n", "<br/>"), st_head_w) for h in detail_head],
        ]
        styles: List[tuple] = [
            ("SPAN", (0, 0), (-1, 0)),
            ("BACKGROUND", (0, 0), (-1, 1), colors.HexColor(NAVY)),
            ("GRID", (0, 1), (-1, -1), 0.5, colors.HexColor("#d0d5dd")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
        for i, row in enumerate(chunk, start=2):
            anomalous = _is_anomalous(row)
            left_red = row["open_by_side"]["left"]
            right_red = row["open_by_side"]["right"]

            def cell(text: str, red: bool) -> Paragraph:
                if red:
                    return Paragraph(
                        f"<b><font color='{RED}'>{text}</font></b>", st_cell)
                return Paragraph(text, st_cell)

            type_color = (BLUE_TXT if row["wagon_type"] == "LOADED"
                          else ORANGE if row["wagon_type"] == "EMPTY" else GREY)
            table_rows.append([
                Paragraph(f"<b>{row['sr_no']}</b>", st_cell),
                Paragraph(row["wagon_number"], st_cell),
                cell(row["left_text"], left_red),
                cell(row["right_text"], right_red),
                cell("DAMAGE" if row["r_top_damage"] else "OK",
                     row["r_top_damage"]),
                cell("DAMAGE" if row["l_top_damage"] else "OK",
                     row["l_top_damage"]),
                Paragraph(f"<b><font color='{type_color}'>"
                          f"{row['wagon_type']}</font></b>", st_cell),
            ])
            if anomalous:
                styles.append(("BACKGROUND", (0, i), (-1, i),
                               colors.HexColor(PINK)))
            elif i % 2 == 0:
                styles.append(("BACKGROUND", (0, i), (-1, i),
                               colors.HexColor(ROW_ALT)))
        tbl = Table(table_rows, colWidths=detail_widths, repeatRows=2)
        tbl.setStyle(TableStyle(styles))
        el.append(tbl)

    # ---- Damaged Wagon Report ------------------------------------------
    if damaged:
        el.append(PageBreak())
        el.append(Paragraph(
            "<b>Damaged Wagon Report</b>",
            ParagraphStyle("DR", parent=base["Normal"], fontSize=13,
                           alignment=TA_CENTER)))
        el.append(Spacer(1, 3 * mm))
        el.append(Paragraph(f"Total Damaged Wagons: {len(damaged)}", st_total))
        el.append(Spacer(1, 5 * mm))

        dmg_head = ["SN", "Wagon ID", "Wagon No.", "Camera Angles", "Issues",
                    "Date & Time"]
        dmg_widths = [16 * mm, 24 * mm, 0, 46 * mm, 20 * mm, 52 * mm]
        dmg_widths[2] = width - sum(w for w in dmg_widths if w)

        for n, entry in enumerate(damaged, start=1):
            block: List[Any] = []
            info = Table(
                [[Paragraph(f"<b>{h}</b>", st_head) for h in dmg_head],
                 [Paragraph(f"{n}.", st_cell),
                  Paragraph(str(entry["wagon_id"]), st_cell),
                  Paragraph(str(entry["wagon_number"]), st_cell),
                  Paragraph(f"<b>{entry['angles']}</b>", st_cell),
                  Paragraph(f"<b>{entry['issues']}</b>", st_cell),
                  Paragraph(entry["date_time"], st_cell)]],
                colWidths=dmg_widths)
            info.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(SUBHEAD)),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#5f6368")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]))
            block.append(info)
            block.append(Spacer(1, 4 * mm))
            if entry["images"]:
                block.append(_evidence_panel(entry["images"][:2], width,
                                             st_cap, colors, Table, TableStyle,
                                             Image, Paragraph, mm))
            el.append(KeepTogether(block))
            el.append(Spacer(1, 6 * mm))
            # A third image belongs to the same wagon but does not fit beside
            # the first two, so it gets its own panel -- same as the production
            # report's overflow page.
            for extra in _chunks(entry["images"][2:], 2):
                el.append(PageBreak())
                el.append(_evidence_panel(extra, width, st_cap, colors, Table,
                                          TableStyle, Image, Paragraph, mm))

    doc.build(el)
    return output_path


def _chunks(items: Sequence[Any], size: int) -> List[List[Any]]:
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def _evidence_panel(images, width, st_cap, colors, Table, TableStyle, Image,
                    Paragraph, mm):
    """A bordered panel of captioned evidence images, one row."""
    cells: List[Any] = []
    caps: List[Any] = []
    cell_w = width / max(1, len(images))
    img_w = cell_w - 16 * mm
    for entry in images:
        caps.append(Paragraph(entry["caption"], st_cap))
        try:
            img = Image(entry["path"])
            ratio = (img.imageHeight / img.imageWidth) if img.imageWidth else 0.62
            img.drawWidth = img_w
            img.drawHeight = min(img_w * ratio, 78 * mm)
            img.drawWidth = img.drawHeight / ratio if ratio else img_w
            cells.append(img)
        except Exception:                              # noqa: BLE001
            cells.append(Paragraph("<i>evidence image unavailable</i>", st_cap))
    panel = Table([caps, cells], colWidths=[cell_w] * len(images))
    panel.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#d0d5dd")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e4e7ec")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return panel
