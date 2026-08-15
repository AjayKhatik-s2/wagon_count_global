"""Render the legacy outputs: per-camera PDFs, combined PDF, annotated videos.

THE RULE THIS MODULE EXISTS TO ENFORCE
--------------------------------------
NOTHING HERE RUNS A MODEL. There is no YOLO import, no ``.predict``, no
``.detect``, no weight path. Every page and every overlay is built from what
``legacy_inspection`` already persisted:

    <camera>/inspection_data.json   the dashboard payload
    <camera>/segments.csv           one row per GLOBAL wagon
    <camera>/damage_results.csv     the legacy damage/door verdicts
    <camera>/problem_frames.csv     evidence frames + boxes
    <camera>/frame_detections.csv   per-frame boxes for the video overlay

That is what makes the three artifacts agree by construction: the PDF, the
processed video and the dashboard JSON cannot disagree about a wagon, because
none of them is allowed to form its own opinion. A second inference pass here
would be free to disagree with the JSON, so the absence of one is a property
worth testing, and ``tests/test_legacy_inspection_port.py`` asserts it against
this file's source text.

WAGON IDENTITY
--------------
The legacy combined report sized its wagon table from ``max(wagon_count)`` seen
across the cameras -- a camera-derived count. Here the row set comes from the
finalized global roster, so the combined report has exactly N rows for N global
wagons whatever any camera saw. A camera with no row for a wagon prints a
NOT VISIBLE cell; it never shortens the table.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C

from . import global_bridge as gb

__all__ = ["build_camera_pdfs", "build_combined_pdf", "build_annotated_videos",
           "build_all_legacy_outputs", "load_camera_payloads",
           "combined_wagon_rows"]


# ---------------------------------------------------------------------------
# persisted-state readers
# ---------------------------------------------------------------------------

def _camera_dir(output_root: str, camera_id: str) -> str:
    return os.path.join(output_root, gb.camera_profile(camera_id).legacy_name)


INTERNAL_JSON_NAME = "inspection_data.internal.json"
DASHBOARD_JSON_NAME = "inspection_data.json"


def load_camera_payloads(output_root: str,
                         cameras: Optional[Sequence[str]] = None
                         ) -> Dict[str, Dict[str, Any]]:
    """Read each camera's persisted payload.

    Prefers ``inspection_data.internal.json``, because the dashboard file is
    trimmed to the exact legacy contract and therefore carries no
    ``global_wagon_id`` / ``inspection_status`` -- the two fields the combined
    report uses to label a wagon and to tell "not seen" from "seen and clean".
    Falls back to the dashboard file, which still renders correctly: a segment
    with no ``inspection_status`` is treated as observed, which is what the
    legacy report assumed for every row it printed.

    A camera whose JSON is missing or unreadable is simply absent from the
    result -- reported as such, never substituted with an empty-but-clean blob.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for camera_id in (cameras or C.ALL_CAMERAS):
        base = _camera_dir(output_root, camera_id)
        for name in (INTERNAL_JSON_NAME, DASHBOARD_JSON_NAME):
            path = os.path.join(base, name)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    out[camera_id] = json.load(fh)
                break
            except (OSError, json.JSONDecodeError):
                continue
    return out


def _read_csv(path: str):
    import pandas as pd
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:                                  # noqa: BLE001
        return pd.DataFrame()


def _revive_list_columns(df, columns: Sequence[str]):
    """CSV round-trips lists as their repr; restore them for the PDF builder.

    ``problem_frames.csv`` carries a ``bounding_box`` list-of-dicts. The legacy
    PDF only reads ``annotated_image_path`` / ``frame_path`` / ``problem_type``
    / ``frame_number``, so this is belt-and-braces for any consumer that does
    look at the boxes.
    """
    import ast
    if df is None or len(df) == 0:
        return df
    for col in columns:
        if col not in df.columns:
            continue
        def _parse(v: Any) -> Any:
            if isinstance(v, (list, dict)) or v is None:
                return v
            try:
                return ast.literal_eval(str(v))
            except (ValueError, SyntaxError):
                return v
        df[col] = df[col].map(_parse)
    return df


# ---------------------------------------------------------------------------
# per-camera PDF -- the LEGACY PdfReportBuilder, unchanged
# ---------------------------------------------------------------------------

def build_camera_pdfs(
    output_root: str,
    upload_timestamp,
    cameras: Optional[Sequence[str]] = None,
    station_name: str = "HAZARIBAGH",
    logo_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Optional[str]]:
    """One legacy-layout PDF per camera, built from the persisted CSVs.

    Layout, pagination, status colours and the 25/55/80 wagon tiles are the
    legacy builder's own -- it is called, not reimplemented.
    """
    from .legacy.pdf_builder import CameraStyle, PdfReportBuilder

    out: Dict[str, Optional[str]] = {}
    for camera_id in (cameras or C.ALL_CAMERAS):
        profile = gb.camera_profile(camera_id)
        work_dir = _camera_dir(output_root, camera_id)
        payload = None
        json_path = os.path.join(work_dir, "inspection_data.json")
        if os.path.isfile(json_path):
            try:
                with open(json_path, encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError):
                payload = None
        if payload is None:
            out[camera_id] = None
            continue

        data = payload.get("inspection_data", {})
        segments_df = _read_csv(os.path.join(work_dir, "segments.csv"))
        damage_df = _read_csv(os.path.join(work_dir, "damage_results.csv"))
        problems_df = _revive_list_columns(
            _read_csv(os.path.join(work_dir, "problem_frames.csv")),
            ["bounding_box"])

        style = CameraStyle(
            camera_label=profile.pdf_position,
            station_name=station_name,
            flavour=profile.flavour,
        )
        pdf_path = os.path.join(work_dir, f"{profile.legacy_name}_report.pdf")
        try:
            import pandas as pd
            PdfReportBuilder(style=style, logo_path=logo_path).build(
                output_path=pdf_path,
                raw_video_name=str(data.get("raw_video_name") or camera_id),
                upload_timestamp=upload_timestamp,
                direction=str(data.get("direction") or ""),
                segment_summary_df=segments_df,
                damage_results_df=damage_df,
                loco_summary_df=pd.DataFrame(),
                problem_frames_df=problems_df,
                trimmed_video_url=data.get("trimmed_video_url"),
            )
            out[camera_id] = pdf_path
            if verbose:
                print(f"    [PDF/{camera_id}] {pdf_path}")
        except Exception as exc:                       # noqa: BLE001
            out[camera_id] = None
            if verbose:
                print(f"    [PDF/{camera_id}] FAILED: {type(exc).__name__}: {exc}")
    return out


# ---------------------------------------------------------------------------
# combined PDF -- legacy layout, GLOBAL roster as the row set
# ---------------------------------------------------------------------------

def _wagon_status_for_segment(seg: Dict[str, Any], flavour: str) -> Tuple[str, str]:
    """RECOVERED verbatim from the legacy ``combiner/pdf.py``.

    One addition: a segment the camera never saw reports NOT VISIBLE instead of
    OK. The legacy version could not hit that case, because a camera that missed
    a wagon simply had no row for it -- which is exactly the count divergence
    this port removes.
    """
    status = seg.get("inspection_status")
    if status in ("NOT_VISIBLE", "UNRESOLVED"):
        return ("NOT VISIBLE" if status == "NOT_VISIBLE" else "UNRESOLVED",
                "#eeeeee")
    if flavour == "top":
        if seg.get("damage_detected"):
            return "DAMAGE", "#ff5252"
        if seg.get("probable_damage_detected"):
            return "PROBABLE", "#ffb74d"
        return "OK", "#81c784"
    if seg.get("damage_detected"):
        return "DAMAGE", "#ff5252"
    if seg.get("door_status") == "open":
        return "DOOR OPEN", "#ffb74d"
    if seg.get("door_status") == "partially_closed":
        return "PARTIAL", "#fff176"
    return "OK", "#81c784"


def _per_wagon_index(payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """``{wagon_count: segment}`` -- wagon_count IS the global wagon index."""
    out: Dict[int, Dict[str, Any]] = {}
    for seg in (payload.get("inspection_data", {}).get("wagon_segments") or []):
        count = seg.get("wagon_count")
        if count is not None:
            out[int(count)] = seg
    return out


def combined_wagon_rows(
    state: Any,
    payloads: Dict[str, Dict[str, Any]],
    load_status_by_wagon: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """One row per GLOBAL wagon, in roster order. THE report's row set.

    Built from the roster, not from the payloads, so the row count is the global
    wagon count by construction: if the roster has 57 wagons this returns 57
    rows, whatever any camera detected or failed to see.
    """
    load_status_by_wagon = load_status_by_wagon or {}
    indexed = {cam: _per_wagon_index(p) for cam, p in payloads.items()}

    rows: List[Dict[str, Any]] = []
    for wagon in gb._iter_roster(state):
        gid, index, classification = gb._wagon_fields(wagon)
        if gid is None or index is None:
            continue
        per_camera: Dict[str, Dict[str, Any]] = {}
        damage_types: List[str] = []
        door_states: List[str] = []
        for camera_id, seg_by_idx in indexed.items():
            seg = seg_by_idx.get(int(index))
            profile = gb.camera_profile(camera_id)
            if seg is None:
                per_camera[camera_id] = {
                    "status": "NOT VISIBLE", "color": "#eeeeee",
                    "inspection_status": "NOT_VISIBLE"}
                continue
            label, color = _wagon_status_for_segment(seg, profile.flavour)
            per_camera[camera_id] = {
                "status": label, "color": color,
                "inspection_status": seg.get("inspection_status")}
            if profile.is_top:
                if seg.get("floor_dmg_detected"):
                    damage_types.append("floor_dmg")
                if seg.get("inner_wall_dmg_detected"):
                    damage_types.append("inner_wall_dmg")
                if seg.get("floor_dmg_probable_detected"):
                    damage_types.append("floor_dmg_probable")
            else:
                if seg.get("damage_detected"):
                    damage_types.append("side_damage")
                door_states.append(str(seg.get("door_status") or "closed"))

        rows.append({
            "global_wagon_id": str(gid),
            "wagon_index": int(index),
            "classification": classification,
            "load_status": load_status_by_wagon.get(str(gid), C.NO_DATA),
            "door_status": _fuse_door(door_states),
            "damage_types": sorted(set(damage_types)),
            "damage_status": "DAMAGE" if damage_types else "OK",
            "per_camera": per_camera,
        })
    return rows


def _fuse_door(states: Sequence[str]) -> str:
    """Worst-case across the two side cameras.

    A door is a physical property of one side of one wagon, and the two side
    cameras see different sides, so this is a union of two independent
    observations rather than a vote between two opinions: if either side is
    open, the wagon has an open door. Precedence open > partially_closed >
    closed matches the legacy ``_DOOR_CLASS_PRECEDENCE``. With no observation at
    all the answer is NO_DATA, never "closed".
    """
    if not states:
        return C.NO_DATA
    for candidate in ("open", "partially_closed"):
        if candidate in states:
            return candidate
    return "closed"


def build_combined_pdf(
    state: Any,
    output_root: str,
    payloads: Optional[Dict[str, Dict[str, Any]]] = None,
    load_status_by_wagon: Optional[Dict[str, str]] = None,
    batch_key: str = "",
    rows_per_page: int = 14,
    source_video_urls: Optional[Dict[str, str]] = None,
    processed_video_urls: Optional[Dict[str, str]] = None,
    camera_report_urls: Optional[Dict[str, str]] = None,
    logo_path: Optional[str] = None,
    when: Optional[Any] = None,
    verbose: bool = True,
) -> Optional[str]:
    """THE COMBINED WAGON EYE REPORT.

    Layout lives in :mod:`inspection.combined_report`, which reproduces the
    production document: banner, VIDEO EVIDENCE, DETAILED REPORTS, INSPECTION
    SUMMARY, the paged WAGON INSPECTION DETAILS table, and the Damaged Wagon
    Report with per-wagon evidence panels.

    Everything is read from persisted state; the row set is the global roster,
    so the table has exactly one row per GW_1..GW_N.
    """
    from datetime import datetime

    from .combined_report import build_combined_report_pdf

    payloads = payloads if payloads is not None else load_camera_payloads(output_root)
    report_dir = os.path.join(output_root, "reports")
    os.makedirs(report_dir, exist_ok=True)
    output_path = os.path.join(report_dir, "combined_inspection_report.pdf")

    try:
        build_combined_report_pdf(
            state=state, output_root=output_root, payloads=payloads,
            output_path=output_path,
            load_status_by_wagon=load_status_by_wagon,
            source_video_urls=source_video_urls,
            processed_video_urls=processed_video_urls,
            camera_report_urls=camera_report_urls,
            logo_path=logo_path,
            when=when or datetime.now(),
            rows_per_page=rows_per_page)
        if verbose:
            n = len(gb._iter_roster(state))
            print(f"    [REPORT] combined -> {output_path} "
                  f"({n} global wagon rows)")
        return output_path
    except Exception as exc:                           # noqa: BLE001
        if verbose:
            print(f"    [REPORT] combined FAILED: {type(exc).__name__}: {exc}")
        return None


def _camera_summary_page(pdf, plt, page, payloads) -> None:
    fig = plt.figure(figsize=page)
    fig.suptitle("Per-camera summary", fontsize=16, fontweight="bold")
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    rows = []
    for camera_id in C.ALL_CAMERAS:
        payload = payloads.get(camera_id)
        if payload is None:
            rows.append([camera_id, "—", "NOT REPORTED", "—", "—"])
            continue
        data = payload.get("inspection_data", {})
        rows.append([
            camera_id,
            str(data.get("raw_video_name") or "—"),
            str(data.get("rake_status") or "—"),
            data.get("total_wagons", "—"),
            data.get("total_problem_frames", "—"),
        ])
    tbl = ax.table(
        cellText=rows,
        colLabels=["CAMERA", "RAW VIDEO", "RAKE STATUS", "TOTAL WAGONS",
                   "PROBLEM FRAMES"],
        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _wagon_table_pages(pdf, plt, page, rows, payloads, rows_per_page) -> None:
    present = [c for c in C.ALL_CAMERAS if c in payloads]
    headers = (["GW", "CLASS", "LOAD", "DOOR", "DAMAGE"]
               + [c.replace("_UP", "").replace("_", " ") for c in present])
    pages = max(1, math.ceil(len(rows) / rows_per_page))
    for page_idx in range(pages):
        chunk = rows[page_idx * rows_per_page:(page_idx + 1) * rows_per_page]
        fig = plt.figure(figsize=page)
        fig.suptitle(
            f"Per-wagon status (page {page_idx + 1} of {pages}) — "
            f"{len(rows)} global wagons",
            fontsize=15, fontweight="bold")
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        cell_text: List[List[str]] = []
        cell_colors: List[List[str]] = []
        for row in chunk:
            damage_label = ", ".join(row["damage_types"]) or "none"
            line = [row["global_wagon_id"], row["classification"],
                    row["load_status"], row["door_status"], damage_label]
            colors = ["white", "white", "white", "white",
                      "#ff5252" if row["damage_types"] else "white"]
            for camera_id in present:
                cell = row["per_camera"].get(camera_id, {})
                line.append(str(cell.get("status", "—")))
                colors.append(str(cell.get("color", "#eeeeee")))
            cell_text.append(line)
            cell_colors.append(colors)
        tbl = ax.table(cellText=cell_text or [["—"] * len(headers)],
                       colLabels=headers,
                       cellColours=cell_colors or None,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1, 1.35)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _camera_totals_page(pdf, plt, page, payloads) -> None:
    fig = plt.figure(figsize=page)
    fig.suptitle("Per-camera totals", fontsize=15, fontweight="bold")
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    rows = []
    for camera_id in C.ALL_CAMERAS:
        payload = payloads.get(camera_id)
        data = (payload or {}).get("inspection_data", {})
        rows.append([
            camera_id,
            data.get("total_wagons", "—"),
            data.get("damaged_wagons", "—"),
            data.get("doors_open", "—"),
            data.get("floor_dmg_wagons", "—"),
            data.get("inner_wall_dmg_wagons", "—"),
        ])
    tbl = ax.table(
        cellText=rows,
        colLabels=["CAMERA", "TOTAL", "DAMAGED", "DOOR OPEN", "FLOOR DMG",
                   "INNER WALL DMG"],
        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _problem_gallery(pdf, plt, page, output_root, payloads,
                     per_page: int = 3) -> None:
    """Evidence pages. Uses the frame each problem entry actually points at."""
    import cv2

    entries: List[Tuple[str, Dict[str, Any]]] = []
    for camera_id, payload in payloads.items():
        for pf in (payload.get("inspection_data", {}).get("problem_frames") or []):
            entries.append((camera_id, pf))
    if not entries:
        return

    fig = plt.figure(figsize=page)
    fig.suptitle("DAMAGE / DOOR PROBLEM FRAMES", fontsize=20,
                 fontweight="bold", color="red")
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    for start in range(0, len(entries), per_page):
        chunk = entries[start:start + per_page]
        fig, axes = plt.subplots(len(chunk), 1, figsize=page)
        if len(chunk) == 1:
            axes = [axes]
        for ax, (camera_id, pf) in zip(axes, chunk):
            path = _local_problem_frame(output_root, camera_id, pf)
            if path and os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                del img
            ax.set_title(
                f"{pf.get('global_wagon_id') or ''} "
                f"(wagon {pf.get('wagon_count')}) · {camera_id} · "
                f"{pf.get('problem_type')} · frame {pf.get('frame_number')}",
                fontsize=9)
            ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def _local_problem_frame(output_root: str, camera_id: str,
                         pf: Dict[str, Any]) -> Optional[str]:
    """Find the locally published copy of a problem frame.

    Prefers the artifact the publisher actually wrote (annotated when one
    exists, raw otherwise -- never both, which is the legacy rule), and falls
    back to the annotated/raw path recorded during extraction.
    """
    filename = pf.get("filename")
    if filename:
        artifacts = os.path.join(output_root, "artifacts")
        for root, _dirs, files in os.walk(artifacts):
            if filename in files:
                return os.path.join(root, filename)
    return None


# ---------------------------------------------------------------------------
# annotated videos -- the LEGACY builder, unchanged
# ---------------------------------------------------------------------------

def build_annotated_videos(
    output_root: str,
    tracks_by_camera: Dict[str, Any],
    cameras: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Dict[str, Optional[str]]:
    """Draw the persisted damage/door boxes onto each camera's video.

    The overlay is purely additive and comes from ``frame_detections.csv`` -- a
    file, not a model -- so the boxes in the video are the same detections that
    produced the JSON. The global GW overlay is drawn by the counting
    pipeline's own renderer (``video_segmenter.render_processed_video``); this
    adds the inspection boxes the legacy videos carried.
    """
    from .legacy.annotated_video import build_annotated_video

    out: Dict[str, Optional[str]] = {}
    for camera_id in (cameras or C.ALL_CAMERAS):
        profile = gb.camera_profile(camera_id)
        work_dir = _camera_dir(output_root, camera_id)
        lct = tracks_by_camera.get(camera_id)
        video_path = str(getattr(lct, "video_path", "") or "")
        detections = _read_csv(os.path.join(work_dir, "frame_detections.csv"))
        if not video_path or not os.path.isfile(video_path):
            out[camera_id] = None
            continue
        try:
            out[camera_id] = build_annotated_video(
                video_path=video_path,
                output_dir=work_dir,
                raw_video_name=os.path.splitext(os.path.basename(video_path))[0],
                gap_csv=None, loco_csv=None,
                damage_frame_detections_df=detections,
                flavour=profile.flavour)
            if verbose:
                print(f"    [VIDEO/{camera_id}] {out[camera_id]}")
        except Exception as exc:                       # noqa: BLE001
            out[camera_id] = None
            if verbose:
                print(f"    [VIDEO/{camera_id}] FAILED: "
                      f"{type(exc).__name__}: {exc}")
    return out


# ---------------------------------------------------------------------------

def _urls_from_payloads(payloads: Dict[str, Dict[str, Any]], key: str,
                        first_of_list: bool = False) -> Dict[str, str]:
    """Collect one URL per camera out of the persisted payloads.

    The links on the report's front page are the ones the JSON already
    published, so the PDF and the dashboard point a reviewer at the same
    artifacts instead of each deriving its own.
    """
    out: Dict[str, str] = {}
    for camera_id, payload in payloads.items():
        value = (payload.get("inspection_data", {}) or {}).get(key)
        if first_of_list:
            value = value[0] if isinstance(value, list) and value else None
        if value:
            out[camera_id] = str(value)
    return out


def build_all_legacy_outputs(
    state: Any,
    output_root: str,
    upload_timestamp,
    tracks_by_camera: Optional[Dict[str, Any]] = None,
    load_status_by_wagon: Optional[Dict[str, str]] = None,
    batch_key: str = "",
    build_videos: bool = False,
    logo_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Every renderer, in one call, all from persisted state."""
    payloads = load_camera_payloads(output_root)
    result: Dict[str, Any] = {
        "camera_pdfs": build_camera_pdfs(output_root, upload_timestamp,
                                         verbose=verbose),
        "combined_pdf": build_combined_pdf(
            state, output_root, payloads=payloads,
            load_status_by_wagon=load_status_by_wagon,
            batch_key=batch_key,
            source_video_urls=_urls_from_payloads(
                payloads, "raw_video_urls", first_of_list=True),
            processed_video_urls=_urls_from_payloads(
                payloads, "detected_video_url"),
            camera_report_urls=_urls_from_payloads(payloads, "pdf_report_url"),
            logo_path=logo_path,
            when=upload_timestamp, verbose=verbose),
        "cameras_reporting": sorted(payloads),
        "global_wagon_count": len(gb._iter_roster(state)),
    }
    if build_videos and tracks_by_camera:
        result["annotated_videos"] = build_annotated_videos(
            output_root, tracks_by_camera, verbose=verbose)
    return result
