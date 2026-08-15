"""Build the canonical ``inspection_data.json`` payload.

Two flavours are emitted, matching the original notebooks byte-for-byte.

Top cameras (left_top, right_top)
---------------------------------
* `segment_type` strings in JSON: ``wagon_empty`` / ``wagon_loaded`` / ``engine``
  / ``brakevan``.
* Per-segment ``segment_type_map`` value: ``{type, number, wagon_count}`` with
  ``wagon_count`` only set for wagons.
* ``wagon_segments[i]`` carries 3 damage booleans
  (floor_dmg / inner_wall_dmg / floor_dmg_probable).
* Top-level counts include ``floor_dmg_wagons``, ``inner_wall_dmg_wagons``,
  ``floor_dmg_probable_wagons``, ``probable_damage_wagons``.
* ``wagon_number_results`` and ``loco_number_results`` are present but empty.
* ``rake_status`` (``"Loaded"``/``"Empty"``) is the majority vote across this
  camera's ``wagons_loaded`` vs ``wagons_empty`` segment counts.

Side cameras (left_up, right_up)
-------------------------------
* `segment_type` strings in JSON: ``wagon`` / ``engine`` / ``brakevan``
  (no empty/loaded split — side cameras don't classify load visually).
* Per-segment ``segment_type_map`` value: ``{type, number}``.
* ``wagon_segments[i]`` carries ``door_status`` + ``door_close_detected`` +
  ``damage_detected`` (booleans).
* Top-level counts include ``doors_open``, ``doors_closed``,
  ``damaged_wagons``.
* ``wagon_number_results`` keyed by ``str(wagon_count)``,
  ``loco_number_results`` keyed by ``str(loco_id)``.
* ``rake_status`` (``"Loaded"``/``"Empty"``) has no visual load class to draw
  on, so it's inferred from travel ``direction`` instead (left-to-right =
  loaded), same convention as ``combiner/pdf.py``'s ``_direction_to_load``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd

# Internal-to-display mapping used elsewhere — re-exported here to keep the
# two modules' contracts in lockstep.
from .artifacts import display_segment_type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_wagon(seg_type: str) -> bool:
    return seg_type in {"wagon", "wagon_empty", "wagon_loaded"}


def _load_status(seg_type: str) -> Optional[str]:
    if seg_type == "wagon_loaded":
        return "loaded"
    if seg_type in {"wagon", "wagon_empty"}:
        return "empty"
    return None


def _rake_status_from_wagon_counts(wagons_loaded: int, wagons_empty: int) -> str:
    """Top flavour: majority vote across segment load classes."""
    return "Loaded" if wagons_loaded >= wagons_empty else "Empty"


def _rake_status_from_direction(direction: str) -> str:
    """Side flavour: no per-wagon load class exists, so infer the rake's
    overall status from travel direction (left-to-right = loaded), mirroring
    the side-camera convention in ``combiner/pdf.py``'s ``_direction_to_load``.
    """
    if direction not in ("left-to-right", "right-to-left"):
        return "Unknown"
    return "Loaded" if direction == "left-to-right" else "Empty"


def _strip_camera_prefix(folder: str) -> str:
    """``camera_CCTV_HZBN_DHN_6_LEFT_TOP`` -> ``CCTV_HZBN_DHN_6_LEFT_TOP``."""
    if folder and folder.lower().startswith("camera_"):
        return folder[7:]
    return folder


def _build_loco_frames_block(
    loco_frame_entries: list[dict], loco_numbers: Optional[dict[int, dict]],
) -> list[dict]:
    """Group flat (loco_id, position) frame entries into one block per loco.

    Mirrors ``wagon_segments[i].wagon_frames`` — each loco gets its own
    ``frames`` list of representative frames (same positions/names as wagon
    frames), plus the OCR'd ``loco_number`` when available.
    """
    by_loco: dict[int, list[dict]] = {}
    for e in loco_frame_entries:
        by_loco.setdefault(e["loco_id"], []).append(e)

    result = []
    for loco_id in sorted(by_loco):
        loco_number = (loco_numbers or {}).get(loco_id, {}).get("display_number")
        result.append({
            "loco_id": loco_id,
            "loco_number": loco_number,
            "frames": [
                {
                    "position": f["position"],
                    "filename": f["filename"],
                    "s3_key": f["s3_key"],
                    "s3_url": f["s3_url"],
                    "frame_number": f["frame_num"],
                    "timestamp_sec": f["timestamp_sec"],
                }
                for f in by_loco[loco_id]
            ],
        })
    return result


def _build_segment_type_map_and_wagon_counts(
    segment_summary_df: pd.DataFrame, flavour: str,
) -> tuple[dict[str, dict], dict[int, Optional[int]]]:
    """Per-segment number-by-type counters + per-segment wagon_count.

    The segment_type_map is keyed by ``str(segment_id)`` (notebooks emit
    string keys). For top flavour each entry is
    ``{type, number, wagon_count}`` (wagon_count only set for wagons); for
    side flavour each entry is ``{type, number}``.
    """
    type_counters: dict[str, int] = {}
    seg_map: dict[str, dict] = {}
    wagon_count_map: dict[int, Optional[int]] = {}
    wagon_counter = 0

    for _, seg in segment_summary_df.iterrows():
        seg_id = int(seg["segment_id"])
        internal_type = seg.get("segment_type", "wagon")
        display = display_segment_type(internal_type, flavour)
        type_counters[display] = type_counters.get(display, 0) + 1
        number = type_counters[display]

        if _is_wagon(internal_type):
            wagon_counter += 1
            wagon_count_map[seg_id] = wagon_counter
        else:
            wagon_count_map[seg_id] = None

        entry = {"type": display, "number": number}
        if flavour == "top":
            entry["wagon_count"] = wagon_count_map[seg_id]
        seg_map[str(seg_id)] = entry

    return seg_map, wagon_count_map


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_inspection_json(
    *,
    camera_folder: str,
    raw_video_name: str,
    upload_timestamp: datetime,
    direction: str,
    flavour: str,
    segment_summary_df: pd.DataFrame,
    damage_results_df: pd.DataFrame,
    loco_summary_df: pd.DataFrame,
    problem_frames_df: pd.DataFrame,
    wagon_frames_index: dict[int, list[dict]],
    loco_frame_entries: list[dict],
    problem_frame_entries: list[dict],
    wagon_count_map: dict[int, Optional[int]],
    segment_type_map: dict[str, dict],
    trimmed_video_url: Optional[str] = None,
    pdf_report_url: Optional[str] = None,
    detected_video_url: Optional[str] = None,
    raw_video_urls: Optional[list[str]] = None,
    wagon_number_results: Optional[dict[int, dict]] = None,
    loco_numbers: Optional[dict[int, dict]] = None,
    damage_model_active: bool = True,
    version: str = "v4",
) -> dict:
    """Return the JSON-serialisable inspection payload matching the v2 schema."""

    inspection_data: dict[str, Any] = {
        "raw_video_name": raw_video_name,
        "identified_by": "model-v3",
        "upload_timestamp": (
            upload_timestamp.strftime("%Y-%m-%dT%H:%M:%S")
            if upload_timestamp else None
        ),
        "upload_timestamp_readable": (
            upload_timestamp.strftime("%d-%m-%Y %H:%M:%S IST")
            if upload_timestamp else None
        ),
        "direction": direction,
        "rake_status": (
            None if flavour == "top" else _rake_status_from_direction(direction)
        ),
        "pdf_report_url": pdf_report_url,
        "trimmed_video_url": trimmed_video_url,
        "detected_video_url": detected_video_url,
        "raw_video_urls": raw_video_urls or [],
    }

    damage_lookup = (
        damage_results_df.set_index("wagon_id").to_dict("index")
        if damage_results_df is not None and not damage_results_df.empty else {}
    )

    # Counts that both flavours expose.
    n_engines = 0
    n_brakevans = 0

    if flavour == "top":
        wagons_loaded = 0
        wagons_empty = 0
        damaged_wagons = 0
        probable_damage_wagons = 0
        floor_dmg_wagons = 0
        inner_wall_dmg_wagons = 0
        floor_dmg_probable_wagons = 0
        wagon_segments: list[dict] = []

        for _, seg in segment_summary_df.iterrows():
            seg_id = int(seg["segment_id"])
            internal_type = seg.get("segment_type", "wagon")
            display = display_segment_type(internal_type, "top")
            damage_row = damage_lookup.get(seg_id, {})

            if display == "engine":
                n_engines += 1
                continue
            if display == "brakevan":
                n_brakevans += 1
                continue

            # Wagon row — V4_top_damage classes: floor_dmg, inner_wall_dmg, floor_dmg_probable
            floor_dmg = bool(damage_row.get("floor_dmg_detected"))
            inner_wall_dmg = bool(damage_row.get("inner_wall_dmg_detected"))
            floor_prob = bool(damage_row.get("floor_dmg_probable_detected"))
            damage_any = floor_dmg or inner_wall_dmg
            probable_any = floor_prob

            if internal_type == "wagon_loaded":
                wagons_loaded += 1
            else:
                wagons_empty += 1
            if damage_any:
                damaged_wagons += 1
            if probable_any:
                probable_damage_wagons += 1
            if floor_dmg:
                floor_dmg_wagons += 1
            if inner_wall_dmg:
                inner_wall_dmg_wagons += 1
            if floor_prob:
                floor_dmg_probable_wagons += 1

            wagon_count = wagon_count_map.get(seg_id)
            wn_result = (wagon_number_results or {}).get(seg_id) if wagon_count is not None else None
            wagon_seg_top: dict[str, Any] = {
                "segment_id": seg_id,
                "segment_type": display,
                "wagon_count": wagon_count,
                "load_status": _load_status(internal_type),
                "load_condition": None,
                "damage_detected": damage_any,
                "probable_damage_detected": probable_any,
                "floor_dmg_detected": floor_dmg,
                "inner_wall_dmg_detected": inner_wall_dmg,
                "floor_dmg_probable_detected": floor_prob,
                "wagon_frames": [
                    {"position": f["position"], "s3_url": f["s3_url"]}
                    for f in wagon_frames_index.get(seg_id, [])
                ],
            }
            if wn_result and wn_result.get("is_valid_11_digit"):
                wagon_seg_top["wagon_number"] = wn_result.get("display_number")
                wagon_seg_top["is_valid_wagon_id"] = True
            else:
                wagon_seg_top["is_valid_wagon_id"] = False
            wagon_segments.append(wagon_seg_top)

        total_wagons = wagons_loaded + wagons_empty
        problem_by_type = {
            "floor_dmg":
                sum(1 for pf in problem_frame_entries if pf["problem_type"] == "floor_dmg"),
            "inner_wall_dmg":
                sum(1 for pf in problem_frame_entries if pf["problem_type"] == "inner_wall_dmg"),
            "floor_dmg_probable":
                sum(1 for pf in problem_frame_entries if pf["problem_type"] == "floor_dmg_probable"),
        }

        # OCR per-wagon dict keyed by str(wagon_count). Only valid wagons are emitted.
        wn_results_block_top: dict[str, dict] = {}
        for seg_id, wn in (wagon_number_results or {}).items():
            wc = wagon_count_map.get(seg_id)
            if wc is None:
                continue
            wn_results_block_top[str(wc)] = {
                "is_valid_11_digit": bool(wn.get("is_valid_11_digit")),
                "display_number": wn.get("display_number", "-"),
                "is_manipulated": bool(wn.get("fallback_triggered")),
                "original_number": wn.get("raw_number", wn.get("display_number", "-")),
            }

        inspection_data["rake_status"] = _rake_status_from_wagon_counts(
            wagons_loaded, wagons_empty,
        )
        inspection_data.update({
            "total_wagons": total_wagons,
            "wagons_loaded": wagons_loaded,
            "wagons_empty": wagons_empty,
            "damaged_wagons": damaged_wagons,
            "probable_damage_wagons": probable_damage_wagons,
            "floor_dmg_wagons": floor_dmg_wagons,
            "inner_wall_dmg_wagons": inner_wall_dmg_wagons,
            "floor_dmg_probable_wagons": floor_dmg_probable_wagons,
            "num_engines": n_engines,
            "num_brakevans": n_brakevans,
            "total_loco_frames": len(loco_frame_entries),
            "total_problem_frames": len(problem_frame_entries),
            "problem_frames_by_type": problem_by_type,
            "wagon_number_results": wn_results_block_top,
            "loco_number_results": {},
            "segment_type_map": segment_type_map,
            "wagon_segments": wagon_segments,
            "loco_frames": _build_loco_frames_block(loco_frame_entries, loco_numbers),
            "problem_frames": [
                _problem_frame_top_entry(pf, wagon_count_map, damage_lookup, segment_type_map)
                for pf in problem_frame_entries
            ],
            "damage_model_active": damage_model_active,
        })

    else:  # side flavour
        damaged_wagons = 0
        doors_open = 0
        doors_partially_closed = 0
        doors_closed = 0
        door_close_wagons = 0
        wagon_segments_side: list[dict] = []

        for _, seg in segment_summary_df.iterrows():
            seg_id = int(seg["segment_id"])
            internal_type = seg.get("segment_type", "wagon")
            damage_row = damage_lookup.get(seg_id, {})

            if internal_type == "engine":
                n_engines += 1
                continue
            if internal_type == "brakevan":
                n_brakevans += 1
                continue

            damage_any = bool(damage_row.get("damage_detected"))
            door_status = damage_row.get("door_status", "closed")
            door_close_detected = bool(damage_row.get("door_close_detected"))
            door_partial_detected = bool(damage_row.get("door_partial_detected"))

            if damage_any:
                damaged_wagons += 1
            if door_status == "open":
                doors_open += 1
            elif door_status == "partially_closed":
                doors_partially_closed += 1
            else:
                doors_closed += 1
            if door_close_detected:
                door_close_wagons += 1

            wagon_count = wagon_count_map.get(seg_id)
            wagon_seg_entry: dict[str, Any] = {
                "segment_id": seg_id,
                "segment_type": "wagon",
                "wagon_count": wagon_count,
                "door_status": door_status,
                "door_close_detected": door_close_detected,
                "door_partial_detected": door_partial_detected,
                "damage_detected": damage_any,
                "wagon_frames": [
                    {"position": f["position"], "s3_url": f["s3_url"]}
                    for f in wagon_frames_index.get(seg_id, [])
                ],
            }
            # OCR data lives flat per-segment when available (matches notebook).
            wn_result = (wagon_number_results or {}).get(seg_id) if wagon_count is not None else None
            if wn_result and wn_result.get("is_valid_11_digit"):
                wagon_seg_entry["wagon_number"] = wn_result.get("display_number")
                wagon_seg_entry["is_valid_wagon_id"] = True
            else:
                wagon_seg_entry["is_valid_wagon_id"] = False
            wagon_segments_side.append(wagon_seg_entry)

        # OCR per-wagon dict keyed by str(wagon_count). Every read is emitted
        # (even invalid ones) so a bad OCR attempt is still auditable — only
        # is_valid_11_digit distinguishes a usable number from a raw dump.
        wn_results_block: dict[str, dict] = {}
        for seg_id, wn in (wagon_number_results or {}).items():
            wc = wagon_count_map.get(seg_id)
            if wc is None:
                continue
            wn_results_block[str(wc)] = {
                "is_valid_11_digit": bool(wn.get("is_valid_11_digit")),
                "display_number": wn.get("display_number", "-"),
                "is_manipulated": bool(wn.get("fallback_triggered")),
                "original_number": wn.get("raw_number", wn.get("display_number", "-")),
                "ocr_frame_s3_url": wn.get("ocr_frame_s3_url"),
            }

        # Loco numbers: one entry per loco band, keyed by str(loco_id). Every
        # read is emitted (even invalid ones), same as wagon numbers above.
        loco_results_block: dict[str, dict] = {}
        for loco_id, loco_number in sorted((loco_numbers or {}).items()):
            if not loco_number:
                continue
            loco_results_block[str(loco_id)] = {
                "is_valid_5_digit": bool(loco_number.get("is_valid_5_digit")),
                "display_number": loco_number.get("display_number", "-"),
                "raw_number": loco_number.get("raw_number", ""),
                "confidence": loco_number.get("confidence", 0.0),
                "ocr_confidence": loco_number.get("ocr_confidence", 0.0),
                "ocr_frame_s3_url": loco_number.get("ocr_frame_s3_url"),
            }

        problem_by_type = {
            "damage":
                sum(1 for pf in problem_frame_entries if pf["problem_type"] == "damage"),
            "open_door":
                sum(1 for pf in problem_frame_entries if pf["problem_type"] == "open_door"),
            "closed_door":
                sum(1 for pf in problem_frame_entries if pf["problem_type"] == "closed_door"),
            "partially_closed":
                sum(1 for pf in problem_frame_entries if pf["problem_type"] == "partially_closed"),
        }

        inspection_data.update({
            "total_wagons": len(wagon_segments_side),
            "doors_open": doors_open,
            "doors_partially_closed": doors_partially_closed,
            "doors_closed": doors_closed,
            "damaged_wagons": damaged_wagons,
            "num_engines": n_engines,
            "total_loco_frames": len(loco_frame_entries),
            "total_problem_frames": len(problem_frame_entries),
            "problem_frames_by_type": problem_by_type,
            "wagon_number_results": wn_results_block,
            "loco_number_results": loco_results_block,
            "segment_type_map": segment_type_map,
            "wagon_segments": wagon_segments_side,
            "loco_frames": _build_loco_frames_block(loco_frame_entries, loco_numbers),
            "problem_frames": [
                _problem_frame_side_entry(pf, wagon_count_map, damage_lookup)
                for pf in problem_frame_entries
            ],
            "damage_model_active": damage_model_active,
        })

    return {
        "camera_id": _strip_camera_prefix(camera_folder),
        "version": version,
        "inspection_data": inspection_data,
    }


# ---------------------------------------------------------------------------
# Per-problem-frame transformers (per-flavour structures differ slightly)
# ---------------------------------------------------------------------------


def _problem_frame_top_entry(
    pf: dict,
    wagon_count_map: dict[int, Optional[int]],
    damage_lookup: dict[int, dict],
    segment_type_map: dict[str, dict],
) -> dict:
    seg_id = pf.get("wagon_id")
    seg_map_entry = segment_type_map.get(str(seg_id), {}) if seg_id is not None else {}
    return {
        "wagon_count": pf.get("wagon_count"),
        "segment_type": pf.get("segment_type"),
        "segment_number": seg_map_entry.get("number"),
        "problem_type": pf.get("problem_type"),
        "frame_number": pf.get("frame_number"),
        "filename": pf.get("filename"),
        "s3_key": pf.get("s3_key"),
        "s3_url": pf.get("s3_url"),
        "is_annotated": pf.get("is_annotated", False),
        "annotated_image_url": pf.get("annotated_image_url"),
        "bounding_box": pf.get("bounding_box"),
        "load_status": _load_status_for_pf(pf, damage_lookup),
        "load_condition": None,
        "damage_detected": pf.get("problem_type") in {"floor_dmg", "inner_wall_dmg"},
    }


def _problem_frame_side_entry(
    pf: dict,
    wagon_count_map: dict[int, Optional[int]],
    damage_lookup: dict[int, dict],
) -> dict:
    seg_id = pf.get("wagon_id")
    dmg = damage_lookup.get(seg_id, {}) if seg_id is not None else {}
    return {
        "wagon_count": pf.get("wagon_count"),
        "segment_type": pf.get("segment_type"),
        "segment_number": None,  # side notebooks do not surface this for problem frames
        "problem_type": pf.get("problem_type"),
        "frame_number": pf.get("frame_number"),
        "filename": pf.get("filename"),
        "s3_key": pf.get("s3_key"),
        "s3_url": pf.get("s3_url"),
        "is_annotated": pf.get("is_annotated", False),
        "annotated_image_url": pf.get("annotated_image_url"),
        "bounding_box": pf.get("bounding_box"),
        "door_status": dmg.get("door_status", "closed"),
        "door_close_detected": bool(dmg.get("door_close_detected")),
        "door_partial_detected": bool(dmg.get("door_partial_detected")),
        "damage_detected": bool(dmg.get("damage_detected")),
    }


def _load_status_for_pf(pf: dict, damage_lookup: dict[int, dict]) -> Optional[str]:
    seg_type = pf.get("segment_type")
    if seg_type == "wagon_loaded":
        return "loaded"
    if seg_type == "wagon_empty":
        return "empty"
    return None
