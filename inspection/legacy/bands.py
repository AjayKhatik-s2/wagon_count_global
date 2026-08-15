"""Detection-band identification from per-frame detections."""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import pandas as pd


def identify_bands(
    csv_path: Optional[str],
    confidence_threshold: float,
    gap_tolerance: int,
    band_type: str,
    logger: Optional[logging.Logger] = None,
    class_filter: Optional[str] = None,
    min_band_frames: int = 1,
) -> Tuple[List[dict], pd.DataFrame]:
    """Group sorted frame indices into 'bands' separated by a configurable gap.

    ``class_filter`` restricts input rows to a single detection class
    (case-insensitive, matched against the ``class_name`` column) before
    banding — e.g. isolating ``locono`` (loco-number plate) detections from
    ``engine_head`` ones when both share the same loco detections CSV.
    Ignored if the CSV has no ``class_name`` column.

    ``min_band_frames`` implements "Minimum Track Duration": a band supported by
    fewer than this many distinct frames is discarded. A band is this pipeline's
    equivalent of a track, and with a ``gap_tolerance`` of 15-18 frames a genuine
    gap or loco is observed over many frames as it passes, so a one- or two-frame
    band is detector noise. It matters most for **gap** bands, where a spurious
    band inserts a false wagon boundary and splits one wagon into two. The filter
    runs on the already-grouped band list — no extra pass over the CSV. ``1``
    (the default) disables it, preserving the original behaviour for callers that
    do not opt in.
    """
    log = logger or logging.getLogger(__name__)

    if not csv_path or not os.path.exists(csv_path):
        log.info("No %s detections", band_type)
        return [], pd.DataFrame()

    df_all = pd.read_csv(csv_path)
    if class_filter and "class_name" in df_all.columns:
        df_all = df_all[df_all["class_name"].str.lower() == class_filter.lower()]
    df = df_all[df_all["confidence"] >= confidence_threshold].copy()
    frames = sorted(df["frame"].unique().tolist())
    if not frames:
        return [], pd.DataFrame()

    bands: List[dict] = []
    current = {
        "band_id": 1,
        "start_frame": frames[0],
        "end_frame": frames[0],
        "frames": [frames[0]],
    }

    for i in range(1, len(frames)):
        gap = frames[i] - frames[i - 1]
        if gap <= gap_tolerance + 1:
            current["end_frame"] = frames[i]
            current["frames"].append(frames[i])
        else:
            bands.append(dict(current))
            current = {
                "band_id": len(bands) + 1,
                "start_frame": frames[i],
                "end_frame": frames[i],
                "frames": [frames[i]],
            }
    bands.append(current)

    # Minimum Track Duration — drop noise bands, then renumber so band_id stays a
    # dense 1..N sequence (WagonSegmenter sorts on band_id and pairs consecutive
    # bands, so a gap in the numbering would be harmless but confusing in logs).
    if min_band_frames > 1:
        kept = [b for b in bands if len(set(b["frames"])) >= min_band_frames]
        n_dropped = len(bands) - len(kept)
        if n_dropped:
            for band in bands:
                if len(set(band["frames"])) < min_band_frames:
                    log.info(
                        "REJECT band type=%s band_id=%d frames=%d-%d n_frames=%d "
                        "validation=min_track_duration reason=fewer than %d frame(s)",
                        band_type, band["band_id"], band["start_frame"],
                        band["end_frame"], len(set(band["frames"])), min_band_frames,
                    )
            for new_id, band in enumerate(kept, start=1):
                band["band_id"] = new_id
            bands = kept
        if not bands:
            log.info("All %s bands rejected by min_track_duration", band_type)
            return [], pd.DataFrame()

    summary_rows = []
    for band in bands:
        band_frames = band["frames"]
        band_dets = df[df["frame"].isin(band_frames)]
        summary_rows.append({
            "band_id": band["band_id"],
            "start_frame": band["start_frame"],
            "end_frame": band["end_frame"],
            "total_frames": len(band_frames),
            "total_detections": len(band_dets),
            "avg_confidence": float(band_dets["confidence"].mean()),
        })

    log.info("Identified %d %s bands", len(bands), band_type)
    return bands, pd.DataFrame(summary_rows)


def analyze_detection_bands(
    detections: List[tuple], gap_tolerance: int = 6
) -> List[dict]:
    """Group ``(frame, confidence, ...)`` detections into bands.

    Used by per-class damage analysis where each detection carries extra
    bbox metadata in trailing tuple slots.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda x: x[0])
    bands: List[dict] = []
    current = {
        "band_id": 1,
        "start_frame": sorted_dets[0][0],
        "end_frame": sorted_dets[0][0],
        "frames": [sorted_dets[0][0]],
        "confidences": [sorted_dets[0][1]],
    }
    for det in sorted_dets[1:]:
        frame_num, conf = det[0], det[1]
        if frame_num - current["end_frame"] <= gap_tolerance + 1:
            current["end_frame"] = frame_num
            if frame_num not in current["frames"]:
                current["frames"].append(frame_num)
            current["confidences"].append(conf)
        else:
            bands.append(dict(current))
            current = {
                "band_id": len(bands) + 1,
                "start_frame": frame_num,
                "end_frame": frame_num,
                "frames": [frame_num],
                "confidences": [conf],
            }
    bands.append(current)

    for band in bands:
        band["frame_count"] = len(set(band["frames"]))
        band["avg_confidence"] = sum(band["confidences"]) / len(band["confidences"])
        band["detection_count"] = len(band["confidences"])
    return bands
