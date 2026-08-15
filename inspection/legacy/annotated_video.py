"""Combined annotated ("detected") video builder.

Runs *after* gap/loco detection, wagon segmentation, and damage detection have
all completed, so the emitted video can draw gap boxes, loco boxes, and damage
boxes together — instead of the old gap-only video written mid-pipeline
before damage detection even ran.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import pandas as pd

from .damage import SIDE_CLASS_STYLE, TOP_CLASS_STYLE
from .video_io import compress_video


GAP_STYLE = {"color": (0, 255, 255), "label": "GAP"}          # yellow (BGR)
LOCO_STYLE = {"color": (255, 128, 0), "label": "LOCO"}         # orange-blue (BGR)


def _boxes_by_frame(csv_path: Optional[str]) -> dict[int, list[dict]]:
    """Load a gap/loco detections CSV into {frame -> [detection, ...]}."""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001
        return {}
    out: dict[int, list[dict]] = {}
    for _, row in df.iterrows():
        out.setdefault(int(row["frame"]), []).append({
            "class_name": row["class_name"],
            "confidence": float(row["confidence"]),
            "x1": float(row["x1"]), "y1": float(row["y1"]),
            "x2": float(row["x2"]), "y2": float(row["y2"]),
        })
    return out


def _damage_boxes_by_frame(
    damage_frame_detections_df: Optional[pd.DataFrame],
) -> dict[int, list[dict]]:
    if damage_frame_detections_df is None or damage_frame_detections_df.empty:
        return {}
    out: dict[int, list[dict]] = {}
    for _, row in damage_frame_detections_df.iterrows():
        out.setdefault(int(row["frame_number"]), []).append({
            "class_name": row["class_name"],
            "confidence": float(row["confidence"]),
            "x1": float(row["x1"]), "y1": float(row["y1"]),
            "x2": float(row["x2"]), "y2": float(row["y2"]),
        })
    return out


def _draw_box(frame, det: dict, style: dict) -> None:
    x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
    color = style["color"]
    label = f"{style['label']} {det['confidence']:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y = max(y1, th + 8)
    cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw, label_y), color, -1)
    cv2.putText(
        frame, label, (x1, label_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )


def build_annotated_video(
    video_path: str,
    output_dir: str,
    raw_video_name: str,
    gap_csv: Optional[str],
    loco_csv: Optional[str],
    damage_frame_detections_df: Optional[pd.DataFrame],
    flavour: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Draw gap + loco + damage boxes onto every frame of ``video_path``,
    writing ``{raw_video_name}_detected_video.mp4``.

    Returns the output path, or None if the source video can't be opened.
    """
    log = logger or logging.getLogger(__name__)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning("Cannot open video for annotation: %s", video_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = os.path.join(output_dir, f"{raw_video_name}_detected_video.mp4")
    # cv2.VideoWriter's mp4v codec has no rate control and produces bloated
    # files, so frames are drawn into a raw intermediate here and then
    # re-encoded to H.264 (compress_video) before being handed back — the
    # raw file never leaves this function.
    raw_video = os.path.join(output_dir, f"{raw_video_name}_detected_video_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_video, fourcc, fps, (width, height))

    gap_by_frame = _boxes_by_frame(gap_csv)
    loco_by_frame = _boxes_by_frame(loco_csv)
    damage_by_frame = _damage_boxes_by_frame(damage_frame_detections_df)
    damage_style = TOP_CLASS_STYLE if flavour == "top" else SIDE_CLASS_STYLE

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        for det in gap_by_frame.get(frame_number, []):
            _draw_box(frame, det, GAP_STYLE)
        for det in loco_by_frame.get(frame_number, []):
            _draw_box(frame, det, LOCO_STYLE)
        for det in damage_by_frame.get(frame_number, []):
            style = damage_style.get(
                det["class_name"], {"color": (255, 255, 0), "label": det["class_name"].upper()}
            )
            _draw_box(frame, det, style)

        writer.write(frame)

    cap.release()
    writer.release()
    log.info("Raw annotated video written: %s (%d frames)", raw_video, frame_number)

    duration_sec = frame_number / fps if fps else 0.0
    try:
        compress_video(raw_video, output_video, log, duration_sec=duration_sec)
    finally:
        try:
            os.remove(raw_video)
        except OSError:
            pass

    log.info(
        "Annotated video compressed: %s (%.1f MB)",
        output_video, os.path.getsize(output_video) / 1e6,
    )
    return output_video
