"""Deterministic evidence-frame selection for confirmed inspection findings.

TWO RULES.

First, EVERY REPORTED FINDING MUST BE SHOWABLE. Reporting "GW_17 has damage"
without being able to display the camera, frame, box, class and confidence that
justify it is not auditable. So each confirmed event gets at least one evidence
frame, or is explicitly marked evidence-unavailable.

Second, NO FLOOD OF NEAR-DUPLICATES. A finding observed on 51 consecutive frames
would yield 51 nearly identical crops. Selection is therefore capped and picks
frames that carry distinct information:

    peak      the highest-confidence sighting -- the best possible proof
    first     where the object entered, for temporal context
    last      where it left

Frames are chosen by INDEX only; the image is read later, so nothing here holds a
frame in memory. Ordering and tie-breaks are fully determined by the data, so the
same run always selects the same frames.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from .state import InspectionConfig, InspectionEvent
from .tracking import InspectionTrack


def select_evidence_frames(
    track: InspectionTrack,
    cfg: InspectionConfig,
) -> List[Dict[str, Any]]:
    """Pick up to `max_evidence_frames_per_event` sightings from one track.

    Deterministic: the peak is chosen by (-confidence, frame) so ties resolve by
    the earlier frame, and the remaining picks are taken in a fixed order.
    """
    if not track.observations:
        return []
    ordered = sorted(track.observations, key=lambda o: o.frame)
    peak = track.peak_observation

    wanted: List[Any] = [peak]
    for candidate in (ordered[0], ordered[-1]):
        if all(candidate.frame != w.frame for w in wanted):
            wanted.append(candidate)
    limit = max(1, int(cfg.max_evidence_frames_per_event))
    wanted = wanted[:limit]

    roles = {}
    roles[peak.frame] = "peak"
    roles.setdefault(ordered[0].frame, "first")
    roles.setdefault(ordered[-1].frame, "last")

    out: List[Dict[str, Any]] = []
    for o in sorted(wanted, key=lambda x: x.frame):
        out.append({
            "camera_id": track.camera_id,
            "frame": o.frame,
            "time_local": round(o.time_local, 4),
            "bbox": [round(v, 2) for v in o.bbox],
            "class_name": o.class_name,
            "class_id": o.class_id,
            "confidence": round(o.confidence, 4),
            "selection": roles.get(o.frame, "context"),
            "image_path": None,      # filled in when the image is extracted
            "available": True,
        })
    return out


def attach_evidence(
    events: Sequence[InspectionEvent],
    tracks_by_key: Dict[Any, InspectionTrack],
    cfg: InspectionConfig,
) -> None:
    """Attach selected frames to each event, in place.

    An event whose track is missing is marked evidence-unavailable rather than
    left silently empty -- the report must be able to distinguish "no evidence
    was selected" from "evidence exists but could not be produced".
    """
    for ev in events:
        track = tracks_by_key.get((ev.camera_id, ev.track_id))
        if track is None:
            ev.evidence_frames = [{
                "camera_id": ev.camera_id, "frame": ev.peak_frame,
                "available": False,
                "reason": "source track not retained, evidence unavailable",
            }]
            continue
        ev.evidence_frames = select_evidence_frames(track, cfg)


def extract_evidence_images(
    events: Sequence[InspectionEvent],
    video_paths: Dict[str, str],
    out_dir: str,
    draw_box: bool = True,
    verbose: bool = False,
) -> int:
    """Write one image per selected evidence frame. Returns how many were written.

    Frames are grouped by camera and read in ASCENDING frame order with a single
    capture per camera, so a video is opened once and never seeked backwards --
    the memory- and IO-cheap ordering. Each image is written and released
    immediately; none is retained.
    """
    import cv2                                          # noqa: WPS433 (lazy)

    os.makedirs(out_dir, exist_ok=True)
    wanted: Dict[str, List[Any]] = {}
    for ev in events:
        for frame_rec in ev.evidence_frames:
            if not frame_rec.get("available", True):
                continue
            wanted.setdefault(frame_rec["camera_id"], []).append((ev, frame_rec))

    written = 0
    for cam, items in sorted(wanted.items()):
        path = video_paths.get(cam)
        if not path or not os.path.isfile(path):
            for _, rec in items:
                rec["available"] = False
                rec["reason"] = f"video for {cam} unavailable"
            continue
        items.sort(key=lambda pair: pair[1]["frame"])
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            for _, rec in items:
                rec["available"] = False
                rec["reason"] = f"cannot open {cam} video"
            continue
        try:
            for ev, rec in items:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(rec["frame"]))
                ok, img = cap.read()
                if not ok or img is None:
                    rec["available"] = False
                    rec["reason"] = f"frame {rec['frame']} unreadable"
                    continue
                if draw_box and rec.get("bbox"):
                    x1, y1, x2, y2 = (int(v) for v in rec["bbox"])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = (f"{ev.global_id or 'UNRESOLVED'} "
                             f"{rec['class_name']} {rec['confidence']:.2f}")
                    cv2.putText(img, label, (x1, max(18, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                                cv2.LINE_AA)
                name = (f"{ev.event_id}_{cam}_f{int(rec['frame']):06d}_"
                        f"{rec['selection']}.jpg")
                dest = os.path.join(out_dir, name)
                cv2.imwrite(dest, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                rec["image_path"] = dest
                written += 1
                del img
        finally:
            cap.release()
    if verbose:
        print(f"    [EVIDENCE] wrote {written} inspection evidence frame(s)")
    return written
