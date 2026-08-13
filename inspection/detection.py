"""Run the inspection models over video and produce the global inspection state.

MEMORY POLICY -- read this before changing anything here
-------------------------------------------------------
The EC2 box has already suffered an OOM in a large Python process, so this module
is written to hold O(detections), never O(frames):

  * Cameras are processed STRICTLY SEQUENTIALLY. One capture is open at a time and
    is released before the next opens.
  * A decoded frame is used for inference and then dropped. Frames are never
    accumulated, never copied wholesale, and never stored on a track.
  * Evidence keeps only (camera, frame index, bbox) and the frame is re-read from
    the video later, when the report needs it -- the same strategy the existing
    evidence report already uses.
  * Each model is loaded ONCE and reused for every camera it applies to.
  * A per-frame detection ceiling stops a pathological frame from ballooning the
    observation list.

WHICH MODEL RUNS ON WHICH CAMERA
--------------------------------
Doors are on wagon sides, damage is seen from above, so the door model runs on the
side cameras and the damage model on the top cameras. A camera a role does not
apply to is reported NOT_APPLICABLE -- explicitly distinct from "inspected and
found nothing".
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .association import (
    associate_track, camera_visibility, trusted_offsets, wagon_intervals_from_state,
)
from .models import DAMAGE_ROLE, DOOR_ROLE, InspectionModel, ModelAvailability
from .state import (
    ASSOCIATION_RESOLVED, CAMERA_CONFIRMED, CAMERA_NOT_APPLICABLE,
    InspectionConfig, InspectionEvent, InspectionState, WagonInspection,
)
from .tracking import (
    DetectionObservation, InspectionTrack, classify_track, track_detections,
)


def default_role_cameras(all_cameras: Sequence[str]) -> Dict[str, List[str]]:
    """Map each role to the cameras it applies to, by camera NAME convention.

    Derived from the project's own naming ('..._TOP' are the overhead views)
    rather than a hardcoded list, so a deployment with a different camera set
    still routes correctly.
    """
    tops = [c for c in all_cameras if str(c).upper().endswith("_TOP")]
    sides = [c for c in all_cameras if c not in tops]
    return {DOOR_ROLE: sides, DAMAGE_ROLE: tops}


# ---------------------------------------------------------------------------
# per-camera inference
# ---------------------------------------------------------------------------

def iter_frame_detections(
    model: InspectionModel,
    video_path: str,
    cfg: InspectionConfig,
    fps: float,
    total_frames: int,
    verbose: bool = False,
) -> Iterable[Tuple[int, float, List[DetectionObservation]]]:
    """Yield (frame, time_local, observations) for every sampled frame.

    A generator, so the caller can stream frames into the tracker without any
    list of frames ever existing. Sequential reads (no random seeking) because
    seeking per frame is markedly slower on long videos.
    """
    import cv2                                          # noqa: WPS433 (lazy)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for inspection: {video_path}")
    stride = max(1, int(cfg.frame_stride))
    handle = model.handle
    frame_idx = 0
    try:
        while True:
            ok = cap.grab()             # cheap: advances without decoding
            if not ok:
                break
            if frame_idx % stride == 0:
                ok, img = cap.retrieve()
                if not ok or img is None:
                    frame_idx += 1
                    continue
                res = handle.predict(img, conf=cfg.min_detection_confidence,
                                     verbose=False)[0]
                obs: List[DetectionObservation] = []
                boxes = getattr(res, "boxes", None)
                if boxes is not None:
                    for b in boxes:
                        cid = int(b.cls.item())
                        conf = float(b.conf.item())
                        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                        obs.append(DetectionObservation(
                            frame=frame_idx,
                            time_local=(frame_idx / fps) if fps else 0.0,
                            class_id=cid, class_name=model.class_name(cid),
                            confidence=conf, bbox=(x1, y1, x2, y2)))
                del img                  # drop the frame immediately
                yield frame_idx, (frame_idx / fps) if fps else 0.0, obs
            frame_idx += 1
    finally:
        cap.release()


def inspect_camera(
    model: InspectionModel,
    camera_id: str,
    video_path: str,
    fps: float,
    frame_width: int,
    total_frames: int,
    cfg: InspectionConfig,
    verbose: bool = True,
) -> Tuple[List[InspectionTrack], Dict[str, Any]]:
    """Detect, then track, for ONE camera. Returns (tracks, statistics)."""
    t0 = time.time()
    raw_count = 0

    def _counting_stream():
        nonlocal raw_count
        for frame, t_local, obs in iter_frame_detections(
                model, video_path, cfg, fps, total_frames, verbose):
            raw_count += len(obs)
            yield frame, t_local, obs

    tracks = track_detections(_counting_stream(), camera_id, model.role, cfg,
                              fps=fps, frame_width=frame_width)
    elapsed = time.time() - t0
    stats = {
        "camera_id": camera_id, "role": model.role,
        "raw_detections": raw_count, "tracks": len(tracks),
        "frame_stride": cfg.frame_stride,
        "seconds": round(elapsed, 2),
    }
    if verbose:
        print(f"    [{model.role.upper()}/{camera_id}] raw_detections={raw_count} "
              f"tracks={len(tracks)}  ({elapsed:.1f}s)")
    return tracks, stats


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run_inspection(
    state: Any,
    tracks_by_camera: Dict[str, Any],
    availability: Dict[str, ModelAvailability],
    all_cameras: Sequence[str],
    cfg: Optional[InspectionConfig] = None,
    role_cameras: Optional[Dict[str, List[str]]] = None,
    verbose: bool = True,
) -> InspectionState:
    """Detect, track, confirm and associate -- then annotate the global wagons.

    `state` is READ ONLY. The wagon roster and camera offsets are consumed; not
    one field of them is written. The returned `InspectionState` is a separate
    object that references wagons by their existing global ids.
    """
    cfg = cfg or InspectionConfig()
    role_cameras = role_cameras or default_role_cameras(all_cameras)

    insp = InspectionState(
        enabled=cfg.enabled,
        model_availability={r: a.to_dict() for r, a in availability.items()},
        config=_describe_config(cfg))

    wagons = wagon_intervals_from_state(state)
    offsets = trusted_offsets(state)

    # Seed one record per EXISTING wagon, so wagons with no findings are still
    # present in the report rather than silently missing.
    for w in wagons:
        insp.wagons[w.global_id] = WagonInspection(
            global_id=w.global_id, classification=w.classification)

    # Pre-fill every (wagon, camera) with what the camera COULD have seen. This
    # is what stops "no detection" being read as "no damage".
    for role, cams in sorted(role_cameras.items()):
        for cam in all_cameras:
            if cam not in cams:
                for w in wagons:
                    insp.wagons[w.global_id].camera_status.setdefault(cam, {})[
                        role] = CAMERA_NOT_APPLICABLE
                continue
            lct = tracks_by_camera.get(cam)
            duration = None
            if lct is not None and getattr(lct, "fps", 0):
                duration = getattr(lct, "total_frames", 0) / float(lct.fps)
            vis = camera_visibility(wagons, cam, offsets, duration)
            for gid, status in vis.items():
                insp.wagons[gid].camera_status.setdefault(cam, {})[role] = status

    if not cfg.enabled:
        insp.warnings.append("inspection disabled by configuration")
        return insp

    event_seq = 0
    for role in (DOOR_ROLE, DAMAGE_ROLE):
        avail = availability.get(role)
        if avail is None or not avail.is_available:
            reason = avail.reason if avail else "not configured"
            insp.warnings.append(f"{role} inspection skipped: {reason}")
            if verbose:
                print(f"  [{role.upper()}] SKIPPED -- {reason}")
            continue

        cams = [c for c in role_cameras.get(role, []) if c in tracks_by_camera]
        if not cams:
            insp.warnings.append(f"{role} inspection: no applicable camera present")
            continue

        role_t0 = time.time()
        # ONE model handle for every camera in this role.
        with InspectionModel(avail) as model:
            for cam in cams:
                lct = tracks_by_camera[cam]
                video = getattr(lct, "video_path", "")
                if not video or not os.path.isfile(video):
                    insp.warnings.append(
                        f"{role}/{cam}: video unavailable, camera not inspected")
                    continue
                try:
                    cam_tracks, stats = inspect_camera(
                        model, cam, video, fps=float(getattr(lct, "fps", 0.0)),
                        frame_width=int(getattr(lct, "width", 0)),
                        total_frames=int(getattr(lct, "total_frames", 0)),
                        cfg=cfg, verbose=verbose)
                except Exception as exc:
                    # A model failure must be reported clearly, not swallowed and
                    # not allowed to destroy a completed wagon count.
                    msg = (f"{role}/{cam}: inference failed: "
                           f"{type(exc).__name__}: {exc}")
                    insp.warnings.append(msg)
                    if verbose:
                        print(f"    WARNING {msg}")
                    continue

                insp.per_camera.setdefault(cam, {})[role] = stats
                for tr in cam_tracks:
                    insp.tracks_by_key[(cam, tr.track_id)] = tr
                confirmed_n = 0
                for tr in cam_tracks:
                    event_seq += 1
                    ok, why = classify_track(tr, cfg,
                                             int(getattr(lct, "width", 0)))
                    ev = _event_from_track(tr, model, event_seq, ok, why)
                    if ok:
                        assoc = associate_track(tr, wagons, offsets)
                        _apply_association(ev, assoc)
                        insp.events.append(ev)
                        confirmed_n += 1
                        if ev.global_id and ev.global_id in insp.wagons:
                            wi = insp.wagons[ev.global_id]
                            bucket = (wi.door_events if _is_door_state(model, tr)
                                      else wi.damage_events)
                            bucket.append(ev)
                            wi.camera_status.setdefault(cam, {})[role] = \
                                CAMERA_CONFIRMED
                    else:
                        insp.rejected_events.append(ev)
                stats["confirmed_tracks"] = confirmed_n
                stats["rejected_tracks"] = len(cam_tracks) - confirmed_n
        insp.timings[f"{role}_seconds"] = round(time.time() - role_t0, 2)

    return insp


def _is_door_state(model: InspectionModel, track: InspectionTrack) -> bool:
    """Does this track carry a DOOR-STATE class, or a DAMAGE class?

    Decided per TRACK, not per model, because the door model ships a `damage`
    class: a `damage` track from door_state.pt is a damage finding and must not be
    filed as a door state.
    """
    groups = model.availability.class_groups or {}
    _, cname, _ = track.dominant_class()
    if cname in (groups.get("door_state") or []):
        return True
    if cname in (groups.get("damage") or []):
        return False
    # Unrecognised class: file it under the model's own role so it is still
    # reported rather than dropped.
    return model.role == DOOR_ROLE


def _event_from_track(track: InspectionTrack, model: InspectionModel,
                      seq: int, confirmed: bool, reason: str) -> InspectionEvent:
    cid, cname, _ = track.dominant_class()
    peak = track.peak_observation
    return InspectionEvent(
        event_id=f"{model.role.upper()}_{track.camera_id}_{seq:04d}",
        role=model.role, model_path=str(model.availability.path or ""),
        model_class_id=cid, model_class_name=cname,
        camera_id=track.camera_id, track_id=track.track_id,
        start_frame=track.start_frame, end_frame=track.end_frame,
        start_time_local=track.start_time, end_time_local=track.end_time,
        n_observations=track.n_observations,
        peak_confidence=track.peak_confidence,
        mean_confidence=track.mean_confidence,
        peak_frame=peak.frame, peak_bbox=list(peak.bbox),
        displacement_px=track.displacement_px,
        confirmed=confirmed, rejection_reason=reason)


def _apply_association(ev: InspectionEvent, assoc: Dict[str, Any]) -> None:
    ev.global_id = assoc["global_id"]
    ev.association_status = assoc["association_status"]
    ev.association_method = assoc["association_method"]
    ev.association_detail = assoc["association_detail"]
    ev.global_time_start = assoc["global_time_start"]
    ev.global_time_end = assoc["global_time_end"]
    ev.camera_offset = assoc["camera_offset"]
    ev.overlap_fraction = assoc["overlap_fraction"]
    ev.candidate_global_ids = list(assoc["candidate_global_ids"])


def _describe_config(cfg: InspectionConfig) -> Dict[str, Any]:
    return {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
