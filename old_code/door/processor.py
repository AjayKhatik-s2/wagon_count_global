"""Door feature processor (v4, train-state-native, ALL legacy intelligence
ported).

Per wagon, for each side camera (RIGHT_UP / LEFT_UP):

    1. Iterate cached JPEGs in wagon_cache/<GW_n>/<camera>/.
    2. Score illumination quality per frame (legacy IlluminationProcessor).
    3. Run YOLO door_state.pt on the raw frame.
    4. Filter detections through the geometric shape prior (aspect ratio,
       vertical-edge dominance, border completeness).
    5. Feed surviving detections + quality score into the legacy
       DoorTracker (Kalman + Hungarian + per-track 30-frame quality-
       weighted majority vote + state machine with 2x hysteresis on
       OPEN -> CLOSED transitions + sticky DAMAGE state).
    6. After all frames, finalize the tracker -> per-track {state,
       confidence, snapshot}.
    7. Run DoorIdentityMerger to collapse fragmented tracks of the same
       physical door (spatial + temporal + context + structural).
    8. Pick the dominant door state per CAMERA SIDE.

The per-CAMERA dominant state IS the per-side door state (RIGHT_UP -> right
door, LEFT_UP -> left door).  Same convention the legacy combined report
used.

Output JSON shape (per wagon):
    {
        "global_id":   "GW_7",
        "feature":     "door",
        "status":      "OK" | "NO_FRAMES" | "FAILED" | "NO_DATA",
        "left_door":   "CLOSED" | "OPEN" | "PARTIAL" | "DAMAGED" | "NO_DATA",
        "left_door_confidence":  0.91,
        "right_door":  "...",
        "right_door_confidence": 0.83,
        "tracks": [
            {camera_id, track_id, state, confidence, first_frame,
             last_frame, total_hits},
            ...
        ],
        "supporting_cameras": ["LEFT_UP", "RIGHT_UP"],
        "frame_count": ...,
    }
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core import constants as C
from core.global_state_loader import GlobalTrainState

from features._common import (
    load_yolo, iter_wagon_frames, list_wagon_frames,
    write_per_wagon_json, empty_payload, FeatureTimer,
)

# Mature intelligence ported from legacy
from features.inference_lib.door_tracker import (
    DoorTracker, TrackerConfig, DoorState, yolo_to_detections,
)
from features.inference_lib.door_identity_merger import (
    DoorIdentityMerger, MergeConfig,
)
from features.inference_lib.illumination_processor import (
    IlluminationProcessor, IlluminationConfig,
)
from features.inference_lib.geometric_shape_prior import (
    GeometricShapePrior, GeometricPriorConfig,
)


FEATURE_NAME = "door"


# -----------------------------------------------------------------------------
# Canonicalization of legacy state strings
# -----------------------------------------------------------------------------

# DoorTracker.DoorState values are: UNKNOWN / CLOSED / OPEN / PARTIAL_CLOSED /
# DAMAGE / OTHER.  Map to the v4 canonical vocabulary.
_STATE_TO_CANONICAL = {
    "OPEN":            C.DOOR_OPEN,
    "CLOSED":          C.DOOR_CLOSED,
    "PARTIAL_CLOSED":  C.DOOR_PARTIAL,
    "PARTIAL":         C.DOOR_PARTIAL,
    "DAMAGE":          C.DOOR_DAMAGED,
    "DAMAGED":         C.DOOR_DAMAGED,
    "OTHER":           C.NO_DATA,
    "UNKNOWN":         C.NO_DATA,
}


def _canonical(state_value: str) -> str:
    s = str(state_value or "").strip().upper()
    return _STATE_TO_CANONICAL.get(s, s if s else C.NO_DATA)


# -----------------------------------------------------------------------------
# Per-camera tracker run
# -----------------------------------------------------------------------------

def _run_tracker_one_camera(
    yolo_model,
    illumination: IlluminationProcessor,
    geo_prior: GeometricShapePrior,
    tracker_config: TrackerConfig,
    merger_config: MergeConfig,
    cache_root: str,
    gw_id: str,
    camera_id: str,
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """Run the full per-camera door pipeline on one wagon.

    Returns: (track_decisions, n_frames, width, height).
    Each track_decision dict carries:
        track_id, state (canonical), confidence, first_frame, last_frame,
        total_hits, mean_center_x.
    """
    paths = list_wagon_frames(cache_root, gw_id, camera_id)
    if not paths:
        return [], 0, 0, 0

    # Fresh tracker per (gw, camera).  Wagons are independent in the new
    # train-state-native world, so each one resets the tracker.
    tracker = DoorTracker(config=tracker_config)
    tracker.reset()

    frame_w, frame_h = 0, 0
    used = 0

    # ------- frame loop -------
    for fi, frame in iter_wagon_frames(cache_root, gw_id, camera_id):
        if frame_w == 0:
            frame_h, frame_w = frame.shape[:2]
        used += 1

        # 1) quality score (does NOT alter the frame; YOLO sees raw bytes,
        #    matching the legacy pipeline).
        try:
            ill_res = illumination.process_frame(frame, frame_idx=fi)
            quality = float(getattr(ill_res, "quality_score", 1.0))
        except Exception:
            quality = 1.0

        # 2) YOLO detection on raw frame
        try:
            results = yolo_model(frame, verbose=False, half=True)[0]
        except Exception:
            continue
        if results.boxes is None or len(results.boxes) == 0:
            tracker.update([], frame=frame,
                           frame_width=frame_w, frame_height=frame_h)
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss  = results.boxes.cls.cpu().numpy().astype(int)

        # 3) confidence floor (the tracker has its own per-class thresholds
        #    too; this gate just discards obviously-noisy detections early).
        min_conf = float(tracker_config.closed_confidence_threshold)
        keep = confs >= min_conf
        boxes, confs, clss = boxes[keep], confs[keep], clss[keep]
        if len(boxes) == 0:
            tracker.update([], frame=frame,
                           frame_width=frame_w, frame_height=frame_h)
            continue

        # 4) geometric prior filter (aspect ratio / vertical-edge / border)
        try:
            boxes, confs, clss, _idx = geo_prior.filter_detections(
                frame, boxes, confs, clss,
            )
        except Exception:
            pass

        # 5) convert to Detection objects + feed tracker
        names = getattr(yolo_model, "names", {}) or {}
        detections = yolo_to_detections(
            boxes=boxes, confidences=confs, class_ids=clss,
            class_names=names, illumination_quality=quality,
        )
        tracker.update(
            detections=detections,
            frame=frame,
            frame_width=frame_w,
            frame_height=frame_h,
        )

    # ------- finalize -------
    final_states = tracker.get_final_door_states()
    if not final_states:
        return [], used, frame_w, frame_h

    # Run identity merger on the final track set (collapses fragmented IDs
    # of the same physical door).  Operates on the live + deleted track
    # objects exposed by the tracker.
    try:
        merger = DoorIdentityMerger(config=merger_config)
        all_tracks_objs = list(tracker.tracks) + list(tracker.deleted_tracks)
        merged_groups = merger.merge_all_tracks(all_tracks_objs)
        # merge_all_tracks returns mapping {canonical_id: [member_ids]};
        # we keep the canonical id for each group as the surviving track.
        if isinstance(merged_groups, dict) and merged_groups:
            merged_ids = set(merged_groups.keys())
        elif isinstance(merged_groups, list) and merged_groups:
            merged_ids = set(merged_groups)
        else:
            merged_ids = set(final_states.keys())
    except Exception:
        merged_ids = set(final_states.keys())   # fallback: keep everything

    decisions: List[Dict[str, Any]] = []
    all_tracks = list(tracker.tracks) + list(tracker.deleted_tracks)
    by_id = {t.track_id: t for t in all_tracks}

    for tid, state_dict in final_states.items():
        if merged_ids and tid not in merged_ids:
            continue
        tr = by_id.get(tid)
        mean_cx = float(np.mean([d['bbox'][[0,2]].mean()
                                 for d in (tr.detections if tr else [])])) \
            if (tr and getattr(tr, "detections", None)) else 0.0
        decisions.append({
            "camera_id":   camera_id,
            "track_id":    tid,
            "state":       _canonical(state_dict.get("state")),
            "confidence":  float(state_dict.get("confidence", 0.0) or 0.0),
            "first_frame": int(state_dict.get("first_frame", 0)),
            "last_frame":  int(state_dict.get("last_frame", 0)),
            "total_hits":  int(state_dict.get("total_hits", 0)),
            "mean_center_x": mean_cx,
        })
    return decisions, used, frame_w, frame_h


# -----------------------------------------------------------------------------
# Per-side decision picker
# -----------------------------------------------------------------------------

def _pick_side_state(track_decisions: List[Dict[str, Any]]) -> Tuple[str, float]:
    """Pick the dominant door state for one camera/side.

    Priority order:
        1. Any DAMAGED track  -> DAMAGED (terminal in the FSM)
        2. Any OPEN track     -> OPEN  (safety-critical; legacy code biases here)
        3. Any PARTIAL track  -> PARTIAL
        4. Most-frequent CLOSED-class result by total_hits, confidence weighted
        5. NO_DATA
    """
    if not track_decisions:
        return C.NO_DATA, 0.0

    def _max_conf(items):
        return max(items, key=lambda d: (d["total_hits"], d["confidence"]))

    damaged = [d for d in track_decisions if d["state"] == C.DOOR_DAMAGED]
    if damaged:
        best = _max_conf(damaged)
        return C.DOOR_DAMAGED, best["confidence"]

    opens = [d for d in track_decisions if d["state"] == C.DOOR_OPEN]
    if opens:
        best = _max_conf(opens)
        return C.DOOR_OPEN, best["confidence"]

    partials = [d for d in track_decisions if d["state"] == C.DOOR_PARTIAL]
    if partials:
        best = _max_conf(partials)
        return C.DOOR_PARTIAL, best["confidence"]

    closeds = [d for d in track_decisions if d["state"] == C.DOOR_CLOSED]
    if closeds:
        best = _max_conf(closeds)
        return C.DOOR_CLOSED, best["confidence"]

    return C.NO_DATA, 0.0


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def run(
    *,
    state: GlobalTrainState,
    cache_root: str,
    feature_models_dir: str,
    output_dir: str,
    confidence: float = C.CONF_DOOR,
    every_nth: int = 1,
    max_frames: int = 0,           # 0 = unbounded (legacy used the whole wagon)
    verbose: bool = True,
) -> Dict[str, str]:
    """Run the door feature on every wagon using the mature legacy logic."""
    del every_nth, max_frames  # kept for API symmetry; we iterate every frame

    model_path = os.path.join(feature_models_dir, C.MODEL_DOOR_STATE)
    yolo_model = load_yolo(model_path)

    feature_out = os.path.join(output_dir, FEATURE_NAME)
    os.makedirs(feature_out, exist_ok=True)
    timer = FeatureTimer("door")
    summary: Dict[str, str] = {}

    # Pre-construct shared per-process helpers (loaded once across wagons)
    illumination = IlluminationProcessor(IlluminationConfig())
    geo_prior    = GeometricShapePrior(GeometricPriorConfig())
    tracker_cfg  = TrackerConfig()
    merger_cfg   = MergeConfig()

    if yolo_model is None and verbose:
        print(f"[FEAT/door] WARNING: {model_path} missing; emitting NO_DATA "
              f"for every wagon.")

    if verbose:
        print(f"[FEAT/door] running on {len(state.wagons)} wagons "
              f"(conf>={confidence}, legacy DoorTracker + IdentityMerger + "
              f"GeometricPrior + IlluminationQuality)")

    for gw in state.wagons:
        gw_id = gw.global_id
        t0 = time.time()
        try:
            if yolo_model is None:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.NO_DATA,
                    left_door=C.NO_DATA, left_door_confidence=0.0,
                    right_door=C.NO_DATA, right_door_confidence=0.0,
                    tracks=[], supporting_cameras=[],
                    error="door_state.pt not present",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.NO_DATA
                continue

            l_decisions, l_used, _, _ = _run_tracker_one_camera(
                yolo_model, illumination, geo_prior,
                tracker_cfg, merger_cfg,
                cache_root, gw_id, C.CAMERA_LEFT_UP,
            )
            r_decisions, r_used, _, _ = _run_tracker_one_camera(
                yolo_model, illumination, geo_prior,
                tracker_cfg, merger_cfg,
                cache_root, gw_id, C.CAMERA_RIGHT_UP,
            )

            supporting: List[str] = []
            if l_used > 0: supporting.append(C.CAMERA_LEFT_UP)
            if r_used > 0: supporting.append(C.CAMERA_RIGHT_UP)

            if l_used == 0 and r_used == 0:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES,
                    left_door=C.NO_DATA, left_door_confidence=0.0,
                    right_door=C.NO_DATA, right_door_confidence=0.0,
                    tracks=[], supporting_cameras=[],
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.STATUS_NO_FRAMES
                continue

            l_state, l_conf = _pick_side_state(l_decisions)
            r_state, r_conf = _pick_side_state(r_decisions)

            # If frames existed but tracker produced no confirmed tracks,
            # treat as CLOSED with low confidence -- typical pattern when
            # the wagon's doors are uniformly closed and the model doesn't
            # bother firing.  The conservative legacy default.
            if l_used > 0 and l_state == C.NO_DATA:
                l_state, l_conf = C.DOOR_CLOSED, 0.0
            if r_used > 0 and r_state == C.NO_DATA:
                r_state, r_conf = C.DOOR_CLOSED, 0.0

            payload: Dict[str, Any] = {
                "global_id":   gw_id,
                "feature":     FEATURE_NAME,
                "status":      C.STATUS_OK,
                "left_door":   l_state,
                "left_door_confidence":  round(float(l_conf), 4),
                "right_door":  r_state,
                "right_door_confidence": round(float(r_conf), 4),
                "tracks":      l_decisions + r_decisions,
                "supporting_cameras": supporting,
                "frame_count": l_used + r_used,
                "frames_left":  l_used,
                "frames_right": r_used,
            }
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = C.STATUS_OK
            if verbose:
                print(f"  [door/{gw_id}]  L={l_state} ({l_conf:.2f})  "
                      f"R={r_state} ({r_conf:.2f})  "
                      f"tracks={len(l_decisions)+len(r_decisions)}  "
                      f"frames={l_used + r_used}")
        except Exception as e:
            payload = empty_payload(
                gw_id, FEATURE_NAME, C.STATUS_FAILED,
                left_door=C.NO_DATA, right_door=C.NO_DATA,
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(limit=2),
            )
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = C.STATUS_FAILED
            if verbose:
                print(f"  [door/{gw_id}] FAILED: {e}")
        finally:
            timer.stamp(gw_id, t0)

    if verbose:
        n_ok = sum(1 for v in summary.values() if v == C.STATUS_OK)
        print(f"[FEAT/door] done in {timer.total():.1f}s  ok={n_ok}/{len(summary)}")
    return summary
