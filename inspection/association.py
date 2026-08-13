"""Attach a tracked finding to the correct EXISTING global wagon id.

    detection track
        -> camera + local time
        -> camera offset (already resolved by the counting pipeline)
        -> global timeline
        -> global wagon interval
        -> GW_ID

WHAT MAKES THIS SAFE
--------------------
Association is a LOOKUP into a finished roster. The wagon list is read-only, and
the only outputs are (global_id, status) pairs. There is no code path here that
appends a wagon, removes one, renumbers one, or edits a boundary -- so no
detection, however confident, can alter the count.

A finding that lands nowhere is recorded as UNRESOLVED and keeps its camera-local
identity. It is never given a wagon of its own, and never rounded to the nearest
one.

WHY IT REUSES THE EXISTING SYNCHRONIZATION
------------------------------------------
The counting pipeline already resolves each camera's clock offset and marks it
REFERENCE, RESOLVED or unresolved. Inventing a second mechanism here would let
door evidence and gap evidence disagree about when the same wagon passed. So the
offsets are taken as given, and a camera whose offset was never resolved yields
UNRESOLVED findings rather than a guessed timestamp -- the same policy the
evidence report already applies.

THE SAME PHYSICAL WAGON IN FOUR CAMERAS
---------------------------------------
Because every detection is converted into MASTER time before lookup, a door seen
by RIGHT_UP and the same door seen by LEFT_UP resolve to the SAME GW id. There is
deliberately no per-camera wagon numbering anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .state import (
    ASSOCIATION_AMBIGUOUS, ASSOCIATION_RESOLVED, ASSOCIATION_UNRESOLVED,
)
from .tracking import InspectionTrack

# Offset statuses the counting pipeline uses that mean "trustworthy".
TRUSTED_OFFSET_STATUSES = ("REFERENCE", "RESOLVED")


@dataclass
class WagonInterval:
    """A global wagon's time window in MASTER seconds. Read-only."""
    global_id: str
    start_time: float
    end_time: float
    classification: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


def wagon_intervals_from_state(state: Any) -> List[WagonInterval]:
    """Read the finished wagon roster. Never modifies it.

    Accepts either the live `GlobalTrainState` or its serialized dict, so the
    association layer works against a completed run's JSON as well as an
    in-process state.
    """
    wagons = []
    raw = getattr(state, "wagons", None)
    if raw is None and isinstance(state, dict):
        raw = state.get("wagons") or []
    for w in raw or []:
        if isinstance(w, dict):
            gid = w.get("global_id")
            start, end = w.get("start_time"), w.get("end_time")
            cls = w.get("classification", "")
        else:
            gid = getattr(w, "global_id", None)
            start, end = getattr(w, "start_time", None), getattr(w, "end_time", None)
            cls = getattr(w, "classification", "")
        if gid is None or start is None or end is None:
            continue
        wagons.append(WagonInterval(str(gid), float(start), float(end), str(cls)))
    wagons.sort(key=lambda w: (w.start_time, w.global_id))
    return wagons


def trusted_offsets(state: Any) -> Dict[str, float]:
    """Per-camera offset, ONLY where synchronization was decisive.

    A camera that is absent from the result is unresolved: its findings must not
    be placed on the global timeline at all. Returning 0.0 for it would silently
    assert perfect synchronization, which is exactly the fabrication this avoids.
    """
    raw = getattr(state, "camera_offsets", None)
    if raw is None and isinstance(state, dict):
        raw = state.get("camera_offsets") or {}
    out: Dict[str, float] = {}
    for cam, off in (raw or {}).items():
        if isinstance(off, dict) and off.get("status") in TRUSTED_OFFSET_STATUSES:
            out[str(cam)] = float(off.get("delta", 0.0) or 0.0)
    return out


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def associate_track(
    track: InspectionTrack,
    wagons: Sequence[WagonInterval],
    offsets: Dict[str, float],
    min_overlap_fraction: float = 0.0,
) -> Dict[str, Any]:
    """Resolve one track to a global wagon id.

    Chooses the wagon whose window overlaps the track's global time span the
    most. Overlap rather than midpoint, because a finding can legitimately span a
    wagon boundary (the object is visible while the boundary passes) and the
    wagon it mostly belongs to is the right answer.

    Reports AMBIGUOUS when two wagons overlap almost equally, so a coin-flip is
    never silently presented as a fact.
    """
    result: Dict[str, Any] = {
        "global_id": None,
        "association_status": ASSOCIATION_UNRESOLVED,
        "association_method": "",
        "association_detail": "",
        "global_time_start": None,
        "global_time_end": None,
        "camera_offset": None,
        "overlap_fraction": None,
        "candidate_global_ids": [],
    }

    if track.camera_id not in offsets:
        result["association_detail"] = (
            f"camera {track.camera_id} has no resolved clock offset, so this "
            f"finding cannot be placed on the global timeline; it stays "
            f"camera-local rather than being attributed to a guessed wagon")
        result["association_method"] = "none:unresolved_camera"
        return result

    delta = offsets[track.camera_id]
    g_start = track.start_time + delta
    g_end = track.end_time + delta
    result.update({"global_time_start": g_start, "global_time_end": g_end,
                   "camera_offset": delta})

    if not wagons:
        result["association_detail"] = "no global wagons exist to associate with"
        result["association_method"] = "none:empty_roster"
        return result

    span = max(1e-9, g_end - g_start)
    scored: List[Tuple[float, WagonInterval]] = []
    for w in wagons:
        ov = _overlap(g_start, g_end, w.start_time, w.end_time)
        if ov > 0:
            scored.append((ov / span, w))

    if not scored:
        # Outside every wagon window: before the first wagon, after the last, or
        # inside a non-wagon region (engine / brake van carry no GW id).
        result["association_method"] = "overlap:none"
        result["association_detail"] = (
            f"global time {g_start:.2f}-{g_end:.2f}s falls outside every global "
            f"wagon window (before the first wagon, after the last, or in a "
            f"non-wagon region such as an engine or brake van)")
        return result

    # Deterministic: best overlap, ties broken by wagon order.
    scored.sort(key=lambda p: (-p[0], p[1].start_time, p[1].global_id))
    best_frac, best = scored[0]
    result["candidate_global_ids"] = [w.global_id for _, w in scored]
    result["overlap_fraction"] = best_frac
    result["association_method"] = "overlap:max"

    if best_frac < min_overlap_fraction:
        result["association_detail"] = (
            f"best overlap {best_frac:.2f} with {best.global_id} is below the "
            f"{min_overlap_fraction:.2f} floor")
        return result

    if len(scored) > 1:
        second_frac = scored[1][0]
        # "Almost equal" = within a tenth of each other. Below that the wagon with
        # the larger share is a real answer, not a tie.
        if abs(best_frac - second_frac) < 0.10:
            result.update({
                "global_id": best.global_id,
                "association_status": ASSOCIATION_AMBIGUOUS,
                "association_detail": (
                    f"overlaps {best.global_id} ({best_frac:.2f}) and "
                    f"{scored[1][1].global_id} ({second_frac:.2f}) almost "
                    f"equally; attributed to the larger share but flagged"),
            })
            return result

    result.update({
        "global_id": best.global_id,
        "association_status": ASSOCIATION_RESOLVED,
        "association_detail": (
            f"{best_frac:.2f} of the finding's global span lies inside "
            f"{best.global_id} ({best.start_time:.2f}-{best.end_time:.2f}s "
            f"master), via {track.camera_id} offset {delta:+.3f}s"),
    })
    return result


def associate_tracks_to_wagons(
    tracks: Sequence[InspectionTrack],
    state: Any,
    min_overlap_fraction: float = 0.0,
) -> List[Dict[str, Any]]:
    """Resolve many tracks. Order-stable and free of side effects on `state`."""
    wagons = wagon_intervals_from_state(state)
    offsets = trusted_offsets(state)
    return [associate_track(tr, wagons, offsets, min_overlap_fraction)
            for tr in sorted(tracks,
                             key=lambda t: (t.camera_id, t.start_frame, t.track_id))]


def camera_visibility(
    wagons: Sequence[WagonInterval],
    camera_id: str,
    offsets: Dict[str, float],
    camera_duration: Optional[float],
) -> Dict[str, str]:
    """Which wagons a camera could POSSIBLY have seen.

    This is what keeps "no detection" from being reported as "no damage". A wagon
    whose window lies outside this camera's footage was never observed, and the
    report must say so rather than implying a clean inspection.
    """
    from .state import CAMERA_NOT_VISIBLE, CAMERA_NO_DETECTION, CAMERA_UNRESOLVED

    out: Dict[str, str] = {}
    if camera_id not in offsets:
        return {w.global_id: CAMERA_UNRESOLVED for w in wagons}
    delta = offsets[camera_id]
    for w in wagons:
        # Master window -> this camera's own clock.
        local_start = w.start_time - delta
        local_end = w.end_time - delta
        if local_end < 0 or (camera_duration is not None
                             and local_start > camera_duration):
            out[w.global_id] = CAMERA_NOT_VISIBLE
        else:
            out[w.global_id] = CAMERA_NO_DETECTION
    return out
