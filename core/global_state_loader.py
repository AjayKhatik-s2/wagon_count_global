"""Adapter exposing the CURRENT global train state under the name old_code expects.

old_code does `from core.global_state_loader import GlobalTrainState` and then only
ever reads:

    state.wagons              -> iterable of wagons, each with .global_id
    wagon.global_id           -> 'GW_n'
    wagon.classification      -> ENGINE / BRAKE_VAN / WAGON
    state.master_camera, .master_fps, .master_total_frames
    state.fallback_used, .fallback_reason, .corrections_applied
    state.per_camera_local_counts

The current pipeline's `GlobalTrainState` already provides every one of those with
the same names, so this module re-exports it directly instead of wrapping it. That
is deliberate: a wrapper would be a second wagon representation that could drift
from the real one, and the whole point is that there is exactly ONE global wagon
structure.

READ-ONLY BY CONTRACT. Nothing in old_code assigns to `state` -- it only iterates
wagons and writes its own per-wagon JSON. `snapshot_roster()` and
`assert_roster_unchanged()` below make that contract enforceable rather than
assumed: the roster is hashed before inspection and verified afterwards, so if any
feature ever mutates a wagon the run fails loudly instead of shipping a changed
count.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List

# Re-export the real thing. There is no second wagon structure.
from global_train_state import (            # noqa: F401  (re-export)
    ALL_CAMERAS, GlobalTrainState, GlobalWagon, LocalCameraTracks, MASTER_CAMERA,
    SegmentClass,
)

__all__ = [
    "GlobalTrainState", "GlobalWagon", "LocalCameraTracks", "SegmentClass",
    "ALL_CAMERAS", "MASTER_CAMERA",
    "snapshot_roster", "roster_hash", "assert_roster_unchanged",
]


def snapshot_roster(state: GlobalTrainState) -> List[Dict[str, Any]]:
    """The identity-defining fields of every wagon, in order.

    Deliberately narrow: only what inspection must never change. Confidence
    scores and support lists are excluded because a downstream feature may
    legitimately annotate, while ids, boundaries, ordering, classification and the
    count may not.
    """
    return [
        {
            "global_id": w.global_id,
            "wagon_index": w.wagon_index,
            "start_frame_master": w.start_frame_master,
            "end_frame_master": w.end_frame_master,
            "start_time": round(float(w.start_time), 6),
            "end_time": round(float(w.end_time), 6),
            "classification": w.classification,
        }
        for w in state.wagons
    ]


def roster_hash(state: GlobalTrainState) -> str:
    """Deterministic hash of the wagon roster plus the counting invariant.

    Includes `total_wagons` and the MASTER == GLOBAL bookkeeping so that a change
    to the count or the invariant is caught even if individual wagons look intact.
    """
    payload = {
        "wagons": snapshot_roster(state),
        "total_wagons": state.total_wagons,
        "master_camera": state.master_camera,
        "global_gap_count": len(state.global_gaps or []),
        "right_up_final_gap_count": (state.invariant_checks or {}).get(
            "right_up_final_gap_count"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class RosterMutatedError(RuntimeError):
    """Raised when inspection changed the finalized global wagon structure."""


def assert_roster_unchanged(state: GlobalTrainState, before: str) -> None:
    """Fail loudly if inspection altered the wagon roster.

    This is the guard that makes "inspection cannot change the count" a checked
    property rather than a claim. It runs after every feature has completed.
    """
    after = roster_hash(state)
    if after != before:
        raise RosterMutatedError(
            "inspection modified the finalized global wagon structure "
            f"(roster hash {before[:12]} -> {after[:12]}). Door/load/damage/OCR "
            "must only annotate wagons that already exist: they may never create, "
            "delete, split, merge or renumber a wagon, move a boundary, or disturb "
            "the MASTER == GLOBAL invariant.")
