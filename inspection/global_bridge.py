"""Bridge: finalized GLOBAL wagon roster -> the legacy inspection DataFrames.

THIS MODULE IS THE WHOLE POINT OF THE PORT
------------------------------------------
The legacy Train-Inspection-Engine feeds every one of its inspection modules
(``DamageDetector``, ``ProblemFrameExtractor``, ``ArtifactPublisher``,
``build_inspection_json``, ``PdfReportBuilder``) from ONE object: a
``segment_summary_df`` produced by its own ``WagonSegmenter``. That segmenter is
the legacy CAMERA-WISE COUNTER -- it bands gap detections per camera, pairs
consecutive bands into segments, numbers them 1..N per camera, and classifies
each one. It is exactly the component we must not reuse, because it would give
LEFT_UP its own wagon count and RIGHT_UP a different one.

So this module builds the same DataFrame from the finalized global roster
instead. Every legacy consumer downstream then works unchanged, and every number
it prints traces back to GW_1..GW_N rather than to a camera-local recount.

    state.wagons (GW_1..GW_N, FINAL)        <- protected, never read-modified
        x camera offset  (RESOLVED only)    <- finished synchronization
        = per-camera frame window
        -> segment_summary_df rows          <- what the legacy modules consume
                segment_id   = global wagon_index     (NOT a camera-local counter)
                directory    = wagon_cache/GW_n/CAM/  (association by construction)
                segment_type = roster class + load    (legacy vocabulary)

WHY ``segment_id`` IS THE GLOBAL INDEX
--------------------------------------
The legacy schema keys everything on ``segment_id`` and derives the dashboard's
``wagon_count`` from a *separate* running counter that increments once per wagon
segment (``json_builder._build_segment_type_map_and_wagon_counts``). Two
independent numbers, both camera-local. Here both are pinned to the same global
value: ``segment_id == wagon_index`` and ``wagon_count == wagon_index``, for
every camera. That is what makes four cameras agree, and it is asserted by
``assert_wagon_count_map_is_global`` rather than assumed.

WHAT THIS MODULE MAY NOT DO
---------------------------
It only READS the roster. It never creates, drops, renumbers, reorders, splits,
merges or re-times a wagon, and it never touches camera offsets. A camera that
cannot be associated safely produces NO rows for that wagon -- it is recorded as
NOT_VISIBLE / UNRESOLVED, never guessed onto a neighbouring wagon.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C

__all__ = [
    "CameraProfile", "CAMERA_PROFILES", "camera_profile",
    "SEGMENT_SUMMARY_COLUMNS",
    "LEGACY_SEG_WAGON", "LEGACY_SEG_WAGON_LOADED", "LEGACY_SEG_ENGINE",
    "LEGACY_SEG_BRAKEVAN",
    "roster_segment_type", "build_segment_summary", "build_wagon_count_map",
    "build_segment_type_map", "assert_wagon_count_map_is_global",
    "GlobalAssociationError",
]


# ---------------------------------------------------------------------------
# legacy segment-type vocabulary
# ---------------------------------------------------------------------------

LEGACY_SEG_WAGON = "wagon"
LEGACY_SEG_WAGON_LOADED = "wagon_loaded"
LEGACY_SEG_ENGINE = "engine"
LEGACY_SEG_BRAKEVAN = "brakevan"
"""The internal segment_type strings the legacy modules branch on.

RECOVERED, not invented: ``damage.NON_WAGON_SEGMENT_TYPES`` is
``{"engine", "brakevan"}``, ``damage.LOADED_SEGMENT_TYPE`` is ``"wagon_loaded"``,
and ``artifacts.TOP_DISPLAY`` / ``SIDE_DISPLAY`` key on exactly these four.
Emitting anything else would silently fall through those maps as a wagon.
"""

SEGMENT_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "segment_id", "segment_type", "start_frame", "end_frame", "directory",
    "type_dominance", "global_wagon_id", "camera_id", "truncated",
)
"""Columns the legacy modules actually read, plus two additive ones.

REQUIRED BY LEGACY  segment_id / segment_type / start_frame / end_frame /
                    directory  (damage.py, artifacts.py, pdf_builder.py)
OPTIONAL, LEGACY    type_dominance  (damage._dominance_of; guards the
                    engine/brakevan skip so a thin-margin label is still scanned)
ADDITIVE, OURS      global_wagon_id / camera_id / truncated -- carried so every
                    downstream row can be traced back to a GW id and to how
                    completely that window was cached. Legacy code ignores extra
                    columns, so adding them cannot change its behaviour.
"""


class GlobalAssociationError(RuntimeError):
    """Raised when a bridged row would not map onto an existing global wagon."""


# ---------------------------------------------------------------------------
# camera identity: current pipeline <-> legacy dashboard
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraProfile:
    """How one current camera presents itself to the legacy JSON/PDF layer.

    ``folder`` and ``pdf_position`` are RECOVERED verbatim from the legacy
    ``configs/cameras/*.yaml`` so the dashboard keeps receiving the camera
    identifiers it already indexes on. ``flavour`` selects which legacy code
    path runs and is the single switch between the two JSON schemas.
    """

    camera_id: str            # current pipeline id, e.g. RIGHT_UP
    legacy_name: str          # legacy config name, e.g. right_up
    folder: str               # legacy camera_id -> JSON "camera_id" after prefix strip
    flavour: str              # "side" | "top"
    pdf_position: str
    loaded_direction: str
    """Travel direction that the legacy side flavour reads as a LOADED rake.

    Side cameras have no visual load class, so ``json_builder``'s side flavour
    derives ``rake_status`` from direction alone. Preserved as legacy behaviour;
    it annotates only and never feeds counting.
    """

    @property
    def is_side(self) -> bool:
        return self.flavour == "side"

    @property
    def is_top(self) -> bool:
        return self.flavour == "top"


CAMERA_PROFILES: Dict[str, CameraProfile] = {
    C.CAMERA_RIGHT_UP: CameraProfile(
        camera_id=C.CAMERA_RIGHT_UP, legacy_name="right_up",
        folder="camera_CCTV_HZBN_DHN_2_RIGHT_UP", flavour="side",
        pdf_position="RIGHT UP", loaded_direction="left-to-right"),
    C.CAMERA_LEFT_UP: CameraProfile(
        camera_id=C.CAMERA_LEFT_UP, legacy_name="left_up",
        folder="camera_CCTV_HZBN_DHN_1_LEFT_UP", flavour="side",
        pdf_position="LEFT UP", loaded_direction="left-to-right"),
    C.CAMERA_RIGHT_UP_TOP: CameraProfile(
        camera_id=C.CAMERA_RIGHT_UP_TOP, legacy_name="right_top",
        folder="camera_CCTV_HZBN_DHN_5_RIGHT_TOP", flavour="top",
        pdf_position="RIGHT TOP", loaded_direction="right-to-left"),
    C.CAMERA_LEFT_UP_TOP: CameraProfile(
        camera_id=C.CAMERA_LEFT_UP_TOP, legacy_name="left_top",
        folder="camera_CCTV_HZBN_DHN_6_LEFT_TOP", flavour="top",
        pdf_position="LEFT TOP", loaded_direction="left-to-right"),
}
"""RECOVERED from the four legacy camera YAMLs.

The dashboard keys artifacts on the camera folder, so these strings are part of
the compatibility contract -- changing one would orphan that camera's history.
"""


def camera_profile(camera_id: str) -> CameraProfile:
    try:
        return CAMERA_PROFILES[camera_id]
    except KeyError:
        raise GlobalAssociationError(
            f"unknown camera {camera_id!r}; expected one of "
            f"{sorted(CAMERA_PROFILES)}") from None


# ---------------------------------------------------------------------------
# roster -> legacy segment_type
# ---------------------------------------------------------------------------

def roster_segment_type(classification: str,
                        load_status: Optional[str] = None) -> str:
    """Map a GLOBAL wagon's classification (+ load) to the legacy vocabulary.

    The roster is the authority for what a vehicle IS. The load model may only
    refine a wagon into loaded/empty -- it can never turn a wagon into an engine
    or vice versa, which is why classification is checked first and load second.

    ``load_status`` is the fused verdict from the load feature:
    ``LOADED`` -> ``wagon_loaded``; ``EMPTY`` -> ``wagon``; anything else
    (``NO_DATA``, ``None``, an abstaining ``Unlabeled`` prediction) -> ``wagon``.
    That last fallback matches the legacy default and is deliberately NOT a
    claim that the wagon is empty: the top JSON reports it under
    ``wagons_empty`` exactly as the legacy pipeline did for an unclassified
    segment, while ``load_status``/``load_condition`` in the per-wagon block
    still carry the real state.
    """
    cls = (classification or "").strip().upper()
    if cls == C.CLASS_ENGINE:
        return LEGACY_SEG_ENGINE
    if cls == C.CLASS_BRAKE_VAN:
        return LEGACY_SEG_BRAKEVAN
    if (load_status or "").strip().upper() == C.LOAD_LOADED:
        return LEGACY_SEG_WAGON_LOADED
    return LEGACY_SEG_WAGON


# ---------------------------------------------------------------------------
# the bridge
# ---------------------------------------------------------------------------

def _wagon_fields(w: Any) -> Tuple[Optional[str], Optional[int], str]:
    """(global_id, wagon_index, classification) from a wagon object or dict."""
    if isinstance(w, dict):
        return (w.get("global_id"), w.get("wagon_index"),
                str(w.get("classification") or C.CLASS_WAGON))
    return (getattr(w, "global_id", None), getattr(w, "wagon_index", None),
            str(getattr(w, "classification", C.CLASS_WAGON) or C.CLASS_WAGON))


def _iter_roster(state: Any) -> List[Any]:
    wagons = getattr(state, "wagons", None)
    if wagons is None and isinstance(state, dict):
        wagons = state.get("wagons") or []
    return list(wagons or [])


def build_segment_summary(
    state: Any,
    camera_id: str,
    windows: Sequence[Any],
    cache_root: str,
    load_status_by_wagon: Optional[Dict[str, str]] = None,
    require_frames: bool = True,
    include_unwindowed: bool = False,
):
    """Build one camera's ``segment_summary_df`` from the finalized roster.

    Parameters
    ----------
    windows
        The ``WagonWindow`` objects this camera got from
        ``inspection.wagon_cache.plan_cache`` -- i.e. the wagons that are
        genuinely inside this camera's footage at its RESOLVED offset. A wagon
        with no window is simply absent from this camera's DataFrame; it is
        still reported globally, as NOT_VISIBLE for this camera.
    load_status_by_wagon
        ``{GW_id: LOADED|EMPTY|NO_DATA}``. Supplied for TOP cameras so the
        legacy loaded-floor suppression and the ``wagons_loaded`` /
        ``wagons_empty`` counts behave exactly as they did. Omitted for side
        cameras, which the legacy schema does not load-classify.
    require_frames
        Drop a window whose cache directory holds no frames. On (default)
        because the legacy modules silently produce an all-False damage row for
        an empty directory, which would read as "inspected, clean" rather than
        "never seen".
    include_unwindowed
        Also emit a row for every roster wagon this camera has NO window for.

    THE TWO CALLS, AND WHY THERE ARE TWO
    ------------------------------------
    The orchestrator builds this DataFrame twice per camera, and the pair is
    what resolves a real conflict in the requirements:

      * ``scan_df`` -- ``require_frames=True, include_unwindowed=False``. Only
        wagons this camera genuinely observed. This is what the model scans, so
        the detector is never asked to certify a wagon it could not see.
      * ``full_df`` -- ``require_frames=False, include_unwindowed=True``. Every
        wagon in the roster, so ``total_wagons`` and ``wagon_segments`` are the
        GLOBAL count in all four JSON files -- never 55 here and 57 there.

    A wagon in ``full_df`` but not ``scan_df`` has no damage row, so the legacy
    builder falls back to its own defaults for it (``door_status: "closed"``,
    all booleans False) exactly as it always did for a segment with no damage
    row. That legacy conflation of "unseen" with "clean" is preserved in the
    legacy fields for schema compatibility, and disambiguated by the ADDITIVE
    ``inspection_status`` field the orchestrator attaches.

    Returns a DataFrame with :data:`SEGMENT_SUMMARY_COLUMNS`, ordered by global
    wagon index. Row order is the train's physical order, for every camera.
    """
    import pandas as pd

    profile = camera_profile(camera_id)
    load_status_by_wagon = load_status_by_wagon or {}

    by_gid: Dict[str, Any] = {}
    for w in windows:
        gid = getattr(w, "global_id", None)
        if gid is not None:
            by_gid[str(gid)] = w

    rows: List[Dict[str, Any]] = []
    for wagon in _iter_roster(state):
        gid, index, classification = _wagon_fields(wagon)
        if gid is None or index is None:
            continue
        window = by_gid.get(str(gid))
        if window is None and not include_unwindowed:
            continue                      # not visible to this camera -- never guessed

        directory = os.path.join(
            cache_root, str(gid), C.CAMERA_FOLDER.get(camera_id, camera_id))
        if window is not None and require_frames and not _has_frames(directory):
            continue

        seg_type = roster_segment_type(
            classification,
            load_status_by_wagon.get(str(gid)) if profile.is_top else None)

        # A wagon with no window has no frame range in THIS camera's clock. Zero
        # is used rather than a projected guess: the row exists only so the
        # global count is complete, and the legacy readers all guard on
        # os.path.exists before touching a frame, so an empty range yields no
        # evidence instead of evidence from the wrong place.
        rows.append({
            "segment_id": int(index),          # == global wagon index. See module docstring.
            "segment_type": seg_type,
            "start_frame": int(getattr(window, "start_frame", 0)) if window else 0,
            "end_frame": int(getattr(window, "end_frame", 0)) if window else 0,
            "directory": directory,
            # The roster's classification is a fused, cross-camera decision that
            # has already survived validation, so it is trusted outright. 1.0
            # keeps damage.py's `_is_trusted_non_wagon` guard satisfied instead
            # of re-deriving a vote share the roster no longer carries.
            "type_dominance": 1.0,
            "global_wagon_id": str(gid),
            "camera_id": camera_id,
            "truncated": bool(getattr(window, "truncated", False)),
        })

    rows.sort(key=lambda r: r["segment_id"])
    if not rows:
        return pd.DataFrame(columns=list(SEGMENT_SUMMARY_COLUMNS))
    return pd.DataFrame(rows, columns=list(SEGMENT_SUMMARY_COLUMNS))


def _has_frames(directory: str) -> bool:
    if not os.path.isdir(directory):
        return False
    try:
        for name in os.listdir(directory):
            if name.startswith("frame_") and name.endswith(".jpg"):
                return True
    except OSError:
        return False
    return False


# ---------------------------------------------------------------------------
# wagon_count / segment_type_map -- the dashboard-facing identity
# ---------------------------------------------------------------------------

def build_wagon_count_map(segment_summary_df) -> Dict[int, Optional[int]]:
    """``{segment_id: wagon_count}`` where wagon_count IS the global index.

    This function replaces ``json_builder._build_segment_type_map_and_wagon_counts``'s
    running counter. The legacy counter restarted at 1 for every camera and
    skipped non-wagons, so a camera that saw one fewer wagon renumbered every
    wagon after it. Here the value is read from the roster instead of counted,
    so GW_17 is ``wagon_count == 17`` in all four JSON files no matter what any
    camera did or did not see.

    Non-wagon segments map to ``None``, which is the legacy convention and is
    what makes engines and brakevans fall out of the wagon blocks.
    """
    out: Dict[int, Optional[int]] = {}
    if segment_summary_df is None or len(segment_summary_df) == 0:
        return out
    for _, seg in segment_summary_df.iterrows():
        seg_id = int(seg["segment_id"])
        seg_type = seg.get("segment_type", LEGACY_SEG_WAGON)
        is_wagon = seg_type in (LEGACY_SEG_WAGON, LEGACY_SEG_WAGON_LOADED)
        out[seg_id] = seg_id if is_wagon else None
    return out


def build_segment_type_map(segment_summary_df, flavour: str) -> Dict[str, Dict[str, Any]]:
    """Legacy ``segment_type_map``, keyed by ``str(segment_id)``.

    Shape is preserved exactly: ``{type, number}`` for side, plus ``wagon_count``
    for top. ``number`` is the per-display-type ordinal (1st engine, 2nd
    brakevan, ...) exactly as the legacy builder computed it -- that one IS a
    per-type counter in the original and stays one, because it numbers engines
    and brakevans, which the global roster does not assign GW ids to.
    ``wagon_count`` is the global index.
    """
    from .legacy.artifacts import display_segment_type

    type_counters: Dict[str, int] = {}
    seg_map: Dict[str, Dict[str, Any]] = {}
    if segment_summary_df is None or len(segment_summary_df) == 0:
        return seg_map

    wagon_counts = build_wagon_count_map(segment_summary_df)
    for _, seg in segment_summary_df.iterrows():
        seg_id = int(seg["segment_id"])
        internal = seg.get("segment_type", LEGACY_SEG_WAGON)
        display = display_segment_type(internal, flavour)
        type_counters[display] = type_counters.get(display, 0) + 1
        entry: Dict[str, Any] = {"type": display, "number": type_counters[display]}
        if flavour == "top":
            entry["wagon_count"] = wagon_counts.get(seg_id)
        seg_map[str(seg_id)] = entry
    return seg_map


def assert_wagon_count_map_is_global(
    state: Any,
    wagon_count_map: Dict[int, Optional[int]],
    camera_id: str = "",
) -> None:
    """Fail loudly if a camera invented, dropped or renumbered a wagon.

    This is the checked form of the acceptance criterion. It proves three
    things at once, per camera:

      * every non-null ``wagon_count`` is a wagon index that EXISTS in the
        finalized roster -- so inspection cannot invent a wagon;
      * ``wagon_count == segment_id`` -- so no camera-local renumbering slipped
        back in;
      * no two segments share a ``wagon_count`` -- so a wagon cannot appear
        twice under different ids.

    It deliberately does NOT require every roster wagon to be present: a wagon
    outside this camera's footage is legitimately absent here, and is reported
    globally as NOT_VISIBLE. Completeness of the *global* roster in the output
    is asserted separately, over the union of cameras.
    """
    valid = {int(idx) for _, idx, _ in map(_wagon_fields, _iter_roster(state))
             if idx is not None}
    seen: Dict[int, int] = {}
    where = f" ({camera_id})" if camera_id else ""
    for seg_id, count in sorted(wagon_count_map.items()):
        if count is None:
            continue
        if int(count) not in valid:
            raise GlobalAssociationError(
                f"inspection{where} produced wagon_count={count} for segment "
                f"{seg_id}, which is not a wagon in the finalized global roster "
                f"({len(valid)} wagons). Inspection may only annotate wagons "
                f"that already exist.")
        if int(count) != int(seg_id):
            raise GlobalAssociationError(
                f"inspection{where} renumbered segment {seg_id} to "
                f"wagon_count={count}. The global wagon index is the only "
                f"wagon identity; a camera-local counter must never be used.")
        if int(count) in seen:
            raise GlobalAssociationError(
                f"inspection{where} emitted wagon_count={count} twice "
                f"(segments {seen[int(count)]} and {seg_id}). A wagon must "
                f"appear exactly once.")
        seen[int(count)] = seg_id
