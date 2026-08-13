"""Inspection state: what was found, on which wagon, with what provenance.

DESIGN CONSTRAINT: this mirrors the global wagon roster, it does not replace it.
A `WagonInspection` is keyed by the `global_id` that fusion already minted and
holds no geometry of its own -- so it cannot drift from, contradict, or renumber
the counting result.

THE DISTINCTION THAT MATTERS MOST
---------------------------------
"No detection" and "no damage" are NOT the same statement, and conflating them
would make the report actively misleading -- a wagon never seen by a top camera
would be reported as undamaged. Every (wagon, camera) pair therefore carries an
explicit status:

    CONFIRMED       a tracked, confirmed finding exists here
    NO_DETECTION    the camera saw this wagon and found nothing
    NOT_VISIBLE     this wagon's time window falls outside the camera's footage
    UNRESOLVED      the camera's clock offset was never resolved, so nothing
                    here can be attributed to a wagon at all
    NOT_APPLICABLE  this role does not run on this camera (top damage on a
                    side camera, for instance)

Only NO_DETECTION licenses the words "nothing found".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# ---- association outcome ---------------------------------------------------
ASSOCIATION_RESOLVED = "RESOLVED"
ASSOCIATION_UNRESOLVED = "UNRESOLVED"
ASSOCIATION_AMBIGUOUS = "AMBIGUOUS"

# ---- per (wagon, camera) evidence status -----------------------------------
CAMERA_CONFIRMED = "CONFIRMED"
CAMERA_NO_DETECTION = "NO_DETECTION"
CAMERA_NOT_VISIBLE = "NOT_VISIBLE"
CAMERA_UNRESOLVED = "UNRESOLVED"
CAMERA_NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class InspectionConfig:
    """Thresholds for detection, tracking and confirmation.

    Camera-independent by construction: seconds, frame-width fractions,
    dimensionless ratios and counts only, resolved against each camera's own
    width and fps at runtime. Nothing here encodes one train's speed, geometry,
    wagon count or class frequencies.

    PROVENANCE. Measured by running both shipped models over real frames of this
    project's own footage (door model on a side camera, RT-DETR damage model on a
    top camera):

      * At conf >= 0.01 the models emit a median of 27 and 19.5 detections per
        frame. That bulk is noise: its median confidence is 0.013-0.016 and its
        p90 is 0.028-0.079.
      * Genuine detections are sparse and strong -- 0.55 to 0.945 (door) and up
        to 0.876 (damage).
      * On CONSECUTIVE frames a genuine door persisted for 28 frames with its
        centre advancing ~28 px/frame, i.e. at the train's own speed, while false
        positives formed 1-5 frame runs at 0.21-0.55, several pinned near the
        frame edge.

    Every default below is placed in the gap those measurements opened up.
    """

    enabled: bool = True

    # ---- detection ---------------------------------------------------------
    min_detection_confidence: float = 0.25
    """Floor for admitting a raw detection into tracking.

    Sits 3-9x above the measured noise p90 (0.028-0.079) and far below the
    weakest genuine detection (0.55). At 0.01 the models emit ~27 and ~19.5
    detections per frame; at this floor that collapses to under one."""

    frame_stride: int = 4
    """Inspect every Nth frame.

    Chosen from a measurement on this project's own CPU box: inference costs
    ~934 ms per frame per model, so covering four cameras end to end takes about
    221 minutes at stride 1 and 111 at stride 2 -- untenable per train. Stride 4
    brings that to ~55 minutes.

    4 is the FASTEST SAFE value, and that boundary was measured rather than
    assumed. Detecting once at stride 1 on two real windows and then subsampling
    -- so only the stride varies -- gave identical verdicts at 1, 2 and 4: the
    same classes, the same peak confidences (0.95 / 0.90 / 0.89 / 0.82) and the
    same wagon attributions. At stride 8 every finding was LOST, and the damage
    window fragmented from 4 tracks into 11 because consecutive sightings no
    longer associate.

    So verdicts are stride-invariant up to 4 and break at 8. Persistence
    thresholds are in SECONDS and the association allowance is per frame
    interval, which is what makes that invariance hold; raising this dial past 4
    is not a cost/quality trade but a loss of findings."""

    max_center_jump_frac: float = 0.06
    """Positional association allowance, as a fraction of frame width PER FRAME
    INTERVAL.

    At 960 px this is ~58 px per interval against a measured train advance of
    ~28 px/frame -- roughly 2x headroom. Being per-interval is what lets
    `frame_stride` be raised for speed without fragmenting tracks."""

    max_detections_per_frame: int = 20
    """Hard ceiling per frame, so a pathological frame cannot balloon memory."""

    # ---- tracking ----------------------------------------------------------
    iou_match_threshold: float = 0.20
    """Minimum box overlap to continue an existing track across frames."""

    max_track_miss_seconds: float = 0.30
    """How long a track may go unobserved before it is closed."""

    min_track_seconds: float = 0.40
    """Persistence required to confirm a finding.

    At 15 fps this is 6 frames. Every observed false-positive run was 1-5 frames;
    the genuine door ran 28. So this floor sits above all measured noise and 4.7x
    below the measured signal."""

    min_track_observations: int = 3
    """A confirmed finding needs at least three sightings regardless of fps, so a
    single frame -- or a single frame plus one flicker -- can never be a finding."""

    # ---- confirmation ------------------------------------------------------
    min_peak_confidence: float = 0.60
    """A confirmed track must peak above this.

    Genuine findings peaked at 0.945 and 0.876; the strongest false-positive run
    peaked at 0.55. 0.60 separates them with margin on both sides."""

    min_mean_confidence: float = 0.35
    """Guards against a track that is noise throughout except for one strong
    frame: the run as a whole must be more than incidental."""

    # ---- motion (the static-artefact test that worked for gaps) ------------
    require_motion: bool = True
    min_motion_frac: float = 0.02
    """A confirmed finding must travel at least this fraction of frame width.

    The measured false positives sat pinned near cx=20 while the genuine door
    swept 53->847 px (0.83 of frame width). This is the same discriminator that
    caught the gap false positives, which passed confidence, hit-count and
    coverage filters and were only ever separated by motion."""

    # ---- evidence ----------------------------------------------------------
    max_evidence_frames_per_event: int = 3
    """Highest-confidence sighting plus temporal context. Deliberately small:
    thousands of near-duplicate crops would bloat the report for no gain."""


# ---------------------------------------------------------------------------
# findings
# ---------------------------------------------------------------------------

@dataclass
class InspectionEvent:
    """One confirmed finding, with full provenance from pixel to GW id.

    Every field needed to defend the finding is retained, because a report that
    says "GW_17 has damage" without being able to show the camera, frame, box,
    class, confidence and how it was attributed is not auditable.
    """
    event_id: str
    role: str                          # 'door' | 'top_damage'
    model_path: str
    model_class_id: int
    model_class_name: str              # the model's OWN name, never invented
    camera_id: str
    track_id: int

    start_frame: int
    end_frame: int
    start_time_local: float
    end_time_local: float
    n_observations: int

    peak_confidence: float
    mean_confidence: float

    # association into the global train
    global_id: Optional[str] = None
    association_status: str = ASSOCIATION_UNRESOLVED
    association_method: str = ""
    association_detail: str = ""
    global_time_start: Optional[float] = None
    global_time_end: Optional[float] = None
    camera_offset: Optional[float] = None
    overlap_fraction: Optional[float] = None
    candidate_global_ids: List[str] = field(default_factory=list)

    # geometry + evidence
    peak_frame: int = 0
    peak_bbox: Optional[List[float]] = None
    displacement_px: float = 0.0
    evidence_frames: List[Dict[str, Any]] = field(default_factory=list)

    # why a candidate did NOT become a finding
    confirmed: bool = True
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id, "role": self.role,
            "model": self.model_path,
            "model_class_id": self.model_class_id,
            "model_class_name": self.model_class_name,
            "camera_id": self.camera_id, "track_id": self.track_id,
            "start_frame": self.start_frame, "end_frame": self.end_frame,
            "start_time_local": round(self.start_time_local, 4),
            "end_time_local": round(self.end_time_local, 4),
            "n_observations": self.n_observations,
            "peak_confidence": round(self.peak_confidence, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "global_id": self.global_id,
            "association_status": self.association_status,
            "association_method": self.association_method,
            "association_detail": self.association_detail,
            "global_time_start": (round(self.global_time_start, 4)
                                  if self.global_time_start is not None else None),
            "global_time_end": (round(self.global_time_end, 4)
                                if self.global_time_end is not None else None),
            "camera_offset": (round(self.camera_offset, 4)
                              if self.camera_offset is not None else None),
            "overlap_fraction": (round(self.overlap_fraction, 4)
                                 if self.overlap_fraction is not None else None),
            "candidate_global_ids": list(self.candidate_global_ids),
            "peak_frame": self.peak_frame,
            "peak_bbox": ([round(v, 2) for v in self.peak_bbox]
                          if self.peak_bbox else None),
            "displacement_px": round(self.displacement_px, 2),
            "evidence_frames": list(self.evidence_frames),
            "confirmed": self.confirmed,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class WagonInspection:
    """Inspection findings for ONE existing global wagon.

    Holds `global_id` only -- no boundaries, no times, no index. The wagon's
    geometry lives in the roster the counting pipeline produced and is read from
    there, so this object cannot contradict it.
    """
    global_id: str
    classification: str = ""
    door_events: List[InspectionEvent] = field(default_factory=list)
    damage_events: List[InspectionEvent] = field(default_factory=list)
    camera_status: Dict[str, Dict[str, str]] = field(default_factory=dict)
    """camera -> {role -> status}. See the status constants above."""

    def _dominant(self, events: Sequence[InspectionEvent]) -> Dict[str, Any]:
        """The reported state for a role: the strongest confirmed finding.

        Confidence-weighted rather than most-frequent, because a single
        long-lived weak track should not outvote a decisive one, and because
        several classes routinely fire on the same physical object (measured:
        `partially_closed` 0.46, `open_door` 0.33 and `closed_door` 0.24 all on
        one door in one frame).
        """
        confirmed = [e for e in events if e.confirmed]
        if not confirmed:
            return {"state": None, "confidence": 0.0, "n_events": 0}
        weights: Dict[str, float] = {}
        for e in confirmed:
            weights[e.model_class_name] = (weights.get(e.model_class_name, 0.0)
                                           + e.peak_confidence * e.n_observations)
        best = max(sorted(weights), key=lambda k: weights[k])
        peak = max(e.peak_confidence for e in confirmed
                   if e.model_class_name == best)
        return {"state": best, "confidence": round(peak, 4),
                "n_events": len(confirmed)}

    @property
    def door_state(self) -> Dict[str, Any]:
        return self._dominant(self.door_events)

    @property
    def top_damage(self) -> Dict[str, Any]:
        return self._dominant(self.damage_events)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_id": self.global_id,
            "classification": self.classification,
            "door_state": self.door_state,
            "top_damage": self.top_damage,
            "door_events": [e.to_dict() for e in self.door_events],
            "damage_events": [e.to_dict() for e in self.damage_events],
            "camera_status": {c: dict(v) for c, v in sorted(self.camera_status.items())},
        }


@dataclass
class InspectionState:
    """Everything inspection produced, for the whole train.

    This is the single source of truth the PDF and the overlay renderer read.
    Neither re-runs a model, so neither can disagree with what was found.
    """
    enabled: bool = True
    model_availability: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    wagons: Dict[str, WagonInspection] = field(default_factory=dict)
    events: List[InspectionEvent] = field(default_factory=list)
    rejected_events: List[InspectionEvent] = field(default_factory=list)
    per_camera: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    tracks_by_key: Dict[Any, Any] = field(default_factory=dict, repr=False)
    """(camera_id, track_id) -> InspectionTrack, retained ONLY long enough for
    evidence selection. Never serialized: the per-observation detail would bloat
    the JSON, and the evidence frames carry everything the report needs."""

    # ---- summaries the report and console banner both read ----------------
    def counts(self) -> Dict[str, Any]:
        conf_door = [e for e in self.events if e.role == "door" and e.confirmed]
        conf_dmg = [e for e in self.events
                    if e.role == "top_damage" and e.confirmed]
        unresolved = [e for e in self.events
                      if e.association_status != ASSOCIATION_RESOLVED]
        return {
            "confirmed_door_events": len(conf_door),
            "confirmed_damage_events": len(conf_dmg),
            "rejected_events": len(self.rejected_events),
            "unresolved_associations": len(unresolved),
            "wagons_with_door_finding": sum(
                1 for w in self.wagons.values() if w.door_state["state"]),
            "wagons_with_damage_finding": sum(
                1 for w in self.wagons.values() if w.top_damage["state"]),
            "evidence_frames": sum(len(e.evidence_frames) for e in self.events),
        }

    def association_status(self) -> str:
        """RESOLVED only when every confirmed finding landed on a wagon."""
        if not self.events:
            return ASSOCIATION_RESOLVED
        bad = sum(1 for e in self.events
                  if e.association_status != ASSOCIATION_RESOLVED)
        if bad == 0:
            return ASSOCIATION_RESOLVED
        return ASSOCIATION_UNRESOLVED if bad == len(self.events) else "PARTIAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model_availability": dict(self.model_availability),
            "config": dict(self.config),
            "summary": {**self.counts(),
                        "association_status": self.association_status()},
            "wagons": {gid: w.to_dict() for gid, w in sorted(self.wagons.items(),
                                                             key=_gw_sort_key)},
            "events": [e.to_dict() for e in self.events],
            "rejected_events": [e.to_dict() for e in self.rejected_events],
            "per_camera": {c: dict(v) for c, v in sorted(self.per_camera.items())},
            "timings": {k: round(v, 3) for k, v in self.timings.items()},
            "warnings": list(self.warnings),
        }


def _gw_sort_key(item: Any) -> Any:
    """Sort GW ids numerically ('GW_2' before 'GW_10'), for stable output."""
    gid = item[0] if isinstance(item, tuple) else item
    tail = str(gid).split("_")[-1]
    return (0, int(tail)) if tail.isdigit() else (1, str(gid))
