"""
gap_validation.py  --  turn raw gap tracks into VALIDATED wagon boundaries
=========================================================================

A raw YOLO gap detection is a CANDIDATE, not a wagon boundary.

This module sits between the existing tracker and the fusion stage:

    GapTracker.process_video()        (UNCHANGED -- detection + tracking)
            |
            v   List[GapEvent]  with center_x_trajectory + hit_frames + bbox_history
            |
    validate_gap_events()            <-- THIS MODULE
            |
            +--> accepted : List[GapEvent]      -> wagon-boundary candidates
            +--> rejected : List[GapRejection]  -> diagnostics with reasons
            |
            v
    wagon window / fusion / global ids

WHY A SEPARATE LAYER
--------------------
The tracker is deliberately untouched: no detection threshold, Kalman parameter,
association gate, `min_hits` or `max_miss` value changes. Everything this module
needs is already recorded on each emitted `GapEvent`:

    center_x_trajectory   Kalman-smoothed bbox centre per hit
    hit_frames            the frame index of each hit (parallel array)
    bbox_history          the raw bbox per hit
    start_frame/end_frame track extent
    confidence            mean detection confidence over the track
    hit_count             number of frames the gap was actually detected

So validation is a pure, deterministic function of data the pipeline already
produces. No new model, no optical flow, no extra video pass -- which matters on
the CPU-only EC2 target.

THE PHYSICAL PRINCIPLE
----------------------
The train moves, so a real inter-wagon gap sweeps across the image. A detection
that keeps firing at the same pixel column is far more likely to be track
furniture, a shadow, a pole or a lighting artefact than a gap between two
wagons.

But "moves => valid" is NOT sufficient on its own: a moving false positive is
possible, and perspective makes apparent speed vary a lot between cameras and
across the frame. So several independent signals are combined, each with its own
recorded rejection reason:

    1. temporal persistence    enough hits, over enough frames
    2. detection continuity    no excessively long blind stretch inside the track
    3. motion                  enough absolute displacement
    4. speed plausibility      apparent speed inside a plausible band
    5. trajectory consistency  mostly one direction, not jittering back and forth
    6. confidence              mean and floor
    7. train-motion context    speed comparable to the other gaps in this camera
    8. duplicate suppression   one physical gap yields at most one GapEvent

Nothing is silently dropped: every rejection is returned with the reason and the
measured features, so `RAW -> TRACKED -> VALID` is fully auditable.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from global_train_state import GapEvent

# =============================================================================
# Rejection reasons
# =============================================================================

REJECTED_TOO_SHORT = "REJECTED_TOO_SHORT"
REJECTED_LOW_CONFIDENCE = "REJECTED_LOW_CONFIDENCE"
REJECTED_STATIC = "REJECTED_STATIC"
REJECTED_LOW_MOTION = "REJECTED_LOW_MOTION"
REJECTED_IMPLAUSIBLE_SPEED = "REJECTED_IMPLAUSIBLE_SPEED"
REJECTED_INCONSISTENT_TRAJECTORY = "REJECTED_INCONSISTENT_TRAJECTORY"
REJECTED_DETECTION_GAP = "REJECTED_DETECTION_GAP"
REJECTED_DUPLICATE = "REJECTED_DUPLICATE"
REJECTED_TRAIN_MOTION_MISMATCH = "REJECTED_TRAIN_MOTION_MISMATCH"
REJECTED_WRONG_DIRECTION = "REJECTED_WRONG_DIRECTION"
REJECTED_NO_TRAJECTORY = "REJECTED_NO_TRAJECTORY"

# Assigned later in the pipeline (wagon window), listed here so the vocabulary
# lives in one place.
REJECTED_OUTSIDE_WAGON_WINDOW = "REJECTED_OUTSIDE_WAGON_WINDOW"
REJECTED_NON_WAGON_REGION = "REJECTED_NON_WAGON_REGION"

ALL_REJECTION_REASONS = (
    REJECTED_TOO_SHORT, REJECTED_LOW_CONFIDENCE, REJECTED_STATIC,
    REJECTED_LOW_MOTION, REJECTED_IMPLAUSIBLE_SPEED,
    REJECTED_INCONSISTENT_TRAJECTORY, REJECTED_DETECTION_GAP,
    REJECTED_DUPLICATE, REJECTED_TRAIN_MOTION_MISMATCH,
    REJECTED_WRONG_DIRECTION, REJECTED_NO_TRAJECTORY,
    REJECTED_OUTSIDE_WAGON_WINDOW, REJECTED_NON_WAGON_REGION,
)


# =============================================================================
# Configuration -- every threshold named, documented and CLI-reachable
# =============================================================================

@dataclass
class ResolvedThresholds:
    """Config thresholds resolved into this camera's own pixels and frames.

    Produced by ``GapValidationConfig.resolve(frame_width, fps)``. Every value
    here is camera-specific and computed at runtime -- nothing is baked in from
    any particular train or camera geometry.
    """
    frame_width: int
    fps: float
    min_track_frames: int
    max_detection_gap_frames: int
    min_motion_px: float
    static_max_motion_px: float
    min_motion_px_per_sec: float
    max_motion_px_per_sec: float
    duplicate_max_center_px: float
    min_separation_frames: int

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


@dataclass
class GapValidationConfig:
    """Thresholds for turning gap tracks into wagon-boundary candidates.

    GENERALIZATION: thresholds are stored in CAMERA-INDEPENDENT units -- fractions
    of frame width for distances, seconds for durations, and dimensionless ratios
    -- and are resolved to this camera's pixels and frames at runtime by
    ``resolve(frame_width, fps)``.

    This matters because the same pipeline processes many trains on differing
    hardware. An absolute "4 px" static threshold calibrated at 848x480 becomes
    2.3x more permissive at 1920x1080 (the same physical jitter spans more
    pixels), so a stationary artefact would stop being rejected. An absolute
    "4 frame" minimum becomes half the wall-clock time at 30 fps. Relative units
    remove both failure modes.

    The numeric defaults below were MEASURED on the local development train and
    are initial defaults, not production invariants. They are expressed relative
    to that train's 848x480 @ 15 fps geometry so they carry over to other
    geometries; they still need confirming against additional trains before being
    treated as settled.

    Defaults were chosen from MEASURED behaviour of the real gap tracks on this
    project's videos (848x480 @ 15 fps). The measurement ran the EXISTING tracker
    over the first 1500 frames of three cameras:

      camera        tracks  |displacement| px    speed px/s      monotonic  conf   coverage
      RIGHT_UP        16     401 / 468 / 547     317 / 470 / 554   1.00      0.81+  1.00
      RIGHT_UP_TOP    18     111 / 340 / 614      74 / 387 / 546   0.61+     0.72+  0.58+
      LEFT_UP_TOP      8       0 / 248 / 354       0 / 237 / 356   0.52+     0.51+  0.43+
                             (min / median / max)

    The three zero-displacement LEFT_UP_TOP tracks are real false positives:
    30 hits each, coverage 1.00, confidence **0.93**, lasting 2.0 s, centre
    pinned at x=436, x=487 and x=283 respectively. Confidence, hit-count and
    coverage filters ALL pass them. Only motion rejects them -- which is why this
    layer exists.

    Every threshold below therefore has a measured margin against real gaps. The
    intent is to remove physically implausible artefacts, NOT to reach any
    particular wagon count.
    """

    enabled: bool = True
    """Master switch. False = emit every tracked gap (previous behaviour)."""

    # ---- 1. temporal persistence ----------------------------------------
    min_track_seconds: float = 0.27
    """Minimum track extent in SECONDS (fps-independent). The tracker already
    requires min_hits=3 to confirm a track, so this mostly guards against a
    track that is confirmed but collapses into almost no time span.
    Measured default: 4 frames at 15 fps."""

    min_hits: int = 3
    """Minimum number of frames the gap was actually detected. Matches the
    tracker's own confirmation rule, restated here so the validation layer is
    self-contained and auditable."""

    # ---- 2. detection continuity ----------------------------------------
    max_detection_gap_seconds: float = 1.33
    """Longest MISSED stretch tolerated inside one track, in SECONDS. A track may
    legitimately look like HIT HIT HIT MISS HIT HIT; a mostly-blind track is
    treated cautiously. Measured default: 20 frames at 15 fps."""

    min_coverage: float = 0.20
    """hits / track_extent. Guards the HIT MISS MISS MISS MISS HIT shape."""

    # ---- 3. motion ------------------------------------------------------
    min_motion_frac: float = 0.0142
    """Minimum centre displacement over the track, as a FRACTION OF FRAME WIDTH.
    Measured: the smallest displacement of any real gap was 110.7 px of 848
    (13.1% of width), so this ~1.4% floor leaves roughly a 9x margin while still
    excluding near-stationary detections."""

    static_max_motion_frac: float = 0.0047
    """At or below this displacement (FRACTION OF FRAME WIDTH) the track is
    reported as REJECTED_STATIC rather than REJECTED_LOW_MOTION, so pinned
    artefacts (rails, sleepers, poles, markings, shadows) are distinguishable in
    the diagnostics. Measured: the three confirmed false positives moved <=0.2 px
    of 848 while the smallest real gap moved 110.7 px, so anything in between
    separates them; ~0.5% of width absorbs Kalman jitter without approaching a
    real gap's motion. Being width-relative is essential -- a fixed 4 px would
    stop rejecting static objects at higher resolutions."""

    # ---- 4. speed plausibility ------------------------------------------
    min_motion_frac_per_sec: float = 0.0094
    max_motion_frac_per_sec: float = 2.36
    """Plausible band for apparent speed, in FRACTIONS OF FRAME WIDTH per second.
    Wide on purpose: perspective makes the same physical gap move at very
    different rates on side versus top cameras and across the frame, and trains
    accelerate. The band excludes the physically absurd, not a specific expected
    speed. Measured defaults: 8 and 2000 px/s at 848 px width."""

    # ---- 5. trajectory consistency --------------------------------------
    min_monotonic_fraction: float = 0.60
    """Fraction of consecutive inter-hit steps that must share the dominant
    direction. A real gap carried by the train moves consistently one way;
    a detection that jitters left-right-left is not tracking a passing object.
    Not 1.0, because Kalman smoothing plus detection noise legitimately produces
    the occasional backward step."""

    min_steps_for_trajectory: int = 3
    """Below this many inter-hit steps the direction statistic is meaningless,
    so the trajectory test is skipped rather than guessed."""

    # ---- 6. confidence ---------------------------------------------------
    min_mean_confidence: float = 0.45
    """Mean detection confidence over the track. Note the detector's own
    threshold (0.4, UNCHANGED) already applies per frame; this asks the track as
    a whole to be a little better than the per-frame floor."""

    # ---- 7. train-motion context ----------------------------------------
    train_motion_check_enabled: bool = True
    min_tracks_for_train_reference: int = 5
    """A per-camera reference speed is only computed when at least this many
    tracks survived the earlier tests -- otherwise the median is not meaningful
    and the check is skipped."""

    train_motion_tolerance: float = 4.0
    """A track's speed may differ from the camera's median gap speed by up to
    this FACTOR (either direction) before it is rejected.

    Measured: within one camera the real gaps span only ~1.9x (RIGHT_UP_TOP
    285-546 px/s), and the train's own deceleration across a full pass adds
    roughly another 1.8x, so ~2.5x is the realistic worst case. 4.0 keeps a
    margin above that while still catching gross outliers -- e.g. the measured
    RIGHT_UP_TOP track at 74 px/s against that camera's 387 px/s median, which
    is a tracker latch onto something other than a passing gap.

    Erring toward rejection here is deliberate: an under-count is a reported
    number, whereas a fabricated wagon is a wrong one."""

    direction_check_enabled: bool = True
    """Reject a track that travels against the camera's dominant gap direction.

    Measured: gap motion direction is per-camera, not global -- RIGHT_UP and
    RIGHT_UP_TOP gaps move in -x, LEFT_UP_TOP gaps move in +x. The dominant
    direction is therefore derived from each camera's own surviving tracks, never
    assumed."""

    min_tracks_for_direction_reference: int = 5
    """Below this many survivors the dominant direction is not established and
    the check is skipped rather than guessed."""

    # ---- 8. duplicate suppression ---------------------------------------
    duplicate_suppression_enabled: bool = True
    duplicate_min_time_overlap: float = 0.30
    """Two tracks are candidates for being the SAME physical gap only when their
    frame ranges genuinely OVERLAP by at least this fraction of the shorter
    track. Time overlap (rather than mere proximity) is used deliberately: two
    distinct inter-wagon gaps are temporally disjoint, so this rule cannot merge
    two real wagons."""

    duplicate_max_center_frac: float = 0.1415
    """...and their centre columns must also be within this distance (FRACTION OF
    FRAME WIDTH), i.e. the two tracks follow the same object in the same part of
    the image. Measured default: 120 px at 848 px width."""

    min_separation_seconds: float = 0.67
    """Minimum time between two consecutive VALIDATED physical gap events.

    A measured physical constraint of the observed train: consecutive real wagon
    gaps were never closer than ~10 frames at 15 fps. Stored in SECONDS so it
    transfers to other frame rates, and applied ONLY to final validated events --
    never to raw detections, which legitimately cluster because several belong to
    one track.

    Treated as an initial measured default, not a production invariant: a shorter
    wagon or a faster train could in principle produce closer boundaries, so a
    violation is resolved as a suspected duplicate/fragmentation WITH diagnostics
    rather than silently deleted."""

    def resolve(self, frame_width: int, fps: float) -> ResolvedThresholds:
        """Resolve camera-independent thresholds into this camera's units.

        Falls back to the development geometry only when a camera reports no
        usable width or fps, so a malformed stream cannot silently disable
        validation.
        """
        w = int(frame_width) if frame_width and frame_width > 0 else 848
        f = float(fps) if fps and fps > 0 else 15.0
        return ResolvedThresholds(
            frame_width=w, fps=f,
            min_track_frames=max(2, int(round(self.min_track_seconds * f))),
            max_detection_gap_frames=max(
                1, int(round(self.max_detection_gap_seconds * f))),
            min_motion_px=self.min_motion_frac * w,
            static_max_motion_px=self.static_max_motion_frac * w,
            min_motion_px_per_sec=self.min_motion_frac_per_sec * w,
            max_motion_px_per_sec=self.max_motion_frac_per_sec * w,
            duplicate_max_center_px=self.duplicate_max_center_frac * w,
            min_separation_frames=max(1, int(round(self.min_separation_seconds * f))),
        )

    def describe(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


DEFAULT_GAP_VALIDATION = GapValidationConfig()


# =============================================================================
# Motion features
# =============================================================================

@dataclass
class GapMotionFeatures:
    """Deterministic motion description of one gap track."""
    track_id: int
    camera_id: str
    frame_start: int
    frame_end: int
    time_start: float
    time_end: float
    duration_s: float
    track_frames: int
    hits: int
    coverage: float
    max_detection_gap: int
    center_start: float
    center_end: float
    displacement_px: float
    abs_displacement_px: float
    velocity_px_per_sec: float
    direction: int                     # +1, -1 or 0
    monotonic_fraction: float
    n_steps: int
    step_velocity_median: Optional[float]
    mean_confidence: float
    min_confidence: Optional[float]
    bbox_height_median: Optional[float]
    bbox_width_median: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for k, v in list(d.items()):
            if isinstance(v, float):
                d[k] = round(v, 4)
        return d


@dataclass
class GapRejection:
    """One rejected candidate, with the reason and the measured evidence."""
    reason: str
    detail: str
    features: GapMotionFeatures

    def to_dict(self) -> Dict[str, Any]:
        return {"reason": self.reason, "detail": self.detail,
                "features": self.features.to_dict()}


@dataclass
class GapValidationResult:
    """Everything one camera's validation pass produced."""
    camera_id: str
    accepted: List[GapEvent] = field(default_factory=list)
    rejected: List[GapRejection] = field(default_factory=list)
    features: List[GapMotionFeatures] = field(default_factory=list)
    raw_detection_count: int = 0
    tracked_candidate_count: int = 0
    train_reference_speed: Optional[float] = None
    config_used: Dict[str, Any] = field(default_factory=dict)
    resolved_thresholds: Dict[str, Any] = field(default_factory=dict)
    """The camera-independent config resolved into this camera's px/frames."""

    @property
    def rejection_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self.rejected:
            out[r.reason] = out.get(r.reason, 0) + 1
        return out

    def to_dict(self, include_rejections: bool = True) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "camera_id": self.camera_id,
            "raw_detections": self.raw_detection_count,
            "tracked_candidates": self.tracked_candidate_count,
            "valid_gap_events": len(self.accepted),
            "rejected_total": len(self.rejected),
            "rejection_counts": self.rejection_counts,
            "train_reference_speed_px_per_sec": (
                round(self.train_reference_speed, 2)
                if self.train_reference_speed is not None else None),
            "resolved_thresholds": dict(self.resolved_thresholds),
        }
        if include_rejections:
            d["rejections"] = [r.to_dict() for r in self.rejected]
        return d


# =============================================================================
# Feature extraction
# =============================================================================

def compute_motion_features(gap: GapEvent) -> Optional[GapMotionFeatures]:
    """Derive motion features from data the tracker already recorded.

    Returns None when the track carries no usable trajectory (fewer than two
    hits), which the caller reports as REJECTED_NO_TRAJECTORY.
    """
    traj = list(gap.center_x_trajectory or [])
    hits = list(gap.hit_frames or [])
    fps = gap.fps or 0.0

    if len(traj) < 2 or len(hits) < 2 or fps <= 0:
        return None

    n = min(len(traj), len(hits))
    traj, hits = traj[:n], hits[:n]

    track_frames = max(1, gap.end_frame - gap.start_frame + 1)
    duration = track_frames / fps

    # Longest run of consecutive missed frames inside the track.
    max_gap = 0
    for a, b in zip(hits, hits[1:]):
        max_gap = max(max_gap, b - a - 1)

    # Per-step apparent velocity between consecutive detections.
    steps: List[float] = []
    for (f0, x0), (f1, x1) in zip(zip(hits, traj), zip(hits[1:], traj[1:])):
        df = f1 - f0
        if df > 0:
            steps.append((x1 - x0) / (df / fps))

    signs = [1 if s > 0 else (-1 if s < 0 else 0) for s in steps]
    n_pos = sum(1 for s in signs if s > 0)
    n_neg = sum(1 for s in signs if s < 0)
    dominant = 0
    if n_pos > n_neg:
        dominant = 1
    elif n_neg > n_pos:
        dominant = -1
    monotonic = (max(n_pos, n_neg) / len(signs)) if signs else 0.0

    displacement = traj[-1] - traj[0]
    abs_disp = abs(displacement)

    heights = [b[3] - b[1] for b in (gap.bbox_history or []) if len(b) >= 4]
    widths = [b[2] - b[0] for b in (gap.bbox_history or []) if len(b) >= 4]

    return GapMotionFeatures(
        track_id=gap.track_id, camera_id=gap.camera_id,
        frame_start=gap.start_frame, frame_end=gap.end_frame,
        time_start=gap.start_time, time_end=gap.end_time,
        duration_s=duration, track_frames=track_frames,
        hits=gap.hit_count, coverage=min(1.0, gap.hit_count / track_frames),
        max_detection_gap=max_gap,
        center_start=traj[0], center_end=traj[-1],
        displacement_px=displacement, abs_displacement_px=abs_disp,
        velocity_px_per_sec=(abs_disp / duration) if duration > 0 else 0.0,
        direction=dominant, monotonic_fraction=monotonic, n_steps=len(steps),
        step_velocity_median=(statistics.median(steps) if steps else None),
        mean_confidence=gap.confidence, min_confidence=None,
        bbox_height_median=(statistics.median(heights) if heights else None),
        bbox_width_median=(statistics.median(widths) if widths else None),
    )


# =============================================================================
# Validation
# =============================================================================

def _time_overlap_fraction(a: GapEvent, b: GapEvent) -> float:
    """Overlap of two frame ranges as a fraction of the shorter range."""
    lo = max(a.start_frame, b.start_frame)
    hi = min(a.end_frame, b.end_frame)
    overlap = max(0, hi - lo + 1)
    shorter = min(a.end_frame - a.start_frame + 1, b.end_frame - b.start_frame + 1)
    return (overlap / shorter) if shorter > 0 else 0.0


def validate_gap_events(
    gaps: Sequence[GapEvent],
    camera_id: str,
    cfg: GapValidationConfig = DEFAULT_GAP_VALIDATION,
    raw_detection_count: int = 0,
    verbose: bool = True,
    frame_width: int = 0,
    fps: float = 0.0,
) -> GapValidationResult:
    """Filter tracked gap candidates down to physically plausible wagon boundaries.

    Deterministic and order-independent: the same input always yields the same
    accepted/rejected split. Nothing is discarded silently -- every rejection
    carries its reason and its measured features.
    """
    # Resolve camera-independent thresholds into THIS camera's pixels/frames.
    # Geometry comes from the caller, or from the gaps themselves as a fallback,
    # so nothing is assumed about resolution or frame rate.
    if not fps:
        fps = next((g.fps for g in gaps if g.fps), 0.0)
    res_thr = cfg.resolve(frame_width, fps)

    result = GapValidationResult(
        camera_id=camera_id,
        raw_detection_count=raw_detection_count,
        tracked_candidate_count=len(gaps),
        config_used=cfg.describe(),
        resolved_thresholds=res_thr.to_dict(),
    )

    if not cfg.enabled:
        result.accepted = list(gaps)
        for g in gaps:
            f = compute_motion_features(g)
            if f:
                result.features.append(f)
        if verbose:
            print(f"  [GAPVAL/{camera_id}] validation disabled -- "
                  f"passing all {len(gaps)} tracked candidate(s) through")
        return result

    # ---- pass 1: per-track tests, independent of the other tracks ----
    survivors: List[Tuple[GapEvent, GapMotionFeatures]] = []
    for g in sorted(gaps, key=lambda x: (x.center_frame, x.track_id)):
        f = compute_motion_features(g)
        if f is None:
            result.rejected.append(GapRejection(
                REJECTED_NO_TRAJECTORY,
                "fewer than two tracked hits, or no fps: no motion can be measured",
                GapMotionFeatures(
                    track_id=g.track_id, camera_id=g.camera_id,
                    frame_start=g.start_frame, frame_end=g.end_frame,
                    time_start=g.start_time, time_end=g.end_time,
                    duration_s=0.0, track_frames=0, hits=g.hit_count,
                    coverage=0.0, max_detection_gap=0, center_start=0.0,
                    center_end=0.0, displacement_px=0.0, abs_displacement_px=0.0,
                    velocity_px_per_sec=0.0, direction=0, monotonic_fraction=0.0,
                    n_steps=0, step_velocity_median=None,
                    mean_confidence=g.confidence, min_confidence=None,
                    bbox_height_median=None, bbox_width_median=None)))
            continue

        result.features.append(f)
        reason: Optional[str] = None
        detail = ""

        # 1. temporal persistence
        if f.track_frames < res_thr.min_track_frames:
            reason = REJECTED_TOO_SHORT
            detail = (f"track spans {f.track_frames} frame(s) "
                      f"< min_track_frames={res_thr.min_track_frames}")
        elif f.hits < cfg.min_hits:
            reason = REJECTED_TOO_SHORT
            detail = f"only {f.hits} hit(s) < min_hits={cfg.min_hits}"

        # 2. detection continuity
        elif f.max_detection_gap > res_thr.max_detection_gap_frames:
            reason = REJECTED_DETECTION_GAP
            detail = (f"longest blind run {f.max_detection_gap} frame(s) "
                      f"> max_detection_gap_frames={res_thr.max_detection_gap_frames}")
        elif f.coverage < cfg.min_coverage:
            reason = REJECTED_DETECTION_GAP
            detail = (f"coverage {f.coverage:.2f} < min_coverage={cfg.min_coverage} "
                      f"({f.hits} hits over {f.track_frames} frames)")

        # 3. confidence
        elif f.mean_confidence < cfg.min_mean_confidence:
            reason = REJECTED_LOW_CONFIDENCE
            detail = (f"mean confidence {f.mean_confidence:.2f} "
                      f"< min_mean_confidence={cfg.min_mean_confidence}")

        # 4. motion: static artefacts are called out explicitly
        elif f.abs_displacement_px <= res_thr.static_max_motion_px:
            reason = REJECTED_STATIC
            detail = (f"centre moved {f.abs_displacement_px:.1f} px over "
                      f"{f.duration_s:.2f}s (<= static_max_motion_px="
                      f"{res_thr.static_max_motion_px}); the train is moving, so a "
                      f"pinned detection is background, not a wagon gap")
        elif f.abs_displacement_px < res_thr.min_motion_px:
            reason = REJECTED_LOW_MOTION
            detail = (f"centre moved only {f.abs_displacement_px:.1f} px "
                      f"< min_motion_px={res_thr.min_motion_px}")

        # 5. speed plausibility
        elif f.velocity_px_per_sec < res_thr.min_motion_px_per_sec:
            reason = REJECTED_IMPLAUSIBLE_SPEED
            detail = (f"apparent speed {f.velocity_px_per_sec:.1f} px/s "
                      f"< min_motion_px_per_sec={res_thr.min_motion_px_per_sec}")
        elif f.velocity_px_per_sec > res_thr.max_motion_px_per_sec:
            reason = REJECTED_IMPLAUSIBLE_SPEED
            detail = (f"apparent speed {f.velocity_px_per_sec:.1f} px/s "
                      f"> max_motion_px_per_sec={res_thr.max_motion_px_per_sec}")

        # 6. trajectory consistency
        elif (f.n_steps >= cfg.min_steps_for_trajectory
                and f.monotonic_fraction < cfg.min_monotonic_fraction):
            reason = REJECTED_INCONSISTENT_TRAJECTORY
            detail = (f"only {f.monotonic_fraction:.2f} of {f.n_steps} steps share "
                      f"the dominant direction "
                      f"< min_monotonic_fraction={cfg.min_monotonic_fraction}")

        if reason:
            result.rejected.append(GapRejection(reason, detail, f))
        else:
            survivors.append((g, f))

    # ---- pass 2a: dominant direction, derived per camera (never assumed) ----
    if (cfg.direction_check_enabled
            and len(survivors) >= cfg.min_tracks_for_direction_reference):
        n_pos = sum(1 for _, f in survivors if f.direction > 0)
        n_neg = sum(1 for _, f in survivors if f.direction < 0)
        dominant = 1 if n_pos > n_neg else (-1 if n_neg > n_pos else 0)
        if dominant != 0:
            kept: List[Tuple[GapEvent, GapMotionFeatures]] = []
            for g, f in survivors:
                if f.direction != 0 and f.direction != dominant:
                    result.rejected.append(GapRejection(
                        REJECTED_WRONG_DIRECTION,
                        f"track travels in {'+x' if f.direction > 0 else '-x'} but "
                        f"this camera's gaps travel in "
                        f"{'+x' if dominant > 0 else '-x'} "
                        f"({max(n_pos, n_neg)} of {len(survivors)} tracks); a gap "
                        f"moving against the train is not a wagon boundary", f))
                else:
                    kept.append((g, f))
            survivors = kept

    # ---- pass 2b: train-motion context (needs the surviving population) ----
    if (cfg.train_motion_check_enabled
            and len(survivors) >= cfg.min_tracks_for_train_reference):
        speeds = [f.velocity_px_per_sec for _, f in survivors
                  if f.velocity_px_per_sec > 0]
        if speeds:
            ref = statistics.median(speeds)
            result.train_reference_speed = ref
            kept: List[Tuple[GapEvent, GapMotionFeatures]] = []
            lo = ref / cfg.train_motion_tolerance
            hi = ref * cfg.train_motion_tolerance
            for g, f in survivors:
                v = f.velocity_px_per_sec
                if v < lo or v > hi:
                    result.rejected.append(GapRejection(
                        REJECTED_TRAIN_MOTION_MISMATCH,
                        f"apparent speed {v:.1f} px/s is outside "
                        f"[{lo:.1f}, {hi:.1f}] px/s, i.e. more than "
                        f"{cfg.train_motion_tolerance}x from this camera's median "
                        f"gap speed {ref:.1f} px/s", f))
                else:
                    kept.append((g, f))
            survivors = kept

    # ---- pass 3: duplicate suppression -- one physical gap, one GapEvent ----
    if cfg.duplicate_suppression_enabled:
        deduped: List[Tuple[GapEvent, GapMotionFeatures]] = []
        for g, f in survivors:
            clash = None
            for kg, kf in deduped:
                if (_time_overlap_fraction(g, kg) >= cfg.duplicate_min_time_overlap
                        and abs(f.center_start - kf.center_start)
                        <= res_thr.duplicate_max_center_px):
                    clash = (kg, kf)
                    break
            if clash is None:
                deduped.append((g, f))
            else:
                kg, _kf = clash
                # Keep the better-evidenced track: more hits, then higher conf.
                if (f.hits, f.mean_confidence) > (kg.hit_count, kg.confidence):
                    deduped = [(x, y) for x, y in deduped if x is not kg]
                    deduped.append((g, f))
                    loser, loser_f = kg, _kf
                else:
                    loser, loser_f = g, f
                result.rejected.append(GapRejection(
                    REJECTED_DUPLICATE,
                    f"track {loser.track_id} overlaps track "
                    f"{(kg if loser is g else g).track_id} in time and position: "
                    f"same physical gap, so only one GapEvent is kept",
                    loser_f))
        survivors = sorted(deduped, key=lambda t: (t[0].center_frame, t[0].track_id))

    result.accepted = [g for g, _ in survivors]

    if verbose:
        rc = result.rejection_counts
        print(f"  [GAPVAL/{camera_id}] raw_detections={result.raw_detection_count}  "
              f"tracked_candidates={result.tracked_candidate_count}  "
              f"valid={len(result.accepted)}  rejected={len(result.rejected)}")
        if rc:
            for reason in ALL_REJECTION_REASONS:
                if reason in rc:
                    print(f"      {reason:<36} {rc[reason]}")
        if result.train_reference_speed is not None:
            print(f"      train reference speed: "
                  f"{result.train_reference_speed:.1f} px/s (median of survivors)")

    return result


def renumber_gap_events(gaps: Sequence[GapEvent]) -> List[GapEvent]:
    """Re-assign track_id 1..N in temporal order, as the tracker does.

    Validation removes tracks, which would otherwise leave holes in the id
    sequence. The pipeline's downstream code treats `track_id` as a temporal
    rank within the camera, so it is restored here.
    """
    out: List[GapEvent] = []
    for new_id, g in enumerate(sorted(gaps, key=lambda x: (x.center_frame, x.track_id)),
                               start=1):
        g.track_id = new_id
        out.append(g)
    return out
