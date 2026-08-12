"""
global_fusion.py  --  fixed-master global fusion (Phase-1 wagon counting)
=========================================================================

THE HARD INVARIANT
------------------
    RIGHT_UP final gaps  ==  global gaps
    total_wagons         ==  global gaps + 1        (existing N -> N+1 convention)

RIGHT_UP is the master camera and its gap sequence is COMPLETE, FINAL and
IMMUTABLE.  The global gap sequence is created *first*, solely by enumerating
RIGHT_UP's gaps.  It is frozen from that moment on.

LEFT_UP, RIGHT_UP_TOP and LEFT_UP_TOP are auxiliary observers.  The only
question they answer is:

    "which already-existing RIGHT_UP global gap does this observation of mine
     belong to?"

They may therefore produce, per observation, exactly one of:

    MATCH    -- this observation is evidence for global gap G_i
    MISSING  -- (per global gap) this camera has no observation for G_i
    EXTRA    -- this observation corresponds to no global gap  (diagnostic only)

None of those three outcomes can create, delete, split or merge a global gap.
There is deliberately NO support-camera insertion / quorum / synthetic-gap
mechanism in this module.  Compare `global_alignment.decide_inserted_gaps`,
which this module replaces: that function promoted unmatched support gaps into
the authoritative timeline (`master_gaps + synth_gaps`) and inflated the count.

WHY THE COUNT IS INDEPENDENT OF SYNCHRONIZATION
-----------------------------------------------
Camera temporal offsets are estimated (they are genuinely unknown -- the four
videos do not share t=0), but they are used ONLY to associate support
observations with the fixed global sequence.  Consequently:

    wrong offset -> worse evidence association          (acceptable)
    wrong offset -> new / changed global gap            (IMPOSSIBLE here)

If an offset cannot be resolved decisively the camera is marked UNRESOLVED and
contributes no evidence at all.  The count does not move.

MODULE LAYOUT
-------------
    FusionConfig                     every tunable, in one place
    GapObservation                   one local detection, fully traceable
    CameraOffset                     t_global = t_local + delta, plus status
    SupportAlignment                 MATCH / MISSING / EXTRA for one camera
    GlobalGap                        one master gap + its evidence children

    to_gap_observations()            GapEvent -> GapObservation (pure adapter)
    align_to_master()                order-preserving DP  (Phase 3b)
    estimate_camera_offset()         offset search + anti-aliasing (Phase 3c)
    build_global_gap_sequence()      RIGHT_UP -> the frozen global sequence
    attach_support_evidence()        hang observations off existing gaps
    diagnose_intervals()             report-only physical plausibility
    assert_invariants()              fail loudly, never silently
    assemble_global_train_state_master_fixed()      end-to-end
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from global_train_state import (
    ALL_CAMERAS,
    MASTER_CAMERA,
    GapEvent,
    GlobalTrainState,
    GlobalWagon,
    LocalCameraTracks,
    SegmentClass,
    _MasterClassification,
)
# build_global_wagons is REUSED UNCHANGED: its `b <= prev` collapse rule,
# N gaps -> N+1 segmentation and GW_{i} numbering already provide exactly the
# required counting behaviour.
from global_alignment import build_global_wagons


# =============================================================================
# Status / operation vocabulary
# =============================================================================

OFFSET_REFERENCE = "REFERENCE"     # the master; delta is 0 by definition
OFFSET_RESOLVED = "RESOLVED"       # decisive winner found
OFFSET_UNRESOLVED = "UNRESOLVED"   # ambiguous / insufficient -> contributes nothing

OP_MATCH = "MATCH"
OP_MISSING = "MISSING"
OP_EXTRA = "EXTRA"

REASON_OUT_OF_RANGE = "out of range: event lies outside this camera's footage"
REASON_OFFSET_UNRESOLVED = "offset unresolved: camera could not be synchronized"
REASON_ALIGNMENT_ERROR = "alignment error"
REASON_NO_GAPS = "camera produced no gap observations"
REASON_NO_METADATA = "camera reported no usable video metadata"


class FusionInvariantError(AssertionError):
    """Raised when the fixed-master invariant is violated. Never silenced."""


# =============================================================================
# Configuration -- every tunable lives here, nothing hidden inside functions
# =============================================================================

@dataclass
class FusionConfig:
    """All fusion parameters. Values are starting points traceable to measured
    behaviour of this project's data (see GLOBAL_FUSION_DESIGN.md sections 4-5);
    none of them can affect the wagon count."""

    # ---- camera offset search -------------------------------------------
    offset_search_s: float = 35.0
    """Half-range of the offset sweep, seconds. Measured offsets on the current
    dataset reach ~29 s, so 35 s covers them with margin."""

    offset_coarse_step_s: float = 0.25
    """Coarse sweep step (~4 frames at 15 fps)."""

    offset_fine_step_s: float = 0.0
    """Refinement step. 0.0 means 'one frame' (1/fps), resolved per camera."""

    offset_fine_window_s: float = 1.0
    """Half-width of the fine refinement window around the coarse winner."""

    offset_min_margin_ratio: float = 0.10
    """A candidate offset is RESOLVED only if it beats the best rival that is at
    least `alias_separation` away by this relative score margin. The wagon
    sequence is quasi-periodic, so a shift by a whole number of wagon periods
    produces a deceptively good alignment; this is the primary guard.

    Choice of 0.10 on this dataset: the estimator and a fully independent
    method (max hit-count within +/-0.5 s, swept separately) agree within 0.5 s
    on all three support cameras -- LEFT_UP +16.63 vs +16.20, RIGHT_UP_TOP
    -3.32 vs -3.15, LEFT_UP_TOP +28.50 vs +28.80 -- yet their margins are only
    10.2%, 11.5% and 2.2%. A 0.15 default therefore rejected offsets that two
    independent methods corroborate, losing all support evidence for no gain:
    the wagon count is provably independent of these values (verified: 53 wagons
    at every margin from 0.00 to 0.30). 0.10 accepts the two corroborated
    cameras and still refuses the weakest (LEFT_UP_TOP, 2.2%).

    Raise it with --offset-min-margin to be stricter; the count will not move."""

    offset_min_match_fraction: float = 0.30
    """RESOLVED also requires matching at least this fraction of
    min(len(master), len(support)) observations."""

    alias_separation_s: float = 0.0
    """Minimum separation for two offsets to count as genuinely different
    hypotheses. 0.0 means 'derive from the data' = median master gap spacing."""

    pattern_weight: float = 2.0
    """Weight of the interval-pattern disagreement term in the offset objective.
    This term is what distinguishes the true offset from a whole-wagon-period
    alias: a k-wagon shift matches gaps whose *local spacing* disagrees wherever
    the train's speed varies."""

    # ---- match cost ------------------------------------------------------
    match_cost_mode: str = "fixed"
    """'fixed'  -> tolerance-normalized |dt|            (default; no derived values)
       'sigma'  -> |dt| / sqrt(sigma_i^2 + sigma_j^2)   (sigma from track span)"""

    match_tolerance_s: float = 0.50
    """'fixed' mode normalizer. Median gap-track half-span on this dataset is
    0.30-0.53 s per camera, so 0.5 s is the observed timing scale.

    NOTE: because missing_penalty + extra_penalty = 2.0 by default, a MATCH is
    only preferred over MISSING+EXTRA while match_cost < 2.0, i.e. while
    |dt| < 2 * match_tolerance_s. The gate, tolerance and penalties therefore
    define the effective matching window JOINTLY and must be read together."""

    match_gate_s: float = 1.50
    """Hard gate in 'fixed' mode: beyond this, a match is never considered."""

    sigma_floor_s: float = 0.10
    sigma_cap_s: float = 1.00
    """'sigma' mode bounds. The cap matters: a merged track on this dataset had a
    78-frame (5.2 s) span, whose half-span would otherwise swamp the metric."""

    match_gate_sigmas: float = 3.0
    """Hard gate in 'sigma' mode."""

    confidence_weight: float = 0.25
    """Small multiplicative penalty for low-confidence pairs."""

    missing_penalty: float = 1.0
    extra_penalty: float = 1.0
    """DP penalties for MISSING / EXTRA. See the note on match_tolerance_s."""

    # ---- physical interval diagnostics (REPORT ONLY) ---------------------
    local_base_window: int = 6
    """Intervals either side used for the local median spacing. The train's
    speed drifts (~3.7 s spacing early, ~6.6 s late on this dataset), so a
    single global threshold would be wrong -- plausibility must be local."""

    short_interval_ratio: float = 0.50
    long_interval_ratio: float = 1.60
    """Flag thresholds. Measured: the one suspicious short interval sits at
    ratio 0.14 (next-smallest interval is 0.62), and every suspicious long
    interval is >= 1.67."""

    # ---- safety ----------------------------------------------------------
    strict_invariants: bool = True
    """True -> violations raise FusionInvariantError. False -> loud warning plus
    a state note (field diagnosis only; never use in production)."""


DEFAULT_CONFIG = FusionConfig()


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class GapObservation:
    """One local gap detection, with full provenance back to source frames."""
    camera_id: str
    local_track_id: int
    local_frame: float
    local_time: float
    confidence: float
    start_frame: int
    end_frame: int
    fps: float
    hit_count: int = 0
    temporal_consistency_score: float = 0.0
    center_x: Optional[float] = None
    global_time: Optional[float] = None      # local_time + delta_c

    @property
    def span_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)

    def sigma(self, cfg: FusionConfig) -> float:
        """Timing tolerance derived from how long the gap was actually visible."""
        if self.fps <= 0:
            return cfg.sigma_cap_s
        half_span = (self.span_frames / 2.0) / self.fps
        return min(cfg.sigma_cap_s, max(cfg.sigma_floor_s, half_span))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "local_track_id": self.local_track_id,
            "local_frame": round(self.local_frame, 4),
            "local_time": round(self.local_time, 4),
            "global_time": (round(self.global_time, 4)
                            if self.global_time is not None else None),
            "confidence": round(self.confidence, 4),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "span_frames": self.span_frames,
            "hit_count": self.hit_count,
            "temporal_consistency_score": round(self.temporal_consistency_score, 4),
        }


@dataclass
class CameraOffset:
    """t_global = t_local + delta, with an explicit resolution status."""
    camera_id: str
    delta: float = 0.0
    status: str = OFFSET_UNRESOLVED
    score: float = float("inf")
    margin_ratio: float = 0.0
    runner_up_delta: Optional[float] = None
    n_match: int = 0
    n_missing: int = 0
    n_extra: int = 0
    pattern_penalty: float = 0.0
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.status in (OFFSET_REFERENCE, OFFSET_RESOLVED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "delta": round(self.delta, 4),
            "status": self.status,
            "score": (round(self.score, 4) if math.isfinite(self.score) else None),
            "margin_ratio": round(self.margin_ratio, 4),
            "runner_up_delta": (round(self.runner_up_delta, 4)
                                if self.runner_up_delta is not None else None),
            "n_match": self.n_match,
            "n_missing": self.n_missing,
            "n_extra": self.n_extra,
            "pattern_penalty": round(self.pattern_penalty, 4),
            "reason": self.reason,
        }


@dataclass
class SupportAlignment:
    """Result of aligning one support camera to the FIXED master sequence."""
    camera_id: str
    offset: CameraOffset
    matches: Dict[int, GapObservation] = field(default_factory=dict)
    missing_global_gap_ids: List[int] = field(default_factory=list)
    extra_observations: List[GapObservation] = field(default_factory=list)
    non_wagon_observations: List[GapObservation] = field(default_factory=list)
    """Observations in this camera's engine / brake-van region: excluded from
    wagon alignment, kept for diagnostics. They can never create a global gap."""
    total_cost: float = 0.0
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "status": self.status,
            "offset": self.offset.to_dict(),
            "n_match": len(self.matches),
            "n_missing": len(self.missing_global_gap_ids),
            "n_extra": len(self.extra_observations),
            "n_non_wagon_excluded": len(self.non_wagon_observations),
            "total_cost": (round(self.total_cost, 4)
                           if math.isfinite(self.total_cost) else None),
            "matched_global_gap_ids": sorted(self.matches.keys()),
            "missing_global_gap_ids": list(self.missing_global_gap_ids),
        }


@dataclass
class GlobalGap:
    """ONE physical wagon boundary.

    `master_observation` is mandatory: a GlobalGap cannot exist without a
    RIGHT_UP source. That is the hard invariant expressed in the type itself.
    Support observations are *children* / evidence, never peers.
    """
    global_gap_id: int
    master_observation: GapObservation
    master_camera: str = MASTER_CAMERA
    support_observations: Dict[str, GapObservation] = field(default_factory=dict)
    missing_cameras: List[str] = field(default_factory=list)
    unavailable_cameras: Dict[str, str] = field(default_factory=dict)
    time_residuals: Dict[str, float] = field(default_factory=dict)
    weighted_time: Optional[float] = None     # DIAGNOSTIC ONLY
    flags: List[str] = field(default_factory=list)

    @property
    def master_frame(self) -> int:
        return int(round(self.master_observation.local_frame))

    @property
    def master_time(self) -> float:
        return self.master_observation.local_time

    @property
    def support_count(self) -> int:
        return len(self.support_observations)

    @property
    def supporting_camera_ids(self) -> List[str]:
        return [c for c in ALL_CAMERAS if c in self.support_observations]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_gap_id": self.global_gap_id,
            "master_camera": self.master_camera,
            "master_frame": self.master_frame,
            "master_time": round(self.master_time, 4),
            "master_track_id": self.master_observation.local_track_id,
            "master_confidence": round(self.master_observation.confidence, 4),
            "support_count": self.support_count,
            "supporting_cameras": self.supporting_camera_ids,
            "support_observations": {c: o.to_dict()
                                     for c, o in self.support_observations.items()},
            "missing_cameras": list(self.missing_cameras),
            "unavailable_cameras": dict(self.unavailable_cameras),
            "time_residuals": {c: round(v, 4) for c, v in self.time_residuals.items()},
            "weighted_time": (round(self.weighted_time, 4)
                              if self.weighted_time is not None else None),
            "flags": list(self.flags),
        }


# =============================================================================
# Phase 3a -- adapter: GapEvent -> GapObservation
# =============================================================================

def to_gap_observations(tracks: LocalCameraTracks) -> List[GapObservation]:
    """Wrap a camera's GapEvents as GapObservations, in temporal order.

    Pure adapter. Detection and tracking are untouched; nothing is discarded.
    """
    out: List[GapObservation] = []
    for g in sorted(tracks.gaps, key=lambda x: (x.center_frame, x.track_id)):
        cx = None
        if g.center_x_trajectory:
            cx = float(g.center_x_trajectory[-1])
        out.append(GapObservation(
            camera_id=g.camera_id or tracks.camera_id,
            local_track_id=g.track_id,
            local_frame=g.center_frame,
            local_time=g.center_time,
            confidence=g.confidence,
            start_frame=g.start_frame,
            end_frame=g.end_frame,
            fps=g.fps or tracks.fps,
            hit_count=g.hit_count,
            temporal_consistency_score=g.temporal_consistency_score,
            center_x=cx,
            global_time=g.center_time,      # delta applied later
        ))
    return out


def _with_offset(obs: Sequence[GapObservation], delta: float) -> List[GapObservation]:
    return [replace(o, global_time=o.local_time + delta) for o in obs]


# =============================================================================
# Phase 3b -- order-preserving alignment against the FIXED master sequence
# =============================================================================

def _match_cost(m: GapObservation, s: GapObservation,
                cfg: FusionConfig) -> float:
    """Cost of declaring m (master) and s (support) the same physical gap.

    Returns +inf when the pair is beyond the hard gate, which makes the match
    structurally unavailable to the DP.
    """
    tm = m.global_time if m.global_time is not None else m.local_time
    ts = s.global_time if s.global_time is not None else s.local_time
    d = abs(tm - ts)

    if cfg.match_cost_mode == "sigma":
        denom = math.sqrt(m.sigma(cfg) ** 2 + s.sigma(cfg) ** 2)
        if denom <= 0:
            return float("inf")
        d_norm = d / denom
        if d_norm > cfg.match_gate_sigmas:
            return float("inf")
        base = d_norm / cfg.match_gate_sigmas
    else:
        if d > cfg.match_gate_s:
            return float("inf")
        base = d / max(cfg.match_tolerance_s, 1e-9)

    mean_conf = 0.5 * (m.confidence + s.confidence)
    return base * (1.0 + cfg.confidence_weight * max(0.0, 1.0 - mean_conf))


def align_to_master(
    master_obs: Sequence[GapObservation],
    support_obs: Sequence[GapObservation],
    cfg: FusionConfig = DEFAULT_CONFIG,
) -> Tuple[float, List[Tuple[int, int]], List[int], List[int]]:
    """Order-preserving alignment of a support sequence to the MASTER sequence.

    The master sequence is a FIXED reference: this function never adds to it,
    removes from it or reorders it. It only decides, for each master index and
    each support index, whether they are the same physical gap.

    Train order is guaranteed *structurally*: the DP indices only ever advance,
    so if master i matches support j and master k matches support l with i < k,
    then j < l necessarily. Crossing matches are unrepresentable, not merely
    rejected after the fact.

    Returns
    -------
    (total_cost, matched_pairs, missing_master_idx, extra_support_idx)
        matched_pairs      : [(master_idx, support_idx), ...] strictly increasing
        missing_master_idx : master indices with no support observation
        extra_support_idx  : support indices corresponding to no master gap
    """
    n, m = len(master_obs), len(support_obs)
    INF = float("inf")

    if n == 0:
        return (m * cfg.extra_penalty, [], [], list(range(m)))
    if m == 0:
        return (n * cfg.missing_penalty, [], list(range(n)), [])

    # D[i][j] = best cost aligning master[:i] with support[:j]
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    P: List[List[Optional[str]]] = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + cfg.missing_penalty
        P[i][0] = OP_MISSING
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + cfg.extra_penalty
        P[0][j] = OP_EXTRA

    for i in range(1, n + 1):
        mi = master_obs[i - 1]
        for j in range(1, m + 1):
            best = D[i - 1][j] + cfg.missing_penalty
            op = OP_MISSING

            cand = D[i][j - 1] + cfg.extra_penalty
            if cand < best:
                best, op = cand, OP_EXTRA

            mc = _match_cost(mi, support_obs[j - 1], cfg)
            if mc < INF and D[i - 1][j - 1] < INF:
                cand = D[i - 1][j - 1] + mc
                if cand < best:
                    best, op = cand, OP_MATCH

            D[i][j], P[i][j] = best, op

    # Backtrack
    pairs: List[Tuple[int, int]] = []
    missing: List[int] = []
    extra: List[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = P[i][j]
        if op == OP_MATCH:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif op == OP_MISSING:
            missing.append(i - 1); i -= 1
        elif op == OP_EXTRA:
            extra.append(j - 1); j -= 1
        else:                      # pragma: no cover - defensive
            break

    pairs.reverse(); missing.reverse(); extra.reverse()
    return D[n][m], pairs, missing, extra


# =============================================================================
# Phase 3c -- camera offset estimation with anti-aliasing safeguards
# =============================================================================

def _median_spacing(obs: Sequence[GapObservation]) -> float:
    times = [o.local_time for o in obs]
    diffs = [b - a for a, b in zip(times, times[1:]) if b > a]
    return statistics.median(diffs) if diffs else 0.0


def interval_pattern_penalty(
    master_obs: Sequence[GapObservation],
    support_obs: Sequence[GapObservation],
    pairs: Sequence[Tuple[int, int]],
) -> float:
    """Disagreement between master and support LOCAL SPACING at matched pairs.

    This is the anti-aliasing discriminator. A pure constant offset with correct
    matching gives spacing agreement at every consecutive matched pair, so the
    penalty is ~0. An offset wrong by a whole number of wagon periods still
    yields tight timestamp matches, but it pairs gaps whose neighbourhood
    spacing differs -- which this term detects wherever the train's speed
    varies (it does, measurably, on this dataset).
    """
    if len(pairs) < 2:
        return 0.0
    pen = 0.0
    for (i0, j0), (i1, j1) in zip(pairs, pairs[1:]):
        dm = master_obs[i1].local_time - master_obs[i0].local_time
        ds = support_obs[j1].local_time - support_obs[j0].local_time
        if dm <= 0 or ds <= 0:
            continue
        pen += abs(dm - ds) / max(dm, ds)
    return pen


def _offset_score(
    master_obs: Sequence[GapObservation],
    support_obs: Sequence[GapObservation],
    delta: float,
    cfg: FusionConfig,
) -> Tuple[float, float, List[Tuple[int, int]], List[int], List[int]]:
    """Full objective at one candidate offset.

    score = DP alignment cost + pattern_weight * interval-pattern disagreement

    The DP term rewards many tight matches; the pattern term rejects
    whole-wagon-period aliases.
    """
    shifted = _with_offset(support_obs, delta)
    cost, pairs, missing, extra = align_to_master(master_obs, shifted, cfg)
    pattern = interval_pattern_penalty(master_obs, shifted, pairs)
    return cost + cfg.pattern_weight * pattern, pattern, pairs, missing, extra


def estimate_camera_offset(
    master_obs: Sequence[GapObservation],
    support_obs: Sequence[GapObservation],
    cfg: FusionConfig = DEFAULT_CONFIG,
    camera_id: str = "",
) -> CameraOffset:
    """Estimate delta for one support camera from the COMPLETE ordered sequences.

    Never estimated from a single detection. Returns UNRESOLVED rather than
    guessing when the evidence is not decisive -- an offset wrong by a whole
    number of wagon periods produces a deceptively good alignment, and a
    confidently wrong offset is worse than an admitted unknown.

    The count never depends on this result.
    """
    cam = camera_id or (support_obs[0].camera_id if support_obs else "?")
    off = CameraOffset(camera_id=cam)

    if not support_obs:
        off.status = OFFSET_UNRESOLVED
        off.reason = REASON_NO_GAPS
        return off
    if not master_obs:
        off.status = OFFSET_UNRESOLVED
        off.reason = "master produced no gap observations"
        return off

    # ---- coarse sweep over the whole search range ----
    step = max(1e-3, cfg.offset_coarse_step_s)
    n_steps = int(round(2 * cfg.offset_search_s / step)) + 1
    curve: List[Tuple[float, float, float, int, int, int]] = []
    for k in range(n_steps):
        delta = -cfg.offset_search_s + k * step
        score, pattern, pairs, missing, extra = _offset_score(
            master_obs, support_obs, delta, cfg)
        curve.append((delta, score, pattern, len(pairs), len(missing), len(extra)))

    curve.sort(key=lambda r: (r[1], abs(r[0])))
    best_delta, best_score, best_pattern, n_match, n_missing, n_extra = curve[0]

    # ---- alias separation: two offsets are distinct hypotheses only if they
    # differ by more than roughly one wagon period ----
    sep = cfg.alias_separation_s
    if sep <= 0:
        sep = _median_spacing(master_obs) or 1.0
    sep = max(sep * 0.75, 2 * cfg.match_gate_s)

    runner: Optional[Tuple[float, float]] = None
    for delta, score, *_ in curve[1:]:
        if abs(delta - best_delta) >= sep:
            runner = (delta, score)
            break

    # ---- fine refinement around the coarse winner ----
    fine_step = cfg.offset_fine_step_s
    if fine_step <= 0:
        fps = support_obs[0].fps or master_obs[0].fps or 25.0
        fine_step = 1.0 / fps
    lo = best_delta - cfg.offset_fine_window_s
    steps = int(round(2 * cfg.offset_fine_window_s / fine_step)) + 1
    for k in range(steps):
        delta = lo + k * fine_step
        score, pattern, pairs, missing, extra = _offset_score(
            master_obs, support_obs, delta, cfg)
        if score < best_score:
            best_delta, best_score, best_pattern = delta, score, pattern
            n_match, n_missing, n_extra = len(pairs), len(missing), len(extra)

    off.delta = best_delta
    off.score = best_score
    off.pattern_penalty = best_pattern
    off.n_match, off.n_missing, off.n_extra = n_match, n_missing, n_extra
    off.runner_up_delta = runner[0] if runner else None

    # ---- decisiveness tests (M2) ----
    if runner is None:
        off.margin_ratio = float("inf")
    else:
        # Normalize by the LARGER of the two scores (floored at 1.0) rather than
        # by the best score. Dividing by the best score is unstable: a perfect
        # synthetic alignment scores ~0, which would make any rival look
        # infinitely worse and mask a genuine ambiguity. With this form a clean
        # alignment yields a margin near 1.0, while the real-data case where two
        # offsets score 43.22 and 43.27 correctly yields ~0.001.
        denom = max(abs(best_score), abs(runner[1]), 1.0)
        off.margin_ratio = (runner[1] - best_score) / denom

    min_match = max(1, int(math.ceil(
        cfg.offset_min_match_fraction * min(len(master_obs), len(support_obs)))))

    if n_match < min_match:
        off.status = OFFSET_UNRESOLVED
        off.reason = (f"only {n_match} of a possible "
                      f"{min(len(master_obs), len(support_obs))} observations matched "
                      f"(need >= {min_match})")
    elif off.margin_ratio < cfg.offset_min_margin_ratio:
        off.status = OFFSET_UNRESOLVED
        off.reason = (f"ambiguous: best delta={best_delta:+.2f}s scores "
                      f"{best_score:.2f} but rival delta="
                      f"{off.runner_up_delta:+.2f}s scores {runner[1]:.2f} "
                      f"(margin {off.margin_ratio:.1%} < "
                      f"{cfg.offset_min_margin_ratio:.0%}); a whole-wagon-period "
                      f"alias cannot be ruled out")
    else:
        off.status = OFFSET_RESOLVED
        off.reason = ""

    return off


# =============================================================================
# Phase 3d -- the frozen global gap sequence, then evidence attachment
# =============================================================================

def build_global_gap_sequence(master_tracks: LocalCameraTracks) -> List[GlobalGap]:
    """Create the global gap sequence from RIGHT_UP ALONE.

    One GlobalGap per RIGHT_UP GapEvent, numbered 1..N in master temporal
    order. This is the only function in the codebase that may mint a
    global_gap_id, and it consults no support camera. After it returns, the
    sequence is treated as immutable.
    """
    master_obs = to_gap_observations(master_tracks)
    return [
        GlobalGap(global_gap_id=i, master_observation=o, master_camera=master_tracks.camera_id)
        for i, o in enumerate(master_obs, start=1)
    ]


def project_global_time_to_local(
    t_global: float, delta: float, fps: float, total_frames: int,
) -> Optional[int]:
    """Map a global instant onto a camera's local frame index, or None.

    Returns None when the instant lies outside the camera's real footage.
    NEVER clamps -- clamping to the last frame is what fabricates evidence.
    (`video_segmenter.map_global_wagon_to_local_frames` does clamp and is left
    untouched for the existing renderer; fusion uses this function instead.)
    """
    if fps <= 0 or total_frames <= 0:
        return None
    local_t = t_global - delta
    frame = int(round(local_t * fps))
    if frame < 0 or frame > total_frames - 1:
        return None
    return frame


def camera_covers(t_global: float, delta: float, fps: float,
                  total_frames: int) -> bool:
    return project_global_time_to_local(t_global, delta, fps, total_frames) is not None


def filter_observations_to_wagon_region(
    obs: Sequence[GapObservation],
    region: Any = None,
) -> Tuple[List[GapObservation], List[GapObservation]]:
    """Split a support camera's observations into (in-wagon-region, outside).

    Observations in the camera's engine or brake-van region are excluded from
    wagon synchronization: an engine-to-wagon transition is not a wagon
    boundary, so letting it anchor the alignment would corrupt the offset.
    Excluded observations are returned separately and reported, never deleted.

    `region` is a ``train_structure.LocalWagonRegion`` or None. When the region
    is unknown, nothing is excluded -- a missing classification must not silently
    drop evidence.
    """
    if region is None:
        return list(obs), []
    inside, outside = [], []
    for o in obs:
        (inside if region.contains_time(o.local_time) else outside).append(o)
    return inside, outside


def attach_support_evidence(
    global_gaps: List[GlobalGap],
    support_tracks: Sequence[LocalCameraTracks],
    cfg: FusionConfig = DEFAULT_CONFIG,
    verbose: bool = True,
    wagon_regions: Optional[Dict[str, Any]] = None,
) -> Dict[str, SupportAlignment]:
    """Align each support camera to the frozen sequence and hang evidence off it.

    Mutates `global_gaps` only by populating support_observations /
    missing_cameras / unavailable_cameras / time_residuals / weighted_time.
    It cannot add, remove or reorder a GlobalGap.

    Each camera is processed inside its own try/except: a support-side failure
    degrades that camera's evidence and can never affect the count.
    """
    if not global_gaps:
        return {}

    master_obs = [g.master_observation for g in global_gaps]
    alignments: Dict[str, SupportAlignment] = {}
    wagon_regions = wagon_regions or {}

    for st in support_tracks:
        cam = st.camera_id
        try:
            sup_obs = to_gap_observations(st)

            # Keep engine / brake-van observations out of wagon synchronization.
            sup_obs, non_wagon_obs = filter_observations_to_wagon_region(
                sup_obs, wagon_regions.get(cam))
            if non_wagon_obs and verbose:
                print(f"  [FUSE/{cam}] {len(non_wagon_obs)} observation(s) lie "
                      f"outside this camera's wagon region -> excluded from wagon "
                      f"alignment (kept as diagnostics)")

            if st.fps <= 0 or st.total_frames <= 0:
                off = CameraOffset(camera_id=cam, status=OFFSET_UNRESOLVED,
                                   reason=REASON_NO_METADATA)
                alignments[cam] = SupportAlignment(
                    camera_id=cam, offset=off, status="no-metadata",
                    non_wagon_observations=list(non_wagon_obs))
                for g in global_gaps:
                    g.unavailable_cameras[cam] = REASON_NO_METADATA
                continue

            off = estimate_camera_offset(master_obs, sup_obs, cfg, camera_id=cam)

            if not off.usable:
                # Safe degradation: no evidence, no effect on the count.
                alignments[cam] = SupportAlignment(
                    camera_id=cam, offset=off, status="offset-unresolved",
                    extra_observations=_with_offset(sup_obs, 0.0),
                    non_wagon_observations=list(non_wagon_obs),
                )
                for g in global_gaps:
                    g.unavailable_cameras[cam] = REASON_OFFSET_UNRESOLVED
                if verbose:
                    print(f"  [FUSE/{cam}] offset UNRESOLVED -> contributes no "
                          f"evidence ({off.reason})")
                continue

            shifted = _with_offset(sup_obs, off.delta)
            cost, pairs, missing_idx, extra_idx = align_to_master(
                master_obs, shifted, cfg)

            al = SupportAlignment(camera_id=cam, offset=off, total_cost=cost)
            for mi, sj in pairs:
                gap = global_gaps[mi]
                obs = shifted[sj]
                gap.support_observations[cam] = obs
                gap.time_residuals[cam] = obs.global_time - gap.master_time
                al.matches[gap.global_gap_id] = obs
            for mi in missing_idx:
                gap = global_gaps[mi]
                if camera_covers(gap.master_time, off.delta, st.fps, st.total_frames):
                    gap.missing_cameras.append(cam)
                    al.missing_global_gap_ids.append(gap.global_gap_id)
                else:
                    gap.unavailable_cameras[cam] = REASON_OUT_OF_RANGE
            al.extra_observations = [shifted[j] for j in extra_idx]
            al.non_wagon_observations = list(non_wagon_obs)
            alignments[cam] = al

            if verbose:
                print(f"  [FUSE/{cam}] delta={off.delta:+.2f}s ({off.status})  "
                      f"MATCH={len(al.matches)}  MISSING={len(al.missing_global_gap_ids)}  "
                      f"EXTRA={len(al.extra_observations)}  "
                      f"(EXTRA create no global gaps)")

        except Exception as e:                      # pragma: no cover - safety net
            off = CameraOffset(camera_id=cam, status=OFFSET_UNRESOLVED,
                               reason=f"{REASON_ALIGNMENT_ERROR}: {type(e).__name__}: {e}")
            alignments[cam] = SupportAlignment(camera_id=cam, offset=off,
                                               status="error")
            for g in global_gaps:
                g.unavailable_cameras.setdefault(cam, off.reason)
            if verbose:
                print(f"  [FUSE/{cam}] ERROR -> no evidence from this camera: {e}")

    # Confidence-weighted timestamp: DIAGNOSTIC ONLY. The master coordinate is
    # never replaced by it.
    for g in global_gaps:
        ws = [(g.master_observation.confidence, g.master_time)]
        for obs in g.support_observations.values():
            ws.append((obs.confidence, obs.global_time))
        denom = sum(w for w, _ in ws)
        g.weighted_time = (sum(w * t for w, t in ws) / denom) if denom > 0 else None

    return alignments


# =============================================================================
# Phase 3e -- physical interval diagnostics (REPORT ONLY, never changes count)
# =============================================================================

def diagnose_intervals(
    global_gaps: Sequence[GlobalGap],
    cfg: FusionConfig = DEFAULT_CONFIG,
) -> List[Dict[str, Any]]:
    """Flag implausible spacing between consecutive global gaps.

    REPORT ONLY. RIGHT_UP's sequence is final: nothing here modifies, inserts or
    deletes a gap. The output exists so a detection-quality problem is visible
    as a detection-quality problem, rather than being silently absorbed into the
    count.

    The train's speed drifts measurably, so plausibility is judged against a
    LOCAL median spacing rather than any global constant.
    """
    out: List[Dict[str, Any]] = []
    if len(global_gaps) < 2:
        return out

    times = [g.master_time for g in global_gaps]
    intervals = [b - a for a, b in zip(times, times[1:])]
    w = max(1, cfg.local_base_window)

    for k, length in enumerate(intervals):
        lo, hi = max(0, k - w), min(len(intervals), k + w + 1)
        window = intervals[lo:hi]
        base = statistics.median(window) if window else 0.0
        ratio = (length / base) if base > 0 else float("inf")

        flag = None
        implied = 0
        if ratio <= cfg.short_interval_ratio:
            flag = "SUSPICIOUSLY_SHORT"
        elif ratio >= cfg.long_interval_ratio:
            flag = "POSSIBLE_MISSING_GAP"
            implied = max(1, int(round(ratio)) - 1)
        if flag is None:
            continue

        rec = {
            "flag": flag,
            "between_global_gap_ids": [global_gaps[k].global_gap_id,
                                       global_gaps[k + 1].global_gap_id],
            "start_time": round(times[k], 4),
            "end_time": round(times[k + 1], 4),
            "interval_sec": round(length, 4),
            "local_base_sec": round(base, 4),
            "ratio": round(ratio, 4),
            "implied_missing_gaps": implied,
            "note": ("diagnostic only -- the RIGHT_UP gap sequence is final and "
                     "was NOT modified"),
        }
        out.append(rec)
        global_gaps[k].flags.append(f"{flag}_AFTER")
        global_gaps[k + 1].flags.append(f"{flag}_BEFORE")
    return out


# =============================================================================
# Phase 3f -- invariant assertions
# =============================================================================

def assert_invariants(
    *,
    global_gaps: Sequence[GlobalGap],
    master_tracks: LocalCameraTracks,
    wagons: Sequence[GlobalWagon],
    alignments: Dict[str, SupportAlignment],
    support_tracks: Sequence[LocalCameraTracks],
    strict: bool = True,
    wagon_window: Any = None,
) -> Dict[str, Any]:
    """Verify the fixed-master invariant. Fails loudly by default.

    Returns a machine-readable record of every check for the JSON output.
    """
    problems: List[str] = []
    n_master = len(master_tracks.gaps)

    # 1. the hard invariant
    if len(global_gaps) != n_master:
        problems.append(f"global_gap_count={len(global_gaps)} != "
                        f"right_up_final_gap_count={n_master}")

    # 2/3/6/12. every global gap is master-sourced
    for g in global_gaps:
        if g.master_observation is None:
            problems.append(f"global gap {g.global_gap_id} has no master observation")
        elif g.master_observation.camera_id != master_tracks.camera_id:
            problems.append(f"global gap {g.global_gap_id} sourced from "
                            f"'{g.master_observation.camera_id}', not "
                            f"'{master_tracks.camera_id}'")
    support_ids = {st.camera_id for st in support_tracks}
    for g in global_gaps:
        if g.master_camera in support_ids:
            problems.append(f"global gap {g.global_gap_id} claims support camera "
                            f"'{g.master_camera}' as its master")

    # 4. ids are exactly 1..N and strictly increasing
    ids = [g.global_gap_id for g in global_gaps]
    if ids != list(range(1, len(global_gaps) + 1)):
        problems.append(f"global_gap_ids are not 1..N in order: {ids[:10]}...")

    # 5. train order preserved on the master clock
    frames = [g.master_frame for g in global_gaps]
    if any(b < a for a, b in zip(frames, frames[1:])):
        problems.append("global gaps are not ordered by master frame")

    # 7/8. wagon count follows from the gap count, then the wagon window
    expected = len(global_gaps) + 1
    collapsed = expected - len(wagons)
    if len(wagons) > expected:
        problems.append(f"total_wagons={len(wagons)} exceeds global_gaps+1={expected}")

    # 13/14. ONLY WAGONS ARE COUNTED -- no engine or brake van may hold a GW id.
    # Checked only under wagon-only counting; `--no-wagon-only` deliberately
    # reproduces the older "every segment is a wagon" behaviour for A/B runs.
    n_bad_class = 0
    if wagon_window is not None:
        for w in wagons:
            if w.classification in (SegmentClass.ENGINE, SegmentClass.BRAKE_VAN):
                n_bad_class += 1
                problems.append(f"{w.global_id} is classified {w.classification} "
                                f"but holds a wagon id; engines and brake vans "
                                f"must never receive one")

    if wagon_window is not None:
        if len(wagons) != wagon_window.master_wagon_count:
            problems.append(f"total_wagons={len(wagons)} != wagon window count "
                            f"{wagon_window.master_wagon_count}")
        # Non-wagon objects are preserved, and are outside the counted set.
        excluded = (len(wagon_window.leading_non_wagon_objects)
                    + len(wagon_window.trailing_non_wagon_objects)
                    + len(wagon_window.interior_non_wagon_objects))
        if len(wagons) + excluded != wagon_window.total_segments:
            problems.append(f"wagons({len(wagons)}) + non-wagon({excluded}) != "
                            f"segments({wagon_window.total_segments}): a segment "
                            f"was lost or double-counted")
        collapsed = expected - wagon_window.total_segments
    if collapsed < 0:
        problems.append(f"negative collapsed-boundary count ({collapsed})")

    # 9/10. each support observation is exactly one of MATCH / MISSING / EXTRA
    for st in support_tracks:
        al = alignments.get(st.camera_id)
        if al is None:
            continue
        n_obs = len(st.gaps)
        if al.offset.usable:
            accounted = len(al.matches) + len(al.extra_observations)
            if accounted != n_obs:
                problems.append(
                    f"{st.camera_id}: MATCH({len(al.matches)}) + "
                    f"EXTRA({len(al.extra_observations)}) = {accounted} != "
                    f"{n_obs} observations")
        if len(al.matches) > len(global_gaps):
            problems.append(f"{st.camera_id}: more matches than global gaps")
        matched_ids = sorted(al.matches.keys())
        if len(set(matched_ids)) != len(matched_ids):
            problems.append(f"{st.camera_id}: a global gap matched twice")

    # 11. GW ids ordered and contiguous
    gw = [w.global_id for w in wagons]
    if gw != [f"GW_{i}" for i in range(1, len(wagons) + 1)]:
        problems.append("GW ids are not GW_1..GW_N in order")

    record = {
        "right_up_final_gap_count": n_master,
        "global_gap_count": len(global_gaps),
        "total_wagons": len(wagons),
        "master_wagon_count": (wagon_window.master_wagon_count
                               if wagon_window is not None else len(wagons)),
        "segments_before_wagon_window": expected,
        "collapsed_boundaries": collapsed,
        "non_wagon_excluded": (
            len(wagon_window.leading_non_wagon_objects)
            + len(wagon_window.trailing_non_wagon_objects)
            + len(wagon_window.interior_non_wagon_objects)
            if wagon_window is not None else 0),
        "engine_or_brakevan_with_wagon_id": n_bad_class,
        "invariant_holds": not problems,
        "violations": problems,
        "checks_run": 14,
    }

    if problems:
        msg = ("FIXED-MASTER INVARIANT VIOLATED:\n  - " + "\n  - ".join(problems))
        if strict:
            raise FusionInvariantError(msg)
        print("WARNING: " + msg)
    return record


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def assemble_global_train_state_master_fixed(
    *,
    master_tracks: LocalCameraTracks,
    support_tracks: Sequence[LocalCameraTracks],
    initial_classifications: Sequence[_MasterClassification],
    config: Optional[FusionConfig] = None,
    verbose: bool = True,
    wagon_regions: Optional[Dict[str, Any]] = None,
    wagon_only: bool = True,
) -> GlobalTrainState:
    """Build the GlobalTrainState under the fixed-master invariant.

        global gaps  == RIGHT_UP validated gaps    (always)
        total_wagons == the WAGON units of the master's wagon window

    ENGINE and BRAKE_VAN are preserved as metadata but never receive a GW id and
    never extend the wagon timeline. Support cameras contribute association +
    evidence + diagnostics only.

    Parameters
    ----------
    wagon_only :
        True (default) restricts the count to the master's wagon window, so
        engines and brake vans are excluded. False keeps the previous
        "every segment is a wagon" behaviour for A/B comparison.
    wagon_regions :
        Optional ``{camera_id: LocalWagonRegion}``, used to keep support
        observations from engine / brake-van regions out of wagon alignment.
    """
    cfg = config or DEFAULT_CONFIG

    # ---- 1. the frozen global sequence, from RIGHT_UP alone ----
    global_gaps = build_global_gap_sequence(master_tracks)
    if verbose:
        print(f"[FUSE] master({master_tracks.camera_id}) is authoritative: "
              f"{len(global_gaps)} validated gap(s) -> {len(global_gaps)} global gap(s)")

    # ---- 2. support alignment (evidence only; cannot touch the sequence) ----
    alignments = attach_support_evidence(global_gaps, support_tracks, cfg, verbose,
                                         wagon_regions=wagon_regions)

    # ---- 3. segments: build_global_wagons REUSED UNCHANGED, fed master.gaps ----
    # No synthetic gap can reach it: the input is literally the master's own
    # validated GapEvent list.
    all_segments = build_global_wagons(
        list(master_tracks.gaps),
        master_total_frames=master_tracks.total_frames,
        master_fps=master_tracks.fps,
        initial_classifications=list(initial_classifications),
        support_camera_ids=[st.camera_id for st in support_tracks],
    )

    # ---- 3b. WAGON-ONLY selection: engines and brake vans get no GW id ----
    wagon_window = None
    if wagon_only:
        from train_structure import get_master_wagon_window
        wagon_window = get_master_wagon_window(all_segments, verbose=verbose)
        wagons = wagon_window.wagon_units
    else:
        wagons = all_segments

    # ---- 4. real supporting_cameras, replacing the old static all-four list ----
    _assign_real_supporting_cameras(wagons, global_gaps, master_tracks.camera_id)

    # ---- 5. diagnostics (report only) ----
    interval_diags = diagnose_intervals(global_gaps, cfg)
    if verbose and interval_diags:
        n_short = sum(1 for d in interval_diags if d["flag"] == "SUSPICIOUSLY_SHORT")
        n_long = sum(1 for d in interval_diags if d["flag"] == "POSSIBLE_MISSING_GAP")
        print(f"[FUSE] interval diagnostics: {n_short} suspiciously short, "
              f"{n_long} possibly-missing (REPORT ONLY -- master sequence unchanged)")

    # ---- 6. invariants ----
    checks = assert_invariants(
        global_gaps=global_gaps, master_tracks=master_tracks, wagons=wagons,
        alignments=alignments, support_tracks=support_tracks,
        strict=cfg.strict_invariants, wagon_window=wagon_window,
    )

    # ---- 7. assemble state ----
    per_local_counts = {master_tracks.camera_id: master_tracks.local_wagon_count}
    per_gap_counts = {master_tracks.camera_id: len(master_tracks.gaps)}
    per_status = {master_tracks.camera_id: "master/authoritative"}
    for st in support_tracks:
        per_local_counts[st.camera_id] = st.local_wagon_count
        per_gap_counts[st.camera_id] = len(st.gaps)
        al = alignments.get(st.camera_id)
        if al is None:
            per_status[st.camera_id] = "not aligned"
        elif not al.offset.usable:
            per_status[st.camera_id] = f"support/{al.status}"
        else:
            per_status[st.camera_id] = (
                f"support/aligned delta={al.offset.delta:+.2f}s "
                f"M{len(al.matches)}/X{len(al.extra_observations)}")

    state = GlobalTrainState(
        total_wagons=len(wagons),
        wagons=wagons,
        master_camera=master_tracks.camera_id,
        master_fps=master_tracks.fps,
        master_total_frames=master_tracks.total_frames,
        per_camera_local_counts=per_local_counts,
        per_camera_gap_counts=per_gap_counts,
        per_camera_status=per_status,
        corrections_applied=[],          # no insertion mechanism exists
        fallback_used=False,
        fallback_reason="",
    )

    # New, additive blocks (all default-empty, so old consumers keep working).
    state.fusion_mode = "master-fixed"
    state.global_gaps = [g.to_dict() for g in global_gaps]
    state.camera_offsets = {
        master_tracks.camera_id: CameraOffset(
            camera_id=master_tracks.camera_id, delta=0.0,
            status=OFFSET_REFERENCE, reason="master camera is the reference clock",
        ).to_dict(),
        **{cam: al.offset.to_dict() for cam, al in alignments.items()},
    }
    state.support_alignment_summary = {cam: al.to_dict() for cam, al in alignments.items()}
    state.extra_support_observations = {
        cam: [o.to_dict() for o in al.extra_observations]
        for cam, al in alignments.items() if al.extra_observations
    }
    state.interval_diagnostics = interval_diags
    state.invariant_checks = checks

    if wagon_window is not None:
        state.wagon_window = wagon_window.summary()
        state.master_wagon_count = wagon_window.master_wagon_count
    else:
        state.master_wagon_count = len(wagons)
    if wagon_regions:
        state.support_wagon_regions = {
            cam: (reg.to_dict() if hasattr(reg, "to_dict") else {})
            for cam, reg in wagon_regions.items()}

    if verbose:
        n_extra = sum(len(v) for v in state.extra_support_observations.values())
        print(f"[FUSE] {n_extra} EXTRA support observation(s) recorded as "
              f"diagnostics; none became a global gap")
        print(f"[FUSE] INVARIANT: right_up_validated_gaps="
              f"{checks['right_up_final_gap_count']} "
              f"== global_gaps={checks['global_gap_count']}")
        if wagon_window is not None:
            print(f"[FUSE] WAGON-ONLY: {checks['non_wagon_excluded']} non-wagon "
                  f"object(s) excluded (no GW id)  ->  "
                  f"total_wagons={checks['total_wagons']}")

    return state


def _assign_real_supporting_cameras(
    wagons: Sequence[GlobalWagon],
    global_gaps: Sequence[GlobalGap],
    master_camera: str,
) -> None:
    """Replace the old static all-four `supporting_cameras` with the truth.

    A wagon lists a support camera only when that camera actually has a matched
    observation on one of the two gaps bounding the wagon. The master is always
    listed: it defines the wagon.
    """
    by_frame: Dict[int, GlobalGap] = {}
    by_track: Dict[int, GlobalGap] = {}
    for g in global_gaps:
        by_frame[g.master_frame] = g
        by_track[g.master_observation.local_track_id] = g

    for w in wagons:
        cams: List[str] = [master_camera]
        for boundary in (w.leading_gap, w.trailing_gap):
            if not boundary:
                continue
            tid = boundary.get("track_id")
            gap = by_track.get(tid) if isinstance(tid, int) else None
            if gap is None:
                continue
            for cam in gap.supporting_camera_ids:
                if cam not in cams:
                    cams.append(cam)
        w.supporting_cameras = [c for c in ALL_CAMERAS if c in cams] or [master_camera]
