"""Temporal tracking and aggregation of inspection detections.

WHY NOT COUNT DETECTIONS PER FRAME
----------------------------------
Measured on real footage, per-frame detection counts are dominated by noise: at
conf >= 0.01 the two shipped models emit a median of 27 and 19.5 detections per
frame, with a median confidence of ~0.014. Counting detections would report
thousands of findings on a train that has a handful.

The signal is in TIME, not in any single frame. A genuine door was observed on 28
consecutive frames, its centre advancing ~28 px/frame -- the train's own speed --
with confidence climbing 0.46 -> 0.95 and tapering at the edges. A genuine
`Inner_wall_damage` ran 51+ frames, 0.62 -> 0.90. Every false positive was a 1-5
frame run at 0.21-0.55, and several were PINNED near the frame edge.

So a finding is a TRACK that persisted, moved with the train, and reached real
confidence -- the same three properties that made gap counting reliable.

THIS MODULE DELIBERATELY DOES NOT TOUCH tracker_engine.py
---------------------------------------------------------
The gap tracker is protected, and its Kalman/association parameters are tuned for
gap geometry. Inspection needs a simpler, independent association over boxes, so
it gets its own -- rather than perturbing the tracker the wagon count depends on.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .state import InspectionConfig


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------

@dataclass
class DetectionObservation:
    """One raw detection in one frame. The atom everything else is built from."""
    frame: int
    time_local: float
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]      # x1, y1, x2, y2

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        return {"frame": self.frame, "time_local": round(self.time_local, 4),
                "class_id": self.class_id, "class_name": self.class_name,
                "confidence": round(self.confidence, 4),
                "bbox": [round(v, 2) for v in self.bbox]}


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def suppress_duplicates_in_frame(
    observations: Sequence[DetectionObservation],
    iou_threshold: float = 0.60,
) -> List[DetectionObservation]:
    """Collapse near-identical boxes within ONE frame, keeping the strongest.

    Necessary and measured: the damage model returned three `Floor_damage` boxes
    at cx 943, 943 and 944 in a single frame. Without this each would seed its own
    track and the same physical spot would be reported repeatedly.

    Suppression is within a class only -- two DIFFERENT classes overlapping is
    real and informative (three door classes were observed firing on one door),
    and is resolved later by confidence-weighted voting over the whole track,
    where there is far more evidence than in any one frame.
    """
    kept: List[DetectionObservation] = []
    # Deterministic: strongest first, then by geometry to break ties.
    for obs in sorted(observations,
                      key=lambda o: (-o.confidence, o.class_id, o.bbox)):
        if any(o.class_id == obs.class_id and iou(o.bbox, obs.bbox) >= iou_threshold
               for o in kept):
            continue
        kept.append(obs)
    return sorted(kept, key=lambda o: (o.class_id, o.bbox))


# ---------------------------------------------------------------------------
# tracks
# ---------------------------------------------------------------------------

@dataclass
class InspectionTrack:
    """One physical object observed across several frames."""
    track_id: int
    camera_id: str
    role: str
    observations: List[DetectionObservation] = field(default_factory=list)

    # ---- extent ----
    @property
    def start_frame(self) -> int:
        return min(o.frame for o in self.observations)

    @property
    def end_frame(self) -> int:
        return max(o.frame for o in self.observations)

    @property
    def start_time(self) -> float:
        return min(o.time_local for o in self.observations)

    @property
    def end_time(self) -> float:
        return max(o.time_local for o in self.observations)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    # ---- confidence ----
    @property
    def peak_confidence(self) -> float:
        return max(o.confidence for o in self.observations)

    @property
    def mean_confidence(self) -> float:
        return statistics.fmean(o.confidence for o in self.observations)

    @property
    def peak_observation(self) -> DetectionObservation:
        """Strongest sighting; ties broken by frame so selection is deterministic."""
        return min(self.observations, key=lambda o: (-o.confidence, o.frame))

    # ---- motion ----
    @property
    def displacement_px(self) -> float:
        xs = [o.center_x for o in sorted(self.observations, key=lambda o: o.frame)]
        return xs[-1] - xs[0]

    @property
    def abs_displacement_px(self) -> float:
        return abs(self.displacement_px)

    def dominant_class(self) -> Tuple[int, str, float]:
        """The track's class, by confidence-weighted vote across every sighting.

        Weighted rather than most-frequent because several classes routinely fire
        on one object -- measured, one door produced `partially_closed` 0.46,
        `open_door` 0.33 and `closed_door` 0.24 in the same frame. Summing
        confidence over the whole track lets the sustained, strong label win
        instead of whichever label happened to appear most often at the noisy
        edges of the pass.
        """
        weights: Dict[Tuple[int, str], float] = {}
        for o in self.observations:
            weights[(o.class_id, o.class_name)] = (
                weights.get((o.class_id, o.class_name), 0.0) + o.confidence)
        # Sort keys first so equal weights resolve identically on every run.
        best = max(sorted(weights), key=lambda k: weights[k])
        total = sum(weights.values())
        return best[0], best[1], (weights[best] / total if total else 0.0)

    def to_dict(self) -> Dict[str, Any]:
        cid, cname, share = self.dominant_class()
        return {
            "track_id": self.track_id, "camera_id": self.camera_id,
            "role": self.role,
            "start_frame": self.start_frame, "end_frame": self.end_frame,
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "duration": round(self.duration, 4),
            "n_observations": self.n_observations,
            "class_id": cid, "class_name": cname,
            "class_vote_share": round(share, 4),
            "peak_confidence": round(self.peak_confidence, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "displacement_px": round(self.displacement_px, 2),
        }


# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------

def _vertical_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    """Fraction of the shorter box's height that the two boxes share."""
    top, bottom = max(a[1], b[1]), min(a[3], b[3])
    inter = max(0.0, bottom - top)
    shorter = min(max(0.0, a[3] - a[1]), max(0.0, b[3] - b[1]))
    return (inter / shorter) if shorter > 0 else 0.0


def association_score(
    last: DetectionObservation, cand: DetectionObservation,
    cfg: InspectionConfig, frame_width: int,
) -> float:
    """How well `cand` continues `last`. 0 means no match.

    IoU is preferred, but a fast-moving object legitimately produces barely
    overlapping boxes, so a positional fallback applies: same class (checked by
    the caller), centre within `max_center_jump_frac` of frame width, and the two
    boxes sharing an image band. Scored below any real IoU match so genuine
    overlap always wins when both apply.
    """
    overlap = iou(last.bbox, cand.bbox)
    if overlap >= cfg.iou_match_threshold:
        return 1.0 + overlap
    if frame_width <= 0:
        return 0.0
    # The allowance is PER FRAME INTERVAL, so it scales with the real gap between
    # the two sightings. This is what makes `frame_stride` a pure cost/quality
    # dial: sampling every 4th frame legitimately quadruples how far an object
    # moves between observations, and a fixed allowance would silently shatter
    # exactly the long genuine tracks that higher strides are meant to keep.
    interval = max(1, cand.frame - last.frame)
    budget = cfg.max_center_jump_frac * frame_width * interval
    jump = abs(cand.center_x - last.center_x)
    if jump > budget:
        return 0.0
    if _vertical_overlap(last.bbox, cand.bbox) < 0.30:
        return 0.0
    # Nearer is better, but always ranked under a true IoU match.
    return 1.0 - (jump / budget)


def track_detections(
    per_frame: Iterable[Tuple[int, float, Sequence[DetectionObservation]]],
    camera_id: str,
    role: str,
    cfg: InspectionConfig,
    fps: float,
    frame_width: int,
) -> List[InspectionTrack]:
    """Group per-frame detections into tracks.

    `per_frame` yields (frame_index, time_local, observations) in ASCENDING frame
    order. Association is greedy by IoU within a class, which is sufficient here:
    inspection targets are large, sparse and move steadily, unlike the crowded
    small-object case that would need a motion model.

    Pure function of its arguments -- no module state -- so trains processed one
    after another in the same process cannot influence each other.
    """
    open_tracks: List[InspectionTrack] = []
    closed: List[InspectionTrack] = []
    next_id = 1
    max_miss_frames = max(1, int(round(cfg.max_track_miss_seconds * fps)))

    for frame, t_local, raw in per_frame:
        obs = [o for o in raw if o.confidence >= cfg.min_detection_confidence]
        obs = suppress_duplicates_in_frame(obs)
        if cfg.max_detections_per_frame and len(obs) > cfg.max_detections_per_frame:
            # Keep the strongest; a pathological frame must not balloon memory.
            obs = sorted(obs, key=lambda o: (-o.confidence, o.class_id))[
                :cfg.max_detections_per_frame]

        # Retire tracks that have gone unseen too long. Done first so a stale
        # track cannot claim a detection belonging to a new object.
        still_open: List[InspectionTrack] = []
        for tr in open_tracks:
            if frame - tr.end_frame > max_miss_frames:
                closed.append(tr)
            else:
                still_open.append(tr)
        open_tracks = still_open

        # Greedy assignment, strongest detection first, each track claimed once.
        claimed: set = set()
        for o in sorted(obs, key=lambda x: (-x.confidence, x.class_id, x.bbox)):
            best_tr, best_score = None, 0.0
            for tr in open_tracks:
                if id(tr) in claimed:
                    continue
                last = max(tr.observations, key=lambda x: x.frame)
                if last.class_id != o.class_id:
                    continue          # a class change is a different object here
                score = association_score(last, o, cfg, frame_width)
                if score > best_score:
                    best_tr, best_score = tr, score
            if best_tr is not None and best_score > 0.0:
                best_tr.observations.append(o)
                claimed.add(id(best_tr))
            else:
                tr = InspectionTrack(track_id=next_id, camera_id=camera_id,
                                     role=role, observations=[o])
                next_id += 1
                open_tracks.append(tr)
                claimed.add(id(tr))

    closed.extend(open_tracks)
    # Stable order regardless of how tracks happened to close.
    closed.sort(key=lambda tr: (tr.start_frame, tr.track_id))
    return closed


def classify_track(
    track: InspectionTrack, cfg: InspectionConfig, frame_width: int,
) -> Tuple[bool, str]:
    """Is this track a confirmed finding? Returns (confirmed, reason_if_not).

    Every test below was chosen because the measurements showed it separating
    signal from noise on real footage; none is a guess.
    """
    if track.n_observations < cfg.min_track_observations:
        return False, (f"only {track.n_observations} sighting(s), fewer than "
                       f"{cfg.min_track_observations}: a single frame or a "
                       f"flicker is not a finding")
    if track.duration < cfg.min_track_seconds:
        return False, (f"persisted {track.duration:.2f}s, under "
                       f"{cfg.min_track_seconds}s -- every measured false "
                       f"positive lasted this briefly")
    if track.peak_confidence < cfg.min_peak_confidence:
        return False, (f"peak confidence {track.peak_confidence:.2f} below "
                       f"{cfg.min_peak_confidence}; measured false-positive runs "
                       f"peaked at 0.55, genuine findings at 0.88-0.95")
    if track.mean_confidence < cfg.min_mean_confidence:
        return False, (f"mean confidence {track.mean_confidence:.2f} below "
                       f"{cfg.min_mean_confidence}: noise throughout except one "
                       f"strong frame")
    if cfg.require_motion and frame_width > 0:
        floor = cfg.min_motion_frac * frame_width
        if track.abs_displacement_px < floor:
            return False, (f"moved {track.abs_displacement_px:.1f}px, under the "
                           f"{floor:.1f}px floor: pinned to one place while the "
                           f"train moves, which is the measured signature of a "
                           f"static artefact")
    return True, ""
