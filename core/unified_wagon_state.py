"""One wagon's fused inspection verdict, in the shape old_code's report expects.

`old_code/reporting/combined_train_report.py` consumes
`Dict[global_id, UnifiedWagonState]` plus `summarize_wagons(...)`. Neither survived
with old_code, so both are reconstructed here from exactly how the report uses
them -- the field set and the KPI key set are therefore RECOVERED from real call
sites, not invented:

    fields  u.global_id  u.classification  u.wagon_identifier  u.left_door
            u.right_door u.load_status     u.top_damage        u.side_damage
            u.confidence u.has_open_door   u.has_damage        u.anomalies
            u.to_dict()

    KPIs    total_wagons engine_count brake_van_count wagon_count ocr_captured
            left_doors_open right_doors_open loaded empty top_damaged

THIS IS A VIEW, NOT A SECOND WAGON STRUCTURE. A `UnifiedWagonState` carries the
`global_id` that fusion already minted and no geometry of its own -- no boundaries,
no frame ranges, no index. It cannot contradict or renumber the roster because it
does not hold the information that would let it.

STATES STAY DISTINCT. Every feature defaults to `NO_DATA`, never to a negative
finding. "not visible" and "unresolved" are preserved as their own values, so the
report can never present an uninspected wagon as a clean one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

from core import constants as C

__all__ = ["UnifiedWagonState", "summarize_wagons", "ASSOC_RESOLVED",
           "ASSOC_UNRESOLVED", "ASSOC_AMBIGUOUS"]

ASSOC_RESOLVED = "RESOLVED"
ASSOC_UNRESOLVED = "UNRESOLVED"
ASSOC_AMBIGUOUS = "AMBIGUOUS"


@dataclass
class UnifiedWagonState:
    """Fused per-wagon inspection result, keyed by an EXISTING global wagon id."""

    global_id: str
    classification: str = C.CLASS_WAGON

    # ---- features (all default to NO_DATA, never to a negative verdict) ----
    left_door: str = C.NO_DATA
    left_door_confidence: float = 0.0
    right_door: str = C.NO_DATA
    right_door_confidence: float = 0.0

    load_status: str = C.NO_DATA
    load_confidence: float = 0.0

    top_damage: str = C.NO_DATA
    top_damage_confidence: float = 0.0
    side_damage: str = C.NO_DATA
    side_damage_confidence: float = 0.0

    wagon_identifier: str = ""
    """OCR'd wagon number. Empty means not read -- it NEVER affects the GW id."""
    wagon_identifier_confidence: float = 0.0

    # ---- provenance ----
    association_status: str = ASSOC_RESOLVED
    supporting_cameras: List[str] = field(default_factory=list)
    camera_status: Dict[str, Dict[str, str]] = field(default_factory=dict)
    feature_status: Dict[str, str] = field(default_factory=dict)
    """feature -> OK / NO_FRAMES / FAILED / NO_DATA, straight from each processor."""
    evidence: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    tracks: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # ---- derived flags the report highlights rows on ----
    @property
    def has_open_door(self) -> bool:
        """OPEN or PARTIAL on either side. NO_DATA is NOT an anomaly."""
        return any(v in (C.DOOR_OPEN, C.DOOR_PARTIAL)
                   for v in (self.left_door, self.right_door))

    @property
    def has_damage(self) -> bool:
        return (self.top_damage == C.DAMAGE_PRESENT
                or self.side_damage == C.DAMAGE_PRESENT
                or C.DOOR_DAMAGED in (self.left_door, self.right_door))

    @property
    def confidence(self) -> float:
        """Representative confidence: the strongest signal that drove a verdict.

        The report prints one number per wagon, and the value that matters is the
        one behind an anomaly, so anomalous features dominate. With no findings it
        falls back to the best available feature confidence.
        """
        anomaly_confs = []
        if self.left_door in (C.DOOR_OPEN, C.DOOR_PARTIAL, C.DOOR_DAMAGED):
            anomaly_confs.append(self.left_door_confidence)
        if self.right_door in (C.DOOR_OPEN, C.DOOR_PARTIAL, C.DOOR_DAMAGED):
            anomaly_confs.append(self.right_door_confidence)
        if self.top_damage == C.DAMAGE_PRESENT:
            anomaly_confs.append(self.top_damage_confidence)
        if self.side_damage == C.DAMAGE_PRESENT:
            anomaly_confs.append(self.side_damage_confidence)
        if anomaly_confs:
            return round(max(anomaly_confs), 4)
        others = [self.left_door_confidence, self.right_door_confidence,
                  self.load_confidence]
        return round(max(others) if others else 0.0, 4)

    @property
    def anomalies(self) -> List[str]:
        """Human-readable anomaly list, used for the KPI count and flagged pages."""
        out: List[str] = []
        if self.left_door in (C.DOOR_OPEN, C.DOOR_PARTIAL):
            out.append(f"LEFT_DOOR_{self.left_door}")
        if self.right_door in (C.DOOR_OPEN, C.DOOR_PARTIAL):
            out.append(f"RIGHT_DOOR_{self.right_door}")
        if self.left_door == C.DOOR_DAMAGED:
            out.append("LEFT_DOOR_DAMAGED")
        if self.right_door == C.DOOR_DAMAGED:
            out.append("RIGHT_DOOR_DAMAGED")
        if self.top_damage == C.DAMAGE_PRESENT:
            out.append("TOP_DAMAGE")
        if self.side_damage == C.DAMAGE_PRESENT:
            out.append("SIDE_DAMAGE")
        if self.classification in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN):
            # Not a defect -- recorded so the report can tint the row and so an
            # INTERIOR engine/brake-van label stays visible as an anomaly.
            pass
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_id": self.global_id,
            "classification": self.classification,
            "wagon_identifier": self.wagon_identifier,
            "wagon_identifier_confidence": round(
                self.wagon_identifier_confidence, 4),
            "left_door": self.left_door,
            "left_door_confidence": round(self.left_door_confidence, 4),
            "right_door": self.right_door,
            "right_door_confidence": round(self.right_door_confidence, 4),
            "load_status": self.load_status,
            "load_confidence": round(self.load_confidence, 4),
            "top_damage": self.top_damage,
            "top_damage_confidence": round(self.top_damage_confidence, 4),
            "side_damage": self.side_damage,
            "side_damage_confidence": round(self.side_damage_confidence, 4),
            "confidence": self.confidence,
            "has_open_door": self.has_open_door,
            "has_damage": self.has_damage,
            "anomalies": list(self.anomalies),
            "association_status": self.association_status,
            "supporting_cameras": list(self.supporting_cameras),
            "camera_status": {k: dict(v) for k, v in sorted(
                self.camera_status.items())},
            "feature_status": dict(self.feature_status),
            "evidence": {k: list(v) for k, v in sorted(self.evidence.items())},
            "tracks": {k: list(v) for k, v in sorted(self.tracks.items())},
        }


def summarize_wagons(wagons: Sequence[UnifiedWagonState]) -> Dict[str, Any]:
    """Train-level KPIs. Key names are those the old report reads.

    Counts are over CONFIRMED verdicts only: a wagon whose load is NO_DATA counts
    towards neither `loaded` nor `empty`, so the two never silently sum to the
    train size and imply a complete inspection.
    """
    ws = list(wagons)
    engines = sum(1 for u in ws if u.classification == C.CLASS_ENGINE)
    brake_vans = sum(1 for u in ws if u.classification == C.CLASS_BRAKE_VAN)
    return {
        "total_wagons": len(ws),
        "engine_count": engines,
        "brake_van_count": brake_vans,
        "wagon_count": len(ws) - engines - brake_vans,
        "ocr_captured": sum(1 for u in ws if u.wagon_identifier),
        "left_doors_open": sum(1 for u in ws if u.left_door == C.DOOR_OPEN),
        "right_doors_open": sum(1 for u in ws if u.right_door == C.DOOR_OPEN),
        "left_doors_partial": sum(1 for u in ws if u.left_door == C.DOOR_PARTIAL),
        "right_doors_partial": sum(1 for u in ws if u.right_door == C.DOOR_PARTIAL),
        "doors_damaged": sum(1 for u in ws
                             if C.DOOR_DAMAGED in (u.left_door, u.right_door)),
        "loaded": sum(1 for u in ws if u.load_status == C.LOAD_LOADED),
        "empty": sum(1 for u in ws if u.load_status == C.LOAD_EMPTY),
        "load_no_data": sum(1 for u in ws if u.load_status == C.NO_DATA),
        "top_damaged": sum(1 for u in ws if u.top_damage == C.DAMAGE_PRESENT),
        "side_damaged": sum(1 for u in ws if u.side_damage == C.DAMAGE_PRESENT),
        "anomaly_wagons": sum(1 for u in ws if u.anomalies),
        "unresolved_associations": sum(
            1 for u in ws if u.association_status != ASSOC_RESOLVED),
    }
