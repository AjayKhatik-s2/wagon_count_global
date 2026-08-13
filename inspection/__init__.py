"""Downstream inspection features attached to an ALREADY-BUILT global train.

    ALL CAMERA VIDEOS
           |
           v
    GLOBAL TRAIN STRUCTURE      <-- built by the counting pipeline, PROTECTED
           |
      +----+----+
      |         |
      v         v
  GW IDs   GLOBAL TIMELINE
      |
      +---------------+
      |               |
      v               v
  DOOR            TOP DAMAGE
      |               |
      +-------+-------+
              |
              v
    GLOBAL INSPECTION STATE
              |
       +------+------+
       |             |
       v             v
     PDF        PROCESSED VIDEOS

THE ONE RULE THIS PACKAGE OBEYS
-------------------------------
The global wagon roster is the source of truth and is READ-ONLY here. Nothing in
this package can create a wagon, delete a wagon, renumber a GW id, alter a gap,
a boundary, a camera offset or the MASTER == GLOBAL invariant. Inspection
findings are annotations on a structure that already exists.

That is enforced structurally, not by convention: association resolves a
detection to a GW id by LOOKUP into the finished wagon roster, and a detection
that resolves to nothing is recorded as unresolved rather than given a wagon of
its own.

WHY THE TRACKING HERE MIRRORS THE GAP PIPELINE
----------------------------------------------
Measured on real video, an inspection detection behaves exactly like a gap
candidate: a genuine door was detected across 28 consecutive frames with its
centre advancing ~28 px/frame -- the train's own speed -- while false positives
appeared as 1-5 frame runs at low confidence, several of them PINNED near the
frame edge. So the same principles that made gap counting reliable apply:
temporal persistence, motion consistent with the train, static rejection, and
confidence-weighted class voting rather than trusting one frame.
"""

from __future__ import annotations

from .models import (
    InspectionModel, ModelAvailability, ModelSpec, discover_models,
    describe_model_availability,
)
from .state import (
    ASSOCIATION_AMBIGUOUS, ASSOCIATION_RESOLVED, ASSOCIATION_UNRESOLVED,
    CAMERA_NOT_APPLICABLE, CAMERA_NOT_VISIBLE, CAMERA_NO_DETECTION,
    CAMERA_CONFIRMED, CAMERA_UNRESOLVED,
    InspectionConfig, InspectionEvent, InspectionState, WagonInspection,
)
from .tracking import DetectionObservation, InspectionTrack, track_detections
from .association import associate_tracks_to_wagons

__all__ = [
    "InspectionModel", "ModelAvailability", "ModelSpec", "discover_models",
    "describe_model_availability",
    "InspectionConfig", "InspectionEvent", "InspectionState", "WagonInspection",
    "DetectionObservation", "InspectionTrack", "track_detections",
    "associate_tracks_to_wagons",
    "ASSOCIATION_RESOLVED", "ASSOCIATION_UNRESOLVED", "ASSOCIATION_AMBIGUOUS",
    "CAMERA_CONFIRMED", "CAMERA_NO_DETECTION", "CAMERA_NOT_VISIBLE",
    "CAMERA_UNRESOLVED", "CAMERA_NOT_APPLICABLE",
]
