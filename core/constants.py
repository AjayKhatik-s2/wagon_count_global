"""Compatibility constants for the ported `old_code` feature processors.

WHY THIS FILE EXISTS
--------------------
`old_code/` is the behavioural source of truth for door / load / damage / OCR,
and every one of its modules does `from core import constants as C`. That module
was NOT included with old_code, so the 31 symbols it referenced had to be
reconstructed here. Reconstructing them -- rather than rewriting the processors --
is what lets the mature algorithms run VERBATIM.

PROVENANCE OF EACH VALUE, AND WHAT IS INFERRED
----------------------------------------------
Values are marked RECOVERED when old_code pins them unambiguously (a docstring
states the literal, or the code compares against it), and INFERRED when they had
to be reasoned out. Every INFERRED value is flagged, because the user's
instruction was to preserve old thresholds rather than invent them -- so where a
value could not be recovered, that fact must be visible rather than buried.

  RECOVERED
    status sentinels        old_code/_common.py docstring: "OK"/"NO_FRAMES"/"FAILED"
    door states            door/processor.py docstring: CLOSED|OPEN|PARTIAL|DAMAGED
    load states            load/processor.py docstring: LOADED|EMPTY
    damage states          damage/processor.py docstring: DAMAGE|OK
    WAGON_NUMBER_LENGTH    wagon_number_ocr.py: "11-digit wagon number", C1..C11
    CONF_DAMAGE            damage/processor.py docstring: "confidence floor (0.55)"
    ENGINE / BRAKE_VAN     match this project's existing SegmentClass vocabulary

  INFERRED  (documented at each definition)
    CAMERA_FOLDER          the on-disk folder name per camera
    CONF_DOOR              cosmetic in old_code -- see note
    LOAD_LABEL_TO_STATE    depends on load.pt's class names, which are unknown
                           locally; resolved at RUNTIME from the model instead
    MODEL_* filenames      old names differ from the models now supplied

  OUT OF SCOPE
    OCR / wagon-number recognition is removed. Its constants remain only so the
    old ocr module stays importable; nothing calls it and no OCR model or
    easyocr install is required.

NOTHING HERE IS TRAIN-SPECIFIC. No frame numbers, wagon counts, offsets,
timestamps, speeds or geometry -- only vocabulary, thresholds and filenames.
"""

from __future__ import annotations

import os
from typing import Dict, List

# =============================================================================
# Status sentinels  (RECOVERED -- _common.py docstring and empty_payload usage)
# =============================================================================

STATUS_OK = "OK"
STATUS_NO_FRAMES = "NO_FRAMES"
STATUS_FAILED = "FAILED"

NO_DATA = "NO_DATA"
"""Used 41 times across old_code. Means "this feature has no usable evidence
here" -- deliberately NOT the same as a negative finding. Preserving that
distinction is why the reports never turn NO_DATA into CLOSED / EMPTY / OK."""


# =============================================================================
# Feature vocabularies  (RECOVERED from each processor's output docstring)
# =============================================================================

DOOR_CLOSED = "CLOSED"
DOOR_OPEN = "OPEN"
DOOR_PARTIAL = "PARTIAL"
DOOR_DAMAGED = "DAMAGED"

LOAD_LOADED = "LOADED"
LOAD_EMPTY = "EMPTY"

DAMAGE_PRESENT = "DAMAGE"
DAMAGE_OK = "OK"


# =============================================================================
# Wagon classification  (RECOVERED -- aligned with this project's SegmentClass)
# =============================================================================

CLASS_ENGINE = "ENGINE"
CLASS_BRAKE_VAN = "BRAKE_VAN"
CLASS_WAGON = "WAGON"


# =============================================================================
# Cameras  (RECOVERED -- identical ids to the current global pipeline)
# =============================================================================

CAMERA_RIGHT_UP = "RIGHT_UP"
CAMERA_LEFT_UP = "LEFT_UP"
CAMERA_RIGHT_UP_TOP = "RIGHT_UP_TOP"
CAMERA_LEFT_UP_TOP = "LEFT_UP_TOP"

ALL_CAMERAS: List[str] = [
    CAMERA_RIGHT_UP, CAMERA_LEFT_UP, CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP,
]

TOP_CAMERAS: List[str] = [CAMERA_RIGHT_UP_TOP, CAMERA_LEFT_UP_TOP]
"""Load and top-damage cameras. RIGHT_UP_TOP is AUTHORITATIVE and LEFT_UP_TOP
supporting -- old_code/load/processor.py states this explicitly and implements it,
so the order here is meaningful, not incidental."""

SIDE_CAMERAS: List[str] = [CAMERA_RIGHT_UP, CAMERA_LEFT_UP]
"""Door cameras. RIGHT_UP -> right door, LEFT_UP -> left door: old_code treats the
per-camera dominant state AS the per-side state, so this is camera authority, not
a merge."""

CAMERA_FOLDER: Dict[str, str] = {
    CAMERA_RIGHT_UP: CAMERA_RIGHT_UP,
    CAMERA_LEFT_UP: CAMERA_LEFT_UP,
    CAMERA_RIGHT_UP_TOP: CAMERA_RIGHT_UP_TOP,
    CAMERA_LEFT_UP_TOP: CAMERA_LEFT_UP_TOP,
}
"""INFERRED. old_code reads wagon_cache/<gw>/<CAMERA_FOLDER[cam]>/frame_*.jpg but
the original folder naming did not survive. The camera id itself is used, which is
self-consistent: the same mapping writes and reads the cache, so any naming works
as long as it is stable. Only matters if an OLD cache directory has to be read."""


# =============================================================================
# Confidence thresholds
# =============================================================================

CONF_DAMAGE = 0.55
"""RECOVERED. damage/processor.py documents "confidence floor (0.55)" and passes
this as the real per-detection gate. Note DamageTrackerConfig.confidence_threshold
is 0.50 -- the processor floor is deliberately stricter than the tracker's."""

CONF_DOOR = 0.68
"""INFERRED, but harmless. old_code/door/processor.py takes `confidence=C.CONF_DOOR`
and then never gates on it -- the actual floor is
`tracker_config.closed_confidence_threshold` (0.68), applied at line 165. This
value therefore only appears in a log line. Set to 0.68 so the log matches the
gate that is genuinely applied."""

CONF_OCR_BOX = 0.30
"""UNUSED -- OCR is out of scope (see the OCR section below). Retained only so
`old_code/ocr/processor.py` imports cleanly. No run reads this value."""


# =============================================================================
# OCR -- REMOVED FROM SCOPE
#
# OCR / wagon-number recognition is deliberately NOT part of this pipeline. The
# constants below remain only so `old_code/ocr/processor.py` stays importable
# alongside its siblings; no stage calls it, no run depends on it, and neither
# `easyocr` nor a wagon-number detector weight is required.
#
# Consequences, stated so the reports are not misread:
#   * `UnifiedWagonState.wagon_identifier` is always empty.
#   * The old report's OCR column is preserved (dropping it would change the old
#     layout) and renders as "-" for every wagon.
#   * The `ocr_captured` KPI is therefore always 0 -- that is "not attempted",
#     NOT "attempted and failed to read".
# =============================================================================

OCR_ENABLED = False
"""OCR is out of scope. Kept as an explicit flag so the state is visible in the
config rather than implied by an absent stage."""

WAGON_NUMBER_LENGTH = 11
"""RECOVERED from wagon_number_ocr.py (C1-C2 type, C3-C4 owning railway, C5-C6
year, C7-C10 serial, C11 check digit). Retained for importability only."""


# =============================================================================
# Load label mapping
# =============================================================================

LOAD_LABEL_TO_STATE: Dict[str, str] = {
    "loaded": LOAD_LOADED,
    "load": LOAD_LOADED,
    "full": LOAD_LOADED,
    "empty": LOAD_EMPTY,
    "unloaded": LOAD_EMPTY,
    "no_load": LOAD_EMPTY,
}
"""INFERRED, and the highest-risk item in this file.

old_code maps the load model's raw label through this dict, but the original
mapping did not survive and `load.pt` is not available locally, so its real class
names are unknown. The keys above are the plausible label spellings.

This is mitigated rather than guessed at runtime: `resolve_load_label_mapping()`
below inspects the ACTUAL model and reports any class name it cannot map, so an
unmapped label surfaces as a warning instead of being silently scored NO_DATA.
Any label not in the mapping contributes no vote -- which old_code already does
by design ("frames that didn't classify into either bucket ... contribute no
vote"), so an unknown label degrades to abstention, never to a wrong verdict."""


def resolve_load_label_mapping(model_class_names: Dict[int, str]) -> Dict[str, str]:
    """Map a load model's REAL class names onto LOADED / EMPTY.

    Called at startup with the loaded model's own `names`, so the mapping follows
    the checkpoint rather than an assumed class order -- retraining that reorders
    ids cannot mislabel anything. Substring matching mirrors how old_code's door
    FSM classifies ("partial" in name), which is the same retrain-robust idea.

    Returns the mapping actually used; unmapped names are simply absent, and the
    caller is expected to report them.
    """
    out: Dict[str, str] = {}
    for raw in (str(v).strip().lower() for v in (model_class_names or {}).values()):
        if not raw:
            continue
        if raw in LOAD_LABEL_TO_STATE:
            out[raw] = LOAD_LABEL_TO_STATE[raw]
        elif "empty" in raw or "unload" in raw:
            out[raw] = LOAD_EMPTY
        elif "load" in raw or "full" in raw:
            out[raw] = LOAD_LOADED
    return out


# =============================================================================
# Model filenames
# =============================================================================

MODEL_DOOR_STATE = "door_state.pt"
"""RECOVERED -- old_code names this file directly, and it is the model supplied."""

MODEL_DAMAGE = "top_damage.pt"
"""ADAPTED. old_code's docstrings refer to `damage.pt`; the model supplied for
this project is `top_damage.pt`. Filename only -- no behaviour depends on it, and
class handling is by name, so the substitution is safe."""

MODEL_LOADED = "load.pt"
"""ADAPTED. old_code's missing-file message says `loaded.pt`, whereas the model
being supplied via S3 is `load.pt`. Both are accepted -- see
`resolve_model_path()` -- so neither naming breaks a run."""

MODEL_WAGON_ID_COUNTING = "wagon_id_counting.pt"
"""NOT REQUIRED -- OCR is out of scope, so no wagon-number detector weight is
needed. Retained as a name only, for importability."""

MODEL_ALIASES: Dict[str, List[str]] = {
    MODEL_DOOR_STATE: ["door_state.pt", "door.pt"],
    MODEL_DAMAGE: ["top_damage.pt", "damage.pt"],
    MODEL_LOADED: ["load.pt", "loaded.pt"],
    MODEL_WAGON_ID_COUNTING: ["wagon_id_counting.pt", "wagon_number.pt",
                              "wagon_id.pt"],
}
"""Accepted filenames per role, so a weight named with either the old or the new
convention resolves. Order is the search order."""


def resolve_model_path(models_dir: str, canonical_filename: str) -> str:
    """First existing alias for a model role, else the canonical path.

    Returning the canonical path when nothing exists keeps old_code's behaviour
    intact: `load_yolo` returns None for a missing file and the processor emits
    NO_DATA, so a missing model never raises here.
    """
    for name in MODEL_ALIASES.get(canonical_filename, [canonical_filename]):
        candidate = os.path.join(models_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(models_dir, canonical_filename)
