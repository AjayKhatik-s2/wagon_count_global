"""Inspection model discovery, availability and caching.

TWO REQUIREMENTS DRIVE THIS MODULE.

First, CLASS NAMES ARE NEVER ASSUMED. Every class id and name is read from the
checkpoint at load time and carried through unchanged into tracking, the JSON and
the report. The models actually shipped were found to be:

    door_state.pt   detect, imgsz 640
        0 closed_door   1 damage   2 open_door   3 partially_closed
    top_damage.pt   detect (RT-DETR backbone), imgsz 640
        0 Floor__probable_damage   1 Floor_damage   2 Inner_wall_damage

Those are recorded here as documentation of what was observed, NOT as a mapping
the code relies on: nothing below hardcodes them, and a model with different
classes flows through the same path. Note in particular that the DOOR model also
carries a `damage` class, so "the door model" is not synonymous with "door
states" -- its classes are partitioned at runtime, by name, into door-state
classes and damage classes.

Second, A MISSING MODEL MUST FAIL BEFORE THE EXPENSIVE RUN. Availability is
resolved up front and reported, so a train is never tracked for half an hour only
to die at the inspection stage.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# Default filenames, matching the project's drop-files-in-./models convention.
DOOR_MODEL_FILENAME = "door_state.pt"
DAMAGE_MODEL_FILENAME = "top_damage.pt"

DOOR_ROLE = "door"
DAMAGE_ROLE = "top_damage"

AVAILABLE = "AVAILABLE"
UNAVAILABLE = "UNAVAILABLE"


# ---------------------------------------------------------------------------
# class-name partitioning -- by NAME, never by id
# ---------------------------------------------------------------------------

_DAMAGE_TOKEN = re.compile(r"damage|crack|dent|hole|broken|torn", re.I)
_DOOR_TOKEN = re.compile(r"door|shutter|hatch|closed|open", re.I)


def partition_class_names(names: Dict[Any, str]) -> Dict[str, List[str]]:
    """Split a model's own class names into semantic groups.

    Grouping is by NAME, never by class id, because ids are arbitrary and differ
    between checkpoints -- an id-based mapping silently mislabels everything the
    moment a model is retrained with a reordered class list.

    A class can legitimately land in `damage` even when it came from the door
    model: `door_state.pt` ships a `damage` class. Anything unrecognised goes to
    `other` and is still tracked and reported, never silently dropped.
    """
    groups: Dict[str, List[str]] = {"door_state": [], "damage": [], "other": []}
    for name in (str(n) for n in names.values()):
        if _DAMAGE_TOKEN.search(name):
            groups["damage"].append(name)
        elif _DOOR_TOKEN.search(name):
            groups["door_state"].append(name)
        else:
            groups["other"].append(name)
    for k in groups:
        groups[k].sort()
    return groups


# ---------------------------------------------------------------------------
# availability
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Where a model should be and what role it plays."""
    role: str
    filename: str
    path: Optional[str] = None
    cameras: Sequence[str] = ()

    @property
    def exists(self) -> bool:
        return bool(self.path) and os.path.isfile(self.path)


@dataclass
class ModelAvailability:
    """Whether a model can be used, and everything discovered about it."""
    role: str
    status: str
    path: Optional[str] = None
    reason: str = ""
    task: str = ""
    class_names: Dict[int, str] = field(default_factory=dict)
    class_groups: Dict[str, List[str]] = field(default_factory=dict)
    imgsz: Optional[int] = None
    parameters: Optional[int] = None
    trained_version: str = ""

    @property
    def is_available(self) -> bool:
        return self.status == AVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role, "status": self.status, "path": self.path,
            "reason": self.reason, "task": self.task,
            "class_names": {int(k): v for k, v in self.class_names.items()},
            "class_groups": {k: list(v) for k, v in self.class_groups.items()},
            "imgsz": self.imgsz, "parameters": self.parameters,
            "trained_version": self.trained_version,
        }


def _introspect(path: str) -> Dict[str, Any]:
    """Read a checkpoint's real task, classes and input size.

    Imported lazily so that this module -- and therefore the tests that exercise
    availability, tracking and association -- never requires torch/ultralytics.
    """
    from ultralytics import YOLO                       # noqa: WPS433 (lazy)

    model = YOLO(path)
    names = getattr(model, "names", None) or {}
    out: Dict[str, Any] = {
        "task": str(getattr(model, "task", "") or ""),
        "class_names": {int(k): str(v) for k, v in names.items()},
        "imgsz": None,
        "parameters": None,
        "trained_version": "",
        "handle": model,
    }
    ckpt = getattr(model, "ckpt", None)
    if isinstance(ckpt, dict):
        targs = ckpt.get("train_args")
        if isinstance(targs, dict) and targs.get("imgsz"):
            try:
                out["imgsz"] = int(targs["imgsz"])
            except (TypeError, ValueError):
                pass
        out["trained_version"] = str(ckpt.get("version", "") or "")
    inner = getattr(model, "model", None)
    if inner is not None:
        try:
            out["parameters"] = sum(p.numel() for p in inner.parameters())
        except Exception:
            pass
    return out


def discover_models(
    models_dir: str,
    door_filename: str = DOOR_MODEL_FILENAME,
    damage_filename: str = DAMAGE_MODEL_FILENAME,
    door_path: Optional[str] = None,
    damage_path: Optional[str] = None,
    introspect: bool = True,
) -> Dict[str, ModelAvailability]:
    """Resolve both inspection models UP FRONT, before any expensive processing.

    Never raises for a missing or unloadable model: the role is reported
    UNAVAILABLE with the reason, so the counting pipeline still completes and the
    report can say plainly that the stage did not run. A silent class mismatch is
    likewise impossible -- classes are always read and always reported.
    """
    specs = [
        ModelSpec(DOOR_ROLE, door_filename,
                  door_path or os.path.join(models_dir, door_filename)),
        ModelSpec(DAMAGE_ROLE, damage_filename,
                  damage_path or os.path.join(models_dir, damage_filename)),
    ]
    out: Dict[str, ModelAvailability] = {}
    for spec in specs:
        if not spec.exists:
            out[spec.role] = ModelAvailability(
                role=spec.role, status=UNAVAILABLE, path=spec.path,
                reason=(f"{spec.filename} not found -- drop it in {models_dir} "
                        f"to enable {spec.role} inspection"))
            continue
        if not introspect:
            out[spec.role] = ModelAvailability(
                role=spec.role, status=AVAILABLE, path=spec.path,
                reason="present (not introspected)")
            continue
        try:
            info = _introspect(spec.path)
        except Exception as exc:
            out[spec.role] = ModelAvailability(
                role=spec.role, status=UNAVAILABLE, path=spec.path,
                reason=f"failed to load: {type(exc).__name__}: {exc}")
            continue
        names = info["class_names"]
        out[spec.role] = ModelAvailability(
            role=spec.role, status=AVAILABLE, path=spec.path,
            reason="loaded", task=info["task"], class_names=names,
            class_groups=partition_class_names(names), imgsz=info["imgsz"],
            parameters=info["parameters"],
            trained_version=info["trained_version"])
    return out


def describe_model_availability(avail: Dict[str, ModelAvailability]) -> str:
    """The startup banner. Printed BEFORE the expensive stages run."""
    lines: List[str] = []
    for role in (DOOR_ROLE, DAMAGE_ROLE):
        a = avail.get(role)
        label = {DOOR_ROLE: "DOOR MODEL", DAMAGE_ROLE: "TOP DAMAGE MODEL"}[role]
        if a is None:
            lines.append(f"  {label:<18}: UNAVAILABLE (not configured)")
            continue
        lines.append(f"  {label:<18}: {a.status}")
        lines.append(f"      path          : {a.path}")
        if not a.is_available:
            lines.append(f"      reason        : {a.reason}")
            continue
        lines.append(f"      model type    : {a.task or 'unknown'}"
                     + (f"   imgsz={a.imgsz}" if a.imgsz else ""))
        if a.parameters:
            lines.append(f"      parameters    : {a.parameters:,}")
        if a.trained_version:
            lines.append(f"      trained with  : ultralytics {a.trained_version}")
        lines.append(f"      classes ({len(a.class_names)}):")
        for cid in sorted(a.class_names):
            lines.append(f"          id={cid}  {a.class_names[cid]!r}")
        for group, members in sorted(a.class_groups.items()):
            if members:
                lines.append(f"      {group:<13} : {', '.join(members)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# cached handles
# ---------------------------------------------------------------------------

class InspectionModel:
    """A loaded model, held once and reused across every camera.

    Loading a 33M-parameter network per camera (let alone per frame) is the
    obvious way to exhaust a CPU-bound EC2 box, so the handle is created once and
    shared. `close()` drops it so a long-running process that inspects several
    trains in sequence does not accumulate networks.
    """

    def __init__(self, availability: ModelAvailability):
        if not availability.is_available:
            raise ValueError(f"{availability.role} model is {availability.status}: "
                             f"{availability.reason}")
        self.availability = availability
        self.role = availability.role
        self.class_names = dict(availability.class_names)
        self._handle: Any = None

    @property
    def handle(self) -> Any:
        if self._handle is None:
            from ultralytics import YOLO                # noqa: WPS433 (lazy)
            self._handle = YOLO(self.availability.path)
        return self._handle

    def class_name(self, class_id: int) -> str:
        """The model's OWN name for a class id, never a fabricated label."""
        return self.class_names.get(int(class_id), f"class_{int(class_id)}")

    def close(self) -> None:
        self._handle = None

    def __enter__(self) -> "InspectionModel":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
