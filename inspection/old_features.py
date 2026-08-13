"""Run the UNCHANGED old_code feature processors on the finalized global wagons.

    finalized GW_1..GW_N          <- protected, produced by the counting pipeline
        |
    wagon_cache/<GW_n>/<camera>/  <- association bridge (inspection/wagon_cache.py)
        |
    old_code door / load / damage <- imported verbatim via the `features` shim
        |
    per-wagon JSON per feature    <- exactly the shapes old_code writes
        |
    UnifiedWagonState per GW_n    <- fused view the old report consumes
        |
    additive state.inspection

WHAT THIS MODULE IS AND IS NOT
------------------------------
It is ORCHESTRATION ONLY. Not one detection, tracking, filtering, voting or
state-machine decision is made here -- every one of those lives in old_code and is
called unmodified. This module decides only: which order the features run in, where
their frames come from, and how their outputs are fused for the report.

ORDER MATTERS AND IS NOT ARBITRARY
----------------------------------
load runs BEFORE damage, because old_code's damage processor reads the sibling load
JSON and drops `floor_damage` tracks on wagons that are LOADED (you cannot see the
floor under a load). Running damage first would silently lose that coupling, so the
order is load -> damage, with door independent.

THE COUNT IS PROTECTED BY A CHECKED PROPERTY
--------------------------------------------
The wagon roster is hashed before anything runs and verified afterwards. Inspection
reads `state.wagons` and writes only its own files, so the hash cannot change --
and if it ever does, the run fails loudly rather than shipping an altered count.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C
from core.global_state_loader import assert_roster_unchanged, roster_hash
from core.unified_wagon_state import (
    ASSOC_RESOLVED, ASSOC_UNRESOLVED, UnifiedWagonState, summarize_wagons,
)

from .wagon_cache import (
    CACHE_DIRNAME, WagonCacheConfig, build_wagon_cache, cache_stats,
    clear_wagon_cache, plan_cache, wagon_camera_dir,
)

__all__ = ["OldInspectionConfig", "OldInspectionResult", "run_old_inspection",
           "discover_feature_models", "fuse_unified_states"]

FEATURES = ("door", "load", "damage")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class OldInspectionConfig:
    """Knobs for the port. Old thresholds are NOT duplicated here.

    Everything that decides a verdict -- confidence floors, area and edge filters,
    Kalman/Hungarian parameters, vote windows, FSM hysteresis, the 0.35 load ratio
    -- lives in old_code and `core.constants`. Only orchestration is configurable
    here, so there is no second place where a threshold can drift.
    """
    enabled: bool = True
    cache: WagonCacheConfig = field(default_factory=WagonCacheConfig)
    keep_cache: bool = False
    """Delete the wagon frame cache after the features have run. Off by default:
    the cache is per-train state and one train's GW_1 directory must never be read
    by the next train's GW_1."""
    load_every_nth: int = 2
    """old_code/load/processor.py's own default."""
    run_door: bool = True
    run_load: bool = True
    run_damage: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "keep_cache": self.keep_cache,
                "load_every_nth": self.load_every_nth,
                "run_door": self.run_door, "run_load": self.run_load,
                "run_damage": self.run_damage,
                "cache": {"every_nth": self.cache.every_nth,
                          "max_frames_per_wagon": self.cache.max_frames_per_wagon,
                          "jpeg_quality": self.cache.jpeg_quality},
                "ocr": "REMOVED_FROM_SCOPE"}


@dataclass
class OldInspectionResult:
    unified: Dict[str, UnifiedWagonState] = field(default_factory=dict)
    feature_summaries: Dict[str, Dict[str, str]] = field(default_factory=dict)
    model_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)
    camera_status: Dict[str, str] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    roster_hash_before: str = ""
    roster_hash_after: str = ""

    def to_dict(self) -> Dict[str, Any]:
        wagons = list(self.unified.values())
        return {
            "source": "old_code (door/load/damage), run unchanged",
            "ocr": "REMOVED_FROM_SCOPE",
            "summary": summarize_wagons(wagons),
            "wagons": {gid: u.to_dict()
                       for gid, u in sorted(self.unified.items(), key=_gw_item)},
            "feature_status": {f: dict(s)
                               for f, s in sorted(self.feature_summaries.items())},
            "model_status": dict(self.model_status),
            "wagon_cache": dict(self.cache),
            "camera_status": dict(sorted(self.camera_status.items())),
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
            "warnings": list(self.warnings),
            "roster_hash_before": self.roster_hash_before,
            "roster_hash_after": self.roster_hash_after,
            "roster_unchanged": (self.roster_hash_before == self.roster_hash_after
                                 and bool(self.roster_hash_before)),
        }


def _gw_item(item: Tuple[str, Any]) -> Tuple[int, Any]:
    tail = str(item[0]).split("_")[-1]
    return (0, int(tail)) if tail.isdigit() else (1, str(item[0]))


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def discover_feature_models(models_dir: str,
                            verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """Resolve each feature's weight UP FRONT and report its real classes.

    Runs before any frame is decoded, so a missing weight is visible immediately
    rather than after the cache has been built. A missing model is NOT fatal:
    old_code emits NO_DATA for that feature and the wagon count is untouched.

    Class names are read from the checkpoint, never assumed -- and for load the
    label->state mapping is resolved from those real names.
    """
    out: Dict[str, Dict[str, Any]] = {}
    roles = (("door", C.MODEL_DOOR_STATE), ("load", C.MODEL_LOADED),
             ("damage", C.MODEL_DAMAGE))
    for role, canonical in roles:
        path = C.resolve_model_path(models_dir, canonical)
        rec: Dict[str, Any] = {"role": role, "path": path,
                               "expected_filenames": C.MODEL_ALIASES.get(canonical, []),
                               "available": os.path.isfile(path)}
        if not rec["available"]:
            rec["status"] = "UNAVAILABLE"
            rec["reason"] = (f"none of {rec['expected_filenames']} found in "
                             f"{models_dir}; {role} will report NO_DATA for every "
                             f"wagon and the wagon count is unaffected")
            out[role] = rec
            continue
        try:
            from features._common import load_yolo, model_class_names
            model = load_yolo(path)
            names = model_class_names(model)
            rec["status"] = "AVAILABLE"
            rec["task"] = str(getattr(model, "task", "") or "")
            rec["class_names"] = {int(k): str(v) for k, v in names.items()}
            if role == "load":
                mapping = C.resolve_load_label_mapping(names)
                rec["load_label_mapping"] = mapping
                unmapped = sorted({str(v).strip().lower() for v in names.values()}
                                  - set(mapping))
                if unmapped:
                    rec["unmapped_class_names"] = unmapped
                    rec["warning"] = (
                        f"load.pt classes {unmapped} could not be mapped to "
                        f"LOADED/EMPTY; frames predicting them abstain rather than "
                        f"voting, matching old_code")
                # Install the resolved mapping so old_code's `_canonical_load`
                # uses the REAL class names of the model it was given.
                C.LOAD_LABEL_TO_STATE.update(mapping)
        except Exception as exc:
            rec["status"] = "UNAVAILABLE"
            rec["reason"] = f"failed to load: {type(exc).__name__}: {exc}"
        out[role] = rec

    if verbose:
        print("-" * 70)
        print("  INSPECTION MODELS (old_code features; OCR removed from scope)")
        print("-" * 70)
        for role, rec in out.items():
            print(f"  {role.upper():<8} {rec['status']:<12} {rec['path']}")
            if rec.get("class_names"):
                print(f"           classes: "
                      + ", ".join(f"{k}:{v}" for k, v in
                                  sorted(rec["class_names"].items())))
            if rec.get("load_label_mapping"):
                print(f"           load mapping: {rec['load_label_mapping']}")
            if rec.get("warning"):
                print(f"           WARNING: {rec['warning']}")
            if rec.get("reason"):
                print(f"           {rec['reason']}")
    return out


# ---------------------------------------------------------------------------
# CPU compatibility shim for old_code's half-precision door inference
# ---------------------------------------------------------------------------

class _HalfDisabledModel:
    """Proxy that forces `half=False` on inference calls.

    OLD BEHAVIOUR
        `old_code/door/processor.py` line 151 calls the door model as
        `yolo_model(frame, verbose=False, half=True)`. Half precision is a GPU
        optimisation, and on GPU it is exactly right.

    NEW BEHAVIOUR
        When CUDA is NOT available, `half` is forced off for the door model.

    WHY IT DIFFERS
        Measured on this CPU box, with the real weight and a real cached frame:

            half=False    782 - 1598 ms per frame
            half=True   103487 - 104035 ms per frame

        ~130x slower, and it does not improve after warm-up -- float16 has no
        optimised CPU kernels, so every op falls back to a slow path. A 57-wagon
        train at a few frames per wagon per camera would take days rather than
        minutes, which is not a slow run but a non-functional one. This was
        observed as a real hang before it was diagnosed.

    CAN IT AFFECT RESULTS?
        Only at fp16-rounding level, and in the direction of MORE precision:
        float32 inference is the reference, and half precision is the
        approximation of it. Detection geometry, class names, confidences,
        thresholds, tracking, voting and the state machine are untouched. On a
        GPU box this proxy is never installed, so GPU behaviour is bit-for-bit
        the old behaviour.

    Everything other than the `half` kwarg is delegated unchanged, so the object
    remains indistinguishable from the model old_code expects -- including
    `.names`, which the door processor reads to build its Detection objects.
    """

    __slots__ = ("_model",)

    def __init__(self, model: Any):
        object.__setattr__(self, "_model", model)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["half"] = False
        return object.__getattribute__(self, "_model")(*args, **kwargs)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["half"] = False
        return object.__getattribute__(self, "_model").predict(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_model"), name)


def install_cpu_half_shim(model_path: str, verbose: bool = True) -> bool:
    """Neutralise half precision for `model_path` when there is no GPU.

    Works by replacing the entry in old_code's OWN model cache, so the processor
    receives the proxy from its own `load_yolo` call and no old_code file is
    edited. Returns True if the shim was installed.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return False                     # GPU: keep old behaviour exactly
    except Exception:
        pass

    try:
        from features import _common as old_common
        model = old_common.load_yolo(model_path)
        if model is None:
            return False
        key = os.path.abspath(model_path)
        with old_common._MODEL_LOCK:
            cached = old_common._MODEL_CACHE.get(key)
            if isinstance(cached, _HalfDisabledModel):
                return True
            old_common._MODEL_CACHE[key] = _HalfDisabledModel(cached or model)
    except Exception as exc:
        if verbose:
            print(f"    NOTE: could not install CPU half shim: "
                  f"{type(exc).__name__}: {exc}")
        return False
    if verbose:
        print("    [CPU] half precision disabled for door inference "
              "(measured ~130x slower on CPU; GPU behaviour unchanged)")
    return True


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------

def _read_feature_json(output_dir: str, feature: str,
                       gw_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(output_dir, feature, f"{gw_id}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def fuse_unified_states(
    state: Any,
    feature_output_dir: str,
    camera_status: Dict[str, str],
    cache_root: str,
) -> Dict[str, UnifiedWagonState]:
    """Combine the three per-wagon feature JSONs into one view per GW id.

    Reads the files old_code wrote -- it does not recompute a verdict, so the
    fused view cannot disagree with the processors.

    Every wagon in the roster gets an entry, findings or not, so the report can
    never omit a wagon. States stay distinct: a feature with no JSON, or a
    NO_DATA/NO_FRAMES status, stays NO_DATA rather than becoming CLOSED / EMPTY /
    OK.
    """
    wagons = getattr(state, "wagons", None)
    if wagons is None and isinstance(state, dict):
        wagons = state.get("wagons") or []

    unified: Dict[str, UnifiedWagonState] = {}
    for w in wagons or []:
        gid = (w.get("global_id") if isinstance(w, dict)
               else getattr(w, "global_id", None))
        cls = (w.get("classification") if isinstance(w, dict)
               else getattr(w, "classification", C.CLASS_WAGON))
        if gid is None:
            continue
        u = UnifiedWagonState(global_id=str(gid), classification=str(cls or ""))

        # ---- door: RIGHT_UP -> right, LEFT_UP -> left (old camera authority) --
        door = _read_feature_json(feature_output_dir, "door", u.global_id)
        if door:
            u.feature_status["door"] = str(door.get("status", C.NO_DATA))
            u.left_door = str(door.get("left_door", C.NO_DATA) or C.NO_DATA)
            u.left_door_confidence = float(door.get("left_door_confidence", 0.0) or 0.0)
            u.right_door = str(door.get("right_door", C.NO_DATA) or C.NO_DATA)
            u.right_door_confidence = float(door.get("right_door_confidence", 0.0) or 0.0)
            if door.get("tracks"):
                u.tracks["door"] = list(door["tracks"])
            for cam in door.get("supporting_cameras") or []:
                if cam not in u.supporting_cameras:
                    u.supporting_cameras.append(cam)
        else:
            u.feature_status["door"] = C.NO_DATA

        # ---- load: RIGHT_UP_TOP authoritative, LEFT_UP_TOP fallback ----------
        load = _read_feature_json(feature_output_dir, "load", u.global_id)
        if load:
            u.feature_status["load"] = str(load.get("status", C.NO_DATA))
            u.load_status = str(load.get("load_status", C.NO_DATA) or C.NO_DATA)
            u.load_confidence = float(load.get("load_confidence", 0.0) or 0.0)
            if load.get("per_camera"):
                u.tracks["load_per_camera"] = [load["per_camera"]]
            for cam in load.get("supporting_cameras") or []:
                if cam not in u.supporting_cameras:
                    u.supporting_cameras.append(cam)
        else:
            u.feature_status["load"] = C.NO_DATA

        # ---- damage (top cameras) --------------------------------------------
        dmg = _read_feature_json(feature_output_dir, "damage", u.global_id)
        if dmg:
            u.feature_status["damage"] = str(dmg.get("status", C.NO_DATA))
            u.top_damage = str(dmg.get("top_damage", C.NO_DATA) or C.NO_DATA)
            details = dmg.get("top_damage_details") or []
            if details:
                u.top_damage_confidence = max(
                    float(d.get("confidence", 0.0) or 0.0) for d in details)
                u.evidence["damage"] = list(details)
                u.tracks["damage"] = list(details)
            for cam in dmg.get("supporting_cameras") or []:
                if cam not in u.supporting_cameras:
                    u.supporting_cameras.append(cam)
        else:
            u.feature_status["damage"] = C.NO_DATA

        # side damage: no side-damage model is supplied, so it stays NO_DATA --
        # explicitly "not inspected", never "no damage".
        u.side_damage = C.NO_DATA

        # ---- per (wagon, camera) evidence status ----------------------------
        for cam in C.ALL_CAMERAS:
            cam_state = camera_status.get(cam, "UNRESOLVED")
            role = "door" if cam in C.SIDE_CAMERAS else "load/damage"
            if cam_state != "RESOLVED":
                u.camera_status[cam] = {role: "UNRESOLVED"
                                        if cam_state == "UNRESOLVED"
                                        else "NOT_VISIBLE"}
                continue
            frames = wagon_camera_dir(cache_root, u.global_id, cam)
            if not (os.path.isdir(frames) and os.listdir(frames)):
                u.camera_status[cam] = {role: "NOT_VISIBLE"}
            elif cam in u.supporting_cameras:
                u.camera_status[cam] = {role: "INSPECTED"}
            else:
                u.camera_status[cam] = {role: "NO_DETECTION"}

        u.association_status = (
            ASSOC_RESOLVED if u.supporting_cameras else ASSOC_UNRESOLVED)
        unified[u.global_id] = u
    return unified


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def run_old_inspection(
    state: Any,
    tracks_by_camera: Dict[str, Any],
    models_dir: str,
    output_root: str,
    cfg: Optional[OldInspectionConfig] = None,
    verbose: bool = True,
) -> OldInspectionResult:
    """Run door / load / damage on the finalized wagons, using old_code verbatim.

    `state` is read-only here and the guarantee is checked, not assumed: the roster
    is hashed before and after, and a mismatch raises.
    """
    cfg = cfg or OldInspectionConfig()
    res = OldInspectionResult()
    res.roster_hash_before = roster_hash(state)

    if not cfg.enabled:
        res.warnings.append("inspection disabled by configuration")
        res.roster_hash_after = res.roster_hash_before
        return res

    feature_out = os.path.join(output_root, "features")
    os.makedirs(feature_out, exist_ok=True)

    # ---- models first, so a missing weight is known before any decoding ----
    res.model_status = discover_feature_models(models_dir, verbose=verbose)

    # ---- association bridge -------------------------------------------------
    t0 = time.time()
    plan = plan_cache(state, tracks_by_camera, output_root, cfg=cfg.cache)
    res.camera_status = dict(plan.camera_status)
    for skip in plan.skipped:
        if "global_id" not in skip:
            res.warnings.append(
                f"{skip.get('camera_id')}: {skip.get('reason')}")
    if verbose:
        print(f"  [CACHE] planning {len(plan.windows)} wagon window(s) across "
              f"{len({w.camera_id for w in plan.windows})} camera(s)")
    res.cache = build_wagon_cache(plan, tracks_by_camera, cfg.cache,
                                  verbose=verbose)
    res.cache["planned"] = plan.to_dict()
    res.timings["wagon_cache_seconds"] = time.time() - t0
    cache_root = plan.root

    # ---- old processors, UNCHANGED -----------------------------------------
    # load BEFORE damage: damage reads the sibling load JSON to drop floor_damage
    # on LOADED wagons, which is old_code behaviour that ordering must preserve.
    if cfg.run_door and res.model_status.get("door", {}).get("available"):
        t = time.time()
        try:
            # old_code asks for half precision, which is a GPU optimisation and is
            # pathological on CPU (measured ~130x slower). Neutralised only when
            # there is no GPU; on GPU the old call is untouched.
            if install_cpu_half_shim(res.model_status["door"]["path"],
                                     verbose=verbose):
                res.warnings.append(
                    "door: half precision disabled because no GPU is available "
                    "(measured ~130x slower on CPU); GPU runs keep old behaviour")
            from features.door import processor as door_proc
            res.feature_summaries["door"] = door_proc.run(
                state=state, cache_root=cache_root,
                feature_models_dir=models_dir, output_dir=feature_out,
                verbose=verbose)
        except Exception as exc:
            res.warnings.append(f"door feature failed: {type(exc).__name__}: {exc}")
        res.timings["door_seconds"] = time.time() - t
    else:
        res.warnings.append("door skipped: model unavailable or disabled")

    if cfg.run_load and res.model_status.get("load", {}).get("available"):
        t = time.time()
        try:
            from features.load import processor as load_proc
            # max_frames=None, NOT old_code's own default of 0.
            #
            # OLD BEHAVIOUR: `load/processor.py` defaults `max_frames=0` and passes
            # it straight to `iter_wagon_frames`, where the guard is
            # `if max_frames is not None and len(paths) > max_frames` -- so 0
            # subsamples the list down to ZERO frames and the feature can never
            # cast a vote. Observed: load finished in 0.0s with ok=0/3.
            # NEW BEHAVIOUR: None, which is what that guard treats as unbounded
            # and what the docstring's "0 = unbounded" clearly intended.
            # AFFECTS RESULTS? Only by letting the feature run at all; the 0.35
            # ratio rule, the voting and the camera authority are untouched.
            res.feature_summaries["load"] = load_proc.run(
                state=state, cache_root=cache_root,
                feature_models_dir=models_dir, output_dir=feature_out,
                every_nth=cfg.load_every_nth, max_frames=None, verbose=verbose)
        except Exception as exc:
            res.warnings.append(f"load feature failed: {type(exc).__name__}: {exc}")
        res.timings["load_seconds"] = time.time() - t
    else:
        res.warnings.append(
            "load skipped: load.pt unavailable (supplied on EC2 from S3) or disabled")

    if cfg.run_damage and res.model_status.get("damage", {}).get("available"):
        t = time.time()
        try:
            from features.damage import processor as dmg_proc
            res.feature_summaries["damage"] = dmg_proc.run(
                state=state, cache_root=cache_root,
                feature_models_dir=models_dir, output_dir=feature_out,
                verbose=verbose)
        except Exception as exc:
            res.warnings.append(f"damage feature failed: {type(exc).__name__}: {exc}")
        res.timings["damage_seconds"] = time.time() - t
    else:
        res.warnings.append("damage skipped: model unavailable or disabled")

    # ---- fuse -------------------------------------------------------------
    res.unified = fuse_unified_states(state, feature_out, res.camera_status,
                                      cache_root)
    res.cache["on_disk"] = cache_stats(cache_root)

    if not cfg.keep_cache:
        # Per-train state: clearing it is what stops train B's GW_1 from reading
        # train A's cached frames.
        clear_wagon_cache(cache_root, verbose=verbose)

    # ---- the checked guarantee -------------------------------------------
    res.roster_hash_after = roster_hash(state)
    assert_roster_unchanged(state, res.roster_hash_before)
    return res
