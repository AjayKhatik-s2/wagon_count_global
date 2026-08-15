"""Run the LEGACY inspection/output stack on the finalized global roster.

    FOUR VIDEOS
        -> current global counting  (gaps -> tracking -> stitching -> validation
           -> master sequence -> camera sync -> fusion)                PROTECTED
        -> FINAL ROSTER GW_1..GW_N                                     PROTECTED
        -> wagon_cache/GW_n/CAM/                    inspection/wagon_cache.py
        -> segment_summary_df per camera            inspection/global_bridge.py
        -> LOAD (top)                               old_code load processor
        -> DOOR + DAMAGE                            legacy DamageDetector
        -> problem frames                           legacy ProblemFrameExtractor
        -> evidence + S3 artifacts                  legacy ArtifactPublisher
        -> inspection_data.json per camera          legacy build_inspection_json
        -> PDF / annotated video                    legacy renderers, JSON-fed

WHAT THIS MODULE DECIDES, AND WHAT IT DOES NOT
----------------------------------------------
It decides ORDER, MODEL WIRING and PERSISTENCE. It makes no detection, no
threshold decision, no band rule, no door vote and no JSON field choice -- all
of those are in ``inspection/legacy/``, unmodified, and all of them are called
rather than reimplemented.

MODEL AUTHORITY (explicit, per the porting brief)
-------------------------------------------------
    side (LEFT_UP / RIGHT_UP)   door_state.pt   AUTHORITATIVE
    top  (*_UP_TOP)             top_damage.pt   AUTHORITATIVE
    load (top cameras)          load.pt         AUTHORITATIVE
    OCR                         none            DISABLED

``door_state.pt`` REPLACES the legacy ``V4_side_damage.pt``. This is not an
assumption -- the checkpoint's own ``names`` were read and are
``{closed_door, damage, open_door, partially_closed}``, which is exactly the
legacy side model's class set and exactly what ``damage.SIDE_CLASS_STYLE`` keys
on. One model therefore serves the whole side task (door state AND side damage)
in ONE pass, which is what the brief means by not loading both models for the
same side task: there is no second side model, so no detection is counted twice.
The ``damage`` class stays a DAMAGE finding and never becomes a door state --
that separation is the legacy code's own (``door_status`` is voted only over
``open_door`` / ``partially_closed`` / ``closed_door``).

Class names are resolved from the checkpoint at runtime and verified against the
expected set; numeric class ids are never assumed anywhere.

MEMORY
------
One camera, one model handle, one wagon window at a time. Model handles are
cached per weight file, so ``door_state.pt`` is loaded once for both side
cameras. Frames are read from the on-disk cache by the legacy code and released
per frame; no list of full-resolution images is ever accumulated. This is the
shape that avoids the OOM seen previously.

STATE IS PER TRAIN
------------------
``reset_inspection_state()`` clears the model cache and every accumulator, and
the frame cache is deleted by the caller between trains. Loading a second train
cannot reuse the first train's models, evidence, or association.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C
from core.global_state_loader import assert_roster_unchanged, roster_hash

from . import global_bridge as gb
from .wagon_cache import wagon_camera_dir

__all__ = [
    "LegacyInspectionConfig", "LegacyCameraResult", "LegacyInspectionResult",
    "run_legacy_inspection", "reset_inspection_state", "resolve_feature_model",
    "LocalArtifactSink", "EXPECTED_SIDE_CLASSES", "EXPECTED_TOP_CLASSES",
    "STATUS_INSPECTED", "STATUS_NO_DETECTION", "STATUS_NOT_VISIBLE",
    "STATUS_UNRESOLVED", "STATUS_AMBIGUOUS", "apply_to_unified",
]


# ---------------------------------------------------------------------------
# evidence-status vocabulary (ADDITIVE to the legacy schema)
# ---------------------------------------------------------------------------

STATUS_INSPECTED = "INSPECTED"
STATUS_NO_DETECTION = "NO_DETECTION"
STATUS_NOT_VISIBLE = "NOT_VISIBLE"
STATUS_UNRESOLVED = "UNRESOLVED"
STATUS_AMBIGUOUS = "AMBIGUOUS"
"""Kept distinct from an actual negative finding.

The legacy schema has no way to say "not seen": a wagon with no damage row gets
``door_status: "closed"`` and ``damage_detected: false``, identical to a wagon
that was inspected and found clean. Those legacy fields are preserved verbatim
for dashboard compatibility, and this vocabulary is exposed alongside them in the
additive ``inspection_status`` field so the distinction is not lost:

    INSPECTED     model ran on this wagon's frames and found something
    NO_DETECTION  model ran and found nothing -- a real negative finding
    NOT_VISIBLE   wagon lay outside this camera's footage; nothing was run
    UNRESOLVED    this camera's clock offset never resolved; nothing was run
    AMBIGUOUS     association was not safe enough to attribute a finding
"""

EXPECTED_SIDE_CLASSES = frozenset({"closed_door", "damage", "open_door",
                                   "partially_closed"})
EXPECTED_TOP_CLASSES = frozenset({"Floor__probable_damage", "Floor_damage",
                                  "Inner_wall_damage"})


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class LegacyInspectionConfig:
    """Orchestration knobs only.

    No detection threshold is redeclared here that the legacy code already owns.
    The two confidences below ARE legacy values (the camera YAMLs'
    ``inspection.damage_confidence``) and are surfaced because they are the one
    thing an operator legitimately tunes per site.
    """

    enabled: bool = True

    damage_confidence_side: float = 0.75
    damage_confidence_top: float = 0.75
    """RECOVERED from the legacy camera YAMLs (`inspection.damage_confidence`)."""

    band_gap_tolerance: int = 5
    """``DamageDetector`` default. Frames a class may vanish for and still be
    one band."""

    edge_skip_frames: int = 10
    """``DamageDetector`` default. Ignore the first/last N frames of a window,
    where the neighbouring vehicle is still in shot."""

    min_band_frames: int = 3
    """RECOVERED from the legacy YAMLs. THE FLICKER RULE: a damage/door band
    seen on fewer than 3 distinct frames is detector noise, not a finding. This
    is what stops one frame turning a wagon into an open/damaged wagon."""

    skip_non_wagon_segments: bool = True
    suppress_floor_damage_on_loaded: bool = True
    min_non_wagon_dominance: float = 0.80

    representative_positions: Tuple[float, ...] = (0.25, 0.55, 0.80)
    representative_position_names: Tuple[str, ...] = ("start", "mid1", "end")
    """The legacy wagon evidence contract: 25% / 55% / 80% -> start / mid1 / end."""

    side_model: str = C.MODEL_DOOR_STATE
    top_model: str = C.MODEL_DAMAGE
    """Local filename or ``s3://bucket/key``. Resolved through the legacy
    ``model_store``, so either form works and S3 objects are ETag-cached."""

    artifact_bucket: str = ""
    """Empty -> artifacts are written to a local directory instead of S3 and the
    URLs are ``file://``. Set to ``bucket`` or ``bucket/prefix`` to publish."""

    region: str = "ap-south-1"
    upload_to_s3: bool = False
    build_pdf: bool = True
    build_annotated_video: bool = False
    """Off by default: re-encoding four full videos is the most expensive step
    and is opt-in, exactly as overlay rendering already is in the counting run."""

    enable_ocr: bool = False
    """HARD OFF. Present so the disabled state is explicit in the config record
    rather than implied by an absent stage. Setting it True is refused."""

    version: str = "v4"
    """Legacy ``version`` field, unchanged so the dashboard's parser matches."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "damage_confidence_side": self.damage_confidence_side,
            "damage_confidence_top": self.damage_confidence_top,
            "band_gap_tolerance": self.band_gap_tolerance,
            "edge_skip_frames": self.edge_skip_frames,
            "min_band_frames": self.min_band_frames,
            "skip_non_wagon_segments": self.skip_non_wagon_segments,
            "suppress_floor_damage_on_loaded": self.suppress_floor_damage_on_loaded,
            "min_non_wagon_dominance": self.min_non_wagon_dominance,
            "representative_positions": list(self.representative_positions),
            "representative_position_names": list(self.representative_position_names),
            "models": {"side": self.side_model, "top": self.top_model,
                       "side_authoritative_for": "door_state + side damage",
                       "top_authoritative_for": "floor / inner-wall damage"},
            "artifact_bucket": self.artifact_bucket,
            "upload_to_s3": self.upload_to_s3,
            "build_pdf": self.build_pdf,
            "build_annotated_video": self.build_annotated_video,
            "ocr": "DISABLED",
            "version": self.version,
        }


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass
class LegacyCameraResult:
    camera_id: str
    flavour: str
    status: str = STATUS_UNRESOLVED
    json_path: Optional[str] = None
    pdf_path: Optional[str] = None
    annotated_video_path: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    wagons_in_json: int = 0
    wagons_scanned: int = 0
    problem_frames: int = 0
    warnings: List[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id, "flavour": self.flavour,
            "status": self.status, "json_path": self.json_path,
            "pdf_path": self.pdf_path,
            "annotated_video_path": self.annotated_video_path,
            "wagons_in_json": self.wagons_in_json,
            "wagons_scanned": self.wagons_scanned,
            "problem_frames": self.problem_frames,
            "warnings": list(self.warnings),
            "seconds": round(self.seconds, 2),
        }


@dataclass
class LegacyInspectionResult:
    cameras: Dict[str, LegacyCameraResult] = field(default_factory=dict)
    model_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    global_wagon_count: int = 0
    roster_hash_before: str = ""
    roster_hash_after: str = ""
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    output_dir: str = ""

    @property
    def roster_unchanged(self) -> bool:
        return bool(self.roster_hash_before) and \
            self.roster_hash_before == self.roster_hash_after

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": "legacy Train-Inspection-Engine modules, run unchanged",
            "counting_authority": "current global wagon counting pipeline",
            "ocr": "DISABLED",
            "global_wagon_count": self.global_wagon_count,
            "cameras": {c: r.to_dict() for c, r in sorted(self.cameras.items())},
            "model_status": dict(self.model_status),
            "warnings": list(self.warnings),
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
            "output_dir": self.output_dir,
            "roster_hash_before": self.roster_hash_before,
            "roster_hash_after": self.roster_hash_after,
            "roster_unchanged": self.roster_unchanged,
        }


# ---------------------------------------------------------------------------
# per-train state (models + caches). Cleared between trains.
# ---------------------------------------------------------------------------

_MODEL_HANDLES: Dict[str, Any] = {}


def reset_inspection_state() -> None:
    """Drop every cached model handle.

    Called at the START of each run, not the end, so a crashed previous run
    cannot leave a handle behind for the next train to inherit. Weights are
    stateless, but the handle is not the only thing that could leak, and making
    "train B reuses nothing from train A" a single explicit call is cheaper than
    reasoning about it per object.
    """
    _MODEL_HANDLES.clear()


def resolve_feature_model(
    spec: str,
    models_dir: str,
    *,
    s3_client: Any = None,
    cache_dir: Optional[str] = None,
    logger: Any = None,
) -> str:
    """Resolve a model spec to a local path.

    Accepts, in order:
      * ``s3://bucket/key``  -> legacy ``model_store.resolve_path`` (ETag-cached)
      * an existing path      -> used as-is
      * a bare filename       -> looked up in ``models_dir`` through
        ``core.constants.resolve_model_path``, so either the old or the new
        naming convention resolves.
    """
    import logging

    from .legacy.model_store import is_remote_uri, resolve_path

    log = logger or logging.getLogger(__name__)
    if is_remote_uri(spec):
        return resolve_path(spec, s3_client=s3_client, cache_dir=cache_dir,
                            logger=log)
    if os.path.isfile(spec):
        return spec
    return C.resolve_model_path(models_dir, spec)


def _load_model(path: str) -> Any:
    """One handle per weight file, reused across cameras of the same flavour."""
    key = os.path.abspath(path)
    handle = _MODEL_HANDLES.get(key)
    if handle is None:
        from ultralytics import YOLO
        handle = YOLO(path)
        _MODEL_HANDLES[key] = handle
    return handle


def _class_names(model: Any) -> Dict[int, str]:
    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {int(i): str(v) for i, v in enumerate(names)}


def _discover_models(cfg: LegacyInspectionConfig, models_dir: str,
                     s3_client: Any, verbose: bool) -> Dict[str, Dict[str, Any]]:
    """Resolve + load both damage models and VERIFY their real class names.

    Class names are read from the checkpoint, never assumed. A model whose
    classes do not match the flavour it was wired to is reported and disabled
    rather than run -- a side model silently used for top damage would map every
    class to None and report a clean train.
    """
    out: Dict[str, Dict[str, Any]] = {}
    roles = (("side", cfg.side_model, EXPECTED_SIDE_CLASSES),
             ("top", cfg.top_model, EXPECTED_TOP_CLASSES))
    for role, spec, expected in roles:
        rec: Dict[str, Any] = {"role": role, "spec": spec,
                               "authoritative": True}
        try:
            path = resolve_feature_model(spec, models_dir, s3_client=s3_client)
            rec["path"] = path
            if not os.path.isfile(path):
                rec["status"] = "UNAVAILABLE"
                rec["reason"] = (
                    f"{path} not found; {role} damage/door reports "
                    f"{STATUS_NOT_VISIBLE} for every wagon and the wagon count "
                    f"is unaffected")
                out[role] = rec
                continue
            model = _load_model(path)
            names = _class_names(model)
            rec["class_names"] = names
            rec["task"] = str(getattr(model, "task", "") or "")
            found = set(names.values())
            rec["expected_class_names"] = sorted(expected)
            missing = sorted(expected - found)
            extra = sorted(found - expected)
            rec["status"] = "AVAILABLE"
            if missing:
                rec["status"] = "CLASS_MISMATCH"
                rec["reason"] = (
                    f"{os.path.basename(path)} does not expose {missing}; it is "
                    f"not a {role} model. Disabled rather than run, because an "
                    f"unmatched class set silently reports a clean train.")
            if extra:
                rec["unexpected_class_names"] = extra
        except Exception as exc:                       # noqa: BLE001
            rec["status"] = "UNAVAILABLE"
            rec["reason"] = f"failed to load: {type(exc).__name__}: {exc}"
        out[role] = rec

    if verbose:
        print("-" * 70)
        print("  LEGACY INSPECTION MODELS  (classes verified at runtime; OCR disabled)")
        print("-" * 70)
        for role, rec in out.items():
            print(f"  {role.upper():<5} {rec.get('status', '?'):<14} "
                  f"{rec.get('path', rec.get('spec'))}")
            if rec.get("class_names"):
                print("        classes: " + ", ".join(
                    f"{k}:{v}" for k, v in sorted(rec["class_names"].items())))
            if rec.get("reason"):
                print(f"        {rec['reason']}")
    return out


# ---------------------------------------------------------------------------
# artifact sink
# ---------------------------------------------------------------------------

class LocalArtifactSink:
    """Local stand-in for ``S3Client`` used when no bucket is configured.

    ``ArtifactPublisher`` only ever calls ``self.s3.client.upload_file(src,
    bucket, key)``, so a stub with that one method lets the legacy publisher run
    verbatim -- with its filename patterns, its directory layout and its
    annotated-or-raw problem-frame rule intact -- while writing to disk. That
    keeps local tests and EC2 dry-runs honest: the same code produces the same
    layout whether or not S3 is reachable.
    """

    def __init__(self, root: str):
        self.root = root
        self.client = self
        self.uploads: List[Tuple[str, str]] = []

    def upload_file(self, local_path: str, bucket: str, key: str) -> str:
        dest = os.path.join(self.root, bucket, key.replace("/", os.sep))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(local_path, dest)
        self.uploads.append((bucket, key))
        return key


def _make_publisher(cfg: LegacyInspectionConfig, profile: gb.CameraProfile,
                    local_root: str, s3_client: Any):
    from .legacy.artifacts import ArtifactPublisher

    if cfg.upload_to_s3 and cfg.artifact_bucket and s3_client is not None:
        sink, bucket = s3_client, cfg.artifact_bucket
    else:
        sink = LocalArtifactSink(os.path.join(local_root, "artifacts"))
        bucket = cfg.artifact_bucket or "local-artifacts"
    return ArtifactPublisher(
        s3=sink, artifact_bucket=bucket, region=cfg.region,
        camera_folder=profile.folder, damage_flavour=profile.flavour,
        representative_positions=tuple(cfg.representative_positions),
        representative_position_names=tuple(cfg.representative_position_names),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _empty_damage_df(flavour: str):
    from .legacy.damage import DamageDetector
    probe = DamageDetector(damage_model=None, flavour=flavour, confidence=0.0)
    return probe._empty_top_df() if flavour == "top" else probe._empty_side_df()


def _empty_frame_detections_df():
    from .legacy.damage import DamageDetector
    return DamageDetector._empty_frame_detections_df()


def _raw_video_name(tracks_by_camera: Dict[str, Any], camera_id: str) -> str:
    lct = tracks_by_camera.get(camera_id)
    path = str(getattr(lct, "video_path", "") or "")
    return os.path.splitext(os.path.basename(path))[0] if path else camera_id


def _direction_for(state: Any, profile: gb.CameraProfile) -> str:
    """Travel direction reported to the legacy JSON/PDF.

    The current counting pipeline does not publish a visual travel direction --
    it never needed one -- so the camera's configured ``loaded_direction`` is
    reported unless the state carries an explicit override. Direction only feeds
    the side flavour's ``rake_status`` and the PDF's rake banner; it takes no
    part in counting, association or any verdict.
    """
    for attr in ("travel_direction", "direction"):
        value = getattr(state, attr, None)
        if isinstance(value, str) and value in ("left-to-right", "right-to-left"):
            return value
        if isinstance(value, dict):
            per_cam = value.get(profile.camera_id)
            if per_cam in ("left-to-right", "right-to-left"):
                return per_cam
    return profile.loaded_direction


def _evidence_status(camera_status: str, has_window: bool,
                     scanned: bool, found: bool) -> str:
    if camera_status == "UNRESOLVED":
        return STATUS_UNRESOLVED
    if camera_status != "RESOLVED" or not has_window:
        return STATUS_NOT_VISIBLE
    if not scanned:
        return STATUS_NOT_VISIBLE
    return STATUS_INSPECTED if found else STATUS_NO_DETECTION


# ---------------------------------------------------------------------------
# per-camera run
# ---------------------------------------------------------------------------

def _run_camera(
    *,
    state: Any,
    camera_id: str,
    cfg: LegacyInspectionConfig,
    plan: Any,
    cache_root: str,
    models: Dict[str, Dict[str, Any]],
    tracks_by_camera: Dict[str, Any],
    load_status_by_wagon: Dict[str, str],
    output_dir: str,
    s3_client: Any,
    upload_timestamp: datetime,
    verbose: bool,
) -> LegacyCameraResult:
    """One camera, start to finish. Nothing here is shared with another camera
    except the model handle, which is stateless."""
    import pandas as pd

    from .legacy.damage import DamageDetector, ProblemFrameExtractor
    from .legacy.json_builder import build_inspection_json

    profile = gb.camera_profile(camera_id)
    res = LegacyCameraResult(camera_id=camera_id, flavour=profile.flavour)
    t0 = time.time()

    cam_status = (plan.camera_status or {}).get(camera_id, "UNRESOLVED")
    res.status = {"RESOLVED": STATUS_INSPECTED}.get(cam_status, cam_status)
    windows = [w for w in plan.windows if w.camera_id == camera_id]

    work_dir = os.path.join(output_dir, profile.legacy_name)
    os.makedirs(work_dir, exist_ok=True)

    # ---- the two DataFrames (see global_bridge.build_segment_summary) -------
    full_df = gb.build_segment_summary(
        state, camera_id, windows, cache_root,
        load_status_by_wagon=load_status_by_wagon,
        require_frames=False, include_unwindowed=True)
    scan_df = gb.build_segment_summary(
        state, camera_id, windows, cache_root,
        load_status_by_wagon=load_status_by_wagon,
        require_frames=True, include_unwindowed=False)
    res.wagons_scanned = int(len(scan_df))

    wagon_count_map = gb.build_wagon_count_map(full_df)
    gb.assert_wagon_count_map_is_global(state, wagon_count_map, camera_id)
    segment_type_map = gb.build_segment_type_map(full_df, profile.flavour)

    # ---- detection: the LEGACY detector, unchanged --------------------------
    model_rec = models.get(profile.flavour, {})
    model_ok = model_rec.get("status") == "AVAILABLE"
    damage_conf = (cfg.damage_confidence_side if profile.is_side
                   else cfg.damage_confidence_top)

    if model_ok and len(scan_df):
        detector = DamageDetector(
            damage_model=_load_model(model_rec["path"]),
            flavour=profile.flavour,
            confidence=damage_conf,
            band_gap_tolerance=cfg.band_gap_tolerance,
            edge_skip_frames=cfg.edge_skip_frames,
            skip_non_wagon_segments=cfg.skip_non_wagon_segments,
            suppress_floor_damage_on_loaded=cfg.suppress_floor_damage_on_loaded,
            min_band_frames=cfg.min_band_frames,
            min_non_wagon_dominance=cfg.min_non_wagon_dominance,
        )
        damage_df, frame_detections_df = detector.detect(scan_df)
        extractor = ProblemFrameExtractor(
            damage_model=_load_model(model_rec["path"]),
            flavour=profile.flavour, damage_confidence=damage_conf)
        problem_frames_df = extractor.extract(scan_df, damage_df, work_dir)
    else:
        if not model_ok:
            res.warnings.append(
                f"{profile.flavour} model unavailable "
                f"({model_rec.get('reason', 'not configured')}) -- every wagon "
                f"reports {STATUS_NOT_VISIBLE} for this feature; the wagon "
                f"count is unaffected")
        damage_df = _empty_damage_df(profile.flavour)
        frame_detections_df = _empty_frame_detections_df()
        problem_frames_df = pd.DataFrame(columns=[
            "wagon_id", "problem_type", "frame_number", "frame_path",
            "annotated_image_path", "bounding_box"])

    # ---- evidence + artifacts: the LEGACY publisher, unchanged -------------
    loco_summary_df = pd.DataFrame()   # loco evidence needs OCR-era loco bands
    publisher = _make_publisher(cfg, profile, output_dir, s3_client)
    timestamp_str, wagon_frames_index, loco_frames, problem_frames = publisher.publish(
        upload_timestamp=upload_timestamp,
        segment_summary_df=full_df,
        loco_summary_df=loco_summary_df,
        problem_frames_df=problem_frames_df,
        wagon_count_map=wagon_count_map,
        local_workdir=work_dir,
    )
    res.problem_frames = len(problem_frames)

    # ---- JSON: the LEGACY builder, unchanged -------------------------------
    payload = build_inspection_json(
        camera_folder=profile.folder,
        raw_video_name=_raw_video_name(tracks_by_camera, camera_id),
        upload_timestamp=upload_timestamp,
        direction=_direction_for(state, profile),
        flavour=profile.flavour,
        segment_summary_df=full_df,
        damage_results_df=damage_df,
        loco_summary_df=loco_summary_df,
        problem_frames_df=problem_frames_df,
        wagon_frames_index=wagon_frames_index,
        loco_frame_entries=loco_frames,
        problem_frame_entries=problem_frames,
        wagon_count_map=wagon_count_map,
        segment_type_map=segment_type_map,
        # OCR is disabled: these stay None so the legacy builder emits the empty
        # dicts the dashboard expects, rather than the fields being removed.
        wagon_number_results=None,
        loco_numbers=None,
        damage_model_active=model_ok,
        version=cfg.version,
    )

    # The set of wagons this camera genuinely has a frame window for. Passed
    # explicitly rather than inferred from the DataFrame, because an unwindowed
    # row and a window that legitimately starts at frame 0 are indistinguishable
    # once they are both rows.
    by_gid_index: Dict[str, int] = {}
    for wagon in gb._iter_roster(state):
        gid, index, _ = gb._wagon_fields(wagon)
        if gid is not None and index is not None:
            by_gid_index[str(gid)] = int(index)
    windowed_indices = {by_gid_index[str(w.global_id)] for w in windows
                        if str(w.global_id) in by_gid_index}

    _annotate_globally(payload, state, scan_df, damage_df, cam_status, profile,
                       windowed_indices)

    res.payload = payload
    res.wagons_in_json = len(payload["inspection_data"].get("wagon_segments", []))

    json_path = os.path.join(work_dir, "inspection_data.json")
    _write_json(json_path, payload)
    res.json_path = json_path
    if cfg.upload_to_s3 and cfg.artifact_bucket and s3_client is not None:
        publisher.upload_inspection_json(timestamp_str, payload, work_dir)

    # ---- persisted inputs for the renderers (NO model runs downstream) -----
    _persist_frames(work_dir, full_df, damage_df, problem_frames_df,
                    frame_detections_df)

    res.seconds = time.time() - t0
    if verbose:
        print(f"    [{camera_id}] {res.wagons_in_json} wagon(s) in JSON, "
              f"{res.wagons_scanned} scanned, {res.problem_frames} problem "
              f"frame(s), {res.seconds:.1f}s")
    return res


def _annotate_globally(payload: Dict[str, Any], state: Any, scan_df, damage_df,
                       cam_status: str, profile: gb.CameraProfile,
                       windowed_indices: Optional[set] = None) -> None:
    """Attach the ADDITIVE global-identity and evidence-status fields.

    Everything added here is new-key-only. No legacy key is renamed, removed or
    retyped, so a dashboard written against the old schema reads this payload
    unchanged and simply ignores the extras.

        wagon_segments[i].global_wagon_id   "GW_17"
        wagon_segments[i].inspection_status INSPECTED | NO_DETECTION | ...
        problem_frames[i].global_wagon_id   "GW_17"
        inspection_data.counting_source     provenance marker
        inspection_data.global_wagon_count  the roster's count
    """
    data = payload.get("inspection_data", {})
    by_index: Dict[int, str] = {}
    for wagon in gb._iter_roster(state):
        gid, index, _ = gb._wagon_fields(wagon)
        if gid is not None and index is not None:
            by_index[int(index)] = str(gid)

    scanned_ids = {int(s) for s in scan_df["segment_id"]} if len(scan_df) else set()
    windowed_ids = set(windowed_indices or ())
    damaged_ids: set = set()
    if damage_df is not None and len(damage_df) and "wagon_id" in damage_df.columns:
        for _, row in damage_df.iterrows():
            flags = [bool(row.get(k)) for k in row.index if str(k).endswith("_detected")]
            door = str(row.get("door_status", "closed"))
            if any(flags) or door != "closed":
                damaged_ids.add(int(row["wagon_id"]))

    for seg in data.get("wagon_segments", []):
        seg_id = seg.get("segment_id")
        if seg_id is None:
            continue
        seg_id = int(seg_id)
        seg["global_wagon_id"] = by_index.get(seg_id)
        seg["inspection_status"] = _evidence_status(
            cam_status,
            has_window=seg_id in windowed_ids,
            scanned=seg_id in scanned_ids,
            found=seg_id in damaged_ids)

    for pf in data.get("problem_frames", []):
        count = pf.get("wagon_count")
        if count is not None:
            pf["global_wagon_id"] = by_index.get(int(count))

    data["counting_source"] = "global_wagon_counting_v1"
    data["global_wagon_count"] = len(by_index)
    data["ocr_enabled"] = False
    data["camera_role"] = profile.camera_id


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    from .legacy.serialization import json_safe
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=json_safe)


def _persist_frames(work_dir: str, full_df, damage_df, problem_frames_df,
                    frame_detections_df) -> None:
    """Write the DataFrames the renderers consume.

    The PDF and the annotated video are built from THESE files, never from a
    fresh model pass, which is what guarantees the PDF, the video and the JSON
    all describe one set of detections.
    """
    for name, df in (("segments.csv", full_df),
                     ("damage_results.csv", damage_df),
                     ("problem_frames.csv", problem_frames_df),
                     ("frame_detections.csv", frame_detections_df)):
        try:
            df.to_csv(os.path.join(work_dir, name), index=False)
        except Exception:                              # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# reconciliation: one set of verdicts behind every artifact
# ---------------------------------------------------------------------------

_DOOR_STATUS_TO_STATE = {
    "open": C.DOOR_OPEN,
    "partially_closed": C.DOOR_PARTIAL,
    "closed": C.DOOR_CLOSED,
}
"""Legacy ``door_status`` -> the vocabulary the combined report reads.

RECOVERED on both sides: the left column is ``damage._DOOR_STATUS_LABEL``'s
value set, the right is ``core.constants``' door states. There is no third
spelling anywhere.
"""


def apply_to_unified(unified: Dict[str, Any],
                     result: "LegacyInspectionResult") -> Dict[str, int]:
    """Fold the legacy per-camera verdicts into the fused per-wagon view.

    WHY THIS EXISTS
    ---------------
    Two documents describe the same train: the dashboard's per-camera
    ``inspection_data.json`` and the combined report built from
    ``UnifiedWagonState``. If the door verdict were computed for one and left
    absent from the other, the PDF would say NO_DATA where the JSON says "open"
    -- two artifacts of one run disagreeing about a wagon. Folding the results
    across means there is ONE verdict per wagon, rendered twice.

    Camera authority is the old_code convention, unchanged:
    ``RIGHT_UP -> right_door``, ``LEFT_UP -> left_door``. Top cameras carry
    damage. A wagon a camera never saw stays ``NO_DATA`` -- never CLOSED, never
    OK -- so an uninspected wagon can still not be presented as a clean one.

    Returns per-feature counts of how many wagons were updated, for the log.
    """
    side_by_camera = {C.CAMERA_RIGHT_UP: "right", C.CAMERA_LEFT_UP: "left"}
    applied = {"door": 0, "side_damage": 0, "top_damage": 0}

    for camera_id, cam_res in (result.cameras or {}).items():
        segments = (cam_res.payload or {}).get(
            "inspection_data", {}).get("wagon_segments") or []
        for seg in segments:
            gid = seg.get("global_wagon_id")
            state_obj = unified.get(gid) if gid else None
            if state_obj is None:
                continue
            inspected = seg.get("inspection_status") in (
                STATUS_INSPECTED, STATUS_NO_DETECTION)
            if not inspected:
                continue

            if camera_id in side_by_camera:
                side = side_by_camera[camera_id]
                door_state = _DOOR_STATUS_TO_STATE.get(
                    str(seg.get("door_status") or "closed"), C.NO_DATA)
                setattr(state_obj, f"{side}_door", door_state)
                applied["door"] += 1
                if seg.get("damage_detected"):
                    state_obj.side_damage = C.DAMAGE_PRESENT
                    applied["side_damage"] += 1
                elif state_obj.side_damage == C.NO_DATA:
                    state_obj.side_damage = C.DAMAGE_OK
            else:
                if seg.get("damage_detected"):
                    state_obj.top_damage = C.DAMAGE_PRESENT
                    applied["top_damage"] += 1
                elif state_obj.top_damage == C.NO_DATA:
                    state_obj.top_damage = C.DAMAGE_OK

            if camera_id not in state_obj.supporting_cameras:
                state_obj.supporting_cameras.append(camera_id)
    return applied


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def run_legacy_inspection(
    state: Any,
    tracks_by_camera: Dict[str, Any],
    plan: Any,
    models_dir: str,
    output_root: str,
    cfg: Optional[LegacyInspectionConfig] = None,
    load_status_by_wagon: Optional[Dict[str, str]] = None,
    s3_client: Any = None,
    cameras: Optional[Sequence[str]] = None,
    upload_timestamp: Optional[datetime] = None,
    verbose: bool = True,
) -> LegacyInspectionResult:
    """Produce the legacy inspection outputs for every camera.

    ``state`` is READ-ONLY, and that is checked rather than trusted: the roster
    is hashed before anything runs and re-verified in a ``finally``, so the
    guarantee holds even if a feature raises.
    """
    cfg = cfg or LegacyInspectionConfig()
    if cfg.enable_ocr:
        raise ValueError(
            "OCR is out of scope for this pipeline and cannot be enabled. The "
            "legacy wagon_number_results / loco_number_results fields are still "
            "emitted, as empty dicts, so the dashboard schema is unchanged.")

    res = LegacyInspectionResult(output_dir=output_root)
    res.roster_hash_before = roster_hash(state)
    res.global_wagon_count = len(gb._iter_roster(state))

    try:
        if not cfg.enabled:
            res.warnings.append("legacy inspection disabled by configuration")
            return res

        reset_inspection_state()          # train B inherits nothing from train A
        os.makedirs(output_root, exist_ok=True)
        upload_timestamp = upload_timestamp or datetime.now()
        load_status_by_wagon = dict(load_status_by_wagon or {})
        cache_root = getattr(plan, "root", "")

        t0 = time.time()
        res.model_status = _discover_models(cfg, models_dir, s3_client, verbose)
        res.timings["model_load_seconds"] = time.time() - t0

        for camera_id in (cameras or C.ALL_CAMERAS):
            t = time.time()
            try:
                res.cameras[camera_id] = _run_camera(
                    state=state, camera_id=camera_id, cfg=cfg, plan=plan,
                    cache_root=cache_root, models=res.model_status,
                    tracks_by_camera=tracks_by_camera,
                    load_status_by_wagon=load_status_by_wagon,
                    output_dir=output_root, s3_client=s3_client,
                    upload_timestamp=upload_timestamp, verbose=verbose)
            except Exception as exc:                   # noqa: BLE001
                # One camera failing must not cost the other three their output,
                # and must never cost the run its wagon count.
                msg = f"{camera_id}: {type(exc).__name__}: {exc}"
                res.warnings.append(msg)
                failed = LegacyCameraResult(
                    camera_id=camera_id,
                    flavour=gb.CAMERA_PROFILES[camera_id].flavour
                    if camera_id in gb.CAMERA_PROFILES else "",
                    status=STATUS_UNRESOLVED)
                failed.warnings.append(msg)
                failed.seconds = time.time() - t
                res.cameras[camera_id] = failed
                if verbose:
                    print(f"    [{camera_id}] FAILED: {msg}")
        return res
    finally:
        # Runs even if a feature raised. This is the checked guarantee that
        # inspection cannot alter the finalized global wagon structure.
        res.roster_hash_after = roster_hash(state)
        assert_roster_unchanged(state, res.roster_hash_before)
