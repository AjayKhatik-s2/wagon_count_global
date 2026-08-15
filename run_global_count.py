"""
run_global_count.py  --  Wagon Eye Phase-1 (standalone)
========================================================

Self-contained entry point for AWS / SageMaker / EC2 deployment.

CONVENTIONS (just-drop-files-and-run)
-------------------------------------
Place the 4 trimmed train videos in ./inputs/ with these exact names:
    inputs/right_up.mp4
    inputs/left_up.mp4
    inputs/right_up_top.mp4
    inputs/left_up_top.mp4

Place the 4 YOLO model weights in ./models/ with these exact names:
    models/right_up_wagon_gap.pt     (used by RIGHT_UP -- master)
    models/left_up_wagon_gap.pt      (used by LEFT_UP)
    models/top_gap.pt                (used by RIGHT_UP_TOP and LEFT_UP_TOP)
    models/side_classification.pt    (used by RIGHT_UP for ENGINE/WAGON/BRAKE_VAN)

Then run:
    python run_global_count.py

Outputs land in ./results/ (configurable with --output).

OVERRIDES
---------
You can override any path explicitly:
    python run_global_count.py \
        --right_up      /abs/path/cam_right_up.mp4 \
        --left_up       /abs/path/cam_left_up.mp4 \
        --right_up_top  /abs/path/cam_right_up_top.mp4 \
        --left_up_top   /abs/path/cam_left_up_top.mp4 \
        --models-dir    /abs/path/models \
        --output        /abs/path/results

WHAT THIS PRODUCES
------------------
    results/
        global_train_state.json          <-- canonical Phase-1 output
        per_camera_tracking.json
        combined_report.pdf              <-- combined evidence report

The report holds, for every global event (GW_n), up to 16 representative
evidence frames: 20% / 40% / 60% / 80% through each camera's own valid
evidence interval, for each of RIGHT_UP, LEFT_UP, RIGHT_UP_TOP and
LEFT_UP_TOP.  Full per-frame sequences are NOT written to disk any more --
the frames are extracted to a temporary directory and deleted once the PDF
has been written and verified.

Overlay videos are opt-in (--render-videos) because they are purely visual
evidence and cost hundreds of MB per run.

WHAT THIS DOES NOT DO
---------------------
No door / damage / OCR detection.  No PDF or email.  No S3 upload.
Phase-1 is gap counting + global synchronization + classification only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import List, Dict, Optional

from global_train_state import (
    GlobalTrainState,
    LocalCameraTracks,
    SegmentClass,
    MASTER_CAMERA,
    ALL_CAMERAS,
    CAMERA_LEFT_UP,
    CAMERA_RIGHT_UP,
    CAMERA_RIGHT_UP_TOP,
    CAMERA_LEFT_UP_TOP,
    summarize_state,
)
from tracker_engine import GapTracker, MasterClassifier, segments_from_gaps
import global_alignment as ga
import global_fusion as gf
import gap_validation as gval
import fragment_stitching as fstitch
import train_structure as ts
import temporal_classification as tcls
import video_segmenter as vs
import evidence_report as er
from inspection import old_features as oldf
from inspection import old_report as oldr
from inspection import wagon_cache as iwc
from inspection import legacy_inspection as legi
from inspection import legacy_render as legr
from core import constants as C
from core import global_state_loader as core_state


# =============================================================================
# Auto-discovery: default file conventions
# =============================================================================

DEFAULT_INPUT_FILENAMES = {
    CAMERA_RIGHT_UP:     "right_up.mp4",
    CAMERA_LEFT_UP:      "left_up.mp4",
    CAMERA_RIGHT_UP_TOP: "right_up_top.mp4",
    CAMERA_LEFT_UP_TOP:  "left_up_top.mp4",
}

# Some users may use these alternative names; we'll fall back to them.
_INPUT_FALLBACK_PATTERNS = {
    CAMERA_RIGHT_UP:     ["right_up.mp4", "RIGHT_UP.mp4", "cam_right_up.mp4"],
    CAMERA_LEFT_UP:      ["left_up.mp4", "LEFT_UP.mp4", "cam_left_up.mp4"],
    CAMERA_RIGHT_UP_TOP: ["right_up_top.mp4", "RIGHT_UP_TOP.mp4", "cam_right_up_top.mp4"],
    CAMERA_LEFT_UP_TOP:  ["left_up_top.mp4", "LEFT_UP_TOP.mp4", "cam_left_up_top.mp4"],
}


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_input(explicit: Optional[str], inputs_dir: str, camera_id: str) -> str:
    """Return a path to the camera's video.

    Search order:
        1. explicit (if provided)
        2. inputs_dir/<filename>  for each fallback name
    """
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"--{camera_id.lower()} path does not exist: {explicit}")
        return os.path.abspath(explicit)

    for name in _INPUT_FALLBACK_PATTERNS[camera_id]:
        p = os.path.join(inputs_dir, name)
        if os.path.exists(p):
            return os.path.abspath(p)

    raise FileNotFoundError(
        f"No input video found for {camera_id}. "
        f"Looked in {inputs_dir} for: "
        f"{_INPUT_FALLBACK_PATTERNS[camera_id]}. "
        f"Either drop the file there or pass --{camera_id.lower()} <path>."
    )


def _resolve_model(name: str, models_dir: str) -> str:
    p = os.path.join(models_dir, name)
    if os.path.exists(p):
        return os.path.abspath(p)
    raise FileNotFoundError(
        f"Model not found: {name}. Expected at {p}. "
        f"Drop the .pt file in {models_dir} or pass --models-dir <path>."
    )


# =============================================================================
# Per-camera processing
# =============================================================================

def _process_side_camera(
    camera_id: str, video_path: str, gap_model_path: str,
    confidence: float, min_height_ratio: float,
    keep_raw_detections: bool, verbose: bool,
) -> LocalCameraTracks:
    tracker = GapTracker(
        camera_id=camera_id, model_path=gap_model_path,
        confidence=confidence, min_height_ratio=min_height_ratio,
        verbose=verbose,
    )
    return tracker.process_video(video_path, keep_raw_detections=keep_raw_detections)


def _process_top_camera(
    camera_id: str, video_path: str, top_gap_model_path: str,
    confidence: float, min_height_ratio: float,
    keep_raw_detections: bool, verbose: bool,
) -> LocalCameraTracks:
    tracker = GapTracker(
        camera_id=camera_id, model_path=top_gap_model_path,
        confidence=confidence, min_height_ratio=min_height_ratio,
        verbose=verbose,
    )
    return tracker.process_video(video_path, keep_raw_detections=keep_raw_detections)


def _classify_master_pre_fusion(
    master_tracks: LocalCameraTracks,
    side_classification_model_path: str,
    num_samples: int,
    verbose: bool,
):
    pre_segments = segments_from_gaps(master_tracks.gaps, master_tracks.total_frames)
    if not pre_segments:
        if verbose:
            print("[CLASSIFY] no pre-fusion segments to classify")
        return []
    if verbose:
        print(f"[CLASSIFY] classifying {len(pre_segments)} pre-fusion segments on "
              f"{os.path.basename(master_tracks.video_path)}")
    clf = MasterClassifier(side_classification_model_path, num_samples=num_samples, verbose=verbose)
    return clf.classify_segments(master_tracks.video_path, pre_segments)


# =============================================================================
# CLI
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    here = _here()
    default_inputs = os.path.join(here, "inputs")
    default_models = os.path.join(here, "models")
    default_output = os.path.join(here, "results")

    p = argparse.ArgumentParser(
        prog="run_global_count.py",
        description="Phase-1 global wagon counting + classification (standalone).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Default file conventions:\n"
            "  inputs/{right_up,left_up,right_up_top,left_up_top}.mp4\n"
            "  models/right_up_wagon_gap.pt   (RIGHT_UP -- master)\n"
            "  models/left_up_wagon_gap.pt    (LEFT_UP)\n"
            "  models/top_gap.pt              (RIGHT_UP_TOP + LEFT_UP_TOP)\n"
            "  models/side_classification.pt  (RIGHT_UP classification)\n"
            "Drop the files in ./inputs and ./models, then run with no args."
        ),
    )

    p.add_argument("--right_up",     default=None, help="Override path to RIGHT_UP video (master)")
    p.add_argument("--left_up",      default=None, help="Override path to LEFT_UP video")
    p.add_argument("--right_up_top", default=None, help="Override path to RIGHT_UP_TOP video")
    p.add_argument("--left_up_top",  default=None, help="Override path to LEFT_UP_TOP video")

    p.add_argument("--inputs-dir",   default=default_inputs,
                   help=f"Directory containing the 4 input videos (default: {default_inputs})")
    p.add_argument("--models-dir",   default=default_models,
                   help=f"Directory containing the 4 .pt models (default: {default_models})")
    p.add_argument("--output", "-o", default=default_output,
                   help=f"Output root directory (default: {default_output})")

    p.add_argument("--side-confidence", type=float, default=0.4,
                   help="Confidence threshold for the side gap models "
                        "right_up_wagon_gap.pt and left_up_wagon_gap.pt "
                        "(default: 0.4)")
    p.add_argument("--top-confidence",  type=float, default=0.4,
                   help="Confidence threshold for top_gap.pt (default: 0.4)")
    p.add_argument("--side-min-height-ratio", type=float, default=0.35,
                   help="Min bbox height / frame height for SIDE gap detections "
                        "(default: 0.35). Tall gaps are typical on side cameras.")
    p.add_argument("--top-min-height-ratio",  type=float, default=0.05,
                   help="Min bbox height / frame height for TOP gap detections "
                        "(default: 0.05). Top-camera gaps are thin horizontal "
                        "strips, so this MUST be much smaller than the side ratio.")
    p.add_argument("--classification-samples", type=int, default=5,
                   help="Frames per segment for side_classification.pt vote (default: 5)")

    # ---- inspection: DOWNSTREAM features on the finished global train --------
    p.add_argument("--no-inspection", action="store_true",
                   help="Skip door/top-damage inspection entirely. The wagon "
                        "count is unaffected either way.")
    p.add_argument("--door-model", default=None,
                   help="Path to door_state.pt (default: <models-dir>/door_state.pt)")
    p.add_argument("--damage-model", default=None,
                   help="Path to top_damage.pt (default: <models-dir>/top_damage.pt)")
    p.add_argument("--no-door", action="store_true",
                   help="Skip the door feature (old_code DoorTracker path)")
    p.add_argument("--no-load", action="store_true",
                   help="Skip the load feature (needs load.pt; supplied on EC2)")
    p.add_argument("--no-damage", action="store_true",
                   help="Skip the top-damage feature (old_code DamageTracker path)")
    p.add_argument("--keep-wagon-cache", action="store_true",
                   help="Keep wagon_cache/ after inspection. Off by default: the "
                        "cache is per-train state and must not be read by the "
                        "next train.")
    p.add_argument("--wagon-cache-every-nth", type=int,
                   default=iwc.WagonCacheConfig().every_nth,
                   help="Cache every Nth frame inside each wagon window "
                        f"(default {iwc.WagonCacheConfig().every_nth})")
    p.add_argument("--wagon-cache-max-frames", type=int,
                   default=iwc.WagonCacheConfig().max_frames_per_wagon,
                   help="Ceiling on cached frames per (wagon, camera) "
                        f"(default {iwc.WagonCacheConfig().max_frames_per_wagon})")

    # ---- legacy inspection outputs: dashboard JSON / PDF / artifacts ---------
    #
    # This is the ported Train-Inspection-Engine output layer. It consumes the
    # FINALIZED roster and emits the legacy per-camera inspection_data.json, the
    # evidence artifacts and the legacy PDFs. It cannot affect the wagon count.
    p.add_argument("--no-legacy-inspection", action="store_true",
                   help="Skip the legacy inspection output layer (per-camera "
                        "inspection_data.json, evidence artifacts, legacy PDFs). "
                        "The wagon count is unaffected either way.")
    p.add_argument("--door-source", choices=("legacy", "old_code"),
                   default="legacy",
                   help="Which door/side-damage implementation runs. 'legacy' "
                        "(default) uses the ported DamageDetector, which is the "
                        "one that produces the dashboard's door_status / "
                        "door_close_detected / door_partial_detected and the "
                        "problem frames. 'old_code' uses the DoorTracker path "
                        "instead, for A/B comparison. EXACTLY ONE runs: both "
                        "consume door_state.pt on the same frames, so running "
                        "both would load the model twice and produce two "
                        "verdicts for one question.")
    p.add_argument("--side-model", default=legi.LegacyInspectionConfig().side_model,
                   help="Side door+damage weights: a filename in --models-dir, a "
                        "path, or an s3://bucket/key URI. AUTHORITATIVE for the "
                        "side task; replaces the legacy V4_side_damage.pt "
                        "(default: door_state.pt)")
    p.add_argument("--top-model", default=legi.LegacyInspectionConfig().top_model,
                   help="Top damage weights: filename, path or s3:// URI. "
                        "AUTHORITATIVE for top damage (default: top_damage.pt)")
    p.add_argument("--artifact-bucket", default="",
                   help="S3 bucket (or bucket/prefix) for inspection artifacts. "
                        "Empty writes them to <output>/artifacts/ instead, with "
                        "the identical layout and filenames.")
    p.add_argument("--upload-artifacts", action="store_true",
                   help="Actually upload artifacts and inspection_data.json to "
                        "--artifact-bucket. Off by default so a run never "
                        "publishes without being asked.")
    p.add_argument("--aws-region", default="ap-south-1",
                   help="Region used to build artifact https URLs (default: "
                        "ap-south-1)")
    p.add_argument("--annotated-videos", action="store_true",
                   help="Also write the legacy annotated videos (damage/door "
                        "boxes drawn from the PERSISTED detections -- no second "
                        "model pass). Expensive: re-encodes every camera.")

    # ---- fragment reassembly: rebuild physical gaps before validating them ---
    _fs = fstitch.DEFAULT_FRAGMENT_STITCH
    p.add_argument("--no-fragment-stitching", action="store_true",
                   help="Disable fragment reassembly and validate each tracker "
                        "fragment separately (previous behaviour). One physical "
                        "gap split across several short tracks will then be "
                        "rejected piece by piece.")
    p.add_argument("--stitch-max-seam-sec", type=float,
                   default=_fs.max_seam_seconds,
                   help="Largest temporal hole between two fragments of the same "
                        "physical gap, in SECONDS "
                        f"(default {_fs.max_seam_seconds})")
    p.add_argument("--stitch-seam-tolerance", type=float,
                   default=_fs.seam_speed_tolerance,
                   help="How far a seam jump may exceed what the local advance "
                        "rate predicts, as a dimensionless ratio "
                        f"(default {_fs.seam_speed_tolerance})")
    p.add_argument("--stitch-max-seam-frac", type=float,
                   default=_fs.max_seam_frac,
                   help="Hard cap on seam displacement as a FRACTION of frame "
                        f"width (default {_fs.max_seam_frac})")

    # ---- gap validation: raw YOLO gaps are CANDIDATES, not boundaries --------
    _gv = gval.DEFAULT_GAP_VALIDATION
    p.add_argument("--no-gap-validation", action="store_true",
                   help="Disable motion/temporal gap validation and treat every "
                        "tracked candidate as a wagon boundary (previous behaviour)")
    # Primary flags are CAMERA-INDEPENDENT: seconds for durations, fractions of
    # frame width for distances. They are resolved to this camera's pixels and
    # frames at runtime, so one setting behaves the same on every train and every
    # camera geometry.
    p.add_argument("--gap-min-track-sec", type=float, default=_gv.min_track_seconds,
                   help=f"Min track extent in SECONDS "
                        f"(default: {_gv.min_track_seconds})")
    p.add_argument("--gap-max-track-gap-sec", type=float,
                   default=_gv.max_detection_gap_seconds,
                   help=f"Longest blind run tolerated inside one track, SECONDS "
                        f"(default: {_gv.max_detection_gap_seconds})")
    p.add_argument("--gap-min-motion-frac", type=float, default=_gv.min_motion_frac,
                   help=f"Min centre displacement as a FRACTION OF FRAME WIDTH "
                        f"(default: {_gv.min_motion_frac})")
    p.add_argument("--gap-static-max-frac", type=float,
                   default=_gv.static_max_motion_frac,
                   help=f"At or below this displacement (FRACTION OF FRAME WIDTH) a "
                        f"track is REJECTED_STATIC "
                        f"(default: {_gv.static_max_motion_frac})")
    p.add_argument("--gap-min-motion-frac-sec", type=float,
                   default=_gv.min_motion_frac_per_sec,
                   help=f"Min apparent speed, FRAME WIDTHS per second "
                        f"(default: {_gv.min_motion_frac_per_sec})")
    p.add_argument("--gap-max-motion-frac-sec", type=float,
                   default=_gv.max_motion_frac_per_sec,
                   help=f"Max apparent speed, FRAME WIDTHS per second "
                        f"(default: {_gv.max_motion_frac_per_sec})")
    p.add_argument("--gap-min-separation-sec", type=float,
                   default=_gv.min_separation_seconds,
                   help=f"Minimum time between consecutive VALIDATED gap events, "
                        f"SECONDS (default: {_gv.min_separation_seconds})")
    p.add_argument("--gap-motion-tolerance", type=float,
                   default=_gv.train_motion_tolerance,
                   help=f"Max factor by which a gap's speed may differ from its "
                        f"camera's median gap speed (default: "
                        f"{_gv.train_motion_tolerance})")
    p.add_argument("--gap-min-confidence", type=float, default=_gv.min_mean_confidence,
                   help=f"Min mean track confidence (default: "
                        f"{_gv.min_mean_confidence}; the per-frame detector "
                        f"threshold is unchanged)")

    # ---- DEPRECATED absolute-unit flags -------------------------------------
    # Kept so existing EC2 invocations keep working. They default to None and are
    # applied as PER-CAMERA overrides at validation time, converted with that
    # camera's own width/fps -- the config itself never stores absolute units, so
    # an absolute value given for one geometry cannot leak into another.
    p.add_argument("--gap-min-track-frames", type=int, default=None,
                   help="DEPRECATED: use --gap-min-track-sec. Absolute frame "
                        "count, applied per camera as an override.")
    p.add_argument("--gap-max-track-gap", type=int, default=None,
                   help="DEPRECATED: use --gap-max-track-gap-sec.")
    p.add_argument("--gap-min-motion-px", type=float, default=None,
                   help="DEPRECATED: use --gap-min-motion-frac.")
    p.add_argument("--gap-static-max-px", type=float, default=None,
                   help="DEPRECATED: use --gap-static-max-frac.")
    p.add_argument("--gap-min-motion-px-sec", type=float, default=None,
                   help="DEPRECATED: use --gap-min-motion-frac-sec.")
    p.add_argument("--gap-max-motion-px-sec", type=float, default=None,
                   help="DEPRECATED: use --gap-max-motion-frac-sec.")
    p.add_argument("--gap-min-monotonic", type=float,
                   default=_gv.min_monotonic_fraction,
                   help=f"Min fraction of steps sharing the dominant direction "
                        f"(default: {_gv.min_monotonic_fraction})")

    # ---- temporal classification -------------------------------------------
    _tc = tcls.DEFAULT_TEMPORAL_CONFIG
    p.add_argument("--no-temporal-classification", action="store_true",
                   help="Trust each segment's own label instead of smoothing the "
                        "class sequence temporally (previous behaviour). Off by "
                        "default: a single low-confidence misclassification must "
                        "not move a train-structure boundary.")
    p.add_argument("--classification-switch-persistence", type=int,
                   default=_tc.switch_persistence,
                   help=f"Consecutive same-class observations that qualify a class "
                        f"change when each is shorter than the minimum stable "
                        f"region (default: {_tc.switch_persistence}; measured noise "
                        f"bursts are 1 observation)")
    p.add_argument("--classification-min-stable-sec", type=float,
                   default=_tc.min_stable_region_s,
                   help=f"Seconds of confidence-weighted evidence a challenger "
                        f"class needs to take over (default: "
                        f"{_tc.min_stable_region_s}; measured genuine regions are "
                        f">=1.33s, noise bursts 0.33s)")
    p.add_argument("--classification-min-challenge-conf", type=float,
                   default=_tc.min_confidence_to_challenge,
                   help=f"An observation below this confidence never counts as "
                        f"evidence for a class change (default: "
                        f"{_tc.min_confidence_to_challenge})")

    # ---- train structure ----------------------------------------------------
    p.add_argument("--no-wagon-recovery", action="store_true",
                   help="Disable the WAGON_ACTIVE second validation pass. By "
                        "default a master candidate inside the confirmed wagon "
                        "region that failed only a SOFT gate (speed, trajectory "
                        "noise, weaker confidence) is re-examined and accepted if "
                        "it clears every hard gate. Hard gates -- untracked, too "
                        "short, blind track, isolated static, wrong direction, "
                        "duplicate, separation -- always reject.")
    p.add_argument("--no-wagon-only", action="store_true",
                   help="Count every segment instead of only the WAGON region "
                        "between the first and last WAGON. Off by default: "
                        "ENGINE and BRAKE_VAN are not wagons and never receive a "
                        "GW id.")
    p.add_argument("--no-support-classification", action="store_true",
                   help="Skip classifying support cameras (top/side models). "
                        "Support classification only refines evidence and "
                        "synchronization; it never changes the wagon count.")

    # ---- fusion ------------------------------------------------------------
    p.add_argument("--fusion", choices=("master-fixed", "legacy"),
                   default="master-fixed",
                   help="Fusion architecture. 'master-fixed' (default) treats the "
                        "RIGHT_UP gap sequence as complete and final: global gaps "
                        "== RIGHT_UP gaps, and support cameras contribute evidence "
                        "only. 'legacy' is the previous behaviour, in which support "
                        "cameras could insert extra global gaps (kept for A/B "
                        "comparison only).")
    p.add_argument("--fusion-non-strict", action="store_true",
                   help="Downgrade fixed-master invariant violations from an error "
                        "to a warning (field diagnosis only)")
    p.add_argument("--offset-search", type=float, default=gf.DEFAULT_CONFIG.offset_search_s,
                   help=f"Half-range in seconds of the camera time-offset search "
                        f"(default: {gf.DEFAULT_CONFIG.offset_search_s})")
    p.add_argument("--offset-min-margin", type=float,
                   default=gf.DEFAULT_CONFIG.offset_min_margin_ratio,
                   help=f"Relative score margin a candidate offset must beat its "
                        f"nearest rival by to be accepted; below this the camera is "
                        f"marked UNRESOLVED and contributes no evidence "
                        f"(default: {gf.DEFAULT_CONFIG.offset_min_margin_ratio})")
    p.add_argument("--match-tolerance", type=float,
                   default=gf.DEFAULT_CONFIG.match_tolerance_s,
                   help=f"Timing tolerance in seconds used when associating a "
                        f"support observation with a RIGHT_UP gap "
                        f"(default: {gf.DEFAULT_CONFIG.match_tolerance_s})")

    # ---- deprecated: the insertion mechanism these controlled no longer -----
    # ---- exists in master-fixed fusion. Still parsed by 'legacy'. ----------
    p.add_argument("--fuse-min-support", type=int, default=2,
                   help=argparse.SUPPRESS)
    p.add_argument("--fuse-max-spread",  type=float, default=1.5,
                   help=argparse.SUPPRESS)
    p.add_argument("--fuse-min-conf",    type=float, default=0.4,
                   help=argparse.SUPPRESS)

    # ---- output / reporting -------------------------------------------------
    p.add_argument("--render-videos", action="store_true",
                   help="Also render the per-camera overlay videos into "
                        "results/processed_videos/ (OFF by default: they are "
                        "purely visual and cost ~100 MB per camera)")
    p.add_argument("--no-report", "--no-frames", dest="no_report",
                   action="store_true",
                   help="Skip evidence-frame selection and combined_report.pdf")
    p.add_argument("--report-dpi", type=int, default=er.DEFAULT_REPORT_DPI,
                   help=f"Raster resolution of the PDF pages "
                        f"(default: {er.DEFAULT_REPORT_DPI}). Raise for larger "
                        f"evidence images and a larger PDF.")
    p.add_argument("--keep-evidence-frames", action="store_true",
                   help="Keep the temporary evidence JPEGs instead of deleting "
                        "them after the PDF is written (debugging only)")

    # ---- deprecated, accepted so existing invocations keep working ----------
    p.add_argument("--no-videos", action="store_true",
                   help=argparse.SUPPRESS)      # overlay videos are now opt-in
    p.add_argument("--every-nth-frame", type=int, default=1,
                   help=argparse.SUPPRESS)      # full sequences are never written

    p.add_argument("--no-raw-detections", action="store_true",
                   help="Don't keep raw per-frame detections in memory (saves RAM)")
    p.add_argument("--quiet", action="store_true", help="Reduce log verbosity")
    return p


def _derive_wagon_window(master: LocalCameraTracks, classifications, verbose=False):
    """Derive the wagon window from the CURRENT master gaps + classifications.

    Runtime-derived only: the window comes from the classified segments that the
    master's own validated gaps define. No frame numbers or timestamps are
    assumed, so it holds for any train.
    """
    if not classifications:
        return None
    try:
        segments = ga.build_global_wagons(
            list(master.gaps),
            master_total_frames=master.total_frames, master_fps=master.fps,
            initial_classifications=list(classifications),
            support_camera_ids=[])
        return ts.get_master_wagon_window(segments, verbose=verbose)
    except Exception:
        return None


def _resolved_camera_offsets(state: GlobalTrainState) -> Dict[str, float]:
    """Per-camera clock offset, but ONLY where synchronization was decisive.

    An UNRESOLVED camera maps to 0.0 and is therefore treated exactly as it was
    before offsets existed -- never forced to a guessed value.
    """
    out: Dict[str, float] = {}
    for cam, off in (state.camera_offsets or {}).items():
        if off.get("status") in ("REFERENCE", "RESOLVED"):
            out[cam] = float(off.get("delta", 0.0))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    verbose = not args.quiet

    # Deprecated flags: warn, then ignore. Overlay videos are opt-in now, and
    # full per-frame sequences are never written, so --every-nth-frame has
    # nothing left to thin out.
    if args.no_videos:
        print("NOTE: --no-videos is deprecated and ignored -- overlay videos are "
              "already off by default. Use --render-videos to turn them on.",
              file=sys.stderr)
    if args.every_nth_frame != 1:
        print("NOTE: --every-nth-frame is deprecated and ignored -- the pipeline no "
              "longer writes full frame sequences, only the 20/40/60/80% "
              "representative evidence frames.", file=sys.stderr)
    if args.fusion == "master-fixed":
        for flag, val, default in (("--fuse-min-support", args.fuse_min_support, 2),
                                   ("--fuse-max-spread", args.fuse_max_spread, 1.5),
                                   ("--fuse-min-conf", args.fuse_min_conf, 0.4)):
            if val != default:
                print(f"NOTE: {flag} is ignored under --fusion master-fixed -- there "
                      f"is no support-camera insertion mechanism to tune. RIGHT_UP "
                      f"gaps are complete and final.", file=sys.stderr)

    t_start = time.time()
    print("=" * 70)
    print("  WAGON EYE - PHASE 1 GLOBAL TRAIN RECONSTRUCTION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Resolve inputs + models
    # ------------------------------------------------------------------
    try:
        right_up_video     = _resolve_input(args.right_up,     args.inputs_dir, CAMERA_RIGHT_UP)
        left_up_video      = _resolve_input(args.left_up,      args.inputs_dir, CAMERA_LEFT_UP)
        right_up_top_video = _resolve_input(args.right_up_top, args.inputs_dir, CAMERA_RIGHT_UP_TOP)
        left_up_top_video  = _resolve_input(args.left_up_top,  args.inputs_dir, CAMERA_LEFT_UP_TOP)

        # Two separate side gap models -- one per side camera.
        right_up_gap_path = _resolve_model("right_up_wagon_gap.pt", args.models_dir)
        left_up_gap_path  = _resolve_model("left_up_wagon_gap.pt",  args.models_dir)
        top_gap_path      = _resolve_model("top_gap.pt",            args.models_dir)
        side_cls_path     = _resolve_model("side_classification.pt", args.models_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # top_classification.pt is OPTIONAL: it refines top-camera classification,
    # evidence and synchronization, but it is never a counting authority, so a
    # missing file degrades capability instead of failing the run.
    top_cls_path: Optional[str] = None
    _top_candidate = os.path.join(args.models_dir, ts.TOP_CLASSIFICATION_MODEL)
    if os.path.exists(_top_candidate):
        top_cls_path = os.path.abspath(_top_candidate)
    else:
        print(f"NOTE: {ts.TOP_CLASSIFICATION_MODEL} not found in {args.models_dir} -- "
              f"RIGHT_UP_TOP / LEFT_UP_TOP will not be classified. The wagon count "
              f"is unaffected (RIGHT_UP is the only counting authority). Place the "
              f"file there to enable top classification, e.g.\n"
              f"      aws s3 cp s3://<bucket>/{ts.TOP_CLASSIFICATION_MODEL} "
              f"{args.models_dir}/", file=sys.stderr)

    print(f"  RIGHT_UP video           : {right_up_video}")
    print(f"  LEFT_UP video            : {left_up_video}")
    print(f"  RIGHT_UP_TOP video       : {right_up_top_video}")
    print(f"  LEFT_UP_TOP video        : {left_up_top_video}")
    print(f"  right_up_wagon_gap.pt    : {right_up_gap_path}")
    print(f"  left_up_wagon_gap.pt     : {left_up_gap_path}")
    print(f"  top_gap.pt               : {top_gap_path}")
    print(f"  side_classification.pt   : {side_cls_path}")
    print(f"  output root              : {args.output}")
    print()

    # ------------------------------------------------------------------
    # Inspection model availability, resolved UP FRONT.
    #
    # Before any video is decoded, so a missing weight is visible now rather than
    # after half an hour of tracking. Classes are read from each checkpoint and
    # printed -- nothing downstream assumes them -- and for load the
    # label->state mapping is resolved from the model's real class names.
    # ------------------------------------------------------------------
    if not args.no_inspection:
        oldf.discover_feature_models(args.models_dir, verbose=True)
        print()

    os.makedirs(args.output, exist_ok=True)
    # Only the overlay-video directory is pre-created, and only when asked for.
    # Evidence frames live in a temporary directory managed by evidence_report.
    processed_videos_dir = os.path.join(args.output, "processed_videos")
    if args.render_videos:
        os.makedirs(processed_videos_dir, exist_ok=True)

    keep_raw = not args.no_raw_detections
    _pending_notes: List[str] = []

    # ------------------------------------------------------------------
    # STEP 1 -- per-camera gap tracking
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  STEP 1  Per-camera gap tracking")
    print("-" * 70)
    tracks: Dict[str, LocalCameraTracks] = {}
    try:
        tracks[CAMERA_RIGHT_UP] = _process_side_camera(
            CAMERA_RIGHT_UP, right_up_video, right_up_gap_path,
            confidence=args.side_confidence,
            min_height_ratio=args.side_min_height_ratio,
            keep_raw_detections=keep_raw, verbose=verbose,
        )
        tracks[CAMERA_LEFT_UP] = _process_side_camera(
            CAMERA_LEFT_UP, left_up_video, left_up_gap_path,
            confidence=args.side_confidence,
            min_height_ratio=args.side_min_height_ratio,
            keep_raw_detections=keep_raw, verbose=verbose,
        )
        tracks[CAMERA_RIGHT_UP_TOP] = _process_top_camera(
            CAMERA_RIGHT_UP_TOP, right_up_top_video, top_gap_path,
            confidence=args.top_confidence,
            min_height_ratio=args.top_min_height_ratio,
            keep_raw_detections=keep_raw, verbose=verbose,
        )
        tracks[CAMERA_LEFT_UP_TOP] = _process_top_camera(
            CAMERA_LEFT_UP_TOP, left_up_top_video, top_gap_path,
            confidence=args.top_confidence,
            min_height_ratio=args.top_min_height_ratio,
            keep_raw_detections=keep_raw, verbose=verbose,
        )
    except Exception as e:
        print(f"ERROR: per-camera tracking failed: {e}", file=sys.stderr)
        traceback.print_exc()
        return 3

    print()
    print("  Local counts after Step 1 (raw tracked candidates):")
    for cam in ALL_CAMERAS:
        t = tracks[cam]
        print(f"    {cam:<14}  candidates={len(t.gaps):>3}   "
              f"fps={t.fps:.2f}   frames={t.total_frames}")
    print()

    # ------------------------------------------------------------------
    # STEP 1a -- FRAGMENT REASSEMBLY  (before validation, not inside it)
    #
    # One physical gap can leave the tracker as several short tracks: when a
    # detection is missed the object reappears beyond the association gate, so
    # the track closes and a new id opens. Validation would then judge each piece
    # separately, reject each as too short, and lose the gap they jointly prove.
    #
    # Reassembly restores the physical object first, so every existing gate then
    # applies to the whole gap. Nothing is accepted here and no threshold is
    # relaxed -- this layer only decides which observations belong together.
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  STEP 1a  Fragment reassembly (tracker fragments -> physical gaps)")
    print("-" * 70)
    stitch_cfg = fstitch.FragmentStitchConfig(
        enabled=not args.no_fragment_stitching,
        max_seam_seconds=float(args.stitch_max_seam_sec),
        seam_speed_tolerance=float(args.stitch_seam_tolerance),
        max_seam_frac=float(args.stitch_max_seam_frac),
    )
    stitching: Dict[str, fstitch.StitchResult] = {}
    for cam in ALL_CAMERAS:
        t = tracks[cam]
        # Geometry per camera: seam limits resolve from THIS camera's own width
        # and frame rate, so nothing measured on one geometry leaks into another.
        sres = fstitch.reassemble_fragments(
            t.gaps, cam, stitch_cfg, frame_width=t.width, fps=t.fps,
            verbose=verbose)
        stitching[cam] = sres
        t.gaps = sres.events
    print()

    # ------------------------------------------------------------------
    # STEP 1b -- GAP VALIDATION
    #
    # A raw YOLO gap detection is a CANDIDATE, not a wagon boundary. Each
    # tracked candidate is checked for temporal persistence, detection
    # continuity, real motion, plausible speed, trajectory consistency,
    # direction and confidence, and duplicates are collapsed. The train is
    # moving, so a detection pinned to one pixel column is background, not a
    # gap between wagons.
    #
    # Detection and tracking themselves are UNCHANGED: this filters the
    # GapEvents the existing tracker already emitted.
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  STEP 1b  Gap validation (candidates -> valid wagon boundaries)")
    print("-" * 70)
    # Config holds ONLY camera-independent units (seconds / frame-width
    # fractions / ratios). Absolute values, if any operator supplied the
    # deprecated flags, are applied per camera below.
    gv_cfg = gval.GapValidationConfig(
        enabled=not args.no_gap_validation,
        min_track_seconds=float(args.gap_min_track_sec),
        max_detection_gap_seconds=float(args.gap_max_track_gap_sec),
        min_motion_frac=float(args.gap_min_motion_frac),
        static_max_motion_frac=float(args.gap_static_max_frac),
        min_motion_frac_per_sec=float(args.gap_min_motion_frac_sec),
        max_motion_frac_per_sec=float(args.gap_max_motion_frac_sec),
        min_separation_seconds=float(args.gap_min_separation_sec),
        min_monotonic_fraction=float(args.gap_min_monotonic),
        min_mean_confidence=float(args.gap_min_confidence),
        train_motion_tolerance=float(args.gap_motion_tolerance),
    )

    # Deprecated absolute-unit flags -> per-camera overrides, converted with each
    # camera's own width/fps at resolve() time. Nothing absolute is stored on the
    # config, so a value given for one geometry cannot leak into another.
    gv_overrides: Dict[str, float] = {}
    for flag, attr, target in (
        ("--gap-min-track-frames", "gap_min_track_frames", "min_track_frames"),
        ("--gap-max-track-gap", "gap_max_track_gap", "max_detection_gap_frames"),
        ("--gap-min-motion-px", "gap_min_motion_px", "min_motion_px"),
        ("--gap-static-max-px", "gap_static_max_px", "static_max_motion_px"),
        ("--gap-min-motion-px-sec", "gap_min_motion_px_sec", "min_motion_px_per_sec"),
        ("--gap-max-motion-px-sec", "gap_max_motion_px_sec", "max_motion_px_per_sec"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            gv_overrides[target] = float(value)
            print(f"NOTE: {flag} is deprecated -- thresholds are now expressed in "
                  f"seconds and frame-width fractions so they generalize across "
                  f"trains and camera geometry. Your value is applied verbatim as "
                  f"a per-camera override for this run.", file=sys.stderr)
    gap_validation: Dict[str, gval.GapValidationResult] = {}
    for cam in ALL_CAMERAS:
        t = tracks[cam]
        raw_n = sum(len(v) for v in (t.raw_frame_detections or {}).values())
        # Geometry is passed per camera so thresholds resolve from THIS camera's
        # own resolution and frame rate -- nothing is assumed about either.
        res = gval.validate_gap_events(t.gaps, cam, gv_cfg,
                                       raw_detection_count=raw_n, verbose=verbose,
                                       frame_width=t.width, fps=t.fps,
                                       absolute_overrides=gv_overrides or None)
        gap_validation[cam] = res
        # Replace the camera's gap list with the validated subset and restore
        # track_id as a contiguous temporal rank, as the tracker produces.
        t.gaps = gval.renumber_gap_events(res.accepted)

    print()
    print("  Validated counts after Step 1b:")
    for cam in ALL_CAMERAS:
        t = tracks[cam]
        r = gap_validation[cam]
        print(f"    {cam:<14}  raw_det={r.raw_detection_count:>4}  "
              f"candidates={r.tracked_candidate_count:>3}  "
              f"valid_gaps={len(t.gaps):>3}  rejected={len(r.rejected):>3}")
    print()

    # ------------------------------------------------------------------
    # STEP 2 -- master classification
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  STEP 2  RIGHT_UP master classification (ENGINE / WAGON / BRAKE_VAN)")
    print("-" * 70)
    master = tracks[CAMERA_RIGHT_UP]
    try:
        initial_classifications = _classify_master_pre_fusion(
            master, side_cls_path,
            num_samples=args.classification_samples, verbose=verbose,
        )
    except Exception as e:
        print(f"WARNING: master classification failed: {e}", file=sys.stderr)
        traceback.print_exc()
        initial_classifications = []

    # ---- STEP 2a: TEMPORAL SMOOTHING of the master class sequence ----
    # FIRST_VALID_WAGON / LAST_VALID_WAGON are derived from the SMOOTHED
    # sequence, never from a single observation's label.
    tc_cfg = tcls.TemporalClassificationConfig(
        enabled=not args.no_temporal_classification,
        switch_persistence=int(args.classification_switch_persistence),
        min_stable_region_s=float(args.classification_min_stable_sec),
        min_confidence_to_challenge=float(args.classification_min_challenge_conf),
    )
    temporal_results: Dict[str, tcls.TemporalClassificationResult] = {}
    if initial_classifications:
        print()
        print("-" * 70)
        print("  STEP 2a  Temporal classification (segment-level hysteresis)")
        print("-" * 70)
        try:
            initial_classifications, tres = tcls.apply_temporal_classification(
                initial_classifications, master.fps, camera_id=CAMERA_RIGHT_UP,
                cfg=tc_cfg, verbose=verbose,
            )
            temporal_results[CAMERA_RIGHT_UP] = tres
        except Exception as e:
            print(f"WARNING: temporal classification failed for "
                  f"{CAMERA_RIGHT_UP}: {e}", file=sys.stderr)
            _pending_notes.append(f"temporal_classification_failed:"
                                  f"{CAMERA_RIGHT_UP}:{e}")

    # ------------------------------------------------------------------
    # STEP 2b -- SUPPORT-CAMERA CLASSIFICATION
    #
    #   RIGHT_UP      -> side_classification.pt   (Step 2, master authority)
    #   LEFT_UP       -> side_classification.pt   (same side geometry)
    #   RIGHT_UP_TOP  -> top_classification.pt
    #   LEFT_UP_TOP   -> top_classification.pt
    #
    # Purpose: identify each support camera's own engine / wagon / brake-van
    # regions so that engine and brake-van observations are kept OUT of wagon
    # synchronization, and so the PDF and overlay videos can label them.
    # Support cameras are never counting authorities.
    # ------------------------------------------------------------------
    support_regions: Dict[str, ts.LocalWagonRegion] = {}
    classification_models: Dict[str, str] = {CAMERA_RIGHT_UP: os.path.basename(side_cls_path)}
    top_label_mapping = None
    if not args.no_support_classification:
        print()
        print("-" * 70)
        print("  STEP 2b  Support-camera classification (top / side models)")
        print("-" * 70)
        _cache: Dict[str, object] = {}
        for cam in ALL_CAMERAS:
            if cam == CAMERA_RIGHT_UP:
                continue
            want = ts.CAMERA_CLASSIFICATION_MODEL.get(cam)
            path = (top_cls_path if want == ts.TOP_CLASSIFICATION_MODEL
                    else side_cls_path)
            if path is None:
                classification_models[cam] = f"{want} (MISSING)"
                support_regions[cam] = ts.LocalWagonRegion(
                    camera_id=cam, classifier_model=f"{want} (missing)",
                    reason=f"{want} not available; camera not classified")
                print(f"  [CLASSIFY/{cam}] SKIPPED -- {want} is not available")
                continue
            try:
                if path not in _cache:
                    # Load each model ONCE and reuse it across cameras.
                    _cache[path] = ts.load_segment_classifier(
                        path, num_samples=args.classification_samples,
                        verbose=verbose)
                clf, mapping = _cache[path]           # type: ignore[misc]
                if want == ts.TOP_CLASSIFICATION_MODEL:
                    top_label_mapping = mapping
                classification_models[cam] = os.path.basename(path)

                t = tracks[cam]
                segs = segments_from_gaps(t.gaps, t.total_frames)
                labels: List[str] = []
                if segs:
                    cls = clf.classify_segments(t.video_path, segs)
                    # Same temporal smoothing on support cameras, so their local
                    # wagon regions are not moved by a single bad observation.
                    cls, tres = tcls.apply_temporal_classification(
                        cls, t.fps, camera_id=cam, cfg=tc_cfg,
                        sample_history=getattr(clf, "sample_history", None),
                        verbose=verbose)
                    temporal_results[cam] = tres
                    labels = [c.label for c in cls]
                support_regions[cam] = ts.build_local_wagon_region(
                    cam, segs, labels, t.fps,
                    classifier_model=os.path.basename(path),
                    unmapped_classes=mapping.unmapped, verbose=verbose)
            except Exception as e:
                print(f"WARNING: classification failed for {cam}: {e}", file=sys.stderr)
                support_regions[cam] = ts.LocalWagonRegion(
                    camera_id=cam, classifier_model=os.path.basename(str(path)),
                    reason=f"classification error: {type(e).__name__}: {e}")
                state_note = f"support_classification_failed:{cam}:{e}"
                _pending_notes.append(state_note)

    # ------------------------------------------------------------------
    # STEP 2c -- WAGON_ACTIVE RECOVERY  (second validation pass)
    #
    # Validation had to run in STEP 1b, before classification, because
    # classification needs the segments that validated gaps define. So the first
    # pass could not know the train state. Now that the wagon window exists,
    # re-examine the master candidates that fell INSIDE it and failed only a SOFT
    # gate (speed vs the local reference, absolute speed band, sub-floor
    # displacement, weaker confidence, noisier trajectory).
    #
    # Every HARD gate still rejects: untracked, insufficient confirmation,
    # mostly-blind track, isolated static artefact, wrong direction, duplicate,
    # minimum-separation duplicate. Recovery also re-checks duplicate and
    # separation against the accepted set, so it cannot crowd an existing gap.
    #
    # This is what makes a genuine wagon gap inside the wagon run count
    # immediately: no further classification event is awaited.
    # ------------------------------------------------------------------
    recovery = None
    if (not args.no_gap_validation and not args.no_wagon_recovery
            and gap_validation.get(CAMERA_RIGHT_UP)):
        print()
        print("-" * 70)
        print("  STEP 2c  WAGON_ACTIVE recovery (soft-failed gaps inside the "
              "wagon region)")
        print("-" * 70)
        _win = _derive_wagon_window(master, initial_classifications, verbose=False)
        if _win is not None and _win.wagon_start_frame is not None:
            print(f"  wagon window (runtime-derived): frames "
                  f"{_win.wagon_start_frame}-{_win.wagon_end_frame}")
            recovery = gval.recover_wagon_active_candidates(
                gap_validation[CAMERA_RIGHT_UP].rejected,
                master.gaps,
                _win.wagon_start_frame, _win.wagon_end_frame,
                CAMERA_RIGHT_UP, gv_cfg,
                frame_width=master.width, fps=master.fps,
                absolute_overrides=gv_overrides or None, verbose=verbose)
            if recovery.recovered:
                master.gaps = gval.renumber_gap_events(
                    list(master.gaps) + list(recovery.recovered))
                # The master gap sequence changed, so the segments and therefore
                # the classification must be rebuilt from it.
                print(f"  recovered {len(recovery.recovered)} gap(s) -> "
                      f"re-deriving master segments and classification")
                try:
                    initial_classifications = _classify_master_pre_fusion(
                        master, side_cls_path,
                        num_samples=args.classification_samples, verbose=False)
                    if initial_classifications:
                        initial_classifications, tres2 = \
                            tcls.apply_temporal_classification(
                                initial_classifications, master.fps,
                                camera_id=CAMERA_RIGHT_UP, cfg=tc_cfg,
                                verbose=False)
                        temporal_results[CAMERA_RIGHT_UP] = tres2
                except Exception as e:
                    print(f"WARNING: re-classification after recovery failed: {e}",
                          file=sys.stderr)
                    _pending_notes.append(f"reclassification_after_recovery:{e}")
            else:
                print("  no gap recovered -- every wagon-window candidate either "
                      "passed already or failed a hard gate")
        else:
            print("  no wagon window derived -- recovery skipped")

    # ------------------------------------------------------------------
    # STEP 3 -- cross-camera fusion
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    if args.fusion == "master-fixed":
        print("  STEP 3  Cross-camera alignment  (master-fixed: RIGHT_UP is final)")
    else:
        print("  STEP 3  Cross-camera gap fusion  (LEGACY: support cameras may insert)")
    print("-" * 70)
    support = [tracks[c] for c in ALL_CAMERAS if c != CAMERA_RIGHT_UP]

    if args.fusion == "master-fixed":
        # The global gap sequence IS the RIGHT_UP gap sequence. Support cameras
        # are aligned to it to attach evidence; they cannot create, delete,
        # split or merge a global gap, so the count is independent of both the
        # support detections and the camera-offset estimation.
        fusion_cfg = gf.FusionConfig(
            offset_search_s=float(args.offset_search),
            offset_min_margin_ratio=float(args.offset_min_margin),
            match_tolerance_s=float(args.match_tolerance),
            strict_invariants=not args.fusion_non_strict,
        )
        state: GlobalTrainState = gf.assemble_global_train_state_master_fixed(
            master_tracks=master,
            support_tracks=support,
            initial_classifications=initial_classifications,
            config=fusion_cfg,
            verbose=verbose,
            wagon_regions=support_regions,
            wagon_only=not args.no_wagon_only,
        )
    else:
        print("WARNING: --fusion legacy allows support-camera observations to "
              "insert global gaps, which breaks the RIGHT_UP-is-final invariant. "
              "Use it only to reproduce old results.", file=sys.stderr)
        fuse_cfg = dict(ga.PHASE1_DEFAULTS)
        fuse_cfg.update({
            "insert_min_support": int(args.fuse_min_support),
            "insert_max_spread_sec": float(args.fuse_max_spread),
            "insert_min_confidence": float(args.fuse_min_conf),
        })
        state = ga.assemble_global_train_state(
            master_tracks=master,
            support_tracks=support,
            initial_classifications=initial_classifications,
            config=fuse_cfg,
            verbose=verbose,
        )
        state.fusion_mode = "legacy"

    # ---- attach validation / classification diagnostics to the state ----
    state.gap_validation_statistics = {
        cam: gap_validation[cam].to_dict(include_rejections=False)
        for cam in ALL_CAMERAS if cam in gap_validation
    }
    state.gap_rejection_details = {
        cam: gap_validation[cam].to_dict(include_rejections=True)["rejections"]
        for cam in ALL_CAMERAS
        if cam in gap_validation and gap_validation[cam].rejected
    }
    state.gap_validation_config = gv_cfg.describe()
    state.fragment_stitching = {
        cam: stitching[cam].to_dict()
        for cam in ALL_CAMERAS if cam in stitching
    }
    if recovery is not None:
        state.wagon_active_recovery = recovery.to_dict()
    state.classification_model_by_camera = classification_models
    state.temporal_classification = {
        cam: tres.to_dict(include_samples=False)
        for cam, tres in temporal_results.items()
    }
    state.temporal_classification_config = tc_cfg.describe()
    if top_label_mapping is not None:
        state.top_classification_model_info = top_label_mapping.to_dict()
    for note in _pending_notes:
        state.add_note(note)

    # ------------------------------------------------------------------
    # STEPS 8-11 -- INSPECTION  (door / load / damage, from old_code)
    #
    # Runs ONLY here, after the global wagon roster and GW ids are final. The
    # mature door / load / damage algorithms are old_code's own and are invoked
    # UNCHANGED through the `features` shim; this stage only supplies them with
    # per-wagon frames and fuses their output.
    #
    # Three protections apply:
    #   * the roster is hashed before and after, and a mismatch raises -- so
    #     "inspection cannot change the count" is checked, not asserted;
    #   * association is by construction, via wagon_cache/<GW_n>/<camera>/, so a
    #     detection cannot be attributed to the wrong wagon or invent one;
    #   * any inspection failure degrades to a warning, because a wagon count
    #     that already succeeded must never be lost to a downstream feature.
    # ------------------------------------------------------------------
    inspection_result = None
    report_paths: Dict[str, Any] = {}
    if not args.no_inspection:
        print()
        print("-" * 70)
        print("  STEPS 8-11  Inspection (door / load / damage from old_code)")
        print("-" * 70)
        _insp_t0 = time.time()
        _roster_before = core_state.roster_hash(state)
        try:
            # The frame cache must survive the old features when the legacy
            # output layer still has to read it. It is cleared below instead.
            _legacy_enabled = not args.no_legacy_inspection
            insp_cfg = oldf.OldInspectionConfig(
                enabled=True,
                keep_cache=bool(args.keep_wagon_cache) or _legacy_enabled,
                # ONE OWNER PER TASK.
                #
                # old_code and the legacy layer can both detect on the SAME
                # weights over the SAME cached frames: door here and the legacy
                # side pass both read door_state.pt; damage here and the legacy
                # top pass both read top_damage.pt. Running a pair would load
                # one model twice and produce two verdicts for one question,
                # with whichever was reconciled last silently winning.
                #
                #   door   -> old_code only when --door-source old_code
                #   damage -> old_code only when the legacy layer is off, since
                #             the legacy top pass is what feeds the dashboard
                #             JSON's damage fields
                #   load   -> always old_code; the legacy engine has no load
                #             feature at all, so there is no pair to resolve
                run_door=(not args.no_door and args.door_source == "old_code"),
                run_load=not args.no_load,
                run_damage=(not args.no_damage and not _legacy_enabled),
                cache=iwc.WagonCacheConfig(
                    every_nth=int(args.wagon_cache_every_nth),
                    max_frames_per_wagon=int(args.wagon_cache_max_frames)),
            )
            inspection_result = oldf.run_old_inspection(
                state=state, tracks_by_camera=tracks,
                models_dir=args.models_dir, output_root=args.output,
                cfg=insp_cfg, verbose=verbose)

            _s = inspection_result.to_dict()["summary"]
            print()
            print(f"  wagons inspected          : {len(inspection_result.unified)}")
            print(f"  L doors OPEN / PARTIAL    : {_s['left_doors_open']} / "
                  f"{_s['left_doors_partial']}")
            print(f"  R doors OPEN / PARTIAL    : {_s['right_doors_open']} / "
                  f"{_s['right_doors_partial']}")
            print(f"  doors DAMAGED             : {_s['doors_damaged']}")
            print(f"  LOADED / EMPTY / NO_DATA  : {_s['loaded']} / {_s['empty']} / "
                  f"{_s['load_no_data']}")
            print(f"  top damaged               : {_s['top_damaged']}")
            print(f"  anomaly wagons            : {_s['anomaly_wagons']}")
            for w in inspection_result.warnings:
                print(f"  NOTE: {w}")

            # The old combined report is built LATER, after the legacy layer has
            # produced its verdicts and they have been folded into `unified` --
            # otherwise the PDF would report NO_DATA for doors while the
            # dashboard JSON reported a real state, which is two artifacts of one
            # run disagreeing about the same wagon.
        except Exception as exc:
            msg = (f"inspection stage failed: {type(exc).__name__}: {exc} -- "
                   f"the wagon count above is unaffected")
            print(f"WARNING: {msg}", file=sys.stderr)
            state.add_note(msg)
            traceback.print_exc(limit=3)
        finally:
            # The checked guarantee, enforced even if a feature raised.
            core_state.assert_roster_unchanged(state, _roster_before)
            _elapsed = time.time() - _insp_t0
            if inspection_result is not None:
                inspection_result.timings["total_seconds"] = _elapsed
            print(f"  inspection elapsed        : {_elapsed:.1f}s")
    else:
        print()
        print("  Inspection SKIPPED (--no-inspection)")

    if inspection_result is not None:
        state.inspection = inspection_result.to_dict()
        if report_paths:
            state.inspection["reports"] = {
                "combined_json": (report_paths.get("combined") or {}).get("json_path"),
                "combined_pdf": (report_paths.get("combined") or {}).get("pdf_path"),
                "camera_pdfs": {k: v for k, v in
                                (report_paths.get("cameras") or {}).items()},
            }

    # ------------------------------------------------------------------
    # STEPS 12-14  Legacy inspection OUTPUT layer
    #
    #   finalized roster -> per-camera legacy DataFrames -> legacy
    #   DamageDetector / ProblemFrameExtractor / ArtifactPublisher /
    #   json_builder -> inspection_data.json -> legacy PDFs -> annotated videos
    #
    # This is the dashboard-compatible half of the port. It READS the roster and
    # the frame cache and writes only its own files. Three protections, same as
    # above: the roster is hashed and re-verified in a `finally`; every wagon
    # identity it emits is checked against the roster; and any failure degrades
    # to a warning, because a wagon count that already succeeded must never be
    # lost to a downstream report.
    # ------------------------------------------------------------------
    legacy_result = None
    if not args.no_legacy_inspection:
        print()
        print("-" * 70)
        print("  STEPS 12-14  Legacy inspection outputs "
              "(dashboard JSON / evidence / PDF)")
        print("-" * 70)
        _leg_t0 = time.time()
        _roster_before_legacy = core_state.roster_hash(state)
        try:
            # Planning touches no video: it is the same deterministic mapping
            # the cache was built from, recomputed so the windows are available
            # here without threading the plan object through the old stage.
            _cache_cfg = iwc.WagonCacheConfig(
                every_nth=int(args.wagon_cache_every_nth),
                max_frames_per_wagon=int(args.wagon_cache_max_frames))
            legacy_plan = iwc.plan_cache(state, tracks, args.output,
                                         cfg=_cache_cfg)

            # The frame cache is normally built by the stage above. When that
            # stage was skipped (--no-inspection) this layer still needs it, so
            # it is built here rather than silently reporting every wagon as
            # NOT_VISIBLE. `skip_existing` makes this a no-op when the cache is
            # already populated, so the videos are never decoded twice.
            if args.no_inspection:
                print("  [CACHE] building wagon frame cache "
                      "(--no-inspection skipped the stage that normally does)")
                iwc.build_wagon_cache(legacy_plan, tracks, _cache_cfg,
                                      verbose=verbose)

            # LOAD comes from the old_code load feature and is the ONLY input
            # the legacy top flavour needs from it: it decides wagon_loaded vs
            # wagon, which drives both the wagons_loaded/empty counts and the
            # legacy suppression of floor damage on a loaded wagon.
            load_by_wagon = {}
            if inspection_result is not None:
                load_by_wagon = {
                    gid: u.load_status
                    for gid, u in inspection_result.unified.items()
                    if u.load_status in (C.LOAD_LOADED, C.LOAD_EMPTY)
                }

            s3_client = None
            if args.upload_artifacts and args.artifact_bucket:
                from inspection.legacy.s3 import S3Client
                s3_client = S3Client(region=args.aws_region)

            legacy_result = legi.run_legacy_inspection(
                state=state, tracks_by_camera=tracks, plan=legacy_plan,
                models_dir=args.models_dir,
                output_root=os.path.join(args.output, "inspection"),
                cfg=legi.LegacyInspectionConfig(
                    side_model=args.side_model,
                    top_model=args.top_model,
                    # The other half of the one-owner rule above. The side pass
                    # runs only when this layer owns the door task; the top pass
                    # owns damage whenever this layer is enabled.
                    run_side_damage=(not args.no_door
                                     and args.door_source == "legacy"),
                    run_top_damage=not args.no_damage,
                    artifact_bucket=args.artifact_bucket,
                    region=args.aws_region,
                    # Opt-in, and only with somewhere to put them.
                    upload_to_s3=bool(args.upload_artifacts
                                      and args.artifact_bucket),
                    build_annotated_video=bool(args.annotated_videos)),
                load_status_by_wagon=load_by_wagon,
                s3_client=s3_client, verbose=verbose)

            print()
            print(f"  global wagons (authority) : {legacy_result.global_wagon_count}")
            for cam, cam_res in sorted(legacy_result.cameras.items()):
                print(f"  {cam:<13} JSON wagons: {cam_res.wagons_in_json:<4} "
                      f"scanned: {cam_res.wagons_scanned:<4} "
                      f"problem frames: {cam_res.problem_frames}")
            for w in legacy_result.warnings:
                print(f"  NOTE: {w}")

            # ---- ONE set of verdicts behind every artifact ------------------
            if inspection_result is not None and inspection_result.unified:
                applied = legi.apply_to_unified(inspection_result.unified,
                                                legacy_result)
                print(f"  reconciled into report    : "
                      f"door={applied['door']} side_damage={applied['side_damage']} "
                      f"top_damage={applied['top_damage']}")

            # ---- renderers: persisted state only, never a second model pass --
            if not args.no_report:
                from datetime import datetime as _dt
                # The old combined + per-camera reports, now that `unified`
                # carries the legacy verdicts. Built here so every document in
                # this run describes the same detections.
                if inspection_result is not None and inspection_result.unified:
                    report_paths = oldr.build_all_reports(
                        state=state, unified=inspection_result.unified,
                        output_root=args.output,
                        batch_key=os.path.basename(os.path.abspath(args.output)),
                        camera_status=inspection_result.camera_status,
                        verbose=verbose)
                    state.inspection = inspection_result.to_dict()
                    state.inspection["reports"] = {
                        "combined_json": (report_paths.get("combined") or {}).get("json_path"),
                        "combined_pdf": (report_paths.get("combined") or {}).get("pdf_path"),
                        "camera_pdfs": dict(report_paths.get("cameras") or {}),
                    }
                legacy_paths = legr.build_all_legacy_outputs(
                    state=state,
                    output_root=os.path.join(args.output, "inspection"),
                    upload_timestamp=_dt.now(),
                    tracks_by_camera=tracks,
                    load_status_by_wagon=load_by_wagon,
                    batch_key=os.path.basename(os.path.abspath(args.output)),
                    build_videos=bool(args.annotated_videos),
                    verbose=verbose)
            else:
                legacy_paths = {}
        except Exception as exc:
            msg = (f"legacy inspection outputs failed: {type(exc).__name__}: "
                   f"{exc} -- the wagon count above is unaffected")
            print(f"WARNING: {msg}", file=sys.stderr)
            state.add_note(msg)
            traceback.print_exc(limit=3)
            legacy_paths = {}
        finally:
            core_state.assert_roster_unchanged(state, _roster_before_legacy)
            print(f"  legacy inspection elapsed : {time.time() - _leg_t0:.1f}s")

        if legacy_result is not None:
            state.inspection = dict(state.inspection or {})
            state.inspection["legacy_outputs"] = legacy_result.to_dict()
            if legacy_paths:
                state.inspection["legacy_outputs"]["reports"] = legacy_paths
    elif (not args.no_report and inspection_result is not None
            and inspection_result.unified):
        # Legacy outputs disabled: the old reports are the only ones, so they
        # are built here instead. Same call, same view model -- only the point
        # in the run differs, because there is no legacy verdict to fold in.
        report_paths = oldr.build_all_reports(
            state=state, unified=inspection_result.unified,
            output_root=args.output,
            batch_key=os.path.basename(os.path.abspath(args.output)),
            camera_status=inspection_result.camera_status, verbose=verbose)
        state.inspection = inspection_result.to_dict()
        state.inspection["reports"] = {
            "combined_json": (report_paths.get("combined") or {}).get("json_path"),
            "combined_pdf": (report_paths.get("combined") or {}).get("pdf_path"),
            "camera_pdfs": dict(report_paths.get("cameras") or {}),
        }

    # The frame cache is per-train state: leaving GW_1..GW_N of one train on
    # disk would let the next train's GW_1 read the previous train's frames.
    if not args.keep_wagon_cache:
        iwc.clear_wagon_cache(
            os.path.join(args.output, iwc.CACHE_DIRNAME), verbose=verbose)

    # ------------------------------------------------------------------
    # STEP 4 -- write JSON
    # ------------------------------------------------------------------
    state_json_path = os.path.join(args.output, "global_train_state.json")
    with open(state_json_path, "w", encoding="utf-8") as f:
        f.write(state.to_json())
    print()
    print(f"[OUTPUT] wrote {state_json_path}")

    tracking_dump = {
        cam: tracks[cam].to_dict(include_classifications=(cam == CAMERA_RIGHT_UP))
        for cam in ALL_CAMERAS
    }
    if initial_classifications:
        tracking_dump[CAMERA_RIGHT_UP]["pre_fusion_classifications"] = [
            c.to_dict() for c in initial_classifications
        ]
    tracking_path = os.path.join(args.output, "per_camera_tracking.json")
    with open(tracking_path, "w", encoding="utf-8") as f:
        json.dump(tracking_dump, f, indent=2)
    print(f"[OUTPUT] wrote {tracking_path}")

    # ------------------------------------------------------------------
    # STEP 5 -- overlay videos (opt-in: purely visual, ~100 MB per camera)
    # ------------------------------------------------------------------
    if args.render_videos:
        print()
        print("-" * 70)
        print("  STEP 5  Overlay videos  (--render-videos)")
        print("-" * 70)
        # Under master-fixed fusion each camera's estimated clock offset is used
        # so the projected global boundaries land on the right local frames, and
        # wagons outside a camera's footage are not drawn at all. Videos are
        # visualization only: they use the same GW ids as combined_report.pdf and
        # cannot change the count.
        offsets = _resolved_camera_offsets(state)
        # The videos show the FULL train. Engine and brake-van regions are
        # labelled with their classification but carry no GW id.
        _ww = state.wagon_window or {}
        non_wagon_regions = (list(_ww.get("leading_non_wagon_objects", []))
                             + list(_ww.get("interior_non_wagon_objects", []))
                             + list(_ww.get("trailing_non_wagon_objects", [])))
        for cam in ALL_CAMERAS:
            try:
                out_mp4 = os.path.join(processed_videos_dir, f"{cam}_processed.mp4")
                vs.render_processed_video(
                    local_tracks=tracks[cam],
                    state=state,
                    output_path=out_mp4,
                    draw_raw_detections=keep_raw,
                    verbose=verbose,
                    time_offset=offsets.get(cam, 0.0),
                    drop_out_of_range=(state.fusion_mode == "master-fixed"),
                    non_wagon_regions=non_wagon_regions,
                    inspection_state=(state.inspection or {}),
                )
            except Exception as e:
                print(f"WARNING: render failed for {cam}: {e}", file=sys.stderr)
                state.add_note(f"render_failed:{cam}:{e}")

    # ------------------------------------------------------------------
    # STEP 6 -- combined evidence report
    #
    # Replaces the old "write every frame of every wagon" behaviour.  For each
    # global event produced above, four representative frames (20/40/60/80% of
    # that camera's own valid evidence interval) are pulled from each of the
    # four cameras -- at most 16 images per event -- composed into
    # results/combined_report.pdf, and the temporary JPEGs are then deleted.
    #
    # This step only READS `state` and `tracks`.  It cannot influence the
    # wagon count, the global ids, or any fusion decision.
    # ------------------------------------------------------------------
    report_info: Optional[Dict[str, object]] = None
    if not args.no_report:
        print()
        print("-" * 70)
        print("  STEP 6  Combined evidence report (20/40/60/80% per camera)")
        print("-" * 70)
        er.warn_about_legacy_frame_dirs(args.output, verbose=verbose)
        try:
            report_info = er.build_combined_report(
                state=state,
                tracks=tracks,
                output_root=args.output,
                dpi=args.report_dpi,
                keep_evidence_frames=args.keep_evidence_frames,
                verbose=verbose,
            )
            if not report_info.get("verified"):
                state.add_note(f"report_unverified:{report_info.get('detail')}")
        except Exception as e:
            print(f"WARNING: combined report failed: {e}", file=sys.stderr)
            traceback.print_exc()
            state.add_note(f"report_failed:{type(e).__name__}:{e}")

    # ------------------------------------------------------------------
    # STEP 7 -- final summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print()
    print(summarize_state(state))
    print(f"  total elapsed: {elapsed:.1f}s")
    print(f"  output root  : {os.path.abspath(args.output)}")
    if report_info:
        print(f"  report       : {report_info.get('pdf_path')}  "
              f"({report_info.get('pages')} pages, "
              f"{report_info.get('slots_available')}/{report_info.get('slots_total')} "
              f"evidence frames, {report_info.get('detail')})")
        if report_info.get("frames_cleaned"):
            print(f"  temp frames  : {report_info.get('frames_cleaned')} "
                  f"deleted after the PDF was verified")
    print()

    # Re-write JSON so any added notes are persisted
    with open(state_json_path, "w", encoding="utf-8") as f:
        f.write(state.to_json())

    return 0


if __name__ == "__main__":
    sys.exit(main())
