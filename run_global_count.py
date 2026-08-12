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
import video_segmenter as vs
import evidence_report as er


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

    os.makedirs(args.output, exist_ok=True)
    # Only the overlay-video directory is pre-created, and only when asked for.
    # Evidence frames live in a temporary directory managed by evidence_report.
    processed_videos_dir = os.path.join(args.output, "processed_videos")
    if args.render_videos:
        os.makedirs(processed_videos_dir, exist_ok=True)

    keep_raw = not args.no_raw_detections

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
    print("  Local counts after Step 1:")
    for cam in ALL_CAMERAS:
        t = tracks[cam]
        print(f"    {cam:<14}  wagons={t.local_wagon_count:>3}   gaps={len(t.gaps):>3}   "
              f"fps={t.fps:.2f}   frames={t.total_frames}")
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
