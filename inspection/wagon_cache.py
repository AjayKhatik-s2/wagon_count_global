"""Build the per-wagon frame cache the old feature processors read.

    wagon_cache/<GW_n>/<CAMERA>/frame_NNNNNN.jpg

THIS IS THE ASSOCIATION MECHANISM
---------------------------------
old_code's processors are already global-wagon-native: they iterate
`state.wagons`, take `gw.global_id`, and read that wagon's frames for one camera.
So the entire "which GW does this observation belong to?" question is answered
HERE, once, by slicing each camera's video using the finalized global wagon
interval and that camera's already-resolved clock offset.

That has a property worth stating plainly: association is by CONSTRUCTION, not by
inference. Every frame under `wagon_cache/GW_17/RIGHT_UP/` is, by the way it was
produced, a frame of GW_17 as seen by RIGHT_UP. A detection found in it cannot be
mis-attributed, and no detection can invent a wagon, because the only directories
that exist are the ones the finalized roster named.

    global wagon window (MASTER seconds)   <- from the finished roster
        - camera offset (RESOLVED only)    <- from the finished synchronization
        = that camera's own time window
        x camera fps
        = that camera's frame range

UNRESOLVED CAMERAS ARE NOT GUESSED
----------------------------------
A camera whose offset was never resolved gets NO directories written. Its features
then report NO_FRAMES / NO_DATA rather than being attributed to a guessed wagon --
the same policy the existing evidence report already applies. `plan_cache()`
records that as an explicit per-camera status so the reports can distinguish
"inspected, found nothing" from "never inspected".

MEMORY AND IO
-------------
One camera at a time, one sequential pass per camera, frames written and released
immediately. Frames are never accumulated and the video is never decoded twice:
wagon windows are sorted and walked in a single forward sweep, so a 4-camera train
costs exactly 4 decodes regardless of wagon count. `every_nth` and
`max_frames_per_wagon` bound the on-disk and per-wagon cost.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core import constants as C

__all__ = ["WagonCacheConfig", "WagonWindow", "CachePlan", "plan_cache",
           "build_wagon_cache", "cache_stats", "clear_wagon_cache"]

CACHE_DIRNAME = "wagon_cache"


@dataclass
class WagonCacheConfig:
    """How many frames to cache per wagon per camera.

    All camera-independent: counts and strides, no pixel or frame constants tied
    to one train's geometry or speed.
    """

    every_nth: int = 1
    """Sample every Nth frame inside a wagon window.

    DENSITY IS A CORRECTNESS CONSTRAINT HERE, NOT A COST DIAL, and this was
    measured the hard way. old_code's DoorTracker associates detections with
    `max_center_distance = 150 px`. A door crosses the frame at roughly 28 px per
    frame, so a stride of 6 moves it ~170 px between sightings -- beyond the gate.
    Every detection then opens a NEW track, `n_init = 3` is never reached, and the
    wagon reports CLOSED with zero confidence even though the door is plainly
    visible at 0.945. Observed exactly that: stride 6 gave `tracks=0` on a wagon
    with a real PARTIAL door.

    The safe ceiling is 150 / 28 ~= 5 frames; 1 keeps full fidelity to how
    old_code was run, and 2 is the most that should be used without re-measuring
    against the train's own speed."""

    max_frames_per_wagon: int = 150
    """Runaway guard per (wagon, camera), applied as a CONTIGUOUS RUN.

    Sized NOT to bind on a normal wagon. Measured window lengths on real data are
    53-70 frames, so an earlier value of 40 truncated ordinary wagons -- and that
    is not a harmless economy: truncating from the start biases every verdict to
    the first part of the wagon. Observed directly, a window of frames 576-645 was
    cut at 597, which excluded the frame carrying a 0.945 PARTIAL door, and the
    wagon was reported CLOSED. The verdict was right for the frames inspected and
    wrong for the wagon.

    So this exists only to stop a pathological window (a stopped train, a
    collapsed segment) writing thousands of JPEGs. When it DOES bind, the window
    is recorded as truncated so partial coverage is visible instead of silent.

    It exists so an unusually long window (a stopped train, a collapsed segment)
    cannot write thousands of JPEGs. When it binds, the frames kept are a
    contiguous run from the START of the window -- deliberately NOT evenly spaced
    across it. Even spacing looks fairer but destroys the temporal continuity the
    Kalman/Hungarian trackers need, which is the same failure as too large a
    stride: a wide jump between kept frames breaks association and silently
    produces zero tracks. A shorter continuous observation beats a longer
    discontinuous one."""

    jpeg_quality: int = 85
    """Matches the quality the existing evidence report already writes."""

    skip_existing: bool = True
    """Reuse a cache directory that is already populated, so re-running the
    inspection stage after a crash does not redo the decode."""


@dataclass
class WagonWindow:
    """One (wagon, camera) frame range, already in that camera's own clock."""
    global_id: str
    camera_id: str
    start_frame: int
    end_frame: int
    truncated: bool = False
    """True when the runaway guard cut this window short, so the feature saw only
    part of the wagon. Recorded because partial coverage must never look like a
    complete inspection."""

    @property
    def n_frames(self) -> int:
        return max(0, self.end_frame - self.start_frame + 1)


@dataclass
class CachePlan:
    """What will be cached, and why anything was left out."""
    root: str
    windows: List[WagonWindow] = field(default_factory=list)
    camera_status: Dict[str, str] = field(default_factory=dict)
    """camera -> RESOLVED | UNRESOLVED | NO_VIDEO. UNRESOLVED cameras are skipped
    entirely rather than cached at a guessed offset."""
    skipped: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "planned_windows": len(self.windows),
            "camera_status": dict(sorted(self.camera_status.items())),
            "wagons": sorted({w.global_id for w in self.windows},
                             key=_gw_key),
            "skipped": list(self.skipped),
        }


def _gw_key(gid: str) -> Tuple[int, Any]:
    tail = str(gid).split("_")[-1]
    return (0, int(tail)) if tail.isdigit() else (1, str(gid))


def wagon_camera_dir(root: str, global_id: str, camera_id: str) -> str:
    """Must match `old_code/_common.wagon_camera_dir` exactly."""
    return os.path.join(root, global_id, C.CAMERA_FOLDER.get(camera_id, camera_id))


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------

def plan_cache(
    state: Any,
    tracks_by_camera: Dict[str, Any],
    output_root: str,
    cameras: Optional[Sequence[str]] = None,
    cfg: Optional[WagonCacheConfig] = None,
) -> CachePlan:
    """Compute every (wagon, camera) frame range WITHOUT touching a video.

    Separated from extraction so the mapping can be tested, inspected and
    reported without decoding anything -- and so a bad plan is visible before an
    hour of IO is spent on it.
    """
    del cfg  # planning does not depend on sampling density
    cameras = list(cameras or C.ALL_CAMERAS)
    plan = CachePlan(root=os.path.join(output_root, CACHE_DIRNAME))

    # Offsets ONLY where synchronization was decisive. Absence means unresolved.
    raw_offsets = getattr(state, "camera_offsets", None)
    if raw_offsets is None and isinstance(state, dict):
        raw_offsets = state.get("camera_offsets") or {}
    offsets: Dict[str, float] = {}
    for cam, off in (raw_offsets or {}).items():
        if isinstance(off, dict) and off.get("status") in ("REFERENCE", "RESOLVED"):
            offsets[str(cam)] = float(off.get("delta", 0.0) or 0.0)

    wagons = getattr(state, "wagons", None)
    if wagons is None and isinstance(state, dict):
        wagons = state.get("wagons") or []

    for cam in cameras:
        lct = tracks_by_camera.get(cam)
        if lct is None or not getattr(lct, "video_path", "") or \
                not os.path.isfile(getattr(lct, "video_path", "")):
            plan.camera_status[cam] = "NO_VIDEO"
            plan.skipped.append({"camera_id": cam, "reason": "video unavailable"})
            continue
        if cam not in offsets:
            plan.camera_status[cam] = "UNRESOLVED"
            plan.skipped.append({
                "camera_id": cam,
                "reason": ("clock offset unresolved -- no frames cached, so no "
                           "finding from this camera can be attributed to a "
                           "wagon rather than being placed at a guessed time")})
            continue

        plan.camera_status[cam] = "RESOLVED"
        delta = offsets[cam]
        fps = float(getattr(lct, "fps", 0.0) or 0.0)
        total = int(getattr(lct, "total_frames", 0) or 0)
        if fps <= 0:
            plan.camera_status[cam] = "NO_VIDEO"
            plan.skipped.append({"camera_id": cam, "reason": "non-positive fps"})
            continue

        for w in wagons:
            gid = (w.get("global_id") if isinstance(w, dict)
                   else getattr(w, "global_id", None))
            t0 = (w.get("start_time") if isinstance(w, dict)
                  else getattr(w, "start_time", None))
            t1 = (w.get("end_time") if isinstance(w, dict)
                  else getattr(w, "end_time", None))
            if gid is None or t0 is None or t1 is None:
                continue
            # MASTER seconds -> this camera's own clock -> its frames.
            f0 = int(round((float(t0) - delta) * fps))
            f1 = int(round((float(t1) - delta) * fps))
            f0, f1 = max(0, min(f0, f1)), max(f0, f1)
            if total > 0:
                f1 = min(f1, total - 1)
            if f1 < f0 or f1 < 0:
                # The wagon passed outside this camera's footage: NOT VISIBLE.
                plan.skipped.append({
                    "camera_id": cam, "global_id": gid,
                    "reason": "wagon window lies outside this camera's footage"})
                continue
            plan.windows.append(WagonWindow(str(gid), cam, f0, f1))

    plan.windows.sort(key=lambda w: (w.camera_id, w.start_frame, _gw_key(w.global_id)))
    return plan


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def build_wagon_cache(
    plan: CachePlan,
    tracks_by_camera: Dict[str, Any],
    cfg: Optional[WagonCacheConfig] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Write the planned frames. ONE sequential decode pass per camera.

    A single forward sweep per camera is what keeps this affordable: seeking per
    wagon would cost thousands of seeks on a long train, and decoding per wagon
    would decode the video once per wagon.
    """
    import cv2                                          # noqa: WPS433 (lazy)

    cfg = cfg or WagonCacheConfig()
    stats: Dict[str, Any] = {"frames_written": 0, "wagons": 0, "per_camera": {},
                             "root": plan.root}

    by_camera: Dict[str, List[WagonWindow]] = {}
    for w in plan.windows:
        by_camera.setdefault(w.camera_id, []).append(w)

    for cam, windows in sorted(by_camera.items()):
        lct = tracks_by_camera.get(cam)
        path = getattr(lct, "video_path", "")
        written = 0
        # Precompute which frames each window wants, so the sweep is a lookup.
        wanted: Dict[int, List[WagonWindow]] = {}
        for w in windows:
            frames = list(range(w.start_frame, w.end_frame + 1,
                                max(1, cfg.every_nth)))
            if cfg.max_frames_per_wagon and len(frames) > cfg.max_frames_per_wagon:
                # CONTIGUOUS truncation, never even spacing: the trackers need
                # temporal continuity, and a spread-out sample breaks their
                # association gate exactly as an oversized stride does.
                frames = frames[:cfg.max_frames_per_wagon]
                w.truncated = True
                stats.setdefault("truncated_windows", []).append(
                    {"global_id": w.global_id, "camera_id": cam,
                     "kept_frames": len(frames),
                     "window_frames": w.n_frames,
                     "note": ("runaway guard bound: this wagon was only partly "
                              "inspected, so its verdict covers the cached "
                              "frames rather than the whole wagon")})
            outdir = wagon_camera_dir(plan.root, w.global_id, cam)
            if cfg.skip_existing and os.path.isdir(outdir) and os.listdir(outdir):
                continue
            os.makedirs(outdir, exist_ok=True)
            for f in frames:
                wanted.setdefault(f, []).append(w)

        if not wanted:
            stats["per_camera"][cam] = {"frames_written": 0,
                                        "windows": len(windows),
                                        "note": "nothing to write (cache reused)"}
            continue

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            stats["per_camera"][cam] = {"frames_written": 0, "error": "cannot open"}
            continue
        try:
            last = max(wanted)
            idx = 0
            while idx <= last:
                ok = cap.grab()          # advance without decoding
                if not ok:
                    break
                if idx in wanted:
                    ok, img = cap.retrieve()
                    if ok and img is not None:
                        for w in wanted[idx]:
                            dest = os.path.join(
                                wagon_camera_dir(plan.root, w.global_id, cam),
                                f"frame_{idx:06d}.jpg")
                            cv2.imwrite(dest, img,
                                        [cv2.IMWRITE_JPEG_QUALITY,
                                         int(cfg.jpeg_quality)])
                            written += 1
                    del img              # release immediately
                idx += 1
        finally:
            cap.release()

        stats["per_camera"][cam] = {"frames_written": written,
                                    "windows": len(windows)}
        stats["frames_written"] += written
        if verbose:
            print(f"    [CACHE/{cam}] {written} frame(s) across "
                  f"{len(windows)} wagon window(s)")

    stats["wagons"] = len({w.global_id for w in plan.windows})
    return stats


def cache_stats(root: str) -> Dict[str, Any]:
    """Count what is actually on disk, for diagnostics and tests."""
    out: Dict[str, Any] = {"root": root, "wagons": 0, "frames": 0,
                           "per_camera": {}}
    if not os.path.isdir(root):
        return out
    for gid in os.listdir(root):
        gdir = os.path.join(root, gid)
        if not os.path.isdir(gdir):
            continue
        out["wagons"] += 1
        for cam in os.listdir(gdir):
            cdir = os.path.join(gdir, cam)
            if not os.path.isdir(cdir):
                continue
            n = sum(1 for f in os.listdir(cdir)
                    if f.startswith("frame_") and f.endswith(".jpg"))
            out["frames"] += n
            rec = out["per_camera"].setdefault(cam, {"wagons": 0, "frames": 0})
            rec["frames"] += n
            if n:
                rec["wagons"] += 1
    return out


def clear_wagon_cache(root: str, verbose: bool = True) -> int:
    """Delete the cache. Returns how many wagon directories were removed.

    Called between trains: the cache is per-train state, and leaving GW_1..GW_57
    of one train on disk would let the next train's GW_1 read the previous
    train's frames. That is the state-leak this prevents.
    """
    if not os.path.isdir(root):
        return 0
    n = sum(1 for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    shutil.rmtree(root, ignore_errors=True)
    if verbose and n:
        print(f"    [CACHE] cleared {n} wagon directory(ies) from {root}")
    return n
