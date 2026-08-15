"""Relative frame-position selection for locomotive report frames.

The same locomotive is seen by four cameras with different timelines, frame
rates and visible spans, so nothing absolute can be held constant between
them — not the frame number, not the timestamp, not the number of frames the
detector fired on. The only comparable quantity is each frame's **relative
position within that camera's view of the loco**.

These helpers turn a span of available frames into N frames at evenly spaced
fractions of it, so ``start``/``mid1``/``mid2``/``end`` mean the same 0 / 33 /
66 / 100 % of the loco's pass in *every* view, and a reviewer can put the four
``inspection_data.json`` blobs side by side and compare like for like.

Why the loco path needs this and the wagon path does not: a wagon segment is
bounded by the two gap bands around it, so every wagon already spans a
comparable slice of its own vehicle, and the wagon fractions
(0.25 / 0.55 / 0.80) are cosmetic offsets tuned per camera to frame the wagon
body. A loco band is bounded by *detections*, which fire unevenly across the
pass — so indexing into the detected-frame list put one camera's "start" in
the middle of the loco and another's "end" three-quarters of the way through.
Hence: span-relative here, unchanged fractions for wagons. The two sets are
**not** interchangeable.
"""
from __future__ import annotations

import bisect
from typing import Iterable, Optional, Sequence, Tuple

# Used when neither the camera YAML nor the caller names the loco positions.
# Four names → 0 / 33 / 66 / 100 %. Must stay in the backend's position
# vocabulary, which is shared with wagon_frames[].position.
DEFAULT_LOCO_POSITION_NAMES: Tuple[str, ...] = ("start", "mid1", "mid2", "end")


def even_relative_positions(n: int) -> Tuple[float, ...]:
    """Return ``n`` fractions evenly spaced over the closed range [0.0, 1.0].

    ``n=3`` → ``(0.0, 0.5, 1.0)``; ``n=4`` → ``(0.0, 0.333…, 0.666…, 1.0)``.
    Both endpoints are included, so the first frame is the loco's entry and the
    last is its exit. A single requested frame has no start-to-end progression
    to preserve, so it is placed at the midpoint of the pass instead of at the
    entry, which is the more representative single image.
    """
    if n <= 0:
        return ()
    if n == 1:
        return (0.5,)
    return tuple(i / (n - 1) for i in range(n))


def resolve_loco_positions(
    position_names: Optional[Sequence[str]] = None,
    positions: Optional[Sequence[float]] = None,
) -> Tuple[Tuple[float, ...], Tuple[str, ...]]:
    """Resolve the (fractions, names) pair used for loco report frames.

    ``positions`` is an optional explicit override; when it is omitted (the
    recommended setup) the fractions are derived from the *number of names* via
    :func:`even_relative_positions`. Deriving rather than copying decimals into
    four camera YAMLs is what keeps the cameras in step — the uniformity cannot
    drift because there is only one place it is computed.

    Either argument being absent *or empty* falls back to the default names,
    matching how the rest of the config layer treats an unset list.
    """
    names = tuple(position_names or DEFAULT_LOCO_POSITION_NAMES)
    fractions = (
        tuple(float(p) for p in positions)
        if positions
        else even_relative_positions(len(names))
    )
    if len(fractions) != len(names):
        raise ValueError(
            f"loco_representative_positions ({fractions}) and "
            f"loco_representative_position_names ({names}) must have the same length"
        )
    if any(not 0.0 <= f <= 1.0 for f in fractions):
        raise ValueError(
            f"loco_representative_positions must all be fractions in [0.0, 1.0]: {fractions}"
        )
    return fractions, names


def _nearest_index(frames: Sequence[int], target: float) -> int:
    """Index in sorted ``frames`` of the value closest to ``target``."""
    pos = bisect.bisect_left(frames, target)
    if pos == 0:
        return 0
    if pos >= len(frames):
        return len(frames) - 1
    before, after = frames[pos - 1], frames[pos]
    return pos if (after - target) < (target - before) else pos - 1


def select_relative_frames(
    available_frames: Iterable[int], positions: Sequence[float],
) -> list[int]:
    """Pick one frame per fraction in ``positions``, span-relative.

    ``available_frames`` is every frame that can actually be produced for this
    loco in this camera — the inclusive entry→exit range for a video-backed
    band, or the frames whose JPG exists on disk for a segment-backed one. Its
    first and last values define the span; fraction ``f`` targets
    ``first + f * (last - first)`` and is then snapped to the nearest available
    frame.

    Two properties the callers rely on:

    * **One frame per position, always** — a band with fewer available frames
      than requested positions yields repeats of the closest valid frames
      rather than dropping positions, so the JSON shape is identical across
      cameras even when one of them barely saw the loco. (Same convention as
      :func:`ocr.three_frame_sheet.select_frame_positions`.)
    * **Monotonic** — the returned frames never step backwards, so the
      start-to-end progression survives both the snapping above and any
      unsorted ``positions``.

    Snapping is nearest-value, not spread-out: given an ``available_frames``
    with a large interior hole, two adjacent positions can land on the same
    frame at the hole's edge. Both callers pass a contiguous range (the band's
    entry→exit span, or a segment's on-disk frames, which are written
    consecutively until the first read failure), so that case does not arise in
    the pipeline — but do not assume distinct frames when passing a sparse list.
    """
    frames = sorted({int(f) for f in available_frames})
    if not frames or not positions:
        return []

    first, last = frames[0], frames[-1]
    span = last - first

    picked: list[int] = []
    floor_idx = 0
    for fraction in positions:
        idx = _nearest_index(frames, first + fraction * span)
        idx = max(idx, floor_idx)
        picked.append(frames[idx])
        floor_idx = idx
    return picked
