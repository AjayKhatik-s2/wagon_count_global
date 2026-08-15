"""Damage detection + problem-frame extraction.

Two damage-model flavours are supported:

* ``"top"`` — 3-class model (V4_top_damage.pt) emitting:
    Floor__probable_damage, Floor_damage, Inner_wall_damage
    Internally mapped to: floor_dmg_probable, floor_dmg, inner_wall_dmg
* ``"side"`` — 4-class model (V4_side_damage.pt) emitting:
    closed_door, damage, open_door, partially_closed
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional

import cv2
import pandas as pd
from ultralytics import YOLO

from .bands import analyze_detection_bands

# VENDORING NOTE (the ONLY edit to this file besides the import block above).
# The legacy source did `from tqdm import tqdm`. tqdm is a progress-bar library
# and is not a dependency of this project, so an unconditional import would make
# the whole damage feature unimportable on a box without it. The fallback below
# is a pass-through iterator: it changes what is PRINTED, never what is
# computed. Every detection, band, filter and verdict below is byte-for-byte the
# legacy implementation.
try:  # pragma: no cover - trivial import guard
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):  # type: ignore[misc]
        return iterable


# ---------------------------------------------------------------------------
# Class-style maps used for drawing bounding boxes.
# (BGR colors — matches OpenCV.)
# ---------------------------------------------------------------------------

TOP_CLASS_STYLE: Mapping[str, dict] = {
    "floor_dmg_probable": {"color": (0, 165, 255), "label": "FLOOR PROBABLE DMG"},
    "Floor__probable_damage": {"color": (0, 165, 255), "label": "FLOOR PROBABLE DMG"},
    "floor_dmg":          {"color": (0, 0, 255),   "label": "FLOOR DMG"},
    "Floor_damage":       {"color": (0, 0, 255),   "label": "FLOOR DMG"},
    "inner_wall_dmg":     {"color": (0, 0, 255),   "label": "INNER WALL DMG"},
    "Inner_wall_damage":  {"color": (0, 0, 255),   "label": "INNER WALL DMG"},
}

SIDE_CLASS_STYLE: Mapping[str, dict] = {
    "damage":           {"color": (0, 0, 255),   "label": "DAMAGE"},
    "open_door":        {"color": (0, 0, 255),   "label": "DOOR OPEN"},
    "closed_door":      {"color": (0, 255, 0),   "label": "DOOR CLOSE"},
    "partially_closed": {"color": (0, 165, 255), "label": "DOOR PARTIAL"},
}

TOP_CLASSES = list(TOP_CLASS_STYLE.keys())
SIDE_CLASSES = list(SIDE_CLASS_STYLE.keys())

# Maps raw model class name → internal snake_case column name (TOP flavour only).
# V4_top_damage.pt emits: Floor__probable_damage, Floor_damage, Inner_wall_damage
TOP_MODEL_CLASS_MAP: dict[str, str] = {
    "Floor__probable_damage": "floor_dmg_probable",
    "Floor_damage":           "floor_dmg",
    "Inner_wall_damage":      "inner_wall_dmg",
}
_TOP_INTERNAL_TO_MODEL: dict[str, str] = {v: k for k, v in TOP_MODEL_CLASS_MAP.items()}


# ---------------------------------------------------------------------------
# Damage-scan filters
# ---------------------------------------------------------------------------

# Segment types with no wagon body to inspect. reporting/json_builder.py already
# discards damage for these (it counts the segment and `continue`s before reading
# any damage field), so scanning them was pure waste — one full model pass per
# frame of every engine and brakevan, thrown away. Worse, a hit still reached
# ProblemFrameExtractor and shipped an annotated "damaged engine" frame into the
# report. Skipping them is therefore both a false-positive fix and a net CPU
# saving. Membership matches json_builder's own checks exactly, so nothing that
# would have been reported can be skipped here: `display_segment_type` maps
# engine→engine and brakevan→brakevan in both flavours, and every other class
# (including unexpected ones) is treated as a wagon on both sides.
NON_WAGON_SEGMENT_TYPES = frozenset({"engine", "brakevan"})

# TOP flavour only. A loaded wagon's floor is not observable from above — cargo
# covers it — so a floor_dmg / floor_dmg_probable box on a loaded wagon is cargo
# texture misread as floor damage. Inner-wall damage is NOT suppressed: the walls
# stay visible above the load, so those detections remain meaningful.
FLOOR_DAMAGE_CLASSES = frozenset({"floor_dmg", "floor_dmg_probable"})

# Internal segment_type emitted by classify_segment_type for a loaded wagon
# (see train_detection/classifier.DEFAULT_CLASS_ALIASES: wagon_filled→wagon_loaded).
LOADED_SEGMENT_TYPE = "wagon_loaded"

# Door classes voted over for the side flavour's door_status, in the precedence
# order used as a tie-break (see _door_status). Precedence-only was the previous
# behaviour; it is now the tie-break for an actual majority vote.
_DOOR_CLASS_PRECEDENCE = ("open_door", "partially_closed", "closed_door")
_DOOR_STATUS_LABEL = {
    "open_door": "open",
    "partially_closed": "partially_closed",
    "closed_door": "closed",
}


@dataclass
class DamageDetector:
    damage_model: Optional[YOLO]
    flavour: str  # "top" or "side"
    confidence: float
    band_gap_tolerance: int = 5
    edge_skip_frames: int = 10
    logger: Optional[logging.Logger] = None
    # Skip the whole per-frame model pass on engine / brakevan segments — their
    # damage rows are discarded by json_builder anyway. See
    # NON_WAGON_SEGMENT_TYPES. Set False to restore the previous scan-everything
    # behaviour.
    skip_non_wagon_segments: bool = True
    # TOP flavour only: drop floor_dmg / floor_dmg_probable boxes on loaded
    # wagons. See FLOOR_DAMAGE_CLASSES. No-op for the side flavour, which has no
    # floor classes.
    suppress_floor_damage_on_loaded: bool = True
    # PDF §4 "Minimum Track Duration". A damage band spanning fewer than this many
    # distinct frames is a flicker, not a feature: the wagon is in view for tens of
    # frames, so a real defect is seen repeatedly. Bands already carry frame_count
    # (analyze_detection_bands), so this is a comparison, not a computation. 1
    # disables the filter and restores the previous behaviour.
    min_band_frames: int = 3
    # PDF §11 "Classification Stability", applied as a GUARD rather than a
    # rejection: only trust an engine/brakevan label (and therefore skip the scan)
    # when that class won at least this share of the segment's classification
    # votes. A thin-margin non-wagon label falls through and is scanned as a wagon,
    # so an unstable segment is never silently dropped from inspection. 0.0 trusts
    # any winning label.
    min_non_wagon_dominance: float = 0.80

    def __post_init__(self) -> None:
        self.log = self.logger or logging.getLogger(__name__)
        if self.flavour not in ("top", "side"):
            raise ValueError(f"Unknown damage flavour: {self.flavour}")
        self._style = TOP_CLASS_STYLE if self.flavour == "top" else SIDE_CLASS_STYLE
        self._classes = TOP_CLASSES if self.flavour == "top" else SIDE_CLASSES

    # ------------------------------------------------------------------

    @staticmethod
    def _segment_type_of(segment) -> str:
        """Read a segment's classified type, defaulting to ``"wagon"``.

        WagonSegmenter merges the classification result with ``how="left"``, so a
        segment_type can in principle arrive missing or NaN. Defaulting to
        ``"wagon"`` keeps an unclassifiable segment in the scan rather than
        silently skipping a real wagon.
        """
        value = segment.get("segment_type")
        return value if isinstance(value, str) else "wagon"

    @staticmethod
    def _dominance_of(segment) -> float:
        """Share of classification votes won by the segment's chosen class.

        Reads ``type_dominance`` if WagonSegmenter supplied it, else derives it
        from the ``vote_counts`` dict that classify_segment_type already returns.
        Returns 1.0 when there is nothing to go on, so a segment carrying no vote
        record is treated as confidently labelled (matching the previous
        behaviour, where the label was trusted unconditionally).
        """
        dominance = segment.get("type_dominance")
        if isinstance(dominance, (int, float)) and dominance == dominance:  # not NaN
            return float(dominance)
        votes = segment.get("vote_counts")
        if isinstance(votes, dict) and votes:
            total = sum(votes.values())
            if total > 0:
                return max(votes.values()) / total
        return 1.0

    def _is_trusted_non_wagon(self, segment) -> bool:
        """True when this segment is an engine/brakevan we can trust enough to skip.

        Combines PDF §9 (engine/tail filtering) with PDF §11 (classification
        stability): the label must BE engine/brakevan *and* have won a dominant
        share of the votes. A flip-flopping segment fails the second test and is
        scanned as a wagon rather than dropped from inspection.
        """
        if self._segment_type_of(segment) not in NON_WAGON_SEGMENT_TYPES:
            return False
        return self._dominance_of(segment) >= self.min_non_wagon_dominance

    def _drop_short_bands(self, bands: list[dict]) -> tuple[list[dict], int]:
        """PDF §4: discard bands spanning fewer than ``min_band_frames`` frames.

        ``frame_count`` is already set by analyze_detection_bands, so this is a
        filter over a short list — O(bands), no new computation and no image work.
        Returns ``(kept, n_dropped)``.
        """
        if self.min_band_frames <= 1 or not bands:
            return bands, 0
        kept = [b for b in bands if b.get("frame_count", 0) >= self.min_band_frames]
        return kept, len(bands) - len(kept)

    def _door_status(self, bands_by_door: Mapping[str, list]) -> str:
        """PDF §5 majority voting for door state.

        Previously this was pure precedence — ANY surviving open_door band made the
        door "open", so one spurious band outvoted a whole segment of closed_door
        evidence. Now the classes are ranked by how many distinct frames actually
        support each, using the frame_count the bands already carry. Ties keep the
        original precedence (open > partial > closed), so the outcome only changes
        when another state genuinely dominates.
        """
        best_label = "closed"
        best_frames = 0
        for cls_name in _DOOR_CLASS_PRECEDENCE:
            frames = sum(b.get("frame_count", 0) for b in bands_by_door.get(cls_name, ()))
            if frames > best_frames:
                best_frames = frames
                best_label = _DOOR_STATUS_LABEL[cls_name]
        return best_label

    def _log_filter_summary(
        self, label: str, skipped: int, suppressed: int, short_bands: int = 0,
    ) -> None:
        """One line per scan, not per object — keeps disk I/O off the hot path."""
        if not (skipped or suppressed or short_bands):
            return
        self.log.info(
            "Damage scan (%s) filters: skipped %d non-wagon segment(s) "
            "[engine/brakevan]; suppressed %d floor-damage box(es) on loaded "
            "wagon(s); dropped %d damage band(s) shorter than %d frame(s)",
            label, skipped, suppressed, short_bands, self.min_band_frames,
        )

    # ------------------------------------------------------------------

    def detect(
        self, segment_summary_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Returns ``(summary_df, frame_detections_df)``.

        ``frame_detections_df`` carries every raw per-frame damage detection
        (not just each band's best frame) — columns ``frame_number,
        class_name, confidence, x1, y1, x2, y2, wagon_id`` — so the combined
        annotated video (build_annotated_video) can draw a box on every frame
        a damage class was actually detected on, not just band summaries.
        ``class_name`` uses the same internal names that key
        ``TOP_CLASS_STYLE`` / ``SIDE_CLASS_STYLE``.
        """
        if self.flavour == "top":
            return self._detect_top(segment_summary_df)
        return self._detect_side(segment_summary_df)

    # ------------------------------------------------------------------

    def _empty_top_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "wagon_id",
            "floor_dmg_detected", "inner_wall_dmg_detected", "floor_dmg_probable_detected",
            "damage_detected", "probable_damage_detected",
            "floor_dmg_best_frames", "inner_wall_dmg_best_frames", "floor_dmg_probable_best_frames",
            "floor_dmg_band_info", "inner_wall_dmg_band_info", "floor_dmg_probable_band_info",
        ])

    def _empty_side_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "wagon_id",
            "damage_detected", "door_status", "door_close_detected", "door_partial_detected",
            "damage_best_frames", "open_door_best_frames", "closed_door_best_frames",
            "partially_closed_best_frames",
            "damage_band_info", "open_door_band_info", "closed_door_band_info",
            "partially_closed_band_info",
        ])

    @staticmethod
    def _empty_frame_detections_df() -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "frame_number", "class_name", "confidence", "x1", "y1", "x2", "y2", "wagon_id",
        ])

    # ------------------------------------------------------------------

    def _iter_segment_frames(self, segment):
        start_frame = int(segment["start_frame"])
        end_frame = int(segment["end_frame"])
        segment_dir = segment["directory"]
        safe_start = start_frame + self.edge_skip_frames
        safe_end = end_frame - self.edge_skip_frames
        if safe_end <= safe_start:
            return
        for frame_num in range(safe_start, safe_end + 1):
            path = os.path.join(segment_dir, f"frame_{frame_num:06d}.jpg")
            if not os.path.exists(path):
                continue
            frame = cv2.imread(path)
            if frame is None:
                continue
            yield frame_num, frame

    def _model_predict(self, frame):
        return self.damage_model.predict(
            source=frame, verbose=False, conf=self.confidence
        )

    def _band_best_frames(self, detections, bands):
        """For each band, find its highest-confidence frame and attach it."""
        best = []
        for band in bands:
            band_frames = set(band["frames"])
            band_dets = [(fn, c) for fn, c, *_ in detections if fn in band_frames]
            if not band_dets:
                continue
            best_frame, best_conf = max(band_dets, key=lambda x: x[1])
            band["best_frame"] = best_frame
            band["best_confidence"] = best_conf
            best.append(best_frame)
        return best

    # ------------------------------------------------------------------

    def _detect_top(
        self, segment_summary_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.damage_model is None or segment_summary_df.empty:
            return self._empty_top_df(), self._empty_frame_detections_df()

        rows = []
        all_frame_detections: list[dict] = []
        n_skipped = n_suppressed = n_short = 0
        for _, segment in tqdm(
            segment_summary_df.iterrows(),
            total=len(segment_summary_df),
            desc="Damage scan (top)",
        ):
            seg_type = self._segment_type_of(segment)

            # Engine / brakevan (with a dominant label): emit no row at all.
            # json_builder discards damage for these, and ProblemFrameExtractor
            # joins on wagon_id with how="inner", so an absent row also stops the
            # engine from ever producing a problem frame. Placed before
            # _iter_segment_frames so the segment's JPEGs are never decoded either.
            if self.skip_non_wagon_segments and self._is_trusted_non_wagon(segment):
                n_skipped += 1
                continue

            suppress_floor = (
                self.suppress_floor_damage_on_loaded
                and seg_type == LOADED_SEGMENT_TYPE
            )

            # Keys are internal names; model names are mapped via TOP_MODEL_CLASS_MAP.
            detections_by_class: dict[str, list] = {c: [] for c in TOP_CLASSES}

            iter_frames = list(self._iter_segment_frames(segment))
            if not iter_frames:
                rows.append({
                    "wagon_id": int(segment["segment_id"]),
                    "floor_dmg_detected": False,
                    "inner_wall_dmg_detected": False,
                    "floor_dmg_probable_detected": False,
                    "damage_detected": False,
                    "probable_damage_detected": False,
                    "floor_dmg_best_frames": [],
                    "inner_wall_dmg_best_frames": [],
                    "floor_dmg_probable_best_frames": [],
                    "floor_dmg_band_info": [],
                    "inner_wall_dmg_band_info": [],
                    "floor_dmg_probable_band_info": [],
                })
                continue

            seg_id = int(segment["segment_id"])
            for frame_num, frame in iter_frames:
                for r in self._model_predict(frame):
                    for box in r.boxes:
                        raw_name = self.damage_model.names[int(box.cls[0])]
                        cls_name = TOP_MODEL_CLASS_MAP.get(raw_name)
                        if cls_name is None:
                            continue
                        # Loaded wagon: the floor is under the cargo, so a floor
                        # box here is cargo texture. Dropped before banding, so it
                        # reaches neither the report nor the annotated video.
                        if suppress_floor and cls_name in FLOOR_DAMAGE_CLASSES:
                            n_suppressed += 1
                            continue
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                        detections_by_class[cls_name].append(
                            (frame_num, conf, x1, y1, x2, y2)
                        )
                        all_frame_detections.append({
                            "frame_number": frame_num,
                            "class_name": cls_name,
                            "confidence": conf,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "wagon_id": seg_id,
                        })

            bands_by_class = {}
            for cls, dets in detections_by_class.items():
                kept, dropped = self._drop_short_bands(
                    analyze_detection_bands(dets, gap_tolerance=self.band_gap_tolerance)
                )
                bands_by_class[cls] = kept
                n_short += dropped
            best_by_class = {
                cls: self._band_best_frames(detections_by_class[cls], bands_by_class[cls])
                for cls in TOP_CLASSES
            }
            floor_dmg = bool(bands_by_class["floor_dmg"])
            inner_wall_dmg = bool(bands_by_class["inner_wall_dmg"])
            floor_prob = bool(bands_by_class["floor_dmg_probable"])

            rows.append({
                "wagon_id": int(segment["segment_id"]),
                "floor_dmg_detected": floor_dmg,
                "inner_wall_dmg_detected": inner_wall_dmg,
                "floor_dmg_probable_detected": floor_prob,
                "damage_detected": floor_dmg or inner_wall_dmg,
                "probable_damage_detected": floor_prob,
                "floor_dmg_best_frames": best_by_class["floor_dmg"],
                "inner_wall_dmg_best_frames": best_by_class["inner_wall_dmg"],
                "floor_dmg_probable_best_frames": best_by_class["floor_dmg_probable"],
                "floor_dmg_band_info": bands_by_class["floor_dmg"],
                "inner_wall_dmg_band_info": bands_by_class["inner_wall_dmg"],
                "floor_dmg_probable_band_info": bands_by_class["floor_dmg_probable"],
            })

        self._log_filter_summary("top", n_skipped, n_suppressed, n_short)
        # Typed empty frames when nothing was scanned (e.g. a rake whose only
        # segments were engine/brakevan): a bare pd.DataFrame([]) has no columns,
        # which would trip ProblemFrameExtractor's missing-column warning and
        # pdf_builder's damage_detected lookup.
        if not rows:
            return self._empty_top_df(), self._empty_frame_detections_df()
        if not all_frame_detections:
            return pd.DataFrame(rows), self._empty_frame_detections_df()
        return pd.DataFrame(rows), pd.DataFrame(all_frame_detections)

    def _detect_side(
        self, segment_summary_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.damage_model is None or segment_summary_df.empty:
            return self._empty_side_df(), self._empty_frame_detections_df()

        rows = []
        all_frame_detections: list[dict] = []
        n_skipped = n_short = 0
        for _, segment in tqdm(
            segment_summary_df.iterrows(),
            total=len(segment_summary_df),
            desc="Damage scan (side)",
        ):
            # Engine / brakevan (with a dominant label): skipped for the same reason
            # as the top flavour — json_builder discards their damage and the
            # problem-frame join is inner. There is no floor class on the side
            # model, so the loaded-wagon filter does not apply here.
            if self.skip_non_wagon_segments and self._is_trusted_non_wagon(segment):
                n_skipped += 1
                continue

            # Internal keys match V4_side_damage model class names directly.
            detections_by_class: dict[str, list] = {c: [] for c in SIDE_CLASSES}

            seg_id = int(segment["segment_id"])
            for frame_num, frame in self._iter_segment_frames(segment):
                for r in self._model_predict(frame):
                    for box in r.boxes:
                        cls_name = self.damage_model.names[int(box.cls[0])]
                        if cls_name not in detections_by_class:
                            continue
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                        detections_by_class[cls_name].append(
                            (frame_num, conf, x1, y1, x2, y2)
                        )
                        all_frame_detections.append({
                            "frame_number": frame_num,
                            "class_name": cls_name,
                            "confidence": conf,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "wagon_id": seg_id,
                        })

            banded: dict[str, list] = {}
            for cls_name in SIDE_CLASSES:
                kept, dropped = self._drop_short_bands(
                    analyze_detection_bands(
                        detections_by_class[cls_name],
                        gap_tolerance=self.band_gap_tolerance,
                    )
                )
                banded[cls_name] = kept
                n_short += dropped

            damage_bands = banded["damage"]
            open_door_bands = banded["open_door"]
            closed_door_bands = banded["closed_door"]
            partial_bands = banded["partially_closed"]

            # PDF §5: majority vote over the door classes instead of the previous
            # any-open-band-wins precedence.
            door_status = self._door_status(banded)

            rows.append({
                "wagon_id": int(segment["segment_id"]),
                "damage_detected": bool(damage_bands),
                "door_status": door_status,
                "door_close_detected": bool(closed_door_bands),
                "door_partial_detected": bool(partial_bands),
                "damage_best_frames": self._band_best_frames(
                    detections_by_class["damage"], damage_bands
                ),
                "open_door_best_frames": self._band_best_frames(
                    detections_by_class["open_door"], open_door_bands
                ),
                "closed_door_best_frames": self._band_best_frames(
                    detections_by_class["closed_door"], closed_door_bands
                ),
                "partially_closed_best_frames": self._band_best_frames(
                    detections_by_class["partially_closed"], partial_bands
                ),
                "damage_band_info": damage_bands,
                "open_door_band_info": open_door_bands,
                "closed_door_band_info": closed_door_bands,
                "partially_closed_band_info": partial_bands,
            })

        self._log_filter_summary("side", n_skipped, 0, n_short)
        if not rows:
            return self._empty_side_df(), self._empty_frame_detections_df()
        if not all_frame_detections:
            return pd.DataFrame(rows), self._empty_frame_detections_df()
        return pd.DataFrame(rows), pd.DataFrame(all_frame_detections)


# ---------------------------------------------------------------------------
# Problem-frame extraction (annotated cropped frames per detection)
# ---------------------------------------------------------------------------


class ProblemFrameExtractor:
    """Renders annotated images for each high-confidence damage detection."""

    def __init__(
        self,
        damage_model: Optional[YOLO],
        flavour: str,
        damage_confidence: float,
        logger: Optional[logging.Logger] = None,
    ):
        self.damage_model = damage_model
        self.flavour = flavour
        self.damage_confidence = damage_confidence
        self.logger = logger or logging.getLogger(__name__)
        self._style = TOP_CLASS_STYLE if flavour == "top" else SIDE_CLASS_STYLE

    # ------------------------------------------------------------------

    def annotate(
        self, frame_path: str, detection_class: str, save_dir: Optional[str]
    ) -> list[dict]:
        if self.damage_model is None or not os.path.exists(frame_path):
            return []
        frame = cv2.imread(frame_path)
        if frame is None:
            return []

        results = self.damage_model.predict(
            source=frame, verbose=False, conf=self.damage_confidence
        )
        annotated = frame.copy()
        all_boxes: list[dict] = []
        found = False
        # For TOP flavour, internal names differ from raw model class names.
        model_cls_target = (
            _TOP_INTERNAL_TO_MODEL.get(detection_class, detection_class)
            if self.flavour == "top" else detection_class
        )

        for r in results:
            for box in r.boxes:
                cls_name = self.damage_model.names[int(box.cls[0])]
                if cls_name != model_cls_target:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                style_cls = (
                    TOP_MODEL_CLASS_MAP.get(cls_name, cls_name)
                    if self.flavour == "top" else cls_name
                )
                style = self._style.get(
                    style_cls, {"color": (255, 255, 0), "label": cls_name.upper()}
                )
                color = style["color"]
                label = f"{style['label']} {conf:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                ls, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                label_y = max(y1, ls[1] + 10)
                cv2.rectangle(
                    annotated, (x1, label_y - ls[1] - 10),
                    (x1 + ls[0], label_y), color, -1,
                )
                cv2.putText(
                    annotated, label, (x1, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                )
                all_boxes.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_name": cls_name,
                })
                found = True

        annotated_path = None
        if found and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(frame_path))[0]
            annotated_path = os.path.join(
                save_dir, f"{base}_{detection_class}_annotated.jpg"
            )
            cv2.imwrite(annotated_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
        for b in all_boxes:
            b["annotated_image_path"] = annotated_path
        return all_boxes

    # ------------------------------------------------------------------

    def extract(
        self,
        segment_summary_df: pd.DataFrame,
        damage_results_df: pd.DataFrame,
        work_dir: str,
    ) -> pd.DataFrame:
        empty_cols = [
            "wagon_id", "problem_type", "frame_number",
            "frame_path", "annotated_image_path", "bounding_box",
        ]
        if damage_results_df.empty:
            return pd.DataFrame(columns=empty_cols)

        if self.flavour == "top":
            band_cols = {
                "floor_dmg":          "floor_dmg_band_info",
                "inner_wall_dmg":     "inner_wall_dmg_band_info",
                "floor_dmg_probable": "floor_dmg_probable_band_info",
            }
        else:
            band_cols = {
                "damage":           "damage_band_info",
                "open_door":        "open_door_band_info",
                "closed_door":      "closed_door_band_info",
                "partially_closed": "partially_closed_band_info",
            }

        required = ["wagon_id"] + list(band_cols.values())
        missing = [c for c in required if c not in damage_results_df.columns]
        if missing:
            self.logger.warning("Missing damage columns: %s", missing)
            return pd.DataFrame(columns=empty_cols)

        annotated_dir = os.path.join(work_dir, "annotated_problem_frames")
        os.makedirs(annotated_dir, exist_ok=True)

        merged = segment_summary_df.merge(
            damage_results_df[required],
            left_on="segment_id", right_on="wagon_id", how="inner",
        )
        if merged.empty:
            return pd.DataFrame(columns=empty_cols)

        rows: list[dict] = []
        for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Problem frames"):
            wagon_id = row["segment_id"]
            segment_dir = row["directory"]
            for class_name, band_col in band_cols.items():
                bands = row.get(band_col, [])
                if not isinstance(bands, list):
                    continue
                for band in bands:
                    if "best_frame" not in band:
                        continue
                    frame_num = band["best_frame"]
                    frame_path = os.path.join(segment_dir, f"frame_{frame_num:06d}.jpg")
                    if not os.path.exists(frame_path):
                        continue
                    bboxes = self.annotate(frame_path, class_name, annotated_dir)
                    annotated_path = bboxes[0]["annotated_image_path"] if bboxes else None
                    clean = [{k: v for k, v in b.items() if k != "annotated_image_path"} for b in bboxes]
                    rows.append({
                        "wagon_id": wagon_id,
                        "problem_type": class_name,
                        "frame_number": frame_num,
                        "frame_path": frame_path,
                        "annotated_image_path": annotated_path,
                        "bounding_box": clean,
                    })
        return pd.DataFrame(rows)
