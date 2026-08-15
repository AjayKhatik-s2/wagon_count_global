"""Upload inspection artifacts (frames + JSON) to S3.

This module is the SINGLE owner of the file-naming convention seen by the
backend ingestion pipeline. Patterns below match the original notebooks
verbatim so the consumer schema does not break.

S3 layout:

    s3://{artifact_bucket}/{camera_folder}/{YYYY-MM-DD_HH-MM-SS}/
        wagon_frames/
            # top flavour:
            {seg_type_label}_{wagon_count:03d}_frame_{frame_num:06d}_{pos_label}.jpg
            # side flavour:
            w{wagon_count}_frame_{frame_num:06d}.jpg
        loco_frames/
            loco_{loco_id:03d}.jpg
        problem_frames/
            {seg_type_label}_{wagon_count:03d}_{problem_type}_frame_{frame_num:06d}.jpg
        wagon_numbers/
            wagon_number_segment_{segment_id:03d}.jpg  # frame sent to Rekognition
        loco_numbers/
            loco_number_band_{loco_id:03d}.jpg          # frame sent to Rekognition
        inspection_data.json

JPEG quality is 95 throughout (matches notebooks).

The publisher returns rich per-segment metadata that the JSON builder folds
into the v2 schema — each wagon frame carries its position name (start /
mid1 / mid2 / end) which appears verbatim under
`wagon_segments[i].wagon_frames[j].position`.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import pandas as pd

from .s3 import S3Client
from .serialization import json_safe
from .url_utils import s3_object_url, split_bucket_prefix


# Internal display names — match the notebooks' `_display_type()` helper.
TOP_DISPLAY = {
    "wagon": "wagon_empty",
    "wagon_loaded": "wagon_loaded",
    "engine": "engine",
    "brakevan": "brakevan",
}
SIDE_DISPLAY = {
    "wagon": "wagon",
    "wagon_loaded": "wagon",  # side cameras don't differentiate
    "engine": "engine",
    "brakevan": "brakevan",
}


def display_segment_type(seg_type: str, flavour: str) -> str:
    """Map internal segment_type to the JSON-emitted label.

    Top flavour normalises ``wagon`` → ``wagon_empty``. Side flavour collapses
    ``wagon`` and ``wagon_loaded`` to a single ``wagon`` label.
    """
    if flavour == "top":
        return TOP_DISPLAY.get(seg_type, seg_type)
    return SIDE_DISPLAY.get(seg_type, seg_type)


class ArtifactPublisher:
    """Owns the artifact directory layout and uploads it to S3."""

    DEFAULT_REPRESENTATIVE_POSITIONS = (0.25, 0.55, 0.80)
    DEFAULT_REPRESENTATIVE_POSITION_NAMES = ("start", "mid1", "end")

    def __init__(
        self,
        s3: S3Client,
        artifact_bucket: str,
        region: str,
        camera_folder: str,
        damage_flavour: str = "top",
        representative_positions: Optional[tuple[float, ...]] = None,
        representative_position_names: Optional[tuple[str, ...]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.s3 = s3
        self.artifact_bucket = artifact_bucket
        self.region = region
        self.camera_folder = camera_folder
        self.damage_flavour = damage_flavour
        self.logger = logger or logging.getLogger(__name__)

        positions = tuple(
            representative_positions or self.DEFAULT_REPRESENTATIVE_POSITIONS
        )
        names = tuple(
            representative_position_names or self.DEFAULT_REPRESENTATIVE_POSITION_NAMES
        )
        if len(positions) != len(names):
            raise ValueError(
                f"representative_positions ({positions}) and "
                f"representative_position_names ({names}) must have the same length"
            )
        self.representative_positions = positions
        self.representative_position_names = names

    # ------------------------------------------------------------------

    def _key(self, timestamp_str: str, sub: str) -> str:
        return f"{self.camera_folder}/{timestamp_str}/{sub}".replace("\\", "/")

    def _https(self, key: str) -> str:
        return s3_object_url(self.artifact_bucket, key, self.region)

    # ------------------------------------------------------------------
    # filename builders (the byte-for-byte sensitive bit)
    # ------------------------------------------------------------------

    def _wagon_frame_filename(
        self, seg_type_label: str, wagon_count: Optional[int],
        frame_num: int, pos_label: str,
    ) -> str:
        if self.damage_flavour == "side":
            # `w{wagon_count}_frame_{frame_num:06d}.jpg` — wagon_count is an int
            # (no zero-padding); side notebooks do not include the position label
            # in the filename (it's carried in the JSON instead).
            return f"w{wagon_count}_frame_{frame_num:06d}.jpg"
        # top flavour — seg_type_label encodes wagon_empty/wagon_loaded.
        wc = int(wagon_count or 0)
        return f"{seg_type_label}_{wc:03d}_frame_{frame_num:06d}_{pos_label}.jpg"

    def _problem_frame_filename(
        self, seg_type_label: str, wagon_count: Optional[int],
        problem_type: str, frame_number: int,
    ) -> str:
        wc = int(wagon_count or 0)
        return f"{seg_type_label}_{wc:03d}_{problem_type}_frame_{frame_number:06d}.jpg"

    @staticmethod
    def _loco_frame_filename(loco_id: int, pos_label: str) -> str:
        return f"loco_{loco_id:03d}_{pos_label}.jpg"

    # ------------------------------------------------------------------
    # publish
    # ------------------------------------------------------------------

    def publish(
        self,
        *,
        upload_timestamp: datetime,
        segment_summary_df: pd.DataFrame,
        loco_summary_df: pd.DataFrame,
        problem_frames_df: pd.DataFrame,
        wagon_count_map: dict[int, Optional[int]],
        local_workdir: str,
    ) -> tuple[str, dict[int, list[dict]], list[dict], list[dict]]:
        """Upload all frames and return the per-section indices used by the
        JSON builder.

        Returns
        -------
        timestamp_str
            Folder name under ``{artifact_bucket}/{camera_folder}/``.
        wagon_frames_index
            ``{segment_id: [{position, s3_key, s3_url, frame_num}, ...]}``
        loco_frames
            ``[{loco_id, position, filename, s3_key, s3_url, frame_num,
                timestamp_sec}, ...]`` — one entry per (loco, position), same
            shape family as ``wagon_frames_index``.
        problem_frames
            ``[{wagon_id, problem_type, frame_number, filename, s3_key,
                s3_url, is_annotated, annotated_image_url, bounding_box}, ...]``
        """
        timestamp_str = upload_timestamp.strftime("%Y-%m-%d_%H-%M-%S")

        # --- wagon frames ---------------------------------------------
        wagon_index: dict[int, list[dict]] = {}
        if segment_summary_df is not None and not segment_summary_df.empty:
            for _, seg in segment_summary_df.iterrows():
                seg_id = int(seg["segment_id"])
                seg_type = seg.get("segment_type", "wagon")
                wagon_count = wagon_count_map.get(seg_id)
                seg_type_label = display_segment_type(seg_type, self.damage_flavour)
                seg_dir = seg["directory"]
                start = int(seg["start_frame"])
                end = int(seg["end_frame"])
                total = max(end - start + 1, 1)

                entries: list[dict] = []
                for pos_name, pos_fraction in zip(
                    self.representative_position_names,
                    self.representative_positions,
                ):
                    frame_idx = start + int(total * pos_fraction)
                    src = os.path.join(seg_dir, f"frame_{frame_idx:06d}.jpg")
                    if not os.path.exists(src):
                        continue
                    filename = self._wagon_frame_filename(
                        seg_type_label, wagon_count, frame_idx, pos_name,
                    )
                    key = self._key(timestamp_str, f"wagon_frames/{filename}")
                    self.s3.client.upload_file(src, self.artifact_bucket, key)
                    entries.append({
                        "position": pos_name,
                        "frame_num": frame_idx,
                        "filename": filename,
                        "s3_key": key,
                        "s3_url": self._https(key),
                    })
                wagon_index[seg_id] = entries

        # --- loco frames ----------------------------------------------
        # One row per (loco_id, position) — same representative_positions /
        # representative_position_names used for wagon frames.
        loco_frames: list[dict] = []
        if loco_summary_df is not None and not loco_summary_df.empty:
            for _, row in loco_summary_df.iterrows():
                loco_id = int(row["loco_id"])
                position = row["position"]
                src = row["frame_path"]
                if not src or not os.path.exists(src):
                    continue
                filename = self._loco_frame_filename(loco_id, position)
                key = self._key(timestamp_str, f"loco_frames/{filename}")
                self.s3.client.upload_file(src, self.artifact_bucket, key)
                loco_frames.append({
                    "loco_id": loco_id,
                    "position": position,
                    "filename": filename,
                    "s3_key": key,
                    "s3_url": self._https(key),
                    "frame_num": int(row["frame_num"]),
                    "timestamp_sec": float(row["timestamp_sec"]),
                })

        # --- problem frames ------------------------------------------
        # Notebook behaviour: upload either the annotated image (preferred) or
        # the raw frame — never both. is_annotated flag distinguishes which.
        problem_frames: list[dict] = []
        if problem_frames_df is not None and not problem_frames_df.empty:
            for _, row in problem_frames_df.iterrows():
                wagon_id = int(row["wagon_id"])
                problem_type = str(row["problem_type"])
                frame_num = int(row["frame_number"])

                # Look up display label + wagon_count from the segment df.
                seg_row = segment_summary_df[
                    segment_summary_df["segment_id"] == wagon_id
                ]
                if seg_row.empty:
                    continue
                seg_type = seg_row.iloc[0].get("segment_type", "wagon")
                seg_type_label = display_segment_type(seg_type, self.damage_flavour)
                wagon_count = wagon_count_map.get(wagon_id)

                filename = self._problem_frame_filename(
                    seg_type_label, wagon_count, problem_type, frame_num,
                )
                key = self._key(timestamp_str, f"problem_frames/{filename}")

                annotated_src = row.get("annotated_image_path")
                raw_src = row.get("frame_path")
                is_annotated = bool(annotated_src) and os.path.exists(annotated_src)
                upload_src = annotated_src if is_annotated else raw_src
                if not upload_src or not os.path.exists(upload_src):
                    continue
                self.s3.client.upload_file(upload_src, self.artifact_bucket, key)
                url = self._https(key)

                # Bounding-box block — top flavour wraps it as a dict with
                # coordinates / confidence / class_name; side flavour stores
                # the raw [x1, y1, x2, y2] list (or null when missing).
                bboxes = row.get("bounding_box") or []
                bbox_block = self._format_bounding_box(bboxes)

                problem_frames.append({
                    "wagon_id": wagon_id,
                    "wagon_count": wagon_count,
                    "segment_type": seg_type_label,
                    "problem_type": problem_type,
                    "frame_number": frame_num,
                    "filename": filename,
                    "s3_key": key,
                    "s3_url": url,
                    "is_annotated": is_annotated,
                    "annotated_image_url": url if is_annotated else None,
                    "bounding_box": bbox_block,
                })

        return timestamp_str, wagon_index, loco_frames, problem_frames

    # ------------------------------------------------------------------

    def publish_ocr_frames(
        self,
        timestamp_str: str,
        wagon_number_results: dict[int, dict],
        loco_number_results: dict[int, dict],
    ) -> None:
        """Upload the frame actually sent to Rekognition for each wagon/loco
        number read, so a bad OCR result can be visually audited.

        Mutates each result dict in place, adding ``ocr_frame_s3_key`` /
        ``ocr_frame_s3_url`` (``None`` when no local debug image exists — the
        detector found no plate at all, as opposed to reading invalid digits,
        which still saves a frame). Uploads unconditionally on validity —
        even a failed read's source frame is kept for review.
        """
        for seg_id, result in (wagon_number_results or {}).items():
            result["ocr_frame_s3_key"] = None
            result["ocr_frame_s3_url"] = None
            src = result.get("enhanced_image_path")
            if not src or not os.path.exists(src):
                continue
            filename = f"wagon_number_segment_{int(seg_id):03d}.jpg"
            key = self._key(timestamp_str, f"wagon_numbers/{filename}")
            self.s3.client.upload_file(src, self.artifact_bucket, key)
            result["ocr_frame_s3_key"] = key
            result["ocr_frame_s3_url"] = self._https(key)

        for loco_id, result in (loco_number_results or {}).items():
            result["ocr_frame_s3_key"] = None
            result["ocr_frame_s3_url"] = None
            src = result.get("enhanced_image_path")
            if not src or not os.path.exists(src):
                continue
            filename = f"loco_number_band_{int(loco_id):03d}.jpg"
            key = self._key(timestamp_str, f"loco_numbers/{filename}")
            self.s3.client.upload_file(src, self.artifact_bucket, key)
            result["ocr_frame_s3_key"] = key
            result["ocr_frame_s3_url"] = self._https(key)

    def _format_bounding_box(self, bboxes):
        """Return the per-flavour bounding-box block.

        Top notebooks emit the rich dict form
        ``{bounding_box_coordinates, confidence, class_name}``; side notebooks
        emit a plain ``[x1, y1, x2, y2]`` list (or ``None`` if no detection).
        """
        if not bboxes or not isinstance(bboxes, list) or not isinstance(bboxes[0], dict):
            return None
        primary = bboxes[0]
        if self.damage_flavour == "side":
            return primary.get("bbox")
        return {
            "bounding_box_coordinates": primary.get("bbox"),
            "confidence": primary.get("confidence"),
            "class_name": primary.get("class_name"),
        }

    # ------------------------------------------------------------------

    def upload_inspection_json(
        self,
        timestamp_str: str,
        inspection_payload: dict,
        local_workdir: str,
    ) -> str:
        """Write inspection_data.json to disk + S3. Returns the S3 key."""
        local_path = os.path.join(local_workdir, "inspection_data.json")
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(inspection_payload, f, indent=2, default=json_safe)
        key = self._key(timestamp_str, "inspection_data.json")
        self.s3.client.upload_file(local_path, self.artifact_bucket, key)
        self.logger.info(
            "inspection_data.json uploaded: s3://%s/%s", self.artifact_bucket, key,
        )
        return key
