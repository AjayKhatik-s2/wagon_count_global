"""Vendored legacy Train-Inspection-Engine inspection/output modules.

PROVENANCE
----------
Every file in this package is a copy of
``rithish__code_1/CCTV-TrainVideo-ML-V2-wagon-Rithish/Train-Inspection-Engine/
src/train_inspection_engine/...`` and is the BEHAVIOURAL SOURCE OF TRUTH for
door state, damage, evidence selection, S3 artifact layout, the dashboard
``inspection_data.json`` contract, the per-camera PDF and the annotated video.

    bands.py           <- inspection/bands.py
    frame_positions.py <- inspection/frame_positions.py
    damage.py          <- inspection/damage.py
    annotated_video.py <- inspection/annotated_video.py
    artifacts.py       <- reporting/artifacts.py
    json_builder.py    <- reporting/json_builder.py
    pdf_builder.py     <- reporting/pdf_builder.py
    model_store.py     <- core/model_store.py
    s3.py              <- core/s3.py
    video_io.py        <- core/video_io.py
    url_utils.py       <- utils/url_utils.py
    serialization.py   <- utils/serialization.py

WHAT WAS CHANGED, AND NOTHING ELSE
----------------------------------
1. Import paths only: ``from ..core.s3`` -> ``from .s3`` etc., because the
   files now live in one flat package instead of the engine's tree. No symbol
   was renamed, added or removed.
2. ``damage.py`` gained a two-line ImportError guard around ``tqdm`` (a progress
   bar this project does not depend on). It changes what is printed, never what
   is computed.

No threshold, filter, vote, band rule, class map, filename pattern, JSON key or
layout decision was touched. That is the point: the dashboard contract and the
proven detection behaviour must not drift, so they are not re-implemented here.

WHAT WAS DELIBERATELY *NOT* VENDORED
------------------------------------
``inspection/segments.py`` (WagonSegmenter / classify_segment_type),
``train_detection/*``, ``pipelines/base_pipeline.py`` and ``combiner/*`` are the
legacy engine's OWN camera-wise wagon segmentation and counting. They are a
second counting authority and are excluded on purpose -- the current global
wagon-counting pipeline is the only authority for wagon identity and count.
``inspection/ocr/*`` and ``core/rekognition.py`` are excluded because OCR is
out of scope; the JSON fields they used to fill are emitted as ``{}``.

The bridge that lets these modules run against the finalized global roster is
``inspection/global_bridge.py``; the orchestrator is
``inspection/legacy_inspection.py``.
"""

from __future__ import annotations

__all__ = [
    "bands", "frame_positions", "damage", "annotated_video", "artifacts",
    "json_builder", "pdf_builder", "model_store", "s3", "video_io",
    "url_utils", "serialization",
]
