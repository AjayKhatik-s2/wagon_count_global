"""OCR feature processor (v4, train-state-native, ALL legacy intelligence
ported).

Pipeline:
    1. YOLO `wagon_id_counting.pt` detects wagon-number bbox regions on
       RIGHT_UP frames (master / OCR authority).
    2. Each crop is fed through the legacy `WagonNumberOCR`:
           padding 10 -> 3x cubic upscale -> NLMeans denoise (h=8) ->
           CLAHE (clipLimit=3.5, tile 8x8) -> unsharp masking ->
           easyocr (allowlist='0123456789') -> digit extraction ->
           wagon-type confusion-map correction (first 2 digits in 10-39)
           -> WagonNumberValidator (length=11, structure check).
    3. Surviving candidates per frame are added to the legacy
       `WagonNumberAggregator` which performs:
           exact-string grouping with digit-level voting at each position
           min 2 frames + min OCR conf 0.3
    4. Best aggregated number is picked by (observations, mean conf).

Output JSON shape:
    {
        "global_id":  "GW_7",
        "feature":    "ocr",
        "status":     "OK" | "NO_FRAMES" | "FAILED" | "NO_DATA",
        "wagon_identifier":  "32145678901",
        "wagon_identifier_confidence": 0.83,
        "candidates":  [...],
        "supporting_cameras": ["RIGHT_UP"],
        "frame_count": ...,
    }
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from core import constants as C
from core.global_state_loader import GlobalTrainState

from features._common import (
    load_yolo, run_detection, iter_wagon_frames, crop_bbox,
    write_per_wagon_json, empty_payload, FeatureTimer,
)

# Mature intelligence ported from legacy
from features.inference_lib.wagon_number_ocr import WagonNumberOCR, WagonNumber
from features.inference_lib.wagon_number_aggregator import (
    WagonNumberAggregator, AggregatorConfig,
)


FEATURE_NAME = "ocr"


# -----------------------------------------------------------------------------
# Per-process singleton (easyocr Reader is heavy; load once)
# -----------------------------------------------------------------------------

_OCR_SINGLETON: Optional[WagonNumberOCR] = None


def _get_ocr() -> Optional[WagonNumberOCR]:
    global _OCR_SINGLETON
    if _OCR_SINGLETON is not None:
        return _OCR_SINGLETON
    try:
        _OCR_SINGLETON = WagonNumberOCR(
            use_gpu=True,
            min_confidence=0.30,        # legacy default for cross-frame aggregation
            resize_factor=3.0,
        )
        if getattr(_OCR_SINGLETON, "reader", None) is None:
            _OCR_SINGLETON = None
    except Exception as e:
        print(f"[FEAT/ocr] WagonNumberOCR init failed: {e}")
        _OCR_SINGLETON = None
    return _OCR_SINGLETON


# -----------------------------------------------------------------------------
# Per-wagon driver
# -----------------------------------------------------------------------------

def _process_one_wagon(
    yolo_model,
    ocr: WagonNumberOCR,
    cache_root: str,
    gw_id: str,
    det_confidence: float,
) -> Dict[str, Any]:
    """Iterate cached RIGHT_UP frames, run YOLO + OCR, aggregate."""
    aggregator = WagonNumberAggregator(AggregatorConfig(
        min_frame_count=2,
        min_confidence=0.3,
        require_validation=True,
    ))

    used = 0
    raw_candidates: List[Dict[str, Any]] = []

    for fi, frame in iter_wagon_frames(cache_root, gw_id, C.CAMERA_RIGHT_UP):
        used += 1

        # Stage A: YOLO detection -- locate wagon-number bbox regions
        try:
            results = yolo_model(frame, verbose=False, half=True)[0]
        except Exception:
            continue
        if results.boxes is None or len(results.boxes) == 0:
            continue

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        # Stage B: per-detection OCR pipeline (preprocess + easyocr +
        # validate + reconstruct)
        for bbox, yolo_conf in zip(boxes, confs):
            if float(yolo_conf) < det_confidence:
                continue
            crop = crop_bbox(frame, [float(b) for b in bbox], pad=10)
            if crop is None or crop.size == 0:
                continue
            try:
                wagon_num = ocr.reconstruct_wagon_number(
                    crop, float(yolo_conf), debug=False,
                )
            except Exception:
                continue
            if wagon_num is None:
                continue
            # `reconstruct_wagon_number` returns a WagonNumber dataclass when
            # validation succeeds.  The aggregator handles dedup + voting.
            aggregator.add_wagon_number(wagon_num, frame_idx=fi)

            # Keep a per-frame breadcrumb for the wagon JSON (debug)
            full = getattr(wagon_num, "full_number", None)
            if full:
                raw_candidates.append({
                    "frame_idx":       int(fi),
                    "full_number":     str(full),
                    "ocr_confidence":  float(getattr(wagon_num, "ocr_confidence", 0.0)),
                    "yolo_confidence": float(getattr(wagon_num, "yolo_confidence", 0.0)),
                })

    # Stage C: pick the dominant aggregated wagon number
    aggregated = aggregator.get_aggregated_numbers()
    return {
        "frame_count": used,
        "aggregated":  aggregated,
        "raw":         raw_candidates,
    }


# -----------------------------------------------------------------------------
# Public entry
# -----------------------------------------------------------------------------

def run(
    *,
    state: GlobalTrainState,
    cache_root: str,
    feature_models_dir: str,
    output_dir: str,
    det_confidence: float = C.CONF_OCR_BOX,
    wagon_number_length: int = C.WAGON_NUMBER_LENGTH,
    every_nth: int = 1,
    max_frames: int = 0,
    verbose: bool = True,
) -> Dict[str, str]:
    """Run OCR on every wagon, using the mature legacy pipeline."""
    del every_nth, max_frames, wagon_number_length  # legacy code uses its own thresholds

    model_path = os.path.join(feature_models_dir, C.MODEL_WAGON_ID_COUNTING)
    yolo_model = load_yolo(model_path)
    ocr = _get_ocr()

    feature_out = os.path.join(output_dir, FEATURE_NAME)
    os.makedirs(feature_out, exist_ok=True)
    timer = FeatureTimer("ocr")
    summary: Dict[str, str] = {}

    if yolo_model is None and verbose:
        print(f"[FEAT/ocr] WARNING: {model_path} missing -- NO_DATA for all wagons.")
    if ocr is None and verbose:
        print(f"[FEAT/ocr] WARNING: easyocr unavailable -- NO_DATA for all wagons.")

    if verbose:
        print(f"[FEAT/ocr] running on {len(state.wagons)} wagons "
              f"(legacy WagonNumberOCR + WagonNumberAggregator, RIGHT_UP only)")

    for gw in state.wagons:
        gw_id = gw.global_id
        t0 = time.time()
        try:
            if yolo_model is None or ocr is None:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.NO_DATA,
                    wagon_identifier=C.NO_DATA,
                    wagon_identifier_confidence=0.0,
                    candidates=[], supporting_cameras=[],
                    error="detector or OCR engine unavailable",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.NO_DATA
                continue

            # ENGINE / BRAKE_VAN wagons rarely carry the standard 11-digit
            # wagon number; running OCR on them produces noise.  Skip but
            # still record the wagon entry.
            if gw.classification in (C.CLASS_ENGINE, C.CLASS_BRAKE_VAN):
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_OK,
                    wagon_identifier=C.NO_DATA,
                    wagon_identifier_confidence=0.0,
                    candidates=[],
                    supporting_cameras=[C.CAMERA_RIGHT_UP],
                    skipped_reason=f"classification={gw.classification}",
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.STATUS_OK
                continue

            outcome = _process_one_wagon(
                yolo_model, ocr, cache_root, gw_id, det_confidence,
            )
            used = outcome["frame_count"]
            aggregated = outcome["aggregated"]

            if used == 0:
                payload = empty_payload(
                    gw_id, FEATURE_NAME, C.STATUS_NO_FRAMES,
                    wagon_identifier=C.NO_DATA,
                    wagon_identifier_confidence=0.0,
                    candidates=[],
                    supporting_cameras=[],
                )
                write_per_wagon_json(feature_out, gw_id, payload)
                summary[gw_id] = C.STATUS_NO_FRAMES
                continue

            # Build serialized candidate list from the aggregator's output
            candidates_out: List[Dict[str, Any]] = []
            for agg in aggregated:
                candidates_out.append({
                    "full_number":     str(getattr(agg, "wagon_number", "")),
                    "observations":    int(getattr(agg, "frame_count", 0)),
                    "mean_conf":       float(getattr(agg, "avg_confidence", 0.0)),
                    "yolo_conf":       float(getattr(agg, "avg_yolo_confidence",
                                              getattr(agg, "yolo_confidence", 0.0))),
                    "is_full_length":  len(str(getattr(agg, "wagon_number", "")))
                                       == C.WAGON_NUMBER_LENGTH,
                })

            # Aggregator already enforces min_frame_count + min_confidence.
            # The "best" candidate is the one with the highest combined
            # (observations, mean_conf) score.
            candidates_out.sort(
                key=lambda c: (
                    -int(c["is_full_length"]),
                    -c["observations"],
                    -c["mean_conf"],
                    c["full_number"],
                )
            )

            if candidates_out and candidates_out[0]["is_full_length"]:
                best = candidates_out[0]
                ident = best["full_number"]
                conf  = best["mean_conf"]
            else:
                ident = C.NO_DATA
                conf  = 0.0

            payload: Dict[str, Any] = {
                "global_id":   gw_id,
                "feature":     FEATURE_NAME,
                "status":      C.STATUS_OK,
                "wagon_identifier":            ident,
                "wagon_identifier_confidence": round(float(conf), 4),
                "candidates":  candidates_out[:8],
                "raw_candidates_first_8":      outcome["raw"][:8],
                "supporting_cameras": [C.CAMERA_RIGHT_UP],
                "frame_count": used,
            }
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = C.STATUS_OK
            if verbose:
                print(f"  [ocr/{gw_id}] {ident} (conf={conf:.2f}, "
                      f"candidates={len(candidates_out)}, frames={used})")
        except Exception as e:
            payload = empty_payload(
                gw_id, FEATURE_NAME, C.STATUS_FAILED,
                wagon_identifier=C.NO_DATA,
                error=f"{type(e).__name__}: {e}",
                traceback=traceback.format_exc(limit=2),
            )
            write_per_wagon_json(feature_out, gw_id, payload)
            summary[gw_id] = C.STATUS_FAILED
            if verbose:
                print(f"  [ocr/{gw_id}] FAILED: {e}")
        finally:
            timer.stamp(gw_id, t0)

    if verbose:
        n_ok = sum(1 for v in summary.values() if v == C.STATUS_OK)
        print(f"[FEAT/ocr] done in {timer.total():.1f}s  ok={n_ok}/{len(summary)}")
    return summary
