"""Downstream inspection: old_code intelligence on the finalized global wagons.

    GLOBAL TRAIN STRUCTURE (protected)
        -> FINAL GW_1..GW_N
        -> wagon_cache/<GW_n>/<camera>/      association by construction
        -> old_code door / load / damage     imported UNCHANGED
        -> UnifiedWagonState per GW_n
        -> additive state.inspection
        -> old combined report + camera reports
        -> processed videos

THERE IS EXACTLY ONE IMPLEMENTATION OF THE FEATURE INTELLIGENCE, and it lives in
`old_code/`. This package contains only the bridge: the per-wagon frame cache that
performs GW association, the orchestrator that calls the old processors in the
right order, and the report entry points. No detection, tracking, filtering,
voting or state-machine logic is reimplemented here -- an earlier simplified
version was removed precisely so it could not drift from the source of truth.

OCR is out of scope.
"""

from .wagon_cache import (
    CACHE_DIRNAME, CachePlan, WagonCacheConfig, WagonWindow, build_wagon_cache,
    cache_stats, clear_wagon_cache, plan_cache, wagon_camera_dir,
)
from .old_features import (
    OldInspectionConfig, OldInspectionResult, discover_feature_models,
    fuse_unified_states, run_old_inspection,
)
from .old_report import (
    build_all_reports, build_camera_reports, build_combined_report,
)

__all__ = [
    "WagonCacheConfig", "WagonWindow", "CachePlan", "plan_cache",
    "build_wagon_cache", "cache_stats", "clear_wagon_cache", "wagon_camera_dir",
    "CACHE_DIRNAME",
    "OldInspectionConfig", "OldInspectionResult", "run_old_inspection",
    "discover_feature_models", "fuse_unified_states",
    "build_all_reports", "build_camera_reports", "build_combined_report",
]
