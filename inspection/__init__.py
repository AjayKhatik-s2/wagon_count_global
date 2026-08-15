"""Downstream inspection on the finalized global wagons.

    GLOBAL TRAIN STRUCTURE (protected)
        -> FINAL GW_1..GW_N
        -> wagon_cache/<GW_n>/<camera>/      association by construction
        -> LOAD                              old_code load processor, UNCHANGED
        -> DOOR + DAMAGE                     legacy TIE DamageDetector, UNCHANGED
        -> evidence / problem frames         legacy ProblemFrameExtractor + publisher
        -> inspection_data.json per camera   legacy json_builder (dashboard contract)
        -> combined PDF + camera PDFs        legacy layouts, persisted state only
        -> processed / annotated videos

TWO LEGACY SOURCES, ONE ROLE EACH -- NOT TWO IMPLEMENTATIONS OF ONE ROLE
------------------------------------------------------------------------
`old_code/` and `inspection/legacy/` are both legacy, from different generations
of the system, and each owns a role the other does not implement:

    old_code/load/processor.py      the load verdict (loaded_ratio > 0.35,
                                    RIGHT_UP_TOP authoritative). The vendored
                                    engine has no load feature at all.
    inspection/legacy/damage.py     door state AND damage, per camera, with the
                                    band/best-frame/problem-frame structure the
                                    dashboard JSON is defined in terms of.
                                    old_code's door processor emits a single
                                    fused per-side verdict with no bands, no
                                    best frames and no per-camera
                                    door_close_detected / door_partial_detected,
                                    so it cannot express the legacy contract.

Because BOTH consume `door_state.pt` on the same side frames, running both would
be two verdicts for one question and two passes of one model. They are therefore
mutually exclusive, selected by `door_source` in the runner:

    legacy   (default)  inspection/legacy/damage.py       <- dashboard contract
    old_code (opt-in)   old_code/door/processor.py        <- kept for A/B on EC2

Exactly one runs per execution. Nothing proven was deleted; nothing is counted
twice.

Neither path may create, remove, renumber or re-time a wagon. That is enforced,
not assumed: the roster is hashed before and after and re-verified in a
`finally`, and every emitted wagon identity is checked against the roster by
`global_bridge.assert_wagon_count_map_is_global`.

OCR is out of scope and hard-disabled; its JSON fields are still emitted, empty,
so the dashboard schema is unchanged.
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
from .global_bridge import (
    CAMERA_PROFILES, CameraProfile, GlobalAssociationError,
    assert_wagon_count_map_is_global, build_segment_summary,
    build_segment_type_map, build_wagon_count_map, camera_profile,
    roster_segment_type,
)
from .legacy_inspection import (
    LegacyCameraResult, LegacyInspectionConfig, LegacyInspectionResult,
    reset_inspection_state, resolve_feature_model, run_legacy_inspection,
)
from .legacy_render import (
    build_all_legacy_outputs, build_annotated_videos, build_camera_pdfs,
    build_combined_pdf, combined_wagon_rows, load_camera_payloads,
)

__all__ = [
    # association bridge
    "WagonCacheConfig", "WagonWindow", "CachePlan", "plan_cache",
    "build_wagon_cache", "cache_stats", "clear_wagon_cache", "wagon_camera_dir",
    "CACHE_DIRNAME",
    # old_code features (load authority) + its reports
    "OldInspectionConfig", "OldInspectionResult", "run_old_inspection",
    "discover_feature_models", "fuse_unified_states",
    "build_all_reports", "build_camera_reports", "build_combined_report",
    # global roster -> legacy DataFrame bridge
    "CameraProfile", "CAMERA_PROFILES", "camera_profile", "roster_segment_type",
    "build_segment_summary", "build_wagon_count_map", "build_segment_type_map",
    "assert_wagon_count_map_is_global", "GlobalAssociationError",
    # legacy inspection + output contract
    "LegacyInspectionConfig", "LegacyInspectionResult", "LegacyCameraResult",
    "run_legacy_inspection", "reset_inspection_state", "resolve_feature_model",
    # renderers (no inference)
    "build_camera_pdfs", "build_combined_pdf", "build_annotated_videos",
    "build_all_legacy_outputs", "load_camera_payloads", "combined_wagon_rows",
]
