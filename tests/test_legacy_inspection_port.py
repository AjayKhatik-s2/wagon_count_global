"""The legacy inspection/output port: legacy behaviour, GLOBAL wagon identity.

What these tests defend, in order of importance:

  1. THE COUNT IS THE GLOBAL COUNT, EVERYWHERE. If the roster has 57 wagons then
     all four inspection JSONs and the combined report contain exactly 57 wagon
     entries -- no matter how many detections fired, how many wagons a camera
     saw, or whether a camera failed entirely.
  2. THE ROSTER IS IMMUTABLE. Inspection cannot create, delete, renumber or
     re-time a wagon, and cannot touch camera offsets. Checked by hash, including
     when a feature raises.
  3. THE JSON IS THE LEGACY JSON. Same keys, same nesting, same types, same
     semantics. Only explicitly documented additive fields are new.
  4. OCR IS OFF, but its fields are still emitted as empty dicts.
  5. STATES STAY DISTINCT. NOT_VISIBLE / UNRESOLVED / NO_DETECTION are never
     collapsed into a clean finding.
  6. THE RENDERERS DO NOT INFER. The PDF/video layer consumes persisted state.

Model-dependent behaviour is exercised through fixture DataFrames, so the whole
suite runs with no weights and no GPU -- which is also what lets it assert
things a real run cannot (a camera that saw 2 of 57 wagons, a duplicate problem
frame, an inspection that tries to mutate the roster).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from core import constants as C
from core.global_state_loader import RosterMutatedError, roster_hash
from global_train_state import GlobalTrainState, GlobalWagon, SegmentClass
from inspection import global_bridge as gb
from inspection import legacy_inspection as li
from inspection import legacy_render as lr
from inspection import wagon_cache as wc
from inspection.legacy import damage as legacy_damage
from inspection.legacy.artifacts import ArtifactPublisher, display_segment_type
from inspection.legacy.json_builder import build_inspection_json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 15.0
EXISTING_PATH = os.path.abspath(__file__)
TS = datetime(2026, 8, 15, 12, 30, 0)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

class FakeTracks:
    def __init__(self, fps=FPS, total=100000, path=EXISTING_PATH):
        self.fps = fps
        self.total_frames = total
        self.width = 960
        self.height = 540
        self.video_path = path


def roster(n=57, span=4.0, classifications=None):
    """A finalized roster: GW_1..GW_n, contiguous in MASTER seconds."""
    state = GlobalTrainState(total_wagons=n, master_camera=C.CAMERA_RIGHT_UP,
                             master_fps=FPS, master_total_frames=100000)
    classifications = classifications or {}
    for i in range(1, n + 1):
        state.wagons.append(GlobalWagon(
            global_id=f"GW_{i}", wagon_index=i,
            start_frame_master=int((i - 1) * span * FPS),
            end_frame_master=int(i * span * FPS),
            start_time=(i - 1) * span, end_time=i * span,
            classification=classifications.get(i, SegmentClass.WAGON)))
    state.camera_offsets = {
        cam: {"status": "REFERENCE" if cam == C.CAMERA_RIGHT_UP else "RESOLVED",
              "delta": 0.0}
        for cam in C.ALL_CAMERAS}
    return state


def windows_for(state, camera_id, only_indices=None):
    """WagonWindows as plan_cache would produce them at offset 0."""
    out = []
    for wagon in state.wagons:
        if only_indices is not None and wagon.wagon_index not in only_indices:
            continue
        out.append(wc.WagonWindow(
            global_id=wagon.global_id, camera_id=camera_id,
            start_frame=int(wagon.start_time * FPS),
            end_frame=int(wagon.end_time * FPS)))
    return out


class FakePlan:
    def __init__(self, root, windows, camera_status):
        self.root = root
        self.windows = list(windows)
        self.camera_status = dict(camera_status)


def seed_cache(cache_root, state, cameras, only_indices=None, frames=3):
    """Write placeholder JPEGs so a wagon counts as observed by a camera."""
    import numpy as np
    import cv2
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    for camera_id in cameras:
        for wagon in state.wagons:
            if only_indices is not None and wagon.wagon_index not in only_indices:
                continue
            d = wc.wagon_camera_dir(cache_root, wagon.global_id, camera_id)
            os.makedirs(d, exist_ok=True)
            base = int(wagon.start_time * FPS)
            for k in range(frames):
                cv2.imwrite(os.path.join(d, f"frame_{base + k:06d}.jpg"), img)


def side_damage_row(seg_id, *, door_status="closed", damage=False,
                    close_detected=True, partial=False):
    return {
        "wagon_id": seg_id, "damage_detected": damage,
        "door_status": door_status, "door_close_detected": close_detected,
        "door_partial_detected": partial,
        "damage_best_frames": [], "open_door_best_frames": [],
        "closed_door_best_frames": [], "partially_closed_best_frames": [],
        "damage_band_info": [], "open_door_band_info": [],
        "closed_door_band_info": [], "partially_closed_band_info": [],
    }


def top_damage_row(seg_id, *, floor=False, inner=False, probable=False):
    return {
        "wagon_id": seg_id,
        "floor_dmg_detected": floor, "inner_wall_dmg_detected": inner,
        "floor_dmg_probable_detected": probable,
        "damage_detected": floor or inner, "probable_damage_detected": probable,
        "floor_dmg_best_frames": [], "inner_wall_dmg_best_frames": [],
        "floor_dmg_probable_best_frames": [],
        "floor_dmg_band_info": [], "inner_wall_dmg_band_info": [],
        "floor_dmg_probable_band_info": [],
    }


def build_json_for(state, camera_id, cache_root, *, damage_rows=None,
                   problem_entries=None, only_indices=None,
                   load_status_by_wagon=None):
    """Drive the legacy JSON builder exactly as the orchestrator does."""
    profile = gb.camera_profile(camera_id)
    windows = windows_for(state, camera_id, only_indices)
    full_df = gb.build_segment_summary(
        state, camera_id, windows, cache_root,
        load_status_by_wagon=load_status_by_wagon,
        require_frames=False, include_unwindowed=True)
    scan_df = gb.build_segment_summary(
        state, camera_id, windows, cache_root,
        load_status_by_wagon=load_status_by_wagon,
        require_frames=True, include_unwindowed=False)
    count_map = gb.build_wagon_count_map(full_df)
    gb.assert_wagon_count_map_is_global(state, count_map, camera_id)
    type_map = gb.build_segment_type_map(full_df, profile.flavour)
    damage_df = pd.DataFrame(damage_rows) if damage_rows else \
        li._empty_damage_df(profile.flavour)
    payload = build_inspection_json(
        camera_folder=profile.folder, raw_video_name="clip",
        upload_timestamp=TS, direction=profile.loaded_direction,
        flavour=profile.flavour, segment_summary_df=full_df,
        damage_results_df=damage_df, loco_summary_df=pd.DataFrame(),
        problem_frames_df=pd.DataFrame(),
        wagon_frames_index={}, loco_frame_entries=[],
        problem_frame_entries=list(problem_entries or []),
        wagon_count_map=count_map, segment_type_map=type_map,
        wagon_number_results=None, loco_numbers=None,
        damage_model_active=True)
    windowed = {w.wagon_index for w in state.wagons
                if only_indices is None or w.wagon_index in only_indices}
    li._annotate_globally(payload, state, scan_df, damage_df, "RESOLVED",
                          profile, windowed)
    return payload


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="legacy_insp_")
        self.cache = os.path.join(self.tmp, "wagon_cache")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. COUNT CONSISTENCY -- the acceptance criterion
# ---------------------------------------------------------------------------

class TestCountConsistency(TempCase):
    """57 global wagons => 57 entries in every output, always."""

    N = 57

    def test_all_four_cameras_report_the_global_count(self):
        state = roster(self.N)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        for camera_id in C.ALL_CAMERAS:
            payload = build_json_for(state, camera_id, self.cache)
            data = payload["inspection_data"]
            self.assertEqual(len(data["wagon_segments"]), self.N, camera_id)
            self.assertEqual(data["total_wagons"], self.N, camera_id)
            self.assertEqual(data["global_wagon_count"], self.N, camera_id)

    def test_partial_camera_still_reports_the_global_count(self):
        """A camera that observed 2 of 57 wagons must still report 57.

        This is the exact failure the port exists to prevent: the legacy
        segmenter would have produced a 2-wagon JSON here and the dashboard
        would have seen two different trains.
        """
        state = roster(self.N)
        seed_cache(self.cache, state, [C.CAMERA_LEFT_UP], only_indices={4, 5})
        payload = build_json_for(state, C.CAMERA_LEFT_UP, self.cache,
                                 only_indices={4, 5})
        data = payload["inspection_data"]
        self.assertEqual(data["total_wagons"], self.N)
        self.assertEqual(len(data["wagon_segments"]), self.N)
        statuses = {s["wagon_count"]: s["inspection_status"]
                    for s in data["wagon_segments"]}
        self.assertEqual(statuses[4], li.STATUS_NO_DETECTION)
        self.assertEqual(statuses[1], li.STATUS_NOT_VISIBLE)

    def test_count_is_independent_of_detection_volume(self):
        state = roster(self.N)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        none_rows = build_json_for(state, C.CAMERA_RIGHT_UP, self.cache)
        many_rows = build_json_for(
            state, C.CAMERA_RIGHT_UP, self.cache,
            damage_rows=[side_damage_row(i, door_status="open", damage=True)
                         for i in range(1, self.N + 1)])
        self.assertEqual(none_rows["inspection_data"]["total_wagons"],
                         many_rows["inspection_data"]["total_wagons"])
        self.assertEqual(many_rows["inspection_data"]["total_wagons"], self.N)
        self.assertEqual(many_rows["inspection_data"]["damaged_wagons"], self.N)

    def test_combined_report_rows_equal_global_count(self):
        state = roster(self.N)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        payloads = {cam: build_json_for(state, cam, self.cache)
                    for cam in C.ALL_CAMERAS}
        rows = lr.combined_wagon_rows(state, payloads)
        self.assertEqual(len(rows), self.N)
        self.assertEqual([r["global_wagon_id"] for r in rows],
                         [f"GW_{i}" for i in range(1, self.N + 1)])

    def test_pdf_rows_equal_json_rows(self):
        state = roster(12)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        payloads = {cam: build_json_for(state, cam, self.cache)
                    for cam in C.ALL_CAMERAS}
        rows = lr.combined_wagon_rows(state, payloads)
        for cam, payload in payloads.items():
            self.assertEqual(
                len(rows), len(payload["inspection_data"]["wagon_segments"]),
                f"{cam}: PDF row set and JSON wagon set must be the same size")

    def test_failed_camera_does_not_shrink_the_report(self):
        state = roster(20)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        payloads = {C.CAMERA_RIGHT_UP: build_json_for(
            state, C.CAMERA_RIGHT_UP, self.cache)}
        rows = lr.combined_wagon_rows(state, payloads)
        self.assertEqual(len(rows), 20)
        self.assertEqual(rows[0]["per_camera"][C.CAMERA_RIGHT_UP]["status"], "OK")


# ---------------------------------------------------------------------------
# 2. THE ROSTER IS IMMUTABLE
# ---------------------------------------------------------------------------

class TestRosterProtection(TempCase):

    def test_inspection_does_not_change_the_roster(self):
        state = roster(9)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        before = roster_hash(state)
        plan = FakePlan(self.cache, windows_for(state, C.CAMERA_RIGHT_UP),
                        {c: "RESOLVED" for c in C.ALL_CAMERAS})
        result = li.run_legacy_inspection(
            state=state, tracks_by_camera={c: FakeTracks() for c in C.ALL_CAMERAS},
            plan=plan, models_dir=os.path.join(ROOT, "models"),
            output_root=os.path.join(self.tmp, "out"),
            cfg=li.LegacyInspectionConfig(
                side_model="__absent__.pt", top_model="__absent__.pt",
                build_pdf=False),
            verbose=False)
        self.assertEqual(roster_hash(state), before)
        self.assertTrue(result.roster_unchanged)
        self.assertEqual(result.global_wagon_count, 9)

    def test_roster_check_runs_even_when_a_feature_raises(self):
        """The finally block is the guarantee; a raise must not bypass it."""
        state = roster(4)
        before = roster_hash(state)
        plan = FakePlan(self.cache, [], {c: "RESOLVED" for c in C.ALL_CAMERAS})

        original = li._run_camera
        li._run_camera = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            result = li.run_legacy_inspection(
                state=state, tracks_by_camera={}, plan=plan,
                models_dir=os.path.join(ROOT, "models"),
                output_root=os.path.join(self.tmp, "out"),
                cfg=li.LegacyInspectionConfig(build_pdf=False), verbose=False)
        finally:
            li._run_camera = original
        self.assertEqual(roster_hash(state), before)
        self.assertTrue(result.roster_unchanged)
        self.assertTrue(any("boom" in w for w in result.warnings))

    def test_mutated_roster_is_detected(self):
        state = roster(5)
        before = roster_hash(state)
        state.wagons.append(GlobalWagon(
            global_id="GW_6", wagon_index=6, start_frame_master=0,
            end_frame_master=1, start_time=0.0, end_time=1.0))
        from core.global_state_loader import assert_roster_unchanged
        with self.assertRaises(RosterMutatedError):
            assert_roster_unchanged(state, before)

    def test_camera_offsets_are_untouched(self):
        state = roster(6)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        before = json.dumps(state.camera_offsets, sort_keys=True)
        plan = FakePlan(self.cache, windows_for(state, C.CAMERA_LEFT_UP),
                        {c: "RESOLVED" for c in C.ALL_CAMERAS})
        li.run_legacy_inspection(
            state=state, tracks_by_camera={c: FakeTracks() for c in C.ALL_CAMERAS},
            plan=plan, models_dir=os.path.join(ROOT, "models"),
            output_root=os.path.join(self.tmp, "out"),
            cfg=li.LegacyInspectionConfig(side_model="__absent__.pt",
                                          top_model="__absent__.pt"),
            verbose=False)
        self.assertEqual(json.dumps(state.camera_offsets, sort_keys=True), before)

    def test_master_equals_global_is_preserved(self):
        state = roster(31)
        self.assertEqual(state.total_wagons, len(state.wagons))
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        payload = build_json_for(state, C.CAMERA_RIGHT_UP, self.cache)
        self.assertEqual(payload["inspection_data"]["total_wagons"],
                         state.total_wagons)


# ---------------------------------------------------------------------------
# 3. NO FEATURE MAY CREATE / RENUMBER A WAGON
# ---------------------------------------------------------------------------

class TestNoWagonCreation(TempCase):

    def test_wagon_count_map_is_the_global_index(self):
        state = roster(10)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        df = gb.build_segment_summary(
            state, C.CAMERA_RIGHT_UP, windows_for(state, C.CAMERA_RIGHT_UP),
            self.cache, require_frames=False, include_unwindowed=True)
        count_map = gb.build_wagon_count_map(df)
        self.assertEqual(count_map, {i: i for i in range(1, 11)})

    def test_engine_and_brakevan_get_no_wagon_count(self):
        state = roster(6, classifications={1: SegmentClass.ENGINE,
                                           6: SegmentClass.BRAKE_VAN})
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        df = gb.build_segment_summary(
            state, C.CAMERA_RIGHT_UP, windows_for(state, C.CAMERA_RIGHT_UP),
            self.cache, require_frames=False, include_unwindowed=True)
        count_map = gb.build_wagon_count_map(df)
        self.assertIsNone(count_map[1])
        self.assertIsNone(count_map[6])
        self.assertEqual(count_map[3], 3)

    def test_unknown_wagon_count_is_refused(self):
        state = roster(5)
        with self.assertRaises(gb.GlobalAssociationError):
            gb.assert_wagon_count_map_is_global(state, {1: 1, 2: 99})

    def test_renumbering_is_refused(self):
        state = roster(5)
        with self.assertRaises(gb.GlobalAssociationError):
            gb.assert_wagon_count_map_is_global(state, {3: 2})

    def test_duplicate_wagon_is_refused(self):
        state = roster(5)
        with self.assertRaises(gb.GlobalAssociationError):
            gb.assert_wagon_count_map_is_global(state, {2: 2, 3: 2})

    def test_ids_are_monotonic_and_contiguous(self):
        state = roster(40)
        seed_cache(self.cache, state, [C.CAMERA_LEFT_UP_TOP])
        payload = build_json_for(state, C.CAMERA_LEFT_UP_TOP, self.cache)
        counts = [s["wagon_count"]
                  for s in payload["inspection_data"]["wagon_segments"]]
        self.assertEqual(counts, list(range(1, 41)))

    def test_a_detection_cannot_add_a_wagon(self):
        """Damage rows for wagons 1..N leave the wagon set exactly N."""
        state = roster(8)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        payload = build_json_for(
            state, C.CAMERA_RIGHT_UP, self.cache,
            damage_rows=[side_damage_row(i, damage=True) for i in range(1, 9)])
        self.assertEqual(len(payload["inspection_data"]["wagon_segments"]), 8)


# ---------------------------------------------------------------------------
# 4. LEGACY JSON CONTRACT (golden schema)
# ---------------------------------------------------------------------------

SIDE_REQUIRED = {
    "raw_video_name": str, "identified_by": str, "upload_timestamp": str,
    "direction": str, "rake_status": str, "total_wagons": int,
    "doors_open": int, "doors_partially_closed": int, "doors_closed": int,
    "damaged_wagons": int, "num_engines": int, "total_loco_frames": int,
    "total_problem_frames": int, "problem_frames_by_type": dict,
    "wagon_number_results": dict, "loco_number_results": dict,
    "segment_type_map": dict, "wagon_segments": list, "loco_frames": list,
    "problem_frames": list, "damage_model_active": bool,
}
TOP_REQUIRED = {
    "raw_video_name": str, "identified_by": str, "upload_timestamp": str,
    "direction": str, "rake_status": str, "total_wagons": int,
    "wagons_loaded": int, "wagons_empty": int, "damaged_wagons": int,
    "probable_damage_wagons": int, "floor_dmg_wagons": int,
    "inner_wall_dmg_wagons": int, "floor_dmg_probable_wagons": int,
    "num_engines": int, "num_brakevans": int, "total_loco_frames": int,
    "total_problem_frames": int, "problem_frames_by_type": dict,
    "wagon_number_results": dict, "loco_number_results": dict,
    "segment_type_map": dict, "wagon_segments": list, "loco_frames": list,
    "problem_frames": list, "damage_model_active": bool,
}
ADDITIVE_ALLOWED = {"global_wagon_id", "inspection_status", "counting_source",
                    "global_wagon_count", "ocr_enabled", "camera_role"}
"""The ONLY new keys. Everything else must be a legacy key, so a dashboard
written against the old schema cannot tell the backends apart."""


class TestLegacyJsonContract(TempCase):

    def test_side_schema(self):
        state = roster(7)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        data = build_json_for(state, C.CAMERA_RIGHT_UP,
                              self.cache)["inspection_data"]
        for key, typ in SIDE_REQUIRED.items():
            self.assertIn(key, data, f"legacy side field {key} was dropped")
            self.assertIsInstance(data[key], typ, key)

    def test_top_schema(self):
        state = roster(7)
        seed_cache(self.cache, state, [C.CAMERA_LEFT_UP_TOP])
        data = build_json_for(state, C.CAMERA_LEFT_UP_TOP,
                              self.cache)["inspection_data"]
        for key, typ in TOP_REQUIRED.items():
            self.assertIn(key, data, f"legacy top field {key} was dropped")
            self.assertIsInstance(data[key], typ, key)

    def test_envelope(self):
        state = roster(3)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        payload = build_json_for(state, C.CAMERA_RIGHT_UP, self.cache)
        self.assertEqual(set(payload), {"camera_id", "version", "inspection_data"})
        self.assertEqual(payload["version"], "v4")
        self.assertEqual(payload["camera_id"], "CCTV_HZBN_DHN_2_RIGHT_UP")

    def test_only_documented_additive_fields_are_new(self):
        state = roster(4)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        data = build_json_for(state, C.CAMERA_RIGHT_UP,
                              self.cache)["inspection_data"]
        legacy_top_level = set(SIDE_REQUIRED) | {
            "upload_timestamp_readable", "pdf_report_url", "trimmed_video_url",
            "detected_video_url", "raw_video_urls", "num_brakevans"}
        extras = set(data) - legacy_top_level
        self.assertTrue(extras <= ADDITIVE_ALLOWED,
                        f"undocumented new top-level fields: "
                        f"{sorted(extras - ADDITIVE_ALLOWED)}")

    def test_side_wagon_segment_fields(self):
        state = roster(3)
        seed_cache(self.cache, state, [C.CAMERA_LEFT_UP])
        seg = build_json_for(state, C.CAMERA_LEFT_UP,
                             self.cache)["inspection_data"]["wagon_segments"][0]
        for key in ("segment_id", "segment_type", "wagon_count", "door_status",
                    "door_close_detected", "door_partial_detected",
                    "damage_detected", "wagon_frames", "is_valid_wagon_id"):
            self.assertIn(key, seg)
        self.assertEqual(seg["segment_type"], "wagon")
        self.assertIsInstance(seg["door_close_detected"], bool)
        self.assertEqual(seg["global_wagon_id"], "GW_1")

    def test_top_wagon_segment_fields(self):
        state = roster(3)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP_TOP])
        seg = build_json_for(state, C.CAMERA_RIGHT_UP_TOP,
                             self.cache)["inspection_data"]["wagon_segments"][0]
        for key in ("segment_id", "segment_type", "wagon_count", "load_status",
                    "load_condition", "damage_detected",
                    "probable_damage_detected", "floor_dmg_detected",
                    "inner_wall_dmg_detected", "floor_dmg_probable_detected",
                    "wagon_frames", "is_valid_wagon_id"):
            self.assertIn(key, seg)

    def test_segment_type_map_shape(self):
        state = roster(4, classifications={1: SegmentClass.ENGINE})
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP_TOP, C.CAMERA_RIGHT_UP])
        top = build_json_for(state, C.CAMERA_RIGHT_UP_TOP,
                             self.cache)["inspection_data"]["segment_type_map"]
        side = build_json_for(state, C.CAMERA_RIGHT_UP,
                              self.cache)["inspection_data"]["segment_type_map"]
        self.assertEqual(set(top["1"]), {"type", "number", "wagon_count"})
        self.assertEqual(set(side["1"]), {"type", "number"})
        self.assertEqual(top["1"]["type"], "engine")
        self.assertIsNone(top["1"]["wagon_count"])
        self.assertEqual(top["2"]["wagon_count"], 2)

    def test_json_is_serialisable(self):
        state = roster(5)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        payload = build_json_for(state, C.CAMERA_RIGHT_UP, self.cache)
        path = os.path.join(self.tmp, "inspection_data.json")
        li._write_json(path, payload)
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["inspection_data"]["total_wagons"], 5)


# ---------------------------------------------------------------------------
# 5. OCR IS OFF -- but its fields survive
# ---------------------------------------------------------------------------

class TestNoOcr(TempCase):

    def test_ocr_result_blocks_are_empty_dicts(self):
        state = roster(5)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        for camera_id in C.ALL_CAMERAS:
            data = build_json_for(state, camera_id,
                                  self.cache)["inspection_data"]
            self.assertEqual(data["wagon_number_results"], {}, camera_id)
            self.assertEqual(data["loco_number_results"], {}, camera_id)
            self.assertIs(data["ocr_enabled"], False)

    def test_is_valid_wagon_id_is_preserved_and_false(self):
        state = roster(3)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        for seg in build_json_for(state, C.CAMERA_RIGHT_UP,
                                  self.cache)["inspection_data"]["wagon_segments"]:
            self.assertIn("is_valid_wagon_id", seg)
            self.assertFalse(seg["is_valid_wagon_id"])
            self.assertNotIn("wagon_number", seg)

    def test_enabling_ocr_is_refused(self):
        state = roster(2)
        plan = FakePlan(self.cache, [], {c: "RESOLVED" for c in C.ALL_CAMERAS})
        with self.assertRaises(ValueError):
            li.run_legacy_inspection(
                state=state, tracks_by_camera={}, plan=plan,
                models_dir=os.path.join(ROOT, "models"),
                output_root=self.tmp,
                cfg=li.LegacyInspectionConfig(enable_ocr=True), verbose=False)

    def test_no_ocr_module_is_imported(self):
        for module in (li, lr, gb):
            src = open(module.__file__, encoding="utf-8").read()
            for banned in ("rekognition", "easyocr", "wagon_number_ocr",
                           "detect_text"):
                self.assertNotIn(banned, src.lower(),
                                 f"{module.__name__} references {banned}")

    def test_ocr_modules_were_not_vendored(self):
        legacy_dir = os.path.join(ROOT, "inspection", "legacy")
        names = set(os.listdir(legacy_dir))
        self.assertNotIn("ocr", names)
        self.assertNotIn("rekognition.py", names)


# ---------------------------------------------------------------------------
# 6. FEATURE SEMANTICS (door / load / damage), on the right wagon
# ---------------------------------------------------------------------------

class TestFeatureSemantics(TempCase):

    def test_door_states_land_on_the_right_global_wagon(self):
        state = roster(10)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        rows = [side_damage_row(3, door_status="open", close_detected=False),
                side_damage_row(7, door_status="partially_closed",
                                close_detected=False, partial=True),
                side_damage_row(5, door_status="closed")]
        data = build_json_for(state, C.CAMERA_RIGHT_UP, self.cache,
                              damage_rows=rows)["inspection_data"]
        by_count = {s["wagon_count"]: s for s in data["wagon_segments"]}
        self.assertEqual(by_count[3]["door_status"], "open")
        self.assertEqual(by_count[3]["global_wagon_id"], "GW_3")
        self.assertEqual(by_count[7]["door_status"], "partially_closed")
        self.assertTrue(by_count[7]["door_partial_detected"])
        self.assertEqual(by_count[5]["door_status"], "closed")
        self.assertEqual(data["doors_open"], 1)
        self.assertEqual(data["doors_partially_closed"], 1)
        self.assertEqual(data["doors_closed"], 8)

    def test_side_damage_is_a_damage_finding_not_a_door_state(self):
        """door_state.pt's `damage` class must never become a door state."""
        state = roster(4)
        seed_cache(self.cache, state, [C.CAMERA_LEFT_UP])
        data = build_json_for(
            state, C.CAMERA_LEFT_UP, self.cache,
            damage_rows=[side_damage_row(2, damage=True)])["inspection_data"]
        by_count = {s["wagon_count"]: s for s in data["wagon_segments"]}
        self.assertTrue(by_count[2]["damage_detected"])
        self.assertEqual(by_count[2]["door_status"], "closed")
        self.assertEqual(data["damaged_wagons"], 1)

    def test_door_vote_only_covers_door_classes(self):
        detector = legacy_damage.DamageDetector(
            damage_model=None, flavour="side", confidence=0.5)
        self.assertEqual(
            detector._door_status({"damage": [{"frame_count": 99}]}), "closed")
        self.assertEqual(
            detector._door_status({"open_door": [{"frame_count": 4}],
                                   "closed_door": [{"frame_count": 2}]}), "open")
        self.assertEqual(
            detector._door_status({"open_door": [{"frame_count": 2}],
                                   "closed_door": [{"frame_count": 9}]}), "closed")

    def test_one_frame_flicker_cannot_open_a_wagon(self):
        """The legacy min_band_frames rule, exercised through the real filter."""
        detector = legacy_damage.DamageDetector(
            damage_model=None, flavour="side", confidence=0.5, min_band_frames=3)
        kept, dropped = detector._drop_short_bands(
            [{"frame_count": 1}, {"frame_count": 2}, {"frame_count": 5}])
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped, 2)
        self.assertEqual(detector._door_status({"open_door": kept[:0]}), "closed")

    def test_top_damage_subtypes_and_counts(self):
        state = roster(6)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP_TOP])
        rows = [top_damage_row(2, floor=True), top_damage_row(4, inner=True),
                top_damage_row(5, probable=True)]
        data = build_json_for(state, C.CAMERA_RIGHT_UP_TOP, self.cache,
                              damage_rows=rows)["inspection_data"]
        self.assertEqual(data["floor_dmg_wagons"], 1)
        self.assertEqual(data["inner_wall_dmg_wagons"], 1)
        self.assertEqual(data["floor_dmg_probable_wagons"], 1)
        self.assertEqual(data["damaged_wagons"], 2)
        self.assertEqual(data["probable_damage_wagons"], 1)
        by_count = {s["wagon_count"]: s for s in data["wagon_segments"]}
        self.assertTrue(by_count[4]["inner_wall_dmg_detected"])
        self.assertFalse(by_count[4]["floor_dmg_detected"])

    def test_top_class_map_matches_the_real_checkpoint_names(self):
        self.assertEqual(
            legacy_damage.TOP_MODEL_CLASS_MAP,
            {"Floor__probable_damage": "floor_dmg_probable",
             "Floor_damage": "floor_dmg",
             "Inner_wall_damage": "inner_wall_dmg"})

    def test_side_classes_match_door_state_pt(self):
        self.assertEqual(set(legacy_damage.SIDE_CLASSES),
                         li.EXPECTED_SIDE_CLASSES)

    def test_loaded_wagon_becomes_wagon_loaded(self):
        state = roster(4)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP_TOP])
        data = build_json_for(
            state, C.CAMERA_RIGHT_UP_TOP, self.cache,
            load_status_by_wagon={"GW_2": C.LOAD_LOADED,
                                  "GW_3": C.LOAD_EMPTY})["inspection_data"]
        by_count = {s["wagon_count"]: s for s in data["wagon_segments"]}
        self.assertEqual(by_count[2]["segment_type"], "wagon_loaded")
        self.assertEqual(by_count[2]["load_status"], "loaded")
        self.assertEqual(by_count[3]["load_status"], "empty")
        self.assertEqual(data["wagons_loaded"], 1)
        self.assertEqual(data["wagons_empty"], 3)
        self.assertEqual(data["wagons_loaded"] + data["wagons_empty"],
                         data["total_wagons"])

    def test_unlabeled_load_abstains(self):
        """load.pt's `Unlabeled` must map to neither LOADED nor EMPTY."""
        mapping = C.resolve_load_label_mapping(
            {0: "Empty", 1: "Loaded", 2: "Unlabeled"})
        self.assertEqual(mapping.get("empty"), C.LOAD_EMPTY)
        self.assertEqual(mapping.get("loaded"), C.LOAD_LOADED)
        self.assertNotIn("unlabeled", mapping)

    def test_loaded_floor_suppression_is_the_legacy_rule(self):
        self.assertEqual(legacy_damage.LOADED_SEGMENT_TYPE, "wagon_loaded")
        self.assertEqual(legacy_damage.FLOOR_DAMAGE_CLASSES,
                         frozenset({"floor_dmg", "floor_dmg_probable"}))
        self.assertNotIn("inner_wall_dmg", legacy_damage.FLOOR_DAMAGE_CLASSES)

    def test_side_camera_never_load_classifies(self):
        state = roster(3)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        data = build_json_for(
            state, C.CAMERA_RIGHT_UP, self.cache,
            load_status_by_wagon={"GW_1": C.LOAD_LOADED})["inspection_data"]
        self.assertEqual(data["wagon_segments"][0]["segment_type"], "wagon")
        self.assertNotIn("wagons_loaded", data)


# ---------------------------------------------------------------------------
# 7. ENGINE / BRAKEVAN EXCLUSION
# ---------------------------------------------------------------------------

class TestNonWagonSegments(TempCase):

    def test_engine_and_brakevan_are_counted_not_reported_as_wagons(self):
        state = roster(8, classifications={1: SegmentClass.ENGINE,
                                           8: SegmentClass.BRAKE_VAN})
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP_TOP])
        data = build_json_for(state, C.CAMERA_RIGHT_UP_TOP,
                              self.cache)["inspection_data"]
        self.assertEqual(data["num_engines"], 1)
        self.assertEqual(data["num_brakevans"], 1)
        self.assertEqual(data["total_wagons"], 6)
        self.assertEqual(len(data["wagon_segments"]), 6)
        self.assertNotIn(1, [s["segment_id"] for s in data["wagon_segments"]])

    def test_damage_on_an_engine_is_not_a_damaged_wagon(self):
        state = roster(5, classifications={1: SegmentClass.ENGINE})
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP_TOP])
        data = build_json_for(
            state, C.CAMERA_RIGHT_UP_TOP, self.cache,
            damage_rows=[top_damage_row(1, floor=True)])["inspection_data"]
        self.assertEqual(data["damaged_wagons"], 0)
        self.assertEqual(data["num_engines"], 1)

    def test_detector_skips_dominant_non_wagon_segments(self):
        detector = legacy_damage.DamageDetector(
            damage_model=None, flavour="top", confidence=0.5,
            min_non_wagon_dominance=0.80)
        self.assertTrue(detector._is_trusted_non_wagon(
            {"segment_type": "engine", "type_dominance": 1.0}))
        self.assertFalse(detector._is_trusted_non_wagon(
            {"segment_type": "engine", "type_dominance": 0.5}))
        self.assertFalse(detector._is_trusted_non_wagon(
            {"segment_type": "wagon", "type_dominance": 1.0}))

    def test_bridge_maps_roster_classes_to_legacy_vocabulary(self):
        self.assertEqual(gb.roster_segment_type(SegmentClass.ENGINE), "engine")
        self.assertEqual(gb.roster_segment_type(SegmentClass.BRAKE_VAN), "brakevan")
        self.assertEqual(gb.roster_segment_type(SegmentClass.WAGON), "wagon")
        self.assertEqual(
            gb.roster_segment_type(SegmentClass.WAGON, C.LOAD_LOADED),
            "wagon_loaded")
        self.assertEqual(
            gb.roster_segment_type(SegmentClass.ENGINE, C.LOAD_LOADED), "engine",
            "load must never override the roster's classification")


# ---------------------------------------------------------------------------
# 8. ASSOCIATION: unresolved / not visible / ambiguous
# ---------------------------------------------------------------------------

class TestAssociation(TempCase):

    def test_unresolved_camera_is_not_guessed(self):
        state = roster(5)
        status = li._evidence_status("UNRESOLVED", has_window=True,
                                     scanned=True, found=False)
        self.assertEqual(status, li.STATUS_UNRESOLVED)

    def test_not_visible_is_distinct_from_no_detection(self):
        self.assertEqual(
            li._evidence_status("RESOLVED", has_window=False, scanned=False,
                                found=False), li.STATUS_NOT_VISIBLE)
        self.assertEqual(
            li._evidence_status("RESOLVED", has_window=True, scanned=True,
                                found=False), li.STATUS_NO_DETECTION)
        self.assertEqual(
            li._evidence_status("RESOLVED", has_window=True, scanned=True,
                                found=True), li.STATUS_INSPECTED)

    def test_all_four_statuses_are_distinct_values(self):
        values = {li.STATUS_INSPECTED, li.STATUS_NO_DETECTION,
                  li.STATUS_NOT_VISIBLE, li.STATUS_UNRESOLVED,
                  li.STATUS_AMBIGUOUS}
        self.assertEqual(len(values), 5)

    def test_a_wagon_outside_the_footage_produces_no_scan_row(self):
        state = roster(10)
        seed_cache(self.cache, state, [C.CAMERA_LEFT_UP], only_indices={2, 3})
        scan_df = gb.build_segment_summary(
            state, C.CAMERA_LEFT_UP,
            windows_for(state, C.CAMERA_LEFT_UP, {2, 3}), self.cache,
            require_frames=True)
        self.assertEqual(sorted(scan_df["segment_id"]), [2, 3])

    def test_frames_are_never_read_from_another_wagons_directory(self):
        state = roster(4)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        df = gb.build_segment_summary(
            state, C.CAMERA_RIGHT_UP, windows_for(state, C.CAMERA_RIGHT_UP),
            self.cache, require_frames=True)
        for _, row in df.iterrows():
            self.assertIn(f"GW_{int(row['segment_id'])}", row["directory"])
            self.assertTrue(row["directory"].endswith(C.CAMERA_RIGHT_UP))


# ---------------------------------------------------------------------------
# 9. EVIDENCE / PROBLEM FRAMES / ARTIFACTS
# ---------------------------------------------------------------------------

class TestEvidence(TempCase):

    def test_representative_positions_are_the_legacy_contract(self):
        cfg = li.LegacyInspectionConfig()
        self.assertEqual(cfg.representative_positions, (0.25, 0.55, 0.80))
        self.assertEqual(cfg.representative_position_names,
                         ("start", "mid1", "end"))

    def test_publisher_writes_three_frames_per_wagon(self):
        state = roster(3)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP], frames=60)
        df = gb.build_segment_summary(
            state, C.CAMERA_RIGHT_UP, windows_for(state, C.CAMERA_RIGHT_UP),
            self.cache, require_frames=True)
        sink = li.LocalArtifactSink(os.path.join(self.tmp, "artifacts"))
        publisher = ArtifactPublisher(
            s3=sink, artifact_bucket="b", region="r",
            camera_folder="cam", damage_flavour="side")
        _ts, index, _loco, _pf = publisher.publish(
            upload_timestamp=TS, segment_summary_df=df,
            loco_summary_df=pd.DataFrame(), problem_frames_df=pd.DataFrame(),
            wagon_count_map=gb.build_wagon_count_map(df),
            local_workdir=self.tmp)
        self.assertEqual(set(index), {1, 2, 3})
        self.assertEqual([e["position"] for e in index[1]],
                         ["start", "mid1", "end"])
        self.assertEqual(len(sink.uploads), 9)

    def test_missing_evidence_frame_is_skipped_not_faked(self):
        state = roster(2)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP], only_indices={1},
                   frames=60)
        df = gb.build_segment_summary(
            state, C.CAMERA_RIGHT_UP, windows_for(state, C.CAMERA_RIGHT_UP),
            self.cache, require_frames=False, include_unwindowed=True)
        sink = li.LocalArtifactSink(os.path.join(self.tmp, "artifacts"))
        publisher = ArtifactPublisher(
            s3=sink, artifact_bucket="b", region="r", camera_folder="cam",
            damage_flavour="side")
        _ts, index, _loco, _pf = publisher.publish(
            upload_timestamp=TS, segment_summary_df=df,
            loco_summary_df=pd.DataFrame(), problem_frames_df=pd.DataFrame(),
            wagon_count_map=gb.build_wagon_count_map(df),
            local_workdir=self.tmp)
        self.assertEqual(len(index[1]), 3)
        self.assertEqual(index[2], [])

    def test_problem_frame_uploads_annotated_or_raw_never_both(self):
        state = roster(1)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP], frames=30)
        df = gb.build_segment_summary(
            state, C.CAMERA_RIGHT_UP, windows_for(state, C.CAMERA_RIGHT_UP),
            self.cache, require_frames=True)
        raw = os.path.join(df.iloc[0]["directory"], "frame_000000.jpg")
        annotated = os.path.join(self.tmp, "annotated.jpg")
        shutil.copyfile(raw, annotated)
        problems = pd.DataFrame([{
            "wagon_id": 1, "problem_type": "open_door", "frame_number": 0,
            "frame_path": raw, "annotated_image_path": annotated,
            "bounding_box": [{"bbox": [1.0, 2.0, 3.0, 4.0], "confidence": 0.9,
                              "class_name": "open_door"}]}])
        sink = li.LocalArtifactSink(os.path.join(self.tmp, "artifacts"))
        publisher = ArtifactPublisher(
            s3=sink, artifact_bucket="b", region="r", camera_folder="cam",
            damage_flavour="side")
        _ts, _idx, _loco, entries = publisher.publish(
            upload_timestamp=TS, segment_summary_df=df,
            loco_summary_df=pd.DataFrame(), problem_frames_df=problems,
            wagon_count_map=gb.build_wagon_count_map(df),
            local_workdir=self.tmp)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertTrue(entry["is_annotated"])
        self.assertEqual(entry["annotated_image_url"], entry["s3_url"])
        self.assertEqual(entry["bounding_box"], [1.0, 2.0, 3.0, 4.0])
        keys = [k for _b, k in sink.uploads if "problem_frames/" in k]
        self.assertEqual(len(keys), 1, "one upload per evidence entry")

    def test_duplicate_problem_frames_are_kept_distinct_per_type(self):
        """Two problem types on one frame are two entries, one file each."""
        state = roster(1)
        seed_cache(self.cache, state, [C.CAMERA_LEFT_UP], frames=30)
        df = gb.build_segment_summary(
            state, C.CAMERA_LEFT_UP, windows_for(state, C.CAMERA_LEFT_UP),
            self.cache, require_frames=True)
        raw = os.path.join(df.iloc[0]["directory"], "frame_000000.jpg")
        problems = pd.DataFrame([
            {"wagon_id": 1, "problem_type": "open_door", "frame_number": 0,
             "frame_path": raw, "annotated_image_path": None,
             "bounding_box": []},
            {"wagon_id": 1, "problem_type": "damage", "frame_number": 0,
             "frame_path": raw, "annotated_image_path": None,
             "bounding_box": []}])
        sink = li.LocalArtifactSink(os.path.join(self.tmp, "artifacts"))
        publisher = ArtifactPublisher(
            s3=sink, artifact_bucket="b", region="r", camera_folder="cam",
            damage_flavour="side")
        _ts, _idx, _loco, entries = publisher.publish(
            upload_timestamp=TS, segment_summary_df=df,
            loco_summary_df=pd.DataFrame(), problem_frames_df=problems,
            wagon_count_map=gb.build_wagon_count_map(df),
            local_workdir=self.tmp)
        self.assertEqual(len(entries), 2)
        self.assertEqual(len({e["s3_key"] for e in entries}), 2,
                         "distinct problem types must not overwrite each other")
        for entry in entries:
            self.assertFalse(entry["is_annotated"])
            self.assertIsNone(entry["annotated_image_url"])

    def test_problem_frames_carry_the_global_wagon_id(self):
        state = roster(9)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        entries = [{"wagon_id": 6, "wagon_count": 6, "segment_type": "wagon",
                    "problem_type": "damage", "frame_number": 120,
                    "filename": "f.jpg", "s3_key": "k", "s3_url": "u",
                    "is_annotated": True, "annotated_image_url": "u",
                    "bounding_box": [1.0, 2.0, 3.0, 4.0]}]
        data = build_json_for(
            state, C.CAMERA_RIGHT_UP, self.cache,
            damage_rows=[side_damage_row(6, damage=True)],
            problem_entries=entries)["inspection_data"]
        self.assertEqual(data["total_problem_frames"], 1)
        self.assertEqual(data["problem_frames_by_type"]["damage"], 1)
        pf = data["problem_frames"][0]
        self.assertEqual(pf["wagon_count"], 6)
        self.assertEqual(pf["global_wagon_id"], "GW_6")
        self.assertTrue(pf["damage_detected"])
        self.assertTrue(pf["is_annotated"])

    def test_side_filename_pattern_is_the_legacy_one(self):
        publisher = ArtifactPublisher(
            s3=li.LocalArtifactSink(self.tmp), artifact_bucket="b", region="r",
            camera_folder="cam", damage_flavour="side")
        self.assertEqual(
            publisher._wagon_frame_filename("wagon", 17, 1234, "start"),
            "w17_frame_001234.jpg")

    def test_top_filename_pattern_is_the_legacy_one(self):
        publisher = ArtifactPublisher(
            s3=li.LocalArtifactSink(self.tmp), artifact_bucket="b", region="r",
            camera_folder="cam", damage_flavour="top")
        self.assertEqual(
            publisher._wagon_frame_filename("wagon_loaded", 17, 1234, "mid1"),
            "wagon_loaded_017_frame_001234_mid1.jpg")

    def test_display_segment_type_matches_legacy(self):
        self.assertEqual(display_segment_type("wagon", "top"), "wagon_empty")
        self.assertEqual(display_segment_type("wagon_loaded", "top"), "wagon_loaded")
        self.assertEqual(display_segment_type("wagon_loaded", "side"), "wagon")
        self.assertEqual(display_segment_type("engine", "top"), "engine")


# ---------------------------------------------------------------------------
# 10. RENDERERS DO NOT INFER
# ---------------------------------------------------------------------------

class TestRenderersAreInferenceFree(TempCase):
    """The renderer may describe inference in prose; it may not perform it.

    The check is over the parsed SYNTAX TREE, not the file text, so it cannot be
    satisfied or broken by a comment -- and it catches the things that actually
    run a model: an import of a model library, a call to a detector, or a
    reference to a weights file.
    """

    BANNED_IMPORTS = {"ultralytics", "torch", "torchvision"}
    BANNED_CALLS = {"YOLO", "DamageDetector", "ProblemFrameExtractor",
                    "predict", "load_yolo", "_load_model"}

    def _tree(self):
        import ast
        return ast.parse(open(lr.__file__, encoding="utf-8").read())

    def test_render_module_imports_no_model_library(self):
        import ast
        for node in ast.walk(self._tree()):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                self.assertNotIn(
                    root, self.BANNED_IMPORTS,
                    f"legacy_render imports {name!r}: the PDF/video layer must "
                    f"consume persisted state, never run a model")

    def test_render_module_calls_no_detector(self):
        import ast
        for node in ast.walk(self._tree()):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute) else "")
            self.assertNotIn(
                name, self.BANNED_CALLS,
                f"legacy_render calls {name!r}: a second inference pass here "
                f"would be free to disagree with the persisted JSON")

    def test_render_module_references_no_weights(self):
        src = open(lr.__file__, encoding="utf-8").read()
        self.assertNotIn(".pt", src, "no weight file may be named here")

    def test_render_module_reads_persisted_files(self):
        src = open(lr.__file__, encoding="utf-8").read()
        self.assertIn("inspection_data.json", src)
        self.assertIn("frame_detections.csv", src)

    def test_combined_pdf_is_built_from_json_only(self):
        state = roster(6)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        out = os.path.join(self.tmp, "out")
        for camera_id in C.ALL_CAMERAS:
            work = os.path.join(out, gb.camera_profile(camera_id).legacy_name)
            os.makedirs(work, exist_ok=True)
            li._write_json(os.path.join(work, "inspection_data.json"),
                           build_json_for(state, camera_id, self.cache))
        payloads = lr.load_camera_payloads(out)
        self.assertEqual(len(payloads), 4)
        pdf_path = lr.build_combined_pdf(state, out, payloads=payloads,
                                         verbose=False)
        self.assertIsNotNone(pdf_path)
        self.assertTrue(os.path.getsize(pdf_path) > 0)

    def test_combined_report_covers_every_wagon_with_no_findings(self):
        state = roster(15)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        payloads = {cam: build_json_for(state, cam, self.cache)
                    for cam in C.ALL_CAMERAS}
        rows = lr.combined_wagon_rows(state, payloads)
        self.assertEqual(len(rows), 15)
        for row in rows:
            self.assertEqual(row["damage_status"], "OK")
            self.assertEqual(row["damage_types"], [])
            self.assertEqual(row["door_status"], "closed")

    def test_door_fusion_never_invents_closed(self):
        self.assertEqual(lr._fuse_door([]), C.NO_DATA)
        self.assertEqual(lr._fuse_door(["closed", "open"]), "open")
        self.assertEqual(lr._fuse_door(["closed", "partially_closed"]),
                         "partially_closed")
        self.assertEqual(lr._fuse_door(["closed", "closed"]), "closed")


# ---------------------------------------------------------------------------
# 11. MODELS: authority, S3 URIs, class verification, CPU safety
# ---------------------------------------------------------------------------

class TestModelResolution(TempCase):

    def test_local_path_passes_through(self):
        path = os.path.join(self.tmp, "w.pt")
        open(path, "wb").close()
        self.assertEqual(
            li.resolve_feature_model(path, self.tmp), path)

    def test_bare_filename_resolves_in_models_dir(self):
        path = os.path.join(self.tmp, "door_state.pt")
        open(path, "wb").close()
        self.assertEqual(
            li.resolve_feature_model("door_state.pt", self.tmp), path)

    def test_s3_uri_is_detected(self):
        from inspection.legacy.model_store import is_remote_uri
        self.assertTrue(is_remote_uri("s3://bucket/key.pt"))
        self.assertFalse(is_remote_uri("/models/key.pt"))

    def test_s3_uri_downloads_and_caches_by_etag(self):
        from inspection.legacy.model_store import resolve_path

        source = os.path.join(self.tmp, "remote.pt")
        with open(source, "wb") as fh:
            fh.write(b"weights")
        calls = {"download": 0, "head": 0}

        class FakeS3:
            def __init__(self):
                self.client = self

            def head_object(self, Bucket, Key):   # noqa: N803 (boto3 casing)
                calls["head"] += 1
                return {"ETag": '"abc123"'}

            def download_file(self, bucket, key, dest):
                calls["download"] += 1
                shutil.copyfile(source, dest)

        cache = os.path.join(self.tmp, "cache")
        first = resolve_path("s3://bucket/models/remote.pt",
                             s3_client=FakeS3(), cache_dir=cache)
        second = resolve_path("s3://bucket/models/remote.pt",
                              s3_client=FakeS3(), cache_dir=cache)
        self.assertEqual(first, second)
        self.assertTrue(os.path.isfile(first))
        self.assertEqual(calls["download"], 1, "cached copy must be reused")

    def test_missing_local_model_raises_not_silently_passes(self):
        from inspection.legacy.model_store import resolve_path
        with self.assertRaises(FileNotFoundError):
            resolve_path(os.path.join(self.tmp, "nope.pt"))

    def test_side_model_is_authoritative_and_singular(self):
        cfg = li.LegacyInspectionConfig()
        self.assertEqual(cfg.side_model, C.MODEL_DOOR_STATE)
        self.assertEqual(cfg.top_model, C.MODEL_DAMAGE)
        self.assertNotEqual(cfg.side_model, cfg.top_model)

    def test_no_gpu_only_half_path(self):
        for module in (li, lr):
            src = open(module.__file__, encoding="utf-8").read()
            self.assertNotIn("half=True", src)
        src = open(legacy_damage.__file__, encoding="utf-8").read()
        self.assertNotIn("half=True", src,
                         "the legacy damage detector must not force fp16 on CPU")

    def test_class_names_are_resolved_by_name_never_by_id(self):
        src = open(legacy_damage.__file__, encoding="utf-8").read()
        self.assertIn("self.damage_model.names[int(box.cls[0])]", src)
        self.assertIn("TOP_MODEL_CLASS_MAP.get(raw_name)", src)


# ---------------------------------------------------------------------------
# 12. NO STATE LEAKS BETWEEN TRAINS
# ---------------------------------------------------------------------------

class TestNoCrossTrainLeak(TempCase):

    def test_model_handles_are_reset_per_run(self):
        li._MODEL_HANDLES["stale"] = object()
        li.reset_inspection_state()
        self.assertEqual(li._MODEL_HANDLES, {})

    def test_run_resets_state_at_start(self):
        state = roster(2)
        li._MODEL_HANDLES["train_a"] = object()
        plan = FakePlan(self.cache, [], {c: "UNRESOLVED" for c in C.ALL_CAMERAS})
        li.run_legacy_inspection(
            state=state, tracks_by_camera={}, plan=plan,
            models_dir=os.path.join(ROOT, "models"),
            output_root=os.path.join(self.tmp, "out"),
            cfg=li.LegacyInspectionConfig(side_model="__absent__.pt",
                                          top_model="__absent__.pt"),
            verbose=False)
        self.assertNotIn("train_a", li._MODEL_HANDLES)

    def test_second_train_cannot_read_the_first_trains_frames(self):
        train_a = roster(3)
        seed_cache(self.cache, train_a, [C.CAMERA_RIGHT_UP])
        self.assertTrue(os.path.isdir(os.path.join(self.cache, "GW_1")))
        wc.clear_wagon_cache(self.cache, verbose=False)
        train_b = roster(3)
        scan_df = gb.build_segment_summary(
            train_b, C.CAMERA_RIGHT_UP, windows_for(train_b, C.CAMERA_RIGHT_UP),
            self.cache, require_frames=True)
        self.assertEqual(len(scan_df), 0,
                         "train B must not inherit train A's cached frames")

    def test_two_runs_in_one_process_are_independent(self):
        first = roster(4)
        second = roster(9)
        seed_cache(self.cache, first, [C.CAMERA_RIGHT_UP])
        a = build_json_for(first, C.CAMERA_RIGHT_UP, self.cache)
        shutil.rmtree(self.cache, ignore_errors=True)
        seed_cache(self.cache, second, [C.CAMERA_RIGHT_UP])
        b = build_json_for(second, C.CAMERA_RIGHT_UP, self.cache)
        self.assertEqual(a["inspection_data"]["total_wagons"], 4)
        self.assertEqual(b["inspection_data"]["total_wagons"], 9)


# ---------------------------------------------------------------------------
# 13. DETERMINISM
# ---------------------------------------------------------------------------

class TestDeterminism(TempCase):

    def test_repeat_runs_produce_identical_json(self):
        state = roster(23)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        rows = [side_damage_row(i, door_status="open") for i in (3, 11, 20)]
        first = build_json_for(state, C.CAMERA_RIGHT_UP, self.cache,
                               damage_rows=rows)
        second = build_json_for(state, C.CAMERA_RIGHT_UP, self.cache,
                                damage_rows=rows)
        self.assertEqual(json.dumps(first, sort_keys=True, default=str),
                         json.dumps(second, sort_keys=True, default=str))

    def test_segment_summary_is_ordered_by_global_index(self):
        state = roster(30)
        seed_cache(self.cache, state, [C.CAMERA_LEFT_UP_TOP])
        windows = list(reversed(windows_for(state, C.CAMERA_LEFT_UP_TOP)))
        df = gb.build_segment_summary(
            state, C.CAMERA_LEFT_UP_TOP, windows, self.cache,
            require_frames=True)
        self.assertEqual(list(df["segment_id"]), list(range(1, 31)))

    def test_combined_rows_are_stable(self):
        state = roster(18)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        payloads = {cam: build_json_for(state, cam, self.cache)
                    for cam in C.ALL_CAMERAS}
        a = lr.combined_wagon_rows(state, payloads)
        b = lr.combined_wagon_rows(state, payloads)
        self.assertEqual(json.dumps(a, sort_keys=True),
                         json.dumps(b, sort_keys=True))


# ---------------------------------------------------------------------------
# 14. THE LEGACY SOURCE IS THE ONE RUNNING
# ---------------------------------------------------------------------------

class TestLegacyProvenance(unittest.TestCase):
    """The vendored files must stay identical to the added legacy codebase."""

    SRC = os.path.join(
        ROOT, "rithish__code_1", "CCTV-TrainVideo-ML-V2-wagon-Rithish",
        "Train-Inspection-Engine", "src", "train_inspection_engine")
    VENDORED = os.path.join(ROOT, "inspection", "legacy")

    PAIRS = {
        "bands.py": "inspection/bands.py",
        "frame_positions.py": "inspection/frame_positions.py",
        "json_builder.py": "reporting/json_builder.py",
        "pdf_builder.py": "reporting/pdf_builder.py",
        "model_store.py": "core/model_store.py",
        "url_utils.py": "utils/url_utils.py",
        "serialization.py": "utils/serialization.py",
        "video_io.py": "core/video_io.py",
    }
    """Files vendored byte-for-byte. damage.py / artifacts.py / s3.py /
    annotated_video.py are excluded only because their import lines were
    rewritten for the flat package (and damage.py's tqdm guard); their bodies
    are covered by the behavioural tests above."""

    def test_vendored_files_are_byte_identical_to_the_legacy_source(self):
        if not os.path.isdir(self.SRC):
            self.skipTest("legacy source tree not present")
        for vendored, original in self.PAIRS.items():
            with open(os.path.join(self.VENDORED, vendored), "rb") as fh:
                got = fh.read()
            with open(os.path.join(self.SRC, *original.split("/")), "rb") as fh:
                want = fh.read()
            self.assertEqual(got, want, f"{vendored} drifted from the legacy source")

    def test_legacy_counting_modules_were_not_vendored(self):
        names = set(os.listdir(self.VENDORED))
        for banned in ("segments.py", "segment_finder.py", "extractor.py",
                       "classifier.py", "base_pipeline.py", "pipeline.py"):
            self.assertNotIn(banned, names,
                             f"{banned} is legacy camera-wise COUNTING and must "
                             f"not be part of the inspection layer")

    def test_orchestrator_calls_legacy_rather_than_reimplementing(self):
        src = open(li.__file__, encoding="utf-8").read()
        for symbol in ("DamageDetector", "ProblemFrameExtractor",
                       "ArtifactPublisher", "build_inspection_json"):
            self.assertIn(symbol, src)
        self.assertNotIn("def _detect_", src,
                         "detection must live in the legacy module, not here")


# ---------------------------------------------------------------------------
# 15. PROTECTED COUNTING MODULES ARE UNTOUCHED
# ---------------------------------------------------------------------------

class TestCountingModulesProtected(unittest.TestCase):

    PROTECTED = ("global_fusion.py", "gap_validation.py", "fragment_stitching.py",
                 "train_structure.py", "temporal_classification.py",
                 "tracker_engine.py", "global_alignment.py",
                 "global_train_state.py", "video_segmenter.py")

    def test_inspection_never_imports_a_counting_module_for_writing(self):
        for module in (li, lr, gb):
            src = open(module.__file__, encoding="utf-8").read()
            for banned in ("import global_fusion", "import gap_validation",
                           "import fragment_stitching", "import train_structure"):
                self.assertNotIn(banned, src)

    def test_inspection_does_not_assign_to_state_wagons(self):
        for module in (li, lr, gb):
            src = open(module.__file__, encoding="utf-8").read()
            for banned in ("state.wagons =", "state.total_wagons =",
                           "state.camera_offsets =", "state.global_gaps ="):
                self.assertNotIn(banned, src,
                                 f"{module.__name__} must never write {banned}")

    def test_protected_modules_exist(self):
        for name in self.PROTECTED:
            self.assertTrue(os.path.isfile(os.path.join(ROOT, name)), name)


# ---------------------------------------------------------------------------
# 16. ONE SET OF VERDICTS BEHIND EVERY ARTIFACT
# ---------------------------------------------------------------------------

class TestReconciliation(TempCase):
    """The dashboard JSON and the combined report must not disagree."""

    def _result(self, state, payload_by_camera):
        result = li.LegacyInspectionResult(
            global_wagon_count=len(state.wagons))
        for camera_id, payload in payload_by_camera.items():
            cam = li.LegacyCameraResult(
                camera_id=camera_id,
                flavour=gb.camera_profile(camera_id).flavour)
            cam.payload = payload
            result.cameras[camera_id] = cam
        return result

    def _unified(self, state):
        from core.unified_wagon_state import UnifiedWagonState
        return {w.global_id: UnifiedWagonState(global_id=w.global_id,
                                               classification=w.classification)
                for w in state.wagons}

    def test_door_verdicts_reach_the_combined_report_view(self):
        state = roster(6)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP])
        payloads = {
            C.CAMERA_RIGHT_UP: build_json_for(
                state, C.CAMERA_RIGHT_UP, self.cache,
                damage_rows=[side_damage_row(2, door_status="open")]),
            C.CAMERA_LEFT_UP: build_json_for(
                state, C.CAMERA_LEFT_UP, self.cache,
                damage_rows=[side_damage_row(4, door_status="partially_closed")]),
        }
        unified = self._unified(state)
        applied = li.apply_to_unified(unified, self._result(state, payloads))
        self.assertEqual(unified["GW_2"].right_door, C.DOOR_OPEN)
        self.assertEqual(unified["GW_4"].left_door, C.DOOR_PARTIAL)
        self.assertEqual(unified["GW_1"].right_door, C.DOOR_CLOSED)
        self.assertTrue(unified["GW_2"].has_open_door)
        self.assertEqual(applied["door"], 12)

    def test_uninspected_wagon_stays_no_data(self):
        state = roster(6)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP], only_indices={2})
        payloads = {C.CAMERA_RIGHT_UP: build_json_for(
            state, C.CAMERA_RIGHT_UP, self.cache, only_indices={2})}
        unified = self._unified(state)
        li.apply_to_unified(unified, self._result(state, payloads))
        self.assertEqual(unified["GW_2"].right_door, C.DOOR_CLOSED)
        self.assertEqual(unified["GW_5"].right_door, C.NO_DATA,
                         "a wagon this camera never saw must not be reported "
                         "as a closed door")
        self.assertEqual(unified["GW_5"].side_damage, C.NO_DATA)

    def test_damage_reaches_both_documents_identically(self):
        state = roster(5)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP_TOP])
        payload = build_json_for(
            state, C.CAMERA_RIGHT_UP_TOP, self.cache,
            damage_rows=[top_damage_row(3, inner=True)])
        unified = self._unified(state)
        li.apply_to_unified(unified, self._result(
            state, {C.CAMERA_RIGHT_UP_TOP: payload}))
        json_damaged = {s["wagon_count"]
                        for s in payload["inspection_data"]["wagon_segments"]
                        if s["damage_detected"]}
        report_damaged = {int(gid.split("_")[1]) for gid, u in unified.items()
                          if u.top_damage == C.DAMAGE_PRESENT}
        self.assertEqual(json_damaged, report_damaged)
        self.assertEqual(json_damaged, {3})

    def test_camera_authority_matches_old_code(self):
        self.assertEqual(li._DOOR_STATUS_TO_STATE,
                         {"open": C.DOOR_OPEN,
                          "partially_closed": C.DOOR_PARTIAL,
                          "closed": C.DOOR_CLOSED})


# ---------------------------------------------------------------------------
# 17. RUNNER WIRING AND STAGE ORDER
# ---------------------------------------------------------------------------

class TestRunnerWiring(unittest.TestCase):
    """Order is a correctness property here, so it is asserted statically.

    Running the real pipeline is an EC2 activity; what can be checked locally is
    that the stages are wired in the only order that is correct.
    """

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "run_global_count.py"),
                  encoding="utf-8") as fh:
            cls.src = fh.read()

    def _pos(self, needle):
        idx = self.src.find(needle)
        self.assertNotEqual(idx, -1, f"{needle!r} not found in run_global_count.py")
        return idx

    def test_cli_exposes_the_new_switches(self):
        for flag in ("--no-legacy-inspection", "--door-source", "--side-model",
                     "--top-model", "--artifact-bucket", "--upload-artifacts",
                     "--annotated-videos", "--aws-region"):
            self.assertIn(flag, self.src)

    def test_exactly_one_door_implementation_runs(self):
        self.assertIn('args.door_source == "old_code"', self.src,
                      "the old_code door path must be gated on --door-source, "
                      "so both implementations never run on the same frames")

    def test_inspection_runs_after_fusion_and_before_state_json(self):
        self.assertLess(self._pos("run_legacy_inspection("),
                        self._pos('"global_train_state.json"'))
        self.assertLess(self._pos("STEPS 8-11"), self._pos("STEPS 12-14"))

    def test_renderers_run_after_detection_is_persisted(self):
        self.assertLess(self._pos("legi.run_legacy_inspection("),
                        self._pos("legr.build_all_legacy_outputs("))

    def test_reconciliation_precedes_the_old_report(self):
        self.assertLess(self._pos("legi.apply_to_unified("),
                        self._pos("oldr.build_all_reports("))

    def test_roster_guard_is_in_a_finally(self):
        tail = self.src[self._pos("STEPS 12-14"):]
        finally_idx = tail.find("finally:")
        guard_idx = tail.find("assert_roster_unchanged")
        self.assertNotEqual(finally_idx, -1)
        self.assertLess(finally_idx, guard_idx,
                        "the roster check must run even when a feature raises")

    def test_cache_is_cleared_unless_explicitly_kept(self):
        self.assertIn("if not args.keep_wagon_cache:", self.src)
        self.assertIn("iwc.clear_wagon_cache(", self.src)

    def test_upload_is_opt_in(self):
        self.assertIn("args.upload_artifacts and args.artifact_bucket", self.src)

    def test_old_code_damage_is_off_when_the_legacy_layer_owns_it(self):
        """Both read top_damage.pt; only one may run."""
        self.assertIn("run_damage=(not args.no_damage and not _legacy_enabled)",
                      self.src)

    def test_legacy_side_pass_is_off_when_old_code_owns_the_door(self):
        """Both read door_state.pt; only one may run."""
        self.assertIn('args.door_source == "legacy"', self.src)
        self.assertIn("run_side_damage=", self.src)

    def test_legacy_inspection_is_genuinely_skippable(self):
        self.assertIn("if not args.no_legacy_inspection:", self.src)


# ---------------------------------------------------------------------------
# 18. ONE OWNER PER TASK  (behavioural, not textual)
# ---------------------------------------------------------------------------

class TestOneOwnerPerTask(TempCase):
    """No model may be run twice over the same frames in one execution.

    These drive the real config objects rather than reading the runner's text,
    so they fail if the gating stops working even while the wiring line stays.
    """

    def _run(self, cfg):
        state = roster(4)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        plan = FakePlan(self.cache, windows_for(state, C.CAMERA_RIGHT_UP)
                        + windows_for(state, C.CAMERA_RIGHT_UP_TOP),
                        {c: "RESOLVED" for c in C.ALL_CAMERAS})
        return state, li.run_legacy_inspection(
            state=state,
            tracks_by_camera={c: FakeTracks() for c in C.ALL_CAMERAS},
            plan=plan, models_dir=os.path.join(ROOT, "models"),
            output_root=os.path.join(self.tmp, "out"), cfg=cfg, verbose=False)

    def test_disabled_side_pass_does_not_run_a_model(self):
        cfg = li.LegacyInspectionConfig(run_side_damage=False,
                                        run_top_damage=False)
        _state, res = self._run(cfg)
        side = res.cameras[C.CAMERA_RIGHT_UP]
        self.assertFalse(
            side.payload["inspection_data"]["damage_model_active"],
            "a disabled pass must advertise that no model ran")
        self.assertTrue(any("disabled" in w for w in side.warnings))

    def test_disabled_pass_still_reports_every_global_wagon(self):
        """Turning a pass off must not shrink the count in that camera's JSON."""
        cfg = li.LegacyInspectionConfig(run_side_damage=False,
                                        run_top_damage=False)
        state, res = self._run(cfg)
        for camera_id, cam in res.cameras.items():
            data = cam.payload["inspection_data"]
            self.assertEqual(data["global_wagon_count"], len(state.wagons),
                             camera_id)
            self.assertEqual(
                data["total_wagons"] + data.get("num_engines", 0)
                + data.get("num_brakevans", 0), len(state.wagons), camera_id)

    def test_disabled_pass_never_claims_a_negative_finding(self):
        """NO_DETECTION asserts the model looked and found nothing."""
        cfg = li.LegacyInspectionConfig(run_side_damage=False,
                                        run_top_damage=False)
        _state, res = self._run(cfg)
        for camera_id, cam in res.cameras.items():
            for seg in cam.payload["inspection_data"]["wagon_segments"]:
                self.assertNotEqual(
                    seg["inspection_status"], li.STATUS_NO_DETECTION,
                    f"{camera_id} GW {seg['wagon_count']}: a check that never "
                    f"ran must not report a real negative finding")

    def test_disabled_pass_cannot_overwrite_the_other_implementation(self):
        """The `--door-source old_code` guarantee, exercised end to end."""
        from core.unified_wagon_state import UnifiedWagonState

        state = roster(4)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        # What old_code's DoorTracker would have concluded.
        unified = {w.global_id: UnifiedWagonState(global_id=w.global_id)
                   for w in state.wagons}
        unified["GW_2"].right_door = C.DOOR_OPEN

        payload = build_json_for(state, C.CAMERA_RIGHT_UP, self.cache)
        payload["inspection_data"]["damage_model_active"] = False   # pass was off
        result = li.LegacyInspectionResult(global_wagon_count=len(state.wagons))
        cam = li.LegacyCameraResult(camera_id=C.CAMERA_RIGHT_UP, flavour="side")
        cam.payload = payload
        result.cameras[C.CAMERA_RIGHT_UP] = cam

        applied = li.apply_to_unified(unified, result)
        self.assertEqual(applied["door"], 0)
        self.assertEqual(
            unified["GW_2"].right_door, C.DOOR_OPEN,
            "a pass that did not run must not overwrite the verdict of the "
            "implementation that was actually selected")

    def test_active_pass_still_reconciles(self):
        """The guard must not block the normal path."""
        from core.unified_wagon_state import UnifiedWagonState

        state = roster(4)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        unified = {w.global_id: UnifiedWagonState(global_id=w.global_id)
                   for w in state.wagons}
        payload = build_json_for(
            state, C.CAMERA_RIGHT_UP, self.cache,
            damage_rows=[side_damage_row(2, door_status="open")])
        self.assertTrue(payload["inspection_data"]["damage_model_active"])
        result = li.LegacyInspectionResult(global_wagon_count=len(state.wagons))
        cam = li.LegacyCameraResult(camera_id=C.CAMERA_RIGHT_UP, flavour="side")
        cam.payload = payload
        result.cameras[C.CAMERA_RIGHT_UP] = cam

        li.apply_to_unified(unified, result)
        self.assertEqual(unified["GW_2"].right_door, C.DOOR_OPEN)

    def test_config_records_which_passes_ran(self):
        cfg = li.LegacyInspectionConfig(run_side_damage=False)
        self.assertIs(cfg.to_dict()["run_side_damage"], False)
        self.assertIs(cfg.to_dict()["run_top_damage"], True)


# ---------------------------------------------------------------------------
# 19. TEST COLLECTION HYGIENE
# ---------------------------------------------------------------------------

class TestCollectionConfig(unittest.TestCase):
    """`pytest -q` with no path must run this suite and nothing else.

    The added legacy tree contains `scripts/colab_ocr_gap_wagon_test.py`, which
    matches pytest's default `*_test.py` pattern and imports tqdm + the OCR
    stack. Without the config below, collecting it aborts the entire run before
    any real test executes.
    """

    def test_pytest_ini_scopes_collection_to_tests(self):
        path = os.path.join(ROOT, "pytest.ini")
        self.assertTrue(os.path.isfile(path), "pytest.ini is required")
        text = open(path, encoding="utf-8").read()
        self.assertIn("testpaths = tests", text)
        self.assertIn("rithish__code_1", text)

    def test_the_legacy_tree_is_still_on_disk(self):
        """It is excluded from collection, NOT deleted -- provenance needs it."""
        legacy = os.path.join(ROOT, "rithish__code_1")
        if not os.path.isdir(legacy):
            self.skipTest("legacy source tree not present")
        self.assertTrue(os.path.isdir(legacy))


# ---------------------------------------------------------------------------
# 20. GOLDEN SCHEMA -- real payloads from the OLD pipeline
# ---------------------------------------------------------------------------

FIXTURES = os.path.join(ROOT, "tests", "fixtures")


def golden(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as fh:
        return json.load(fh)


def walk_shape(node, path="", out=None):
    """Every leaf path and its JSON type. `null` is recorded as its own type
    because the legacy schema uses it as a real value (`load_condition`).

    Id-keyed maps (`segment_type_map`, `wagon_number_results`) collapse to a
    single `*` key, so this compares SHAPE and not how many wagons the fixture
    happened to contain -- a 9-wagon golden payload and a 3-wagon run have the
    same schema.
    """
    out = {} if out is None else out
    if isinstance(node, dict):
        id_keyed = bool(node) and all(str(k).isdigit() for k in node)
        for key, value in node.items():
            step = "*" if id_keyed else key
            walk_shape(value, f"{path}.{step}" if path else str(step), out)
    elif isinstance(node, list):
        out[path] = "list"
        if node:
            # The element's own paths hang off "[]" so the container marker
            # above is not overwritten by its first element's type.
            walk_shape(node[0], path + "[]", out)
    else:
        out[path] = ("null" if node is None
                     else "bool" if isinstance(node, bool)
                     else "int" if isinstance(node, int)
                     else "float" if isinstance(node, float) else "str")
    return out


class TestGoldenSchema(TempCase):
    """The emitted payload must be indistinguishable from the old pipeline's.

    Both fixtures are REAL ``inspection_data.json`` documents produced by the
    old system -- one side camera, one top camera. They are the contract: if the
    dashboard accepted these, it must accept ours.
    """

    def _dashboard(self, state, camera_id, **kw):
        return li.dashboard_payload(
            build_json_for(state, camera_id, self.cache, **kw))

    # ---- key sets ----------------------------------------------------

    def test_side_keys_match_the_golden_payload_exactly(self):
        ref = golden("legacy_inspection_data_side.json")["inspection_data"]
        state = roster(3)
        seed_cache(self.cache, state, [C.CAMERA_LEFT_UP])
        got = self._dashboard(state, C.CAMERA_LEFT_UP)["inspection_data"]
        self.assertEqual(list(got), list(ref),
                         "side payload keys/order drifted from the old pipeline")

    def test_top_keys_match_the_golden_payload_exactly(self):
        ref = golden("legacy_inspection_data_top.json")["inspection_data"]
        state = roster(3)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP_TOP])
        got = self._dashboard(state, C.CAMERA_RIGHT_UP_TOP)["inspection_data"]
        self.assertEqual(list(got), list(ref),
                         "top payload keys/order drifted from the old pipeline")

    def test_envelope_matches(self):
        for name, camera_id in (("legacy_inspection_data_side.json", C.CAMERA_LEFT_UP),
                                ("legacy_inspection_data_top.json", C.CAMERA_RIGHT_UP_TOP)):
            ref = golden(name)
            state = roster(2)
            seed_cache(self.cache, state, [camera_id])
            got = self._dashboard(state, camera_id)
            self.assertEqual(list(got), list(ref))
            self.assertEqual(got["version"], ref["version"])
            self.assertEqual(got["camera_id"], ref["camera_id"])

    # ---- wagon_segments ------------------------------------------------

    def test_top_wagon_segment_matches_the_golden_shape(self):
        ref = golden("legacy_inspection_data_top.json")
        ref_seg = ref["inspection_data"]["wagon_segments"][0]
        state = roster(3)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP_TOP])
        seg = self._dashboard(
            state, C.CAMERA_RIGHT_UP_TOP)["inspection_data"]["wagon_segments"][0]
        self.assertEqual(list(seg), list(ref_seg))
        for key, want in ref_seg.items():
            if key in ("wagon_frames", "segment_id", "wagon_count"):
                continue
            self.assertEqual(type(seg[key]), type(want), key)

    def test_wagon_frames_entry_shape(self):
        ref = golden("legacy_inspection_data_top.json")
        ref_frame = ref["inspection_data"]["wagon_segments"][0]["wagon_frames"][0]
        self.assertEqual(sorted(ref_frame), ["position", "s3_url"])
        state = roster(2)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP_TOP], frames=80)
        payload = _publish_and_build(state, C.CAMERA_RIGHT_UP_TOP, self.cache,
                                     self.tmp)
        frames = payload["inspection_data"]["wagon_segments"][0]["wagon_frames"]
        self.assertTrue(frames, "evidence frames must be emitted")
        for frame in frames:
            self.assertEqual(sorted(frame), ["position", "s3_url"])
        self.assertEqual([f["position"] for f in frames],
                         ["start", "mid1", "end"])

    # ---- types ---------------------------------------------------------

    def test_leaf_types_match_the_golden_payload(self):
        for name, camera_id in (("legacy_inspection_data_side.json", C.CAMERA_LEFT_UP),
                                ("legacy_inspection_data_top.json", C.CAMERA_RIGHT_UP_TOP)):
            ref = golden(name)
            state = roster(3)
            seed_cache(self.cache, state, [camera_id])
            got = self._dashboard(state, camera_id)
            ref_shape = walk_shape(ref)
            got_shape = walk_shape(got)
            for path, want in ref_shape.items():
                if path not in got_shape:
                    # Only ELEMENT paths may be absent, and only because the
                    # corresponding list is empty in this run (no problem
                    # frames, no configured buckets). Container and scalar
                    # paths must always be present -- those are the schema.
                    self.assertIn("[]", path, f"missing leaf {path}")
                    continue
                got_type = got_shape[path]
                if want == "null" or got_type == "null":
                    continue          # nullable in the legacy schema
                self.assertEqual(got_type, want,
                                 f"{name}: {path} is {got_type}, golden has {want}")

    # ---- the url block -------------------------------------------------

    def test_url_fields_follow_the_golden_naming(self):
        from datetime import datetime as _dt
        ref = golden("legacy_inspection_data_side.json")["inspection_data"]
        raw = ref["raw_video_name"]
        urls = li._video_urls(
            li.LegacyInspectionConfig(),
            gb.camera_profile(C.CAMERA_LEFT_UP), raw,
            _dt(2026, 8, 15, 18, 34, 33))
        self.assertEqual(urls["pdf_report_url"], ref["pdf_report_url"])
        self.assertEqual(urls["trimmed_video_url"], ref["trimmed_video_url"])
        self.assertEqual(urls["detected_video_url"], ref["detected_video_url"])
        self.assertEqual(urls["raw_video_urls"][0], ref["raw_video_urls"][0])

    def test_absent_bucket_yields_null_not_a_fabricated_url(self):
        from datetime import datetime as _dt
        cfg = li.LegacyInspectionConfig(
            raw_video_bucket="", trimmed_video_bucket="",
            detected_video_bucket="", report_bucket="")
        urls = li._video_urls(cfg, gb.camera_profile(C.CAMERA_LEFT_UP), "clip",
                              _dt(2026, 1, 1))
        self.assertIsNone(urls["pdf_report_url"])
        self.assertEqual(urls["raw_video_urls"], [])

    def test_explicit_urls_override_the_derived_ones(self):
        from datetime import datetime as _dt
        cfg = li.LegacyInspectionConfig(video_urls_by_camera={
            C.CAMERA_LEFT_UP: {"pdf_report_url": "https://example/x.pdf",
                               "raw_video_urls": ["https://example/a.mp4",
                                                  "https://example/b.mp4"]}})
        urls = li._video_urls(cfg, gb.camera_profile(C.CAMERA_LEFT_UP), "clip",
                              _dt(2026, 1, 1))
        self.assertEqual(urls["pdf_report_url"], "https://example/x.pdf")
        self.assertEqual(len(urls["raw_video_urls"]), 2)
        self.assertIn("biro-wagon-pre-processed", urls["trimmed_video_url"])

    # ---- the additive fields are OFF by default ------------------------

    def test_no_additive_field_reaches_the_dashboard_payload(self):
        state = roster(4)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        for camera_id in C.ALL_CAMERAS:
            data = self._dashboard(state, camera_id)["inspection_data"]
            for key in li.GLOBAL_TOP_LEVEL_FIELDS:
                self.assertNotIn(key, data, f"{camera_id}: {key} must not ship")
            for seg in data["wagon_segments"]:
                for key in li.GLOBAL_SEGMENT_FIELDS:
                    self.assertNotIn(key, seg, f"{camera_id}: {key} must not ship")

    def test_default_config_keeps_the_dashboard_file_legacy_exact(self):
        self.assertFalse(li.LegacyInspectionConfig().emit_global_fields)

    def test_the_internal_payload_keeps_the_global_identity(self):
        """Stripping is for the wire, not for us -- provenance is not lost."""
        state = roster(4)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        full = build_json_for(state, C.CAMERA_RIGHT_UP, self.cache)
        self.assertEqual(full["inspection_data"]["global_wagon_count"], 4)
        self.assertEqual(
            [s["global_wagon_id"] for s in full["inspection_data"]["wagon_segments"]],
            ["GW_1", "GW_2", "GW_3", "GW_4"])

    def test_stripping_does_not_mutate_the_source_payload(self):
        state = roster(3)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        full = build_json_for(state, C.CAMERA_RIGHT_UP, self.cache)
        li.dashboard_payload(full)
        self.assertIn("global_wagon_count", full["inspection_data"])
        self.assertIn("global_wagon_id",
                      full["inspection_data"]["wagon_segments"][0])

    def test_problem_frame_global_id_is_stripped_too(self):
        state = roster(5)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        entries = [{"wagon_id": 3, "wagon_count": 3, "segment_type": "wagon",
                    "problem_type": "damage", "frame_number": 10,
                    "filename": "f.jpg", "s3_key": "k", "s3_url": "u",
                    "is_annotated": False, "annotated_image_url": None,
                    "bounding_box": None}]
        full = build_json_for(state, C.CAMERA_RIGHT_UP, self.cache,
                              damage_rows=[side_damage_row(3, damage=True)],
                              problem_entries=entries)
        self.assertIn("global_wagon_id", full["inspection_data"]["problem_frames"][0])
        dash = li.dashboard_payload(full)
        pf = dash["inspection_data"]["problem_frames"][0]
        self.assertNotIn("global_wagon_id", pf)
        ref_keys = {"wagon_count", "segment_type", "segment_number",
                    "problem_type", "frame_number", "filename", "s3_key",
                    "s3_url", "is_annotated", "annotated_image_url",
                    "bounding_box", "door_status", "door_close_detected",
                    "door_partial_detected", "damage_detected"}
        self.assertEqual(set(pf), ref_keys)

    # ---- renderers still see the identity ------------------------------

    def test_renderers_read_the_internal_file(self):
        state = roster(4)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        out = os.path.join(self.tmp, "out")
        work = os.path.join(out, gb.camera_profile(C.CAMERA_RIGHT_UP).legacy_name)
        os.makedirs(work, exist_ok=True)
        full = build_json_for(state, C.CAMERA_RIGHT_UP, self.cache)
        li._write_json(os.path.join(work, "inspection_data.json"),
                       li.dashboard_payload(full))
        li._write_json(os.path.join(work, li.INTERNAL_JSON_NAME), full)

        payloads = lr.load_camera_payloads(out)
        self.assertIn("global_wagon_count",
                      payloads[C.CAMERA_RIGHT_UP]["inspection_data"])
        rows = lr.combined_wagon_rows(state, payloads)
        self.assertEqual(len(rows), 4)

    def test_renderers_survive_a_dashboard_only_directory(self):
        state = roster(4)
        seed_cache(self.cache, state, [C.CAMERA_RIGHT_UP])
        out = os.path.join(self.tmp, "out")
        work = os.path.join(out, gb.camera_profile(C.CAMERA_RIGHT_UP).legacy_name)
        os.makedirs(work, exist_ok=True)
        li._write_json(os.path.join(work, "inspection_data.json"),
                       li.dashboard_payload(
                           build_json_for(state, C.CAMERA_RIGHT_UP, self.cache)))
        rows = lr.combined_wagon_rows(state, lr.load_camera_payloads(out))
        self.assertEqual(len(rows), 4, "the report must not shrink without the "
                                       "internal file")


# ---------------------------------------------------------------------------
# 21. THE COMBINED WAGON EYE REPORT -- matched to the production document
# ---------------------------------------------------------------------------

class TestCombinedReport(TempCase):
    """Structure, vocabulary and row set of the combined report.

    The reference is the production `Rake_Inspection_Report` PDF: a banner,
    VIDEO EVIDENCE / DETAILED REPORTS / INSPECTION SUMMARY on page 1, paged
    WAGON INSPECTION DETAILS, then a Damaged Wagon Report with evidence panels.
    """

    def _band(self, start=100):
        return {"band_id": 1, "start_frame": start, "end_frame": start + 9,
                "frames": list(range(start, start + 10)), "confidences": [0.9],
                "frame_count": 10, "avg_confidence": 0.9,
                "detection_count": 10, "best_frame": start + 4,
                "best_confidence": 0.9}

    def _write_camera(self, state, camera_id, out, rows=None, load=None):
        profile = gb.camera_profile(camera_id)
        work = os.path.join(out, profile.legacy_name)
        os.makedirs(work, exist_ok=True)
        payload = build_json_for(state, camera_id, self.cache,
                                 damage_rows=rows, load_status_by_wagon=load)
        li._write_json(os.path.join(work, li.INTERNAL_JSON_NAME), payload)
        li._write_json(os.path.join(work, "inspection_data.json"),
                       li.dashboard_payload(payload))
        pd.DataFrame(rows or []).to_csv(
            os.path.join(work, "damage_results.csv"), index=False)
        return work

    # ---- DOOR 1 / DOOR 2 from bands ---------------------------------

    def test_door_cell_vocabulary(self):
        from inspection.combined_report import door_cell_text
        self.assertEqual(door_cell_text([]), "NO DOOR DETECTED")
        self.assertEqual(
            door_cell_text([{"class": "closed_door", "start_frame": 1}]),
            "DOOR 1 CLOSED")
        self.assertEqual(
            door_cell_text([{"class": "open_door", "start_frame": 1}]),
            "DOOR 1 OPEN")
        self.assertEqual(
            door_cell_text([{"class": "partially_closed", "start_frame": 1}]),
            "DOOR 1 PARTIAL CLOSED")
        self.assertEqual(
            door_cell_text([{"class": "closed_door", "start_frame": 1},
                            {"class": "partially_closed", "start_frame": 9}]),
            "DOOR 1 CLOSED / DOOR 2 PARTIAL CLOSED")

    def test_not_visible_is_not_no_door(self):
        from inspection.combined_report import door_cell_text
        self.assertEqual(door_cell_text([], visible=False), "NOT VISIBLE")

    def test_two_bands_become_two_doors(self):
        """Bands are the door instances -- rebuilt from persisted state only."""
        from inspection.combined_report import wagon_report_rows

        state = roster(3)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        out = os.path.join(self.tmp, "out")
        row = side_damage_row(2, door_status="partially_closed", partial=True)
        row["closed_door_band_info"] = [self._band(10)]
        row["partially_closed_band_info"] = [self._band(90)]
        self._write_camera(state, C.CAMERA_LEFT_UP, out, [row])
        for cam in (C.CAMERA_RIGHT_UP, C.CAMERA_RIGHT_UP_TOP, C.CAMERA_LEFT_UP_TOP):
            self._write_camera(state, cam, out, [])
        rows = wagon_report_rows(state, lr.load_camera_payloads(out), out)
        by = {r["sr_no"]: r for r in rows}
        self.assertEqual(by[2]["left_text"],
                         "DOOR 1 CLOSED / DOOR 2 PARTIAL CLOSED")
        self.assertEqual(by[1]["left_text"], "NO DOOR DETECTED")

    # ---- row set is the global roster --------------------------------

    def test_one_row_per_global_wagon(self):
        from inspection.combined_report import wagon_report_rows
        state = roster(57)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        out = os.path.join(self.tmp, "out")
        for cam in C.ALL_CAMERAS:
            self._write_camera(state, cam, out, [])
        rows = wagon_report_rows(state, lr.load_camera_payloads(out), out)
        self.assertEqual(len(rows), 57)
        self.assertEqual([r["sr_no"] for r in rows], list(range(1, 58)))
        self.assertEqual([r["global_wagon_id"] for r in rows],
                         [f"GW_{i}" for i in range(1, 58)])

    def test_missing_camera_never_shortens_the_table(self):
        from inspection.combined_report import wagon_report_rows
        state = roster(20)
        seed_cache(self.cache, state, [C.CAMERA_LEFT_UP])
        out = os.path.join(self.tmp, "out")
        self._write_camera(state, C.CAMERA_LEFT_UP, out, [])
        rows = wagon_report_rows(state, lr.load_camera_payloads(out), out)
        self.assertEqual(len(rows), 20)

    # ---- INSPECTION SUMMARY -----------------------------------------

    def test_summary_fields_and_vocabulary(self):
        from inspection.combined_report import summary_row, wagon_report_rows
        state = roster(10)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        out = os.path.join(self.tmp, "out")
        left = side_damage_row(3, door_status="partially_closed", partial=True)
        left["partially_closed_band_info"] = [self._band()]
        right = side_damage_row(4, door_status="open")
        right["open_door_band_info"] = [self._band()]
        self._write_camera(state, C.CAMERA_LEFT_UP, out, [left])
        self._write_camera(state, C.CAMERA_RIGHT_UP, out, [right])
        self._write_camera(state, C.CAMERA_RIGHT_UP_TOP, out,
                           [top_damage_row(5, floor=True)])
        self._write_camera(state, C.CAMERA_LEFT_UP_TOP, out,
                           [top_damage_row(6, inner=True)])
        rows = wagon_report_rows(state, lr.load_camera_payloads(out), out,
                                 {f"GW_{i}": C.LOAD_LOADED for i in range(1, 11)})
        s = summary_row(rows, TS)
        self.assertEqual(sorted(s), ["date_time", "l_top_damages",
                                     "left_open_doors", "loco_number",
                                     "partial_closed", "r_top_damages",
                                     "rake_type", "right_open_doors", "status",
                                     "total_wagons"])
        self.assertEqual(s["total_wagons"], 10)
        self.assertEqual(s["left_open_doors"], 0)
        self.assertEqual(s["right_open_doors"], 1)
        self.assertEqual(s["r_top_damages"], 1)
        self.assertEqual(s["l_top_damages"], 1)
        self.assertEqual(s["partial_closed"], "L 1 / R 0")
        self.assertEqual(s["rake_type"], "LOADED RAKE")
        self.assertEqual(s["status"], "NOT OK")
        self.assertEqual(s["loco_number"], "-", "OCR is disabled")

    def test_clean_rake_is_ok(self):
        from inspection.combined_report import summary_row, wagon_report_rows
        state = roster(6)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        out = os.path.join(self.tmp, "out")
        for cam in C.ALL_CAMERAS:
            self._write_camera(state, cam, out, [])
        rows = wagon_report_rows(state, lr.load_camera_payloads(out), out)
        self.assertEqual(summary_row(rows, TS)["status"], "OK")

    def test_total_wagons_is_the_roster_not_a_camera(self):
        from inspection.combined_report import summary_row, wagon_report_rows
        state = roster(31)
        seed_cache(self.cache, state, [C.CAMERA_LEFT_UP], only_indices={2, 3})
        out = os.path.join(self.tmp, "out")
        self._write_camera(state, C.CAMERA_LEFT_UP, out, [])
        rows = wagon_report_rows(state, lr.load_camera_payloads(out), out)
        self.assertEqual(summary_row(rows, TS)["total_wagons"], 31)

    # ---- WAGON TYPE ---------------------------------------------------

    def test_wagon_type_vocabulary(self):
        from inspection.combined_report import wagon_report_rows
        state = roster(3)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        out = os.path.join(self.tmp, "out")
        load = {"GW_1": C.LOAD_LOADED, "GW_2": C.LOAD_EMPTY}
        for cam in C.ALL_CAMERAS:
            self._write_camera(state, cam, out, [], load)
        rows = {r["sr_no"]: r for r in
                wagon_report_rows(state, lr.load_camera_payloads(out), out, load)}
        self.assertEqual(rows[1]["wagon_type"], "LOADED")
        self.assertEqual(rows[2]["wagon_type"], "EMPTY")
        self.assertEqual(rows[3]["wagon_type"], "-",
                         "an unknown load must not be guessed as EMPTY")

    def test_wagon_number_is_dash_because_ocr_is_off(self):
        from inspection.combined_report import wagon_report_rows
        state = roster(4)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        out = os.path.join(self.tmp, "out")
        for cam in C.ALL_CAMERAS:
            self._write_camera(state, cam, out, [])
        for row in wagon_report_rows(state, lr.load_camera_payloads(out), out):
            self.assertEqual(row["wagon_number"], "-")

    # ---- Damaged Wagon Report ----------------------------------------

    def test_damaged_entries_match_the_reference_shape(self):
        from inspection.combined_report import (damaged_wagon_entries,
                                                wagon_report_rows)
        state = roster(12)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        out = os.path.join(self.tmp, "out")
        right = side_damage_row(9, door_status="open")
        right["open_door_band_info"] = [self._band()]
        self._write_camera(state, C.CAMERA_LEFT_UP, out, [])
        self._write_camera(state, C.CAMERA_RIGHT_UP, out, [right])
        self._write_camera(state, C.CAMERA_RIGHT_UP_TOP, out,
                           [top_damage_row(9, floor=True)])
        self._write_camera(state, C.CAMERA_LEFT_UP_TOP, out,
                           [top_damage_row(9, inner=True)])
        rows = wagon_report_rows(state, lr.load_camera_payloads(out), out)
        entries = damaged_wagon_entries(rows, out, TS)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["wagon_id"], 9)
        self.assertEqual(entry["global_wagon_id"], "GW_9")
        self.assertEqual(entry["wagon_number"], "-")
        self.assertEqual(entry["angles"], "Left-Top, Right, Right-Top",
                         "angles read alphabetically, as in the reference")
        self.assertIn("IST", entry["date_time"])

    def test_clean_rake_has_no_damaged_section(self):
        from inspection.combined_report import (damaged_wagon_entries,
                                                wagon_report_rows)
        state = roster(5)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        out = os.path.join(self.tmp, "out")
        for cam in C.ALL_CAMERAS:
            self._write_camera(state, cam, out, [])
        rows = wagon_report_rows(state, lr.load_camera_payloads(out), out)
        self.assertEqual(damaged_wagon_entries(rows, out, TS), [])

    def test_evidence_images_are_captioned_per_camera_and_problem(self):
        from inspection.combined_report import (damaged_wagon_entries,
                                                wagon_report_rows)
        import cv2
        import numpy as np

        state = roster(6)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        out = os.path.join(self.tmp, "out")
        right = side_damage_row(4, door_status="open")
        right["open_door_band_info"] = [self._band()]
        self._write_camera(state, C.CAMERA_RIGHT_UP, out, [right])
        work = self._write_camera(state, C.CAMERA_LEFT_UP_TOP, out,
                                  [top_damage_row(4, inner=True)])
        self._write_camera(state, C.CAMERA_LEFT_UP, out, [])
        self._write_camera(state, C.CAMERA_RIGHT_UP_TOP, out, [])
        img = os.path.join(self.tmp, "ev.jpg")
        cv2.imwrite(img, np.zeros((40, 60, 3), dtype=np.uint8))
        pd.DataFrame([{"wagon_id": 4, "problem_type": "inner_wall_dmg",
                       "frame_number": 5, "frame_path": img,
                       "annotated_image_path": img, "bounding_box": []}]
                     ).to_csv(os.path.join(work, "problem_frames.csv"),
                              index=False)
        rows = wagon_report_rows(state, lr.load_camera_payloads(out), out)
        entry = damaged_wagon_entries(rows, out, TS)[0]
        self.assertEqual([i["caption"] for i in entry["images"]],
                         ["Left-Top Camera – Damage"])

    # ---- the document -------------------------------------------------

    def test_report_renders_with_the_expected_sections(self):
        from pypdf import PdfReader

        from inspection.combined_report import build_combined_report_pdf

        state = roster(30)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        out = os.path.join(self.tmp, "out")
        right = side_damage_row(7, door_status="open")
        right["open_door_band_info"] = [self._band()]
        self._write_camera(state, C.CAMERA_RIGHT_UP, out, [right])
        self._write_camera(state, C.CAMERA_LEFT_UP, out, [])
        self._write_camera(state, C.CAMERA_RIGHT_UP_TOP, out, [])
        self._write_camera(state, C.CAMERA_LEFT_UP_TOP, out, [])
        path = os.path.join(self.tmp, "combined.pdf")
        build_combined_report_pdf(
            state=state, output_root=out,
            payloads=lr.load_camera_payloads(out), output_path=path,
            source_video_urls={c: "https://e/r.mp4" for c in C.ALL_CAMERAS},
            processed_video_urls={c: "https://e/p.mp4" for c in C.ALL_CAMERAS},
            camera_report_urls={c: "https://e/x.pdf" for c in C.ALL_CAMERAS},
            when=TS)
        self.assertTrue(os.path.getsize(path) > 0)
        text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
        for heading in ("COMBINED WAGON EYE REPORT", "VIDEO EVIDENCE",
                        "DETAILED REPORTS", "INSPECTION SUMMARY",
                        "WAGON INSPECTION DETAILS", "Damaged Wagon Report",
                        "Total Damaged Wagons: 1"):
            self.assertIn(heading, text, f"missing section: {heading}")
        for column in ("SR.NO", "WAGON NUMBER", "LEFT CAMERA", "RIGHT CAMERA",
                       "R-TOP", "L-TOP", "WAGON", "Click to View",
                       "LEFT Detail Report", "L-TOP Detail Report"):
            self.assertIn(column, text, f"missing column/label: {column}")

    def test_every_global_wagon_appears_in_the_rendered_table(self):
        from pypdf import PdfReader

        from inspection.combined_report import build_combined_report_pdf

        state = roster(45)
        seed_cache(self.cache, state, C.ALL_CAMERAS)
        out = os.path.join(self.tmp, "out")
        for cam in C.ALL_CAMERAS:
            self._write_camera(state, cam, out, [])
        path = os.path.join(self.tmp, "c.pdf")
        build_combined_report_pdf(state=state, output_root=out,
                                  payloads=lr.load_camera_payloads(out),
                                  output_path=path, when=TS)
        reader = PdfReader(path)
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        # SR.NO appears once per wagon; the table header repeats per page.
        self.assertEqual(text.count("NO DOOR DETECTED"), 45 * 2,
                         "every wagon must have a left and a right door cell")
        self.assertGreaterEqual(reader.pages.__len__(), 4)

    def test_renderer_runs_no_model(self):
        """Same guarantee as legacy_render: layout consumes persisted state."""
        import ast

        from inspection import combined_report as cr
        tree = ast.parse(open(cr.__file__, encoding="utf-8").read())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                self.assertNotIn(name.split(".")[0],
                                 {"ultralytics", "torch", "torchvision"},
                                 f"combined_report imports {name}")
            if isinstance(node, ast.Call):
                func = node.func
                called = (func.id if isinstance(func, ast.Name)
                          else func.attr if isinstance(func, ast.Attribute)
                          else "")
                self.assertNotIn(called, {"YOLO", "DamageDetector", "predict",
                                          "ProblemFrameExtractor"})


def _publish_and_build(state, camera_id, cache_root, tmp):
    """Build a payload with real published evidence frames."""
    profile = gb.camera_profile(camera_id)
    windows = windows_for(state, camera_id)
    full_df = gb.build_segment_summary(
        state, camera_id, windows, cache_root,
        require_frames=False, include_unwindowed=True)
    count_map = gb.build_wagon_count_map(full_df)
    sink = li.LocalArtifactSink(os.path.join(tmp, "artifacts"))
    publisher = ArtifactPublisher(
        s3=sink, artifact_bucket="bucket", region="ap-south-1",
        camera_folder=profile.folder, damage_flavour=profile.flavour)
    _ts, index, loco, problems = publisher.publish(
        upload_timestamp=TS, segment_summary_df=full_df,
        loco_summary_df=pd.DataFrame(), problem_frames_df=pd.DataFrame(),
        wagon_count_map=count_map, local_workdir=tmp)
    return build_inspection_json(
        camera_folder=profile.folder, raw_video_name="clip",
        upload_timestamp=TS, direction=profile.loaded_direction,
        flavour=profile.flavour, segment_summary_df=full_df,
        damage_results_df=li._empty_damage_df(profile.flavour),
        loco_summary_df=pd.DataFrame(), problem_frames_df=pd.DataFrame(),
        wagon_frames_index=index, loco_frame_entries=loco,
        problem_frame_entries=problems, wagon_count_map=count_map,
        segment_type_map=gb.build_segment_type_map(full_df, profile.flavour),
        wagon_number_results=None, loco_numbers=None, damage_model_active=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
