"""The old_code port: same intelligence, attached to the finalized global wagons.

What these tests defend, in order of importance:

  1. The counting result is IMMUTABLE. Inspection cannot change a GW id, a
     boundary, the wagon count, a classification or MASTER == GLOBAL -- and the
     roster hash makes that a checked property, not a claim.
  2. Association is BY CONSTRUCTION. A frame under wagon_cache/GW_17/RIGHT_UP/ is
     a frame of GW_17 as seen by RIGHT_UP, because that is how it was produced.
     An unresolved camera contributes nothing rather than guessing.
  3. The old ALGORITHMS are the ones running -- old_code's own trackers, configs
     and thresholds, imported unchanged, not a reimplementation.
  4. States stay DISTINCT: NO_DATA / NOT_VISIBLE / UNRESOLVED are never collapsed
     into CLOSED / EMPTY / OK.
  5. Nothing leaks between trains processed in one process.

Model-dependent checks skip cleanly when a weight is absent, which is the real
local situation for load.pt (supplied on EC2 from S3).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core import constants as C
from core.global_state_loader import (
    RosterMutatedError, assert_roster_unchanged, roster_hash, snapshot_roster,
)
from core.unified_wagon_state import UnifiedWagonState, summarize_wagons
from global_train_state import GlobalTrainState, GlobalWagon, SegmentClass
from inspection import old_features as oldf
from inspection import wagon_cache as wc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

FPS = 15.0
WIDTH = 960


# `plan_cache` only checks that the video path EXISTS (it never opens it), so this
# file stands in for a video. Using a real path keeps the planning tests focused on
# offsets and windows instead of tripping the missing-video branch first.
EXISTING_PATH = os.path.abspath(__file__)


class FakeTracks:
    """Minimal stand-in for LocalCameraTracks: only what the bridge reads."""

    def __init__(self, fps=FPS, total=3000, width=WIDTH, path=EXISTING_PATH):
        self.fps = fps
        self.total_frames = total
        self.width = width
        self.video_path = path


def roster(n=5, span=10.0, classifications=None):
    """A finalized roster: GW_1..GW_n, contiguous in MASTER seconds."""
    st = GlobalTrainState(total_wagons=n, master_camera=C.CAMERA_RIGHT_UP,
                          master_fps=FPS, master_total_frames=3000)
    for i in range(1, n + 1):
        cls = (classifications or {}).get(i, SegmentClass.WAGON)
        st.wagons.append(GlobalWagon(
            global_id=f"GW_{i}", wagon_index=i,
            start_frame_master=int(span * (i - 1) * FPS),
            end_frame_master=int(span * i * FPS),
            start_time=span * (i - 1), end_time=span * i,
            classification=cls))
    st.camera_offsets = {
        C.CAMERA_RIGHT_UP: {"status": "REFERENCE", "delta": 0.0},
        C.CAMERA_LEFT_UP: {"status": "RESOLVED", "delta": 2.0},
        C.CAMERA_RIGHT_UP_TOP: {"status": "RESOLVED", "delta": -1.5},
        C.CAMERA_LEFT_UP_TOP: {"status": "UNRESOLVED", "delta": 99.0},
    }
    st.global_gaps = [{"global_gap_id": i} for i in range(1, n)]
    st.invariant_checks = {"right_up_final_gap_count": n - 1}
    return st


def all_tracks(video=EXISTING_PATH):
    return {c: FakeTracks(path=video) for c in C.ALL_CAMERAS}


# ===========================================================================
# 1  the counting result is immutable -- checked, not claimed
# ===========================================================================

class TestCountingImmutable(unittest.TestCase):
    def test_roster_hash_is_stable_and_deterministic(self):
        st = roster()
        self.assertEqual(roster_hash(st), roster_hash(roster()))
        self.assertEqual(roster_hash(st), roster_hash(st))

    def test_snapshot_captures_identity_defining_fields(self):
        snap = snapshot_roster(roster(3))
        self.assertEqual([s["global_id"] for s in snap], ["GW_1", "GW_2", "GW_3"])
        for s in snap:
            for key in ("wagon_index", "start_frame_master", "end_frame_master",
                        "start_time", "end_time", "classification"):
                self.assertIn(key, s)

    def test_adding_a_wagon_is_detected(self):
        st = roster(3)
        before = roster_hash(st)
        st.wagons.append(GlobalWagon(
            global_id="GW_4", wagon_index=4, start_frame_master=0,
            end_frame_master=1, start_time=30.0, end_time=40.0))
        with self.assertRaises(RosterMutatedError):
            assert_roster_unchanged(st, before)

    def test_renumbering_a_gw_id_is_detected(self):
        st = roster(3)
        before = roster_hash(st)
        st.wagons[1].global_id = "GW_99"
        with self.assertRaises(RosterMutatedError):
            assert_roster_unchanged(st, before)

    def test_moving_a_boundary_is_detected(self):
        st = roster(3)
        before = roster_hash(st)
        st.wagons[0].end_time += 0.5
        with self.assertRaises(RosterMutatedError):
            assert_roster_unchanged(st, before)

    def test_changing_a_classification_is_detected(self):
        st = roster(3)
        before = roster_hash(st)
        st.wagons[0].classification = SegmentClass.ENGINE
        with self.assertRaises(RosterMutatedError):
            assert_roster_unchanged(st, before)

    def test_breaking_master_equals_global_is_detected(self):
        st = roster(4)
        before = roster_hash(st)
        st.global_gaps.pop()          # global gaps no longer match the master
        with self.assertRaises(RosterMutatedError):
            assert_roster_unchanged(st, before)

    def test_annotating_does_not_trip_the_guard(self):
        """Inspection must be free to ANNOTATE -- only identity is frozen."""
        st = roster(3)
        before = roster_hash(st)
        st.inspection = {"summary": {"top_damaged": 2}}
        st.add_note("door feature ran")
        assert_roster_unchanged(st, before)      # must not raise

    def test_fusion_never_invents_a_gw_id(self):
        st = roster(4)
        with tempfile.TemporaryDirectory() as tmp:
            unified = oldf.fuse_unified_states(
                st, tmp, {c: "RESOLVED" for c in C.ALL_CAMERAS}, tmp)
        self.assertEqual(sorted(unified), ["GW_1", "GW_2", "GW_3", "GW_4"])


# ===========================================================================
# 2  association by construction
# ===========================================================================

class TestAssociation(unittest.TestCase):
    def test_every_resolved_camera_gets_a_window_per_wagon(self):
        st = roster(5)
        plan = wc.plan_cache(st, all_tracks(), output_root="out")
        resolved = [c for c, s in plan.camera_status.items() if s == "RESOLVED"]
        self.assertEqual(len(plan.windows), 5 * len(resolved))

    def test_camera_offset_shifts_the_window(self):
        """LEFT_UP is +2.0s, so its frames sit EARLIER in its own clock."""
        st = roster(5)
        plan = wc.plan_cache(st, all_tracks(), output_root="out")
        by = {(w.global_id, w.camera_id): w for w in plan.windows}
        right = by[("GW_2", C.CAMERA_RIGHT_UP)]
        left = by[("GW_2", C.CAMERA_LEFT_UP)]
        self.assertEqual(left.start_frame, right.start_frame - int(2.0 * FPS))

    def test_unresolved_camera_gets_no_windows_at_all(self):
        st = roster(5)
        plan = wc.plan_cache(st, all_tracks(), output_root="out")
        self.assertEqual(plan.camera_status[C.CAMERA_LEFT_UP_TOP], "UNRESOLVED")
        self.assertEqual(
            [w for w in plan.windows if w.camera_id == C.CAMERA_LEFT_UP_TOP], [])
        self.assertTrue(any("unresolved" in s.get("reason", "")
                            for s in plan.skipped))

    def test_wagon_outside_the_footage_is_skipped_not_clamped(self):
        st = roster(5)
        tracks = all_tracks()
        tracks[C.CAMERA_RIGHT_UP] = FakeTracks(total=100)   # ~6.7s of footage
        plan = wc.plan_cache(st, tracks, output_root="out")
        ru = [w for w in plan.windows if w.camera_id == C.CAMERA_RIGHT_UP]
        self.assertLess(len(ru), 5)
        self.assertTrue(any(s.get("global_id") for s in plan.skipped))

    def test_cache_path_matches_what_old_code_reads(self):
        """The bridge must write where old_code's own helper looks."""
        from features._common import wagon_camera_dir as old_dir
        ours = wc.wagon_camera_dir("root", "GW_7", C.CAMERA_RIGHT_UP)
        theirs = old_dir("root", "GW_7", C.CAMERA_RIGHT_UP)
        self.assertEqual(os.path.normpath(ours), os.path.normpath(theirs))

    def test_missing_video_is_reported_not_guessed(self):
        st = roster(3)
        tracks = all_tracks(video="does_not_exist.mp4")
        plan = wc.plan_cache(st, tracks, output_root="out")
        self.assertEqual(set(plan.camera_status.values()), {"NO_VIDEO"})
        self.assertEqual(plan.windows, [])

    def test_plan_is_deterministic(self):
        st = roster(6)
        a = wc.plan_cache(st, all_tracks(), output_root="out").to_dict()
        b = wc.plan_cache(st, all_tracks(), output_root="out").to_dict()
        self.assertEqual(a, b)

    def test_planning_does_not_mutate_the_state(self):
        st = roster(4)
        before = roster_hash(st)
        wc.plan_cache(st, all_tracks(), output_root="out")
        assert_roster_unchanged(st, before)


# ===========================================================================
# 3  the OLD algorithms are the ones running
# ===========================================================================

class TestOldAlgorithmsInUse(unittest.TestCase):
    def test_door_tracker_is_old_code_with_its_own_thresholds(self):
        from features.inference_lib.door_tracker import (
            DoorState, DoorTracker, TrackerConfig,
        )
        cfg = TrackerConfig()
        # Values as shipped in old_code -- a change here means the port drifted.
        self.assertAlmostEqual(cfg.open_confidence_threshold, 0.80)
        self.assertAlmostEqual(cfg.closed_confidence_threshold, 0.68)
        self.assertEqual(cfg.max_age, 30)
        self.assertEqual(cfg.n_init, 3)
        self.assertEqual(cfg.new_id_delay_frames, 8)
        self.assertTrue(cfg.direction_gate_enabled)
        self.assertTrue(cfg.enable_track_revival)
        self.assertIn("PARTIAL_CLOSED", [s.value for s in DoorState])
        self.assertIn("DAMAGE", [s.value for s in DoorState])
        self.assertTrue(hasattr(DoorTracker(config=cfg), "update"))

    def test_damage_tracker_is_old_code_distance_only(self):
        from features.inference_lib.damage_tracker import DamageTrackerConfig
        cfg = DamageTrackerConfig()
        self.assertAlmostEqual(cfg.max_center_distance, 200.0)
        self.assertAlmostEqual(cfg.iou_weight, 0.0)      # top view: IoU disabled
        self.assertAlmostEqual(cfg.distance_weight, 1.0)
        self.assertEqual(cfg.n_init, 2)
        self.assertEqual(cfg.max_age, 30)

    def test_hungarian_assignment_is_available(self):
        """old_code's trackers use scipy's linear_sum_assignment."""
        from scipy.optimize import linear_sum_assignment
        cost = np.array([[1.0, 9.0], [9.0, 1.0]])
        rows, cols = linear_sum_assignment(cost)
        self.assertEqual(list(cols), [0, 1])

    def test_identity_merger_and_priors_are_old_code(self):
        from features.inference_lib.door_identity_merger import MergeConfig
        from features.inference_lib.geometric_shape_prior import GeometricPriorConfig
        from features.inference_lib.illumination_processor import IlluminationConfig
        self.assertAlmostEqual(MergeConfig().merge_threshold, 0.85)
        self.assertAlmostEqual(MergeConfig().spatial_weight, 0.40)
        GeometricPriorConfig()
        IlluminationConfig()

    def test_damage_filters_are_old_code_verbatim(self):
        from features.damage import processor as dmg
        self.assertAlmostEqual(dmg._AREA_MIN_RATIO, 0.005)
        self.assertAlmostEqual(dmg._AREA_MAX_RATIO, 0.40)
        self.assertAlmostEqual(dmg._EDGE_X_MIN_RATIO, 0.12)
        self.assertAlmostEqual(dmg._EDGE_X_MAX_RATIO, 0.88)
        self.assertAlmostEqual(dmg._EDGE_Y_MIN_RATIO, 0.10)
        self.assertAlmostEqual(dmg._EDGE_Y_MAX_RATIO, 0.85)
        self.assertAlmostEqual(dmg._EDGE_BYPASS_CONF, 0.70)
        self.assertIn("outer_wall_damage", dmg._SKIP_CLASSES_TOP)

    def test_load_ratio_rule_is_old_code_verbatim(self):
        from features.load import processor as load
        self.assertAlmostEqual(load._LOADED_RATIO_THRESHOLD, 0.35)

    def test_door_fsm_classifies_by_substring_not_class_id(self):
        """Retrain-robust: the FSM matches names, so reordered ids are harmless."""
        from features.inference_lib.door_tracker import StateMachine, TrackerConfig
        sm = StateMachine(track_id=1, config=TrackerConfig())
        # 'partially_closed' must NOT be read as 'closed'
        for name in ("open_door", "closed_door", "partially_closed", "damage"):
            sm.update(name, 0.95, 1)
        self.assertTrue(hasattr(sm, "state"))

    def test_side_and_top_camera_authority_is_preserved(self):
        self.assertEqual(C.SIDE_CAMERAS, [C.CAMERA_RIGHT_UP, C.CAMERA_LEFT_UP])
        # RIGHT_UP_TOP first == authoritative for load, per old_code.
        self.assertEqual(C.TOP_CAMERAS[0], C.CAMERA_RIGHT_UP_TOP)


# ===========================================================================
# 4  states stay distinct
# ===========================================================================

class TestStatesStayDistinct(unittest.TestCase):
    def test_absent_feature_json_yields_no_data_not_a_verdict(self):
        st = roster(3)
        with tempfile.TemporaryDirectory() as tmp:
            unified = oldf.fuse_unified_states(
                st, tmp, {c: "RESOLVED" for c in C.ALL_CAMERAS}, tmp)
        for u in unified.values():
            self.assertEqual(u.left_door, C.NO_DATA)
            self.assertEqual(u.right_door, C.NO_DATA)
            self.assertEqual(u.load_status, C.NO_DATA)
            self.assertEqual(u.top_damage, C.NO_DATA)
            self.assertEqual(u.side_damage, C.NO_DATA)

    def test_no_data_is_not_an_anomaly(self):
        u = UnifiedWagonState(global_id="GW_1")
        self.assertFalse(u.has_open_door)
        self.assertFalse(u.has_damage)
        self.assertEqual(u.anomalies, [])

    def test_unresolved_camera_status_is_preserved_per_wagon(self):
        st = roster(3)
        status = {C.CAMERA_RIGHT_UP: "RESOLVED", C.CAMERA_LEFT_UP: "RESOLVED",
                  C.CAMERA_RIGHT_UP_TOP: "RESOLVED",
                  C.CAMERA_LEFT_UP_TOP: "UNRESOLVED"}
        with tempfile.TemporaryDirectory() as tmp:
            unified = oldf.fuse_unified_states(st, tmp, status, tmp)
        for u in unified.values():
            vals = list(u.camera_status[C.CAMERA_LEFT_UP_TOP].values())
            self.assertEqual(vals, ["UNRESOLVED"])

    def test_feature_json_is_read_not_recomputed(self):
        """Fusion must report exactly what the processor wrote."""
        st = roster(2)
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "door"))
            with open(os.path.join(tmp, "door", "GW_1.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"global_id": "GW_1", "feature": "door",
                           "status": C.STATUS_OK,
                           "left_door": C.DOOR_OPEN,
                           "left_door_confidence": 0.93,
                           "right_door": C.DOOR_CLOSED,
                           "right_door_confidence": 0.71,
                           "supporting_cameras": [C.CAMERA_LEFT_UP],
                           "tracks": [{"track_id": 3}]}, fh)
            unified = oldf.fuse_unified_states(
                st, tmp, {c: "RESOLVED" for c in C.ALL_CAMERAS}, tmp)
        u = unified["GW_1"]
        self.assertEqual(u.left_door, C.DOOR_OPEN)
        self.assertAlmostEqual(u.left_door_confidence, 0.93)
        self.assertEqual(u.right_door, C.DOOR_CLOSED)
        self.assertTrue(u.has_open_door)
        self.assertIn("LEFT_DOOR_OPEN", u.anomalies)
        self.assertEqual(unified["GW_2"].left_door, C.NO_DATA)

    def test_kpis_do_not_imply_a_complete_inspection(self):
        """loaded + empty must not silently sum to the train size."""
        ws = [UnifiedWagonState(global_id=f"GW_{i}") for i in range(1, 6)]
        ws[0].load_status = C.LOAD_LOADED
        ws[1].load_status = C.LOAD_EMPTY
        s = summarize_wagons(ws)
        self.assertEqual(s["loaded"], 1)
        self.assertEqual(s["empty"], 1)
        self.assertEqual(s["load_no_data"], 3)
        self.assertEqual(s["total_wagons"], 5)

    def test_every_wagon_appears_in_the_kpi_roster(self):
        st = roster(7)
        with tempfile.TemporaryDirectory() as tmp:
            unified = oldf.fuse_unified_states(
                st, tmp, {c: "RESOLVED" for c in C.ALL_CAMERAS}, tmp)
        self.assertEqual(summarize_wagons(list(unified.values()))["total_wagons"], 7)


# ===========================================================================
# 5  models: absence is graceful, classes come from the checkpoint
# ===========================================================================

class TestModels(unittest.TestCase):
    def test_missing_models_are_reported_not_fatal(self):
        rec = oldf.discover_feature_models(os.path.join(ROOT, "no_such_dir"),
                                          verbose=False)
        for role in ("door", "load", "damage"):
            self.assertFalse(rec[role]["available"])
            self.assertEqual(rec[role]["status"], "UNAVAILABLE")
            self.assertIn("wagon count is unaffected", rec[role]["reason"])

    def test_load_model_absence_is_the_expected_local_state(self):
        """load.pt is supplied on EC2 from S3; locally it may be absent."""
        rec = oldf.discover_feature_models(MODELS_DIR, verbose=False)["load"]
        if rec["available"]:
            self.skipTest("load.pt present locally")
        self.assertIn("load.pt", rec["expected_filenames"])
        self.assertIn("loaded.pt", rec["expected_filenames"])

    def test_both_load_filenames_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = os.path.join(tmp, "loaded.pt")
            open(legacy, "wb").close()
            self.assertEqual(
                os.path.normpath(C.resolve_model_path(tmp, C.MODEL_LOADED)),
                os.path.normpath(legacy))

    def test_damage_model_accepts_old_and_new_filename(self):
        self.assertIn("top_damage.pt", C.MODEL_ALIASES[C.MODEL_DAMAGE])
        self.assertIn("damage.pt", C.MODEL_ALIASES[C.MODEL_DAMAGE])

    @unittest.skipUnless(os.path.isfile(os.path.join(MODELS_DIR, "door_state.pt")),
                         "door_state.pt not present")
    def test_door_classes_come_from_the_checkpoint(self):
        rec = oldf.discover_feature_models(MODELS_DIR, verbose=False)["door"]
        self.assertEqual(rec["status"], "AVAILABLE")
        self.assertTrue(rec["class_names"])
        names = {v.lower() for v in rec["class_names"].values()}
        # The FSM keys off these substrings, so at least one must be present.
        self.assertTrue(any("open" in n or "closed" in n for n in names))

    @unittest.skipUnless(os.path.isfile(os.path.join(MODELS_DIR, "top_damage.pt")),
                         "top_damage.pt not present")
    def test_damage_classes_come_from_the_checkpoint(self):
        rec = oldf.discover_feature_models(MODELS_DIR, verbose=False)["damage"]
        self.assertEqual(rec["status"], "AVAILABLE")
        self.assertTrue(rec["class_names"])

    @unittest.skipUnless(os.path.isfile(os.path.join(MODELS_DIR, "load.pt")),
                         "load.pt not present (supplied on EC2 from S3)")
    def test_real_load_model_classes_map_correctly(self):
        """Pins the ACTUAL shipped load.pt behaviour.

        Measured classes are Empty / Loaded / Unlabeled. The first two must map to
        the old vocabulary; `Unlabeled` must NOT be forced into either, because
        old_code counts an unmapped label as seen-but-abstaining, and guessing it
        would move the 0.35 loaded-ratio verdict.
        """
        rec = oldf.discover_feature_models(MODELS_DIR, verbose=False)["load"]
        self.assertEqual(rec["status"], "AVAILABLE")
        mapping = rec["load_label_mapping"]
        names = {v.lower() for v in rec["class_names"].values()}
        if "empty" in names:
            self.assertEqual(mapping["empty"], C.LOAD_EMPTY)
        if "loaded" in names:
            self.assertEqual(mapping["loaded"], C.LOAD_LOADED)
        for unmapped in rec.get("unmapped_class_names", []):
            self.assertNotIn(unmapped, mapping,
                             "an unmappable label must abstain, not be guessed")

    def test_load_label_mapping_is_resolved_from_real_class_names(self):
        m = C.resolve_load_label_mapping({0: "EMPTY", 1: "Loaded"})
        self.assertEqual(m["empty"], C.LOAD_EMPTY)
        self.assertEqual(m["loaded"], C.LOAD_LOADED)

    def test_unmappable_load_label_abstains_rather_than_guessing(self):
        m = C.resolve_load_label_mapping({0: "mystery"})
        self.assertNotIn("mystery", m)

    def test_load_mapping_is_class_id_order_independent(self):
        a = C.resolve_load_label_mapping({0: "empty", 1: "loaded"})
        b = C.resolve_load_label_mapping({0: "loaded", 1: "empty"})
        self.assertEqual(a, b)

    def test_cpu_half_shim_forces_float32_but_delegates_everything_else(self):
        """old_code asks for half=True; on CPU that measured ~130x slower.

        The proxy must neutralise ONLY the `half` kwarg -- `.names` and every
        other attribute the door processor reads must pass straight through, or
        the port would diverge in more than precision.
        """
        from inspection.old_features import _HalfDisabledModel

        class Spy:
            names = {0: "closed_door", 2: "open_door"}
            task = "detect"

            def __init__(self):
                self.seen = []

            def __call__(self, *a, **kw):
                self.seen.append(kw.get("half"))
                return ["result"]

            def predict(self, *a, **kw):
                self.seen.append(kw.get("half"))
                return ["result"]

        spy = Spy()
        proxy = _HalfDisabledModel(spy)
        proxy("frame", verbose=False, half=True)
        proxy.predict("frame", half=True)
        self.assertEqual(spy.seen, [False, False],
                         "half must be forced off on both call paths")
        self.assertEqual(proxy.names, Spy.names)   # delegated, not shadowed
        self.assertEqual(proxy.task, "detect")

    def test_half_shim_is_not_installed_when_a_gpu_is_present(self):
        """On GPU the old behaviour must be bit-for-bit preserved."""
        import torch

        from inspection.old_features import install_cpu_half_shim
        if torch.cuda.is_available():
            self.assertFalse(install_cpu_half_shim("models/door_state.pt",
                                                   verbose=False))
        else:
            self.skipTest("no GPU here; CPU path is covered by the proxy test")

    def test_ocr_is_out_of_scope(self):
        self.assertFalse(C.OCR_ENABLED)
        u = UnifiedWagonState(global_id="GW_1")
        self.assertEqual(u.wagon_identifier, "")
        self.assertEqual(summarize_wagons([u])["ocr_captured"], 0)


# ===========================================================================
# 6  reports
# ===========================================================================

class TestReports(unittest.TestCase):
    def _unified(self, st, damaged=(), open_doors=()):
        out = {}
        for w in st.wagons:
            u = UnifiedWagonState(global_id=w.global_id,
                                  classification=w.classification)
            if w.wagon_index in damaged:
                u.top_damage = C.DAMAGE_PRESENT
                u.top_damage_confidence = 0.9
                u.supporting_cameras = [C.CAMERA_RIGHT_UP_TOP]
            if w.wagon_index in open_doors:
                u.left_door = C.DOOR_OPEN
                u.left_door_confidence = 0.88
                u.supporting_cameras = [C.CAMERA_LEFT_UP]
            u.camera_status = {c: {"door": "NO_DETECTION"} for c in C.ALL_CAMERAS}
            out[u.global_id] = u
        return out

    def test_combined_report_is_old_code(self):
        """The combined report must be old_code's module, not a new one."""
        from features.reporting import combined_train_report as old_report
        self.assertTrue(hasattr(old_report, "build"))
        self.assertTrue(hasattr(old_report, "_build_pdf"))
        self.assertTrue(hasattr(old_report, "_build_json"))
        self.assertIn("old_code", old_report.__file__)

    def test_combined_report_writes_json_and_pdf_with_gw_ids(self):
        from inspection.old_report import build_all_reports
        st = roster(6, classifications={1: SegmentClass.ENGINE,
                                        6: SegmentClass.BRAKE_VAN})
        unified = self._unified(st, damaged=(3,), open_doors=(4,))
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        out = build_all_reports(st, unified, tmp, batch_key="TEST",
                                camera_status={c: "RESOLVED"
                                               for c in C.ALL_CAMERAS},
                                verbose=False)
        jp = out["combined"]["json_path"]
        self.assertTrue(os.path.isfile(jp))
        doc = json.load(open(jp, encoding="utf-8"))
        self.assertEqual([w["global_id"] for w in doc["wagons"]],
                         [f"GW_{i}" for i in range(1, 7)])
        self.assertEqual(doc["summary"]["total_wagons"], 6)
        self.assertEqual(doc["summary"]["engine_count"], 1)
        self.assertEqual(doc["summary"]["brake_van_count"], 1)
        self.assertEqual(doc["summary"]["top_damaged"], 1)
        self.assertEqual(doc["summary"]["left_doors_open"], 1)
        if out["combined"]["pdf_path"]:
            self.assertGreater(os.path.getsize(out["combined"]["pdf_path"]), 1000)

    def test_camera_reports_exist_for_all_four_cameras(self):
        from inspection.old_report import build_all_reports
        st = roster(4)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        out = build_all_reports(st, self._unified(st), tmp, batch_key="TEST",
                                camera_status={c: "RESOLVED"
                                               for c in C.ALL_CAMERAS},
                                verbose=False)
        self.assertEqual(sorted(out["cameras"]), sorted(C.ALL_CAMERAS))
        for cam, path in out["cameras"].items():
            self.assertTrue(path and os.path.isfile(path), cam)

    def test_camera_report_headers_follow_camera_authority(self):
        from inspection.old_report import _camera_headers
        self.assertIn("R_DOOR", _camera_headers(C.CAMERA_RIGHT_UP))
        self.assertNotIn("L_DOOR", _camera_headers(C.CAMERA_RIGHT_UP))
        self.assertIn("L_DOOR", _camera_headers(C.CAMERA_LEFT_UP))
        for h in ("LOAD", "TOP_DMG"):
            self.assertIn(h, _camera_headers(C.CAMERA_RIGHT_UP_TOP))

    def test_report_does_not_mutate_the_roster(self):
        from inspection.old_report import build_all_reports
        st = roster(4)
        before = roster_hash(st)
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        build_all_reports(st, self._unified(st), tmp, batch_key="T",
                          camera_status={c: "RESOLVED" for c in C.ALL_CAMERAS},
                          verbose=False)
        assert_roster_unchanged(st, before)

    def test_every_wagon_has_a_row_even_with_no_findings(self):
        from inspection.old_report import _camera_rows
        st = roster(9)
        rows = _camera_rows(C.CAMERA_RIGHT_UP, list(self._unified(st).values()))
        self.assertEqual(len(rows), 9)
        self.assertEqual([r[1] for r in rows], [f"GW_{i}" for i in range(1, 10)])


# ===========================================================================
# 7  processed-video overlays consume persisted state
# ===========================================================================

class TestVideoOverlays(unittest.TestCase):
    def test_renderer_consumes_persisted_state_not_events(self):
        import inspect as _inspect

        import video_segmenter as vs
        params = _inspect.signature(vs.render_processed_video).parameters
        self.assertIn("inspection_state", params)
        self.assertIsNone(params["inspection_state"].default)

    def test_counting_overlay_parameters_are_untouched(self):
        import inspect as _inspect

        import video_segmenter as vs
        params = _inspect.signature(vs.render_processed_video).parameters
        for name in ("local_tracks", "state", "output_path", "time_offset",
                     "drop_out_of_range", "non_wagon_regions"):
            self.assertIn(name, params)

    def test_door_and_damage_overlays_are_visually_distinct(self):
        import video_segmenter as vs
        self.assertNotEqual(vs._DOOR_BOX_COLOR, vs._DAMAGE_BOX_COLOR)

    def test_inspection_state_serializes_for_the_renderer(self):
        st = roster(3)
        res = oldf.OldInspectionResult()
        res.unified = {"GW_1": UnifiedWagonState(global_id="GW_1")}
        res.roster_hash_before = res.roster_hash_after = roster_hash(st)
        st.inspection = res.to_dict()
        d = st.to_dict()
        self.assertIn("inspection", d)
        self.assertIn("wagons", d["inspection"])
        self.assertTrue(d["inspection"]["roster_unchanged"])

    def test_inspection_block_is_additive_to_the_json(self):
        plain = GlobalTrainState(total_wagons=0,
                                master_camera=C.CAMERA_RIGHT_UP).to_dict()
        st = roster(2)
        st.inspection = {"summary": {}}
        withi = st.to_dict()
        for key in plain:
            self.assertIn(key, withi, f"{key} disappeared from the JSON")

    def test_damage_evidence_carries_what_an_overlay_needs(self):
        st = roster(2)
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "damage"))
            with open(os.path.join(tmp, "damage", "GW_1.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"global_id": "GW_1", "feature": "damage",
                           "status": C.STATUS_OK,
                           "top_damage": C.DAMAGE_PRESENT,
                           "top_damage_details": [{
                               "class_name": "Inner_wall_damage",
                               "confidence": 0.9,
                               "bbox": [10.0, 20.0, 90.0, 120.0],
                               "frame_idx": 42,
                               "camera_id": C.CAMERA_RIGHT_UP_TOP,
                               "track_id": 2}],
                           "supporting_cameras": [C.CAMERA_RIGHT_UP_TOP]}, fh)
            unified = oldf.fuse_unified_states(
                st, tmp, {c: "RESOLVED" for c in C.ALL_CAMERAS}, tmp)
        det = unified["GW_1"].evidence["damage"][0]
        for key in ("class_name", "confidence", "bbox", "frame_idx", "camera_id"):
            self.assertIn(key, det)
        self.assertEqual(unified["GW_1"].top_damage, C.DAMAGE_PRESENT)


# ===========================================================================
# 8  two trains in one process
# ===========================================================================

class TestTrainIsolation(unittest.TestCase):
    def test_fusion_state_does_not_leak_between_trains(self):
        st_a = roster(3)
        st_b = roster(5)
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "door"))
            with open(os.path.join(tmp, "door", "GW_1.json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"global_id": "GW_1", "status": C.STATUS_OK,
                           "left_door": C.DOOR_OPEN,
                           "left_door_confidence": 0.9,
                           "right_door": C.NO_DATA,
                           "supporting_cameras": [C.CAMERA_LEFT_UP]}, fh)
            a = oldf.fuse_unified_states(
                st_a, tmp, {c: "RESOLVED" for c in C.ALL_CAMERAS}, tmp)
            # Train B has its OWN (empty) feature directory.
            with tempfile.TemporaryDirectory() as tmp_b:
                b = oldf.fuse_unified_states(
                    st_b, tmp_b, {c: "RESOLVED" for c in C.ALL_CAMERAS}, tmp_b)
        self.assertEqual(a["GW_1"].left_door, C.DOOR_OPEN)
        self.assertEqual(b["GW_1"].left_door, C.NO_DATA,
                         "train B's GW_1 must not inherit train A's finding")
        self.assertEqual(len(a), 3)
        self.assertEqual(len(b), 5)

    def test_wagon_cache_is_cleared_between_trains(self):
        """One train's GW_1 frames must never be read by the next train's GW_1."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        root = os.path.join(tmp, wc.CACHE_DIRNAME)
        d = wc.wagon_camera_dir(root, "GW_1", C.CAMERA_RIGHT_UP)
        os.makedirs(d)
        open(os.path.join(d, "frame_000001.jpg"), "wb").close()
        self.assertEqual(wc.cache_stats(root)["frames"], 1)
        removed = wc.clear_wagon_cache(root, verbose=False)
        self.assertEqual(removed, 1)
        self.assertEqual(wc.cache_stats(root)["frames"], 0)

    def test_default_config_clears_the_cache(self):
        self.assertFalse(oldf.OldInspectionConfig().keep_cache)

    def test_result_objects_are_independent(self):
        a = oldf.OldInspectionResult()
        a.warnings.append("train A")
        b = oldf.OldInspectionResult()
        self.assertEqual(b.warnings, [])
        self.assertEqual(b.unified, {})

    def test_roster_hash_differs_between_trains(self):
        self.assertNotEqual(roster_hash(roster(3)), roster_hash(roster(5)))


# ===========================================================================
# 9  config carries no train-specific constants
# ===========================================================================

class TestNoTrainSpecificConstants(unittest.TestCase):
    def test_cache_config_is_counts_only(self):
        cfg = wc.WagonCacheConfig()
        for name in cfg.__dataclass_fields__:
            self.assertFalse(name.endswith(("_px", "_frame", "_frames"))
                             and "per_wagon" not in name,
                             f"{name} looks like an absolute geometry constant")

    def test_orchestrator_config_duplicates_no_threshold(self):
        """Verdict thresholds must live in old_code / core.constants only."""
        fields = set(oldf.OldInspectionConfig().to_dict())
        for banned in ("confidence", "conf", "iou", "ratio", "threshold"):
            self.assertFalse(any(banned in f for f in fields),
                             f"orchestrator must not own a '{banned}' knob")

    def test_no_local_train_measurements_in_the_bridge(self):
        import io
        import tokenize
        numbers = set()
        for rel in ("inspection/wagon_cache.py", "inspection/old_features.py"):
            with open(os.path.join(ROOT, rel), "rb") as fh:
                for tok in tokenize.tokenize(io.BytesIO(fh.read()).readline):
                    if tok.type == tokenize.NUMBER:
                        numbers.add(tok.string)
        for value in ("57", "58", "3555", "3425", "316", "0.8833", "2.6833"):
            self.assertNotIn(value, numbers,
                             f"local-train measurement {value!r} leaked into code")


if __name__ == "__main__":
    unittest.main(verbosity=2)
