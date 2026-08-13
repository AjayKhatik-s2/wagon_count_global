"""Inspection annotates the global train; it never changes it.

The properties under test are, in order of importance:

  1. Inspection cannot create, delete or renumber a global wagon.
  2. A finding is attributed to the correct EXISTING GW id, via the offsets the
     counting pipeline already resolved -- and to NO id when a camera's offset was
     never resolved.
  3. Temporal evidence decides a finding, not any single frame.
  4. "No detection" is never reported as "no damage".

Model-dependent tests are skipped when the weights are absent, so the suite still
runs on a machine without them. Everything else is synthetic and geometry-agnostic.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inspection import association as assoc
from inspection import evidence as insp_ev
from inspection import models as insp_models
from inspection.state import (
    ASSOCIATION_AMBIGUOUS, ASSOCIATION_RESOLVED, ASSOCIATION_UNRESOLVED,
    CAMERA_NOT_VISIBLE, CAMERA_NO_DETECTION, CAMERA_UNRESOLVED,
    InspectionConfig, InspectionEvent, InspectionState, WagonInspection,
)
from inspection.tracking import (
    DetectionObservation, InspectionTrack, classify_track, iou,
    suppress_duplicates_in_frame, track_detections,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
FPS = 15.0
WIDTH = 960
ADVANCE = 28.0        # px/frame: the measured train-speed advance of a real door


def obs(frame, x, *, cls="open_door", cid=2, conf=0.90, w=60.0, fps=FPS):
    return DetectionObservation(
        frame=frame, time_local=frame / fps, class_id=cid, class_name=cls,
        confidence=conf, bbox=(x, 100.0, x + w, 300.0))


def stream(frames, *, x0=100.0, dx=ADVANCE, cls="open_door", cid=2, conf=0.90,
           fps=FPS, w=60.0):
    """A steadily advancing detection, one per frame -- the genuine signature."""
    for i, f in enumerate(frames):
        yield f, f / fps, [obs(f, x0 + dx * i, cls=cls, cid=cid, conf=conf,
                               fps=fps, w=w)]


def make_track(frames, **kw):
    cfg = kw.pop("cfg", None) or InspectionConfig()
    cam = kw.pop("camera_id", "RIGHT_UP")
    role = kw.pop("role", "door")
    fps = kw.get("fps", FPS)
    return track_detections(stream(frames, **kw), cam, role, cfg, fps=fps,
                            frame_width=WIDTH)


def roster(n=5, span=10.0):
    """A finished global wagon roster: GW_1..GW_n, contiguous in master time."""
    return {
        "wagons": [{"global_id": f"GW_{i}", "wagon_index": i,
                    "start_time": span * (i - 1), "end_time": span * i,
                    "classification": "WAGON"} for i in range(1, n + 1)],
        "camera_offsets": {
            "RIGHT_UP": {"status": "REFERENCE", "delta": 0.0},
            "LEFT_UP": {"status": "RESOLVED", "delta": 2.0},
            "RIGHT_UP_TOP": {"status": "RESOLVED", "delta": -1.5},
            "LEFT_UP_TOP": {"status": "UNRESOLVED", "delta": 99.0},
        },
    }


# ===========================================================================
# 1-3  models load and their REAL classes are discovered
# ===========================================================================

class TestModelDiscovery(unittest.TestCase):
    def test_missing_model_is_unavailable_not_an_exception(self):
        """A missing model must never crash a run that already counted wagons."""
        av = insp_models.discover_models(os.path.join(ROOT, "no_such_dir"))
        for role in (insp_models.DOOR_ROLE, insp_models.DAMAGE_ROLE):
            self.assertFalse(av[role].is_available)
            self.assertIn("not found", av[role].reason)

    def test_availability_banner_names_both_roles(self):
        av = insp_models.discover_models(os.path.join(ROOT, "no_such_dir"))
        text = insp_models.describe_model_availability(av)
        self.assertIn("DOOR MODEL", text)
        self.assertIn("TOP DAMAGE MODEL", text)
        self.assertIn("UNAVAILABLE", text)

    @unittest.skipUnless(os.path.isfile(os.path.join(MODELS_DIR, "door_state.pt")),
                         "door_state.pt not present")
    def test_door_model_loads_and_reports_its_own_classes(self):
        av = insp_models.discover_models(MODELS_DIR)[insp_models.DOOR_ROLE]
        self.assertTrue(av.is_available, av.reason)
        self.assertTrue(av.class_names, "classes must be discovered, not assumed")
        self.assertEqual(av.task, "detect")
        # Grouping is by name; the door model is known to ship a damage class, and
        # it must NOT be filed as a door state.
        self.assertTrue(av.class_groups["door_state"])
        for name in av.class_groups["damage"]:
            self.assertNotIn(name, av.class_groups["door_state"])

    @unittest.skipUnless(os.path.isfile(os.path.join(MODELS_DIR, "top_damage.pt")),
                         "top_damage.pt not present")
    def test_damage_model_loads_and_reports_its_own_classes(self):
        av = insp_models.discover_models(MODELS_DIR)[insp_models.DAMAGE_ROLE]
        self.assertTrue(av.is_available, av.reason)
        self.assertTrue(av.class_names)
        self.assertTrue(av.class_groups["damage"])

    def test_class_partition_uses_names_never_ids(self):
        """Reordering class ids must not change the grouping."""
        a = insp_models.partition_class_names(
            {0: "closed_door", 1: "damage", 2: "open_door"})
        b = insp_models.partition_class_names(
            {0: "damage", 1: "open_door", 2: "closed_door"})
        self.assertEqual(a, b)
        self.assertIn("damage", a["damage"])
        self.assertIn("open_door", a["door_state"])

    def test_unknown_class_is_kept_not_dropped(self):
        groups = insp_models.partition_class_names({0: "mystery_label"})
        self.assertEqual(groups["other"], ["mystery_label"])


# ===========================================================================
# 4-8  temporal tracking, flicker suppression, persistence
# ===========================================================================

class TestTracking(unittest.TestCase):
    def test_steady_detection_becomes_one_track(self):
        tracks = make_track(range(300, 328))
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].n_observations, 28)

    def test_persistent_detection_is_confirmed(self):
        """The measured genuine door: 28 frames, advancing, high confidence."""
        tr = make_track(range(300, 328))[0]
        ok, why = classify_track(tr, InspectionConfig(), WIDTH)
        self.assertTrue(ok, why)

    def test_one_frame_detection_is_never_a_finding(self):
        tr = make_track(range(300, 301))[0]
        ok, why = classify_track(tr, InspectionConfig(), WIDTH)
        self.assertFalse(ok)
        self.assertIn("sighting", why)

    def test_two_frame_flicker_is_rejected(self):
        tr = make_track(range(300, 302))[0]
        self.assertFalse(classify_track(tr, InspectionConfig(), WIDTH)[0])

    def test_low_confidence_run_is_rejected(self):
        """A long run that never gets strong: the measured FP profile."""
        tr = make_track(range(300, 330), conf=0.45)[0]
        ok, why = classify_track(tr, InspectionConfig(), WIDTH)
        self.assertFalse(ok)
        self.assertIn("peak confidence", why)

    def test_static_detection_is_rejected_however_long(self):
        """Pinned while the train moves -- the measured static artefact."""
        tr = make_track(range(300, 360), dx=0.0)[0]
        ok, why = classify_track(tr, InspectionConfig(), WIDTH)
        self.assertFalse(ok)
        self.assertIn("pinned", why)

    def test_a_break_longer_than_the_miss_window_splits_the_track(self):
        cfg = InspectionConfig()
        gap = int(cfg.max_track_miss_seconds * FPS) + 5
        frames = list(range(300, 310)) + list(range(310 + gap, 320 + gap))
        tracks = track_detections(stream(frames), "RIGHT_UP", "door", cfg,
                                  fps=FPS, frame_width=WIDTH)
        self.assertEqual(len(tracks), 2)

    def test_class_vote_is_confidence_weighted(self):
        """Several classes fire on one object; the sustained strong one wins.

        Measured: one door produced partially_closed 0.46, open_door 0.33 and
        closed_door 0.24 in a single frame.
        """
        per_frame = []
        for i, f in enumerate(range(300, 320)):
            x = 100.0 + ADVANCE * i
            per_frame.append((f, f / FPS, [
                obs(f, x, cls="partially_closed", cid=3, conf=0.90),
                obs(f, x + 1, cls="open_door", cid=2, conf=0.30)]))
        tracks = track_detections(iter(per_frame), "RIGHT_UP", "door",
                                  InspectionConfig(), fps=FPS, frame_width=WIDTH)
        strong = max(tracks, key=lambda t: t.peak_confidence)
        self.assertEqual(strong.dominant_class()[1], "partially_closed")

    def test_verdicts_are_stride_invariant_up_to_the_measured_limit(self):
        """Measured on real windows: strides 1, 2 and 4 agree; 8 loses findings.

        Detect once, then subsample, so only the stride varies. This guards the
        property that makes `frame_stride` a pure cost dial.
        """
        frames = list(range(300, 361))          # a 61-frame genuine finding
        dense = list(stream(frames))
        for stride in (1, 2, 4):
            sub = [r for r in dense if (r[0] - 300) % stride == 0]
            cfg = InspectionConfig(frame_stride=stride)
            tracks = track_detections(iter(sub), "RIGHT_UP", "door", cfg,
                                      fps=FPS, frame_width=WIDTH)
            self.assertEqual(len(tracks), 1, f"fragmented at stride {stride}")
            self.assertTrue(classify_track(tracks[0], cfg, WIDTH)[0],
                            f"finding lost at stride {stride}")

    def test_association_allowance_is_per_frame_interval(self):
        """A wider sampling gap must permit a proportionally larger jump."""
        from inspection.tracking import association_score
        cfg = InspectionConfig()
        a = obs(300, 100.0)
        far = obs(304, 100.0 + 4 * ADVANCE)      # 4 intervals of real motion
        self.assertGreater(association_score(a, far, cfg, WIDTH), 0.0)
        # The same jump in ONE interval is not physically plausible.
        abrupt = obs(301, 100.0 + 4 * ADVANCE * 3)
        self.assertEqual(association_score(a, abrupt, cfg, WIDTH), 0.0)

    def test_stride_does_not_change_confirmation(self):
        """Persistence is in seconds, so sampling density must not decide it."""
        dense = make_track(range(300, 330))[0]
        sparse = track_detections(stream(range(300, 330, 3)), "RIGHT_UP", "door",
                                  InspectionConfig(frame_stride=3), fps=FPS,
                                  frame_width=WIDTH)[0]
        self.assertTrue(classify_track(dense, InspectionConfig(), WIDTH)[0])
        self.assertTrue(classify_track(sparse, InspectionConfig(), WIDTH)[0])


# ===========================================================================
# 17-18  duplicate merging
# ===========================================================================

class TestDuplicates(unittest.TestCase):
    def test_near_identical_boxes_in_one_frame_collapse(self):
        """Measured: three Floor_damage boxes at cx 943/943/944 in one frame."""
        dupes = [obs(10, 900.0, cls="Floor_damage", cid=1, conf=0.26),
                 obs(10, 901.0, cls="Floor_damage", cid=1, conf=0.24),
                 obs(10, 902.0, cls="Floor_damage", cid=1, conf=0.23)]
        kept = suppress_duplicates_in_frame(dupes)
        self.assertEqual(len(kept), 1)
        self.assertAlmostEqual(kept[0].confidence, 0.26)

    def test_different_classes_overlapping_are_both_kept(self):
        """Two classes on one object is real information, resolved later by vote."""
        pair = [obs(10, 500.0, cls="open_door", cid=2, conf=0.5),
                obs(10, 501.0, cls="closed_door", cid=0, conf=0.4)]
        self.assertEqual(len(suppress_duplicates_in_frame(pair)), 2)

    def test_distant_boxes_of_one_class_are_both_kept(self):
        pair = [obs(10, 100.0, cls="Floor_damage", cid=1, conf=0.5),
                obs(10, 800.0, cls="Floor_damage", cid=1, conf=0.5)]
        self.assertEqual(len(suppress_duplicates_in_frame(pair)), 2)

    def test_duplicate_suppression_is_deterministic(self):
        dupes = [obs(10, 900.0 + i, cls="Floor_damage", cid=1, conf=0.3)
                 for i in range(4)]
        a = [o.bbox for o in suppress_duplicates_in_frame(dupes)]
        b = [o.bbox for o in suppress_duplicates_in_frame(list(reversed(dupes)))]
        self.assertEqual(a, b)

    def test_iou_is_symmetric_and_bounded(self):
        a, b = (0, 0, 10, 10), (5, 5, 15, 15)
        self.assertAlmostEqual(iou(a, b), iou(b, a))
        self.assertAlmostEqual(iou(a, a), 1.0)
        self.assertEqual(iou(a, (100, 100, 110, 110)), 0.0)


# ===========================================================================
# 9-13  association to the correct EXISTING GW id
# ===========================================================================

class TestAssociation(unittest.TestCase):
    def setUp(self):
        self.state = roster()
        self.wagons = assoc.wagon_intervals_from_state(self.state)
        self.offsets = assoc.trusted_offsets(self.state)

    def test_finding_lands_on_the_wagon_containing_it(self):
        tr = make_track(range(int(15.0 * FPS), int(15.0 * FPS) + 20))[0]
        a = assoc.associate_track(tr, self.wagons, self.offsets)
        self.assertEqual(a["global_id"], "GW_2")     # master 10-20s
        self.assertEqual(a["association_status"], ASSOCIATION_RESOLVED)

    def test_camera_offset_is_applied(self):
        """LEFT_UP is +2.0s, so the same local frames land 2s later globally."""
        frames = range(int(8.5 * FPS), int(8.5 * FPS) + 15)
        right = make_track(frames, camera_id="RIGHT_UP")[0]
        left = make_track(frames, camera_id="LEFT_UP")[0]
        a = assoc.associate_track(right, self.wagons, self.offsets)
        b = assoc.associate_track(left, self.wagons, self.offsets)
        self.assertEqual(a["global_id"], "GW_1")     # ~8.5-9.4s master
        self.assertEqual(b["global_id"], "GW_2")     # ~10.5-11.4s master
        self.assertAlmostEqual(b["camera_offset"], 2.0)

    def test_same_physical_wagon_across_cameras_maps_to_one_gw_id(self):
        """The core multi-camera requirement: no per-camera wagon identities."""
        # RIGHT_UP at 15.0s and LEFT_UP at 13.0s local are the SAME instant.
        r = make_track(range(int(15.0 * FPS), int(15.0 * FPS) + 12),
                       camera_id="RIGHT_UP")[0]
        l = make_track(range(int(13.0 * FPS), int(13.0 * FPS) + 12),
                       camera_id="LEFT_UP")[0]
        t = make_track(range(int(16.5 * FPS), int(16.5 * FPS) + 12),
                       camera_id="RIGHT_UP_TOP")[0]
        ids = {assoc.associate_track(x, self.wagons, self.offsets)["global_id"]
               for x in (r, l, t)}
        self.assertEqual(ids, {"GW_2"},
                         "three cameras observing one wagon must agree on its id")

    def test_unresolved_camera_never_invents_a_gw_id(self):
        tr = make_track(range(int(15.0 * FPS), int(15.0 * FPS) + 20),
                        camera_id="LEFT_UP_TOP")[0]
        a = assoc.associate_track(tr, self.wagons, self.offsets)
        self.assertIsNone(a["global_id"])
        self.assertEqual(a["association_status"], ASSOCIATION_UNRESOLVED)
        self.assertIsNone(a["global_time_start"])

    def test_unresolved_offset_status_is_not_trusted(self):
        self.assertNotIn("LEFT_UP_TOP", self.offsets)
        self.assertIn("RIGHT_UP", self.offsets)

    def test_detection_outside_every_wagon_is_unresolved(self):
        """Past the last wagon, e.g. over a brake van: no id, and it says why."""
        tr = make_track(range(int(500.0 * FPS), int(500.0 * FPS) + 20))[0]
        a = assoc.associate_track(tr, self.wagons, self.offsets)
        self.assertIsNone(a["global_id"])
        self.assertIn("outside every global wagon", a["association_detail"])

    def test_boundary_straddling_finding_goes_to_the_larger_share(self):
        # 9.0 -> 11.0s master straddles GW_1/GW_2; skewed so one clearly wins.
        frames = range(int(9.8 * FPS), int(11.4 * FPS))
        tr = make_track(frames)[0]
        a = assoc.associate_track(tr, self.wagons, self.offsets)
        self.assertEqual(a["global_id"], "GW_2")
        self.assertIn("GW_1", a["candidate_global_ids"])

    def test_an_even_straddle_is_reported_ambiguous(self):
        frames = range(int(9.5 * FPS), int(10.5 * FPS) + 1)
        tr = make_track(frames)[0]
        a = assoc.associate_track(tr, self.wagons, self.offsets)
        self.assertEqual(a["association_status"], ASSOCIATION_AMBIGUOUS)
        self.assertIsNotNone(a["global_id"])

    def test_empty_roster_yields_no_association(self):
        a = assoc.associate_track(make_track(range(300, 320))[0], [], self.offsets)
        self.assertIsNone(a["global_id"])
        self.assertEqual(a["association_method"], "none:empty_roster")

    def test_association_is_deterministic(self):
        tr = make_track(range(int(15.0 * FPS), int(15.0 * FPS) + 20))[0]
        a = assoc.associate_track(tr, self.wagons, self.offsets)
        b = assoc.associate_track(tr, self.wagons, self.offsets)
        self.assertEqual(a, b)

    def test_association_does_not_mutate_the_roster(self):
        before = [(w.global_id, w.start_time, w.end_time) for w in self.wagons]
        for f in (300, 900, 1500):
            assoc.associate_track(make_track(range(f, f + 20))[0], self.wagons,
                                  self.offsets)
        self.assertEqual([(w.global_id, w.start_time, w.end_time)
                          for w in self.wagons], before)


# ===========================================================================
# 14-16, 24  the counting result is untouchable
# ===========================================================================

class TestCountingIsProtected(unittest.TestCase):
    def test_reading_the_roster_does_not_change_the_state(self):
        state = roster(6)
        import copy
        snapshot = copy.deepcopy(state)
        assoc.wagon_intervals_from_state(state)
        assoc.trusted_offsets(state)
        self.assertEqual(state, snapshot)

    def test_inspection_state_holds_only_existing_gw_ids(self):
        state = roster(4)
        ins = InspectionState()
        for w in assoc.wagon_intervals_from_state(state):
            ins.wagons[w.global_id] = WagonInspection(global_id=w.global_id)
        self.assertEqual(sorted(ins.wagons), ["GW_1", "GW_2", "GW_3", "GW_4"])

    def test_wagons_with_no_findings_are_still_present(self):
        """Requirement: do not omit wagons with no detections."""
        ins = InspectionState()
        for i in range(1, 6):
            ins.wagons[f"GW_{i}"] = WagonInspection(global_id=f"GW_{i}")
        out = ins.to_dict()
        self.assertEqual(len(out["wagons"]), 5)
        for rec in out["wagons"].values():
            self.assertIsNone(rec["door_state"]["state"])
            self.assertIsNone(rec["top_damage"]["state"])

    def test_gw_ids_serialize_in_numeric_order(self):
        ins = InspectionState()
        for i in (1, 2, 10, 11, 3):
            ins.wagons[f"GW_{i}"] = WagonInspection(global_id=f"GW_{i}")
        self.assertEqual(list(ins.to_dict()["wagons"]),
                         ["GW_1", "GW_2", "GW_3", "GW_10", "GW_11"])

    def test_a_finding_cannot_introduce_a_new_gw_id(self):
        """A detection between two wagons must not become a wagon of its own."""
        state = roster(3)
        wagons = assoc.wagon_intervals_from_state(state)
        offsets = assoc.trusted_offsets(state)
        ins = InspectionState()
        for w in wagons:
            ins.wagons[w.global_id] = WagonInspection(global_id=w.global_id)
        known = set(ins.wagons)
        tr = make_track(range(int(999.0 * FPS), int(999.0 * FPS) + 20))[0]
        a = assoc.associate_track(tr, wagons, offsets)
        if a["global_id"]:
            self.assertIn(a["global_id"], known)
        self.assertEqual(set(ins.wagons), known)


# ===========================================================================
# 28  "no detection" is not "no damage"
# ===========================================================================

class TestVisibilitySemantics(unittest.TestCase):
    def test_wagon_outside_a_camera_footage_is_not_visible(self):
        wagons = assoc.wagon_intervals_from_state(roster(5))
        # Camera footage ends at 25s, so GW_4 (30-40s) was never observed.
        vis = assoc.camera_visibility(wagons, "RIGHT_UP", {"RIGHT_UP": 0.0},
                                      camera_duration=25.0)
        self.assertEqual(vis["GW_1"], CAMERA_NO_DETECTION)
        self.assertEqual(vis["GW_4"], CAMERA_NOT_VISIBLE)

    def test_unresolved_camera_marks_every_wagon_unresolved(self):
        wagons = assoc.wagon_intervals_from_state(roster(3))
        vis = assoc.camera_visibility(wagons, "LEFT_UP_TOP", {}, 100.0)
        self.assertEqual(set(vis.values()), {CAMERA_UNRESOLVED})

    def test_no_detection_and_not_visible_are_distinct_values(self):
        self.assertNotEqual(CAMERA_NO_DETECTION, CAMERA_NOT_VISIBLE)
        self.assertNotEqual(CAMERA_NO_DETECTION, CAMERA_UNRESOLVED)


# ===========================================================================
# 19, 22  evidence selection
# ===========================================================================

class TestEvidence(unittest.TestCase):
    def _track(self):
        per = []
        confs = [0.4, 0.6, 0.95, 0.7, 0.5]
        for i, (f, c) in enumerate(zip(range(300, 305), confs)):
            per.append((f, f / FPS, [obs(f, 100.0 + ADVANCE * i, conf=c)]))
        return track_detections(iter(per), "RIGHT_UP", "door",
                                InspectionConfig(), fps=FPS, frame_width=WIDTH)[0]

    def test_peak_frame_is_selected(self):
        picks = insp_ev.select_evidence_frames(self._track(), InspectionConfig())
        self.assertTrue(any(p["selection"] == "peak" and p["frame"] == 302
                            for p in picks))

    def test_selection_is_capped(self):
        cfg = InspectionConfig(max_evidence_frames_per_event=2)
        self.assertLessEqual(len(insp_ev.select_evidence_frames(self._track(), cfg)), 2)

    def test_selection_is_deterministic(self):
        tr, cfg = self._track(), InspectionConfig()
        self.assertEqual(insp_ev.select_evidence_frames(tr, cfg),
                         insp_ev.select_evidence_frames(tr, cfg))

    def test_every_pick_carries_what_the_report_must_show(self):
        for p in insp_ev.select_evidence_frames(self._track(), InspectionConfig()):
            for key in ("camera_id", "frame", "bbox", "class_name", "confidence"):
                self.assertIn(key, p)

    def test_missing_track_is_marked_unavailable_not_silently_empty(self):
        ev = InspectionEvent(
            event_id="D1", role="door", model_path="m.pt", model_class_id=2,
            model_class_name="open_door", camera_id="RIGHT_UP", track_id=7,
            start_frame=1, end_frame=9, start_time_local=0.0, end_time_local=0.6,
            n_observations=9, peak_confidence=0.9, mean_confidence=0.8)
        insp_ev.attach_evidence([ev], {}, InspectionConfig())
        self.assertEqual(len(ev.evidence_frames), 1)
        self.assertFalse(ev.evidence_frames[0]["available"])
        self.assertIn("unavailable", ev.evidence_frames[0]["reason"])


# ===========================================================================
# 26-27, 29-30  isolation, geometry independence, failure reporting, memory
# ===========================================================================

class TestRobustness(unittest.TestCase):
    def test_two_trains_in_one_process_do_not_share_state(self):
        a = InspectionState()
        a.wagons["GW_1"] = WagonInspection(global_id="GW_1")
        b = InspectionState()
        self.assertEqual(b.wagons, {})
        self.assertEqual(b.events, [])
        self.assertEqual(b.warnings, [])

    def test_tracking_holds_no_module_state_between_runs(self):
        first = make_track(range(300, 320))
        second = make_track(range(300, 320))
        self.assertEqual([t.to_dict() for t in first],
                         [t.to_dict() for t in second])

    def test_association_works_at_a_different_fps_and_resolution(self):
        state = roster()
        wagons = assoc.wagon_intervals_from_state(state)
        offsets = assoc.trusted_offsets(state)
        for fps in (10.0, 25.0, 30.0):
            frames = range(int(15.0 * fps), int(15.0 * fps) + 12)
            tr = track_detections(stream(frames, fps=fps), "RIGHT_UP", "door",
                                  InspectionConfig(), fps=fps, frame_width=1920)[0]
            a = assoc.associate_track(tr, wagons, offsets)
            self.assertEqual(a["global_id"], "GW_2", f"failed at fps={fps}")

    def test_motion_floor_scales_with_frame_width(self):
        """Same scene at a different resolution must reach the same verdict."""
        cfg = InspectionConfig()
        # Scale the WHOLE scene, box size included -- a higher-resolution camera
        # sees proportionally larger objects, not same-size ones moving faster.
        for width, scale in ((640, 640 / 960), (1920, 2.0)):
            tracks = make_track(range(300, 330), dx=ADVANCE * scale,
                                x0=100.0 * scale, w=60.0 * scale)
            self.assertEqual(len(tracks), 1,
                             f"scene must stay ONE track at width={width}")
            self.assertTrue(classify_track(tracks[0], cfg, width)[0],
                            f"failed at width={width}")

    def test_per_frame_detection_ceiling_is_enforced(self):
        """A pathological frame must not balloon the observation list."""
        cfg = InspectionConfig(max_detections_per_frame=5)
        per = [(300, 20.0, [obs(300, 50.0 * i, conf=0.9) for i in range(40)])]
        tracks = track_detections(iter(per), "RIGHT_UP", "door", cfg, fps=FPS,
                                  frame_width=WIDTH)
        self.assertLessEqual(len(tracks), 5)

    def test_tracks_store_no_image_data(self):
        """Memory safety: an observation carries geometry, never pixels."""
        tr = make_track(range(300, 320))[0]
        for o in tr.observations:
            for value in vars(o).values():
                self.assertNotHasAttr = getattr(value, "shape", None)
                self.assertIsNone(getattr(value, "shape", None),
                                  "an observation must not hold an array/frame")

    def test_config_is_camera_independent(self):
        cfg = InspectionConfig()
        for name in cfg.__dataclass_fields__:
            self.assertFalse(name.endswith("_px"),
                             f"{name} is absolute; config must stay in seconds, "
                             f"fractions, ratios and counts")

    def test_state_serializes_without_the_retained_tracks(self):
        """Per-observation detail must not bloat the JSON."""
        ins = InspectionState()
        ins.tracks_by_key[("RIGHT_UP", 1)] = make_track(range(300, 320))[0]
        self.assertNotIn("tracks_by_key", ins.to_dict())

    def test_summary_counts_are_present_for_the_console_and_report(self):
        ins = InspectionState()
        out = ins.to_dict()
        for key in ("confirmed_door_events", "confirmed_damage_events",
                    "rejected_events", "unresolved_associations",
                    "association_status"):
            self.assertIn(key, out["summary"])

    def test_association_status_reflects_partial_resolution(self):
        ins = InspectionState()
        def ev(status, n):
            return InspectionEvent(
                event_id=f"E{n}", role="door", model_path="m.pt",
                model_class_id=2, model_class_name="open_door",
                camera_id="RIGHT_UP", track_id=n, start_frame=1, end_frame=9,
                start_time_local=0.0, end_time_local=0.6, n_observations=9,
                peak_confidence=0.9, mean_confidence=0.8,
                association_status=status)
        ins.events = [ev(ASSOCIATION_RESOLVED, 1), ev(ASSOCIATION_UNRESOLVED, 2)]
        self.assertEqual(ins.association_status(), "PARTIAL")
        ins.events = [ev(ASSOCIATION_RESOLVED, 1)]
        self.assertEqual(ins.association_status(), ASSOCIATION_RESOLVED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
