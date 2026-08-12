"""Unit tests for wagon-only counting and the top classification model.

The hard rules under test:

    ENGINE is not a wagon.  BRAKE_VAN is not a wagon.
    Neither ever receives a GW id, and neither extends the wagon timeline.
    GLOBAL WAGON COUNT == validated RIGHT_UP WAGON count.
    Support cameras are evidence only.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import global_fusion as gf
import train_structure as ts
from global_train_state import (
    ALL_CAMERAS, CAMERA_LEFT_UP, CAMERA_LEFT_UP_TOP, CAMERA_RIGHT_UP,
    CAMERA_RIGHT_UP_TOP, GapEvent, GlobalWagon, LocalCameraTracks, SegmentClass,
    _MasterClassification,
)

FPS = 15.0
GAP_SPAN = 12


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def make_tracks(camera_id, gap_times, duration_s=400.0, fps=FPS):
    gaps = []
    for i, t in enumerate(sorted(gap_times), start=1):
        c = t * fps
        s, e = int(round(c - GAP_SPAN / 2)), int(round(c + GAP_SPAN / 2))
        xs = [600.0 - 35.0 * k for k in range(GAP_SPAN + 1)]
        gaps.append(GapEvent(
            track_id=i, camera_id=camera_id, start_frame=max(0, s),
            end_frame=max(1, e), confidence=0.9, hit_count=GAP_SPAN + 1,
            center_x_trajectory=xs, fps=fps, temporal_consistency_score=1.0,
            hit_frames=list(range(max(0, s), max(1, e) + 1)),
            bbox_history=[[x - 20, 100, x + 20, 300] for x in xs]))
    return LocalCameraTracks(
        camera_id=camera_id, video_path=f"/synthetic/{camera_id}.mp4", fps=fps,
        total_frames=int(round(duration_s * fps)), width=848, height=480, gaps=gaps)


def segments_and_labels(labels, spacing=4.0, start=10.0, fps=FPS,
                        duration_s=400.0):
    """Build a master with len(labels) segments carrying the given labels.

    n labels -> n-1 internal gaps, plus the leading/trailing video edges.
    """
    n = len(labels)
    gap_times = [start + i * spacing for i in range(1, n)]
    master = make_tracks(CAMERA_RIGHT_UP, gap_times, duration_s=duration_s, fps=fps)

    # Boundaries the way build_global_wagons computes them.
    bounds = [int(round(g.center_frame)) for g in master.gaps]
    segs, prev = [], 0
    for b in sorted(bounds):
        if b <= prev:
            continue
        segs.append((prev, b - 1)); prev = b
    if prev <= master.total_frames - 1:
        segs.append((prev, master.total_frames - 1))

    assert len(segs) == n, f"expected {n} segments, built {len(segs)}"
    cls = [_MasterClassification(i, s, e, labels[i], 1.0)
           for i, (s, e) in enumerate(segs)]
    return master, cls


def assemble(master, supports, cls, wagon_only=True):
    return gf.assemble_global_train_state_master_fixed(
        master_tracks=master, support_tracks=supports,
        initial_classifications=cls, config=gf.FusionConfig(),
        verbose=False, wagon_only=wagon_only)


E, W, B, U = (SegmentClass.ENGINE, SegmentClass.WAGON,
              SegmentClass.BRAKE_VAN, SegmentClass.UNKNOWN)


# ===========================================================================
# engine / brake-van exclusion
# ===========================================================================

class TestWagonOnlyCounting(unittest.TestCase):
    def test_engine_plus_five_wagons(self):
        master, cls = segments_and_labels([E, W, W, W, W, W])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 5)
        self.assertEqual([w.global_id for w in st.wagons],
                         ["GW_1", "GW_2", "GW_3", "GW_4", "GW_5"])
        self.assertTrue(all(w.classification == W for w in st.wagons))

    def test_five_wagons_plus_brake_van(self):
        master, cls = segments_and_labels([W, W, W, W, W, B])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 5)
        self.assertEqual(st.wagons[-1].global_id, "GW_5")

    def test_engine_five_wagons_brake_van(self):
        master, cls = segments_and_labels([E, W, W, W, W, W, B])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 5)
        self.assertEqual([w.global_id for w in st.wagons],
                         ["GW_1", "GW_2", "GW_3", "GW_4", "GW_5"])

    def test_three_engines_then_wagons_then_brake_van(self):
        master, cls = segments_and_labels([E, E, E, W, W, W, W, B])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 4)
        ww = st.wagon_window
        self.assertEqual(ww["leading_non_wagon_count"], 3)
        self.assertEqual(ww["trailing_non_wagon_count"], 1)
        self.assertEqual(ww["leading_non_wagon_classes"], {E: 3})
        self.assertEqual(ww["trailing_non_wagon_classes"], {B: 1})

    def test_the_worked_example_from_the_brief(self):
        """ENGINE WAGON WAGON WAGON BRAKE_VAN -> GW_1 GW_2 GW_3, not five ids."""
        master, cls = segments_and_labels([E, W, W, W, B])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 3)
        self.assertEqual([w.global_id for w in st.wagons], ["GW_1", "GW_2", "GW_3"])

    def test_no_engine_or_brakevan_ever_holds_a_gw_id(self):
        master, cls = segments_and_labels([E, E, W, W, B, B])
        st = assemble(master, [], cls)
        for w in st.wagons:
            self.assertNotIn(w.classification, (E, B))

    def test_interior_engine_is_excluded_from_the_count(self):
        """An engine between wagons still never receives a GW id."""
        master, cls = segments_and_labels([E, W, W, E, W, W, B])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 4)
        self.assertEqual(st.wagon_window["interior_non_wagon_count"], 1)
        self.assertEqual(st.wagon_window["interior_non_wagon_classes"], {E: 1})

    def test_unknown_inside_the_window_is_counted_and_reported(self):
        """An unlabelled vehicle between two wagons is physically a wagon."""
        master, cls = segments_and_labels([E, W, U, W, B])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 3)
        self.assertIn(U, [w.classification for w in st.wagons])

    def test_unknown_outside_the_window_is_not_counted(self):
        master, cls = segments_and_labels([U, E, W, W, B, U])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 2)

    def test_no_wagons_at_all_yields_zero(self):
        master, cls = segments_and_labels([E, E, B])
        st = assemble(master, [], cls)
        self.assertEqual(st.total_wagons, 0)
        self.assertEqual(st.wagons, [])
        self.assertFalse(st.wagon_window["found"])

    def test_wagon_only_can_be_disabled_for_ab_comparison(self):
        master, cls = segments_and_labels([E, W, W, W, B])
        st = assemble(master, [], cls, wagon_only=False)
        self.assertEqual(st.total_wagons, 5, "legacy behaviour counts every segment")


# ===========================================================================
# the wagon window itself
# ===========================================================================

class TestWagonWindow(unittest.TestCase):
    def _segments(self, labels):
        out = []
        for i, lb in enumerate(labels):
            out.append(GlobalWagon(
                global_id=f"SEG_{i}", wagon_index=i,
                start_frame_master=i * 100, end_frame_master=i * 100 + 99,
                start_time=i * 100 / FPS, end_time=(i * 100 + 100) / FPS,
                classification=lb, classification_confidence=1.0))
        return out

    def test_first_and_last_wagon_bound_the_window(self):
        win = ts.get_master_wagon_window(self._segments([E, E, W, W, W, B]),
                                        verbose=False)
        self.assertTrue(win.found)
        self.assertEqual(win.first_wagon_segment_index, 2)
        self.assertEqual(win.last_wagon_segment_index, 4)
        self.assertEqual(win.master_wagon_count, 3)
        self.assertEqual(win.wagon_start_frame, 200)
        self.assertEqual(win.wagon_end_frame, 499)

    def test_temporal_order_and_timestamps_are_preserved(self):
        """Non-wagon frames are excluded from counting, never deleted or shifted."""
        segs = self._segments([E, W, W, B])
        win = ts.get_master_wagon_window(segs, verbose=False)
        self.assertEqual(win.wagon_units[0].start_frame_master, 100)
        self.assertEqual(win.wagon_units[0].start_time, 100 / FPS)
        self.assertEqual(win.leading_non_wagon_objects[0].start_frame, 0)
        self.assertEqual(win.trailing_non_wagon_objects[0].start_frame, 300)
        # the master frame numbers are untouched by renumbering
        self.assertEqual([w.global_id for w in win.wagon_units], ["GW_1", "GW_2"])

    def test_every_segment_is_accounted_for(self):
        for labels in ([E, W, B], [E, E, W, W, W, B], [W], [E, W, E, W, B],
                       [U, W, U], [E, B]):
            win = ts.get_master_wagon_window(self._segments(labels), verbose=False)
            total = (win.master_wagon_count
                     + len(win.leading_non_wagon_objects)
                     + len(win.trailing_non_wagon_objects)
                     + len(win.interior_non_wagon_objects))
            self.assertEqual(total, len(labels), f"labels={labels}")

    def test_engine_and_brakevan_metadata_is_preserved(self):
        win = ts.get_master_wagon_window(self._segments([E, W, W, B]), verbose=False)
        lead = win.leading_non_wagon_objects[0]
        self.assertEqual(lead.classification, E)
        self.assertEqual(lead.position, "leading")
        self.assertIn("classification", lead.to_dict())
        trail = win.trailing_non_wagon_objects[0]
        self.assertEqual(trail.classification, B)
        self.assertEqual(trail.position, "trailing")

    def test_empty_input(self):
        win = ts.get_master_wagon_window([], verbose=False)
        self.assertFalse(win.found)
        self.assertEqual(win.master_wagon_count, 0)


# ===========================================================================
# support cameras cannot change the count
# ===========================================================================

class TestSupportCannotInflate(unittest.TestCase):
    def test_engine_and_brakevan_on_all_cameras_do_not_inflate(self):
        master, cls = segments_and_labels([E, W, W, W, W, B])
        supports = [make_tracks(c, [t / FPS for t in
                                    [g.center_frame for g in master.gaps]])
                    for c in ALL_CAMERAS if c != CAMERA_RIGHT_UP]
        st = assemble(master, supports, cls)
        self.assertEqual(st.total_wagons, 4)

    def test_support_extra_observations_cannot_increase_the_count(self):
        master, cls = segments_and_labels([E, W, W, W, B])
        base = [g.center_time for g in master.gaps]
        noisy = sorted(base + [b + 1.7 for b in base] + [b + 2.4 for b in base])
        supports = [make_tracks(CAMERA_LEFT_UP, noisy)]
        st = assemble(master, supports, cls)
        self.assertEqual(st.total_wagons, 3)
        self.assertEqual(st.corrections_applied, [])

    def test_support_missing_observations_do_not_change_the_count(self):
        master, cls = segments_and_labels([E, W, W, W, W, W, B])
        supports = [make_tracks(CAMERA_LEFT_UP, [master.gaps[0].center_time])]
        st = assemble(master, supports, cls)
        self.assertEqual(st.total_wagons, 5)

    def test_duplicate_support_detections_cannot_create_ids(self):
        master, cls = segments_and_labels([E, W, W, B])
        base = [g.center_time for g in master.gaps]
        supports = [make_tracks(CAMERA_LEFT_UP_TOP,
                                sorted(base + [b + 0.2 for b in base]))]
        st = assemble(master, supports, cls)
        self.assertEqual(st.total_wagons, 2)
        self.assertEqual([w.global_id for w in st.wagons], ["GW_1", "GW_2"])

    def test_count_equals_master_wagon_count(self):
        master, cls = segments_and_labels([E, E, W, W, W, W, W, W, B])
        supports = [make_tracks(c, [g.center_time for g in master.gaps][:2])
                    for c in ALL_CAMERAS if c != CAMERA_RIGHT_UP]
        st = assemble(master, supports, cls)
        self.assertEqual(st.total_wagons, 6)
        self.assertEqual(st.master_wagon_count, 6)
        self.assertEqual(st.invariant_checks["master_wagon_count"], 6)
        self.assertTrue(st.invariant_checks["invariant_holds"])

    def test_support_observations_outside_the_wagon_region_are_excluded(self):
        master, cls = segments_and_labels([E, W, W, W, B])
        base = [g.center_time for g in master.gaps]
        sup = make_tracks(CAMERA_RIGHT_UP_TOP, base)
        region = ts.LocalWagonRegion(
            camera_id=CAMERA_RIGHT_UP_TOP, found=True,
            start_time=base[1] - 0.1, end_time=base[-1] + 0.1)
        st = gf.assemble_global_train_state_master_fixed(
            master_tracks=master, support_tracks=[sup],
            initial_classifications=cls, config=gf.FusionConfig(), verbose=False,
            wagon_regions={CAMERA_RIGHT_UP_TOP: region})
        self.assertEqual(st.total_wagons, 3, "region filtering must not alter the count")
        summary = st.support_alignment_summary[CAMERA_RIGHT_UP_TOP]
        self.assertGreaterEqual(summary["n_non_wagon_excluded"], 1)


# ===========================================================================
# top classification model
# ===========================================================================

class TestTopClassificationMapping(unittest.TestCase):
    def test_camera_to_classifier_mapping(self):
        m = ts.CAMERA_CLASSIFICATION_MODEL
        self.assertEqual(m[CAMERA_RIGHT_UP], ts.SIDE_CLASSIFICATION_MODEL)
        self.assertEqual(m[CAMERA_RIGHT_UP_TOP], ts.TOP_CLASSIFICATION_MODEL)
        self.assertEqual(m[CAMERA_LEFT_UP_TOP], ts.TOP_CLASSIFICATION_MODEL)
        self.assertEqual(m[CAMERA_LEFT_UP], ts.SIDE_CLASSIFICATION_MODEL)

    def test_mapping_is_built_from_real_names_not_indices(self):
        """Class IDs are never assumed: 0 is not 'wagon' by fiat."""
        lm = ts.build_label_mapping({0: "brakevan", 1: "engine", 2: "wagon"})
        self.assertEqual(lm.semantic_for("wagon"), W)
        self.assertEqual(lm.semantic_for("engine"), E)
        self.assertEqual(lm.semantic_for("brakevan"), B)
        # a differently ordered model must map identically
        lm2 = ts.build_label_mapping({0: "wagon", 1: "brakevan", 2: "engine"})
        self.assertEqual(lm.mapping, lm2.mapping)

    def test_unexpected_class_maps_to_unknown_never_wagon(self):
        lm = ts.build_label_mapping({0: "wagon", 1: "engine", 2: "sheep",
                                     3: "flying_saucer"})
        self.assertEqual(lm.semantic_for("sheep"), U)
        self.assertEqual(lm.semantic_for("flying_saucer"), U)
        self.assertNotEqual(lm.semantic_for("sheep"), W)
        self.assertEqual(sorted(lm.unmapped), ["flying_saucer", "sheep"])

    def test_background_style_classes_map_to_unknown(self):
        lm = ts.build_label_mapping({0: "empty_track", 1: "background",
                                     2: "other", 3: "unknown"})
        for name in ("empty_track", "background", "other", "unknown"):
            self.assertEqual(lm.semantic_for(name), U)
        self.assertEqual(lm.unmapped, [], "these are recognised, not unexpected")

    def test_brakevan_variants_are_not_read_as_wagons(self):
        lm = ts.build_label_mapping({0: "wagon_tail", 1: "guard_van",
                                     2: "brake-van", 3: "tail"})
        for name in ("wagon_tail", "guard_van", "brake-van", "tail"):
            self.assertEqual(lm.semantic_for(name), B, name)

    def test_engine_variants(self):
        lm = ts.build_label_mapping({0: "loco", 1: "locomotive", 2: "engine_head",
                                     3: "locono"})
        for name in lm.names.values():
            self.assertEqual(lm.semantic_for(name), E, name)

    def test_case_and_whitespace_insensitive(self):
        lm = ts.build_label_mapping({0: " Wagon ", 1: "ENGINE"})
        self.assertEqual(lm.semantic_for(" Wagon "), W)
        self.assertEqual(lm.semantic_for("engine"), E)

    def test_unknown_label_at_lookup_time_is_unknown(self):
        lm = ts.build_label_mapping({0: "wagon"})
        self.assertEqual(lm.semantic_for("something_else_entirely"), U)

    def test_mapping_serializes_for_the_json_report(self):
        lm = ts.build_label_mapping({0: "wagon", 1: "mystery"}, "models/top.pt")
        d = lm.to_dict()
        self.assertEqual(d["class_count"], 2)
        self.assertEqual(d["names"], {0: "wagon", 1: "mystery"})
        self.assertEqual(d["unmapped_classes"], ["mystery"])


class TestLocalWagonRegion(unittest.TestCase):
    def test_region_from_labels(self):
        segs = [(0, 99), (100, 199), (200, 299), (300, 399)]
        reg = ts.build_local_wagon_region(
            CAMERA_RIGHT_UP_TOP, segs, [E, W, W, B], FPS, verbose=False)
        self.assertTrue(reg.found)
        self.assertEqual(reg.start_frame, 100)
        self.assertEqual(reg.end_frame, 299)
        self.assertTrue(reg.contains_time(150 / FPS))
        self.assertFalse(reg.contains_time(50 / FPS))
        self.assertFalse(reg.contains_time(350 / FPS))

    def test_unknown_region_accepts_everything(self):
        """A missing classification must not silently discard evidence."""
        reg = ts.build_local_wagon_region(
            CAMERA_LEFT_UP_TOP, [(0, 99)], [E], FPS, verbose=False)
        self.assertFalse(reg.found)
        self.assertTrue(reg.contains_time(0.0))
        self.assertTrue(reg.contains_time(9999.0))

    def test_top_engine_and_brakevan_produce_no_gw_ids(self):
        """A top camera seeing an engine cannot mint a wagon id."""
        master, cls = segments_and_labels([E, W, W, B])
        sup = make_tracks(CAMERA_RIGHT_UP_TOP,
                          [g.center_time for g in master.gaps])
        region = ts.build_local_wagon_region(
            CAMERA_RIGHT_UP_TOP, [(0, 100), (101, 200)], [E, B], FPS, verbose=False)
        st = gf.assemble_global_train_state_master_fixed(
            master_tracks=master, support_tracks=[sup],
            initial_classifications=cls, config=gf.FusionConfig(), verbose=False,
            wagon_regions={CAMERA_RIGHT_UP_TOP: region})
        self.assertEqual(st.total_wagons, 2)
        self.assertEqual([w.global_id for w in st.wagons], ["GW_1", "GW_2"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
