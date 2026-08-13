"""Inspection appears in the PDF and in the overlay videos, from ONE record.

The report and the renderer must both read `state.inspection` and neither may
re-derive a finding, so that the JSON, the PDF and the video can never disagree
about what was found on which wagon.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evidence_report as er
import video_segmenter as vs
from global_train_state import GlobalTrainState, GlobalWagon, SegmentClass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def event(gid="GW_2", *, camera="RIGHT_UP", role="door",
          cls="open_door", conf=0.91, confirmed=True, start=300, end=328,
          reason=""):
    return {
        "event_id": f"{role.upper()}_{camera}_0001", "role": role,
        "model": "models/door_state.pt", "model_class_id": 2,
        "model_class_name": cls, "camera_id": camera, "track_id": 1,
        "start_frame": start, "end_frame": end,
        "start_time_local": start / 15.0, "end_time_local": end / 15.0,
        "n_observations": end - start + 1,
        "peak_confidence": conf, "mean_confidence": conf - 0.1,
        "global_id": gid, "association_status": "RESOLVED",
        "association_method": "overlap:max",
        "association_detail": "0.98 inside GW_2",
        "peak_frame": (start + end) // 2, "peak_bbox": [100.0, 120.0, 260.0, 380.0],
        "displacement_px": 780.0, "confirmed": confirmed,
        "rejection_reason": reason,
        "evidence_frames": [{"camera_id": camera, "frame": (start + end) // 2,
                             "bbox": [100.0, 120.0, 260.0, 380.0],
                             "class_name": cls, "confidence": conf,
                             "selection": "peak", "available": True,
                             "image_path": None}],
    }


def state_with_inspection(n_wagons=4, events=None, rejected=None):
    st = GlobalTrainState(total_wagons=n_wagons, master_camera="RIGHT_UP",
                          master_fps=15.0, master_total_frames=3000)
    for i in range(1, n_wagons + 1):
        st.wagons.append(GlobalWagon(
            global_id=f"GW_{i}", wagon_index=i,
            start_frame_master=100 * i, end_frame_master=100 * (i + 1),
            start_time=10.0 * (i - 1), end_time=10.0 * i,
            classification=SegmentClass.WAGON))
    st.total_wagons = n_wagons
    evs = events if events is not None else [event()]
    rej = rejected or []
    st.inspection = {
        "enabled": True,
        "model_availability": {
            "door": {"role": "door", "status": "AVAILABLE", "task": "detect",
                     "class_names": {0: "closed_door", 1: "damage",
                                     2: "open_door", 3: "partially_closed"}},
            "top_damage": {"role": "top_damage", "status": "AVAILABLE",
                           "task": "detect",
                           "class_names": {0: "Floor__probable_damage",
                                           1: "Floor_damage",
                                           2: "Inner_wall_damage"}},
        },
        "summary": {"confirmed_door_events": len([e for e in evs
                                                  if e["role"] == "door"]),
                    "confirmed_damage_events": len([e for e in evs
                                                    if e["role"] == "top_damage"]),
                    "rejected_events": len(rej), "unresolved_associations": 0,
                    "wagons_with_door_finding": 1,
                    "wagons_with_damage_finding": 0,
                    "evidence_frames": sum(len(e["evidence_frames"]) for e in evs),
                    "association_status": "RESOLVED"},
        "wagons": {
            f"GW_{i}": {
                "global_id": f"GW_{i}", "classification": "WAGON",
                "door_state": ({"state": "open_door", "confidence": 0.91,
                                "n_events": 1} if i == 2 else
                               {"state": None, "confidence": 0.0, "n_events": 0}),
                "top_damage": {"state": None, "confidence": 0.0, "n_events": 0},
                "door_events": [], "damage_events": [],
                "camera_status": {
                    "RIGHT_UP": {"door": "CONFIRMED" if i == 2 else "NO_DETECTION"},
                    "LEFT_UP_TOP": {"top_damage": "NOT_VISIBLE"}},
            } for i in range(1, n_wagons + 1)},
        "events": evs, "rejected_events": rej,
        "per_camera": {"RIGHT_UP": {"door": {"raw_detections": 812, "tracks": 40,
                                             "confirmed_tracks": 1,
                                             "rejected_tracks": 39,
                                             "seconds": 61.2}}},
        "timings": {"door_seconds": 61.2}, "warnings": [],
    }
    return st


# ===========================================================================
# 21-23  the PDF carries the inspection sections
# ===========================================================================

class TestPdfSections(unittest.TestCase):
    def _build(self, st):
        b = er._ReportBuilder(dpi=70)          # small: fast, still real rendering
        b.add_inspection_sections(st)
        return b

    def test_inspection_sections_add_pages(self):
        b = self._build(state_with_inspection())
        self.assertGreaterEqual(len(b.pages), 3,
                                "expected an overview, a wagon table and "
                                "diagnostics")

    def test_absent_inspection_still_renders_a_page(self):
        st = state_with_inspection()
        st.inspection = {}
        b = self._build(st)
        self.assertEqual(len(b.pages), 1,
                         "a run without inspection must still say so, not crash")

    def test_every_wagon_appears_even_without_findings(self):
        """Requirement: do not omit wagons with no detections."""
        st = state_with_inspection(n_wagons=7)
        b = er._ReportBuilder(dpi=70)
        rows = []
        real_table = b._table_block

        def spy(p, x, y, w, title, headers, table_rows, max_rows=None):
            rows.extend(table_rows)
            return real_table(p, x, y, w, title, headers, table_rows, max_rows)

        b._table_block = spy
        b.add_inspection_sections(st)
        gw_rows = [r for r in rows if str(r[0]).startswith("GW_")]
        self.assertEqual(len(gw_rows), 7)
        self.assertTrue(any(r[2] == "none" for r in gw_rows),
                        "wagons without a door finding must read 'none'")

    def test_long_train_paginates_the_wagon_table(self):
        b = self._build(state_with_inspection(n_wagons=80))
        self.assertGreaterEqual(len(b.pages), 4)

    def test_rejected_candidates_are_shown_not_hidden(self):
        rej = [event(gid=None, confirmed=False, conf=0.30,
                     reason="peak confidence 0.30 below 0.6")]
        rows = []
        b = er._ReportBuilder(dpi=70)
        real_table = b._table_block

        def spy(p, x, y, w, title, headers, table_rows, max_rows=None):
            rows.append((title, table_rows))
            return real_table(p, x, y, w, title, headers, table_rows, max_rows)

        b._table_block = spy
        b.add_inspection_sections(state_with_inspection(rejected=rej))
        titles = " ".join(t for t, _ in rows)
        self.assertIn("REJECTED", titles)

    def test_model_class_names_are_reported_verbatim(self):
        """The report must show the model's own labels, never invented ones."""
        rows = []
        b = er._ReportBuilder(dpi=70)
        real_table = b._table_block

        def spy(p, x, y, w, title, headers, table_rows, max_rows=None):
            rows.extend(table_rows)
            return real_table(p, x, y, w, title, headers, table_rows, max_rows)

        b._table_block = spy
        b.add_inspection_sections(state_with_inspection())
        blob = " | ".join(" ".join(str(c) for c in r) for r in rows)
        for name in ("closed_door", "partially_closed", "Floor__probable_damage",
                     "Inner_wall_damage"):
            self.assertIn(name, blob, f"{name} must appear verbatim")

    def test_pdf_section_does_not_mutate_the_state(self):
        st = state_with_inspection()
        before = st.to_dict()
        self._build(st)
        self.assertEqual(st.to_dict(), before)


# ===========================================================================
# 20  the overlay renderer consumes the same record
# ===========================================================================

class TestVideoOverlays(unittest.TestCase):
    def test_renderer_accepts_inspection_events(self):
        import inspect as _inspect
        sig = _inspect.signature(vs.render_processed_video)
        self.assertIn("inspection_events", sig.parameters)
        self.assertIsNone(sig.parameters["inspection_events"].default)

    def test_overlay_colours_distinguish_door_from_damage(self):
        self.assertNotEqual(vs._DOOR_BOX_COLOR, vs._DAMAGE_BOX_COLOR)

    def test_renderer_signature_keeps_the_counting_parameters(self):
        """Overlays are additive: nothing the count relies on may disappear."""
        import inspect as _inspect
        params = _inspect.signature(vs.render_processed_video).parameters
        for name in ("local_tracks", "state", "output_path", "time_offset",
                     "drop_out_of_range", "non_wagon_regions"):
            self.assertIn(name, params)

    def test_state_json_carries_inspection_for_the_renderer(self):
        """The renderer reads state.inspection['events']; it must be serialized."""
        st = state_with_inspection()
        d = st.to_dict()
        self.assertIn("inspection", d)
        self.assertTrue(d["inspection"]["events"])
        ev = d["inspection"]["events"][0]
        for key in ("camera_id", "start_frame", "end_frame", "peak_bbox",
                    "model_class_name", "peak_confidence", "global_id", "role"):
            self.assertIn(key, ev, f"the overlay needs {key}")

    def test_inspection_block_is_additive_to_the_json(self):
        """Every pre-existing top-level field must survive."""
        plain = GlobalTrainState(total_wagons=0, master_camera="RIGHT_UP",
                                 master_fps=15.0).to_dict()
        withi = state_with_inspection().to_dict()
        for key in plain:
            self.assertIn(key, withi, f"{key} disappeared from the JSON")

    def test_empty_inspection_serializes_as_empty_not_missing(self):
        d = GlobalTrainState(total_wagons=0, master_camera="RIGHT_UP",
                                 master_fps=15.0).to_dict()
        self.assertEqual(d["inspection"], {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
