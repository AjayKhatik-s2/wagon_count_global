"""STATIC checks on how inspection is wired into the pipeline.

These read source and imports; they execute no video, no model and no pipeline.
They exist because the port's safety properties are ORDERING properties, and
ordering is exactly what a refactor breaks silently:

  * inspection must run AFTER the global wagon roster is final,
  * load must run BEFORE damage (damage reads load's output),
  * the roster guard must be unconditional,
  * old_code must be the only implementation of the feature intelligence,
  * the renderer must consume persisted state rather than re-running inference.

Local scope by design: full four-camera, multi-train validation belongs on EC2.
"""

from __future__ import annotations

import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


class TestOldCodeIsTheOnlyImplementation(unittest.TestCase):
    def test_every_old_module_resolves_to_old_code(self):
        """A local file shadowing an old_code module would fork the source of truth."""
        import importlib
        for name in ("features._common", "features.door.processor",
                     "features.load.processor", "features.damage.processor",
                     "features.reporting.combined_train_report",
                     "features.inference_lib.door_tracker",
                     "features.inference_lib.damage_tracker",
                     "features.inference_lib.door_identity_merger"):
            mod = importlib.import_module(name)
            self.assertIn("old_code", (getattr(mod, "__file__", "") or ""),
                          f"{name} is NOT coming from old_code")

    def test_the_interim_duplicate_implementation_is_gone(self):
        """An earlier simplified detector/tracker was removed; it must not return."""
        for rel in ("inspection/models.py", "inspection/detection.py",
                    "inspection/tracking.py", "inspection/association.py",
                    "inspection/state.py", "inspection/evidence.py"):
            self.assertFalse(os.path.exists(os.path.join(ROOT, rel)),
                             f"{rel} is a second implementation of the feature "
                             f"intelligence and must not exist alongside old_code")

    def test_the_bridge_reimplements_no_tracking(self):
        """The bridge is orchestration; trackers/FSMs/voting live in old_code."""
        for rel in ("inspection/wagon_cache.py", "inspection/old_features.py",
                    "inspection/old_report.py"):
            tree = ast.parse(read(rel))
            names = {n.name.lower() for n in ast.walk(tree)
                     if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
            for banned in ("kalman", "hungarian", "statemachine", "majorityvote"):
                self.assertFalse(any(banned in n for n in names),
                                 f"{rel} appears to reimplement {banned}")


class TestPipelineOrdering(unittest.TestCase):
    def setUp(self):
        self.src = read("run_global_count.py")

    def test_inspection_runs_after_fusion_and_classification(self):
        """GW ids must be final before any feature sees a frame."""
        i_fuse = self.src.index("STEP 3")
        i_insp = self.src.index("STEPS 8-11")
        self.assertLess(i_fuse, i_insp,
                        "inspection must come after the fusion stage")

    def test_inspection_runs_before_the_json_is_written(self):
        i_insp = self.src.index("STEPS 8-11")
        i_json = self.src.index("STEP 4 -- write JSON")
        self.assertLess(i_insp, i_json,
                        "inspection results must reach global_train_state.json")

    def test_roster_hash_is_taken_before_and_verified_after(self):
        self.assertIn("core_state.roster_hash(state)", self.src)
        self.assertIn("core_state.assert_roster_unchanged(state, _roster_before)",
                      self.src)

    def test_the_roster_guard_is_unconditional(self):
        """It must sit in a finally, so a raising feature cannot skip it."""
        idx = self.src.index("core_state.assert_roster_unchanged(state, _roster_before)")
        preceding = self.src[:idx]
        self.assertIn("finally:", preceding.rsplit("try:", 1)[-1],
                      "the roster check must run even if a feature raises")

    def test_load_runs_before_damage(self):
        """damage reads the sibling load JSON to drop floor_damage on LOADED."""
        src = read("inspection/old_features.py")
        self.assertLess(src.index("from features.load import processor"),
                        src.index("from features.damage import processor"),
                        "load must run before damage or the coupling is lost")

    def test_inspection_failure_cannot_lose_the_wagon_count(self):
        block = self.src[self.src.index("STEPS 8-11"):
                         self.src.index("STEP 4 -- write JSON")]
        self.assertIn("except Exception", block)
        self.assertIn("the wagon count above is unaffected", block)

    def test_models_are_resolved_before_any_decoding(self):
        """A missing weight must be visible before the expensive stages."""
        i_models = self.src.index("oldf.discover_feature_models(args.models_dir")
        i_track = self.src.index("STEP 1  Per-camera gap tracking")
        self.assertLess(i_models, i_track,
                        "model availability must be reported before tracking")


class TestRendererConsumesPersistedState(unittest.TestCase):
    def test_renderer_takes_state_not_events(self):
        import inspect

        import video_segmenter as vs
        params = inspect.signature(vs.render_processed_video).parameters
        self.assertIn("inspection_state", params)
        self.assertNotIn("inspection_events", params)

    def test_renderer_runs_no_inference(self):
        """Rendering must not load a model or call predict."""
        src = read("video_segmenter.py")
        for banned in ("YOLO(", ".predict(", "load_yolo", "ultralytics"):
            self.assertNotIn(banned, src,
                             f"the renderer must not {banned!r} -- it consumes "
                             f"persisted inspection state")

    def test_pipeline_passes_persisted_state_to_the_renderer(self):
        self.assertIn("inspection_state=(state.inspection or {})", self.src
                      if hasattr(self, "src") else read("run_global_count.py"))


class TestProtectedCountingModules(unittest.TestCase):
    PROTECTED = ("tracker_engine.py", "global_alignment.py", "gap_validation.py",
                 "fragment_stitching.py", "global_fusion.py", "train_structure.py")

    def test_protected_modules_do_not_import_inspection(self):
        """The counting core must not depend on a downstream feature."""
        for rel in self.PROTECTED:
            src = read(rel)
            for banned in ("import inspection", "from inspection",
                           "from features", "import features",
                           "from core import"):
                self.assertNotIn(banned, src,
                                 f"{rel} must not depend on {banned!r}")

    def test_protected_modules_exist_and_parse(self):
        for rel in self.PROTECTED:
            ast.parse(read(rel))

    def test_inspection_never_writes_to_the_wagon_roster(self):
        """No assignment to state.wagons / total_wagons anywhere in the bridge."""
        for rel in ("inspection/wagon_cache.py", "inspection/old_features.py",
                    "inspection/old_report.py"):
            tree = ast.parse(read(rel))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AugAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for tgt in targets:
                    if isinstance(tgt, ast.Attribute) and tgt.attr in (
                            "wagons", "total_wagons", "global_gaps",
                            "camera_offsets", "invariant_checks"):
                        base = getattr(tgt.value, "id", "")
                        self.assertNotEqual(
                            base, "state",
                            f"{rel} assigns to state.{tgt.attr} -- inspection must "
                            f"never write the counting result")


class TestNoTrainSpecificAssumptions(unittest.TestCase):
    def test_no_local_train_measurements_in_shipped_code(self):
        import io
        import tokenize
        banned = {"57", "58", "3555", "3425", "0.8833", "2.6833", "654.76"}
        for rel in ("inspection/wagon_cache.py", "inspection/old_features.py",
                    "inspection/old_report.py", "core/constants.py",
                    "core/unified_wagon_state.py", "core/global_state_loader.py"):
            with open(os.path.join(ROOT, rel), "rb") as fh:
                nums = {t.string for t in
                        tokenize.tokenize(io.BytesIO(fh.read()).readline)
                        if t.type == tokenize.NUMBER}
            leaked = nums & banned
            self.assertFalse(leaked, f"{rel} contains local-train values {leaked}")

    def test_camera_lists_are_not_hardcoded_per_train(self):
        from core import constants as C
        self.assertEqual(len(C.ALL_CAMERAS), 4)
        self.assertEqual(set(C.SIDE_CAMERAS) | set(C.TOP_CAMERAS),
                         set(C.ALL_CAMERAS))

    def test_ocr_stays_out_of_scope(self):
        from core import constants as C
        self.assertFalse(C.OCR_ENABLED)
        src = read("inspection/old_features.py")
        self.assertNotIn("from features.ocr import", src,
                         "OCR is out of scope and must not be invoked")


if __name__ == "__main__":
    unittest.main(verbosity=2)
