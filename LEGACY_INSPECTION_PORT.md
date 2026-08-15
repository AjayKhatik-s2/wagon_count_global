# Legacy Inspection Port — module map and contract

**One sentence:** the current global wagon-counting pipeline remains the sole
authority for wagon identity and count; the legacy Train-Inspection-Engine's
proven door / damage / evidence / JSON / PDF behaviour is reused unchanged and
attached to the finalized `GW_1 … GW_N` roster.

The legacy source added to this repository is
`rithish__code_1/CCTV-TrainVideo-ML-V2-wagon-Rithish/Train-Inspection-Engine/`
(abbreviated **TIE** below).

---

## 1. Module map

| Legacy module (TIE) | What it does | REUSED | MUST NOT be reused | New global adapter |
|---|---|---|---|---|
| `inspection/damage.py` → `DamageDetector` | top + side damage/door detection, band analysis, min-band flicker filter, door majority vote, engine/brakevan skip, loaded-floor suppression | **all of it, vendored** | — | fed a `segment_summary_df` synthesized from the roster |
| `inspection/damage.py` → `ProblemFrameExtractor` | annotated problem frames at each band's best frame | **vendored** | — | `wagon_id` is the global wagon index |
| `inspection/bands.py` | `analyze_detection_bands` groups per-frame detections into bands | **vendored** | `identify_bands` — gap banding, i.e. **counting** | none |
| `inspection/frame_positions.py` | span-relative loco frame selection | **vendored** | — | none |
| `inspection/segments.py` | `WagonSegmenter`, `classify_segment_type` — **camera-local wagon segmentation and numbering** | only its *column contract* | **all code — this is the old counting authority** | `global_bridge.build_segment_summary()` replaces it |
| `reporting/artifacts.py` | S3 layout, 25/55/80 evidence frames, problem-frame upload rule, `display_segment_type` | **vendored** | — | `wagon_count_map` carries global IDs; local sink for no-S3 runs |
| `reporting/json_builder.py` | **the canonical dashboard contract** | **vendored** | — | consumes the global `wagon_count_map` |
| `reporting/pdf_builder.py` | per-camera legacy PDF | **vendored** | — | fed persisted CSVs; no inference |
| `inspection/annotated_video.py` | gap + loco + damage overlays | **vendored** | — | fed persisted `frame_detections.csv` |
| `core/model_store.py` | `s3://` model resolution + ETag cache | **vendored** | — | wired into model discovery |
| `core/s3.py`, `core/video_io.py`, `utils/url_utils.py`, `utils/serialization.py` | plumbing | **vendored** | — | — |
| `inspection/ocr/*`, `core/rekognition.py` | wagon/loco number OCR | **nothing** | **all — hard disabled** | fields emitted as `{}` |
| `pipelines/base_pipeline.py`, `train_detection/*`, `combiner/pipeline.py` | old end-to-end **counting** | **nothing** | **all** | current global pipeline |
| `combiner/pdf.py` | combined report layout | layout + status vocabulary | `max(wagon_count)` row sizing (camera-derived) | `legacy_render.combined_wagon_rows()` sizes from the roster |

### New files

| File | Role |
|---|---|
| `inspection/legacy/` | vendored TIE modules — the behavioural source of truth |
| `inspection/global_bridge.py` | roster → legacy DataFrames; the whole adapter |
| `inspection/legacy_inspection.py` | orchestrator: models → detect → evidence → JSON |
| `inspection/legacy_render.py` | PDFs + annotated videos, from persisted state only |
| `tests/test_legacy_inspection_port.py` | 100 tests over the above |

---

## 2. Pipeline order

```
FOUR VIDEOS
  → gap detection → tracking → fragment stitching → validation
  → master sequence → camera synchronization → global fusion       PROTECTED
  → FINAL GLOBAL ROSTER  GW_1 … GW_N                               PROTECTED
  ────────────────────────────────────────────────────────────────
  → inspection windows        wagon_cache/GW_n/CAMERA/frame_*.jpg
  → LOAD                      old_code load processor (0.35 rule)
  → DOOR + DAMAGE             legacy DamageDetector, per camera
  → global wagon association  by construction, then asserted
  → evidence / snapshots      legacy ArtifactPublisher (25/55/80)
  → PERSISTED inspection JSON legacy json_builder
  → combined report + PDFs    persisted state only, no inference
  → processed / annotated videos
  → dashboard JSON
```

Inspection never runs before roster finalization and never feeds back into
counting.

---

## 3. Wagon identity

The legacy schema carried two independent camera-local numbers: `segment_id`
(the camera's segment ordinal) and `wagon_count` (a separate running counter
that skipped non-wagons). Both are pinned to the global index here:

```
segment_id == wagon_count == GlobalWagon.wagon_index      for every camera
```

so `GW_17` is `wagon_count: 17` in all four JSON files. `assert_wagon_count_map_is_global()`
proves, per camera, that every emitted identity exists in the roster, that
nothing was renumbered, and that no wagon appears twice.

Engines and brakevans keep the legacy convention of `wagon_count: null`; they are
counted in `num_engines` / `num_brakevans` and excluded from `wagon_segments`.
Therefore:

```
total_wagons + num_engines + num_brakevans == global roster size
```

---

## 4. Model authority

| Task | Model | Classes (read at runtime) | Authority |
|---|---|---|---|
| Side door **and** side damage | `door_state.pt` | `closed_door`, `damage`, `open_door`, `partially_closed` | **authoritative — replaces legacy `V4_side_damage.pt`** |
| Top damage | `top_damage.pt` | `Floor__probable_damage`, `Floor_damage`, `Inner_wall_damage` | **authoritative** |
| Load | `load.pt` | `Empty`, `Loaded`, `Unlabeled` | **authoritative** |
| OCR | none | — | **disabled** |

`door_state.pt` exposes exactly the legacy side model's four classes, so it is a
drop-in replacement and one model serves the whole side task in a single pass.
There is no second side model, so no side detection is counted twice. The
`damage` class remains a damage finding — `door_status` is voted only over the
three door classes.

`--door-source` selects between the two legacy door implementations
(`legacy` = vendored `DamageDetector`, default; `old_code` = `DoorTracker`).
**Exactly one runs**, because both consume `door_state.pt` on the same frames.

`Unlabeled` maps to neither state and abstains, as required.

---

## 5. JSON compatibility

`inspection_data.json` is produced by the vendored `json_builder.py`, so the key
set, nesting, types and semantics are the legacy ones by construction. The only
new keys are additive:

| Key | Where | Why |
|---|---|---|
| `global_wagon_id` | `wagon_segments[i]`, `problem_frames[i]` | the `GW_n` identity, alongside the legacy numeric `wagon_count` |
| `inspection_status` | `wagon_segments[i]` | `INSPECTED` / `NO_DETECTION` / `NOT_VISIBLE` / `UNRESOLVED` — the legacy schema cannot distinguish "not seen" from "seen and clean" |
| `counting_source` | `inspection_data` | provenance marker |
| `global_wagon_count` | `inspection_data` | the roster size, for cross-camera verification |
| `ocr_enabled` | `inspection_data` | explicit `false` |
| `camera_role` | `inspection_data` | the current pipeline's camera id |

No legacy field was renamed, removed or retyped. `wagon_number_results` and
`loco_number_results` are emitted as `{}` and `is_valid_wagon_id` remains on
every wagon segment, so the dashboard's shape is unchanged with OCR off.

---

## 6. Protections

| Invariant | How it is enforced |
|---|---|
| Wagon count unchanged by inspection | roster SHA-256 before/after, re-verified in a `finally` |
| GW ids, boundaries, classifications unchanged | included in the same hash |
| Camera offsets unchanged | included in the same hash |
| `MASTER == GLOBAL` | included in the same hash |
| Every result references an existing wagon | `assert_wagon_count_map_is_global` |
| No wagon twice, no renumbering, monotonic + contiguous | same assertion + schema tests |
| Every wagon appears exactly once per camera JSON | count-consistency tests |
| PDF rows == JSON rows | `combined_wagon_rows` is built from the roster |
| Renderers run no model | AST assertion over `legacy_render.py` |
| No OCR call | source scan + vendoring exclusion |
| No cross-train state reuse | `reset_inspection_state()` at run start; cache cleared between trains |

---

## 7. Deviations from the vendored legacy source

Only two, both documented at the site:

1. **Import paths** — `from ..core.s3` → `from .s3`, because the files now live
   in one flat package. No symbol renamed, added or removed.
2. **`damage.py` `tqdm` guard** — a two-line `ImportError` fallback around a
   progress-bar import this project does not depend on. Changes what is
   printed, never what is computed.

`tests/test_legacy_inspection_port.py::TestLegacyProvenance` asserts the
remaining vendored files are byte-identical to the legacy source, so drift is
caught rather than assumed.

---

## 8. Not yet validated

Everything above is verified at code level only — unit tests, schema tests,
static assertions, and a bounded synthetic run against the real weights.
**A real four-camera EC2 run is still required** before this is called done.
See `EC2_VALIDATION.md`, Part 2.
