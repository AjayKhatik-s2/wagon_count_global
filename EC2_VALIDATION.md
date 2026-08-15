# EC2 Validation Runbook — Global Train Inspection

**EC2 is the validation environment.** The local VS Code machine holds a single
development train and is not representative of production. Everything below has
been implemented and checked at code level locally; none of it has been validated
end to end, and the local train's numbers are **not** production validation.

---

## What was validated locally (code level only)

| Check | Result |
|---|---|
| `pytest -q` and `python -m unittest discover -s tests` | 365 passed, 2 skipped |
| Compile / import of every changed module | clean |
| All 14 `old_code` modules resolve to `old_code/` (nothing shadowed) | verified |
| Stage ordering: inspection after fusion, before JSON; load before damage | asserted statically |
| Roster guard present and in a `finally` | asserted statically |
| Renderer contains no `YOLO(` / `.predict(` / `load_yolo` | asserted statically |
| Bridge never assigns to `state.wagons` / `global_gaps` / `camera_offsets` | asserted by AST |
| Protected counting modules unchanged, and importing no feature code | verified |
| No local-train constants in shipped code | asserted by tokenizer |

The 2 skips are GPU-only and OCR-only paths, both expected on a CPU dev box.

---

## Required models

Place in `--models-dir` (default `./models`). Weights are never committed to Git.

| Role | Filename (aliases accepted) | Notes |
|---|---|---|
| Door | `door_state.pt` (or `door.pt`) | classes read at load time |
| Load | `load.pt` (or `loaded.pt`) | **supplied from S3**, complete-train bucket / `new_local` |
| Top damage | `top_damage.pt` (or `damage.pt`) | old_code called this `damage.pt` |

Download **once at startup**, never per frame or per wagon. A missing weight is
non-fatal: that feature reports `NO_DATA` for every wagon and the wagon count is
unaffected. Class names are read from each checkpoint and printed before any video
is decoded — do not assume class ids.

Observed classes (report anything different, do not assume these):
- `door_state.pt` → `closed_door`, `damage`, `open_door`, `partially_closed`
- `load.pt` → `Empty`, `Loaded`, `Unlabeled` (`Unlabeled` abstains, by design)
- `top_damage.pt` → `Floor__probable_damage`, `Floor_damage`, `Inner_wall_damage`

OCR is **out of scope**: no OCR model and no `easyocr` install is required.

---

## Run

```bash
python run_global_count.py \
    --inputs-dir  /data/train_XXXX \
    --models-dir  /opt/models \
    --output      /data/results/train_XXXX \
    --render-videos
```

Useful switches: `--no-inspection`, `--no-door`, `--no-load`, `--no-damage`,
`--keep-wagon-cache`, `--wagon-cache-every-nth`, `--wagon-cache-max-frames`.

---

## GPU strongly recommended — measured reason

Door inference in `old_code` is called with `half=True`, correct on GPU. On the
CPU dev box, measured with the real weight and a real frame:

```
half=False       782 – 1598 ms / frame
half=True     103487 – 104035 ms / frame     (~130x slower, no warm-up recovery)
```

A CPU-only shim forces float32 **only when CUDA is unavailable**, so a GPU run
keeps old behaviour bit for bit. On GPU, confirm the log does **not** print
`[CPU] half precision disabled`.

Cost driver on CPU is per-frame inference (~0.8–1.6 s). One wagon with full
coverage across four cameras measured ~4.5 minutes, so a 57-wagon train is hours
on CPU and must not be the production path.

---

## Frame-cache density is a CORRECTNESS setting, not a cost dial

`--wagon-cache-every-nth` defaults to **1**. Raising it breaks old_code's
trackers, which is silent rather than loud:

- `DoorTracker.max_center_distance = 150 px`; a door crosses at ~28 px/frame, so a
  stride of 6 moves it ~170 px between sightings — beyond the gate.
- Every detection then opens a new track, `n_init = 3` is never reached, and a
  wagon with a plainly visible 0.945 PARTIAL door reports `CLOSED (0.00)`.

Safe ceiling is ~5 at the speed measured here; **re-measure against production
train speed before raising it.** Similarly `--wagon-cache-max-frames` (default
150) is a runaway guard only: when it binds, the window is recorded as
`truncated` because partial coverage must not look like a complete inspection.

---

## What EC2 must validate

### 1. Counting must be untouched
- [ ] `MASTER == GLOBAL` holds; `global_gap_count == right_up_final_gap_count`
- [ ] `inspection.roster_unchanged == true` in `global_train_state.json`
- [ ] Wagon count, GW ids, boundaries and classifications identical with and
      without `--no-inspection`
- [ ] The run does **not** raise `RosterMutatedError`

### 2. Four-camera inference
- [ ] All three features complete on all applicable cameras
- [ ] Door: RIGHT_UP → right door, LEFT_UP → left door (camera authority)
- [ ] Load: RIGHT_UP_TOP authoritative, LEFT_UP_TOP fallback
- [ ] Damage: `floor_damage` dropped on wagons whose load is `LOADED`
- [ ] Report any `load.pt` class that could not be mapped to LOADED/EMPTY

### 3. GW association
- [ ] Every finding carries a GW id, or an explicit `UNRESOLVED` / `NOT_VISIBLE`
- [ ] An unresolved camera offset yields **no** cached frames and **no** wagon
      attribution — never a guessed one
- [ ] The same physical wagon seen by several cameras maps to ONE GW id

### 4. Reports and videos
- [ ] `reports/combined_train_report.json` + `.pdf`
- [ ] `reports/camera_report_{RIGHT_UP,LEFT_UP,RIGHT_UP_TOP,LEFT_UP_TOP}.pdf`
- [ ] Every global wagon appears, including those with no findings
- [ ] `NO_DATA` / `NOT_VISIBLE` / `UNRESOLVED` never rendered as
      `CLOSED` / `EMPTY` / `OK`
- [ ] Processed videos show GW id, door state, load state and damage boxes
- [ ] PDF, JSON and video agree — all three read the same persisted record

### 5. Memory and performance (previous OOM on this box)
- [ ] Peak RSS across the whole run; cameras are processed sequentially
- [ ] `wagon_cache/` disk peak, and that it is cleared unless
      `--keep-wagon-cache`
- [ ] Per-stage timings from the `timings` block

### 6. Multiple trains, sequentially, in ONE process
- [ ] Run ≥3 trains back to back
- [ ] Train N's GW_1 findings never appear on train N+1's GW_1
- [ ] `wagon_cache/` is cleared between trains
- [ ] Differing wagon counts, camera offsets, fps and resolutions all handled

### 7. Parity against the old system
- [ ] Run the OLD system on the same train and diff door / load / damage verdicts

This last item is the one thing that cannot be asserted from the port itself.
Locally only *algorithmic* identity was established — old_code's own trackers,
thresholds, FSM and voting are the code that runs. **Numerical parity is unproven
until an old-system reference run exists for the same train.**

---

## Known deviations from `old_code`

| Old behaviour | New behaviour | Why | Can it change results? |
|---|---|---|---|
| Door inference `half=True` | float32 when no GPU | ~130x slower on CPU; a hang, not a slow run | Only fp16 rounding, toward more precision. GPU path untouched. |
| Load `max_frames=0` | `max_frames=None` | `0` made `iter_wagon_frames` subsample to **zero** frames, so load could never vote | Only by letting the feature run; the 0.35 rule and voting are untouched |
| `core.constants`, `core.global_state_loader`, `core.unified_wagon_state` shipped | reconstructed | not included with `old_code` | Values marked RECOVERED / INFERRED in `core/constants.py`; `LOAD_LABEL_TO_STATE` is resolved from the real model at runtime |
| Camera-local wagon identity | global `GW_n` | the whole point of the port | Identity only |
| OCR present | removed | out of scope by request | OCR column always `-`, `ocr_captured` always 0 — "not attempted", not "failed" |

### Latent `old_code` bug, deliberately NOT changed

`old_code/door/processor.py:144` calls
`illumination.process_frame(frame, frame_idx=fi)`, but the real signature takes
only `frame`. The call sits inside a bare `except`, so it always fails and
`quality` is always `1.0` — meaning the DoorTracker's quality-weighted vote is
effectively **unweighted** as shipped. Left as-is because it is the old
behaviour; flagged because fixing it would change door verdicts.

---
---

# Part 2 — Legacy Inspection Output Layer (dashboard JSON / evidence / PDF)

Covers the ported Train-Inspection-Engine output stack: `inspection/legacy/`
(vendored legacy modules), `inspection/global_bridge.py` (roster → legacy
DataFrames), `inspection/legacy_inspection.py` (orchestrator) and
`inspection/legacy_render.py` (renderers).

**Nothing here can change the wagon count.** The current global counting
pipeline remains the sole authority for wagon identity and count; this layer
consumes the finalized roster and is verified against it.

---

## Model authority (explicit)

| Task | Model | Authority | Replaces |
|---|---|---|---|
| Side door state **and** side damage | `door_state.pt` | **AUTHORITATIVE** | legacy `V4_side_damage.pt` |
| Top damage | `top_damage.pt` | **AUTHORITATIVE** | legacy `V4_top_damage.pt` |
| Load (top cameras) | `load.pt` | **AUTHORITATIVE** | — |
| Wagon / loco numbers | none | **DISABLED** | OCR out of scope |

`door_state.pt` exposes exactly the legacy side model's four classes —
`closed_door`, `damage`, `open_door`, `partially_closed` — so it is a drop-in
replacement and **one model serves the whole side task in one pass**. There is
no second side model, so no side detection is counted twice. The `damage` class
stays a damage finding: `door_status` is voted only over the three door classes.

Class names are read from each checkpoint at startup and checked against the
expected set. A model whose classes do not match the flavour it was wired to is
**disabled and reported**, never run — an unmatched class set would otherwise
silently report a clean train.

`--door-source` selects which door implementation runs. `legacy` (default) uses
the ported `DamageDetector`; `old_code` uses the `DoorTracker` path. **Exactly
one runs** — both consume `door_state.pt` on the same frames, so enabling both
would load the model twice and produce two verdicts for one question.

### Verify the models before a run

```bash
python - <<'PY'
from ultralytics import YOLO
for f in ("door_state.pt", "top_damage.pt", "load.pt"):
    m = YOLO(f"/opt/models/{f}")
    print(f"{f:16} task={m.task:9} names={m.names}")
PY
```

Expected (report any difference rather than assuming — nothing resolves a class
by numeric id):

```
door_state.pt    detect    {0:'closed_door',1:'damage',2:'open_door',3:'partially_closed'}
top_damage.pt    detect    {0:'Floor__probable_damage',1:'Floor_damage',2:'Inner_wall_damage'}
load.pt          classify  {0:'Empty',1:'Loaded',2:'Unlabeled'}
```

Models may be given as `s3://bucket/key`. They are downloaded once, cached by
ETag under `~/.cache/train-inspection-engine/models/`, and re-downloaded only
when the S3 object changes:

```bash
--side-model s3://wagon-eye-models/door_state.pt \
--top-model  s3://wagon-eye-models/top_damage.pt
```

---

## Run under tmux

```bash
tmux new -s inspect
cd /opt/wagon_count

python run_global_count.py \
    --inputs-dir /data/train_XXXX \
    --models-dir /opt/models \
    --output     /data/results/train_XXXX \
    --door-source legacy \
    --artifact-bucket biro-wagon-report-biro-copy \
    --aws-region ap-south-1 \
    --render-videos \
    2>&1 | tee /data/results/train_XXXX/run.log

# detach: Ctrl-b d       reattach: tmux attach -t inspect
```

Add `--upload-artifacts` **only when you intend to publish**. Without it the
artifacts are written to `<output>/inspection/artifacts/` with the identical
layout and filenames — that is the correct first EC2 run.

Add `--annotated-videos` for the legacy detected-videos (expensive: re-encodes
every camera; needs `ffmpeg` on PATH).

### Monitor from a second pane

```bash
tmux new-window -t inspect
watch -n 5 'free -g; echo; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv; echo; du -sh /data/results/train_XXXX/wagon_cache 2>/dev/null'

# peak RSS of the run
pidstat -r -p $(pgrep -f run_global_count.py) 5
```

**CPU/GPU safety.** No code path forces `half=True`. On CPU the models run
float32; on GPU the approved behaviour is unchanged. One model handle per weight
file is shared across both cameras of a flavour, frames are read one at a time
from the on-disk cache and released immediately, and no list of full-resolution
images is accumulated — this is the shape that avoids the earlier OOM.

---

## Verification checklist

### 1. The global wagon count is the only count

```bash
cd /data/results/train_XXXX
python - <<'PY'
import glob, json, os
state = json.load(open("global_train_state.json"))
n = state["total_wagons"]
print("GLOBAL WAGON COUNT:", n)
ok = True
for path in sorted(glob.glob("inspection/*/inspection_data.json")):
    d = json.load(open(path))["inspection_data"]
    counts = [s["wagon_count"] for s in d["wagon_segments"]]
    good = (
        d["global_wagon_count"] == n
        and d["total_wagons"] == len(d["wagon_segments"])
        and d["total_wagons"] + d.get("num_engines", 0) + d.get("num_brakevans", 0) == n
        and counts == sorted(counts)
        and len(set(counts)) == len(counts)
    )
    ok &= good
    print(f"  {os.path.basename(os.path.dirname(path)):10} "
          f"total_wagons={d['total_wagons']:4} segments={len(d['wagon_segments']):4} "
          f"engines={d.get('num_engines')} brakevans={d.get('num_brakevans')} "
          f"{'OK' if good else 'MISMATCH'}")
print("ALL CAMERAS CONSISTENT:", ok)
PY
```

- [ ] `global_wagon_count` equals `global_train_state.json`'s `total_wagons` in **all four** JSONs
- [ ] `total_wagons + num_engines + num_brakevans` equals the global count in each JSON
- [ ] `wagon_count` values are monotonic, contiguous and unique in each camera
- [ ] No camera reports a different wagon population from any other
- [ ] `inspection.legacy_outputs.roster_unchanged` is `true`
- [ ] Re-run with `--no-legacy-inspection`: wagon count, GW ids and boundaries identical

### 2. Dashboard JSON compatibility

```bash
python - <<'PY'
import json
side = json.load(open("inspection/right_up/inspection_data.json"))["inspection_data"]
req = ["raw_video_name","identified_by","upload_timestamp","direction","rake_status",
       "total_wagons","doors_open","doors_partially_closed","doors_closed",
       "damaged_wagons","num_engines","total_loco_frames","total_problem_frames",
       "problem_frames_by_type","wagon_number_results","loco_number_results",
       "segment_type_map","wagon_segments","loco_frames","problem_frames",
       "damage_model_active"]
print("side missing :", [k for k in req if k not in side] or "NONE")
print("side ocr {}  :", side["wagon_number_results"] == {} and side["loco_number_results"] == {})
print("side seg keys:", sorted(side["wagon_segments"][0]))

top = json.load(open("inspection/right_top/inspection_data.json"))["inspection_data"]
req = ["rake_status","total_wagons","wagons_loaded","wagons_empty","damaged_wagons",
       "probable_damage_wagons","floor_dmg_wagons","inner_wall_dmg_wagons",
       "floor_dmg_probable_wagons","num_engines","num_brakevans","total_loco_frames",
       "total_problem_frames","problem_frames_by_type","wagon_number_results",
       "loco_number_results","segment_type_map","wagon_segments","loco_frames",
       "problem_frames","damage_model_active"]
print("top missing  :", [k for k in req if k not in top] or "NONE")
PY
```

- [ ] No legacy field missing from either flavour
- [ ] `wagon_number_results` and `loco_number_results` are `{}` — present and empty, **not** removed
- [ ] `is_valid_wagon_id` present on every wagon segment, and `false`
- [ ] Envelope is `{camera_id, version, inspection_data}` with `version == "v4"`
- [ ] `camera_id` matches the legacy folder names, e.g. `CCTV_HZBN_DHN_2_RIGHT_UP`
- [ ] The only new keys are `global_wagon_id`, `inspection_status`,
      `counting_source`, `global_wagon_count`, `ocr_enabled`, `camera_role`
- [ ] **Feed one JSON to the real dashboard and confirm it ingests unchanged**

### 3. Feature correctness

- [ ] `door_status` is one of `open` / `partially_closed` / `closed`, and
      `door_close_detected` / `door_partial_detected` are independent booleans
- [ ] A one-frame flicker does **not** flip a wagon (`min_band_frames = 3`)
- [ ] Side `damage` appears as `damage_detected`, never as a door state
- [ ] `Floor_damage` / `Inner_wall_damage` / `Floor__probable_damage` map to
      `floor_dmg` / `inner_wall_dmg` / `floor_dmg_probable`
- [ ] Floor damage suppressed on `wagon_loaded`; inner-wall damage **not** suppressed
- [ ] `wagons_loaded + wagons_empty == total_wagons` on top cameras
- [ ] Engines/brakevans counted in `num_engines` / `num_brakevans`, absent from
      `wagon_segments`, and a damage hit on one does **not** raise `damaged_wagons`
- [ ] Any `load.pt` class that could not be mapped is reported (`Unlabeled` abstains)

### 4. Evidence and artifacts

```bash
find inspection/artifacts -type f | head -20
find inspection/artifacts -name '*.jpg' | wc -l
```

- [ ] `wagon_frames/` holds up to 3 frames per wagon at 25% / 55% / 80%, named
      `start` / `mid1` / `end`
- [ ] Side filenames `w{n}_frame_{nnnnnn}.jpg`; top filenames
      `{seg_type}_{nnn}_frame_{nnnnnn}_{pos}.jpg`
- [ ] `problem_frames/` holds **one** file per evidence entry — annotated when one
      exists, otherwise raw, never both (`is_annotated` says which)
- [ ] Every `problem_frames[i]` carries `global_wagon_id`, `bounding_box`,
      `s3_key`, `s3_url`
- [ ] A wagon with no evidence has `wagon_frames: []`, not a fabricated frame

### 5. PDF and videos agree with the JSON

- [ ] `inspection/reports/combined_inspection_report.pdf` exists
- [ ] Its per-wagon table has **exactly** one row per global wagon — count the rows
- [ ] `inspection/<camera>/<camera>_report.pdf` exists for all four cameras
- [ ] Wagons with no findings appear with explicit `OK` / `NOT VISIBLE`, and
      `NOT_VISIBLE` / `UNRESOLVED` are never rendered as a clean finding
- [ ] Processed videos label wagons with the same GW ids as the JSON
- [ ] No inference in the log between the inspection stage completing and the PDFs
      being written

### 6. Memory, performance, repeatability

- [ ] Peak RSS recorded; no growth across cameras (they run sequentially)
- [ ] `wagon_cache/` peak size recorded, and cleared unless `--keep-wagon-cache`
- [ ] Per-camera timings from `inspection.legacy_outputs.timings`
- [ ] Run at least 3 trains back to back in one process: train N's evidence, model
      state, offsets and cache never appear on train N+1
- [ ] The same train run twice produces a byte-identical `inspection_data.json`

### 7. S3 upload — only after everything above passes

```bash
python run_global_count.py ... --artifact-bucket <bucket> --upload-artifacts
aws s3 ls s3://<bucket>/camera_CCTV_HZBN_DHN_2_RIGHT_UP/ --recursive | head
```

- [ ] Layout is `{bucket}/{camera_folder}/{YYYY-MM-DD_HH-MM-SS}/{wagon_frames,problem_frames}/...`
- [ ] `inspection_data.json` present at that prefix for each camera
- [ ] `s3_url` values in the JSON resolve

---

## Output layout

```
<output>/
  global_train_state.json                  wagon count -- the authority
  inspection/
    right_up/  left_up/  right_top/  left_top/
      inspection_data.json                 dashboard payload
      segments.csv  damage_results.csv
      problem_frames.csv  frame_detections.csv
      <name>_report.pdf                    legacy per-camera PDF
      annotated_problem_frames/
    artifacts/                             S3-identical local layout
    reports/combined_inspection_report.pdf
```

---

## Still required on EC2 — nothing below is proven yet

Local verification was **code level only**: unit tests, schema tests, static
assertions and a bounded synthetic run against the real weights. None of the
following has been observed on real footage, and this port must not be called
validated until it has:

1. A real four-camera end-to-end run completing.
2. The inspection JSON wagon count matching the global count on real footage.
3. Real detections on real wagons attributed to the correct GW id.
4. The dashboard ingesting the new JSON without modification.
5. Memory and runtime within budget for a full-length train.
6. Parity of door/damage verdicts against an old-system run on the same train.
