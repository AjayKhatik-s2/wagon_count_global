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
