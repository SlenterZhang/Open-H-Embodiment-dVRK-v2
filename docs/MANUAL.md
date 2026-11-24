# dVRK Data Collection Manual

This manual walks through a complete data collection: preparing the machine,
recording demonstrations on the dVRK, converting them to a LeRobot dataset,
and using the dataset (including depth and point clouds). It is written as a
procedure; for every option and file field see the [README](../README.md), and
for the internal design see [ARCHITECTURE.md](../ARCHITECTURE.md).

**Contents**

1. [Overview](#1-overview)
2. [One-time setup](#2-one-time-setup)
3. [Recording day checklist](#3-recording-day-checklist)
4. [Recording, step by step](#4-recording-step-by-step)
5. [After recording](#5-after-recording)
6. [Converting to LeRobot](#6-converting-to-lerobot)
7. [Using the dataset](#7-using-the-dataset)
8. [Point clouds](#8-point-clouds)
9. [Memory usage](#9-memory-usage)
10. [Timestamps and clocks](#10-timestamps-and-clocks)
11. [Troubleshooting](#11-troubleshooting)
12. [Quick reference](#12-quick-reference)

---

## 1. Overview

The pipeline has two stages. **Record** runs live next to the robot and writes
raw files. **Convert** runs offline, as often as you like, and turns raw files
into a LeRobot dataset.

```text
 dVRK PSMs (ROS 2, ~100 Hz) ───┐
 endoscope left/right (30 fps) ┼─► record ─► data/raw/<run>/episode_N ─► convert ─► data/lerobot/<name> ─► verify
 D435i colour + depth (30 fps) ┘   (live)      raw master copy           (offline)   LeRobot v2.1 / v3.0
```

| Term | Meaning |
|---|---|
| **Run** | One `record` session: one folder `data/raw/<YYYYMMDD-HHMMSS>/`, one task text |
| **Episode** | One demonstration inside a run, started with `s` and ended with `e`: `episode_N/` |
| **Frameset** | One D435i colour image plus the depth image captured with it; they share an index |
| **Slot** | One row of the converted dataset, every `1 / fps` seconds (33.3 ms at 30 fps) |
| **Raw data** | Everything `record` writes. It is the master copy; `convert` only reads it |

What ends up in each dataset row:

| Feature | Content |
|---|---|
| `observation.state` | 28 values: per arm (PSM1, PSM2) 6 joints, jaw, pose `x y z qx qy qz qw` |
| `action` | The next row's state (placeholder, see the README's Limitations) |
| `observation.images.left` / `.right` | Endoscope frames, 1920x1080 |
| `observation.images.external` | D435i colour, 640x480 |
| `observation.depth.external` | D435i depth, 480x640, exact sensor integers |

---

## 2. One-time setup

1. Install ffmpeg and create the Python environment. It must see the system ROS
   packages, so always use the script:

   ```bash
   sudo apt install ffmpeg
   ./scripts/setup_env.sh
   ```

2. Connect the hardware: both endoscope capture cards and the D435i, on a USB 3
   port (USB 2 cannot carry colour and depth at 30 fps).

3. Check the camera device paths in [configs/record.yaml](../configs/record.yaml).
   Capture cards can change `/dev/videoN` after reconnecting:

   ```bash
   v4l2-ctl --list-devices
   ```

   Set `cameras.left.device` and `cameras.right.device`; swap them if the eyes
   appear reversed in the preview. With more than one RealSense connected, set
   `realsense.serial`.

4. Decide the D435i colour mode (`realsense.color_lossless`):

   | Value | Raw colour | Use when |
   |---|---|---|
   | `false` (default) | H.264, ~0.06 MB/s | Normal collection. The dataset is identical either way |
   | `true` | Lossless FFV1, ~5.5 MB/s | You need exact colours in point clouds rebuilt from raw data |

5. Leave `realsense.record_cloud: false`. Point clouds rebuild exactly from the
   depth (section 8) and cost ~100 MB/s when recorded live.

---

## 3. Recording day checklist

**Before**

- [ ] Free disk space: `df -h .` Budget ~1.2 GB per recorded minute, ~2.5 GB
      per minute including the converted dataset (section 9).
- [ ] dVRK console running, arms powered and homed.
- [ ] `./scripts/doctor.sh` prints `RESULT: PASS`.
- [ ] The preview shows real images from both eyes (a card can pass the check
      while showing "No Signal").
- [ ] The D435i sees the workspace and nothing blocks it.

**During**

- [ ] One run per task. Quit (`q`) and start a new run for a different task.
- [ ] End every demonstration with `e`, including failed ones; leave failed
      episodes out when converting.
- [ ] Note which episodes failed or were unusual (or use `--notes`).

**After**

- [ ] Check every episode status (section 5).
- [ ] Convert and verify (section 6).
- [ ] Copy the raw run to backup storage before deleting anything.

---

## 4. Recording, step by step

### 4.1 Start the robot

Start the dVRK console as usual, power on, home the arms and enable
teleoperation. The recorder only listens to ROS topics; it never moves the
robot.

### 4.2 Pre-flight check

```bash
./scripts/doctor.sh
```

Every line must be `PASS`. Two numbers deserve a look:

- `external RGB-D: 640x480 @ 30 fps, depth scale 0.001 m/unit` confirms the D435i.
- `first receive time minus header stamp` should be a few milliseconds. Seconds
  mean the robot PC and this PC disagree on time.

Optional, without the robot:
`uv run --no-sync python -m dvrk_lerobot.tools.camera_check` runs all cameras
for 15 s and reports frame rates and time offsets between cameras. Flash a
light or move something in view of every camera while it runs.

### 4.3 Choose how to teleoperate

| Setup | Command |
|---|---|
| dVRK master console (MTMs) | Teleoperate as usual, then `./scripts/record.sh --task "..."` |
| TouchX haptic device | `./scripts/touchx_record.sh --task "..."` (starts TouchX teleop in the background, then the recorder) |
| Viewing in an Apple Vision Pro | `./scripts/avp_view.sh` in another terminal, then open the printed `https://...:8443` address in Safari on the headset |

### 4.4 Record

```bash
./scripts/record.sh --task "needle pick"
```

The recorder waits until every robot topic has sent data, opens all cameras,
and shows the preview window. Then:

| Key | Action |
|---|---|
| `s` | Start an episode. The overlay turns red: `REC episode_N 12.3s 370` |
| `e` | End and save the episode (`status: ok`). The overlay returns to green `IDLE` |
| `q` / Ctrl+C | Quit. A running episode is saved as `aborted` |

Keys work in the preview window or in the terminal. While the preview window
has focus, **every key goes to the recorder**.

| Option | Use |
|---|---|
| `--notes "..."` | Free text stored in each episode's `meta.json` |
| `--resume-run data/raw/<run>` | Add episodes to an existing run (same task); numbering continues |
| `--no-realsense` | Stereo only for this run. Convert such runs separately with `--no-realsense` |

A good episode starts with the arm still, contains one complete attempt, and
ends shortly after the task is done. Keep the episodes of one dataset
consistent in how they start and end.

---

## 5. After recording

Each episode's `meta.json` has a `status`:

| Status | Meaning | Convert? |
|---|---|---|
| `ok` | Ended with `e` | Yes |
| `aborted` | Quit with `q` / Ctrl+C during the episode | Only with `--allow-incomplete`, after checking it |
| `failed` | A camera or encoder error stopped the episode | No |
| `recording` | The program was killed; the end may be cut | Only with `--allow-incomplete`, after checking it |

List them for a run:

```bash
grep -H '"status"' data/raw/<run>/episode_*/meta.json
```

Files in each episode:

```text
episode_N/
├── meta.json        task, status, camera settings, D435i intrinsics and depth scale
├── robot.jsonl      every robot message (receive time t_ns, ROS stamp stamp_ns)
└── video/
    ├── left.mkv, right.mkv              endoscope video (H.264)
    ├── external.mkv                     D435i colour (H.264, or FFV1 if lossless)
    ├── external_depth.mkv               D435i depth, lossless 16-bit FFV1
    └── *_timestamps.txt / *.csv         one line per frame, same order as the video
```

---

## 6. Converting to LeRobot

### 6.1 Run convert

```bash
# a whole run (episodes whose status is not ok are skipped)
./scripts/convert.sh --sessions data/raw/20261123-203823 --output data/lerobot/needle_pick

# several runs, or chosen episodes, into one dataset
./scripts/convert.sh --sessions data/raw/runA data/raw/runB/episode_3 --output data/lerobot/needle_pick_all

# for lerobot 0.4 or newer
./scripts/convert.sh --sessions data/raw/20261123-203823 --output data/lerobot/needle_pick --format v3.0
```

| Format | Read by |
|---|---|
| `v2.1` (default) | lerobot 0.3.x and older |
| `v3.0` | lerobot 0.4 and newer (matches lerobot 0.6.0) |

The output folder must not exist. If convert fails, the half-written output is
kept as `<output>.partial`; read the error, delete that folder, and run again.
Convert never changes the raw data, so you can convert the same run again with
other settings at any time.

### 6.2 What convert does

1. Builds a grid with one slot every 1/30 s, covering the time when every
   camera was recording.
2. Gives each slot, per camera, the first frame that arrived inside it. A slot
   with no new frame repeats the previous one, so no row is ever dropped and
   the episode keeps its real length. Colour and depth always come from the
   same frameset.
3. Attaches, per arm, the latest joint, jaw and pose samples received at or
   before the slot.
4. Writes `action[k] = state[k+1]` and `timestamp = k / 30`, re-encodes the
   chosen frames to MP4, writes the exact depth into Parquet, and computes
   statistics.

### 6.3 Read the alignment report

Convert prints a few `[align]` lines per episode:

```text
[align] 20261123-203823/episode_10: grid: 133 frames at 30 fps (4.43 s)
[align] 20261123-203823/episode_10: left: 134 frames captured, 0 slots repeated (0.0%), 1 frames unused, longest gap 47 ms
[align] 20261123-203823/episode_10: right: 133 frames captured, 23 slots repeated (17.3%), 15 frames unused, longest gap 43 ms
[align] 20261123-203823/episode_10: external: 133 frames captured, 0 slots repeated (0.0%), 0 frames unused, longest gap 36 ms
[align] 20261123-203823/episode_10: stereo skew: median 15.0 ms, worst 25.5 ms, 1 frames over 25 ms
[align] 20261123-203823/episode_10: RGB-D vs left: median 7.9 ms, worst 22.4 ms, 0 frames over 25 ms
[align] 20261123-203823/episode_10: robot sample age: median 7.4 ms, worst 11.0 ms, 0 frames over 25 ms
```

| Line | Healthy | If not |
|---|---|---|
| `slots repeated` | a few percent | A camera ran slow or bunched its frames; check the video for freezes during motion |
| `longest gap` | under ~70 ms | A camera stalled; discard the episode if it stalled during motion |
| `stereo skew` | median under ~17 ms | The two capture cards drift; see the README's stereo timing diagnostics |
| `RGB-D vs left` | median under ~17 ms | D435i bandwidth or USB problems |
| `robot sample age` | a few ms | Robot topics stalled; discard the episode if the state was held through motion |

The tolerances in `configs/record.yaml` (`alignment`) only set what is counted
as "over"; they never remove data.

### 6.4 Verify

```bash
uv run --no-sync dvrk-lerobot verify --dataset data/lerobot/needle_pick
```

It must end with `dataset structure: PASS`. As the final check, load the
dataset with LeRobot (section 7).

---

## 7. Using the dataset

Use a separate Python environment for training (not the ROS one):

```bash
python -m venv ~/envs/lerobot && source ~/envs/lerobot/bin/activate
pip install "lerobot==0.3.3"      # v2.1 datasets
pip install "lerobot==0.6.0"      # v3.0 datasets
```

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/needle_pick", root="data/lerobot/needle_pick")
sample = ds[0]
sample["observation.state"]              # (28,)
sample["observation.images.left"]        # (3, 1080, 1920)
sample["observation.images.external"]    # (3, 480, 640)
sample["observation.depth.external"]     # (480, 640) raw depth units
```

### 7.1 Depth in metres

The depth feature stores the sensor's integers exactly (as float32). Its
metadata in `meta/info.json` holds everything needed to use it:

```json
"observation.depth.external": {"info": {
  "depth_scale_m_per_unit": 0.001,
  "invalid_value": 0,
  "aligned_to": "observation.images.external",
  "intrinsics": {"width": 640, "height": 480, "fx": 605.27, "fy": 605.12,
                 "ppx": 315.79, "ppy": 243.94, "model": "inverse_brown_conrady",
                 "coeffs": [0, 0, 0, 0, 0]}
}}
```

`metres = value * depth_scale_m_per_unit`. A value of 0 means "no depth" at
that pixel; never treat it as 0 m. Depth pixel `(v, u)` and colour pixel
`(v, u)` of the same row show the same point.

---

## 8. Point clouds

A point cloud is the depth back-projected through the camera intrinsics. It is
not stored in the dataset because the depth already holds it exactly.

### 8.1 From a dataset row (recommended)

This needs only the dataset, and the cloud matches the row's robot state and
endoscope images automatically:

```python
import json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

root = Path("data/lerobot/needle_pick")
info = json.loads((root / "meta/info.json").read_text())["features"]["observation.depth.external"]["info"]
k, scale = info["intrinsics"], info["depth_scale_m_per_unit"]

table = pq.read_table(root / "data/chunk-000/episode_000000.parquet", columns=["observation.depth.external"])
depth = np.array(table.column(0)[10].as_py(), dtype=np.float32)       # row 10, (480, 640)

v, u = np.indices(depth.shape)
valid = depth > 0
z = depth[valid] * scale                                               # metres
x = (u[valid] - k["ppx"]) / k["fx"] * z
y = (v[valid] - k["ppy"]) / k["fy"] * z
xyz = np.stack([x, y, z], axis=1)                                      # (N, 3)
```

Coordinates are metres in the D435i colour camera frame: x right, y down,
z forward. For colour, take `observation.images.external` of the same row at
`[v[valid], u[valid]]`. This pinhole formula is exact when `coeffs` are all
zero, as the D435i colour stream reports (checked to 0.1 µm against the
RealSense SDK). If they are not zero, use
`dvrk_lerobot.dataset.cloud.rays(intrinsics)` for the per-pixel rays instead.

### 8.2 From a raw episode

```bash
uv run --no-sync python -m dvrk_lerobot.tools.reconstruct_cloud \
    data/raw/<run>/episode_1 --output data/clouds/<run>/episode_1
```

Writes one binary PLY per raw frameset (`frame_000000.ply`, ...) and a
`manifest.json` with timestamps and point counts, using the intrinsics saved in
`meta.json` (the camera does not need to be connected). Geometry is exact.
Colours are exact only for episodes recorded with `color_lossless: true`;
otherwise they come from the H.264 colour video and are very close.

Raw frame order is not the dataset's row order (a dataset slot can repeat a
frameset). To get the cloud of a dataset row, use section 8.1.

---

## 9. Memory usage

### 9.1 Disk

Measured on this setup (two 1080p endoscope streams, D435i 640x480, 30 fps,
two PSMs):

| Data | Per second | Per minute | Per hour |
|---|---|---|---|
| Endoscope left + right (raw H.264, `crf 18`) | 18.6 MB | 1.1 GB | 67 GB |
| D435i depth (raw, lossless) | 0.9–1.3 MB | 55–80 MB | 3.2–4.8 GB |
| D435i colour (raw, default H.264) | 0.04–0.06 MB | ~3 MB | ~0.2 GB |
| D435i colour (raw, `color_lossless: true`) | ~5.5 MB | ~330 MB | ~20 GB |
| Robot topics (`robot.jsonl`) | 0.12 MB | 7 MB | 0.4 GB |
| **Raw total, default settings** | **~20 MB** | **~1.2 GB** | **~72 GB** |
| Point clouds, every frame (`--record-cloud` or a full rebuild) | ~100 MB | ~6 GB | ~360 GB |
| **LeRobot dataset (v2.1)** | **~21 MB** | **~1.3 GB** | **~76 GB** |
| of which endoscope MP4s | 16.2 MB | 970 MB | 58 GB |
| of which depth in Parquet | 5.0 MB | 300 MB | 18 GB |
| of which D435i colour MP4 | 0.05 MB | 3 MB | 0.2 GB |

- Depth is ~4x larger in the dataset than in the raw FFV1 file, because the
  dataset stores each frame as a float32 array in Parquet.
- Image content changes the numbers: a busy, moving scene compresses worse
  than a still one. Plan with some headroom.
- Recording live point clouds costs ~5x the whole raw recording. Avoid it and
  rebuild only the frames you need (section 8).

### 9.2 What can be deleted

| Data | Safe to delete? |
|---|---|
| `cloud/frame_*.ply` in raw episodes | Yes, if the episode's `external_depth.mkv` is complete: geometry rebuilds exactly (colours exactly only with lossless colour) |
| Rebuilt clouds (`data/clouds/...`) | Yes, they can be rebuilt any time |
| A converted dataset | Yes, if the raw run is kept: convert again |
| A raw run | Only after converting, verifying and backing it up: it is the only copy of the recording, and converting with new settings needs it |
| `<output>.partial` folders | Yes, after reading the convert error |

### 9.3 RAM

- **Recording** holds only a few frames per camera in memory. With
  `--record-cloud`, the cloud writer queues up to 30 clouds (~100 MB); if the
  disk cannot keep up, the episode is marked failed instead of using more RAM.
- **Convert** streams video and depth in batches, so its memory use does not
  grow with episode length. It is CPU-bound: about 1–1.6x the episode duration,
  using many cores.
- **Reading depth yourself:** one decoded depth frame is 1.2 MB (480x640
  float32), so one minute is ~2.2 GB. `pq.read_table` on a whole episode loads
  all of it. For long episodes read only the rows you need, or iterate with
  `pq.ParquetFile(path).iter_batches(batch_size=64, columns=[...])`.
- **Training with LeRobot** decodes frames per sample; memory depends on the
  batch size and the number of data-loader workers, not on the dataset size.

---

## 10. Timestamps and clocks

All alignment uses **one clock: the recording PC's time when data arrived**.

| Data | Time used by convert | Also stored |
|---|---|---|
| Robot messages | `t_ns` (receive time) | `stamp_ns`, the ROS header time from the robot PC |
| Endoscope frames | Host time after decode (`<side>_timestamps.txt`) | V4L2 driver times (`<side>_driver_timestamps.csv`) |
| D435i framesets | Host receive time (`external_timestamps.txt`) | Colour and depth sensor times (`external_sensor_timestamps.csv`) |

- Within one frameset, the D435i colour and depth sensor times differ by
  ~0.06 ms, so colour, depth and any point cloud built from them describe the
  same instant. They need no further alignment.
- A frameset arrives ~35 ms after its sensor time (USB and processing). The
  endoscope frames carry a similar delay, so camera-to-camera alignment is
  consistent; robot samples do not include this delay.
- Do not mix `stamp_ns` with camera times: they come from different computers.

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `doctor` FAIL on a camera | Check `v4l2-ctl --list-devices` and the device paths in the config; close other programs using the camera |
| `doctor` FAIL on the D435i | Reconnect it to a USB 3 port; check `realsense.serial` |
| `no data within 20s; missing: /PSM1/...` | Start the dVRK console; check `ros2 topic list \| grep PSM` and the `arms` topics in the config |
| Preview shows "No Signal" | Check the endoscope output and the capture-card cables before recording |
| Keys do nothing | Click the preview window or the terminal |
| Episode marked `failed` with a point-cloud error | The disk could not keep up with `--record-cloud`; record without it |
| `RGB-D presence differs from the conversion config` | Stereo-only episodes with the RealSense config: add `--no-realsense` (and keep them in a separate dataset) |
| `incompatible RGB-D shape, scale, intrinsics, or frame count` | Episodes from different D435i settings or cameras in one dataset: convert them separately |
| `output already exists` | Choose a new `--output`, or delete the old dataset |
| `ModuleNotFoundError: lark` in `pytest` | Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest` |

More symptoms and fixes are in the README's Troubleshooting section.

---

## 12. Quick reference

```bash
./scripts/setup_env.sh                                      # once
./scripts/doctor.sh                                         # before each session
./scripts/record.sh --task "needle pick"                    # s start, e end, q quit
./scripts/touchx_record.sh --task "needle pick"             # same, with TouchX teleop
grep -H '"status"' data/raw/<run>/episode_*/meta.json       # check episodes
./scripts/convert.sh --sessions data/raw/<run> --output data/lerobot/<name> [--format v3.0]
uv run --no-sync dvrk-lerobot verify --dataset data/lerobot/<name>
uv run --no-sync python -m dvrk_lerobot.tools.reconstruct_cloud data/raw/<run>/episode_1 --output data/clouds/<run>/episode_1
```

| Setting (`configs/record.yaml`) | Default | Change when |
|---|---|---|
| `cameras.*.device` | `/dev/video0`, `/dev/video2` | Capture cards were reconnected |
| `realsense.serial` | `null` | More than one RealSense is connected |
| `realsense.color_lossless` | `false` | You need exact colours in clouds rebuilt from raw data |
| `realsense.record_cloud` | `false` | Almost never |
| `dataset.video_crf` | 18 | Smaller dataset videos (higher) or better quality (lower) |
