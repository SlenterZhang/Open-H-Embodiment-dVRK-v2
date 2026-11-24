# dVRK → LeRobot Dataset v2.1

Collect teleoperation demonstrations on the dVRK and turn them into a
[LeRobot v2.1](https://github.com/huggingface/lerobot) dataset.

The tool records three things at the same time:

- **Robot telemetry** from ROS 2: joint positions, jaw angle and Cartesian pose
  of each PSM (about 100 Hz per topic).
- **Stereo endoscope video** from two V4L2 capture devices (1920x1080, 30 fps).
- **External RGB-D observation** from a D435i (color and 16-bit depth aligned to
  color, 640x480 at 30 fps by default).

The D435i is configured in `configs/record.yaml` under `realsense`. It uses one
`pyrealsense2` pipeline for both streams. `doctor.sh` checks this device as well
as the two existing endoscope cameras. Select a serial number in the config if
multiple RealSense cameras are connected.

D435i depth is stored **losslessly** during recording, as raw unsigned 16-bit
values in an FFV1 MKV, aligned to the color image by the device. Color is H.264
by default (tiny; the dataset re-encodes it to H.264 anyway), or lossless FFV1
with `realsense.color_lossless: true`. `meta.json` records `realsense.depth_scale`
(metres per raw unit; zero is an invalid pixel) and the color intrinsics at
recording time. Together these rebuild the exact point-cloud geometry of every
frame offline (see [Point clouds](#point-clouds)), so clouds do not need to be
stored; their colors are exact only with lossless color. The LeRobot dataset gets the color video, the exact depth integers, and
the depth scale and intrinsics in `info.json`, so it rebuilds the same clouds on
its own.

It then aligns both streams in time and writes them in the LeRobot format.

```text
                ┌──────────────────────────┐        ┌────────────────────────────┐
 dVRK (ROS 2) ─►│ record                   │        │ convert                    │
 endoscope    ─►│ s / e / q, live preview  │──raw──►│ align in time, write       │──► LeRobot v2.1
 D435i RGB-D  ─►│                          │        │                            │
                │ data/raw/<run>/episode_N │        │ data/lerobot/<dataset>     │
                └──────────────────────────┘        └────────────────────────────┘
                 needs the robot                      offline, repeatable
```

| Step | Command | What it does |
|---|---|---|
| Check | `./scripts/doctor.sh` | Verifies ffmpeg, both endoscope cameras, the D435i and robot topics |
| Record | `./scripts/record.sh --task "..."` | Records episodes with a preview window; `s` start, `e` end, `q` quit |
| Convert | `./scripts/convert.sh --sessions ... --output ...` | Aligns robot data and video in time, writes the LeRobot dataset |
| Verify | `uv run --no-sync dvrk-lerobot verify --dataset ...` | Checks the dataset structure |

A step-by-step operator guide (recording day checklist, converting, using depth
and point clouds, disk and RAM usage) is in [docs/MANUAL.md](docs/MANUAL.md).
Design details (threading, timing model, file formats) are in
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Requirements

| Item | Version / note |
|---|---|
| OS | Ubuntu 22.04 with a desktop session (the preview window needs a display) |
| ROS 2 | Humble, at `/opt/ros/humble` |
| dVRK | ROS 2 workspace built at `~/ros2_ws` |
| Python | 3.10 (the system Python used by ROS Humble) |
| uv | 0.9 or newer |
| ffmpeg | System package with `libx264` and FFV1 (`sudo apt install ffmpeg`) |
| Cameras | Two V4L2 endoscope devices plus one D435i RGB-D camera |

## 2. One-time setup

```bash
sudo apt install ffmpeg
./scripts/setup_env.sh
```

`setup_env.sh` creates `.venv` **with access to the system ROS packages**
(`--system-site-packages`), so `rclpy` and the ROS message types can be imported,
and installs `numpy`, `opencv-python` (with GUI), `pyarrow`, `PyYAML` and `pyrealsense2` from
the lock file. Do not create the environment with a plain `uv venv`;
`import rclpy` would fail.

All scripts source ROS for you:

```bash
source /opt/ros/humble/setup.bash      # override with ROS_SETUP=...
source ~/ros2_ws/install/setup.bash    # override with DVRK_SETUP=...
```

---

## 3. Collecting data, step by step

### 3.1 Start the dVRK

Start the dVRK console as usual, power on, home the arms and enable
teleoperation. The recorder only listens; it never sends commands.

### 3.2 Pre-flight check

```bash
./scripts/doctor.sh
```

Expected output:

```text
== tools ==
PASS ffmpeg: /usr/bin/ffmpeg
PASS ffprobe: /usr/bin/ffprobe

== cameras ==
PASS left: 1920x1080 @ 30.0 fps fourcc='MJPG' (/dev/video6)
PASS right: 1920x1080 @ 30.0 fps fourcc='MJPG' (/dev/video8)
PASS external RGB-D: 640x480 @ 30 fps, depth scale 0.001 m/unit

== ROS 2 topics ==
listening for 5.0s ...
PASS /PSM1/measured_js: 496 messages (99.2 Hz)
       first receive time minus header stamp: 0.4 ms
PASS /PSM1/jaw/measured_js: 495 messages (99.0 Hz)
...
RESULT: PASS
```

Fix any `FAIL` before recording (see [Troubleshooting](#8-troubleshooting)).
Also inspect the preview: a capture card can pass while displaying "No Signal".
The "receive time minus header stamp" should be a few milliseconds; a value of
seconds means the robot and the PC clocks disagree.

### 3.3 Record episodes

```bash
./scripts/record.sh --task "needle pick"
```

To record without the RealSense today, leave its settings in the YAML and pass
`--no-realsense`. Omit the flag on a future run to record RGB-D again:

```bash
./scripts/record.sh --task "needle pick" --no-realsense
./scripts/record.sh --task "needle pick"
```

Use `--no-realsense` with `./scripts/doctor.sh` and `./scripts/convert.sh` as
well when checking or converting stereo-only sessions. Keep RGB-D and
stereo-only episodes in separate converted datasets.

| Option | Default | Meaning |
|---|---|---|
| `--task` | required | Task description stored with every episode of this run (becomes the LeRobot task text) |
| `--notes` | empty | Free-form notes stored in each episode's `meta.json` |
| `--root` | `data/raw` | Where run folders are created |
| `--resume-run` | unset | Append to an existing run directory, continuing after its highest episode number; existing episodes are never overwritten |
| `-c / --config` | `configs/record.yaml` | Configuration file |
| `--no-realsense` | off | Ignore the configured RealSense for this run |
| `--record-cloud` / `--no-record-cloud` | YAML setting (off) | Also write per-frame colored PLYs while recording (usually unnecessary, see [Point clouds](#point-clouds)) |

What happens:

1. The recorder waits (up to 20 s) until every configured robot topic has sent data.
2. It opens both cameras and shows a **preview window** with both eyes side by side.
3. You control it with the keyboard, either in the preview window or in the terminal:

| Key | Action |
|---|---|
| `s` | Start a new episode (overlay turns red: `REC episode_N  12.3s  370`) |
| `e` | End the current episode and save it (overlay returns to green `IDLE`) |
| `q` | Quit. If an episode is running, it is saved with status `aborted` |
| `Ctrl+C` | Same as `q` (in the terminal) |

Typical session:

```text
[record] waiting for robot topics
[record] opening cameras
[record] ready. keys (preview window or terminal): [s] start episode   [e] end episode   [q] quit
[record] episode_1 started: data/raw/20261027-192407/episode_1
[record] episode_1 saved (1 this run)
[record] episode_2 started: data/raw/20261027-192407/episode_2
[record] episode_2 saved (2 this run)
recorded 2 episode(s)
raw session: data/raw/20261027-192407/episode_1
raw session: data/raw/20261027-192407/episode_2
```

Tips:

- Robot topics and cameras stay open for the whole run, so `s` starts recording
  immediately. Record as many episodes as you like before pressing `q`.
- A failed demonstration: press `e` as usual, and simply leave that episode out
  when converting (or delete its folder).
- When the preview window has keyboard focus, **every key you type goes to the
  recorder**. Click another window before typing elsewhere.
- One run has one `--task`. For a different task, quit and start a new run.

#### Point clouds

You normally do **not** need to record point clouds. The lossless depth video
plus the intrinsics in `meta.json` rebuild their geometry exactly, at a small
fraction of the disk PLY files need (roughly 100 MB/s). Point colors come from
the color video: exact with `color_lossless: true`, close otherwise.

Rebuild the clouds of one raw episode when you need them:

```bash
uv run --no-sync python -m dvrk_lerobot.tools.reconstruct_cloud \
    data/raw/<run>/episode_1 --output data/clouds/<run>/episode_1
```

This writes `frame_000000.ply`, ... (one per RGB-D frame, raw frame order) and a
`manifest.json` with timestamps and point counts. It uses the recording-time
intrinsics from `meta.json`; `--calibration <file.json>` uses other intrinsics
instead. Coordinates are metres in the color camera frame (x right, y down,
z forward); zero-depth pixels are omitted; no fusion or filtering is applied.

To write clouds live during recording anyway, pass `--record-cloud` (or set
`realsense.record_cloud: true`; `--no-record-cloud` overrides it). Each episode
then also gets `cloud/frame_*.ply`, `cloud/timestamps.csv` and
`cloud/metadata.json`. A background worker writes them; if the disk cannot keep
up (30-frame queue full) the episode is marked failed rather than dropping
clouds. Cloud recording needs the RealSense and cannot be combined with
`--no-realsense`.

### 3.4 Where the raw data goes

Each run creates one folder named by the time of its first episode, with one
sub-folder per episode:

```text
data/raw/20261027-192407/              ← one record run
├── episode_1/
│   ├── meta.json                      ← task, status, cameras, topics, joint names, counts
│   ├── robot.jsonl                    ← every robot message, one JSON per line
│   ├── video/
│   │   ├── left.mkv                   ← left eye, H.264, every captured frame
│   │   ├── left_timestamps.txt        ← host time after decode (ns), line i = frame i
│   │   ├── left_driver_timestamps.csv ← driver metadata and host stage times, row i = frame i
│   │   ├── right.mkv
│   │   ├── right_timestamps.txt
│   │   ├── right_driver_timestamps.csv
│   │   ├── external.mkv              ← D435i color, H.264 (FFV1 with color_lossless)
│   │   ├── external_depth.mkv        ← color-aligned uint16 depth, lossless FFV1
│   │   ├── external_timestamps.txt   ← one timestamp per RGB-D frameset
│   │   └── external_sensor_timestamps.csv ← sensor times and frame numbers per frameset
│   └── cloud/                     ← per-frame PLYs, only with --record-cloud
└── episode_2/
    └── ...
```

`meta.json` → `status` tells you how the episode ended:

| Status | Meaning |
|---|---|
| `ok` | Ended with `e` |
| `aborted` | Program quit (`q`, Ctrl+C) while the episode was running |
| `failed` | A camera or encoder error stopped the episode |
| `recording` | Program was killed hard; the files are still readable |

A line of `robot.jsonl`:

```json
{"t_ns":1793096647123456789,"stamp_ns":1793096647123000000,"kind":"jp","arm":"PSM1","values":[0.1,0.2,0.3,0.4,0.5,0.6]}
```

`kind` is `jp` (6 joints), `jaw` (1 value) or `cp` (pose `x y z qx qy qz qw`).
`stamp_ns` is the ROS header time, `t_ns` the time the PC received it.

Raw data is the master copy: `convert` only reads it, and you can convert it
again at any time.

### Stereo driver timestamp diagnostics

New recordings also save `video/left_driver_timestamps.csv` and
`video/right_driver_timestamps.csv`. Each CSV row describes exactly the same
frame as the corresponding video index and line in `<side>_timestamps.txt`.
The recorder reads V4L2 MMAP buffers directly so that image,
sequence and timestamp all come from the **same** `VIDIOC_DQBUF` result; OpenCV
is used to decode the copied image payload.
The currently supported capture formats are MJPG/JPEG/YUYV/UYVY (single plane).
The driver may allocate more buffers than the two requested; `buffer_count`
is recorded in camera metadata.

| CSV field | Meaning |
|---|---|
| `frame_index` | Zero-based frame index within this episode's saved video |
| `driver_timestamp_ns` | Original V4L2 buffer timeval converted to ns; source precision is microseconds |
| `driver_sequence` | Driver's 32-bit stream sequence; it does not restart for each episode and is not a shared left/right frame ID |
| `driver_flags` | Original flags returned by the driver |
| `driver_clock` | `monotonic`, `unknown`, `copy`, or unrecognized flag bits |
| `driver_timestamp_source` | Driver declares `start_of_exposure`, `end_of_frame`, or unrecognized bits |
| `sequence_step` | Sequence increment since the preceding captured frame, modulo 2^32 |
| `host_dequeue_ns` | PC wall-clock time immediately after DQBUF, before copying/decoding |
| `host_dequeue_monotonic_ns` | Paired PC monotonic time (midpoint of two clock reads surrounding the wall-clock read) |
| `host_clock_pair_span_ns` | Time span of that clock pairing; use to assess its uncertainty |
| `host_decode_complete_ns` | PC wall-clock time after decode; also written to the original timestamps.txt used by convert |
| `host_decode_complete_monotonic_ns` | PC monotonic time immediately after the preceding wall-clock read |
| `driver_to_dequeue_ns` | Dequeue minus driver time, **only** when the driver reports MONOTONIC and a positive timestamp; otherwise blank |
| `encoder_write_start_monotonic_ns`, `encoder_write_end_monotonic_ns` | Encoding pipe write boundaries, useful for identifying blocking |

`MONOTONIC` can be compared with host monotonic time, **not** directly with
ROS epoch header time. UNKNOWN/COPY must not be guessed to be realtime.
`start_of_exposure` is a **driver declaration**. An HDMI capture card reporting
this flag does not prove that it reports the upstream endoscope sensor's exact
exposure time or that both sensors are hardware synchronized. `end_of_frame`
denotes receipt of the frame's last pixel. See Linux V4L2's buffer timestamp
and timestamp-source definitions in the official userspace API documentation:

```text
https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/buffer.html
```

To analyze one **newly recorded** episode after it has been saved:

```bash
cd /home/zhang/zst/Open-H-Embodiment-dVRK-v2
.venv/bin/python -m dvrk_lerobot.tools.stereo_timing \
  data/raw/<run>/episode_1 \
  --tolerance-ms 15 \
  --output /tmp/stereo_timing.json
```

The command reads the files, checks video/CSV/legacy timestamp correspondence,
and compares stereo timing at driver, dequeue and decode stages. It also
reports frame-sequence gaps, driver error flags, decode delay, encoder write
duration, and the delay difference for the **same driver-matched frame pairs**.
It refuses cross-driver comparisons when clocks/sources are incompatible.

- A large difference already in driver timestamps places the difference at or
  before that driver timestamp boundary (source/card/driver); these logs alone
  cannot separate those layers.
- Close driver timestamps followed by a large host difference points to delay
  after the driver timestamp boundary; compare dequeue/decode/encoder timings.
- A jump in frame sequence means driver frame numbers went unobserved. It does
  not, by itself, diagnose a broken camera.

Convert still uses the original after-decode host times and current tolerances.
Adding diagnostics does not establish exposure synchronization or eliminate
alignment gaps. Older recordings lack driver metadata and cannot be analyzed
retroactively by this command.

### 3.5 Convert to LeRobot

```bash
./scripts/convert.sh --sessions data/raw/20261027-192407 --output data/lerobot/needle_pick
```

| Option | Meaning |
|---|---|
| `--sessions` | One or more **run folders** and/or **episode folders**. A run folder expands to its `episode_1..N` in numeric order (`episode_10` after `episode_9`) |
| `--output` | New dataset folder. It must not exist yet; nothing is ever overwritten |
| `--format` | `v2.1` (default, `lerobot<=0.3.x`) or `v3.0` (LeRobot dataset format v3.0, `lerobot==0.6.0`); see [4.1](#41-v30-layout---format-v30) |
| `--allow-incomplete` | Also convert episodes whose status is not `ok` |
| `-c / --config` | Configuration file (tolerances and output video settings) |

Examples:

```bash
# Several runs into one dataset (episodes keep the order given)
./scripts/convert.sh --sessions data/raw/20261027-192407 data/raw/20261028-101500 \
  --output data/lerobot/needle_pick_all

# Only selected episodes
./scripts/convert.sh --sessions data/raw/20261027-192407/episode_1 data/raw/20261027-192407/episode_3 \
  --output data/lerobot/needle_pick_best
```

Example output:

```text
[align] 20261027-192407/episode_1: grid: 96 frames at 30 fps (3.20 s)
[align] 20261027-192407/episode_1: left: 90 frames captured, 6 slots repeated (6.2%), 0 frames unused, longest gap 233 ms
[align] 20261027-192407/episode_1: right: 90 frames captured, 6 slots repeated (6.2%), 0 frames unused, longest gap 233 ms
[align] 20261027-192407/episode_1: stereo skew: median 11.8 ms, worst 14.7 ms, 0 frames over 25 ms
[align] 20261027-192407/episode_1: robot sample age: median 3.7 ms, worst 7.0 ms, 0 frames over 25 ms
dataset: /home/zhang/zst/Open-H-Embodiment-dVRK-v2/data/lerobot/needle_pick
episodes: 1, frames: 96
```

How to read it:

| Field | Meaning | Healthy value |
|---|---|---|
| `grid` | Dataset frames for this episode. The duration matches the time both cameras were recording | the length of the episode |
| `frames captured` | Frames recorded by this camera | about 30 per second |
| `slots repeated` | Grid frames with no new picture from this camera, which show the previous one again | a few percent |
| `frames unused` | Frames that arrived in a slot that already had one, or outside the grid | a few |
| `longest gap` | Longest time between two frames from this camera | about 33 ms, a few frames at most |
| `stereo skew` | Time between the left and right frames of one dataset frame | under 33 ms at 30 fps |
| `robot sample age` | Time between a dataset frame and the robot samples held for it | a few ms |

Episodes in a run folder whose status is not `ok` are skipped with a
`[convert] skipping ...` line. If conversion fails, the half-written output is
kept as `<output>.partial` for inspection; delete it before retrying.

### 3.6 How the time alignment works

The robot (about 100 Hz per topic) and the two cameras (about 30 fps each) run
independently. `convert` places them all on one fixed-rate grid, so frame *k* is
`k / fps` seconds into the episode:

1. **Grid.** One slot every `1 / fps` seconds, shared by both eyes (and the D435i
   when present), covering the time when every camera was recording.
2. **Video.** Each camera gives a slot the first frame that arrived inside it. A
   slot with no new frame repeats the previous picture, so a dropped frame does
   not shorten the episode. D435i color and depth share a frameset, so they
   always use the same index.
3. **Robot state.** For each arm, the latest `jp`, `jaw` and `cp` samples received
   at or before the slot's time are attached. Receive time is used because the
   camera frames are stamped on the same PC clock. ROS header stamps are kept in
   the raw data but not used here.
4. **Output.** Video frame *k* matches parquet row *k*, and
   `timestamp = frame_index / fps` (what LeRobot expects). The true capture times
   stay in the raw folder.

Frames are never removed for being outside a tolerance. `stereo_tolerance_ms`,
`external_tolerance_ms` and `robot_tolerance_ms` only set what the `convert`
output counts.

Raw recordings made before this grid was added can be converted again without
recording them again. Datasets already converted keep the old timing: they
dropped frames and then numbered the rest as if nothing was missing.

### 3.7 Verify

```bash
uv run --no-sync dvrk-lerobot verify --dataset data/lerobot/needle_pick
```

```text
episodes: 2
frames: 835
tasks: 1
fps: 30
state_dimension: 28
dataset structure: PASS
```

It checks metadata totals, parquet columns and indices, `timestamp = k / fps`,
and that every video exists with the expected size. It does not decode every
video frame; load the dataset with LeRobot as the final check (section 5).

---

## 4. The LeRobot dataset

```text
data/lerobot/needle_pick/
├── meta/
│   ├── info.json              ← fps, features, shapes, codec, path templates
│   ├── episodes.jsonl         ← {episode_index, tasks, length} per episode
│   ├── tasks.jsonl            ← {task_index, task}
│   ├── episodes_stats.jsonl   ← per-episode min/max/mean/std
│   └── stats.json             ← dataset-wide statistics
├── data/chunk-000/
│   ├── episode_000000.parquet
│   └── episode_000001.parquet
└── videos/chunk-000/
    ├── observation.images.left/episode_000000.mp4 ...
    ├── observation.images.right/episode_000000.mp4 ...
    └── observation.images.external/episode_000000.mp4 ...
```

| Feature | Type / shape | Content |
|---|---|---|
| `observation.state` | float32 `[28]` | per arm (PSM1, then PSM2): 6 joints, jaw, pose `x y z qx qy qz qw` |
| `action` | float32 `[28]` | next state (`state[t+1]`, last row repeated); a placeholder, see Limitations |
| `observation.images.left` | video `[1080, 1920, 3]` | H.264, yuv420p, 30 fps |
| `observation.images.right` | video `[1080, 1920, 3]` | H.264, yuv420p, 30 fps |
| `observation.images.external` | video `[480, 640, 3]` by default | D435i color, re-encoded to H.264 yuv420p by `convert` |
| `observation.depth.external` | float32 `[480, 640]` by default | Exact raw uint16 depth integers in Parquet; metres = value × `depth_scale_m_per_unit`. The feature's `info.intrinsics` (color camera, which depth is aligned to) back-projects it to a point cloud |
| `timestamp` | float32 | `frame_index / fps` |
| `frame_index` | int64 | `0..N-1` within the episode |
| `episode_index` | int64 | episode number in the dataset |
| `index` | int64 | global frame number across episodes |
| `task_index` | int64 | row in `tasks.jsonl` |

### 4.1 v3.0 layout (`--format v3.0`)

Same alignment, features, videos and statistics; only the storage changes. The
exporter writes the v2.1 tree to `<output>.v2.1`, repacks it the way
lerobot 0.6.0's `convert_dataset_v21_to_v30` does, and then deletes the staging copy.

```text
data/lerobot/needle_pick/
├── meta/
│   ├── info.json                         ← codebase_version "v3.0", file-based path templates
│   ├── tasks.parquet                     ← task (index) → task_index
│   ├── stats.json
│   └── episodes/chunk-000/file-000.parquet  ← per episode: tasks, length, data/video file + span, stats/*
├── data/chunk-000/file-000.parquet       ← many episodes per file (rolls over at 100 MB)
└── videos/observation.images.{left,right}/chunk-000/file-000.mp4  ← episodes concatenated (200 MB)
```

Each episode row stores `dataset_from_index/dataset_to_index` (rows in the data
file) and `videos/<key>/from_timestamp/to_timestamp` (its span inside the MP4).

State names, in order (stored in `info.json`):

```text
PSM1.jp.yaw  PSM1.jp.pitch  PSM1.jp.insertion  PSM1.jp.roll  PSM1.jp.wrist_pitch  PSM1.jp.wrist_yaw
PSM1.jaw
PSM1.cp.x  PSM1.cp.y  PSM1.cp.z  PSM1.cp.qx  PSM1.cp.qy  PSM1.cp.qz  PSM1.cp.qw
PSM2.… (same 14 values)
```

Units are those published by the dVRK: radians for revolute joints and the jaw,
metres for insertion and position. The pose is in each arm's base frame
(`PSM1_base`, `PSM2_base`).

## 5. Loading in LeRobot

Use a **separate** Python environment for training, so the PyTorch stack does
not mix with the ROS acquisition environment:

```bash
python -m venv ~/envs/lerobot && source ~/envs/lerobot/bin/activate
pip install "lerobot==0.3.3"      # for --format v2.1 (default)
pip install "lerobot==0.6.0"      # for --format v3.0
```

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/needle_pick", root="data/lerobot/needle_pick")
print(ds)
sample = ds[0]
print(sample["observation.state"].shape)          # torch.Size([28])
print(sample["observation.images.left"].shape)    # torch.Size([3, 1080, 1920])
depth_m = sample["observation.depth.external"] * ds.meta.features["observation.depth.external"]["info"]["depth_scale_m_per_unit"]
```

lerobot 0.4+ only reads v3.0. Export with `--format v3.0` for those versions.

---

## 6. Configuration

All settings are in [configs/record.yaml](configs/record.yaml). `record` and
`convert` read the same file.

### `arms`

```yaml
arms:
  - name: PSM1
    js_topic: /PSM1/measured_js        # sensor_msgs/JointState, 6 joints
    cp_topic: /PSM1/measured_cp        # geometry_msgs/PoseStamped
    jaw_topic: /PSM1/jaw/measured_js   # sensor_msgs/JointState, 1 value
  - name: PSM2
    ...
```

The order of arms is the order inside `observation.state`. Keep it the same for
all recordings that go into one dataset. To record a single arm, delete the
other block.

### `cameras`

| Key | Default | Meaning |
|---|---|---|
| `device` | `/dev/video0` (left), `/dev/video2` (right) | V4L2 device; confirm paths with `v4l2-ctl --list-devices` after reconnecting cameras and swap if eyes are reversed |
| `width`, `height` | 1920, 1080 | Must be a mode the capture card supports |
| `fps` | 30 | Must equal `dataset.fps` |
| `fourcc` | `MJPG` | Pixel format requested from the card |
| `crf` | 18 | Quality of the **raw recording**. Lower = better and larger; 18 is visually lossless, 23 is about 2.5x smaller |

List the modes your card supports:

```bash
ffmpeg -hide_banner -f v4l2 -list_formats all -i /dev/video6
```

### `realsense`

The optional `realsense` block enables one D435i RGB-D observation. If it is
omitted, recording and conversion are stereo-only. Use a stereo-only config (or
`--no-realsense`) when converting older raw episodes without RGB-D.

| Key | Default | Meaning |
|---|---|---|
| `serial` | `null` | Device serial; `null` selects the first connected D435i |
| `width`, `height` | 640, 480 | Resolution of both color and depth |
| `fps` | 30 | Must equal `dataset.fps` |
| `record_cloud` | `false` | Also write per-frame PLYs while recording; see [Point clouds](#point-clouds) |
| `color_lossless` | `false` | `false`: raw color H.264 (CRF 18, ~0.06 MB/s). `true`: lossless FFV1 (~5 MB/s), only needed for exact colors in rebuilt point clouds; the dataset is the same either way |

Raw depth is always lossless FFV1. `meta.json` records the color codec as
`realsense.color_codec` (older episodes record `color_crf` instead).

### `dataset`

| Key | Default | Meaning |
|---|---|---|
| `fps` | 30 | Dataset frame rate |
| `robot_type` | `dvrk` | Written to `info.json` |
| `video_codec` | `h264` | Only `h264` is supported |
| `video_crf` | 18 | Quality of the **dataset videos** written by `convert` |
| `video_preset` | `medium` | x264 speed/size trade-off for `convert` (`veryfast` … `veryslow`) |

### `alignment`

| Key | Default | Meaning |
|---|---|---|
| `stereo_tolerance_ms` | 25 | `convert` counts frames whose left and right pictures are further apart than this |
| `external_tolerance_ms` | 25 | `convert` counts frames whose RGB-D frameset is further than this from the left picture |
| `robot_tolerance_ms` | 25 | `convert` counts frames whose held robot sample is older than this |

Changing `dataset` or `alignment` only needs a new `convert`; the raw data does
not have to be recorded again.

---

## 7. Disk space and performance

| | Size |
|---|---|
| Raw recording, 1080p stereo, `crf: 18` | about 18 MB/s, **about 65 GB per hour** |
| Raw D435i color, 640x480 | about 0.06 MB/s (H.264); about 5 MB/s with `color_lossless: true` |
| Raw D435i depth, 640x480, lossless | about 1–2 MB/s |
| Raw point clouds (`--record-cloud`) | about 100 MB/s, **about 350 GB per hour**; avoid |
| Dataset, 1080p stereo, `video_crf: 18` | about 16 MB/s (for example, 459 MB for 28 s) |

- Each camera is encoded to H.264 while recording, using about 2 CPU cores per
  eye.
- `convert` takes roughly as long as the recording on this machine. It uses many
  cores while encoding.
- Free space: `df -h .`. Delete raw runs only after converting and checking them.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: rclpy` | `.venv` created without system packages | `rm -rf .venv && ./scripts/setup_env.sh` |
| `no data within 20s; missing: /PSM1/...` | dVRK console not running, or a topic name differs | Start the console; `ros2 topic list \| grep PSM`; fix `arms` in the config |
| `cannot open /dev/video6` | Device busy or no permission | Close other camera apps (including another recorder); check the device path and permissions |
| Stereo preview shows `No Signal` | Capture card has no upstream video input | Check endoscope output, HDMI cables and capture-card inputs before recording |
| `negotiated 1280x720, expected 1920x1080` | Mode not supported by the card | Pick a listed mode (section 6) |
| Preview window does not appear / Qt `xcb` error | No display (e.g. SSH session) | Run from the desktop session, or `ssh -X` |
| Keys do nothing | Neither the preview window nor the terminal has focus | Click the preview window or the terminal |
| Many `robot sample age` frames over the tolerance | Robot topics stalled during recording | Check `ros2 topic hz /PSM1/measured_js`; discard the episode if the state was held through motion |
| High `slots repeated` or a long `longest gap` | Cameras dropped frames or run below `fps` (CPU or USB overload) | Close other heavy programs; check the USB connections; discard the episode if the picture was frozen through motion |
| `session status is 'aborted'` | The episode was ended with `q` / Ctrl+C | Add `--allow-incomplete` if the episode is usable |
| `output already exists` | `--output` folder exists | Choose a new name, or delete the old dataset |
| `convert failed ... partial output kept` | Error during conversion | Read the message, delete `<output>.partial`, run again |
| `ModuleNotFoundError: lark` when running `pytest` | ROS's pytest plugin is loaded from `PYTHONPATH` | `env -u PYTHONPATH uv run --no-sync pytest` |

Check a single camera without the recorder:

```bash
ffplay -f v4l2 -input_format mjpeg -video_size 1920x1080 -framerate 30 /dev/video6
```

---

## 9. Limitations

- **`action` is a placeholder.** It is the next measured state, not the
  operator's command. Record a command topic (for example `/PSM1/setpoint_js`)
  before training a policy that needs true actions.
- **Software stereo sync.** Convert uses PC time after each frame is decoded.
  Driver timestamps are now logged for diagnostics. The recorder does not
  configure or verify hardware synchronization of the source or capture cards.
- **Partial camera calibration.** Only the D435i colour intrinsics are stored
  (raw `meta.json` and the dataset's `info.json`). Endoscope intrinsics, stereo
  extrinsics and camera-to-robot registration are not.
- **Positions only.** Joint velocities, efforts and other CRTK topics are not
  recorded.
- **One task per run.** All episodes of one `record` run share its `--task`.

## 10. Development

```bash
env -u PYTHONPATH uv run --no-sync pytest   # unit + integration tests (needs ffmpeg)
uv run --no-sync ruff check src tests
```

| Path | Contents |
|---|---|
| `scripts/` | `setup_env.sh`, `doctor.sh`, `record.sh`, `convert.sh` |
| `configs/record.yaml` | Topics, cameras, tolerances, video settings |
| `src/dvrk_lerobot/acquire/` | Recorder: ROS node, cameras, preview, keyboard, raw session files |
| `src/dvrk_lerobot/align/` | Time alignment |
| `src/dvrk_lerobot/dataset/` | LeRobot writer: parquet, video, metadata, statistics |
| `src/dvrk_lerobot/tools/` | `doctor` and `verify` |
| `tests/` | Unit and integration tests |
