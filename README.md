# dVRK to LeRobot Dataset v2.1

Record dVRK Cartesian and joint telemetry together with a stereo endoscope stream,
align the streams in time, and export the result as a
[LeRobot Dataset v2.1](https://github.com/huggingface/lerobot) directory.

The target platform is Ubuntu 22.04, ROS 2 Humble, a dVRK workspace built at
`/home/zhang/ros2_ws`, and two V4L2 cameras exposed as `/dev/video0` and
`/dev/video2`. `uv` 0.9 or newer, Python 3.10, and a system `ffmpeg` with
`libx264` are required.

The design is deliberately split into two stages:

1. **Acquire** raw robot samples and timestamped stereo JPEGs. This stage only
   needs ROS 2 and OpenCV.
2. **Export** the raw session(s) into LeRobot v2.1 parquet, MP4, and metadata.
   This stage does not need ROS 2 and can be re-run with different tolerances.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed design, timing model,
and design trade-offs.

---

## 1. What is recorded

For each configured arm (default `PSM1`):

| Source | ROS 2 topic | Message | Stored as |
|---|---|---|---|
| Joint state | `/PSM1/measured_js` | `sensor_msgs/msg/JointState` | `measured_js.position` |
| Cartesian pose | `/PSM1/measured_cp` | `geometry_msgs/msg/PoseStamped` | position + quaternion `xyzw` |
| Left eye | `/dev/video0` | V4L2 | JPEG, timestamped filename |
| Right eye | `/dev/video2` | V4L2 | JPEG, timestamped filename |

`measured_jp` is **not** a dVRK topic. In CRTK and `dvrk_python`,
`measured_jp()` is a client-side accessor that returns the `position` field of
`measured_js`. The recorder subscribes to `measured_js` and stores its
`position` array; no separate `/measured_jp` subscription exists.

The exporter produces an `observation.state` vector with this order:

```text
PSM1.jp.<joint 0> ... PSM1.jp.<joint n>
PSM1.cp.x PSM1.cp.y PSM1.cp.z PSM1.cp.qx PSM1.cp.qy PSM1.cp.qz PSM1.cp.qw
```

For multiple arms, the same block is repeated in the order listed in
`configs/record.yaml`. Units and frames are whatever the dVRK publishes
(normally metres, radians, and the `header.frame_id` of `measured_cp`).

`action` is currently defined as the next aligned `observation.state`
(`action[t] = state[t + 1]`; the last action repeats the last state). This makes
the dataset structurally complete for LeRobot, but it is a placeholder until a
commanded-action source is selected. See
[Assumptions and limitations](#12-assumptions-and-limitations).

---

## 2. Quick start

```bash
# 1. System tools (once)
sudo apt install ffmpeg v4l-utils

# 2. Python environment (once; keeps system ROS packages visible)
./scripts/setup_env.sh

# 3. ROS environment (every shell; the wrapper scripts do this for you)
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
# Override ROS_SETUP / DVRK_SETUP if your workspaces live elsewhere.

# 4. Pre-flight: ffmpeg, both cameras, and live topics
uv run --no-sync dvrk-lerobot doctor -c configs/record.yaml

# 5. Record one episode (Ctrl+C or --duration to stop)
./scripts/record.sh --task "needle transfer" --operator zhang --duration 60

# 6. Export one or more recordings
./scripts/export.sh \
  --sessions data/raw/20261027-140000_needle-transfer \
  --output data/lerobot/needle_transfer

# 7. Structural check
uv run --no-sync dvrk-lerobot verify --dataset data/lerobot/needle_transfer
```

Record several demonstrations by repeating step 5 with different sessions and
passing all session directories to step 6. Each raw session becomes one
LeRobot episode.

---

## 3. Environment

### 3.1 Why a system-site-packages virtual environment

`rclpy` and the ROS 2 message packages are installed under
`/opt/ros/humble`, not on PyPI. The project therefore uses a `uv` environment
created with `--system-site-packages`:

```bash
uv venv --system-site-packages --python /usr/bin/python3 .venv
uv sync
```

This gives the environment access to `rclpy`, `sensor_msgs`, and
`geometry_msgs`, while `numpy`, `opencv-python-headless`, `pyarrow`, and
`PyYAML` come from the project lock file. Do **not** create a plain
`uv venv` for the recorder; `import rclpy` will fail.

`scripts/setup_env.sh` performs these steps and refuses to reuse an existing
environment that was created without system site packages.

### 3.2 Pinned dependencies

| Package | Purpose |
|---|---|
| `numpy<2` | array math, compatible with ROS 2 Humble Python 3.10 |
| `opencv-python-headless` | V4L2 capture and JPEG writing |
| `pyarrow` | parquet tables |
| `PyYAML` | acquisition configuration |
| `pytest`, `ruff` | development only |

`ffmpeg` and `ffprobe` are system binaries and are not Python dependencies.
They are used only during export.

### 3.3 Import check

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
uv run --no-sync python -c \
  "import rclpy, cv2, pyarrow, numpy; print('imports ok', numpy.__version__)"
```

If this fails, check `.venv/pyvenv.cfg` for
`include-system-site-packages = true`.

---

## 4. Repository layout

```text
.
├── README.md                     # this file
├── ARCHITECTURE.md               # detailed design and timing model
├── pyproject.toml                # uv project and tool configuration
├── uv.lock                       # locked Python dependency graph
├── configs/
│   └── record.yaml               # topics, cameras, fps, tolerances
├── scripts/
│   ├── setup_env.sh              # create the system-site-packages venv
│   ├── doctor.sh                 # pre-flight checks
│   ├── record.sh                 # source ROS 2 + record
│   └── export.sh                 # export raw sessions
├── src/dvrk_lerobot/
│   ├── cli.py                    # argument parsing and command dispatch
│   ├── config/                   # typed configuration
│   │   ├── model.py
│   │   └── loader.py
│   ├── acquire/                  # ROS 2 + camera acquisition
│   │   ├── camera.py             # V4L2 capture thread
│   │   ├── metadata.py           # session naming and metadata
│   │   ├── node.py               # ROS 2 subscriptions and callbacks
│   │   ├── recorder.py           # camera/ROS orchestration
│   │   └── session.py            # raw session writer/reader
│   ├── align/                    # timestamp alignment
│   │   └── sync.py
│   ├── dataset/                  # LeRobot v2.1 writer
│   │   ├── schema.py             # feature names, paths, info.json
│   │   ├── stats.py              # numeric and video statistics
│   │   ├── video.py              # H.264 encoding/probing
│   │   ├── parquet.py            # per-episode table
│   │   ├── metadata.py           # meta/*.json and meta/*.jsonl
│   │   └── writer.py             # export orchestration
│   └── tools/
│       ├── doctor.py             # camera/topic/tool checks
│       └── verify.py             # read-only dataset validation
└── tests/
    ├── unit/
    └── integration/
```

Each directory has one responsibility. `acquire` is the only package that
runs ROS 2 subscription callbacks or camera capture; `dataset` is the only
package that writes LeRobot files.

---

## 5. ROS 2 topic contract

Check the live graph before recording:

```bash
ros2 topic list | grep -E 'PSM|ECM|MTM'
ros2 topic type /PSM1/measured_js
ros2 topic type /PSM1/measured_cp
ros2 topic hz /PSM1/measured_js
ros2 topic hz /PSM1/measured_cp
ros2 topic info -v /PSM1/measured_js
```

Expected types:

```text
/PSM1/measured_js   sensor_msgs/msg/JointState
/PSM1/measured_cp   geometry_msgs/msg/PoseStamped
```

Topic names are configured in `configs/record.yaml`; the QoS profile is fixed
to `qos_profile_sensor_data` (best effort). A best-effort subscriber is
compatible with both best-effort and reliable publishers, which makes the
recorder tolerant of the dVRK bridge's QoS choice.

### Timestamps

- Camera frames are timestamped with `time.time_ns()` immediately after
  `VideoCapture.read()` returns.
- Robot samples keep the ROS header stamp as their alignment time. If the
  header stamp is zero, the receive time is used and a warning is printed.
- The exporter aligns camera and robot streams using those integer timestamps.
- Run with `use_sim_time=false`. If the header stamp differs from the wall
  clock by more than two seconds, the recorder warns and `doctor` prints the
  observed offset. Fix the clock source before collecting a real dataset.

Inspect the stamps directly if needed:

```bash
ros2 topic echo --once /PSM1/measured_cp --field header.stamp
date +%s%N
```

---

## 6. Camera setup

List devices and supported modes:

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video2 --list-formats-ext
```

Default assumptions:

| Side | Device | Mode | Format |
|---|---|---|---|
| left | `/dev/video0` | 1280x1024 @ 30 | MJPG |
| right | `/dev/video2` | 1280x1024 @ 30 | MJPG |

Change `configs/record.yaml` if the devices enumerate differently, if the
endoscope uses another resolution, or if MJPG is unavailable. The recorder
fails fast when the negotiated size differs from the request; it does not
silently resize.

The two devices are treated as independent cameras with software timestamps.
They are not hardware-triggered. The `alignment.stereo_tolerance_ms` setting
bounds the accepted left/right skew. For quantitative stereo work, use a
hardware-synchronised capture path instead.

Both cameras must use the same resolution and rate, and that rate must equal
`dataset.fps`. This keeps the final MP4 timeline and the LeRobot `fps` field
consistent.

---

## 7. Recording

### 7.1 Configuration

The default file is `configs/record.yaml`:

```yaml
dataset:
  fps: 30
  robot_type: dvrk
  video_codec: h264
  video_crf: 18
  video_preset: medium

arms:
  - name: PSM1
    js_topic: /PSM1/measured_js
    cp_topic: /PSM1/measured_cp

cameras:
  left:
    device: /dev/video0
    width: 1280
    height: 1024
    fps: 30
    fourcc: MJPG
    jpeg_quality: 95
  right:
    device: /dev/video2
    width: 1280
    height: 1024
    fps: 30
    fourcc: MJPG
    jpeg_quality: 95

alignment:
  stereo_tolerance_ms: 15.0
  robot_tolerance_ms: 25.0
```

Add a second arm by uncommenting the `PSM2` block. The arm order defines the
state-vector order and must stay stable for a dataset.

### 7.2 Pre-flight

```bash
./scripts/doctor.sh -c configs/record.yaml
```

`doctor` checks:

1. `ffmpeg` and `ffprobe` on `PATH`.
2. Both V4L2 devices open at the requested size and rate.
3. Every configured topic produces messages during the listen window.

It prints `PASS`/`FAIL` per check and a final summary. Exit code is non-zero on
failure. Run it with the dVRK running and no other process holding the cameras.

### 7.3 Start and stop

```bash
./scripts/record.sh --task "needle transfer" --operator zhang --duration 60
```

- `--duration 0` (the default) records until `Ctrl+C`.
- Recording starts only after every configured `measured_js` and
  `measured_cp` topic has produced at least one valid sample.
- A session directory is created after warm-up; camera threads start at that
  moment.
- `SIGTERM` and `Ctrl+C` stop the threads, flush the robot log, and finalise
  `meta.json`.
- If a camera thread fails, the session is marked `failed` and the error is
  propagated.

### 7.4 Raw session layout

```text
data/raw/20261027-140000_needle-transfer/
├── meta.json                 # task, operator, topics, camera negotiation
├── robot.jsonl               # one JSON object per measured_js/measured_cp sample
└── frames/
    ├── left/
    │   └── 1761544800123456789.jpg
    └── right/
        └── 1761544800123456789.jpg
```

Example robot line:

```json
{"t_ns":1761544800123456789,"stamp_ns":1761544800123000000,"kind":"jp","arm":"PSM1","values":[0.1,0.2,0.3,0.4,0.5,0.6,0.7]}
```

`t_ns` is the receive time, `stamp_ns` is the header stamp, `kind` is `jp` or
`cp`, and `values` is the raw message payload. Raw sessions are immutable once
finalised; re-export does not modify them.

Disk usage is dominated by JPEGs. At 1280x1024, quality 95, 30 fps, stereo,
expect roughly 6-12 MB/s, or 20-40 GB for a one-hour session. The final H.264
videos are much smaller. Plan disk space before long collections.

---

## 8. Export

```bash
./scripts/export.sh \
  --sessions \
    data/raw/20261027-140000_needle-transfer \
    data/raw/20261027-141500_needle-transfer \
  --output data/lerobot/needle_transfer
```

- `--output` must not exist. The exporter writes a sibling `*.partial`
  directory and renames it only after success, so an interrupted export never
  looks complete.
- On failure the partial directory is kept for inspection and the error is
  printed. Remove it manually after inspecting it.
- All sessions must share the same joint names and camera geometry.
- Sessions whose `meta.json` status is not `ok` are refused by default; pass
  `--allow-incomplete` only when you have inspected a crashed/failed session.
- Each session becomes one episode; tasks are taken from `meta.json` and
  deduplicated into `meta/tasks.jsonl`.
- Re-exporting the same raw sessions is safe and produces a new directory.

### Alignment policy

1. **Stereo pairing.** Walk left frames in timestamp order and pair each with
   the nearest unused right frame within `stereo_tolerance_ms`. Left is the
   master clock.
2. **Pair time.** Use the midpoint of the left and right timestamps.
3. **Robot state.** For each arm, find the nearest `measured_js` and
   `measured_cp` samples within `robot_tolerance_ms`. Duplicate header stamps
   are collapsed to the first sample.
4. **Drop incomplete rows.** If either joint or pose data is missing, the pair
   is dropped and counted in the alignment notes.
5. **Fixed output timeline.** Output `timestamp[k] = k / fps`. This is the
   LeRobot playback timeline, not the raw capture time; the raw session keeps
   the actual timestamps.
6. **Action.** `action[t] = state[t + 1]`, with the final action repeated.

The CLI prints alignment notes per session, including stereo pairs, aligned
frames, robot drops, median joint/pose skew, and a warning when the capture span
differs from the nominal duration by more than 10%.

### Tolerances

| Setting | Meaning | Suggested start |
|---|---|---|
| `stereo_tolerance_ms` | max left/right timestamp difference | 15 ms for two UVC cameras |
| `robot_tolerance_ms` | max robot sample age at a stereo pair | 25 ms |

Tighten the stereo tolerance if both eyes are hardware-synchronised. Raising it
accepts more skew but keeps more frames.

---

## 9. Dataset format

The exporter writes LeRobot `codebase_version = "v2.1"`:

```text
data/lerobot/needle_transfer/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   ├── stats.json
│   └── tasks.jsonl
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── episode_000001.parquet
└── videos/
    └── chunk-000/
        ├── observation.images.left/
        │   ├── episode_000000.mp4
        │   └── episode_000001.mp4
        └── observation.images.right/
            ├── episode_000000.mp4
            └── episode_000001.mp4
```

### Frames

| Feature | dtype | Shape | Notes |
|---|---|---|---|
| `observation.state` | `float32` | `[D]` | joints then pose, per arm |
| `action` | `float32` | `[D]` | next state (placeholder) |
| `observation.images.left` | `video` | `[H, W, 3]` | H.264, `yuv420p` |
| `observation.images.right` | `video` | `[H, W, 3]` | H.264, `yuv420p` |
| `timestamp` | `float32` | `[1]` | `frame_index / fps` |
| `frame_index` | `int64` | `[1]` | `0..N-1` per episode |
| `episode_index` | `int64` | `[1]` | global episode id |
| `index` | `int64` | `[1]` | global frame id |
| `task_index` | `int64` | `[1]` | index into `tasks.jsonl` |

`D` is the sum over arms of `len(measured_js.position) + 7`.

### Metadata

- `info.json`: version, fps, splits, path templates, feature schema, and video
  codec/pixel format. `total_videos` counts two videos per episode.
- `episodes.jsonl`: one `{episode_index, tasks, length}` row per episode.
- `tasks.jsonl`: one `{task_index, task}` row per unique task string.
- `episodes_stats.jsonl`: per-episode min/max/mean/std/count for every feature.
- `stats.json`: dataset-level pooled statistics (weighted mean and pooled
  variance, matching LeRobot's aggregation rule).

Videos are encoded with `libx264`, CRF 18, `yuv420p`, and
`-movflags +faststart`. The exporter probes the encoded file and writes the
actual codec, pixel format, dimensions, and fps into `info.json`; it does not
trust the requested settings.

---

## 10. Verification and loading

### 10.1 Built-in verification

```bash
uv run --no-sync dvrk-lerobot verify --dataset data/lerobot/needle_transfer
```

The verifier checks:

- `codebase_version`, episode lengths, `total_frames`, `total_tasks`, and
  `total_videos`;
- parquet column order, fixed-size state/action lists, `frame_index`,
  `episode_index`, global `index`, task indices, and `timestamp = k / fps`;
- every MP4 exists and its dimensions match `info.json`;
- `episodes_stats.jsonl` has one row per episode.

It does not decode every video frame; ffprobe reads the container metadata.
Use a full LeRobot loader as the final compatibility check in the training
environment.

### 10.2 Loading with LeRobot

Install LeRobot in a **separate** environment used for training. Avoid adding
its PyTorch stack to the ROS acquisition environment.

```bash
# Example training environment, separate from this repository's .venv
python -m venv ~/envs/lerobot
source ~/envs/lerobot/bin/activate
pip install "lerobot==0.3.3"
```

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="local/needle_transfer",
    root="data/lerobot/needle_transfer",
)
print(dataset)
sample = dataset[0]
print(sample["observation.state"].shape)
```

LeRobot 0.3.3 uses the v2.1 dataset codebase. Newer LeRobot releases may default
to the v3.0 writer while still reading v2.1; check your installed version. The
exporter targets the v2.1 on-disk schema and does not depend on a particular
LeRobot Python API.

The synthetic integration dataset produced by this repository was loaded with
`lerobot==0.3.3` (CPU Torch, PyAV video backend) during development; all samples,
including both video streams, decoded successfully. Repeat this smoke test with
the exact version used for training.

---

## 11. Testing

```bash
uv run --no-sync pytest
uv run --no-sync ruff check src tests
```

Test layers:

- `tests/unit/test_config.py`: YAML validation and failure messages.
- `tests/unit/test_session.py`: raw session round-trip and malformed input.
- `tests/unit/test_alignment.py`: greedy stereo pairing, state construction,
  timestamps, dropped robot frames, missing pose data.
- `tests/unit/test_stats.py`: numeric statistics and pooled aggregation.
- `tests/integration/test_export.py`: two synthetic sessions, MP4 encoding,
  parquet schema, metadata, export refusal, and `verify`.

The integration test requires `ffmpeg` with `libx264`.

---

## 12. Assumptions and limitations

1. **One raw session is one episode.** There is no in-recording start/stop
   trigger or episode splitter yet. Record separate sessions for separate
   demonstrations.
2. **`action` is a placeholder.** It is the next measured state, not the
   commanded teleoperation action. Replace it with `servo_jp`, `setpoint_js`, or
   the master-side command before training a policy that needs true actions.
3. **Software stereo synchronisation.** Each eye is timestamped in user space;
   there is no hardware trigger or genlock. Check
   `alignment.stereo_tolerance_ms` and the alignment notes.
4. **Fixed-rate timeline.** The output timeline is `k / fps`, not the raw
   capture instants. Large frame drops are reported as a span mismatch but are
   not repaired. The raw JPEGs retain the true capture times for audit.
5. **JPEG intermediate.** Raw frames are JPEG quality 95 and are re-encoded to
   H.264 CRF 18. This is convenient and compact, but not lossless. Add a
   PNG/FFV1 raw mode if the experiment requires lossless intermediates.
6. **No image rectification or calibration.** The videos are the raw camera
   streams. Intrinsics, extrinsics, and rectification maps are not stored in the
   LeRobot dataset.
7. **Only positions are exported.** `measured_js.velocity`, `measured_js.effort`,
   `measured_cv`, `measured_cf`, and other CRTK streams are not stored yet.
   Extend `acquire/node.py` and `acquire/session.py` before adding them to the
   state vector.
8. **No rosbag2 copy.** The raw session format is intentionally small and
   inspectable. If you also need the full ROS graph, record a rosbag alongside
   it.
9. **Single-machine, wall-clock acquisition.** `use_sim_time` must be false.
   The header-stamp/wall-clock offset is checked by `doctor` and warned about by
   the recorder.
10. **H.264 only.** The exporter validates `libx264` output. AV1/HEVC support
    would only require changing `dataset/video.py` and the schema expectations.

---

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: rclpy` | `.venv` was created without system site packages | remove `.venv`, rerun `scripts/setup_env.sh` |
| `cannot open /dev/video0` | permissions, device busy, or wrong node | `sudo usermod -aG video $USER`, log out/in; close other camera apps |
| `negotiated WxH, expected ...` | camera does not support the requested mode | check `v4l2-ctl --list-formats-ext`, update YAML |
| `no data within 20s` | dVRK not running or wrong topic names | `ros2 topic list`, `ros2 topic hz`, update YAML |
| `fewer than two stereo pairs` | cameras not producing frames, or tolerance too tight | run `doctor`, check devices, raise `stereo_tolerance_ms` |
| `no cp samples` | wrong `measured_cp` topic or no pose publisher | `ros2 topic type` / `ros2 topic hz` |
| header stamp skew warning | `use_sim_time` enabled or clock mismatch | set `use_sim_time=false`, check `date` and ROS time |
| `ffmpeg not found` | ffmpeg absent | `sudo apt install ffmpeg` |
| `libx264` encode failure | ffmpeg built without x264 | install Ubuntu's `ffmpeg` package |
| `output already exists` | export target is not new | choose a new `--output` or move the old dataset |
| `export failed ... partial output kept` | inspect the partial directory | fix the reported error, remove partial, re-export |
| ROS log permission error | `~/.ros` not writable | scripts already set `ROS_LOG_DIR` to `./.ros`; override it if needed |

For a minimal camera-only check without ROS:

```bash
ffplay -f v4l2 -input_format mjpeg -video_size 1280x1024 -framerate 30 /dev/video0
```

---

## 14. Next steps

Recommended changes before a large ex-vivo campaign:

1. Decide the action source. Subscribe to `/PSM1/servo_jp` or
   `/PSM1/setpoint_js` and store a `raw_action` stream; export `action` from
   that stream instead of next-state.
2. Add a foot-pedal or keyboard start/stop trigger and per-episode segmentation.
3. Decide whether JPEG quality 95 is sufficient or a lossless intermediate is
   required.
4. Characterise the stereo skew with a flashing LED or checkerboard before
   trusting the default 15 ms tolerance.
5. Record camera intrinsics/extrinsics and the dVRK arm-to-camera registration
   alongside the dataset (for example, in a sidecar `calibration/` directory).
6. Smoke-test one exported dataset with the exact LeRobot version used for
   training, including video decoding.

---

## 15. References

- dVRK: <https://github.com/jhu-dvrk/dvrk2>
- CRTK: <https://github.com/jhu-cisst/cisst-ros>
- LeRobot dataset format: <https://github.com/huggingface/lerobot>
- ROS 2 Humble: <https://docs.ros.org/en/humble/>
- uv: <https://docs.astral.sh/uv/>
