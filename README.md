# dVRK → LeRobot Dataset v2.1

Collect teleoperation demonstrations on the dVRK and turn them into a
[LeRobot v2.1](https://github.com/huggingface/lerobot) dataset.

The tool records two things at the same time:

- **Robot telemetry** from ROS 2: joint positions, jaw angle and Cartesian pose
  of each PSM (about 100 Hz per topic).
- **Stereo endoscope video** from two V4L2 capture devices (1920x1080, 30 fps).

It then aligns both streams in time and writes them in the LeRobot format.

```text
                ┌──────────────────────────┐        ┌────────────────────────────┐
 dVRK (ROS 2) ─►│ record                   │        │ convert                    │
 endoscope    ─►│ s / e / q, live preview  │──raw──►│ align in time, write       │──► LeRobot v2.1
                │ data/raw/<run>/episode_N │        │ data/lerobot/<dataset>     │
                └──────────────────────────┘        └────────────────────────────┘
                 needs the robot                      offline, repeatable
```

| Step | Command | What it does |
|---|---|---|
| Check | `./scripts/doctor.sh` | Verifies ffmpeg, both cameras and all robot topics |
| Record | `./scripts/record.sh --task "..."` | Records episodes with a preview window; `s` start, `e` end, `q` quit |
| Convert | `./scripts/convert.sh --sessions ... --output ...` | Aligns robot data and video in time, writes the LeRobot dataset |
| Verify | `uv run --no-sync dvrk-lerobot verify --dataset ...` | Checks the dataset structure |

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
| ffmpeg | System package with `libx264` (`sudo apt install ffmpeg`) |
| Cameras | Two V4L2 devices: left eye `/dev/video0`, right eye `/dev/video2` |

## 2. One-time setup

```bash
sudo apt install ffmpeg
./scripts/setup_env.sh
```

`setup_env.sh` creates `.venv` **with access to the system ROS packages**
(`--system-site-packages`), so `rclpy` and the ROS message types can be imported,
and installs `numpy`, `opencv-python` (with GUI), `pyarrow` and `PyYAML` from
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
PASS left: 1920x1080 @ 30.0 fps fourcc='MJPG' (/dev/video0)
PASS right: 1920x1080 @ 30.0 fps fourcc='MJPG' (/dev/video2)

== ROS 2 topics ==
listening for 5.0s ...
PASS /PSM1/measured_js: 496 messages (99.2 Hz)
       first receive time minus header stamp: 0.4 ms
PASS /PSM1/jaw/measured_js: 495 messages (99.0 Hz)
...
RESULT: PASS
```

Fix any `FAIL` before recording (see [Troubleshooting](#8-troubleshooting)).
The "receive time minus header stamp" should be a few milliseconds; a value of
seconds means the robot and the PC clocks disagree.

### 3.3 Record episodes

```bash
./scripts/record.sh --task "needle pick"
```

| Option | Default | Meaning |
|---|---|---|
| `--task` | required | Task description stored with every episode of this run (becomes the LeRobot task text) |
| `--notes` | empty | Free-form notes stored in each episode's `meta.json` |
| `--root` | `data/raw` | Where run folders are created |
| `-c / --config` | `configs/record.yaml` | Configuration file |

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

### 3.4 Where the raw data goes

Each run creates one folder named by the time of its first episode, with one
sub-folder per episode:

```text
data/raw/20261027-192407/              ← one record run
├── episode_1/
│   ├── meta.json                      ← task, status, cameras, topics, joint names, counts
│   ├── robot.jsonl                    ← every robot message, one JSON per line
│   └── video/
│       ├── left.mkv                   ← left eye, H.264, every captured frame
│       ├── left_timestamps.txt        ← capture time (ns) of each frame, line i = frame i
│       ├── right.mkv
│       └── right_timestamps.txt
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

### 3.5 Convert to LeRobot

```bash
./scripts/convert.sh --sessions data/raw/20261027-192407 --output data/lerobot/needle_pick
```

| Option | Meaning |
|---|---|
| `--sessions` | One or more **run folders** and/or **episode folders**. A run folder expands to its `episode_1..N` in numeric order (`episode_10` after `episode_9`) |
| `--output` | New dataset folder. It must not exist yet; nothing is ever overwritten |
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
[align] 20261027-192407/episode_1: left frames: 464, stereo pairs: 463, aligned frames: 463, dropped for robot: 0
[align] 20261027-192407/episode_1: median jp/jaw/cp header skew: 0.6 ms
[align] 20261027-192407/episode_2: left frames: 372, stereo pairs: 372, aligned frames: 372, dropped for robot: 0
[align] 20261027-192407/episode_2: median jp/jaw/cp header skew: 0.6 ms
dataset: /home/zhang/zst/Open-H-Embodiment-dVRK-v2/data/lerobot/needle_pick
episodes: 2, frames: 835
```

How to read it:

| Field | Meaning | Healthy value |
|---|---|---|
| `left frames` | Frames recorded by the left camera | about 30 per second |
| `stereo pairs` | Left frames that found a right frame within `stereo_tolerance_ms` | close to `left frames` |
| `aligned frames` | Pairs that also found robot data within `robot_tolerance_ms`. **These become the dataset frames** | close to `stereo pairs` |
| `dropped for robot` | Pairs without robot data close enough in time | 0 or a few |
| `header skew` | Spread between the joint, jaw and pose samples used for one frame | a few ms |
| `capture span ... differs` (warning) | The cameras dropped frames during recording | absent |

Episodes in a run folder whose status is not `ok` are skipped with a
`[convert] skipping ...` line. If conversion fails, the half-written output is
kept as `<output>.partial` for inspection; delete it before retrying.

### 3.6 How the time alignment works

The robot (about 100 Hz per topic) and the two cameras (about 30 fps each) run
on independent clocks. `convert` builds one row per video frame:

1. **Stereo pairing.** Each left frame is paired with the nearest unused right
   frame. Pairs further apart than `stereo_tolerance_ms` are discarded.
2. **Frame time.** The time of a pair is the midpoint of its two capture times.
3. **Robot state.** For each arm, the nearest `jp`, `jaw` and `cp` samples (by
   ROS header time) are attached. If any is further away than
   `robot_tolerance_ms`, the frame is dropped.
4. **Output timeline.** Kept frames are numbered `0..N-1` with
   `timestamp = frame_index / fps` (what LeRobot expects). The true capture
   times stay in the raw folder.
5. **Video.** Only the kept frames are re-encoded into the dataset MP4s, so
   video frame *k* always matches parquet row *k*.

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
    └── observation.images.right/episode_000000.mp4 ...
```

| Feature | Type / shape | Content |
|---|---|---|
| `observation.state` | float32 `[28]` | per arm (PSM1, then PSM2): 6 joints, jaw, pose `x y z qx qy qz qw` |
| `action` | float32 `[28]` | next state (`state[t+1]`, last row repeated); a placeholder, see Limitations |
| `observation.images.left` | video `[1080, 1920, 3]` | H.264, yuv420p, 30 fps |
| `observation.images.right` | video `[1080, 1920, 3]` | H.264, yuv420p, 30 fps |
| `timestamp` | float32 | `frame_index / fps` |
| `frame_index` | int64 | `0..N-1` within the episode |
| `episode_index` | int64 | episode number in the dataset |
| `index` | int64 | global frame number across episodes |
| `task_index` | int64 | row in `tasks.jsonl` |

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
pip install "lerobot==0.3.3"      # uses the v2.1 dataset format
```

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/needle_pick", root="data/lerobot/needle_pick")
print(ds)
sample = ds[0]
print(sample["observation.state"].shape)          # torch.Size([28])
print(sample["observation.images.left"].shape)    # torch.Size([3, 1080, 1920])
```

Newer LeRobot releases default to the v3.0 format. Check that the version you
train with can read v2.1 datasets.

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
| `device` | `/dev/video0` (left), `/dev/video2` (right) | V4L2 device. Swap them if left and right are reversed |
| `width`, `height` | 1920, 1080 | Must be a mode the capture card supports |
| `fps` | 30 | Must equal `dataset.fps` |
| `fourcc` | `MJPG` | Pixel format requested from the card |
| `crf` | 18 | Quality of the **raw recording**. Lower = better and larger; 18 is visually lossless, 23 is about 2.5x smaller |

List the modes your card supports:

```bash
ffmpeg -hide_banner -f v4l2 -list_formats all -i /dev/video0
```

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
| `stereo_tolerance_ms` | 15 | Max time difference between a left and right frame |
| `robot_tolerance_ms` | 25 | Max time difference between a frame and the robot sample attached to it |

Changing `dataset` or `alignment` only needs a new `convert`; the raw data does
not have to be recorded again.

---

## 7. Disk space and performance

| | Size |
|---|---|
| Raw recording, 1080p stereo, `crf: 18` | about 18 MB/s, **about 65 GB per hour** |
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
| `cannot open /dev/video0` | Device busy or no permission | Close other camera apps (including another recorder); `sudo usermod -aG video $USER`, then log in again |
| `negotiated 1280x720, expected 1920x1080` | Mode not supported by the card | Pick a listed mode (section 6) |
| Preview window does not appear / Qt `xcb` error | No display (e.g. SSH session) | Run from the desktop session, or `ssh -X` |
| Keys do nothing | Neither the preview window nor the terminal has focus | Click the preview window or the terminal |
| Many `dropped for robot` frames | Robot topics stalled during recording | Check `ros2 topic hz /PSM1/measured_js`; raise `robot_tolerance_ms` only if the gaps are small |
| `stereo pairs` much lower than `left frames` | The two capture cards started half a frame apart | Re-convert with `stereo_tolerance_ms: 20` |
| `capture span ... differs from nominal` | Cameras dropped frames (CPU or USB overload) | Close other heavy programs; check the USB connections |
| `session status is 'aborted'` | The episode was ended with `q` / Ctrl+C | Add `--allow-incomplete` if the episode is usable |
| `output already exists` | `--output` folder exists | Choose a new name, or delete the old dataset |
| `convert failed ... partial output kept` | Error during conversion | Read the message, delete `<output>.partial`, run again |
| `ModuleNotFoundError: lark` when running `pytest` | ROS's pytest plugin is loaded from `PYTHONPATH` | `env -u PYTHONPATH uv run --no-sync pytest` |

Check a single camera without the recorder:

```bash
ffplay -f v4l2 -input_format mjpeg -video_size 1920x1080 -framerate 30 /dev/video0
```

---

## 9. Limitations

- **`action` is a placeholder.** It is the next measured state, not the
  operator's command. Record a command topic (for example `/PSM1/setpoint_js`)
  before training a policy that needs true actions.
- **Software stereo sync.** Each eye is timestamped by the PC when its frame
  arrives; the capture cards are not hardware-synchronised.
- **No camera calibration.** Intrinsics, stereo extrinsics and
  camera-to-robot registration are not stored.
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
