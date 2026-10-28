# dvrk-touchx: teleoperate a PSM with a TouchX

Drive a dVRK PSM with a [3D Systems Touch X](https://www.3dsystems.com/haptics-devices/touch-x)
in place of an MTM. The TouchX pose is mapped onto the PSM the same way the dVRK
MTM-to-PSM teleoperation maps the MTM, and is sent to the PSM as `servo_cp`.
dVRK then solves the PSM inverse kinematics.

```text
 Touch X ──USB──► sawSensablePhantom ──/arm/measured_cp (500 Hz)──► dvrk-touchx ──/PSM1/servo_cp──────► dVRK PSM1
 (stylus,         (C++ 1 kHz haptic     /arm/front, /arm/back          retarget +    /PSM1/jaw/servo_jp     (IK, joint
  2 buttons)       loop, ROS 2 bridge)                                safety        ◄─/PSM1/measured_cp    control)
```

| Step | Command | What it does |
|---|---|---|
| dVRK | `ros2 run dvrk_robot dvrk_system -j <system.json>` (`dvrk_console_json` on older dVRK) | Powers, homes and exposes PSM1 on ROS 2 |
| TouchX driver | sawSensablePhantom ROS 2 node | Publishes the stylus pose and buttons |
| Teleop | `./scripts/teleop.sh` | Hold the front button to follow, press the back button to open/close the jaw |
| Record (optional) | `./scripts/record.sh --task "..."` | Records demonstrations while you teleoperate |

---

## 1. How it works

### 1.1 Frames

The dVRK surgeon console (see `arm.rst` and
`dVRK-transformations-surgeon-console.png` at the repository root) defines:

- `MTM/local/measured_cp`: MTM tip in the MTM base frame (forward kinematics).
- `base_frame` (console.json): MTM base to the stereo display (HRSV). The display
  frame follows the ISI convention: X to the operator's left, Y up, Z away from
  the operator.
- `MTM/measured_cp = base_frame · local/measured_cp`: MTM tip in the display frame.

On the patient side, each PSM's `measured_cp` is expressed in the camera frame
(via its own `base_frame`). The dVRK teleop assumes the display frame and the
camera frame line up, so moving your hand right moves the tool right on screen.

The TouchX takes the place of the MTM:

| dVRK MTM | TouchX here |
|---|---|
| `MTM/local/measured_cp` | `/arm/measured_cp` from sawSensablePhantom (OpenHaptics frame: X right, Y up, Z toward the user) |
| MTM `base_frame` in console.json | `rotation` in `configs/touchx.yaml` (TouchX to the frame of `/PSM1/measured_cp`) |
| MTM gripper | Back button (each press toggles open/closed) |
| Clutch pedal and operator presence | Front button (held = following) |

### 1.2 Retargeting

When the front button is pressed, the node records the TouchX position `p_M0` (rotated into the
PSM frame) and the measured PSM pose `P0`. For every new TouchX sample `M`:

```text
p_goal = p_P0 + scale · (p_M − p_M0)
R_goal = R_M · tip_rotation        (R_M rotated into the PSM frame)
```

Position is relative and orientation is absolute, as in `mtsTeleOperationPSM`: the tool
tip, as seen on the camera view, points the way the stylus points as seen by you.

- The first goal equals the current PSM pose. The tool then turns to the stylus
  orientation at `align_rate` (1 rad/s), and follows it directly once aligned
  (the log says `tool aligned with the stylus`). The position never jumps.
- Translation is scaled down (default 0.2, as in dVRK).
- The base translation cancels out. `tip_rotation` maps the stylus tip frame to
  the tool tip frame (see section 4).
- The roll about the tool axis is taken at the first engage so the shaft does not
  twist toward its limit, then kept until the teleop restarts. Only the pointing
  direction is absolute from the start.
- Releasing the front button stops sending goals. dVRK holds the last goal, so the PSM
  stays where it is. Move the stylus to a comfortable spot and press again,
  just like clutching on the MTM.

### 1.3 Commands sent to the PSM

| Topic | Type | When |
|---|---|---|
| `/PSM1/servo_cp` | `geometry_msgs/PoseStamped` | On every TouchX sample while following |
| `/PSM1/jaw/servo_jp` | `sensor_msgs/JointState` | With every `servo_cp`; moves toward `jaw_closed` or `jaw_open` (toggled by the back button) at up to 2π rad/s, starting from the measured jaw; while not following, only until the target is reached |

The pose goes out inside the TouchX callback, with no extra control loop. The
added latency is the driver's publish period (2 ms by default) plus a local DDS hop.

### 1.4 Safety checks

Following stops (the log says `released: ...`) when:

- the front button is released;
- the goal is more than `max_error` (2 cm) from the measured PSM position. This
  also catches a PSM that is not following, or a driver that publishes
  millimetres instead of metres;
- `/PSM1/measured_cp` is older than `max_age` (0.1 s).

Engaging is refused when there is no TouchX pose yet or no recent PSM pose.

---

## 2. Requirements

| Item | Note |
|---|---|
| OS / ROS | Ubuntu 22.04, ROS 2 Humble at `/opt/ros/humble` |
| dVRK | ROS 2 workspace at `~/ros2_ws` with `dvrk_robot` |
| TouchX driver | 3D Systems Touch device driver and OpenHaptics for Linux; check the device with `Touch_Diagnostic` |
| ROS 2 TouchX node | [jhu-saw/sawSensablePhantom](https://github.com/jhu-saw/sawSensablePhantom) built with ROS 2 support in the same workspace (see the [jhu-saw/vcs](https://github.com/jhu-saw/vcs) build instructions) |
| This repository | `.venv` created by `./scripts/setup_env.sh` (see the main README) |

---

## 3. Running, step by step

### 3.1 Start the dVRK

Start the dVRK system as usual, then power on and home the arms. **Do not enable
the dVRK console teleoperation for this PSM.** Only one node may send `servo_cp`;
if the console teleop is also running, the two fight for the arm.

Check that the PSM is publishing:

```bash
ros2 topic hz /PSM1/measured_cp
```

### 3.2 Start the TouchX driver

Start the sawSensablePhantom ROS 2 node (see its README for the exact command
and configuration file). Then check its topics:

```bash
ros2 topic list | grep -E "front|back"   # /arm/front /arm/back
ros2 topic hz /arm/measured_cp         # about 500 Hz
ros2 topic echo /arm/measured_cp       # positions should be in metres (|x| < 0.2)
ros2 topic echo /arm/front           # buttons: [1] while pressed
```

The namespace (`/arm` above) is the device name in the driver configuration.
If yours differs, for example `/left`, set `touchx:` in `configs/touchx.yaml`.

### 3.3 Start the teleoperation

```bash
./scripts/teleop.sh                         # uses configs/touchx.yaml
./scripts/teleop.sh -c configs/my_touchx.yaml
```

The script sources ROS 2 and the dVRK workspace (override with `ROS_SETUP` and
`DVRK_SETUP`), then runs `dvrk-touchx`. Expected output:

```text
[INFO] [dvrk_touchx_teleop]: ready: hold /arm/front to follow, press back to open/close the jaw
[INFO] [dvrk_touchx_teleop]: following, aligning the tool with the stylus
[INFO] [dvrk_touchx_teleop]: tool aligned with the stylus
[INFO] [dvrk_touchx_teleop]: released: clutch released
```

| Input | Action |
|---|---|
| Hold the front button | Tool turns to the stylus orientation, then follows the stylus |
| Release the front button | PSM stops and holds; reposition your hand (clutch) |
| Press the back button | Jaw toggles between `jaw_open` and `jaw_closed` |
| `Ctrl+C` | Quit; the PSM holds its last goal |

The back button works with or without the front button held, and the jaw keeps its
state while you clutch, so you do not drop what it holds. The jaw starts open.

### 3.4 Record demonstrations (optional)

The recorder only listens to PSM topics, so it can run next to the teleop in a
second terminal. The rest of the pipeline is unchanged (see the main README):

```bash
./scripts/doctor.sh
./scripts/record.sh --task "needle pick"      # s = start, e = end, q = quit
./scripts/convert.sh --sessions data/raw/<run> --output data/lerobot/<dataset>
```

---

## 4. First-time direction check

`rotation` must match how `/PSM1/measured_cp` is expressed. Do this once per
setup, with the instrument clear of anything:

1. Set `scale: 0.1` in `configs/touchx.yaml` and start the teleop.
2. Hold the front button and move the stylus slowly **right**, **up**, then **away from
   you**. Watching the endoscope view (or the PSM itself), the tool must move
   right, up, and deeper into the scene.
3. Rotate the stylus a little about each axis; the tool must rotate the same way and,
   once aligned, point the same way as the stylus.
4. If an axis is mirrored or swapped, pick the matching `rotation` below,
   restart, and repeat.

| `/PSM1/measured_cp` frame | `rotation` |
|---|---|
| ISI display/camera convention: X left, Y up, Z away (default) | `[[-1,0,0],[0,1,0],[0,0,-1]]` |
| OpenCV camera frame (e.g. from `dvrk_camera_registration`): X right, Y down, Z forward | `[[1,0,0],[0,-1,0],[0,0,-1]]` |
| No PSM `base_frame` (RCM frame) | The RCM-to-user rotation for your mount, composed with the TouchX axes |

`tip_rotation` needs no calibration. The stylus pen points along −z of its frame
(sawSensablePhantom puts the tip at −39 mm along z) and the tool tip along +z, so the
default `[[1,0,0],[0,-1,0],[0,0,-1]]` flips z. The roll between the stylus buttons and the
jaws is set automatically at the first engage.

If no row fits, fit `rotation` from your own movements, with the teleop stopped:

```bash
./scripts/calibrate_touchx.sh        # prompts for pairs, prints the rotation: block
```

It asks for six directions (right, left, up, down, toward you, away from you).
For each, move the stylus 3-5 cm that way, then move the PSM tip (hand clutch on
the arm) the same way. It prints both moves in mm so you can redo a wrong one,
then how many degrees each move is off after the fit.

`rotation` must be a proper rotation matrix (orthonormal, determinant +1);
otherwise the config is rejected.

---

## 5. Configuration reference (`configs/touchx.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `touchx` | `/arm` | Driver namespace: `measured_cp`, `front`, `back` |
| `psm` | `/PSM1` | PSM namespace: `measured_cp`, `servo_cp`, `jaw/servo_jp`, `jaw/measured_js` |
| `rotation` | ISI convention | TouchX base frame, 3x3 rotation into the PSM frame |
| `tip_rotation` | z flipped | Stylus tip frame to tool tip frame, 3x3 rotation |
| `scale` | `0.2` | PSM translation per TouchX translation |
| `jaw_open` | `0.8` | Jaw angle (rad) when open (initial state) |
| `jaw_closed` | `-0.2` | Jaw angle (rad) when closed; negative adds grip force |
| `max_error` | `0.02` | Release if the goal is farther than this from the PSM (m) |
| `max_age` | `0.1` | Release or refuse to engage if PSM pose is older than this (s) |
| `align_rate` | `1.0` | Tool rotation speed (rad/s) on engage until it matches the stylus |

All keys are required; unknown keys are rejected. To drive PSM2, copy the file
and set `psm: /PSM2`.

---

## 6. Troubleshooting

| Message or symptom | Cause and fix |
|---|---|
| `cannot engage: no data on /arm/measured_cp` | Driver not running or wrong namespace. Check `ros2 topic list` and `touchx:` |
| `cannot engage: no recent /PSM1/measured_cp` | dVRK not running, or wrong `psm:` |
| Pressing the front button does nothing, no log line | Wrong button topic, or the driver's button topic is not `sensor_msgs/Joy`. Check `ros2 topic echo /arm/front` |
| `released: goal is X mm from the measured PSM position` right after a small move | Driver publishes millimetres, or the PSM is not enabled/homed and does not follow |
| `released: goal is X mm ...` during fast motion | The PSM lags behind; move slower or raise `max_error` slightly |
| `released: /PSM1/measured_cp is stale` | dVRK stopped publishing or the PC is overloaded |
| PSM moves in a mirrored direction | Wrong `rotation`; redo section 4 |
| PSM shakes or both teleops move it | The dVRK console teleop is still enabled for this PSM; turn it off |
| Jaw does not move | PSM tool not engaged/homed; check `ros2 topic echo /PSM1/jaw/measured_js` |

---

## 7. Code layout

```text
src/dvrk_touchx/
├── cli.py        argument parsing, rclpy spin
├── config.py     YAML loading and validation (no ROS)
├── retarget.py   frame math and relative mapping (no ROS)
└── node.py       ROS 2 subscriptions, servo_cp and jaw commands, safety checks
```

The math and config are covered by `tests/unit/test_touchx.py`, which runs
without ROS:

```bash
uv run --no-sync pytest tests/unit/test_touchx.py
```
