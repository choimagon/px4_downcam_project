# PX4 Gazebo ROS 2 Down-Facing Camera

This project provides a reproducible PX4 SITL platform for the official
`x500_mono_cam_down` quadrotor. It starts the PX4 v1.17.0 model in the PX4
ArUco world, bridges the real Gazebo camera transport stream into ROS 2, and
opens a separate `rqt_image_view` window for the down-facing camera.

## Environment

- Ubuntu 22.04.5 LTS
- PX4 Autopilot `v1.17.0`
- ROS 2 Humble
- Gazebo Harmonic (`gz-sim8` 8.15.0)
- Harmonic-compatible `ros-humble-ros-gzharmonic` / `ros_gz_bridge`
- DDS: Cyclone DDS, selected by the launcher to carry the 1280x960 RGB camera
  frames reliably on this host

## Layout

```text
PX4-Autopilot/                    Official PX4 v1.17.0 source tree
config/down_camera_bridge.yaml    Live Gazebo-to-ROS bridge mapping
config/cyclonedds.xml             Large-frame DDS transport configuration
config/99-px4-downcam-dds.conf    Persistent 8 MiB host socket-buffer setting
scripts/run_all.sh                One-command launcher
scripts/stop_all.sh               Project-scoped cleanup
logs/                              PX4, bridge, viewer, launcher, validation logs
artifacts/screenshots/            Captured execution views
artifacts/px4_downcam_execution_report.pdf
```

## Run

From a new terminal, no `.bashrc` sourcing is required:

```bash
cd ~/px4_downcam_project
./scripts/run_all.sh
```

The launcher validates Ubuntu and dependencies, sources ROS 2 Humble itself,
starts PX4 with `PX4_GZ_WORLD=aruco`, discovers the actual `gz.msgs.Image`
topic, writes the matching bridge config, validates ROS image delivery, then
opens `rqt_image_view`. The ArUco world includes a collision-only elevated
test stand so the official down-facing camera remains above the marker while
normal Gazebo physics runs; neither the official vehicle nor its camera
orientation is altered.

Two GUI windows should be visible at the same time:

1. Gazebo Sim with the X500 and ArUco world.
2. `rqt_image_view` on `/down_camera/image_raw`.

## Stop

Press `Ctrl+C` in the launcher terminal, or use:

```bash
./scripts/stop_all.sh
```

The stop script first uses recorded process groups and only then matches a
Gazebo process whose world path is unique to this project.

## ROS camera interfaces

```text
/down_camera/image_raw   sensor_msgs/msg/Image
/down_camera/camera_info sensor_msgs/msg/CameraInfo
```

Useful checks (the launcher exports these DDS settings internally; source and
export them manually only when checking from a separate shell):

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$HOME/px4_downcam_project/config/cyclonedds.xml
ros2 topic type /down_camera/image_raw
ros2 topic hz /down_camera/image_raw
gz topic -l | grep -Ei 'image|camera'
```

## Troubleshooting

- If no GUI appears, confirm that the active desktop session exposes `DISPLAY`
  and inspect `logs/px4.log` for rendering errors.
- If a previous run was interrupted, run `./scripts/stop_all.sh` before
  launching again.
- The launcher uses the discovered Gazebo image topic rather than a fixed model
  instance suffix, so it remains valid when PX4 changes the suffix.
- HD ROS images may trigger Fast DDS buffer errors on this host. The launcher
  uses its installed Cyclone DDS configuration automatically; do not remove
  `config/cyclonedds.xml`. This project installed
  `/etc/sysctl.d/99-px4-downcam-dds.conf`, which persists the required 8 MiB
  socket limits. If it was removed, restore it with
  `sudo install -m 0644 config/99-px4-downcam-dds.conf /etc/sysctl.d/` and
  `sudo sysctl -p /etc/sysctl.d/99-px4-downcam-dds.conf`.

## Evidence

The four required screenshots are in `artifacts/screenshots/`. The rendered
execution report is `artifacts/px4_downcam_execution_report.pdf`.

## QR precision landing with reinforcement learning

This project also contains a real QR-based precision-landing pipeline. It
adds a decodable `QR` texture to the Gazebo world; it is not an
ArUco approximation. `landing_rl.vision.QrDetector` uses OpenCV
`QRCodeDetector` on the bridged `/down_camera/image_raw` frames and converts
the QR center to normalized camera-center error.

The QR pad is now **0.4 m × 0.4 m** (half of the previous 0.8 m square). A
blue 2 m ring and orange 7 m ring are visibly painted on the Gazebo floor. For
every demo the vehicle begins on the ground at a seeded random point strictly
between those rings, takes off, performs broad QR acquisition, and then relies
on the decoded image-center error for final learned centering and landing.

The training environment is domain randomized for annulus start position,
altitude, wind, image noise, and QR dropout. It compares three actual
Stable-Baselines3 algorithms: PPO, DDPG, and SAC. Each policy learns a residual
lateral velocity over a low-gain QR image-centering safety servo; a guarded
state machine permits descent only after 15 consecutive frames inside the
alignment gate.

Install the Python dependencies once (already installed on this host):

```bash
/usr/bin/python3 -m pip install --user -r requirements-rl.txt
```

Train PPO, DDPG, and SAC on the same randomized 2–7 m annulus task:

```bash
./scripts/train_qr_landing.sh --algorithms ppo,ddpg,sac --timesteps 50000 --eval-episodes 50 --seed 8127
```

The selected model is written to `models/best_qr_landing.zip`; metrics and the
PPO/DDPG/SAC comparison are saved in
`artifacts/rl_training/training_metrics.json`.

Each run writes all three candidate archives (`models/ppo_qr_landing.zip`,
`models/ddpg_qr_landing.zip`, and `models/sac_qr_landing.zip`) before selecting
the highest evaluated policy.

To run camera detection and policy inference against an already running
`run_all.sh` session without commanding PX4:

```bash
./scripts/run_qr_landing_inference.sh --model models/best_qr_landing.zip
```

For one reproducible end-to-end **SITL-only** annulus demo, choose a random
seed. This example spawns at a sample in the requested ring and records the
whole monitor, with Gazebo maximized in third-person:

```bash
./scripts/run_qr_landing_demo.sh \
  --model models/ppo_qr_landing.zip \
  --spawn-seed 7 \
  --gazebo-video-file artifacts/rl_training/ppo_annulus_qr_landing_third_person.mp4
```

`--gazebo-video-file` captures the entire visible **monitor** as an H.264 MP4,
while the demo temporarily disables rqt and maximizes Gazebo so the
third-person scene, 2 m/7 m rings, drone, and QR pad dominate the recording.

```bash
./scripts/run_annulus_landing_suite.sh
```

The suite records PPO, DDPG, and SAC in
`artifacts/rl_training/*_annulus_qr_landing_third_person.mp4`, extracts each
flight's metrics, and builds the local result site:
`artifacts/rl_training/annulus_landing_dashboard.html`.

## Moving QR tracking and landing

The moving-target experiment uses a second Gazebo world with the same 0.4 m
QR landing surface mounted on PX4's moving-platform controller. After a
50-second takeoff/acquisition staging period it follows a seeded, smooth
curved path: base speed, speed modulation, heading, and lateral curve vary
between runs but remain bounded. The flight controller receives the current
target velocity as a feed-forward term, continues lateral tracking through
descent, and pauses descent whenever the decoded QR leaves the alignment gate.

Train a separate moving-target policy set (the static-pad models remain
unchanged):

```bash
./scripts/train_qr_landing.sh \
  --algorithms ppo,ddpg,sac --timesteps 120000 --eval-episodes 60 --seed 20260827 \
  --model-suffix moving_qr_landing --metrics-file moving_qr_training_metrics.json
```

Create the three SITL demonstrations and their local comparison page:

```bash
./scripts/run_moving_qr_landing_suite.sh
```

It produces full-monitor H.264 third-person recordings in
`artifacts/rl_training/*_moving_qr_tracking_landing_third_person.mp4` and the
site at `artifacts/rl_training/moving_qr_landing_dashboard.html`. All moving
target evidence is PX4 SITL/Gazebo evidence only; it is not a hardware-flight
claim.

## Random curved path and wavy QR deck

The current moving-target scenario replaces the fixed straight line with a
seeded smooth trajectory. Each run changes its base heading, speed modulation,
and gentle lateral curve; all values remain bounded (roughly 0.06–0.12 m/s)
so the target remains physically trackable. The QR landing deck also rolls,
pitches, yaws, and heaves with small deterministic boat-like waves. A single
motion-profile seed is shared by the training environment, Gazebo plugin, and
online velocity feed-forward, so every recorded flight is reproducible.

Train the new policy archives:

```bash
PYTHONPATH=. /usr/bin/python3 -m landing_rl.train \
  --algorithms ppo,ddpg,sac --timesteps 180000 --eval-episodes 60 --seed 20260828 \
  --model-suffix wavy_qr_landing --metrics-file wavy_qr_training_metrics.json
```

Record all three algorithms on distinct randomized paths and build the results
site:

```bash
./scripts/run_wavy_qr_landing_suite.sh
```

This produces `artifacts/rl_training/wavy_qr_landing_dashboard.html`, three
full-monitor H.264 third-person MP4s, the per-flight logs/CSVs, and
`wavy_qr_demo_results.json` with the exact spawn and trajectory seeds.

## All-MuJoCo training and ONNX inference

The MuJoCo pipeline is separate from PX4/Gazebo at runtime: both learning and
inference run in MuJoCo.  Its rendered vehicle is not a primitive stand-in:
`scripts/prepare_mujoco_x500_assets.py` converts the checked-in Gazebo
`x500_mono_cam_down → x500 → x500_base` frame, motor, and propeller assets to
MuJoCo OBJ/STL, and the X500 SDF's 2.0 kg base mass/inertia are used.  The
world models gravity, force-based velocity control, ground contact, a
moving/wavy 0.4 m QR deck, and random 2–7 m annulus starts. PPO, DDPG, and SAC
train in that physical world.

```bash
./scripts/train_mujoco_qr_landing.sh
```

The resulting SB3 archives are written as
`models/{ppo,ddpg,sac}_mujoco_moving_qr.zip`.  Export them to ONNX and perform
fresh **ONNX Runtime** inference in MuJoCo, including a third-person camera
that recomputes its look-at point from the drone position on every rendered
frame:

```bash
./scripts/run_mujoco_onnx_suite.sh
```

It writes one MuJoCo ONNX MP4, PNG snapshot, CSV trace, and downloadable ONNX
model per algorithm under `artifacts/rl_training/`, then builds
`artifacts/rl_training/mujoco_qr_landing_dashboard.html`.

`--enable-actuation` is restricted to the demo/inference command and should
never be pointed at a physical vehicle without independent safety review,
flight-boundary validation, and a hardware kill path.

## PX4 SITL + MuJoCo HIL 배포 검증 (현재 추론·영상 기준)

MuJoCo 학습 환경과 PX4 비행제어를 하나로 섞어 구현한 것이 아니다.
학습은 MuJoCo에서 수행하고, 최종 PPO/DDPG/SAC ONNX 추론과 비학습 카메라 MPC
평가·영상 생성은 별도 실행되는 **프로젝트 내부 PX4 SITL 바이너리**와 MAVLink
HIL로 연결한다.
따라서 학습 정책이 모터 PWM을 직접 내지 않는다. 정책은 QR 추적용 제한된
3D 속도 보정 `Δvx, Δvy, Δvz`를 제안하고, PX4가 EKF2·속도/자세/고도 제어·
제어할당·모터 네 개의 출력을 계산한다.

```text
MuJoCo X500 IMU / barometer / GPS
    └─ HIL_SENSOR + HIL_GPS (MAVLink) ────────────────► PX4 SITL EKF2

QR/PnP + PX4 자체 상태 ─┬─► PPO/DDPG/SAC ONNX (3D 제한 residual)
                          └─► camera/PnP MPC (8-step 3D 속도 계획)
                                       └─► companion velocity target
    └─ SET_POSITION_TARGET_LOCAL_NED (vx, vy, vz) ───► PX4 Offboard control

PX4 position/attitude control + control allocation
    └─ HIL_ACTUATOR_CONTROLS (motor 0..3) ───────────► MuJoCo X500 physics
```

### 비학습 Camera/PnP MPC 기준선

MPC는 PPO/DDPG/SAC와 별도로 비교하는 **비학습** 기준선이다. ONNX나 Go2 상태를
읽지 않고, 하향 카메라/PnP에서 얻은 QR 대비 3D 위치·상대속도와 PX4 EKF2의
드론 3D 속도만 사용한다. 100 ms 간격, 8 step horizon에서 PX4 수평/수직 속도
루프를 각각 0.38/0.32 s 1차 응답으로 예측하고, 수평 위치 오차(28.0)·수직 위치
오차(10.0)·수평 상대속도(1.5)·수직 상대속도(8.0)·명령 변화(0.04)·종단 위치
오차(80.0/24.0)의 가중 비용이 가장 낮은 `vx, vy, vz` reference를 선택한다.
5×5×3 후보 lattice, 수평 3.6 m/s·하강 0.65 m/s 상한을 사용하며, 매 100 ms
첫 명령만 내보내고 다시 푼다(receding horizon). Z 목표는 QR 중심이 아니라 stock
X500 스키드의 물리 접지 clearance이며, 그 높이보다 위에서는 최소 하강 후보 0.22 m/s를
MPC feasible set에 넣는다. 이는 별도 수직 제어가 아니라 3D MPC의 물리 제약이다.

PPO/DDPG/SAC도 동일하게 7D 카메라/자체수직속도 입력에서 3D residual
`Δvx, Δvy, Δvz`를 낸다. 이 3축 보정은 카메라/PnP 기준 reference에 합산된다.
QR가 보이지 않거나 QR 중심·상대속도 gate를 통과하지 못하면 safety governor가
하강만 hold한다. 이는 별도의 수직 제어기가 아니라, 잘못된 QR/접촉 추정에서
하강을 금지하는 최후 제한이다. 어떤 경우에도 RL/MPC가 모터 PWM, force/torque,
pose를 직접 출력하지 않고 PX4에는 Offboard `vx/vy/vz`만 보낸다.

### Gazebo PX4 X500에서 MuJoCo HIL로 옮긴 범위

이 경로는 Gazebo 전체를 MuJoCo로 변환해 PX4를 흉내 낸 것이 아니다. 기존
PX4 Gazebo X500의 기체 프레임·질량/관성·스키드 형상·로터 위치와 회전 방향을
MuJoCo 장면에 이식하고, 이미 빌드된
`PX4-Autopilot/build/px4_sitl_default/bin/px4`를 별도 프로세스로 실행한다.
각 실행은 X500 airframe `4001` 기반의 임시 PX4 rootfs를 만들고
`simulator_mavlink`만 선택한다. 기존 Gazebo 프로세스, PX4 소스 트리, 사용자
파라미터 파일은 수정하지 않는다.

| 경계 | MuJoCo 쪽 | PX4 쪽 |
| --- | --- | --- |
| 좌표 | world NWU, body FLU | local NED, body FRD |
| 센서 | 가속도계·자이로·자력계·기압·GPS | `HIL_SENSOR`, `HIL_GPS` 수신 후 EKF2 융합 |
| 상위 명령 | 카메라/PnP 기반 `vx`, `vy`, `vz` 목표 | MAVLink Offboard local-velocity setpoint 수신 |
| 저수준 | PX4가 낸 4개 모터 출력으로 힘/토크 계산 | multicopter 제어기와 X500 control allocation |
| 착지 접촉 | Go2·QR 판·X500 스키드의 MuJoCo 충돌 | 접촉값을 입력으로 사용하지 않음 |

MuJoCo는 PX4의 `HIL_ACTUATOR_CONTROLS` 네 출력을 X500의 실제 로터 위치,
추력 방향, yaw moment ratio로 합산하여 기체의 force/torque에 적용한다. 이는
정책이나 companion이 직접 비행 force, pose, velocity, PWM을 쓰는 경로가 아니다.
단, 개별 로터의 공력/워시와 Gazebo 플러그인 물리는 MuJoCo에서 별도로 재현하지
않는다. 그러므로 이 검증은 **PX4 EKF2·Offboard·multicopter control·control
allocation HIL 검증**이며, Gazebo 플러그인 또는 개별 로터 공력의 동등성 검증은
아니다.

### 재현과 감사 산출물

먼저 프로젝트 내부 PX4 SITL 바이너리가 필요하다.

```bash
cd PX4-Autopilot
make px4_sitl_default
cd ..
bash scripts/train_px4_flat_hil_3d.sh
python3 scripts/run_px4_flat_hil_suite.py
python3 scripts/build_go2_back_qr_dashboard.py
python3 -m http.server 9371 --bind 0.0.0.0 --directory artifacts/rl_training
```

`run_px4_flat_hil_suite.py`는 PPO/DDPG/SAC/MPC × 초급/중급/고급의 12개 독립
실행을 만든다. 각 실행은 H.264 MP4, PNG, trace CSV, metrics JSON, PX4 텍스트
로그, ULog를 남기며 `artifacts/rl_training/px4_flat_hil_suite.json`에 통합된다.
유효한 실행은 PX4 armed/Offboard 상태, HIL 센서·GPS·모터 메시지 수, EKF
innovation, HIL 시간 단조성, Go2 발 접지/낙상, 그리고 양쪽 X500 스키드의 물리
접촉을 함께 기록한다.

### 실제 PX4 기체 연동 가능성 분석

현재 구조는 실기 companion 구조와 같은 경계를 사용하므로, 실기 연결의 출발점으로
사용할 수 있다. MuJoCo HIL에서 이미 검증하는 것은 실제 PX4의 EKF2 입력 경로,
Offboard `vx/vy/vz` 명령, 자세/추력 계산, 모터 할당이다. 실기에서는 MuJoCo HIL
송신부를 다음 입력 어댑터로 교체하면 된다.

1. 하향 카메라의 QR corner 검출·camera calibration·`solvePnP`로 정책의
   `float32[7]` 관측을 만든다.
2. PX4 `vehicle_local_position`의 유효한 수직속도를 NED 부호에서 학습 좌표계로
   변환한다.
3. ONNX 출력을 같은 안전층으로 제한하고 MAVSDK/ROS 2 companion에서 Offboard
   local-velocity setpoint만 보낸다.
4. QR 미검출, 카메라/상태 timestamp 지연, Offboard loss에 대해 hold/상승/RTL과
   독립 kill path를 둔다.
5. 정지 QR 패드, tether/bench, 저속 이동 패드 순서로 검증한 뒤에만 Go2 이동
   착륙을 시험한다.

따라서 이 저장소의 결과는 실제 기체 비행 인증이나 안전 보증이 아니다. 특히 실제
카메라 QR detector/solvePnP, 외부보정, 통신 지연, vibration, failsafe는 별도로
구현·시험해야 한다. 그러나 정책이 raw motor를 우회하지 않고 PX4의 표준 Offboard
속도 인터페이스를 사용하므로, 이 조건들을 충족하면 실제 PX4 기체 연동에 활용할 수
있는 구조다.

## Unitree Go2 dorsal QR-deck landing

`third_party/unitree_mujoco` contains a sparse checkout of Unitree's official
Go2 MJCF and meshes.  The Go2 back-QR environment retains the real 12-DoF
Go2 links, joints, torque limits, and foot contacts.  Its low-profile 0.22 kg
QR mount is a rigid child of `base_link` (no free joint): two dorsal rails and
front/rear cross braces keep the 20 mm QR deck out of the hip/leg sweep
volume.  The natural locomotion controller uses a 58% duty-factor diagonal
trot, sagittal two-link IK, stance-foot world locking, smooth Hermite swing,
60/2 target-velocity PD, and a retrained 450-state/12-action Go2 PPO residual.
The deployed PPO residual is conditioned by a fixed 0.50 safety gain before
the single 0.18 rad joint-residual mapping; this gain passed randomized
held-out ablation against both raw-policy and zero-residual baselines.
The low-level reward measures stance-foot slip, 0.32 m body height, diagonal
contact timing, body attitude, action size, and action rate.  Go2 root force
and torque are identically zero: propulsion and balance come only from the 12
joint actuators and physical foot contacts.

The X500 uses the same two continuous skid-rail collision shapes as PX4's
Gazebo `x500_base`: each is 0.25 × 0.015 × 0.015 m at body-frame
`x = 0`, `y = ±0.132 m`. Their sole is at body `z = -0.22759951 m`, aligned
within 1 mm of the imported stock X500's rendered skid sole. The collision
rails are transparent, so only the original stock mesh is rendered, and both
rails contact the visible 36 cm QR deck with 3D sliding contact. The X500 has
no landing-leg touch, load, or contact sensor; the rails are collision shapes
only. MuJoCo rail-contact count is used only for training episode termination
and its terminal reward label, plus offline evaluation. Count, normal force, and
penetration remain explicitly offline physics diagnostics. No dense reward,
policy observation, or flight-controller term reads a landing-leg contact
value. Stable landing additionally requires both physical skid rails,
relative height no greater than 0.245 m, centre error within the configured
landing limit, and deck-relative speed below 0.40 m/s. The accepted MuJoCo
soft-contact penetration gate is 2 mm; the
exact maximum is retained in every evaluation profile and inference CSV rather
than hidden or converted into a sensor.

The earlier visible skid-through-deck defect came from a 101.6 mm mismatch:
the old collision sole sat that far above the imported visual skid sole, so
physics contact stopped the body while the rendered skids continued through
the deck. The hand-made cylinders were replaced by the two continuous stock
Gazebo skid rails, sharing the rendered sole plane. Their duplicate collision
render and the black camera housing/lens placeholders are hidden; the named
MuJoCo `down_camera` remains active.

The X500 landing policy has a strict stock-sensor-only 7-value observation:
down-camera QR centre u/v, solvePnP depth, QR-valid flag, QR centre-rate u/v,
and PX4-estimated vertical velocity. Those six camera values plus vertical
velocity are the policy's complete input. Go2/base/pad pose, velocity, route
command, simulator target coordinates, and invented landing-gear sensors are
excluded from both policy input and drone control. The flight controller may
additionally use the X500's own position, velocity, attitude, angular-rate,
and accelerometer channels, corresponding to GNSS/IMU/PX4 state estimates on
hardware. In this MuJoCo environment these are noisy 50 Hz sample-and-hold
PX4-output surrogates built from explicit frame-position, frame-velocity,
attitude, and gyro sensors; this is not a claim that a real EKF2 or barometer
model is running in the simulation.

The stock camera contract is 1280×960 at 30 Hz with 99.7°×83.27° FOV and a
0.10 m near plane. While QR is not visible, the X500 uses only its own
position/altitude estimate surrogate and elapsed mission time to fly a
forward-corridor lateral sweep at a decodable altitude. Near touchdown the
controller detects a possible impact without a leg sensor by subtracting the
body-Z specific force predicted from its known collective-thrust command from
the measured body-Z IMU specific force. It latches only inside the recent
final-vision window, below 0.25 m visual height, with at least 4.0 m/s²
innovation and at most 0.45 m/s estimated vertical speed. During the 0.35 s
settle interval, collective is reduced to 88% of hover instead of adding an
upward arrest impulse that can bounce the leading skid. A simulation episode
that has not reached the offline stable
contact terminal climbs at 0.45 m/s until QR depth reaches 0.30 m, then earns
a fresh visual-alignment streak. Contact count is not an input to this
controller state machine.

At corrected skid touchdown the camera-to-marker depth remains about 0.161 m,
which is beyond the 0.10 m near plane, so the QR stays in optical range through
normal contact. For a real detector dropout or brief occlusion, the controller
holds the last visual pose/velocity for at most 2.0 s (including one-frame
reacquisition flicker) and continues at 0.16 m/s relative descent; after three
retries that is reduced to 0.14 m/s. Learned lateral authority is
projected onto the inward QR-error direction, tapers from its 0.001 m/s maximum
below 1.20 m, and is identically zero inside 0.45 m. Thus no saturated actor
can push outward/tangentially or override final camera/IMU landing control.
Training uses a 0.002 m/s exploration envelope; every held-out evaluation,
ONNX export and inference run uses the stricter 0.001 m/s deployment envelope.
Simulator QR geometry is confined to a 30 Hz camera emulator that emits noisy,
sample-and-held solvePnP translation and camera-relative rotation; the world
attitude reference is reconstructed from that PnP output and the X500's own
attitude estimate. Rotation-vector noise uses
`sigma_deg = 0.15 + 0.03 * pnp_depth_m` per axis, clipped at three sigma.
Simulator ground truth is otherwise retained solely for
training terminal reward/termination and offline evaluation metrics.

```bash
./scripts/train_go2_back_qr.sh
./scripts/run_go2_back_qr_suite.sh
```

This writes three MuJoCo-trained policy archives,
`models/{ppo,ddpg,sac}_go2_back_qr.zip`, then exports them to ONNX and records
nine H.264 1920×720 30 fps videos (PPO/DDPG/SAC × easy/medium/hard).  Every video
contains synchronized Go2/X500 third-person and attached down-camera views.
The Korean result page is
`artifacts/rl_training/go2_back_qr_landing_dashboard.html`.

## 최신 지형 검증 기준 (한국어)

이 절은 위의 이전 실험 설명보다 우선하는 현재 공개 기준입니다. 지형은 **10°가 아니라
10% grade**이며, 각도는 다음과 같습니다.

$$
\tan(\theta)=0.10,\qquad \theta=\arctan(0.10)=5.7106^\circ
$$

경사 상승/하강은 길이 16 m의 실제 MuJoCo 회전 box 충돌면이고, Go2는 12 m 코스를 통과한
뒤 지형 끝으로 걸어 내려가지 않도록 감속·정지합니다. 요철은 카메라 배경이나 수직 블록이
아닌 폭 2.4 m의 연속 heightfield 충돌면입니다. Go2 본체와 QR 판의 보수적 외곽이 이 유한
충돌면 안에 있을 때만 성공 착륙을 허용합니다. 전방 경계 접근 시 보행 명령을 감속하며,
한 프레임이라도 이탈하면 성공은 되돌릴 수 없게 차단되고 해당 replay는 공개 검증에서 제외됩니다.

| 구간 | 실제 높이 진폭 | 검증된 기준 보행 |
| --- | ---: | --- |
| 10% 오르막 | 5.71° | 12.01 m 통과, 낙상 0 |
| 10% 내리막 | 5.71° | 12.01 m 통과, 낙상 0 |
| 요철 1단계 | 24 mm | 12.00 m, 낙상 0 |
| 요철 2단계 | 48 mm | 9.16 m/15 s, 낙상 0 |
| 요철 3단계 | 80 mm | 10.81 m/15 s, 낙상 0 |

위 수치는 기준 seed의 독립 Go2 물리 보행 값이다. 결합 X500 착륙 영상에서는 Go2 낙상 0,
최대 기울기 40° 이하, Go2·QR 판 코스 이탈 0, root wrench 0, 좌/우 스키드 두 개의 물리 접촉, 최대 관입 2.1 mm
이하를 모두 통과한 seed만 공개한다. 이는 고정된 물리 replay 검증이며, ±2% 액추에이터·마찰
무작위화에 대한 실기 강건성 인증이라고 주장하지 않는다.

### Go2 제어 상태·출력·물리 경계

legged-loco 호환 PPO 후보의 한 프레임은 45개, 현재+9개 history는 450개다.

$$
o_t=[\omega_B(3),\operatorname{rpy}_B(3),v^{cmd}(3),q-q_{stand}(12),\dot q(12),a_{t-1}(12)]
$$

출력은 12개 관절 잔차이고, 실제 토크는 다음 PD 제어로 발 접촉을 통해서만 전달된다.

$$
q_t^*=q_t^{ref}+0.18\,g\,a_t^{PPO},\qquad
\tau_t=\operatorname{clip}\{60(q_t^*-q_t)+2(\dot q_t^*-\dot q_t)\}
$$

새 10% 지형에서 재학습한 PPO 후보는 낙상·전진·경로 이탈 기준을 충족하지 못해
`*_candidate.zip`으로 보존하고 **공개 영상에는 사용하지 않는다**. 현재 15개 지형 영상의
Go2는 동일한 공식 MJCF의 12 모터, 네 발 접촉, IMU 및 Go2 자신의 odometry만 사용한
terrain-specific 기준 트로트다. 지형 높이, QR pose, 드론 상태, Go2 root 외력은 입력에
넣지 않는다. 특히 다음이 항상 성립한다.

$$
\boldsymbol w_{root}^{applied}=\boldsymbol 0_6
$$

경사에서는 지지 다각형을 grade 쪽으로 40 mm, 연속 요철에서는 20 mm 보정하고, 요철은
더 낮은 1.8 Hz 기반 cadence로 발을 더 오래 지지한다. 모든 전진은 관절 토크와 발-지면
마찰에서 나온다.

### X500 PPO/DDPG/SAC의 실제 입력·출력

세 드론 정책은 서로 같은 `float32[7]`을 입력으로 받고 3D 속도 잔차
`[Δv_x, Δv_y, Δv_z]`를 출력한다. XY는 카메라/PnP 기반 기준 추적을 보정하고,
Z는 가시 정렬이 성립한 하강 기준속도를 보정한다. 최종 접근의 시각 gate·속도 한계·충격
복구는 정책 밖 safety governor가 맡으며, PX4가 최종 `vx/vy/vz`로 자세·collective·모터를 계산한다.

| # | 입력 | 실제/시뮬레이터 획득 경로 | 가공 |
| ---: | --- | --- | --- |
| 0, 1 | QR 중심 `u, v` | 하향 RGB → QR corners | 이미지 중심 기준 `[-1,1]` |
| 2 | PnP 깊이 | 23 cm QR + camera intrinsics | `min(1,z/8)` |
| 3 | `detected` | decoder/PnP 유효성 | 0 또는 1 |
| 4, 5 | 중심 변화율 | 연속 검출 timestamp 차분 | clip ±5/s, 저역통과 |
| 6 | 드론 수직속도 | 드론 자체 GPS/IMU/PX4 estimator | clip ±3 m/s |

Go2 위치·속도·관절, 지형 정답 높이, QR의 simulator 월드 pose, 착륙다리 접촉/힘/관입은
드론 7D 입력이 아니다. 이들은 학습 reward/종단 라벨 또는 `offline_sim_*` 사후 지표에만
쓴다. QR 미검출 시에는 드론 자신의 EKF pose와 경과시간으로만 탐색 회랑을 비행한다.

PPO, DDPG, SAC의 대표 목적함수는 다음과 같으며, 모든 영상에서는 탐색 noise를 끄고 ONNX
Runtime 결정론 출력을 사용한다.

$$
L_{PPO}=-\mathbb E[\min(\rho_tA_t,\operatorname{clip}(\rho_t,0.8,1.2)A_t)]
+0.5\mathbb E[(V_\theta-R_t)^2]
$$

$$
L_{DDPG,Q}=\mathbb E[(Q_\phi(o,a)-y)^2],\quad
L_{DDPG,\mu}=-\mathbb E[Q_\phi(o,\mu_\theta(o))]
$$

$$
L_{SAC,Q_i}=\mathbb E[(Q_{i,\phi}(o,a)-y)^2],\quad
L_{SAC,\pi}=\mathbb E[\alpha\log\pi_\theta-\min(Q_1,Q_2)]
$$

| 기법 | 주요 설정 |
| --- | --- |
| PPO | `lr=2.5e-4`, rollout 512, batch 128, 10 epoch, $\gamma=0.997$, GAE 0.96, clip 0.20 |
| DDPG | `lr=3e-4`, replay 180k, batch 256, $\tau=0.01$, warm-up 2k, action noise 0.18 |
| SAC | `lr=3e-4`, replay 180k, batch 256, $\gamma=0.997$, $\tau=0.01$, entropy 자동 조정 |

### 15개 동기화 영상과 재현

아래 명령은 PPO/DDPG/SAC × 오르막/내리막/요철 1·2·3단계, **15개**의 1920×720 H.264 MP4를
생성한다. 요철 영상의 왼쪽 큰 화면은 Go2 발·실제 heightfield를 가까이 보는 3인칭이고,
같은 5 ms MuJoCo 상태의 전체 X500 3인칭을 동기화된 삽입 화면으로 함께 보인다. 오른쪽은
X500 하향 카메라이며, 하단뷰는 검정 레터박스 없이 전체 패널을 채운다.

```bash
python3 -B scripts/go2_terrain_landing_suite.py
python3 scripts/build_go2_back_qr_dashboard.py
python3 -m http.server 9371 --bind 0.0.0.0 --directory artifacts/rl_training
```

브라우저 주소는 로컬에서는
`http://localhost:9371/go2_back_qr_landing_dashboard.html`이고, Tailscale 장비에서는
`http://<이-호스트의-tailscale-ip>:9371/go2_back_qr_landing_dashboard.html`이다.
`localhost`는 접속하는 장비 자기 자신을 뜻하므로 원격 Tailscale 장비 주소로는 사용할 수 없다.

대시보드는 MP4/PNG/CSV/receipt JSON을 함께 제공한다. receipt에는 실제 선택 seed, ONNX SHA-256,
Go2 기준 보행 종류, 스키드 접촉/관입, 기울기와 root-wrench 검증 결과가 남는다. 대용량
체크포인트·ONNX·MP4는 Git에서 제외하고 재현 명령으로 생성한다.
