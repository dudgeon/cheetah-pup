# Published actuator model: CPU integration

The workbench now runs the actual published BAM XL330 M6 model in MuJoCo CPU
forward dynamics. This replaces the arbitrary ideal-PD actuator when explicitly
selected through `cheetah_pup.actuator`. It does **not** establish an accurate
stock XL330-M288-T simulation or settle the motor purchase decision.

The source is pinned to
[`Rhoban/bam@62bd8ce12154340be97e06f7f41a0ca8f116d967`](https://github.com/Rhoban/bam/tree/62bd8ce12154340be97e06f7f41a0ca8f116d967),
the BAM revision in our pinned Microduck RL lock. It lives in
`vendor/bam_microduck`; the older `vendor/bam` reference remains separate.
The adapter imports the upstream implementation directly and rejects a wrong
commit, tracked modifications, or an already loaded different BAM installation.
Upstream code and fit retain their Apache-2.0 license.

## Run it

From the repository root:

```bash
git submodule update --init vendor/bam_microduck
uv sync --locked
uv run pytest tests/test_actuator.py -q
uv run python -m cheetah_pup.actuator
```

The final command regenerates `reports/actuator-validation.json` from the current
`config/robot.json`. It includes a 3-second comparison with directly invoked BAM,
an electrical torque/speed diagnostic, a 5-second free-base stand simulation,
and a 60-second P400/P200 comparison. It also replays two crawl cycles with
free-base dynamics and reports whether the robot actually makes progress.
The report records the robot-configuration hash, exact BAM commit, parameter-file
hash, electrical settings, and timestep-dependent delay.

```python
import mujoco
from cheetah_pup.model import build_mjcf, load_config
from cheetah_pup.actuator import motor_mjcf, BamPositionController

xml = motor_mjcf(build_mjcf(load_config("config/robot.json")))
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
controller = BamPositionController(model, data)
targets = data.qpos[controller.qadr].copy()
controller.set_targets(targets)  # radians: FL, FR, RL, RR; roll, pitch, knee
for _ in range(2500):
    controller.step()  # update BAM, then a real MuJoCo dynamics step
```

Hold targets between 50 Hz command updates; step the controller at every physics
tick. After `mj_resetData` or `mj_resetDataKeyframe`, call `controller.reset()`.
`update()` is also exposed for callers that invoke `mujoco.mj_step` themselves.
Never drive the BAM motor `data.ctrl` with angle targets directly: its units are
torque, and the controller converts angle targets into those motor torques.

The stand checks use only the MJCF `stand` keyframe generated from
`home_q_rad` (currently roll 0, hip pitch 0.4, knee −0.8 radians).
They hold those joint targets constant in free-base forward dynamics.
The animation's `config.gait` height, support shifts and foot trajectories do
not drive these checks; a successful stand does not validate the animated gait.

## Explicit simulation contract

| Setting | Implementation and meaning |
|---|---|
| Candidate | Stock ROBOTIS XL330-M288-T, still provisional |
| Rail | Regulated **5.0 V**, required by config validation |
| Position P gain | **400**, a control-table integer; published stock XL330 and BAM constructor default, explicitly selected after comparing P200 |
| PWM ceiling | 1.0 normalized duty |
| BAM current limit | **1.75 A**, enforced upstream as a PWM duty-window constraint before the physical PWM bound |
| Command delay | **20 ms assumed**, a single queue of 10 physics steps at 2 ms |
| Friction | Published M6 load-dependent, directional and Stribeck terms, written to MuJoCo frictionloss/damping |
| Reflected inertia | Published fit armature **0.0018077432831600838 kg m²** per controlled joint |
| Torque limits | Electrical motor equation/current-window behavior; the primitive's arbitrary ±0.10 Nm cap is removed |
| Joint coordinates | Robot joint zeros and limits preserved; fitted pendulum `q_offset` is not used as a robot assembly offset |

The [locked motor model](https://github.com/Rhoban/bam/blob/62bd8ce12154340be97e06f7f41a0ca8f116d967/bam/actuator.py)
computes voltage from position error, applies the current-limiter duty window,
then clips to physically achievable PWM. Torque follows the DC motor equation
including back-EMF. At sufficiently high back-driving speed, the current window
cannot be attained with the available voltage; an extra fixed torque clamp would
change upstream behavior. Friction is a separate solver contribution.

`motor_mjcf` removes the ideal-PD gains and caps, clears preexisting joint damping,
friction, spring stiffness and armature, and emits one unit-gear motor per joint.
The adapter checks the compiled transmissions before attaching BAM. Thus the
motor's back-EMF damping, fitted viscous friction and rotor inertia are each
represented once. Physical rigid-body mass/inertia and joint limits are preserved.

The upstream controller constructor calls `mj_setConst`, which resets its supplied
data's positions to `qpos0`. The wrapper saves and restores the caller's physical
state around that call. Its reset also clears targets, command history, friction,
and the timestamp that upstream reset leaves unchanged.

The [Microduck reference](https://github.com/pollen-robotics/microduck_rl/blob/29e887ecfbf5d37144759e5a9f8a176dfb83d547/src/mjlab_microduck/robot/microduck_constants.py)
uses 3–6 **physics-step** actuation delay at a 5 ms physics tick, i.e. 15–30 ms.
Our 20 ms queue is an explicit simulation assumption within that interval, not a
measured latency. The pinned XL330 M6 file contains no fitted command-delay term.
No other action or observation delay is applied by this CPU adapter. Future
training must account separately for command cadence and delayed observations.

## What the evidence resolves, and what remains open

The manufacturer's [XL330-M288 manual](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/)
specifies 3.7–6 V operation, recommends 5 V, and lists P400 as the default. Its documented controller behavior
also exposes unresolved differences from this BAM revision: the manual gives a
P-gain divisor of 128, while BAM uses an empirical divisor of 256; default PWM
Slope 140 implies approximately 18 ms for a zero-to-5 V ramp, which BAM omits.
The manual describes input-rail current measurement and current limiting in
current/current-position modes; BAM's winding-current duty-window approximation
is not proof of equivalent stock position-mode behavior.

The [published fit](https://github.com/Rhoban/bam/blob/62bd8ce12154340be97e06f7f41a0ca8f116d967/bam/params/xl330/m6.json)
contains motor/friction coefficients but no acquisition voltage, exact XL330
variant, firmware version, or firmware-setting provenance. The
[upstream constructor](https://github.com/Rhoban/bam/blob/62bd8ce12154340be97e06f7f41a0ca8f116d967/bam/dynamixel/actuator.py)
defaults to 7.5 V, and Microduck's
[testbench replay script](https://github.com/pollen-robotics/microduck_rl/blob/29e887ecfbf5d37144759e5a9f8a176dfb83d547/scripts/testbench_sim2real.py)
uses 7.4 V. Neither is a valid rail to copy to our stock-servo design, nor proof
of the fit's acquisition conditions. Explicitly selecting 5 V prevents that
configuration mistake but cannot establish empirical accuracy.

There is no thermal model, gearbox wear/backlash model, PWM slew/profile generator,
I/D/feedforward controller, measured supply impedance, or sensor model here.
The electrical torque envelope is **not a continuous rating**. It must not be
used to infer a battery current budget, allowable sustained torque, or hardware
safety from the 1.75 A setting alone.

## Verification and next gate

The deterministic loaded-pendulum test compares separate adapter and directly
invoked upstream instances over 1,500 physics steps, with a sinusoidal command
and a later offset. Position, speed, torque, frictionloss and damping agree
exactly in the checked CPU environment. Regression tests also cover rail
rejection, the current/PWM ordering, preservation of a nonzero stand keyframe,
removal of duplicate dynamics, delay timing and deterministic reset.

This proves correct software integration with the selected BAM revision.
It is not a comparison with recorded motor data. The 5-second stand result is a
forward-dynamics smoke check under these assumptions, not a locomotion or
sim-to-real success criterion; inspect its height drift and tracking error.

The longer check exposed a concrete limitation of copying Microduck's P200:
the refined quadruped slowly crouched until its rear knee-bracket bottoms
contacted the floor. Increasing to the published stock P400 default retained
a stand for 60 simulated seconds within the explicitly recorded bounds
(height loss ≤10 mm, base tilt ≤10°, joint tracking error ≤0.2 rad, no loaded
self contacts or nonfoot ground contacts). This is why `config/actuator.json`
now explicitly uses P400. The P200 failure remains in the report; no larger gain
search was needed. These are screening bounds, not a claim that standing has
been learned or validated on hardware.

Each result records RMS motor torque and a winding-only `I²R` loss proxy derived
from the same fitted Kt/R model. Those values omit driver/electronics losses,
gearbox heat, thermal paths and battery behavior. They cannot establish runtime
or a thermal margin. The report also records active nonfoot ground/self-contact
pairs, final foot normal loads, maximum tilt, and the generated MJCF hash so a
geometry-code change cannot silently preserve an apparently current result.

## Forward crawl replay

`crawl_replay_smoke` provides a deliberately limited bridge from the illustration
to forward dynamics. It generates the illustration on a separate kinematic
model, initializes the live robot once at reference frame zero with zero
velocity, and then sends only the twelve joint targets. Its 96 periodic joint
samples are linearly interpolated at 50 Hz; the controller holds each command
between updates and applies the single 20 ms delay. MuJoCo integrates at 2 ms
for two 6.4-second cycles. No live joint position, velocity, or floating-base
pose is overwritten while that simulation runs.

The frozen assembly's replay stays upright without unwanted contacts but fails
the progress and tracking screen: it requests 40 mm of forward travel and moves
approximately **5.2 mm backward**, with maximum joint tracking error 0.244 rad.
Maximum tilt is 3.62° and peak motor torque 0.161 Nm. The upstream electrical
model reaches neither its PWM bound nor current-window clipping in this run.
Those facts do not isolate a single cause or establish a thermal margin; they
show that clearance and vertical static support checks alone do not produce a
working walking controller.

The report states all failure bounds, source/configuration hashes, update rates,
command-delay semantics, contacts and electrical-limit statistics. This is
prescribed open-loop joint replay, not an RL policy or hardware validation.
The next locomotion work must provide feedback and demonstrate actual forward
progress in this dynamic model before claiming the illustrated crawl works.

Before spending on training or hardware:

1. Finish the refined assembly collision/workspace and gait-load checks, then
   regenerate this report against that configuration.
2. Resolve the public fit's stock-servo/5 V/firmware provenance from existing
   published datasets, settings or validated robot examples. If this cannot be
   resolved, retain a visible uncertainty range or choose a better supported
   component. Do not turn the owner into a motor-characterization operator.
3. Create a separately locked GPU training environment and verify its actuator
   against this CPU reference. The pinned BAM `mjlab` extra targets MuJoCo/Warp
   3.7 and Warp 1.12, whereas the present CPU workbench uses MuJoCo 3.10. Installing
   the extra into this environment is not a validated compatibility resolution.
4. Train standing before walking, under declared model uncertainty, with a
   concrete bounded cloud job. Training success alone will not close the
   hardware-fidelity gate.
