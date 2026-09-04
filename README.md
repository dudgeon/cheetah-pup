# Cheetah Pup

A compact, original 12-joint quadruped for learning robotics reinforcement learning.
MIT Cheetah-inspired leg kinematics, practical Microduck software reuse, and **no
owner servo characterization or motor-model fitting**.

This branch is `codex/microduck-research-20260904`; the parallel agent's branch is
separate. [Implementation plan](docs/implementation/PLAN.md) ·
[decisions](docs/implementation/DECISIONS.md) ·
[motor and cloud readiness](reports/pre-cloud-review.md) ·
[CPU training](reports/cpu-rl-smoke.md) ·
[research](docs/microduck-review/RESEARCH.md).

## Current milestone

The refined assembly anchors each XL330 casing to its real offset output shaft,
models the stock horn separately, and uses manufacturer reference center of mass
and inertia. Original slotted cradles reserve access to the two servo sockets.
The component mass allowance remains **613 g**. This is a study of the assembly;
screw engagement, support bearings, structural stiffness and real cable routing
still need detailed CAD.

![Refined assembly and hip orientation](reports/assembly-review.png)

**[Download the STEP assembly](models/cheetah_pup_assembly.step)** — 83 named original
solids, checked against the compiled MuJoCo model and verified after STEP readback.
Manufacturer CAD is not redistributed. **[Updated crawl animation](reports/gait-demo.md)**
shows the revised, shallower posture; it is a kinematic illustration, not a policy.

- **[Assembly audit](reports/assembly-validation.md):** all twelve shafts align;
  the nominal pose and 192 sampled crawl poses have no solid or reserved-port
  interference. Broad joint-box sampling finds collisions, so scalar joint limits
  alone do not define a valid robot workspace.
- **[Actuator integration](docs/implementation/ACTUATOR.md):** pinned upstream BAM
  CPU model, explicit 5 V/P400 settings, current/PWM behavior and 20 ms delay;
  bit-exact upstream replay parity. A 60-second simulated stand passes the stated
  pose/contact screen. Stock-servo calibration and firmware equivalence remain open.
- **[Walking load screen](reports/gait-load-validation.md):** selected crawl needs
  0.0934 N·m peak static torque, only 1.07× margin to the 0.10 N·m estimate.
  Retiming that illustration to 0.05 m/s exceeds unloaded motor speed. A smoother
  walking controller and more load margin are needed before choosing hardware.
- **[Mass sensitivity](reports/motor-viability.md):** reducing the unchanged-motor
  assembly allowance toward 440 g reaches the proposed 1.5× static reserve for this
  crawl. This is a sizing hypothesis, not a verified lighter build.
- **[Actual CPU PPO](reports/cpu-rl-smoke.md):** 16,384 transitions in 20 seconds on
  eight workers, with saved initial/trained checkpoints. Fixed home targets and both
  networks passed all eight five-second held-out standing episodes. Training did
  not improve on fixed targets. The isolated [training environment](training_cpu/README.md)
  uses the pinned BAM controller and documents exact reset/action/observation rules.

**55 tests pass**, covering kinematics, reference frames, force allocation, geometry,
actuator integration and reset/delay behavior. Sampled CPU resets and evaluated poses
also clear the independent solid/connector audit. Learned walking, real terrain
traversal, thermal capability and battery runtime remain unproven.

## Reproduce

Python 3.12 and `uv`; run from the repository root:

```sh
git submodule update --init vendor/bam_microduck
uv sync --locked
uv run pytest -q
uv run cheetah-pup build
uv run cheetah-pup validate
uv run cheetah-pup render
uv run python -m cheetah_pup.assembly_audit
uv run python -m cheetah_pup.gait_load
uv run python -m cheetah_pup.actuator
```

The ordinary generated XML uses ideal PD for diagnostic comparisons. The BAM
command converts it to torque actuators and applies the actual pinned law; do not
train on the diagnostic XML while describing it as BAM. The two upstream BAM
revisions have separate gitlinks so legacy references stay intact.

All commands above use CPU MuJoCo. Rendering uses Matplotlib without OpenGL.
The optional native `render --renderer mujoco` needs a working OpenGL installation.
For the STEP study, use the optional exporter dependency:

```sh
uv run --locked --with cadquery-ocp==7.9.3.1.1 python scripts/export_assembly_step.py
```

To regenerate the annotated shoulder review:

```sh
uv run python -c 'from pathlib import Path; from cheetah_pup.model import load_config; from cheetah_pup.render import render_assembly_review; render_assembly_review(load_config("config/robot.json"), Path("reports/assembly-review.png"))'
```

Other scenes:

```sh
uv run cheetah-pup build --terrain threshold --output models/cheetah_pup_threshold.xml
uv run cheetah-pup build --terrain carpet --output models/cheetah_pup_carpet_placeholder.xml
```

Carpet is a rigid friction placeholder. The 10 mm obstacle exists in its scene;
traversal is not demonstrated. Geometry/mass assumptions live in
[config/robot.json](config/robot.json); BAM settings in
[config/actuator.json](config/actuator.json).

## Next implementation slice

1. Build a smoother contact-aware walking controller; compare lighter assemblies
   with stronger model-supported servos, including their extra mass and size.
   The [alternative review](reports/pre-cloud-review.md) prioritizes an exact
   7.4 V STS3215 packaging/load study, with XC330 as the compact comparison.
2. Establish the quadruped RL task and CPU/GPU actuator parity. Begin exploratory
   standing/very slow flat-ground learning, with explicit model uncertainty and a
   concrete capped cloud job before any spend.
3. Close hardware/model/load gates before purchases, manufacturing CAD or PCB work.

Original code and generated assembly geometry are [MIT licensed](LICENSE).
Pinned upstreams retain their own licenses; see [reference scope](docs/microduck-review/REFERENCES.md).
