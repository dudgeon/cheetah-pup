# Microduck-based Cheetah Pup: independent research branch

Research date: 2026-09-04. Branch: `codex/microduck-research-20260904`.
Base: `41de690f6d21ff85ba88ab91474836779c02ab35`.

This branch supplements the existing agent's work. It does not replace its handoff or
claim that new architecture choices have already been accepted. The owner explicitly
authorized an independent branch while another agent works on the project.

## Latest requirements

- Original goal: a much smaller quadruped with MIT Cheetah-style geometry, using the
  recent Pollen/Hugging Face Microduck's components and software where practical.
- Research, then interview, then a comprehensive plan, then implementation in stages.
- New explicit constraint: **no owner-performed servo characterization or motor-model
  fitting in the baseline project**. This is the owner's first robotics RL project.
  Prefer components with existing identified models and reproducible integration.
  This preference should influence selection without overriding the overall project.
- Carry forward the existing handoff's recorded decisions as context: 12 DOF,
  smart servos, $600–$1,500 hardware budget, cloud GPU training, scriptable CAD with
  STEP export, and primitive simulation before detailed CAD. Exact motor SKU,
  dimensions, compute, sensor, and power choices remain provisional on this branch.

Read [RESEARCH.md](RESEARCH.md) for evidence and corrections, and
[NEXT_STEPS.md](NEXT_STEPS.md) for the remaining interview and proposed sequence.

## What is completed

Four additional upstreams are pinned as real git submodules: Microduck runtime,
Microduck RL, MIT Cheetah software, and Pollen's open robot HAT. Existing submodule
pins are retained. See [REFERENCES.md](REFERENCES.md) for scope and licenses.

No CAD, PCB, runtime, learned quadruped policy, or hardware validation is claimed.
The existing plan's servo-characterization steps are **not adopted** on this branch.
The next selection pass must identify a usable published motor model before freezing
the actuator, geometry, or battery rail.
