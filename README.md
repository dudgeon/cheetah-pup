# Cheetah Pup

An experimental, hobbyist-scale quadruped robot for reinforcement-learning research:
Mini-Cheetah-style 12-DOF body geometry and leg kinematics, built with Feetech
STS3215-class smart servos and much of the surrounding stack (power, sensors, training
approach) drawn from Hugging Face/Pollen Robotics' open-source **Open Duck Mini v2**
project.

**Status**: Phase 1 complete, Phase 2 CAD first pass done. The design is locked (candidate A ·
direct drive, size M — see [`docs/design/01-candidates.md`](docs/design/01-candidates.md) and
`docs/design/locked.json`), the MuJoCo model is validated
([`docs/design/02-sim-validation.md`](docs/design/02-sim-validation.md)), the RL environment is
built and smoke-tested ([`docs/design/03-rl-environment.md`](docs/design/03-rl-environment.md)),
and the build123d CAD of every printed part now feeds the simulator its masses, inertias and
meshes ([`docs/design/04-cad-detail.md`](docs/design/04-cad-detail.md)). Next: confirm the servo
interface on a real STS3215, the first cloud-GPU training run, and the second CAD pass.

**Start here**: [`docs/HANDOFF.md`](docs/HANDOFF.md) is the project plan — decisions made and
why, architecture, phased build sequence, budget, safety, and open questions.
[`docs/DESIGN_LOG.md`](docs/DESIGN_LOG.md) is the dated record of decisions and milestones.
[`docs/research-appendix.md`](docs/research-appendix.md) is the research backing the decisions.

## Repo layout

```
cheetah_pup/            Design library (SI units): servo + electronics data, parameters and
                        presets (incl. the locked design), 3-DOF leg kinematics, gait generation,
                        sizing analysis, MJCF generation, MuJoCo validation, page builders
cheetah_pup/rl/         MuJoCo Playground (MJX) environment, domain randomization, PPO runner
cad/                    build123d CAD: STS3215 model and interface conventions, printed parts in
                        their MuJoCo body frames, assembly + mass properties + STEP/STL export
cad/exports/            Generated: step/ and stl/ per part, mass_properties.json (read by the
                        MJCF generator), viewer_meshes.json (read by the DR-04 page builder)
sim/                    Generated models (cheetah_pup.xml, cheetah_pup_rl.xml) and validation
                        recordings; checkpoints are written to sim/checkpoints (ignored)
tests/                  pytest suite (design library, MJCF, RL environment)
docs/                   Plan, design log, research; docs/design/ holds each design review
docs/design/review/     DR-01 candidate review page (template + build)
docs/design/replay/     DR-02 sim playback page (template + build)
docs/design/cad/        DR-04 CAD review page with a WebGL viewer (template + build)
vendor/                 Git submodules: external reference repos (see vendor/README.md)
```

PCB design and firmware directories will be added as those build phases start (see
`docs/HANDOFF.md` §5).

## Working with the code

```
python -m venv .venv && .venv/bin/pip install -e ".[sim,rl,dev]"
.venv/bin/pytest tests/test_kinematics.py tests/test_analysis.py tests/test_mjcf.py   # fast
JAX_PLATFORMS=cpu .venv/bin/pytest tests/test_rl_env.py                               # ~2 min JIT
.venv/bin/python -m cheetah_pup.mjcf sim/cheetah_pup.xml            # model from the locked design
.venv/bin/python -m cheetah_pup.mjcf sim/cheetah_pup_rl.xml --rl    # training variant
.venv/bin/python -m cheetah_pup.validate sim/cheetah_pup.xml        # stand + gait recordings
.venv/bin/python -m cheetah_pup.build_replay                        # rebuild the DR-02 page
.venv/bin/python -m cheetah_pup.build_artifact                      # rebuild the DR-01 page
JAX_PLATFORMS=cpu .venv/bin/python -m cheetah_pup.rl.train --smoke  # PPO pipeline check
```

The CAD needs the `cad` extra (build123d pulls in OCP, so use a separate venv if the sim one
is precious):

```
python -m venv .venv-cad && .venv-cad/bin/pip install -e ".[cad]"
.venv-cad/bin/python -m cad.assembly                 # STEP/STL, mass_properties.json, viewer meshes
.venv/bin/python -m cheetah_pup.mjcf sim/cheetah_pup.xml && .venv/bin/python -m cheetah_pup.mjcf sim/cheetah_pup_rl.xml --rl
.venv/bin/python -m cheetah_pup.build_cad_review     # rebuild the DR-04 page
```

`python -m cheetah_pup.mjcf` uses the CAD inertias and meshes whenever `cad/exports` matches the
locked design; `--no-cad` writes the Phase 1 primitive model instead.

`cheetah_pup.design.PRESETS` holds the candidates; `cheetah_pup.analysis.metrics()` sizes any
`DesignParams`. The review page's JavaScript mirrors the same math and cross-checks itself
against the package's numbers at load.

## Getting the code

This repo uses git submodules for vendored reference material:

```
git clone --recurse-submodules <this-repo-url>
# or, if already cloned:
git submodule update --init --recursive
```

## License

This repo's own code is MIT licensed (see `LICENSE`) unless noted otherwise. Vendored
reference material under `vendor/` keeps its own upstream license — see
`vendor/README.md` for exact terms per repo, including a few that are reference-only
pending license clarification from their authors.
