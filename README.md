# Cheetah Pup

An experimental, hobbyist-scale quadruped robot for reinforcement-learning research:
Mini-Cheetah-style 12-DOF body geometry and leg kinematics, built with Feetech
STS3215-class smart servos and much of the surrounding stack (power, sensors, training
approach) drawn from Hugging Face/Pollen Robotics' open-source **Open Duck Mini v2**
project.

**Status**: Phase 1 (kinematic validation). Three leg-architecture candidates are sized and
published for review — see [`docs/design/01-candidates.md`](docs/design/01-candidates.md). The
next milestone after the design is locked is the MuJoCo sim model and the RL environment.

**Start here**: [`docs/HANDOFF.md`](docs/HANDOFF.md) is the project plan — decisions made and
why, architecture, phased build sequence, budget, safety, and open questions.
[`docs/DESIGN_LOG.md`](docs/DESIGN_LOG.md) is the dated record of decisions and milestones.
[`docs/research-appendix.md`](docs/research-appendix.md) is the research backing the decisions.

## Repo layout

```
cheetah_pup/            Design library (SI units): servo + electronics data, parameters and
                        presets, 3-DOF leg kinematics, gait generation, sizing analysis,
                        artifact/JSON export
tests/                  pytest suite for the design library
docs/                   Plan, design log, research; docs/design/ holds each design review
docs/design/review/     Source (template.html) and build (index.html) of the DR-01 review page
vendor/                 Git submodules: external reference repos (see vendor/README.md)
```

Mechanical CAD, PCB design, firmware, and training code directories will be added as
each build phase starts (see `docs/HANDOFF.md` §5).

## Working with the design library

```
python -m venv .venv && .venv/bin/pip install -e ".[sim,dev]"
.venv/bin/pytest                                   # kinematics + sizing tests
.venv/bin/python -m cheetah_pup.export docs/design/candidates.json   # presets + metrics
.venv/bin/python -m cheetah_pup.build_artifact     # rebuild the review page from the template
```

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
