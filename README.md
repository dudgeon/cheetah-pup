# Cheetah Pup

> **Independent Microduck research branch:** see
> [`docs/microduck-review/README.md`](docs/microduck-review/README.md) for updated
> sources, additional submodules, and the owner's requirement to avoid custom servo
> characterization. The original handoff below is retained for comparison.

An experimental, hobbyist-scale quadruped robot for reinforcement-learning research:
Mini-Cheetah-style 12-DOF body geometry and leg kinematics, built with Feetech
STS3215-class smart servos and much of the surrounding stack (power, sensors, training
approach) drawn from Hugging Face/Pollen Robotics' open-source **Open Duck Mini v2**
project.

**Status**: pre-design. Research is done, key architecture decisions are made, and the
repo is scaffolded — no CAD, PCB, firmware, or training code exists yet.

**Start here**: [`docs/HANDOFF.md`](docs/HANDOFF.md) is the comprehensive project plan —
decisions made and why, architecture, phased build sequence, budget, safety, and open
questions. [`docs/research-appendix.md`](docs/research-appendix.md) is the research
backing those decisions (MIT Cheetah, prior-art scaled QDD quadrupeds, the Hugging Face
duck robot family, current legged-RL training stacks).

## Repo layout

```
docs/                   Planning and research documents — read HANDOFF.md first
vendor/                 Git submodules: external reference repos (see vendor/README.md)
```

Mechanical CAD, PCB design, firmware, and training code directories will be added as
each build phase starts (see `docs/HANDOFF.md` §5).

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
