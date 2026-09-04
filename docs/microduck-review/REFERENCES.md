# Pinned additional references

All entries are research references, not automatically imported dependencies.
Existing seven submodules remain at their prior commits.

| Path | Upstream and pinned commit | Scope |
|---|---|---|
| `vendor/microduck` | [Pollen runtime](https://github.com/pollen-robotics/microduck) — `bc41fb5c9a9b39894669c1e022e375cf83800382` | Apache-2.0 root license; preserve nested third-party notices. Runtime architecture and HAL reference. |
| `vendor/microduck_rl` | [Pollen RL](https://github.com/pollen-robotics/microduck_rl) — `29e887ecfbf5d37144759e5a9f8a176dfb83d547` | Apache-2.0 code. README separately labels 3D model files Creative Commons BY-SA-NC without specifying a version there. Do not treat those assets as Apache or copy them into original quadruped CAD. |
| `vendor/mit_cheetah` | [MIT Cheetah](https://github.com/mit-biomimetics/Cheetah-Software) — `c71c5a138d3e418cc833e94e25357ceea8955daa` | MIT license. FK, Jacobian, topology and model references; OBJ visuals are not manufacturing CAD. |
| `vendor/pollen_robot_hat` | [Pollen HAT](https://github.com/pollen-robotics/elec_RPI_Robot_HAT) — `23eab11927f95ceca0dfa35bf182caeb7db39ea0` | Apache-2.0. KiCad 9 board and manufacturing reference. Does not establish the full Microduck production hardware. |

Initialize only these additions with:

```sh
git submodule update --init vendor/microduck vendor/microduck_rl vendor/mit_cheetah vendor/pollen_robot_hat
```

This branch intentionally revises the prior `vendor/README.md` rationale for excluding
Microduck: useful code can be referenced with explicit asset boundaries. The prior
documents are retained to keep this branch easy to compare with the concurrent agent's
work. Their claim that Microduck is not vendored applies to the original base, not this branch.

The existing BAM gitlink is not advanced. The training dependency eventually selected
must pass an exact actuator-model compatibility check, including its supplied parameters,
voltage/controller conventions and CPU/GPU implementation.
