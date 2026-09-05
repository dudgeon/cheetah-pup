"""Parametric CAD of the Cheetah Pup in build123d.

Every part is built in the MuJoCo body frame it belongs to (trunk, <leg>_abad, <leg>_hip,
<leg>_knee) so the exported meshes and mass properties drop straight into the simulation model.
Frames: x forward, y left, z up; units are millimetres inside this package (build123d convention)
and metres everywhere else in the repo.

Run `python -m cad.assembly` (needs the `cad` extra: build123d) to regenerate cad/exports/.
"""
