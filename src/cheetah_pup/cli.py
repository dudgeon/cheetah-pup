"""Small reproducible entry point for the first simulation milestone."""
from __future__ import annotations

import argparse
from pathlib import Path

from .model import build_mjcf, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/robot.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Generate original MJCF geometry")
    build.add_argument("--terrain", choices=("flat", "threshold", "carpet"), default="flat")
    build.add_argument("--output", type=Path, default=Path("models/cheetah_pup_flat.xml"))
    validate = commands.add_parser("validate", help="Check geometry and record load screening")
    validate.add_argument("--output", type=Path, default=Path("reports/primitive-validation.json"))
    render = commands.add_parser("render", help="Render the initial pose; no gait or policy")
    render.add_argument("--output", type=Path, default=Path("reports/primitive-preview.png"))
    render.add_argument("--renderer", choices=("software", "mujoco"), default="software")
    args = parser.parse_args()
    config = load_config(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "build":
        args.output.write_text(build_mjcf(config, args.terrain))
        print(f"Wrote {args.output}")
    elif args.command == "validate":
        from .analysis import write_report
        report = write_report(config, args.output)
        print(f"Geometry: {report['gates']['geometry_implementation']}; mass: {report['structure']['mujoco_mass_kg']:.3f} kg")
        print(f"Wrote {args.output} and {args.output.with_suffix('.md')}")
        print("Motor selection, BAM physics, RL and hardware gates remain open.")
        if report["gates"]["geometry_implementation"] != "pass":
            raise SystemExit(1)
    elif args.command == "render":
        if args.renderer == "software":
            from .render import render_software
            render_software(config, args.output)
            print(f"Wrote {args.output}; software projection of the neutral MuJoCo model")
            return
        import mujoco
        import numpy as np
        from PIL import Image
        from .analysis import standing_model
        model, data = standing_model(config)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = np.array([0, 0, 0.09])
        camera.distance = 0.53
        camera.azimuth = 135
        camera.elevation = -24
        with mujoco.Renderer(model, height=720, width=1080) as renderer:
            renderer.update_scene(data, camera=camera)
            Image.fromarray(renderer.render()).save(args.output)
        print(f"Wrote {args.output}; neutral kinematic pose, not a learned behavior")


if __name__ == "__main__":
    main()
