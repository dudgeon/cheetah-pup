"""Reinforcement-learning environment for Cheetah Pup, built on MuJoCo Playground (MJX).

Modules:
- constants: paths, sensor and site names
- base: CheetahPupEnv — model loading and sensor accessors
- joystick: velocity-command locomotion task (observations, rewards, termination)
- randomize: domain randomization over the MJX model
- train: Brax PPO runner (smoke test on CPU, full runs on GPU)

Structure follows mujoco_playground's Go1 joystick environment (Apache-2.0), which is the same
12-DOF abad/hip/knee quadruped layout as ours, adapted to our scale, sensors, and servo limits.
"""
