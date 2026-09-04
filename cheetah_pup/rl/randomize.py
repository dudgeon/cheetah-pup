"""Domain randomization over the MJX model (vmapped per environment).

Ranges are sized for a 1.4 kg printed robot on Feetech servos: masses and CoM within what a print
and a battery swap can change, floor friction from tile to rubber, servo gain and torque limit
spread for unit-to-unit variation and battery sag. Backlash is not modeled here — BAM's actuator
model (Phase 5) adds it.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 1


def domain_randomize(model: mjx.Model, rng: jax.Array):
    nu = model.nu

    @jax.vmap
    def rand_dynamics(rng):
        rng, key = jax.random.split(rng)
        geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(jax.random.uniform(key, minval=0.5, maxval=1.0))

        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(key, shape=(nu,), minval=0.8, maxval=1.2)
        dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

        rng, key = jax.random.split(rng)
        armature = model.dof_armature[6:] * jax.random.uniform(key, shape=(nu,), minval=0.9, maxval=1.1)
        dof_armature = model.dof_armature.at[6:].set(armature)

        rng, key = jax.random.split(rng)
        dpos = jax.random.uniform(key, (3,), minval=-0.01, maxval=0.01)
        body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(model.body_ipos[TORSO_BODY_ID] + dpos)

        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, shape=(model.nbody,), minval=0.9, maxval=1.1)
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)
        rng, key = jax.random.split(rng)
        body_mass = body_mass.at[TORSO_BODY_ID].set(body_mass[TORSO_BODY_ID] + jax.random.uniform(key, minval=-0.1, maxval=0.1))

        rng, key = jax.random.split(rng)
        qpos0 = model.qpos0.at[7:].set(model.qpos0[7:] + jax.random.uniform(key, shape=(nu,), minval=-0.03, maxval=0.03))

        # Servo gain spread (unit-to-unit, temperature) and torque limit (battery sag) per actuator.
        rng, key = jax.random.split(rng)
        kp_factor = jax.random.uniform(key, shape=(nu,), minval=0.85, maxval=1.15)
        kp = model.actuator_gainprm[:, 0]
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp * kp_factor)
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp * kp_factor)
        rng, key = jax.random.split(rng)
        lim_factor = jax.random.uniform(key, shape=(nu, 1), minval=0.85, maxval=1.05)
        actuator_forcerange = model.actuator_forcerange * lim_factor

        return (geom_friction, dof_frictionloss, dof_armature, body_ipos, body_mass, qpos0,
                actuator_gainprm, actuator_biasprm, actuator_forcerange)

    fields = ("geom_friction", "dof_frictionloss", "dof_armature", "body_ipos", "body_mass", "qpos0",
              "actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
    values = rand_dynamics(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({f: 0 for f in fields})
    model = model.tree_replace(dict(zip(fields, values)))
    return model, in_axes
