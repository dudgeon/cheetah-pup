"""Compatibility shims between the installed Brax and JAX versions.

Brax 0.14.x still calls `jax.device_put_replicated`, which JAX 0.10 removed. This restores it with
the documented drop-in: stack a copy per device along a new leading axis and place the result
across the devices, which is what Brax's pmap-based training loop expects.

Import this module before `brax.training` (train.py does).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _device_put_replicated(x, devices):
    devices = list(devices)
    n = len(devices)
    if n == 1:
        return jax.tree_util.tree_map(lambda a: jax.device_put(jnp.expand_dims(jnp.asarray(a), 0), devices[0]), x)
    mesh = jax.sharding.Mesh(devices, ("d",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
    return jax.tree_util.tree_map(lambda a: jax.device_put(jnp.stack([jnp.asarray(a)] * n), sharding), x)


def install() -> None:
    try:
        jax.device_put_replicated  # noqa: B018 — raises AttributeError once removed
    except AttributeError:
        jax.device_put_replicated = _device_put_replicated


install()
