"""Base environment: loads the RL model variant and exposes sensor accessors."""

from __future__ import annotations

import os

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

from . import constants as C
from ..design import locked
from ..mjcf import build_mjcf


def load_model(xml_path: Optional[str] = None) -> mujoco.MjModel:
    """The RL model: the XML file if present (loaded by path so its mesh directory resolves), else
    generated from the locked design."""
    path = C.RL_XML if xml_path is None else xml_path
    if os.path.exists(path):
        return mujoco.MjModel.from_xml_path(str(path))
    return mujoco.MjModel.from_xml_string(build_mjcf(locked(), rl=True))


class CheetahPupEnv(mjx_env.MjxEnv):
    """Base class for Cheetah Pup environments."""

    def __init__(
        self,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        xml_path: Optional[str] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        self._xml_path = str(C.RL_XML if xml_path is None else xml_path)
        self._mj_model = load_model(xml_path)
        self._mj_model.opt.timestep = self._config.sim_dt
        self._model_assets = {}
        impl = self._config.get("impl", "jax")
        self._mjx_model = mjx.put_model(self._mj_model, impl=impl)
        self._imu_site_id = self._mj_model.site(C.IMU_SITE).id
        self._feet_site_id = np.array([self._mj_model.site(n).id for n in C.FEET_SITES])
        self._feet_geom_id = np.array([self._mj_model.geom(n).id for n in C.FEET_GEOMS])
        self._floor_geom_id = self._mj_model.geom(C.FLOOR_GEOM).id
        self._torso_body_id = self._mj_model.body(C.ROOT_BODY).id
        self._torso_mass = float(self._mj_model.body_subtreemass[self._torso_body_id])
        self._feet_contact_sensor_adr = np.array(
            [self._mj_model.sensor_adr[self._mj_model.sensor(n).id] for n in C.FEET_CONTACT_SENSORS]
        )
        adr = []
        for n in C.FEET_LINVEL_SENSORS:
            sid = self._mj_model.sensor(n).id
            a, d = self._mj_model.sensor_adr[sid], self._mj_model.sensor_dim[sid]
            adr.append(list(range(a, a + d)))
        self._foot_linvel_sensor_adr = jp.array(adr)

    # Sensors
    def get_upvector(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, C.UPVECTOR_SENSOR)

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self._imu_site_id].T @ jp.array([0.0, 0.0, -1.0])

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, C.GLOBAL_LINVEL_SENSOR)

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, C.GLOBAL_ANGVEL_SENSOR)

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, C.LOCAL_LINVEL_SENSOR)

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, C.ACCELEROMETER_SENSOR)

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, C.GYRO_SENSOR)

    def get_feet_contact(self, data: mjx.Data) -> jax.Array:
        """Boolean per foot from the foot-floor contact sensors."""
        return data.sensordata[self._feet_contact_sensor_adr] > 0

    def get_feet_linvel(self, data: mjx.Data) -> jax.Array:
        return data.sensordata[self._foot_linvel_sensor_adr]

    # Accessors
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
