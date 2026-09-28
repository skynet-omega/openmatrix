"""Own ray-based achromatic eye instrument; no heading/flow/behavior decoder.

The arena is an enclosing luminous sphere with an analytic surface texture.
Light depends on current physical eye position, head orientation and world time.
Optical mapping is an explicitly unvalidated cross-individual registration.
"""
from pathlib import Path
import copy
import json

import mujoco
import numpy as np
import pandas as pd

from session_io import sha256


def aperture_rays(rays, angle_rad=np.deg2rad(2.)):
    rays = np.asarray(rays, dtype=np.float64)
    if rays.ndim != 2 or rays.shape[1] != 3 or not np.isfinite(rays).all() or not np.allclose(np.linalg.norm(rays, axis=1), 1., atol=1e-12):
        raise ValueError("Need finite unit viewing directions")
    auxiliary = np.tile([0., 0., 1.], (len(rays), 1))
    auxiliary[np.abs(rays[:, 2]) > .9] = [0., 1., 0.]
    a = np.cross(rays, auxiliary)
    a /= np.linalg.norm(a, axis=1)[:, None]
    b = np.cross(rays, a)
    ring = [np.cos(angle_rad)*rays + np.sin(angle_rad)*(np.cos(phi)*a + np.sin(phi)*b)
            for phi in np.arange(6)*np.pi/3]
    return np.stack([rays] + ring, axis=1)


class LuminousWorld:
    def __init__(self, start_ns):
        self.start_ns = int(start_ns)
        self.radius_mm = 25.
        self.center_mm = np.array([0., 0., 1.])
        self.mean_light = .5
        self.contrast = .8
        self.stripes = 8
        self.motion_start_s = .05
        self.motion_stop_s = .15
        self.angular_speed_rad_s = np.pi

    def luminance(self, origins, rays, time_ns):
        if time_ns < self.start_ns:
            raise ValueError("World cannot sense before its initialization")
        offset = np.asarray(origins) - self.center_mm
        if np.any(np.linalg.norm(offset, axis=-1) >= self.radius_mm):
            raise ValueError("Eye left the declared luminous arena; no visual rescue")
        projection = np.sum(offset * rays, axis=-1)
        distance = -projection + np.sqrt(projection**2 + self.radius_mm**2 - np.sum(offset**2, axis=-1))
        hit = offset + distance[..., None] * rays
        longitude = np.arctan2(hit[..., 1], hit[..., 0])
        elapsed = (time_ns - self.start_ns)*1e-9
        phase = self.angular_speed_rad_s * max(0., min(elapsed, self.motion_stop_s) - self.motion_start_s)
        # Latitudinal taper removes the longitude singularity at the poles.
        taper = np.sqrt(hit[..., 0]**2 + hit[..., 1]**2) / self.radius_mm
        return self.mean_light * (1. + self.contrast*taper*np.sin(self.stripes*(longitude-phase)))

    def state_dict(self):
        return copy.deepcopy(self.__dict__)

    @classmethod
    def from_state(cls, state):
        obj = cls(state["start_ns"])
        if set(state) != set(obj.__dict__):
            raise ValueError("Incomplete light-world state")
        obj.__dict__.update(copy.deepcopy(state))
        return obj


class CompoundEye:
    def __init__(self, brain, body, directory):
        folder = Path(directory)
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest.get("schema") != "matrix_optics_candidate_v1":
            raise ValueError("Unknown optical acquisition")
        for name, digest in manifest["files"].items():
            if sha256(folder / name) != digest:
                raise ValueError("Optical acquisition hash differs")
        table = pd.read_parquet(folder / "photoreceptor_rays.parquet")
        state = dict(ids=table.bodyId.to_numpy(np.int64), sides=table.side.to_numpy(dtype="U1"),
                     rays=table[["ray_x", "ray_y", "ray_z"]].to_numpy(np.float64), manifest=manifest)
        self._initialize(brain, body, state)

    def _initialize(self, brain, body, state):
        if set(state) != {"ids", "sides", "rays", "manifest"}:
            raise ValueError("Incomplete optical mapping")
        self.brain, self.body = brain, body
        self.saved = copy.deepcopy(state)
        self.ids, self.sides, self.rays = state["ids"].copy(), state["sides"].copy(), state["rays"].copy()
        if self.ids.dtype != np.int64 or np.any(np.diff(self.ids) <= 0) or self.sides.shape != self.ids.shape or not np.isin(self.sides, ["L", "R"]).all():
            raise ValueError("Invalid optical identities/sides")
        self.indices = np.searchsorted(brain.node_ids, self.ids)
        if np.any(self.indices >= brain.n_neurons) or not np.array_equal(brain.node_ids[self.indices], self.ids):
            raise ValueError("Photoreceptor identity is absent from the canonical CNS")
        self.apertures = aperture_rays(self.rays)
        self.head_id = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, "matrixfly/Head")
        if self.head_id < 0:
            raise ValueError("Missing physical head")
        self.eye_geometry = {}
        for side in ("L", "R"):
            eye_id = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_BODY, f"matrixfly/{side}Eye")
            geoms = np.where(body.model.geom_bodyid == eye_id)[0]
            meshes = [int(g) for g in geoms if body.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
            if len(meshes) != 1:
                raise ValueError("Need one named physical eye mesh")
            geom = meshes[0]
            mesh = body.model.geom_dataid[geom]
            start, count = body.model.mesh_vertadr[mesh], body.model.mesh_vertnum[mesh]
            vertices = body.model.mesh_vert[start:start+count].astype(float)
            self.eye_geometry[side] = (geom, (vertices.min(axis=0)+vertices.max(axis=0))*.5)

    def pose(self):
        # mj_forward on scratch prevents an observer from changing solver state.
        b = self.body
        integration = np.empty(mujoco.mj_stateSize(b.model, b.spec))
        mujoco.mj_getState(b.model, b.data, integration, b.spec)
        mujoco.mj_setState(b.model, b.scratch, integration, b.spec)
        mujoco.mj_forward(b.model, b.scratch)
        rotation = b.scratch.xmat[self.head_id].reshape(3, 3).copy()
        centers = {}
        for side, (geom, local) in self.eye_geometry.items():
            centers[side] = b.scratch.geom_xpos[geom] + b.scratch.geom_xmat[geom].reshape(3, 3) @ local
        return rotation, centers

    def sample(self, world, time_ns, pose=None):
        rotation, centers = self.pose() if pose is None else pose
        directions = self.apertures @ rotation.T
        origins = np.array([centers[s] for s in self.sides])[:, None, :]
        light = world.luminance(origins, directions, time_ns)
        result = .5*light[:, 0] + np.sum(light[:, 1:], axis=1)/12.
        if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise ValueError("Luminance outside normalized instrument range")
        return result

    def state_dict(self):
        return copy.deepcopy(self.saved)

    @classmethod
    def from_state(cls, brain, body, state):
        obj = cls.__new__(cls)
        obj._initialize(brain, body, state)
        return obj
